use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args};
use comfy_table::{ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use tokio::task::JoinSet;

use crate::{AudioInputDto, TranscribeRequest, TranscribeResponse};

#[derive(Debug, Args)]
pub struct PerfArgs {
    /// Text file containing one audio URL per line.
    #[arg(short, long, value_name = "PATH")]
    input: PathBuf,

    /// Transcribe service endpoint.
    #[arg(long, default_value = "http://127.0.0.1:8867/inference")]
    server: String,

    /// HTTP concurrency. Use 0 to auto-select the machine CPU count.
    #[arg(long, default_value_t = 0)]
    concurrency: usize,

    /// Number of audio URLs per request.
    #[arg(long = "batch-size", default_value_t = 1)]
    batch_size: usize,

    /// Override mono mode for every request item. Omit to let the server decide.
    #[arg(long, action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    mono: Option<bool>,

    /// HTTP request retries.
    #[arg(long, default_value_t = 3)]
    retries: usize,

    /// HTTP timeout in seconds.
    #[arg(long, default_value_t = 3600)]
    timeout_sec: u64,

    /// Directory used when `--output` is not provided.
    #[arg(long, default_value = "results", value_name = "DIR")]
    results_dir: PathBuf,

    /// Optional JSON result output path.
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct PerfRequestResult {
    request_index: usize,
    batch_size: usize,
    latency_seconds: f64,
    ok: bool,
    bad_items: usize,
    audio_seconds: f64,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PerfSummary {
    requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    items: usize,
    bad_items: usize,
    concurrency: usize,
    batch_size: usize,
    wall_seconds: f64,
    requests_per_second: f64,
    items_per_second: f64,
    audio_seconds: f64,
    audio_seconds_per_second: f64,
    rtf: f64,
    latency_p50_seconds: f64,
    latency_p90_seconds: f64,
    latency_p95_seconds: f64,
    latency_p99_seconds: f64,
    latency_max_seconds: f64,
}

#[derive(Debug, Serialize)]
struct PerfReport<'a> {
    timestamp: String,
    input: &'a str,
    service_url: &'a str,
    items: usize,
    summary: PerfSummary,
    requests: Vec<PerfRequestResult>,
}

#[derive(Debug, Clone, Serialize)]
struct PerfRunReport {
    summary: PerfSummary,
    requests: Vec<PerfRequestResult>,
}

pub async fn run_perf(args: PerfArgs) -> Result<()> {
    validate_args(&args)?;
    let urls = read_urls(&args.input)?;
    if urls.is_empty() {
        bail!("input contains no URLs: {}", args.input.display());
    }
    let concurrency = resolve_concurrency(args.concurrency);
    let batch_size = args.batch_size;

    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()
        .context("failed to build HTTP client")?;

    println!(
        "perf START | input={} items={} server={} batch={} concurrency={} mono={}",
        args.input.display(),
        urls.len(),
        args.server,
        batch_size,
        concurrency,
        format_mono(args.mono)
    );
    let run = run_one_perf_trial(
        &client,
        &args.server,
        &urls,
        concurrency,
        batch_size,
        args.mono,
        args.retries,
    )
    .await
    .with_context(|| format!("perf failed: batch_size={batch_size} concurrency={concurrency}"))?;
    let summary = run.summary.clone();
    println!(
        "perf DONE  | items={} bad={} failed={} elapsed={:.3}s audio={:.1}s speed={:.1}x",
        summary.items,
        summary.bad_items,
        summary.failed_requests,
        summary.wall_seconds,
        summary.audio_seconds,
        summary.audio_seconds_per_second,
    );
    print_perf_summary_table(&summary);

    let timestamp = timestamp_file_name();
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.results_dir.join(format!("perf_{timestamp}.json")));
    save_report(
        &output,
        &PerfReport {
            timestamp,
            input: &args.input.to_string_lossy(),
            service_url: &args.server,
            items: urls.len(),
            summary,
            requests: run.requests,
        },
    )?;
    println!("[saved] {}", output.display());
    Ok(())
}

fn validate_args(args: &PerfArgs) -> Result<()> {
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }
    if args.retries == 0 {
        bail!("--retries must be greater than 0");
    }
    if args.batch_size == 0 {
        bail!("--batch-size must be greater than 0");
    }
    Ok(())
}

fn read_urls(path: &Path) -> Result<Vec<String>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect())
}

fn resolve_concurrency(value: usize) -> usize {
    if value > 0 {
        return value;
    }
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}

fn format_mono(mono: Option<bool>) -> &'static str {
    match mono {
        Some(true) => "true",
        Some(false) => "false",
        None => "auto",
    }
}

async fn run_one_perf_trial(
    client: &reqwest::Client,
    server: &str,
    urls: &[String],
    concurrency: usize,
    batch_size: usize,
    mono: Option<bool>,
    retries: usize,
) -> Result<PerfRunReport> {
    let batches = build_batches(urls, batch_size, mono);
    let total_requests = batches.len();
    let progress = progress_bar(total_requests);
    let start = Instant::now();
    let mut next_request_index = 0usize;
    let mut in_flight = JoinSet::new();
    let mut results = Vec::with_capacity(batches.len());

    while next_request_index < batches.len() {
        while in_flight.len() < concurrency && next_request_index < batches.len() {
            let request_index = next_request_index;
            next_request_index = next_request_index.saturating_add(1);
            let request = batches[request_index].clone();
            let server = server.to_string();
            let client = client.clone();
            in_flight.spawn(async move {
                run_one_request(client, server, request_index, request, retries).await
            });
        }

        match in_flight.join_next().await {
            Some(joined) => {
                results.push(joined.context("perf request task panicked")??);
                progress.inc(1);
            }
            None => break,
        }
    }

    while let Some(joined) = in_flight.join_next().await {
        results.push(joined.context("perf request task panicked")??);
        progress.inc(1);
    }

    let wall_seconds = start.elapsed().as_secs_f64();
    progress.finish_with_message(format!(
        "completed {}/{} in {:.1}s",
        results.len(),
        total_requests,
        wall_seconds
    ));
    let summary = summarize(concurrency, batch_size, wall_seconds, &results);
    Ok(PerfRunReport {
        summary,
        requests: results,
    })
}

fn build_batches(urls: &[String], batch_size: usize, mono: Option<bool>) -> Vec<TranscribeRequest> {
    urls.chunks(batch_size)
        .map(|chunk| {
            let inputs = chunk
                .iter()
                .map(|url| AudioInputDto {
                    url: Some(url.clone()),
                    b64_str: None,
                    mono,
                    hotword: None,
                })
                .collect();
            TranscribeRequest { inputs }
        })
        .collect()
}

async fn run_one_request(
    client: reqwest::Client,
    server: String,
    request_index: usize,
    request: TranscribeRequest,
    retries: usize,
) -> Result<PerfRequestResult> {
    let batch_size = request.inputs.len();
    let mut last_error = None;
    for attempt in 0..retries {
        let start = Instant::now();
        match client.post(&server).json(&request).send().await {
            Ok(response) if response.status().is_success() => {
                let latency_seconds = start.elapsed().as_secs_f64();
                let response = response
                    .json::<TranscribeResponse>()
                    .await
                    .context("failed to decode transcribe response JSON")?;
                let bad_items = response.data.iter().filter(|item| item.is_bad).count();
                return Ok(PerfRequestResult {
                    request_index,
                    batch_size,
                    latency_seconds,
                    ok: true,
                    bad_items,
                    audio_seconds: response.inference_performance.total_audio_duration_seconds,
                    error: None,
                });
            }
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                last_error = Some(anyhow!(
                    "status={} body={}",
                    status,
                    truncate_for_error(&body)
                ));
            }
            Err(error) => {
                last_error = Some(anyhow!("{error}"));
            }
        }
        if attempt + 1 < retries {
            tokio::time::sleep(Duration::from_millis(100 * (attempt as u64 + 1))).await;
        }
    }

    Ok(PerfRequestResult {
        request_index,
        batch_size,
        latency_seconds: 0.0,
        ok: false,
        bad_items: batch_size,
        audio_seconds: 0.0,
        error: Some(
            last_error
                .map(|error| error.to_string())
                .unwrap_or_else(|| "request failed".to_string()),
        ),
    })
}

fn summarize(
    concurrency: usize,
    batch_size: usize,
    wall_seconds: f64,
    results: &[PerfRequestResult],
) -> PerfSummary {
    let requests = results.len();
    let successful_requests = results.iter().filter(|result| result.ok).count();
    let failed_requests = requests.saturating_sub(successful_requests);
    let items: usize = results.iter().map(|result| result.batch_size).sum();
    let bad_items: usize = results.iter().map(|result| result.bad_items).sum();
    let audio_seconds: f64 = results.iter().map(|result| result.audio_seconds).sum();
    let mut latencies = results
        .iter()
        .filter(|result| result.ok)
        .map(|result| result.latency_seconds)
        .collect::<Vec<_>>();
    latencies.sort_by(f64::total_cmp);
    let requests_per_second = if wall_seconds > 0.0 {
        requests as f64 / wall_seconds
    } else {
        0.0
    };
    let items_per_second = if wall_seconds > 0.0 {
        items as f64 / wall_seconds
    } else {
        0.0
    };
    let audio_seconds_per_second = if wall_seconds > 0.0 {
        audio_seconds / wall_seconds
    } else {
        0.0
    };
    let rtf = if audio_seconds > 0.0 {
        wall_seconds / audio_seconds
    } else {
        0.0
    };
    PerfSummary {
        requests,
        successful_requests,
        failed_requests,
        items,
        bad_items,
        concurrency,
        batch_size,
        wall_seconds: round4(wall_seconds),
        requests_per_second: round4(requests_per_second),
        items_per_second: round4(items_per_second),
        audio_seconds: round4(audio_seconds),
        audio_seconds_per_second: round4(audio_seconds_per_second),
        rtf: round6(rtf),
        latency_p50_seconds: round4(percentile(&latencies, 50.0)),
        latency_p90_seconds: round4(percentile(&latencies, 90.0)),
        latency_p95_seconds: round4(percentile(&latencies, 95.0)),
        latency_p99_seconds: round4(percentile(&latencies, 99.0)),
        latency_max_seconds: round4(latencies.last().copied().unwrap_or(0.0)),
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((pct / 100.0) * (sorted.len().saturating_sub(1) as f64)).ceil() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn progress_bar(total: usize) -> ProgressBar {
    let progress = ProgressBar::new(total as u64);
    let style = ProgressStyle::with_template(
        "perf requests [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} eta {eta_precise}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("##-");
    progress.set_style(style);
    progress
}

fn print_perf_summary_table(summary: &PerfSummary) {
    let mut table = summary_table();
    table.add_rows([
        vec!["items".to_string(), summary.items.to_string()],
        vec!["bad items".to_string(), summary.bad_items.to_string()],
        vec!["requests".to_string(), summary.requests.to_string()],
        vec![
            "successful".to_string(),
            summary.successful_requests.to_string(),
        ],
        vec!["failed".to_string(), summary.failed_requests.to_string()],
        vec!["batch size".to_string(), summary.batch_size.to_string()],
        vec!["concurrency".to_string(), summary.concurrency.to_string()],
        vec![
            "elapsed".to_string(),
            format!("{:.3}s", summary.wall_seconds),
        ],
        vec![
            "audio".to_string(),
            format!("{:.1}s", summary.audio_seconds),
        ],
        vec![
            "speed".to_string(),
            format!("{:.1}x", summary.audio_seconds_per_second),
        ],
        vec!["rtf".to_string(), format!("{:.6}", summary.rtf)],
        vec![
            "requests/s".to_string(),
            format!("{:.3}", summary.requests_per_second),
        ],
        vec![
            "items/s".to_string(),
            format!("{:.3}", summary.items_per_second),
        ],
        vec![
            "latency p50".to_string(),
            format!("{:.3}s", summary.latency_p50_seconds),
        ],
        vec![
            "latency p90".to_string(),
            format!("{:.3}s", summary.latency_p90_seconds),
        ],
        vec![
            "latency p95".to_string(),
            format!("{:.3}s", summary.latency_p95_seconds),
        ],
        vec![
            "latency p99".to_string(),
            format!("{:.3}s", summary.latency_p99_seconds),
        ],
        vec![
            "latency max".to_string(),
            format!("{:.3}s", summary.latency_max_seconds),
        ],
    ]);
    println!("\n{table}\n");
}

fn summary_table() -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["metric", "value"]);
    table
}

fn save_report(path: &Path, report: &PerfReport<'_>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(report).context("failed to encode report JSON")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn timestamp_file_name() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_utc_timestamp(seconds)
}

fn format_utc_timestamp(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3600;
    let minute = (seconds_of_day % 3600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}{month:02}{day:02}_{hour:02}{minute:02}{second:02}")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u64, u64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };
    (year, month as u64, day as u64)
}

fn truncate_for_error(body: &str) -> String {
    const MAX: usize = 500;
    if body.len() <= MAX {
        body.to_string()
    } else {
        format!("{}...", &body[..MAX])
    }
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::format_utc_timestamp;

    #[test]
    fn timestamp_file_name_uses_utc_calendar_format() {
        assert_eq!(format_utc_timestamp(0), "19700101_000000");
        assert_eq!(format_utc_timestamp(1_782_720_711), "20260629_081151");
    }
}
