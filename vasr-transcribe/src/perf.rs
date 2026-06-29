use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
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

    /// Number of concurrent HTTP requests.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Number of audio URLs per request.
    #[arg(long = "batch-size", default_value_t = 1)]
    batch_size: usize,

    /// Total number of batch requests to send.
    #[arg(long, conflicts_with = "duration_sec")]
    requests: Option<usize>,

    /// Run for this many seconds instead of a fixed request count.
    #[arg(long = "duration-sec", conflicts_with = "requests")]
    duration_sec: Option<u64>,

    /// Send mono=true for every request item.
    #[arg(long, default_value_t = true)]
    mono: bool,

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
    summary: PerfSummary,
    requests: Vec<PerfRequestResult>,
}

pub async fn run_perf(args: PerfArgs) -> Result<()> {
    validate_args(&args)?;
    let urls = read_urls(&args.input)?;
    if urls.is_empty() {
        bail!("input contains no URLs: {}", args.input.display());
    }

    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()
        .context("failed to build HTTP client")?;
    let total_requests = args.requests.unwrap_or(usize::MAX);
    let deadline = args
        .duration_sec
        .map(|seconds| Instant::now() + Duration::from_secs(seconds));
    let start = Instant::now();
    let mut next_request_index = 0usize;
    let mut in_flight = JoinSet::new();
    let mut results = Vec::new();

    while next_request_index < total_requests {
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            break;
        }
        while in_flight.len() < args.concurrency && next_request_index < total_requests {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let request_index = next_request_index;
            next_request_index = next_request_index.saturating_add(1);
            let request = build_request(&urls, request_index, args.batch_size, args.mono);
            let server = args.server.clone();
            let client = client.clone();
            let retries = args.retries;
            in_flight.spawn(async move {
                run_one_request(client, server, request_index, request, retries).await
            });
        }

        match in_flight.join_next().await {
            Some(joined) => results.push(joined.context("perf request task panicked")??),
            None => break,
        }
    }

    while let Some(joined) = in_flight.join_next().await {
        results.push(joined.context("perf request task panicked")??);
    }

    let wall_seconds = start.elapsed().as_secs_f64();
    let summary = summarize(args.concurrency, args.batch_size, wall_seconds, &results);
    print_perf_summary_table(&summary);

    let timestamp = unix_timestamp_string();
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
            summary,
            requests: results,
        },
    )?;
    println!("[saved] {}", output.display());
    Ok(())
}

fn validate_args(args: &PerfArgs) -> Result<()> {
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }
    if args.concurrency == 0 {
        bail!("--concurrency must be greater than 0");
    }
    if args.batch_size == 0 {
        bail!("--batch-size must be greater than 0");
    }
    if args.retries == 0 {
        bail!("--retries must be greater than 0");
    }
    if args.requests.is_none() && args.duration_sec.is_none() {
        bail!("provide --requests or --duration-sec");
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

fn build_request(
    urls: &[String],
    request_index: usize,
    batch_size: usize,
    mono: bool,
) -> TranscribeRequest {
    let start = request_index.saturating_mul(batch_size);
    let inputs = (0..batch_size)
        .map(|offset| {
            let url = urls[(start + offset) % urls.len()].clone();
            AudioInputDto {
                url: Some(url),
                b64_str: None,
                mono: Some(mono),
                hotword: None,
            }
        })
        .collect();
    TranscribeRequest { inputs }
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

fn print_perf_summary_table(summary: &PerfSummary) {
    println!();
    println!("+----------------------+----------------+");
    println!("| metric               | value          |");
    println!("+----------------------+----------------+");
    println!("| requests             | {:>14} |", summary.requests);
    println!(
        "| successful           | {:>14} |",
        summary.successful_requests
    );
    println!("| failed               | {:>14} |", summary.failed_requests);
    println!("| items                | {:>14} |", summary.items);
    println!("| bad items            | {:>14} |", summary.bad_items);
    println!("| concurrency          | {:>14} |", summary.concurrency);
    println!("| batch size           | {:>14} |", summary.batch_size);
    println!("| wall                 | {:>13.4}s |", summary.wall_seconds);
    println!(
        "| requests/s           | {:>14.4} |",
        summary.requests_per_second
    );
    println!(
        "| items/s              | {:>14.4} |",
        summary.items_per_second
    );
    println!(
        "| audio                | {:>13.4}s |",
        summary.audio_seconds
    );
    println!(
        "| audio speed          | {:>10.4}x |",
        summary.audio_seconds_per_second
    );
    println!("| rtf                  | {:>14.6} |", summary.rtf);
    println!(
        "| latency p50          | {:>13.4}s |",
        summary.latency_p50_seconds
    );
    println!(
        "| latency p90          | {:>13.4}s |",
        summary.latency_p90_seconds
    );
    println!(
        "| latency p95          | {:>13.4}s |",
        summary.latency_p95_seconds
    );
    println!(
        "| latency p99          | {:>13.4}s |",
        summary.latency_p99_seconds
    );
    println!(
        "| latency max          | {:>13.4}s |",
        summary.latency_max_seconds
    );
    println!("+----------------------+----------------+");
    println!();
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

fn unix_timestamp_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
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
