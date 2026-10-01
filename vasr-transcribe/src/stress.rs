use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args};
use serde::Serialize;
use tokio::task::JoinSet;

use crate::dataset::read_audio_input_dtos;
use crate::perf::timestamp_file_name;
use crate::{AudioInputDto, TranscribeRequest, TranscribeResponse, table::render_table};

#[derive(Debug, Args)]
pub struct StressArgs {
    /// Text file with one audio path/URL per line, or a vASR dataset directory / `.vasr` AudioDb.
    #[arg(short, long, value_name = "PATH")]
    input: PathBuf,

    /// Transcribe service endpoint.
    #[arg(long, default_value = "http://127.0.0.1:8867/transcribe")]
    server: String,

    /// HTTP concurrency.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Number of audio sources per request.
    #[arg(long = "batch-size", default_value_t = 1)]
    batch_size: usize,

    /// Stress test duration in seconds.
    #[arg(long = "duration-sec", default_value_t = 60)]
    duration_sec: u64,

    /// Override mono mode for every request item. Omit to let the server decide.
    #[arg(long, action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    mono: Option<bool>,

    /// Shuffle input sources once before cycling through them.
    #[arg(long, default_value_t = false)]
    shuffle: bool,

    /// HTTP request timeout in seconds.
    #[arg(long, default_value_t = 3600)]
    timeout_sec: u64,

    /// HTTP request retries.
    #[arg(long, default_value_t = 1)]
    retries: usize,

    /// GPU id for nvidia-smi memory sampling.
    #[arg(long, default_value_t = 0)]
    gpu: usize,

    /// GPU memory sampling interval.
    #[arg(long = "sample-interval-ms", default_value_t = 100)]
    sample_interval_ms: u64,

    /// Disable GPU memory sampling entirely.
    ///
    /// Sampling needs `nvidia-smi`, so on Apple Metal hosts it is skipped automatically;
    /// pass this to skip it explicitly (for example to silence the notice).
    #[arg(long = "no-gpu-sample", default_value_t = false)]
    no_gpu_sample: bool,

    /// Stop when an OOM response is observed.
    #[arg(long, default_value_t = false)]
    stop_on_oom: bool,

    /// Stop after this many failed requests. 0 means unlimited.
    #[arg(long, default_value_t = 0)]
    max_errors: usize,

    /// Directory used when `--output` is not provided.
    #[arg(long, default_value = "results", value_name = "DIR")]
    results_dir: PathBuf,

    /// Optional JSON result output path.
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct StressSummary {
    requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    oom_requests: usize,
    items: usize,
    bad_items: usize,
    concurrency: usize,
    batch_size: usize,
    wall_seconds: f64,
    requests_per_second: f64,
    items_per_second: f64,
    audio_seconds: f64,
    audio_seconds_per_second: f64,
    latency_p50_seconds: f64,
    latency_p90_seconds: f64,
    latency_p95_seconds: f64,
    latency_p99_seconds: f64,
    latency_max_seconds: f64,
}

#[derive(Debug, Clone, Serialize)]
struct StressGpuSummary {
    gpu: usize,
    samples: usize,
    base_used_mib: u64,
    peak_used_mib: u64,
    delta_used_mib: i64,
    avg_used_mib: f64,
    total_mib: u64,
}

#[derive(Debug, Clone, Serialize)]
struct GpuSample {
    elapsed_ms: u128,
    used_mib: u64,
    free_mib: u64,
    total_mib: u64,
}

#[derive(Debug, Clone, Serialize)]
struct StressConfig<'a> {
    input: &'a str,
    server: &'a str,
    concurrency: usize,
    batch_size: usize,
    duration_sec: u64,
    mono: Option<bool>,
    shuffle: bool,
    timeout_sec: u64,
    retries: usize,
    gpu: usize,
    sample_interval_ms: u64,
    gpu_sampling: bool,
    stop_on_oom: bool,
    max_errors: usize,
}

#[derive(Debug, Serialize)]
struct StressReport<'a> {
    timestamp: String,
    config: StressConfig<'a>,
    summary: StressSummary,
    gpu: StressGpuSummary,
    gpu_samples: Vec<GpuSample>,
    errors: Vec<String>,
}

#[derive(Debug)]
struct RequestOutcome {
    latency_seconds: f64,
    ok: bool,
    oom: bool,
    batch_size: usize,
    bad_items: usize,
    audio_seconds: f64,
    error: Option<String>,
}

#[derive(Debug, Default)]
struct StressStats {
    requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    oom_requests: usize,
    items: usize,
    bad_items: usize,
    audio_seconds: f64,
    latencies: Vec<f64>,
    errors: Vec<String>,
}

pub async fn run_stress(args: StressArgs) -> Result<()> {
    validate_args(&args)?;
    let mut sources = read_audio_input_dtos(&args.input, args.mono)?;
    if sources.is_empty() {
        bail!("input contains no audio sources: {}", args.input.display());
    }
    if args.shuffle {
        shuffle_sources(&mut sources);
    }
    let sources = Arc::new(sources);

    // GPU memory sampling needs `nvidia-smi`, which only exists on NVIDIA hosts. On Apple
    // Metal (or any host without it) degrade gracefully instead of aborting the run.
    let gpu_sampling = !args.no_gpu_sample && nvidia_smi_available();
    if gpu_sampling {
        sample_gpu_memory(args.gpu)
            .with_context(|| format!("failed to sample GPU {}", args.gpu))?;
    } else if !args.no_gpu_sample {
        println!(
            "stress NOTE   | nvidia-smi unavailable, GPU memory sampling disabled (use --no-gpu-sample to silence)"
        );
    }

    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()
        .context("failed to build HTTP client")?;

    println!(
        "stress START | input={} items={} server={} batch={} concurrency={} duration={}s mono={}",
        args.input.display(),
        sources.len(),
        args.server,
        args.batch_size,
        args.concurrency,
        args.duration_sec,
        format_mono(args.mono)
    );

    let started = Instant::now();
    let deadline = started + Duration::from_secs(args.duration_sec);
    let stop = Arc::new(AtomicBool::new(false));
    let next_source = Arc::new(AtomicUsize::new(0));
    let stats = Arc::new(Mutex::new(StressStats::default()));
    let gpu_samples = Arc::new(Mutex::new(Vec::new()));

    let sampler = if gpu_sampling {
        Some(tokio::spawn(spawn_gpu_sampler(
            args.gpu,
            started,
            Duration::from_millis(args.sample_interval_ms),
            Arc::clone(&stop),
            Arc::clone(&gpu_samples),
        )))
    } else {
        None
    };

    let mut workers = JoinSet::new();
    for _ in 0..args.concurrency {
        workers.spawn(stress_worker(
            client.clone(),
            args.server.clone(),
            Arc::clone(&sources),
            args.batch_size,
            args.retries,
            deadline,
            Arc::clone(&stop),
            Arc::clone(&next_source),
            Arc::clone(&stats),
            args.stop_on_oom,
            args.max_errors,
        ));
    }

    while let Some(result) = workers.join_next().await {
        result.context("stress worker task panicked")??;
    }
    stop.store(true, Ordering::Release);
    if let Some(sampler) = sampler {
        sampler.await.context("GPU sampler task panicked")??;
    }

    let wall_seconds = started.elapsed().as_secs_f64();
    let stats = take_mutex(stats, "stress stats")?;
    let samples = take_mutex(gpu_samples, "GPU samples")?;
    let summary = summarize(&stats, args.concurrency, args.batch_size, wall_seconds);
    let gpu = summarize_gpu(args.gpu, &samples);

    println!(
        "stress DONE  | requests={} ok={} failed={} oom={} elapsed={:.3}s rps={:.2}",
        summary.requests,
        summary.successful_requests,
        summary.failed_requests,
        summary.oom_requests,
        summary.wall_seconds,
        summary.requests_per_second
    );
    if gpu.samples > 0 {
        println!(
            "gpu MEM      | base={:.2}GiB peak={:.2}GiB delta={:.2}GiB avg={:.2}GiB samples={}",
            mib_to_gib(gpu.base_used_mib),
            mib_to_gib(gpu.peak_used_mib),
            gpu.delta_used_mib as f64 / 1024.0,
            gpu.avg_used_mib / 1024.0,
            gpu.samples
        );
    } else {
        println!("gpu MEM      | unavailable (no nvidia-smi)");
    }
    print_summary_table(&summary, &gpu);

    let timestamp = timestamp_file_name();
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.results_dir.join(format!("stress_{timestamp}.json")));
    let input = args.input.to_string_lossy().to_string();
    save_report(
        &output,
        &StressReport {
            timestamp,
            config: StressConfig {
                input: &input,
                server: &args.server,
                concurrency: args.concurrency,
                batch_size: args.batch_size,
                duration_sec: args.duration_sec,
                mono: args.mono,
                shuffle: args.shuffle,
                timeout_sec: args.timeout_sec,
                retries: args.retries,
                gpu: args.gpu,
                sample_interval_ms: args.sample_interval_ms,
                gpu_sampling,
                stop_on_oom: args.stop_on_oom,
                max_errors: args.max_errors,
            },
            summary,
            gpu,
            gpu_samples: samples,
            errors: stats.errors,
        },
    )?;
    println!("[saved] {}", output.display());
    Ok(())
}

fn validate_args(args: &StressArgs) -> Result<()> {
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }
    if args.concurrency == 0 {
        bail!("--concurrency must be greater than 0");
    }
    if args.batch_size == 0 {
        bail!("--batch-size must be greater than 0");
    }
    if args.duration_sec == 0 {
        bail!("--duration-sec must be greater than 0");
    }
    if args.sample_interval_ms == 0 {
        bail!("--sample-interval-ms must be greater than 0");
    }
    if args.retries == 0 {
        bail!("--retries must be greater than 0");
    }
    Ok(())
}

fn shuffle_sources(sources: &mut [AudioInputDto]) {
    let mut state = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    for i in (1..sources.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        sources.swap(i, (state as usize) % (i + 1));
    }
}

#[allow(clippy::too_many_arguments)]
async fn stress_worker(
    client: reqwest::Client,
    server: String,
    sources: Arc<Vec<AudioInputDto>>,
    batch_size: usize,
    retries: usize,
    deadline: Instant,
    stop: Arc<AtomicBool>,
    next_source: Arc<AtomicUsize>,
    stats: Arc<Mutex<StressStats>>,
    stop_on_oom: bool,
    max_errors: usize,
) -> Result<()> {
    while Instant::now() < deadline && !stop.load(Ordering::Acquire) {
        let request = next_request(&sources, batch_size, &next_source);
        let outcome = run_one_request(&client, &server, request, retries).await;
        let should_stop = record_outcome(&stats, outcome, stop_on_oom, max_errors)?;
        if should_stop {
            stop.store(true, Ordering::Release);
        }
    }
    Ok(())
}

fn next_request(
    sources: &[AudioInputDto],
    batch_size: usize,
    next_source: &AtomicUsize,
) -> TranscribeRequest {
    let start = next_source.fetch_add(batch_size, Ordering::Relaxed);
    let inputs = (0..batch_size)
        .map(|offset| sources[(start + offset) % sources.len()].clone())
        .collect();
    TranscribeRequest { inputs }
}

async fn run_one_request(
    client: &reqwest::Client,
    server: &str,
    request: TranscribeRequest,
    retries: usize,
) -> RequestOutcome {
    let batch_size = request.inputs.len();
    let mut last_error = None;
    for attempt in 0..retries {
        let started = Instant::now();
        match client.post(server).json(&request).send().await {
            Ok(response) if response.status().is_success() => {
                let latency_seconds = started.elapsed().as_secs_f64();
                match response.json::<TranscribeResponse>().await {
                    Ok(response) => {
                        let bad_items = response.data.iter().filter(|item| item.is_bad).count();
                        let error_text = response
                            .data
                            .iter()
                            .filter_map(|item| item.bad_reason.as_deref())
                            .collect::<Vec<_>>()
                            .join(" | ");
                        let oom = contains_oom(&error_text);
                        return RequestOutcome {
                            latency_seconds,
                            ok: !oom,
                            oom,
                            batch_size,
                            bad_items,
                            audio_seconds: response
                                .inference_performance
                                .total_audio_duration_seconds,
                            error: if error_text.is_empty() {
                                None
                            } else {
                                Some(error_text)
                            },
                        };
                    }
                    Err(error) => last_error = Some(format!("decode response JSON: {error}")),
                }
            }
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                last_error = Some(format!("status={status} body={}", truncate(&body)));
            }
            Err(error) => last_error = Some(error.to_string()),
        }
        if attempt + 1 < retries {
            tokio::time::sleep(Duration::from_millis(100 * (attempt as u64 + 1))).await;
        }
    }

    let error = last_error.unwrap_or_else(|| "request failed".to_string());
    RequestOutcome {
        latency_seconds: 0.0,
        ok: false,
        oom: contains_oom(&error),
        batch_size,
        bad_items: batch_size,
        audio_seconds: 0.0,
        error: Some(error),
    }
}

fn record_outcome(
    stats: &Mutex<StressStats>,
    outcome: RequestOutcome,
    stop_on_oom: bool,
    max_errors: usize,
) -> Result<bool> {
    let mut stats = stats
        .lock()
        .map_err(|_| anyhow!("stress stats lock poisoned"))?;
    stats.requests = stats.requests.saturating_add(1);
    stats.items = stats.items.saturating_add(outcome.batch_size);
    stats.bad_items = stats.bad_items.saturating_add(outcome.bad_items);
    stats.audio_seconds += outcome.audio_seconds;
    if outcome.ok {
        stats.successful_requests = stats.successful_requests.saturating_add(1);
        stats.latencies.push(outcome.latency_seconds);
    } else {
        stats.failed_requests = stats.failed_requests.saturating_add(1);
        if stats.errors.len() < 20
            && let Some(error) = outcome.error.as_deref()
        {
            stats.errors.push(error.to_string());
        }
    }
    if outcome.oom {
        stats.oom_requests = stats.oom_requests.saturating_add(1);
    }
    Ok((stop_on_oom && outcome.oom) || (max_errors > 0 && stats.failed_requests >= max_errors))
}

/// Whether `nvidia-smi` can be executed on this host.
///
/// Metal/Apple hosts have no `nvidia-smi`, so GPU memory sampling must be skipped rather
/// than treated as a fatal error.
fn nvidia_smi_available() -> bool {
    Command::new("nvidia-smi")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// Render a GPU memory figure in GiB, or `n/a` when no samples were collected.
fn format_gpu_gib(samples: usize, mib: u64) -> String {
    if samples == 0 {
        "n/a".to_string()
    } else {
        format!("{:.2}GiB", mib_to_gib(mib))
    }
}

/// Poll GPU memory until `stop` is set.
///
/// A sampling failure stops sampling instead of failing the whole stress run, so a
/// transient `nvidia-smi` problem cannot discard an otherwise valid result.
async fn spawn_gpu_sampler(
    gpu: usize,
    started: Instant,
    interval: Duration,
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<GpuSample>>>,
) -> Result<()> {
    loop {
        match sample_gpu_memory(gpu) {
            Ok(sample) => {
                samples
                    .lock()
                    .map_err(|_| anyhow!("GPU samples lock poisoned"))?
                    .push(GpuSample {
                        elapsed_ms: started.elapsed().as_millis(),
                        used_mib: sample.0,
                        free_mib: sample.1,
                        total_mib: sample.2,
                    });
            }
            Err(err) => {
                eprintln!("gpu MEM      | sampling stopped: {err:#}");
                return Ok(());
            }
        }
        if stop.load(Ordering::Acquire) {
            break;
        }
        tokio::time::sleep(interval).await;
    }
    Ok(())
}

fn sample_gpu_memory(gpu: usize) -> Result<(u64, u64, u64)> {
    let output = Command::new("nvidia-smi")
        .arg("--id")
        .arg(gpu.to_string())
        .arg("--query-gpu=memory.used,memory.free,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
        .context("failed to run nvidia-smi")?;
    if !output.status.success() {
        bail!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .next()
        .ok_or_else(|| anyhow!("nvidia-smi returned no memory row"))?;
    let mut parts = line.split(',').map(str::trim);
    let used = parse_mib(parts.next(), "memory.used")?;
    let free = parse_mib(parts.next(), "memory.free")?;
    let total = parse_mib(parts.next(), "memory.total")?;
    Ok((used, free, total))
}

fn parse_mib(value: Option<&str>, field: &str) -> Result<u64> {
    value
        .ok_or_else(|| anyhow!("nvidia-smi missing {field}"))?
        .parse::<u64>()
        .with_context(|| format!("invalid nvidia-smi {field}"))
}

fn summarize(
    stats: &StressStats,
    concurrency: usize,
    batch_size: usize,
    wall_seconds: f64,
) -> StressSummary {
    let mut latencies = stats.latencies.clone();
    latencies.sort_by(f64::total_cmp);
    StressSummary {
        requests: stats.requests,
        successful_requests: stats.successful_requests,
        failed_requests: stats.failed_requests,
        oom_requests: stats.oom_requests,
        items: stats.items,
        bad_items: stats.bad_items,
        concurrency,
        batch_size,
        wall_seconds: round4(wall_seconds),
        requests_per_second: round4(rate(stats.requests, wall_seconds)),
        items_per_second: round4(rate(stats.items, wall_seconds)),
        audio_seconds: round4(stats.audio_seconds),
        audio_seconds_per_second: round4(stats.audio_seconds / wall_seconds.max(f64::EPSILON)),
        latency_p50_seconds: round4(percentile(&latencies, 50.0)),
        latency_p90_seconds: round4(percentile(&latencies, 90.0)),
        latency_p95_seconds: round4(percentile(&latencies, 95.0)),
        latency_p99_seconds: round4(percentile(&latencies, 99.0)),
        latency_max_seconds: round4(latencies.last().copied().unwrap_or(0.0)),
    }
}

fn summarize_gpu(gpu: usize, samples: &[GpuSample]) -> StressGpuSummary {
    let base = samples.first().map(|sample| sample.used_mib).unwrap_or(0);
    let peak = samples
        .iter()
        .map(|sample| sample.used_mib)
        .max()
        .unwrap_or(0);
    let total = samples.last().map(|sample| sample.total_mib).unwrap_or(0);
    let avg = if samples.is_empty() {
        0.0
    } else {
        samples
            .iter()
            .map(|sample| sample.used_mib as f64)
            .sum::<f64>()
            / samples.len() as f64
    };
    StressGpuSummary {
        gpu,
        samples: samples.len(),
        base_used_mib: base,
        peak_used_mib: peak,
        delta_used_mib: peak as i64 - base as i64,
        avg_used_mib: round4(avg),
        total_mib: total,
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((pct / 100.0) * (sorted.len().saturating_sub(1) as f64)).ceil() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn rate(count: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        count as f64 / seconds
    } else {
        0.0
    }
}

fn contains_oom(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    text.contains("cuda_error_out_of_memory") || text.contains("out of memory")
}

fn truncate(text: &str) -> String {
    const MAX: usize = 512;
    if text.len() <= MAX {
        text.to_string()
    } else {
        format!("{}...", &text[..MAX])
    }
}

fn print_summary_table(summary: &StressSummary, gpu: &StressGpuSummary) {
    let table = render_table(
        ["metric", "value"],
        [
            vec!["requests".to_string(), summary.requests.to_string()],
            vec![
                "successful".to_string(),
                summary.successful_requests.to_string(),
            ],
            vec!["failed".to_string(), summary.failed_requests.to_string()],
            vec!["oom".to_string(), summary.oom_requests.to_string()],
            vec!["items".to_string(), summary.items.to_string()],
            vec!["bad items".to_string(), summary.bad_items.to_string()],
            vec!["concurrency".to_string(), summary.concurrency.to_string()],
            vec!["batch size".to_string(), summary.batch_size.to_string()],
            vec![
                "elapsed".to_string(),
                format!("{:.3}s", summary.wall_seconds),
            ],
            vec![
                "requests/s".to_string(),
                format!("{:.3}", summary.requests_per_second),
            ],
            vec![
                "items/s".to_string(),
                format!("{:.3}", summary.items_per_second),
            ],
            vec![
                "audio".to_string(),
                format!("{:.1}s", summary.audio_seconds),
            ],
            vec![
                "audio speed".to_string(),
                format!("{:.1}x", summary.audio_seconds_per_second),
            ],
            vec![
                "latency p50".to_string(),
                format!("{:.3}s", summary.latency_p50_seconds),
            ],
            vec![
                "latency p95".to_string(),
                format!("{:.3}s", summary.latency_p95_seconds),
            ],
            vec![
                "latency max".to_string(),
                format!("{:.3}s", summary.latency_max_seconds),
            ],
            vec!["gpu".to_string(), gpu.gpu.to_string()],
            vec![
                "gpu base".to_string(),
                format_gpu_gib(gpu.samples, gpu.base_used_mib),
            ],
            vec![
                "gpu peak".to_string(),
                format_gpu_gib(gpu.samples, gpu.peak_used_mib),
            ],
            vec![
                "gpu delta".to_string(),
                if gpu.samples == 0 {
                    "n/a".to_string()
                } else {
                    format!("{:.2}GiB", gpu.delta_used_mib as f64 / 1024.0)
                },
            ],
        ],
    );
    println!("\n{table}\n");
}

fn save_report(path: &Path, report: &StressReport<'_>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(report).context("failed to encode report JSON")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn take_mutex<T>(mutex: Arc<Mutex<T>>, name: &str) -> Result<T> {
    Arc::try_unwrap(mutex)
        .map_err(|_| anyhow!("{name} still has references"))?
        .into_inner()
        .map_err(|_| anyhow!("{name} lock poisoned"))
}

fn format_mono(mono: Option<bool>) -> &'static str {
    match mono {
        Some(true) => "true",
        Some(false) => "false",
        None => "auto",
    }
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn mib_to_gib(mib: u64) -> f64 {
    mib as f64 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oom_detector_matches_cuda_errors() {
        assert!(contains_oom("CUDA_ERROR_OUT_OF_MEMORY"));
        assert!(contains_oom("cuda error: out of memory"));
        assert!(!contains_oom("connection refused"));
    }

    fn sample_input(url: &str, mono: Option<bool>) -> AudioInputDto {
        AudioInputDto {
            url: Some(url.to_string()),
            b64_str: None,
            modelscope: None,
            mono,
            hotword: None,
        }
    }

    #[test]
    fn next_request_cycles_sources() {
        let sources = vec![
            sample_input("a.wav", Some(true)),
            sample_input("b.wav", Some(true)),
        ];
        let next = AtomicUsize::new(1);
        let request = next_request(&sources, 3, &next);
        let urls = request
            .inputs
            .iter()
            .map(|input| input.url.as_deref().unwrap_or_default())
            .collect::<Vec<_>>();

        assert_eq!(urls, ["b.wav", "a.wav", "b.wav"]);
        assert_eq!(request.inputs[0].mono, Some(true));
    }

    #[test]
    fn shuffle_preserves_sources() {
        let mut sources = vec![
            sample_input("a", None),
            sample_input("b", None),
            sample_input("c", None),
        ];
        shuffle_sources(&mut sources);
        let mut urls = sources
            .iter()
            .map(|input| input.url.clone().unwrap_or_default())
            .collect::<Vec<_>>();
        urls.sort();

        assert_eq!(urls, ["a", "b", "c"]);
    }
}
