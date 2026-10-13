use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Args};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use tokio::task::JoinSet;

use crate::eval::{EvalNormalizer, EvalTimeline, evaluate_perf_output, has_reference};
use crate::inputs::{audio_input_from_collected, collect_inputs};
use crate::protocol::{audio_channel_text, audio_is_bad, audio_seconds};
use crate::quality::detect_repeated_pattern;
use crate::{AudioInputDto, TranscribeRequest, table::render_table};
use asr_data::{Audio, AudioChannel};

#[derive(Debug, Args)]
pub struct PerfArgs {
    /// Audio file, directory, URL, or vASR AudioDb (`.db` / `.vasr` / `.vasr.msgpack`).
    #[arg(short, long, value_name = "INPUT")]
    input: String,

    /// Transcribe service endpoint.
    #[arg(long, default_value = "http://127.0.0.1:8867/transcribe")]
    server: String,

    /// Recursively scan subdirectories when `--input` is a directory.
    #[arg(long, default_value_t = false)]
    recursive: bool,

    /// HTTP concurrency. Use 0 to auto-select the machine CPU count.
    #[arg(long, default_value_t = 0)]
    concurrency: usize,

    /// Number of audio sources per request.
    #[arg(long = "batch-size", default_value_t = 1)]
    batch_size: usize,

    /// Process at most this many audio sources, preserving input order.
    #[arg(long)]
    limit: Option<usize>,

    /// Override mono mode for every request item. Omit to let the server decide.
    #[arg(long, action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    mono: Option<bool>,

    /// Audio timeline/channel to evaluate. CER runs automatically when references exist.
    #[arg(long, value_enum, default_value_t = EvalTimeline::Mono)]
    timeline: EvalTimeline,

    /// Text normalization used before CER.
    #[arg(long, value_enum, default_value_t = EvalNormalizer::Bench)]
    normalizer: EvalNormalizer,

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

/// 写入 perf 结果的单条音频引用。
///
/// 只保留可再次定位音频的地址和请求参数。base64、`data:` URI 等原始音频载荷不进入该结构。
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PerfSavedInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    info: Option<asr_data::AudioInfo>,
    /// 远程 URL 或本地路径。没有可保存的地址时省略。
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    /// ModelScope identity allows replay without persisting audio bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    modelscope: Option<crate::ModelScopeSourceDto>,
    /// 请求里显式指定的 mono 模式。
    #[serde(skip_serializing_if = "Option::is_none")]
    mono: Option<bool>,
    /// 请求里的热词。
    #[serde(skip_serializing_if = "Option::is_none")]
    hotword: Option<String>,
}

/// 一次请求（`perf` 的 HTTP 请求，或 `run` 的单个进程内批次）的结果。
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PerfRequestResult {
    pub(crate) request_index: usize,
    pub(crate) batch_size: usize,
    pub(crate) inputs: Vec<PerfSavedInput>,
    pub(crate) latency_seconds: f64,
    pub(crate) ok: bool,
    pub(crate) bad_items: usize,
    pub(crate) quality_passed_items: usize,
    pub(crate) quality_audio_seconds: f64,
    pub(crate) quality_anomalies: Vec<PerfQualityAnomaly>,
    pub(crate) audio_seconds: f64,
    /// 服务端返回的原始 `Audio` 记录（wire format 即 Audio）。
    pub(crate) data: Vec<Audio>,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PerfQualityAnomaly {
    item_index: usize,
    reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PerfSummary {
    requests: usize,
    successful_requests: usize,
    failed_requests: usize,
    items: usize,
    bad_items: usize,
    quality_passed_items: usize,
    quality_audio_seconds: f64,
    quality_audio_seconds_per_second: f64,
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

/// `perf` / `run` 共用的报告结构。
///
/// `service_url` 只在 `perf`（HTTP 客户端）里出现；`run` 是进程内执行，序列化时省略。
#[derive(Debug, Serialize)]
pub(crate) struct PerfReport<'a> {
    pub(crate) timestamp: String,
    pub(crate) input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_url: Option<&'a str>,
    pub(crate) items: usize,
    pub(crate) summary: PerfSummary,
    pub(crate) requests: Vec<PerfRequestResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cer: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
struct PerfRunReport {
    summary: PerfSummary,
    requests: Vec<PerfRequestResult>,
}

pub async fn run_perf(args: PerfArgs) -> Result<()> {
    validate_args(&args)?;
    let mut collected = collect_inputs(&args.input, args.recursive)?;
    if let Some(limit) = args.limit {
        collected.truncate(limit);
    }
    if collected.is_empty() {
        bail!("input contains no audio sources: {}", args.input);
    }
    let request_mono = resolve_request_mono(&args)?;
    let sources = collected
        .iter()
        .enumerate()
        .map(|(index, input)| audio_input_from_collected(input, request_mono, index))
        .collect::<Vec<_>>();

    // 只有 audiodb 输入携带参考文本；存在参考时自动评估 CER。
    let records = collected
        .iter()
        .map(|input| input.record.clone())
        .collect::<Option<Vec<_>>>();
    let has_references = records.as_ref().is_some_and(|records| {
        records
            .iter()
            .any(|record| has_reference(record, args.timeline))
    });

    let concurrency = resolve_concurrency(args.concurrency);
    let batch_size = args.batch_size;

    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()
        .context("failed to build HTTP client")?;

    println!(
        "perf START | input={} items={} server={} batch={} concurrency={} mono={} timeline={}",
        args.input,
        sources.len(),
        args.server,
        batch_size,
        concurrency,
        format_mono(request_mono),
        args.timeline.name()
    );
    let run = run_one_perf_trial(
        &client,
        &args.server,
        &sources,
        concurrency,
        batch_size,
        args.retries,
    )
    .await
    .with_context(|| format!("perf failed: batch_size={batch_size} concurrency={concurrency}"))?;
    let summary = run.summary.clone();
    let cer = if has_references {
        let records = records.unwrap_or_default();
        Some(evaluate_cer(
            &records,
            &run.requests,
            &summary,
            args.timeline,
            args.normalizer,
        )?)
    } else {
        println!(
            "perf CER  | skipped: no reference transcripts for timeline {}",
            args.timeline.name()
        );
        None
    };
    println!(
        "perf DONE  | items={} bad={} quality_passed={} failed={} elapsed={:.3}s audio={:.1}s speed={:.1}x quality_speed={:.1}x",
        summary.items,
        summary.bad_items,
        summary.quality_passed_items,
        summary.failed_requests,
        summary.wall_seconds,
        summary.audio_seconds,
        summary.audio_seconds_per_second,
        summary.quality_audio_seconds_per_second,
    );
    print_perf_summary_table(&summary, cer.as_ref());
    if let Some(error) = cer
        .as_ref()
        .and_then(|report| report.get("error"))
        .and_then(serde_json::Value::as_str)
    {
        println!("perf CER  | failed: {error}");
    }

    let timestamp = timestamp_file_name();
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.results_dir.join(format!("perf_{timestamp}.json")));
    save_report(
        &output,
        &PerfReport {
            timestamp,
            input: args.input.as_str(),
            service_url: Some(args.server.as_str()),
            items: sources.len(),
            summary,
            requests: run.requests,
            cer,
        },
    )?;
    println!("[saved] {}", output.display());
    Ok(())
}

fn validate_args(args: &PerfArgs) -> Result<()> {
    if args.retries == 0 {
        bail!("--retries must be greater than 0");
    }
    if args.batch_size == 0 {
        bail!("--batch-size must be greater than 0");
    }
    Ok(())
}

/// `--timeline` 选左/右声道时请求必须是 stereo；`--mono` 只能在 mono 时间轴上覆盖。
fn resolve_request_mono(args: &PerfArgs) -> Result<Option<bool>> {
    if args.timeline == EvalTimeline::Mono {
        return Ok(args.mono);
    }
    if args.mono == Some(true) {
        bail!(
            "--mono=true conflicts with --timeline {}; stereo output is required",
            args.timeline.name()
        );
    }
    Ok(Some(false))
}

/// 把请求输出按输入顺序拼回数据集记录，自动评估 CER。
pub(crate) fn evaluate_cer(
    records: &[asr_data::Audio],
    requests: &[PerfRequestResult],
    summary: &PerfSummary,
    timeline: EvalTimeline,
    normalizer: EvalNormalizer,
) -> Result<serde_json::Value> {
    let mut ordered = requests.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|request| request.request_index);
    if ordered
        .iter()
        .any(|request| !request.ok || request.data.len() != request.batch_size)
    {
        return Ok(serde_json::json!({"error": "incomplete responses; CER not accepted"}));
    }
    let data: Vec<Audio> = ordered
        .into_iter()
        .flat_map(|request| request.data.clone())
        .collect();
    match evaluate_perf_output(
        records,
        data,
        summary.wall_seconds,
        summary.audio_seconds,
        timeline,
        normalizer,
    ) {
        Ok(value) => Ok(value),
        Err(error) => Ok(serde_json::json!({"error": error.to_string()})),
    }
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
    sources: &[AudioInputDto],
    concurrency: usize,
    batch_size: usize,
    retries: usize,
) -> Result<PerfRunReport> {
    let batches = build_batches(sources, batch_size);
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

fn build_batches(sources: &[AudioInputDto], batch_size: usize) -> Vec<TranscribeRequest> {
    sources
        .chunks(batch_size)
        .map(|chunk| TranscribeRequest {
            inputs: chunk.to_vec(),
        })
        .collect()
}

/// 一批完成输出（服务端响应或进程内结果）的质量与失败统计。
pub(crate) struct AudioQualityStats {
    pub(crate) quality_anomalies: Vec<PerfQualityAnomaly>,
    pub(crate) quality_passed_items: usize,
    pub(crate) quality_audio_seconds: f64,
    pub(crate) bad_items: usize,
    pub(crate) audio_seconds: f64,
}

/// 判定一批 `Audio` 输出的异常、坏项和有效音频时长。
pub(crate) fn audio_quality_stats(audios: &[Audio], batch_size: usize) -> AudioQualityStats {
    let data_count = audios.len();
    let mapping_complete = data_count == batch_size;
    let mut quality_anomalies = Vec::new();
    if !mapping_complete {
        quality_anomalies.push(PerfQualityAnomaly {
            item_index: data_count.min(batch_size),
            reason: format!("item count mismatch: expected {batch_size}, got {data_count}"),
        });
    }
    for (item_index, audio) in audios.iter().enumerate() {
        let texts = [
            audio_channel_text(audio, AudioChannel::Mono),
            audio_channel_text(audio, AudioChannel::Left),
            audio_channel_text(audio, AudioChannel::Right),
        ];
        let non_empty = texts.iter().filter(|text| !text.trim().is_empty());
        let anomaly = if audio_seconds(audio) <= 0.0 {
            Some("missing or invalid audio duration".to_string())
        } else if non_empty.clone().next().is_none() {
            Some("empty ASR output for non-empty audio".to_string())
        } else {
            non_empty
                .filter_map(|text| detect_repeated_pattern(text))
                .next()
        };
        if let Some(reason) = anomaly {
            quality_anomalies.push(PerfQualityAnomaly { item_index, reason });
        }
    }
    let has_anomaly = |index: usize| {
        quality_anomalies
            .iter()
            .any(|anomaly| anomaly.item_index == index)
    };
    let quality_passed_items = if mapping_complete {
        audios
            .iter()
            .enumerate()
            .filter(|(index, audio)| !audio_is_bad(audio) && !has_anomaly(*index))
            .count()
    } else {
        0
    };
    let quality_audio_seconds = if mapping_complete {
        audios
            .iter()
            .enumerate()
            .filter(|(index, audio)| !audio_is_bad(audio) && !has_anomaly(*index))
            .map(|(_, audio)| audio_seconds(audio))
            .sum()
    } else {
        0.0
    };
    let bad_items = if mapping_complete {
        audios
            .iter()
            .enumerate()
            .filter(|(index, audio)| audio_is_bad(audio) || has_anomaly(*index))
            .count()
    } else {
        batch_size
    };
    AudioQualityStats {
        quality_anomalies,
        quality_passed_items,
        quality_audio_seconds,
        bad_items,
        audio_seconds: audios.iter().map(audio_seconds).sum(),
    }
}

async fn run_one_request(
    client: reqwest::Client,
    server: String,
    request_index: usize,
    request: TranscribeRequest,
    retries: usize,
) -> Result<PerfRequestResult> {
    let batch_size = request.inputs.len();
    // 结果只记音频引用，避免把 base64 等原始音频写进 JSON。
    let inputs = saved_inputs(&request.inputs);
    let mut last_error = None;
    for attempt in 0..retries {
        let start = Instant::now();
        match client.post(&server).json(&request).send().await {
            Ok(response) if response.status().is_success() => {
                let body = response
                    .text()
                    .await
                    .context("failed to read transcribe JSONL response")?;
                let latency_seconds = start.elapsed().as_secs_f64();
                let audios = crate::protocol::parse_transcribe_audios(&body)
                    .context("failed to decode transcribe response JSONL")?;
                let stats = audio_quality_stats(&audios, batch_size);
                return Ok(PerfRequestResult {
                    request_index,
                    batch_size,
                    inputs,
                    latency_seconds,
                    ok: true,
                    bad_items: stats.bad_items,
                    quality_passed_items: stats.quality_passed_items,
                    quality_audio_seconds: stats.quality_audio_seconds,
                    quality_anomalies: stats.quality_anomalies,
                    audio_seconds: stats.audio_seconds,
                    data: audios,
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
        inputs,
        latency_seconds: 0.0,
        ok: false,
        bad_items: batch_size,
        quality_passed_items: 0,
        quality_audio_seconds: 0.0,
        quality_anomalies: vec![PerfQualityAnomaly {
            item_index: 0,
            reason: "request failed".to_string(),
        }],
        audio_seconds: 0.0,
        data: Vec::new(),
        error: Some(
            last_error
                .map(|error| error.to_string())
                .unwrap_or_else(|| "request failed".to_string()),
        ),
    })
}

pub(crate) fn summarize(
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
    let quality_passed_items: usize = results
        .iter()
        .map(|result| result.quality_passed_items)
        .sum();
    let quality_audio_seconds = if quality_passed_items == 0 {
        0.0
    } else {
        results
            .iter()
            .map(|result| result.quality_audio_seconds)
            .sum()
    };
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
    let quality_audio_seconds_per_second = if wall_seconds > 0.0 {
        quality_audio_seconds / wall_seconds
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
        quality_passed_items,
        quality_audio_seconds: round4(quality_audio_seconds),
        quality_audio_seconds_per_second: round4(quality_audio_seconds_per_second),
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

fn print_perf_summary_table(summary: &PerfSummary, cer: Option<&serde_json::Value>) {
    let cer_summary = cer.and_then(|report| report.get("summary"));
    let cer_percent = |key: &str| {
        cer_summary
            .and_then(|summary| summary[key].as_f64())
            .map(|value| format!("{:.4}%", value * 100.0))
            .unwrap_or_else(|| "n/a".to_string())
    };
    let table = render_table(
        ["metric", "value"],
        [
            vec!["items".to_string(), summary.items.to_string()],
            vec!["bad items".to_string(), summary.bad_items.to_string()],
            vec![
                "quality passed".to_string(),
                summary.quality_passed_items.to_string(),
            ],
            vec!["weighted CER".to_string(), cer_percent("cer_weighted")],
            vec!["average CER".to_string(), cer_percent("cer_average")],
            vec![
                "CER labeled".to_string(),
                cer_summary
                    .and_then(|summary| {
                        Some(format!(
                            "{}/{}",
                            summary["labeled_items"].as_u64()?,
                            summary["total_items"].as_u64()?
                        ))
                    })
                    .unwrap_or_else(|| "n/a".to_string()),
            ],
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
            vec![
                "quality audio".to_string(),
                format!("{:.1}s", summary.quality_audio_seconds),
            ],
            vec![
                "quality speed".to_string(),
                format!("{:.1}x", summary.quality_audio_seconds_per_second),
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
        ],
    );
    println!("\n{table}\n");
}

pub(crate) fn save_report(path: &Path, report: &PerfReport<'_>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(report).context("failed to encode report JSON")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub(crate) fn timestamp_file_name() -> String {
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

/// 从请求输入抽出可写入结果的引用，丢掉原始音频载荷。
///
/// `url` 中的 HTTP(S)、`file://` 和本地路径会保留。`b64_str`，以及写在 `url` 里的
/// `data:` 内嵌音频，不会保存。
pub(crate) fn saved_inputs(inputs: &[AudioInputDto]) -> Vec<PerfSavedInput> {
    inputs
        .iter()
        .map(|input| PerfSavedInput {
            id: input.id.clone(),
            info: input.info.clone(),
            url: persistable_audio_ref(input.url.as_deref()),
            modelscope: input.modelscope.clone(),
            mono: input.mono,
            hotword: input.hotword.clone(),
        })
        .collect()
}

/// 保留音频地址；空串和内嵌音频返回 `None`。
fn persistable_audio_ref(value: Option<&str>) -> Option<String> {
    let value = value?.trim();
    if value.is_empty() || is_embedded_audio(value) {
        return None;
    }
    Some(value.to_string())
}

/// `data:` URI 与 base64 一样是原始音频，不能写入结果。
fn is_embedded_audio(value: &str) -> bool {
    value.trim_start().starts_with("data:")
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
    use super::{
        PerfArgs, PerfReport, format_utc_timestamp, resolve_request_mono, saved_inputs, summarize,
    };
    use crate::AudioInputDto;
    use clap::Parser;

    #[test]
    fn report_is_shared_by_perf_and_run() {
        let report = PerfReport {
            timestamp: "20260101_000000".to_string(),
            input: "a.wav",
            service_url: Some("http://127.0.0.1:8867/transcribe"),
            items: 0,
            summary: summarize(1, 0, 0.0, &[]),
            requests: Vec::new(),
            cer: None,
        };
        let json = serde_json::to_value(&report).expect("serialize perf report");
        assert_eq!(json["service_url"], "http://127.0.0.1:8867/transcribe");
        assert!(json.get("cer").is_none());

        let local = PerfReport {
            timestamp: "20260101_000000".to_string(),
            input: "a.wav",
            service_url: None,
            items: 0,
            summary: summarize(1, 0, 0.0, &[]),
            requests: Vec::new(),
            cer: None,
        };
        let json = serde_json::to_value(&local).expect("serialize run report");
        assert!(json.get("service_url").is_none());
    }

    #[derive(Debug, Parser)]
    struct TestPerfCli {
        #[command(flatten)]
        args: PerfArgs,
    }

    fn perf_args(extra: &[&str]) -> PerfArgs {
        let mut argv = vec!["perf", "-i", "a.wav"];
        argv.extend_from_slice(extra);
        TestPerfCli::try_parse_from(argv)
            .expect("perf args should parse")
            .args
    }

    #[test]
    fn timeline_left_forces_stereo_and_conflicts_with_mono_true() {
        assert_eq!(resolve_request_mono(&perf_args(&[])).unwrap(), None);
        assert_eq!(
            resolve_request_mono(&perf_args(&["--mono", "true"])).unwrap(),
            Some(true)
        );
        assert_eq!(
            resolve_request_mono(&perf_args(&["--timeline", "left"])).unwrap(),
            Some(false)
        );
        assert!(
            resolve_request_mono(&perf_args(&["--timeline", "left", "--mono", "true"])).is_err()
        );
    }

    #[test]
    fn timestamp_file_name_uses_utc_calendar_format() {
        assert_eq!(format_utc_timestamp(0), "19700101_000000");
        assert_eq!(format_utc_timestamp(1_782_720_711), "20260629_081151");
    }

    #[test]
    fn saved_inputs_keep_audio_refs_and_drop_payloads() {
        let inputs = vec![
            AudioInputDto {
                id: None,
                info: None,
                url: Some("https://example.com/a.wav".to_string()),
                b64_str: Some("AAAA".to_string()),
                modelscope: None,
                mono: Some(true),
                hotword: None,
            },
            AudioInputDto {
                id: None,
                info: None,
                url: Some("/data/a.wav".to_string()),
                b64_str: None,
                modelscope: None,
                mono: None,
                hotword: Some("热词".to_string()),
            },
            AudioInputDto {
                id: None,
                info: None,
                url: None,
                b64_str: Some("UklGRg==".to_string()),
                modelscope: None,
                mono: Some(false),
                hotword: None,
            },
            AudioInputDto {
                id: None,
                info: None,
                url: Some("data:audio/wav;base64,UklGRg==".to_string()),
                b64_str: None,
                modelscope: None,
                mono: None,
                hotword: None,
            },
        ];

        let json = serde_json::to_value(saved_inputs(&inputs)).expect("saved inputs serialize");
        let encoded = json.to_string();
        assert!(!encoded.contains("b64_str"));
        assert!(!encoded.contains("UklGRg"));
        assert!(!encoded.contains("AAAA"));
        assert_eq!(json[0]["url"], "https://example.com/a.wav");
        assert_eq!(json[0]["mono"], true);
        assert_eq!(json[1]["url"], "/data/a.wav");
        assert_eq!(json[1]["hotword"], "热词");
        assert!(json[2].get("url").is_none());
        assert!(json[3].get("url").is_none());
    }

    #[test]
    fn saved_inputs_keep_modelscope_identity_for_replay() {
        let source = crate::ModelScopeSourceDto {
            repo_id: "owner/dataset".into(),
            file_path: "nested/audio.wav".into(),
            revision: Some("immutable-revision".into()),
        };
        let input = AudioInputDto {
            id: None,
            info: None,
            url: None,
            b64_str: None,
            modelscope: Some(source.clone()),
            mono: Some(true),
            hotword: None,
        };
        let json = serde_json::to_value(saved_inputs(&[input])).unwrap();
        assert_eq!(json[0]["modelscope"], serde_json::to_value(source).unwrap());
        assert!(json[0].get("b64_str").is_none());
    }
}
