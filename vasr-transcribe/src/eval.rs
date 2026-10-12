use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::dataset::{audio_input_from_source, read_vasr_audios};
use crate::{TranscribeRequest, TranscribeResponse};
use anyhow::{Context, Result, anyhow, bail};
use asr_data::{Audio, AudioChannel, AudioSource, CerStats, normalize_for_cer};
use clap::{Args, ValueEnum};
use serde::Serialize;

use crate::perf::timestamp_file_name;
use crate::quality::detect_repeated_pattern;
use crate::table::render_table;

#[derive(Debug, Args)]
pub struct EvalArgs {
    /// vASR dataset directory (`<name>.db` + `assets/`) or a single `.vasr` AudioDb.
    #[arg(short, long, alias = "dataset", value_name = "PATH")]
    input: PathBuf,

    /// Transcribe service endpoint.
    #[arg(long, default_value = "http://127.0.0.1:8867/transcribe")]
    server: String,

    /// Directory used when `--output` is not provided.
    #[arg(long, default_value = "results", value_name = "DIR")]
    results_dir: PathBuf,

    /// Optional JSON result output path.
    #[arg(short, long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Process at most this many records.
    #[arg(long)]
    limit: Option<usize>,

    /// Audio timeline/channel to evaluate.
    #[arg(long, value_enum, default_value_t = EvalTimeline::Mono)]
    timeline: EvalTimeline,

    /// HTTP request retries.
    #[arg(long, default_value_t = 5)]
    retries: usize,

    /// HTTP timeout in seconds.
    #[arg(long, default_value_t = 3600)]
    timeout_sec: u64,

    /// Text normalization used before CER.
    #[arg(long, value_enum, default_value_t = EvalNormalizer::Bench)]
    normalizer: EvalNormalizer,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum EvalNormalizer {
    /// vASR bench normalizer: width/case/number/erhua plus punctuation and space removal.
    Bench,
    /// Existing simple vASR CER normalizer: punctuation and space removal only.
    Simple,
}

#[derive(Debug, Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum EvalTimeline {
    Mono,
    Left,
    Right,
}

impl EvalTimeline {
    fn name(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    fn channel(self) -> AudioChannel {
        match self {
            Self::Mono => AudioChannel::Mono,
            Self::Left => AudioChannel::Left,
            Self::Right => AudioChannel::Right,
        }
    }

    fn request_mono(self) -> bool {
        matches!(self, Self::Mono)
    }
}

#[derive(Debug, Clone, Serialize)]
struct ItemResult {
    index: usize,
    audio_id: String,
    url: String,
    reference: String,
    hypothesis: String,
    reference_normalized: String,
    hypothesis_normalized: String,
    cer: f64,
    correct: usize,
    insertions: usize,
    deletions: usize,
    substitutions: usize,
    reference_length: usize,
    accuracy: f64,
    error_rate: f64,
    duration_s: Option<f64>,
    is_bad: bool,
    service_is_bad: bool,
    quality_anomaly: Option<String>,
    bad_reason: Option<String>,
    bad_component: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    wall_seconds: f64,
    audio_seconds: f64,
    speedup_audio_seconds_per_second: f64,
    rtf: f64,
    cer_weighted: f64,
    cer_average: f64,
    correct: usize,
    insertions: usize,
    deletions: usize,
    substitutions: usize,
    reference_length: usize,
    accuracy: f64,
    error_rate: f64,
    accuracy_percent: f64,
    error_rate_percent: f64,
    quality_passed_items: usize,
    quality_passed_audio_seconds: f64,
    quality_passed_speed: f64,
    service_performance: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct EvalReport<'a> {
    timestamp: String,
    dataset: &'a str,
    service_url: &'a str,
    timeline: EvalTimeline,
    summary: Summary,
    items: Vec<ItemResult>,
    response: TranscribeResponse,
}

pub async fn run_eval(args: EvalArgs) -> Result<()> {
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }
    if args.retries == 0 {
        bail!("--retries must be greater than 0");
    }

    let timestamp = timestamp_file_name();
    let mut records = read_vasr_audios(&args.input)
        .with_context(|| format!("failed to open AudioDb {}", args.input.display()))?;
    if let Some(limit) = args.limit {
        records.truncate(limit);
    }
    if records.is_empty() {
        bail!("dataset is empty: {}", args.input.display());
    }

    println!(
        "eval START | input={} items={} server={} normalizer={:?} timeline={}",
        args.input.display(),
        records.len(),
        args.server,
        args.normalizer,
        args.timeline.name()
    );

    validate_timeline(&records, args.timeline)?;
    let request = build_request(&records, args.timeline.request_mono())?;
    println!(
        "eval request START | items={} retries={} timeout={}s",
        request.inputs.len(),
        args.retries,
        args.timeout_sec
    );
    let response = request_batch(&args, request).await?;
    let items = extract_items(&records, &response, args.normalizer, args.timeline)?;
    let summary = summarize(&items, &response)?;
    let bad_count = items.iter().filter(|item| item.is_bad).count();
    println!(
        "eval request DONE  | items={} bad={} elapsed={:.3}s audio={:.1}s speed={:.1}x cer={:.4}",
        records.len(),
        bad_count,
        summary.wall_seconds,
        summary.audio_seconds,
        summary.speedup_audio_seconds_per_second,
        summary.cer_weighted,
    );

    print_eval_summary_table(&summary, records.len());

    let bad_items: Vec<_> = items.iter().filter(|item| item.is_bad).collect();
    if !bad_items.is_empty() {
        println!("bad:      {}", bad_items.len());
        for item in bad_items.iter().take(5) {
            println!(
                "  {:03} {} {}",
                item.index,
                item.audio_id,
                item.bad_reason.as_deref().unwrap_or("")
            );
        }
    }
    let quality_anomalies: Vec<_> = items
        .iter()
        .filter(|item| item.quality_anomaly.is_some())
        .collect();
    if !quality_anomalies.is_empty() {
        println!("quality anomalies: {}", quality_anomalies.len());
        for item in quality_anomalies.iter().take(5) {
            println!(
                "  {:03} {} {}",
                item.index,
                item.audio_id,
                item.quality_anomaly.as_deref().unwrap_or("")
            );
        }
    }

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.results_dir.join(format!("eval_{timestamp}.json")));
    println!(
        "eval DONE | items={} bad={} elapsed={:.3}s speed={:.1}x output={}",
        records.len(),
        bad_count,
        summary.wall_seconds,
        summary.speedup_audio_seconds_per_second,
        output.display()
    );
    save_report(
        &output,
        &EvalReport {
            timestamp,
            dataset: &args.input.to_string_lossy(),
            service_url: &args.server,
            timeline: args.timeline,
            summary,
            items,
            response,
        },
    )?;
    println!("[saved] {}", output.display());

    Ok(())
}

fn print_eval_summary_table(summary: &Summary, items: usize) {
    let table = render_table(
        ["metric", "value"],
        [
            vec!["items".to_string(), items.to_string()],
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
                format!("{:.1}x", summary.speedup_audio_seconds_per_second),
            ],
            vec!["rtf".to_string(), format!("{:.6}", summary.rtf)],
            vec![
                "quality passed".to_string(),
                summary.quality_passed_items.to_string(),
            ],
            vec![
                "quality audio".to_string(),
                format!("{:.1}s", summary.quality_passed_audio_seconds),
            ],
            vec![
                "quality speed".to_string(),
                format!("{:.1}x", summary.quality_passed_speed),
            ],
            vec![
                "cer weighted".to_string(),
                format!("{:.6}", summary.cer_weighted),
            ],
            vec![
                "cer average".to_string(),
                format!("{:.6}", summary.cer_average),
            ],
            vec![
                "accuracy".to_string(),
                format!("{:.2}%", summary.accuracy_percent),
            ],
            vec!["correct".to_string(), summary.correct.to_string()],
            vec!["insertions".to_string(), summary.insertions.to_string()],
            vec!["deletions".to_string(), summary.deletions.to_string()],
            vec![
                "substitutions".to_string(),
                summary.substitutions.to_string(),
            ],
        ],
    );
    println!("\n{table}\n");
}

fn record_timeline(record: &Audio, timeline: EvalTimeline) -> Result<&asr_data::Timeline> {
    record
        .timeline(timeline.channel())
        .map_err(|err| anyhow!("{err}"))?
        .ok_or_else(|| {
            anyhow!(
                "audio {} is missing timeline '{}'",
                record.audio_id(),
                timeline.name()
            )
        })
}

fn validate_timeline(records: &[Audio], timeline: EvalTimeline) -> Result<()> {
    let mut missing = Vec::new();
    for record in records {
        let empty = match record_timeline(record, timeline) {
            Ok(found) => found.reference_transcript().text.is_empty(),
            Err(_) => true,
        };
        if empty {
            missing.push(record.audio_id());
        }
    }
    if !missing.is_empty() {
        let examples = missing
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        bail!(
            "timeline '{}' is missing from {} audio(s); examples: {}",
            timeline.name(),
            missing.len(),
            examples
        );
    }
    Ok(())
}

fn build_request(records: &[Audio], mono: bool) -> Result<TranscribeRequest> {
    let mut inputs = Vec::with_capacity(records.len());
    for record in records {
        let mut input =
            audio_input_from_source(&record.source, Some(mono), hotword_from_record(record));
        input.id = Some(record.id.clone());
        input.info = Some(record.info.clone());
        if input.url.is_none() && input.b64_str.is_none() && input.modelscope.is_none() {
            bail!("audio {} has no usable source", record.audio_id());
        }
        inputs.push(input);
    }
    Ok(TranscribeRequest { inputs })
}

async fn request_batch(args: &EvalArgs, request: TranscribeRequest) -> Result<TranscribeResponse> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()
        .context("failed to build HTTP client")?;

    let mut last_error: Option<anyhow::Error> = None;
    for attempt in 0..args.retries {
        let started = Instant::now();
        let response = client.post(&args.server).json(&request).send().await;
        match response {
            Ok(response) => {
                let status = response.status();
                if status == reqwest::StatusCode::SERVICE_UNAVAILABLE && attempt + 1 < args.retries
                {
                    sleep_before_retry(attempt).await;
                    continue;
                }
                if !status.is_success() {
                    let body = response.text().await.unwrap_or_default();
                    last_error = Some(anyhow!(
                        "request failed: status={} body={}",
                        status,
                        truncate_for_error(&body)
                    ));
                } else {
                    let body = response
                        .text()
                        .await
                        .context("failed to read transcribe JSONL response")?;
                    let parsed = crate::protocol::parse_transcribe_jsonl(
                        &body,
                        started.elapsed().as_secs_f64(),
                    )
                    .context("failed to decode transcribe response JSONL")?;
                    return Ok(parsed);
                }
            }
            Err(error) => {
                last_error = Some(anyhow!("request failed: {error}"));
            }
        }

        if attempt + 1 < args.retries {
            sleep_before_retry(attempt).await;
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("request failed after retries")))
}

/// Evaluate already-completed concurrent requests with the same normalizer as `eval`.
/// This performs no additional inference and includes every dataset item.
pub(crate) fn evaluate_perf_output(
    records: &[Audio],
    data: Vec<crate::InferenceData>,
    wall_seconds: f64,
    audio_seconds: f64,
) -> Result<serde_json::Value> {
    if data.len() != records.len() {
        bail!("CER requires one response per dataset item");
    }
    let response = TranscribeResponse {
        data,
        inference_performance: crate::InferencePerformance {
            batch_wall_seconds: wall_seconds,
            num_items: records.len(),
            throughput_items_per_second: records.len() as f64 / wall_seconds,
            total_audio_duration_seconds: audio_seconds,
            speedup: audio_seconds / wall_seconds,
            rtf: wall_seconds / audio_seconds,
        },
    };
    validate_timeline(records, EvalTimeline::Mono)?;
    let items = extract_items(
        records,
        &response,
        EvalNormalizer::Bench,
        EvalTimeline::Mono,
    )?;
    let summary = summarize(&items, &response)?;
    Ok(
        serde_json::json!({"normalizer": "bench", "timeline": "mono", "summary": summary, "items": items}),
    )
}

fn extract_items(
    records: &[Audio],
    response: &TranscribeResponse,
    normalizer: EvalNormalizer,
    timeline: EvalTimeline,
) -> Result<Vec<ItemResult>> {
    records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let response_item = response.data.get(index);
            let reference = reference_text(record, timeline)?;
            let hypothesis = response_item
                .map(|item| hypothesis_text(item, timeline))
                .unwrap_or_default();
            let reference_normalized = normalize_text(&reference, normalizer);
            let hypothesis_normalized = normalize_text(&hypothesis, normalizer);
            let counts = align_counts(&reference_normalized, &hypothesis_normalized);
            let reference_length = counts.reference_chars;
            let cer = counts.cer();
            let quality_anomaly = detect_item_quality_anomaly(
                &reference_normalized,
                &hypothesis_normalized,
                &hypothesis,
                cer,
            );
            let accuracy = if reference_length == 0 {
                0.0
            } else {
                counts.correct() as f64 / reference_length as f64
            };

            let service_is_bad = response_item.map(|item| item.is_bad).unwrap_or(true);
            let response_bad_reason = response_item.and_then(|item| item.bad_reason.clone());
            let response_bad_component = response_item.and_then(|item| item.bad_component.clone());
            let is_bad = service_is_bad || quality_anomaly.is_some();
            let bad_reason = response_bad_reason.or_else(|| quality_anomaly.clone());
            let bad_component = response_bad_component
                .or_else(|| quality_anomaly.as_ref().map(|_| "quality".to_string()));

            Ok(ItemResult {
                index,
                audio_id: record.audio_id(),
                url: url_from_record(record),
                reference,
                hypothesis,
                reference_normalized,
                hypothesis_normalized,
                cer,
                correct: counts.correct(),
                insertions: counts.insertions,
                deletions: counts.deletions,
                substitutions: counts.substitutions,
                reference_length,
                accuracy: round6(accuracy),
                error_rate: round6(cer),
                duration_s: response_item
                    .map(|item| item.timings.audio)
                    .filter(|duration| *duration > 0.0),
                is_bad,
                service_is_bad,
                quality_anomaly,
                bad_reason,
                bad_component,
            })
        })
        .collect()
}

fn summarize(items: &[ItemResult], response: &TranscribeResponse) -> Result<Summary> {
    let valid: Vec<_> = items
        .iter()
        .filter(|item| !item.reference_normalized.is_empty())
        .collect();
    let reference_length: usize = valid.iter().map(|item| item.reference_length).sum();
    let weighted_errors: f64 = valid
        .iter()
        .map(|item| item.cer * item.reference_length as f64)
        .sum();
    let cer_weighted = if reference_length == 0 {
        0.0
    } else {
        weighted_errors / reference_length as f64
    };
    let cer_average = if valid.is_empty() {
        0.0
    } else {
        valid.iter().map(|item| item.cer).sum::<f64>() / valid.len() as f64
    };
    let correct: usize = valid.iter().map(|item| item.correct).sum();
    let insertions: usize = valid.iter().map(|item| item.insertions).sum();
    let deletions: usize = valid.iter().map(|item| item.deletions).sum();
    let substitutions: usize = valid.iter().map(|item| item.substitutions).sum();
    let accuracy = if reference_length == 0 {
        0.0
    } else {
        correct as f64 / reference_length as f64
    };
    let error_rate = if reference_length == 0 {
        0.0
    } else {
        (insertions + deletions + substitutions) as f64 / reference_length as f64
    };
    let performance = &response.inference_performance;
    let quality_passed: Vec<_> = items
        .iter()
        .filter(|item| {
            !item.is_bad
                && !item.reference_normalized.is_empty()
                && !item.hypothesis_normalized.is_empty()
        })
        .collect();
    let quality_passed_audio_seconds = if quality_passed.is_empty() {
        0.0
    } else {
        quality_passed
            .iter()
            .filter_map(|item| item.duration_s)
            .sum()
    };
    let service_performance =
        serde_json::to_value(&response.inference_performance).context("encode service perf")?;

    Ok(Summary {
        wall_seconds: round4(performance.batch_wall_seconds),
        audio_seconds: round4(performance.total_audio_duration_seconds),
        speedup_audio_seconds_per_second: round4(performance.speedup),
        rtf: round6(performance.rtf),
        cer_weighted: round6(cer_weighted),
        cer_average: round6(cer_average),
        correct,
        insertions,
        deletions,
        substitutions,
        reference_length,
        accuracy: round6(accuracy),
        error_rate: round6(error_rate),
        accuracy_percent: round2(accuracy * 100.0),
        error_rate_percent: round2(error_rate * 100.0),
        quality_passed_items: quality_passed.len(),
        quality_passed_audio_seconds: round4(quality_passed_audio_seconds),
        quality_passed_speed: round4(if performance.batch_wall_seconds > 0.0 {
            quality_passed_audio_seconds / performance.batch_wall_seconds
        } else {
            0.0
        }),
        service_performance,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct AlignCounts {
    substitutions: usize,
    deletions: usize,
    insertions: usize,
    reference_chars: usize,
}

impl AlignCounts {
    fn correct(self) -> usize {
        self.reference_chars
            .saturating_sub(self.deletions + self.substitutions)
    }

    fn cer(self) -> f64 {
        CerStats {
            substitutions: self.substitutions,
            deletions: self.deletions,
            insertions: self.insertions,
            reference_chars: self.reference_chars,
        }
        .cer()
    }
}

fn align_counts(reference: &str, hypothesis: &str) -> AlignCounts {
    let reference: Vec<char> = reference.chars().collect();
    let hypothesis: Vec<char> = hypothesis.chars().collect();
    let n = reference.len();
    let m = hypothesis.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    let mut back = vec![vec![Op::None; m + 1]; n + 1];

    for i in 1..=n {
        dp[i][0] = i;
        back[i][0] = Op::Del;
    }
    for j in 1..=m {
        dp[0][j] = j;
        back[0][j] = Op::Ins;
    }

    for i in 1..=n {
        for j in 1..=m {
            if reference[i - 1] == hypothesis[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
                back[i][j] = Op::Ok;
                continue;
            }
            let candidates = [
                (dp[i - 1][j - 1] + 1, Op::Sub),
                (dp[i][j - 1] + 1, Op::Ins),
                (dp[i - 1][j] + 1, Op::Del),
            ];
            // 三个候选固定非空，`unwrap_or` 只用于消除 `Option`，兜底分支不会命中。
            let (cost, op) = candidates
                .into_iter()
                .min_by_key(|(cost, op)| (*cost, op.rank()))
                .unwrap_or(candidates[0]);
            dp[i][j] = cost;
            back[i][j] = op;
        }
    }

    let mut i = n;
    let mut j = m;
    let mut counts = AlignCounts {
        reference_chars: n,
        ..AlignCounts::default()
    };
    while i > 0 || j > 0 {
        match back[i][j] {
            Op::Ok => {
                i -= 1;
                j -= 1;
            }
            Op::Sub => {
                counts.substitutions += 1;
                i -= 1;
                j -= 1;
            }
            Op::Ins => {
                counts.insertions += 1;
                j -= 1;
            }
            Op::Del => {
                counts.deletions += 1;
                i -= 1;
            }
            Op::None => break,
        }
    }

    counts
}

#[derive(Debug, Clone, Copy)]
enum Op {
    None,
    Ok,
    Sub,
    Ins,
    Del,
}

impl Op {
    fn rank(self) -> usize {
        match self {
            Self::Sub => 0,
            Self::Ins => 1,
            Self::Del => 2,
            Self::Ok => 3,
            Self::None => 4,
        }
    }
}

fn reference_text(record: &Audio, timeline: EvalTimeline) -> Result<String> {
    Ok(record_timeline(record, timeline)?
        .reference_transcript()
        .text)
}

fn hypothesis_text(item: &crate::InferenceData, timeline: EvalTimeline) -> String {
    match timeline {
        EvalTimeline::Mono => item.text.mono.clone(),
        EvalTimeline::Left => item.text.left.clone(),
        EvalTimeline::Right => item.text.right.clone(),
    }
}

fn normalize_text(text: &str, normalizer: EvalNormalizer) -> String {
    match normalizer {
        EvalNormalizer::Simple => normalize_for_cer(text, true),
        EvalNormalizer::Bench => normalize_bench_text(text),
    }
}

fn detect_item_quality_anomaly(
    reference: &str,
    hypothesis: &str,
    raw_hypothesis: &str,
    cer: f64,
) -> Option<String> {
    if !reference.is_empty() && hypothesis.is_empty() {
        Some("empty ASR output for non-empty reference".to_string())
    } else if let Some(reason) = detect_repeated_pattern(raw_hypothesis) {
        Some(reason)
    } else if !reference.is_empty() && cer >= 1.0 {
        Some(format!("CER exceeds 100% ({:.1}%)", cer * 100.0))
    } else {
        None
    }
}

fn normalize_bench_text(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let folded = fold_width_and_case(text);
    let chars: Vec<char> = folded.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if is_cer_ignored_char(chars[i]) {
            i += 1;
            continue;
        }
        if let Some((digits, consumed)) = parse_chinese_number(&chars[i..]) {
            normalized.push_str(&digits);
            i += consumed;
            continue;
        }
        if chars[i] == '儿' && should_drop_erhua(&chars, i) {
            i += 1;
            continue;
        }
        normalized.push(chars[i]);
        i += 1;
    }
    normalized
}

fn fold_width_and_case(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '\u{3000}' => ' ',
            '\u{ff01}'..='\u{ff5e}' => char::from_u32(ch as u32 - 0xfee0).unwrap_or(ch),
            _ => ch,
        })
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_cer_ignored_char(ch: char) -> bool {
    ch.is_whitespace() || normalize_for_cer(&ch.to_string(), false).is_empty()
}

fn should_drop_erhua(chars: &[char], index: usize) -> bool {
    if index == 0 {
        return false;
    }
    let prev = chars[index - 1];
    let next = chars.get(index + 1).copied();
    is_cjk(prev) && next.is_none_or(|ch| !is_cjk(ch) || is_chinese_sentence_particle(ch))
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32, 0x4e00..=0x9fff | 0x3400..=0x4dbf)
}

fn is_chinese_sentence_particle(ch: char) -> bool {
    matches!(
        ch,
        '啊' | '呀' | '呢' | '吧' | '吗' | '嘛' | '啦' | '了' | '的'
    )
}

fn parse_chinese_number(chars: &[char]) -> Option<(String, usize)> {
    let mut end = 0;
    while end < chars.len() && is_chinese_number_char(chars[end]) {
        end += 1;
    }
    if end == 0 {
        return None;
    }
    let slice = &chars[..end];
    if slice.len() == 1 {
        if let Some(digit) = chinese_digit(slice[0]) {
            return Some((digit.to_string(), 1));
        }
        if let Some(unit) = chinese_small_unit(slice[0]) {
            return Some((unit.to_string(), 1));
        }
    }
    if slice.iter().all(|ch| chinese_digit(*ch).is_some()) {
        let digits = slice
            .iter()
            .filter_map(|ch| chinese_digit(*ch))
            .map(|digit| char::from(b'0' + digit as u8))
            .collect();
        return Some((digits, end));
    }
    chinese_number_value(slice).map(|value| (value.to_string(), end))
}

fn is_chinese_number_char(ch: char) -> bool {
    chinese_digit(ch).is_some() || matches!(ch, '十' | '百' | '千' | '万' | '亿')
}

fn chinese_digit(ch: char) -> Option<u64> {
    match ch {
        '零' | '〇' => Some(0),
        '一' | '壹' => Some(1),
        '二' | '两' | '贰' => Some(2),
        '三' | '叁' => Some(3),
        '四' | '肆' => Some(4),
        '五' | '伍' => Some(5),
        '六' | '陆' => Some(6),
        '七' | '柒' => Some(7),
        '八' | '捌' => Some(8),
        '九' | '玖' => Some(9),
        _ => None,
    }
}

fn chinese_small_unit(ch: char) -> Option<u64> {
    match ch {
        '十' => Some(10),
        '百' => Some(100),
        '千' => Some(1000),
        _ => None,
    }
}

fn chinese_large_unit(ch: char) -> Option<u64> {
    match ch {
        '万' => Some(10_000),
        '亿' => Some(100_000_000),
        _ => None,
    }
}

fn chinese_number_value(chars: &[char]) -> Option<u64> {
    let mut total = 0u64;
    let mut section = 0u64;
    let mut number = 0u64;
    let mut has_unit = false;

    for &ch in chars {
        if let Some(digit) = chinese_digit(ch) {
            number = digit;
        } else if let Some(unit) = chinese_small_unit(ch) {
            has_unit = true;
            section += number.max(1) * unit;
            number = 0;
        } else {
            let unit = chinese_large_unit(ch)?;
            has_unit = true;
            section += number;
            total += section.max(1) * unit;
            section = 0;
            number = 0;
        }
    }

    if !has_unit {
        return None;
    }
    Some(total + section + number)
}

fn url_from_record(record: &Audio) -> String {
    match &record.source {
        AudioSource::Path(path) => path.display().to_string(),
        AudioSource::Url(url) => url.clone(),
        AudioSource::ModelScope {
            repo_id,
            file_path,
            revision,
        } => format!("modelscope://{repo_id}/{file_path}@{revision}"),
        AudioSource::Base64(_) | AudioSource::EncodedBytes(_) | AudioSource::PcmS16Le { .. } => {
            String::new()
        }
    }
}

fn hotword_from_record(record: &Audio) -> Option<String> {
    for key in ["hotword", "hotwords", "fasr_hotword", "fasr_hotwords"] {
        let Some(value) = record.metadata.get(key) else {
            continue;
        };
        if let Some(text) = value.as_str()
            && !text.trim().is_empty()
        {
            return Some(text.to_string());
        }
        if let Some(values) = value.as_array() {
            let joined = values
                .iter()
                .filter_map(|value| value.as_str())
                .filter(|value| !value.trim().is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !joined.is_empty() {
                return Some(joined);
            }
        }
    }
    None
}

async fn sleep_before_retry(attempt: usize) {
    tokio::time::sleep(Duration::from_secs(2 + attempt as u64 * 2)).await;
}

fn save_report(path: &Path, report: &EvalReport<'_>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    let json = serde_json::to_vec_pretty(report).context("failed to encode report JSON")?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn truncate_for_error(body: &str) -> String {
    const MAX: usize = 500;
    if body.len() <= MAX {
        body.to_string()
    } else {
        format!("{}...", &body[..MAX])
    }
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
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
        EvalNormalizer, EvalTimeline, detect_item_quality_anomaly, hypothesis_text, normalize_text,
        reference_text,
    };
    use crate::InferenceData;
    use asr_data::{
        Audio, AudioChannel, AudioEncoding, AudioEvent, AudioFormat, AudioInfo, AudioSource,
        Speech, Transcription,
    };

    fn test_audio_info(channels: u16) -> AudioInfo {
        AudioInfo {
            sample_rate: 16_000,
            channels,
            frame_count: 16_000,
            source_format: AudioFormat {
                encoding: AudioEncoding::Unknown,
                sample_rate: 16_000,
                channels,
            },
        }
    }

    fn add_reference(audio: &mut Audio, channel: AudioChannel, text: &str) {
        let duration = audio.timeline_duration().unwrap_or(1_000);
        audio
            .ensure_timeline(channel, Some(duration))
            .expect("timeline")
            .annotate_with(
                0,
                duration,
                AudioEvent::Speech(Speech::new().with_transcription(Transcription::new(text))),
                true,
                None,
            )
            .expect("reference");
    }

    #[test]
    fn concurrent_perf_quality_reuses_bench_and_rejects_missing_items() {
        let mut audio = Audio::with_id_from_info(
            "call-1",
            AudioSource::from_url("https://example.com/call.wav"),
            &test_audio_info(1),
        );
        add_reference(&mut audio, AudioChannel::Mono, "我要修改商品帖子的价格");
        let mut output =
            InferenceData::from_timeline("service", &asr_data::Timeline::new("call-1", 1_000));
        output.text.mono = "我要修改商品帖子的价格。".into();
        output.timings.audio = 1.0;
        let quality =
            super::evaluate_perf_output(&[audio.clone()], vec![output], 1.0, 1.0).unwrap();
        assert_eq!(quality["summary"]["cer_weighted"], 0.0);
        assert_eq!(
            quality["items"][0]["reference_normalized"],
            "我要修改商品帖子的价格"
        );
        assert!(super::evaluate_perf_output(&[audio], vec![], 1.0, 1.0).is_err());
    }

    #[test]
    fn timeline_selects_matching_reference_and_hypothesis() {
        let mut audio = Audio::with_id_from_info(
            "call-1",
            AudioSource::from_url("https://example.com/call.wav"),
            &test_audio_info(2),
        );
        add_reference(&mut audio, AudioChannel::Left, "left reference");
        add_reference(&mut audio, AudioChannel::Right, "right reference");

        let dummy = asr_data::Timeline::new("call-1", 1_000);
        let mut inference = InferenceData::from_timeline("service", &dummy);
        inference.text.mono = "mono hypothesis".to_string();
        inference.text.left = "left hypothesis".to_string();
        inference.text.right = "right hypothesis".to_string();

        assert_eq!(
            reference_text(&audio, EvalTimeline::Left).expect("left reference"),
            "left reference"
        );
        assert_eq!(
            reference_text(&audio, EvalTimeline::Right).expect("right reference"),
            "right reference"
        );
        assert_eq!(
            hypothesis_text(&inference, EvalTimeline::Mono),
            "mono hypothesis"
        );
        assert_eq!(
            hypothesis_text(&inference, EvalTimeline::Left),
            "left hypothesis"
        );
        assert_eq!(
            hypothesis_text(&inference, EvalTimeline::Right),
            "right hypothesis"
        );
    }

    #[test]
    fn bench_normalizer_folds_width_case_punctuation_and_spaces() {
        assert_eq!(
            normalize_text("ＡＢＣ， The number １ ２ ３！", EvalNormalizer::Bench),
            "abcthenumber123"
        );
    }

    #[test]
    fn bench_normalizer_removes_common_erhua_suffix() {
        assert_eq!(normalize_text("一会儿。", EvalNormalizer::Bench), "1会");
        assert_eq!(normalize_text("这儿啊", EvalNormalizer::Bench), "这啊");
    }

    #[test]
    fn bench_normalizer_converts_chinese_numbers() {
        assert_eq!(
            normalize_text("一百二十三个，二零二六年，十次", EvalNormalizer::Bench),
            "123个2026年10次"
        );
    }

    #[test]
    fn simple_normalizer_keeps_existing_vasr_behavior() {
        assert_eq!(
            normalize_text("ＡＢＣ，一百二十三", EvalNormalizer::Simple),
            "ＡＢＣ一百二十三"
        );
    }

    #[test]
    fn quality_gate_rejects_empty_repeated_and_extreme_cer_outputs() {
        assert_eq!(
            detect_item_quality_anomaly("reference", "", "", 1.0).as_deref(),
            Some("empty ASR output for non-empty reference")
        );
        assert!(
            detect_item_quality_anomaly(
                "reference",
                "iburibiribiribiribiribiribiribiribiribiribir",
                "iburibiribiribiribiribiribiribiribiribiribir",
                0.8,
            )
            .is_some()
        );
        assert_eq!(
            detect_item_quality_anomaly("reference", "unrelated", "unrelated", 1.2).as_deref(),
            Some("CER exceeds 100% (120.0%)")
        );
        assert!(detect_item_quality_anomaly("reference", "unrelated", "unrelated", 1.0).is_some());
        assert!(detect_item_quality_anomaly("reference", "near", "near", 0.5).is_none());
    }
}
