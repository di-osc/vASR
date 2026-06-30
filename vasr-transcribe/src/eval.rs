use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{AudioInputDto, TranscribeRequest, TranscribeResponse};
use anyhow::{Context, Result, anyhow, bail};
use base64::Engine;
use clap::{ArgAction, Args, ValueEnum};
use comfy_table::{ContentArrangement, Table, modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL};
use serde::Serialize;
use vasr_data::{AudioAsset, CerStats, VasrRecord, VasrRecordList, normalize_for_cer};

#[derive(Debug, Args)]
pub struct EvalArgs {
    /// vASR `VasrRecordList` MessagePack dataset.
    #[arg(short, long, alias = "dataset", value_name = "PATH")]
    input: PathBuf,

    /// Transcribe service endpoint.
    #[arg(long, default_value = "http://127.0.0.1:8867/inference")]
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

    /// Override mono mode for every request item. Omit to let the server decide.
    #[arg(long, action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true")]
    mono: Option<bool>,

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
    service_performance: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct EvalReport<'a> {
    timestamp: String,
    dataset: &'a str,
    service_url: &'a str,
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

    let timestamp = unix_timestamp_string();
    let mut records = VasrRecordList::read_msgpack(&args.input)
        .with_context(|| format!("failed to read dataset {}", args.input.display()))?
        .records;
    if let Some(limit) = args.limit {
        records.truncate(limit);
    }
    if records.is_empty() {
        bail!("dataset is empty: {}", args.input.display());
    }

    println!(
        "eval START | input={} items={} server={} normalizer={:?} mono={}",
        args.input.display(),
        records.len(),
        args.server,
        args.normalizer,
        format_mono(args.mono)
    );

    let request = build_request(&records, args.mono)?;
    println!(
        "eval request START | items={} retries={} timeout={}s",
        request.inputs.len(),
        args.retries,
        args.timeout_sec
    );
    let (response, wall_seconds) = request_batch(&args, request).await?;
    let items = extract_items(&records, &response, args.normalizer);
    let summary = summarize(&items, wall_seconds, &response)?;
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
            summary,
            items,
            response,
        },
    )?;
    println!("[saved] {}", output.display());

    Ok(())
}

fn print_eval_summary_table(summary: &Summary, items: usize) {
    let mut table = summary_table();
    table.add_rows([
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

fn format_mono(mono: Option<bool>) -> &'static str {
    match mono {
        Some(true) => "true",
        Some(false) => "false",
        None => "auto",
    }
}

fn build_request(records: &[VasrRecord], mono: Option<bool>) -> Result<TranscribeRequest> {
    let mut inputs = Vec::with_capacity(records.len());
    for record in records {
        let input = match &record.media {
            AudioAsset::Uri { uri, .. } => AudioInputDto {
                url: Some(uri.clone()),
                b64_str: None,
                mono,
                hotword: hotword_from_record(record),
            },
            AudioAsset::Embedded { bytes, .. } => AudioInputDto {
                url: None,
                b64_str: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
                mono,
                hotword: hotword_from_record(record),
            },
        };
        if input.url.is_none() && input.b64_str.is_none() {
            bail!("record {} has no usable audio source", record.record_id());
        }
        inputs.push(input);
    }
    Ok(TranscribeRequest { inputs })
}

async fn request_batch(
    args: &EvalArgs,
    request: TranscribeRequest,
) -> Result<(TranscribeResponse, f64)> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(args.timeout_sec))
        .build()
        .context("failed to build HTTP client")?;

    let mut last_error: Option<anyhow::Error> = None;
    for attempt in 0..args.retries {
        let start = Instant::now();
        let response = client.post(&args.server).json(&request).send().await;
        let elapsed = start.elapsed().as_secs_f64();
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
                    let parsed = response
                        .json::<TranscribeResponse>()
                        .await
                        .context("failed to decode transcribe response JSON")?;
                    return Ok((parsed, elapsed));
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

fn extract_items(
    records: &[VasrRecord],
    response: &TranscribeResponse,
    normalizer: EvalNormalizer,
) -> Vec<ItemResult> {
    records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let response_item = response.data.get(index);
            let reference = reference_text(record);
            let hypothesis = response_item.map(hypothesis_text).unwrap_or_default();
            let reference_normalized = normalize_text(&reference, normalizer);
            let hypothesis_normalized = normalize_text(&hypothesis, normalizer);
            let counts = align_counts(&reference_normalized, &hypothesis_normalized);
            let reference_length = counts.reference_chars;
            let cer = counts.cer();
            let accuracy = if reference_length == 0 {
                0.0
            } else {
                counts.correct() as f64 / reference_length as f64
            };

            ItemResult {
                index,
                audio_id: record.record_id(),
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
                duration_s: record
                    .media
                    .duration()
                    .map(|duration| duration.0 as f64 / 1000.0),
                is_bad: response_item.map(|item| item.is_bad).unwrap_or(true),
                bad_reason: response_item.and_then(|item| item.bad_reason.clone()),
                bad_component: response_item.and_then(|item| item.bad_component.clone()),
            }
        })
        .collect()
}

fn summarize(
    items: &[ItemResult],
    wall_seconds: f64,
    response: &TranscribeResponse,
) -> Result<Summary> {
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
    let audio_seconds: f64 = items.iter().filter_map(|item| item.duration_s).sum();
    let speedup = if wall_seconds > 0.0 {
        audio_seconds / wall_seconds
    } else {
        0.0
    };
    let rtf = if audio_seconds > 0.0 {
        wall_seconds / audio_seconds
    } else {
        0.0
    };
    let service_performance =
        serde_json::to_value(&response.inference_performance).context("encode service perf")?;

    Ok(Summary {
        wall_seconds: round4(wall_seconds),
        audio_seconds: round4(audio_seconds),
        speedup_audio_seconds_per_second: round4(speedup),
        rtf: round6(rtf),
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
            let (cost, op) = candidates
                .into_iter()
                .min_by_key(|(cost, op)| (*cost, op.rank()))
                .expect("non-empty candidates");
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

fn reference_text(record: &VasrRecord) -> String {
    record.timeline.transcript().text
}

fn hypothesis_text(item: &crate::InferenceData) -> String {
    if !item.text.mono.is_empty() {
        return item.text.mono.clone();
    }
    if !item.text.left.is_empty() {
        return item.text.left.clone();
    }
    item.text.right.clone()
}

fn normalize_text(text: &str, normalizer: EvalNormalizer) -> String {
    match normalizer {
        EvalNormalizer::Simple => normalize_for_cer(text, true),
        EvalNormalizer::Bench => normalize_bench_text(text),
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
        } else if let Some(unit) = chinese_large_unit(ch) {
            has_unit = true;
            section += number;
            total += section.max(1) * unit;
            section = 0;
            number = 0;
        } else {
            return None;
        }
    }

    if !has_unit {
        return None;
    }
    Some(total + section + number)
}

fn url_from_record(record: &VasrRecord) -> String {
    match &record.media {
        AudioAsset::Uri { uri, .. } => uri.clone(),
        AudioAsset::Embedded { .. } => String::new(),
    }
}

fn hotword_from_record(record: &VasrRecord) -> Option<String> {
    for key in ["hotword", "hotwords", "fasr_hotword", "fasr_hotwords"] {
        let Some(value) = record.metadata.get(key) else {
            continue;
        };
        if let Some(text) = value.as_str() {
            if !text.trim().is_empty() {
                return Some(text.to_string());
            }
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
    use super::{EvalNormalizer, normalize_text};

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
}
