use anyhow::{Result, anyhow, bail};
use asr_data::{Audio, AudioChannel, AudioSource, compute_cer, normalize_for_cer};
use clap::ValueEnum;
use serde::Serialize;

use crate::protocol::{
    audio_bad_component, audio_bad_reason, audio_channel_text, audio_is_bad, audio_seconds,
};
use crate::quality::detect_repeated_pattern;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum EvalNormalizer {
    /// vASR bench normalizer: width/case/number/erhua plus punctuation and space removal.
    Bench,
    /// Existing simple vASR CER normalizer: punctuation and space removal only.
    Simple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub(crate) enum EvalTimeline {
    Mono,
    Left,
    Right,
}

impl EvalTimeline {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Mono => "mono",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    pub(crate) fn channel(self) -> AudioChannel {
        match self {
            Self::Mono => AudioChannel::Mono,
            Self::Left => AudioChannel::Left,
            Self::Right => AudioChannel::Right,
        }
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
    /// 带参考文本、参与 CER 的记录数。
    labeled_items: usize,
    /// 输入记录总数。
    total_items: usize,
    service_performance: serde_json::Value,
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

/// 记录在该时间轴上是否带有非空参考文本。
pub(crate) fn has_reference(record: &Audio, timeline: EvalTimeline) -> bool {
    record_timeline(record, timeline)
        .map(|timeline| !timeline.reference_transcript().text.trim().is_empty())
        .unwrap_or(false)
}

/// 评估已经完成的并发请求，不触发额外推理。
///
/// 只有带参考文本的记录参与 CER；其余记录仍保留在 `items` 中，但不计入
/// `labeled_items`。没有任何参考文本时返回错误。
pub(crate) fn evaluate_perf_output(
    records: &[Audio],
    results: Vec<Audio>,
    wall_seconds: f64,
    audio_seconds: f64,
    timeline: EvalTimeline,
    normalizer: EvalNormalizer,
) -> Result<serde_json::Value> {
    if results.len() != records.len() {
        bail!("CER requires one response per input item");
    }
    let items = extract_items(records, &results, normalizer, timeline);
    let summary = summarize(&items, wall_seconds, audio_seconds);
    if summary.labeled_items == 0 {
        bail!(
            "no reference transcripts for timeline '{}'",
            timeline.name()
        );
    }
    Ok(serde_json::json!({
        "normalizer": format!("{normalizer:?}").to_ascii_lowercase(),
        "timeline": timeline.name(),
        "summary": summary,
        "items": items,
    }))
}

fn extract_items(
    records: &[Audio],
    results: &[Audio],
    normalizer: EvalNormalizer,
    timeline: EvalTimeline,
) -> Vec<ItemResult> {
    records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let result = results.get(index);
            let reference = reference_text(record, timeline);
            let hypothesis = result
                .map(|audio| audio_channel_text(audio, timeline.channel()))
                .unwrap_or_default();
            let reference_normalized = normalize_text(&reference, normalizer);
            let hypothesis_normalized = normalize_text(&hypothesis, normalizer);
            let stats = compute_cer(&reference_normalized, &hypothesis_normalized);
            let reference_length = stats.reference_chars;
            let cer = stats.cer();
            let correct = reference_length.saturating_sub(stats.deletions + stats.substitutions);
            let quality_anomaly = detect_item_quality_anomaly(
                &reference_normalized,
                &hypothesis_normalized,
                &hypothesis,
                cer,
            );
            let accuracy = if reference_length == 0 {
                0.0
            } else {
                correct as f64 / reference_length as f64
            };

            let service_is_bad = result.map(audio_is_bad).unwrap_or(true);
            let response_bad_reason = result.and_then(audio_bad_reason);
            let response_bad_component = result.and_then(audio_bad_component);
            let is_bad = service_is_bad || quality_anomaly.is_some();
            let bad_reason = response_bad_reason.or_else(|| quality_anomaly.clone());
            let bad_component = response_bad_component
                .or_else(|| quality_anomaly.as_ref().map(|_| "quality".to_string()));

            ItemResult {
                index,
                audio_id: record.audio_id(),
                url: url_from_record(record),
                reference,
                hypothesis,
                reference_normalized,
                hypothesis_normalized,
                cer,
                correct,
                insertions: stats.insertions,
                deletions: stats.deletions,
                substitutions: stats.substitutions,
                reference_length,
                accuracy: round6(accuracy),
                error_rate: round6(cer),
                duration_s: result.map(audio_seconds).filter(|duration| *duration > 0.0),
                is_bad,
                service_is_bad,
                quality_anomaly,
                bad_reason,
                bad_component,
            }
        })
        .collect()
}

fn summarize(items: &[ItemResult], wall_seconds: f64, audio_seconds: f64) -> Summary {
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
    let throughput_items_per_second = items.len() as f64 / wall_seconds.max(f64::EPSILON);
    let speedup = audio_seconds / wall_seconds.max(f64::EPSILON);
    let rtf = wall_seconds / audio_seconds.max(f64::EPSILON);
    let service_performance = serde_json::json!({
        "batch_wall_seconds": round4(wall_seconds),
        "num_items": items.len(),
        "throughput_items_per_second": round4(throughput_items_per_second),
        "total_audio_duration_seconds": round4(audio_seconds),
        "speedup": round4(speedup),
        "rtf": round6(rtf),
    });

    Summary {
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
        quality_passed_items: quality_passed.len(),
        quality_passed_audio_seconds: round4(quality_passed_audio_seconds),
        quality_passed_speed: round4(if wall_seconds > 0.0 {
            quality_passed_audio_seconds / wall_seconds
        } else {
            0.0
        }),
        labeled_items: valid.len(),
        total_items: items.len(),
        service_performance,
    }
}

/// 该时间轴的参考文本；时间轴缺失时返回空串（视为无参考）。
fn reference_text(record: &Audio, timeline: EvalTimeline) -> String {
    record_timeline(record, timeline)
        .map(|timeline| timeline.reference_transcript().text)
        .unwrap_or_default()
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
        EvalNormalizer, EvalTimeline, detect_item_quality_anomaly, normalize_text, reference_text,
    };
    use crate::protocol::audio_channel_text;
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

    fn add_prediction(audio: &mut Audio, channel: AudioChannel, text: &str) {
        let duration = audio.timeline_duration().unwrap_or(1_000);
        audio
            .ensure_timeline(channel, Some(duration))
            .expect("timeline")
            .prediction
            .push(AudioEvent::Speech(
                Speech::new().with_transcription(Transcription::new(text)),
            ));
    }

    #[test]
    fn concurrent_perf_quality_reuses_bench_and_rejects_missing_items() {
        let mut audio = Audio::with_id_from_info(
            "call-1",
            AudioSource::from_url("https://example.com/call.wav"),
            &test_audio_info(1),
        );
        add_reference(&mut audio, AudioChannel::Mono, "我要修改商品帖子的价格");
        add_prediction(&mut audio, AudioChannel::Mono, "我要修改商品帖子的价格。");
        let quality = super::evaluate_perf_output(
            &[audio.clone()],
            vec![audio.clone()],
            1.0,
            1.0,
            EvalTimeline::Mono,
            EvalNormalizer::Bench,
        )
        .unwrap();
        assert_eq!(quality["summary"]["cer_weighted"], 0.0);
        assert_eq!(quality["summary"]["labeled_items"], 1);
        assert_eq!(
            quality["items"][0]["reference_normalized"],
            "我要修改商品帖子的价格"
        );
        assert!(
            super::evaluate_perf_output(
                &[audio],
                vec![],
                1.0,
                1.0,
                EvalTimeline::Mono,
                EvalNormalizer::Bench,
            )
            .is_err()
        );
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
        add_prediction(&mut audio, AudioChannel::Mono, "mono hypothesis");
        add_prediction(&mut audio, AudioChannel::Left, "left hypothesis");
        add_prediction(&mut audio, AudioChannel::Right, "right hypothesis");

        assert_eq!(reference_text(&audio, EvalTimeline::Left), "left reference");
        assert_eq!(
            reference_text(&audio, EvalTimeline::Right),
            "right reference"
        );
        assert!(super::has_reference(&audio, EvalTimeline::Left));
        assert!(!super::has_reference(&audio, EvalTimeline::Mono));
        assert_eq!(
            audio_channel_text(&audio, AudioChannel::Mono),
            "mono hypothesis"
        );
        assert_eq!(
            audio_channel_text(&audio, AudioChannel::Left),
            "left hypothesis"
        );
        assert_eq!(
            audio_channel_text(&audio, AudioChannel::Right),
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
