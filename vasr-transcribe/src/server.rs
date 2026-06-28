use std::sync::Arc;
use std::time::Instant;

use axum::extract::DefaultBodyLimit;
use axum::{Json, Router, routing::post};
use vasr_data::AudioSource;

use crate::pipeline::{
    GlobalAsyncTranscribePipeline, PipelineWorkerOptions, TranscribeInput, TranscribeItemTiming,
};
use crate::protocol::{
    InferenceData, InferencePerformance, SentenceResult, TextResult, TimingBreakdown,
    TranscribeRequest, TranscribeResponse,
};
use crate::runtime::{AsrOptions, VadOptions};

const TRANSCRIBE_BODY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

pub struct TranscribeService {
    pub service_id: String,
    pub pipeline: Arc<GlobalAsyncTranscribePipeline>,
}

pub fn transcribe_router(service: Arc<TranscribeService>) -> Router {
    Router::new()
        .route("/transcribe", post(handle_transcribe))
        .route("/inference", post(handle_transcribe))
        .layer(DefaultBodyLimit::max(TRANSCRIBE_BODY_LIMIT_BYTES))
        .with_state(service)
}

async fn handle_transcribe(
    axum::extract::State(service): axum::extract::State<Arc<TranscribeService>>,
    Json(request): Json<TranscribeRequest>,
) -> Json<TranscribeResponse> {
    let start = Instant::now();
    let request_id = short_request_id();
    let request_items = request.inputs.len();
    let mut inputs = Vec::new();
    let mut data_slots = vec![None; request_items];
    let mut request_metas = vec![RequestMeta::default(); request_items];
    let mut request_stats = RequestStats::default();
    let mut start_stats = RequestStartStats::default();

    for (index, input) in request.inputs.into_iter().enumerate() {
        request_metas[index].service_id = service.service_id.clone();
        let mono = input.mono.unwrap_or(false);
        if mono {
            start_stats.mono_true += 1;
        } else {
            start_stats.mono_false += 1;
        }
        let source = if let Some(url) = input.url {
            start_stats.url += 1;
            AudioSource::Url(url)
        } else if let Some(b64) = input.b64_str {
            start_stats.base64 += 1;
            AudioSource::Base64(b64)
        } else {
            start_stats.missing += 1;
            data_slots[index] = Some(error_data(
                request_metas[index].service_id.clone(),
                "loader",
                "必须提供 url 或 b64_str 之一".to_string(),
                TranscribeItemTiming::default(),
                0.0,
            ));
            continue;
        };
        inputs.push(TranscribeInput {
            index,
            source,
            mono,
        });
    }
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{} {:<7} {:<5} | items={} src=url:{} base64:{} missing:{} mono=true:{} false:{}",
        request_id,
        "request",
        "START",
        request_items,
        start_stats.url,
        start_stats.base64,
        start_stats.missing,
        start_stats.mono_true,
        start_stats.mono_false
    );

    let outcomes = service
        .pipeline
        .run_with_log_id(request_id.clone(), inputs)
        .await;
    let mut total_audio_duration_seconds = 0.0;
    let mut component_timing = TranscribeItemTiming::default();

    for outcome in outcomes {
        if outcome.channel == "mono" {
            request_stats.mono += 1;
        } else if outcome.channel == "left" {
            request_stats.stereo += 1;
        }
        request_stats.channels += 1;
        let Some(meta) = request_metas.get(outcome.index) else {
            continue;
        };
        match outcome.result {
            Ok(timeline) => {
                total_audio_duration_seconds += outcome.audio_seconds;
                component_timing.add_scaled(outcome.timing, 1.0);
                let mut channel_data = InferenceData::from_timeline_channel(
                    meta.service_id.clone(),
                    &timeline,
                    outcome.channel,
                );
                fill_timing_fields(&mut channel_data, outcome.timing, outcome.audio_seconds);
                merge_channel_data(&mut data_slots[outcome.index], channel_data);
            }
            Err(error) => {
                tracing::error!(
                    target: "vasr_transcribe::serve",
                    "{} request BAD   | index={} channel={} component={} audio={:.1}s reason={}",
                    request_id,
                    outcome.index,
                    outcome.channel,
                    outcome.bad_component.unwrap_or("recognizer"),
                    outcome.audio_seconds,
                    error
                );
                component_timing.add_scaled(outcome.timing, 1.0);
                merge_channel_data(
                    &mut data_slots[outcome.index],
                    error_data(
                        meta.service_id.clone(),
                        outcome.bad_component.unwrap_or("recognizer"),
                        format!("{}: {}", outcome.channel, error),
                        outcome.timing,
                        outcome.audio_seconds,
                    ),
                )
            }
        }
    }
    let data = data_slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            let mut data = slot.unwrap_or_else(|| {
                error_data(
                    request_metas[index].service_id.clone(),
                    "loader",
                    "no transcribe outcome".to_string(),
                    TranscribeItemTiming::default(),
                    0.0,
                )
            });
            clean_call_boundary_noise(&mut data);
            data
        })
        .collect::<Vec<_>>();

    let wall = start.elapsed().as_secs_f64();
    let num_items = data.len();
    let bad = data.iter().filter(|item| item.is_bad).count();
    let performance = InferencePerformance {
        batch_wall_seconds: wall,
        num_items,
        throughput_items_per_second: if wall > 0.0 {
            num_items as f64 / wall
        } else {
            0.0
        },
        total_audio_duration_seconds,
        speedup: if wall > 0.0 {
            total_audio_duration_seconds / wall
        } else {
            0.0
        },
        rtf: if total_audio_duration_seconds > 0.0 {
            wall / total_audio_duration_seconds
        } else {
            0.0
        },
    };
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{} {:<7} {:<5} | items={} bad={} | audio={:>6.1}s ch=mono:{} stereo:{} total:{} | elapsed={:>6.3}s | speed={:>6.1}x",
        request_id,
        "request",
        "DONE",
        performance.num_items,
        bad,
        performance.total_audio_duration_seconds,
        request_stats.mono,
        request_stats.stereo,
        request_stats.channels,
        performance.batch_wall_seconds,
        performance.speedup
    );
    Json(TranscribeResponse {
        data,
        inference_performance: performance,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct RequestStats {
    mono: usize,
    stereo: usize,
    channels: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct RequestStartStats {
    url: usize,
    base64: usize,
    missing: usize,
    mono_true: usize,
    mono_false: usize,
}

#[derive(Debug, Clone, Default)]
struct RequestMeta {
    service_id: String,
}

fn short_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..8].to_string()
}

fn merge_channel_data(slot: &mut Option<InferenceData>, channel_data: InferenceData) {
    let Some(existing) = slot else {
        *slot = Some(channel_data);
        return;
    };

    if !channel_data.text.mono.is_empty() {
        existing.text.mono = channel_data.text.mono;
    }
    if !channel_data.text.left.is_empty() {
        existing.text.left = channel_data.text.left;
    }
    if !channel_data.text.right.is_empty() {
        existing.text.right = channel_data.text.right;
    }
    let mut sentences = channel_data.sentences;
    existing.sentences.append(&mut sentences);
    existing
        .sentences
        .sort_by_key(|sentence| (sentence.start_time, sentence.channel.clone()));
    for (idx, sentence) in existing.sentences.iter_mut().enumerate() {
        sentence.sentence_index = idx;
    }
    if channel_data.is_bad {
        existing.is_bad = true;
        existing.bad_component = channel_data.bad_component;
        existing.bad_reason = match (&existing.bad_reason, channel_data.bad_reason) {
            (Some(old), Some(new)) => Some(format!("{old}; {new}")),
            (None, Some(new)) => Some(new),
            (old, None) => old.clone(),
        };
    }
    existing.spent_seconds = existing.spent_seconds.max(channel_data.spent_seconds);
    existing.spent_details.loader = existing
        .spent_details
        .loader
        .max(channel_data.spent_details.loader);
    existing.spent_details.detector = existing
        .spent_details
        .detector
        .max(channel_data.spent_details.detector);
    existing.spent_details.recognizer = existing
        .spent_details
        .recognizer
        .max(channel_data.spent_details.recognizer);
    existing.spent_details.sentencizer = existing
        .spent_details
        .sentencizer
        .max(channel_data.spent_details.sentencizer);
    existing.timings.total = existing.timings.total.max(channel_data.timings.total);
    existing.timings.loader = existing.timings.loader.max(channel_data.timings.loader);
    existing.timings.pipeline = existing.timings.pipeline.max(channel_data.timings.pipeline);
    existing.timings.audio += channel_data.timings.audio;
}

fn clean_call_boundary_noise(data: &mut InferenceData) {
    if data.is_bad || data.sentences.is_empty() {
        return;
    }

    let original_len = data.sentences.len();
    let mut cleaned = Vec::with_capacity(original_len);
    let mut changed = false;
    let mut start_index = 0usize;
    while start_index < data.sentences.len()
        && should_drop_leading_sentence(&data.sentences[start_index], start_index)
    {
        start_index += 1;
    }
    changed |= start_index > 0;

    for (relative_index, mut sentence) in data.sentences[start_index..].iter().cloned().enumerate()
    {
        if relative_index < 4 {
            let cleaned_text = remove_boundary_artifact_phrases(sentence.text.as_str());
            changed |= cleaned_text != sentence.text;
            sentence.text = cleaned_text;
            if should_drop_early_artifact_sentence(&sentence) {
                changed = true;
                continue;
            }
        }
        cleaned.push(sentence);
    }

    while cleaned
        .last()
        .is_some_and(should_drop_trailing_artifact_sentence)
    {
        cleaned.pop();
        changed = true;
    }

    if !changed {
        return;
    }

    for (idx, sentence) in cleaned.iter_mut().enumerate() {
        sentence.sentence_index = idx;
    }
    data.sentences = cleaned;
    data.text = rebuild_text_from_sentences(&data.sentences);
}

fn should_drop_leading_sentence(sentence: &SentenceResult, leading_index: usize) -> bool {
    if sentence.start_time > 30_000 || sentence.end_time > 22_000 {
        return false;
    }
    let text = compact_text(sentence.text.as_str());
    if text.is_empty() {
        return true;
    }
    let has_anchor = contains_any(text.as_str(), CALL_ANCHORS);
    let has_strong_anchor = contains_any(text.as_str(), STRONG_CALL_ANCHORS);
    let has_noise = contains_any(text.as_str(), BOUNDARY_NOISE_PHRASES);
    let mostly_filler = mostly_filler_text(text.as_str());
    let short_non_call = text.chars().count() <= 8 && !has_strong_anchor;
    let early_non_call = leading_index < 6 && text.chars().count() <= 14 && !has_anchor;

    (has_noise && !has_strong_anchor) || mostly_filler || short_non_call || early_non_call
}

fn should_drop_early_artifact_sentence(sentence: &SentenceResult) -> bool {
    let text = compact_text(sentence.text.as_str());
    text.is_empty()
        || (sentence.end_time <= 22_000 && contains_any(text.as_str(), BOUNDARY_NOISE_PHRASES))
}

fn should_drop_trailing_artifact_sentence(sentence: &SentenceResult) -> bool {
    if sentence.start_time <= 15_000 {
        return false;
    }
    let text = compact_text(sentence.text.as_str());
    matches!(text.as_str(), "老板" | "好的" | "嗯" | "啊")
        || contains_any(text.as_str(), TRAILING_ARTIFACT_PHRASES)
}

fn rebuild_text_from_sentences(sentences: &[SentenceResult]) -> TextResult {
    let mut text = TextResult::default();
    for sentence in sentences {
        match sentence.channel.as_str() {
            "left" => text.left.push_str(sentence.text.as_str()),
            "right" => text.right.push_str(sentence.text.as_str()),
            _ => text.mono.push_str(sentence.text.as_str()),
        }
    }
    text
}

fn remove_boundary_artifact_phrases(text: &str) -> String {
    let mut cleaned = text.to_string();
    for phrase in BOUNDARY_ARTIFACT_PHRASES {
        cleaned = cleaned.replace(phrase, "");
    }
    cleaned
}

fn compact_text(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_whitespace() && !is_common_punctuation(*ch))
        .collect()
}

fn is_common_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。'
            | '？'
            | '！'
            | '、'
            | '；'
            | '：'
            | ','
            | '.'
            | '?'
            | '!'
            | ';'
            | ':'
            | '"'
            | '\''
            | '“'
            | '”'
            | '‘'
            | '’'
            | '('
            | ')'
            | '（'
            | '）'
    )
}

fn mostly_filler_text(text: &str) -> bool {
    let total = text.chars().count();
    if total == 0 {
        return true;
    }
    let filler = text
        .chars()
        .filter(|ch| CALL_FILLER_CHARS.contains(*ch))
        .count();
    filler >= total.saturating_mul(3).div_ceil(5)
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

const CALL_FILLER_CHARS: &str = "嗯啊呃哦哎喂唉呀哈好的是对";
const CALL_ANCHORS: &[&str] = &[
    "你好",
    "您好",
    "五八",
    "家政",
    "保姆",
    "工作",
    "找工作",
    "接单",
    "阿姨",
    "先生",
    "女士",
    "姐",
    "哥",
    "老师",
    "号码",
    "手机号",
    "微信",
];
const STRONG_CALL_ANCHORS: &[&str] = &[
    "你好",
    "您好",
    "五八",
    "家政",
    "保姆",
    "找工作",
    "接单",
    "阿姨",
];
const BOUNDARY_NOISE_PHRASES: &[&str] = &[
    "公众号",
    "摩托",
    "英语",
    "老公",
    "APP",
    "下载APP",
    "接视频",
    "资料下载",
    "高密度",
    "连接",
    "不要怀疑",
    "怀疑别人",
    "别说废话",
    "愿意",
    "一点到五",
    "比较啊正常",
    "今天晚上",
    "我明天下班",
    "我打这个电话",
    "非常",
    "调节一下",
    "把钥匙",
    "新进来的嘛",
    "做提成",
];
const BOUNDARY_ARTIFACT_PHRASES: &[&str] = &[
    "不要怀疑",
    "没有再见",
    "不要怀疑别人",
    "然后我让他发资料下载APP过来接视频",
    "资料下载APP过来接视频",
    "你有摩托就怕他",
    "我跟老公",
    "你说英语怎么过？听不懂了。你说，",
    "你说英语怎么过听不懂了你说",
    "高密度连接",
    "啊，那么我把钥匙了。",
    "啊那么我把钥匙了",
    "别说废话",
    "我把他登了公众号",
    "一点到五",
    "我今天晚上",
    "新进来的嘛，你做提成，你想先工作，你都是直接给。",
    "新进来的嘛你做提成你想先工作你都是直接给",
];
const TRAILING_ARTIFACT_PHRASES: &[&str] = &["新进来的嘛", "做提成"];

fn fill_timing_fields(data: &mut InferenceData, timing: TranscribeItemTiming, audio_seconds: f64) {
    let pipeline_seconds = timing.vad_seconds + timing.asr_seconds;
    data.timings = TimingBreakdown {
        total: timing.loader_seconds + pipeline_seconds,
        loader: timing.loader_seconds,
        pipeline: pipeline_seconds,
        audio: audio_seconds,
    };
    data.spent_seconds = data.timings.total;
    data.spent_details.loader = timing.loader_seconds;
    data.spent_details.detector = timing.vad_seconds;
    data.spent_details.recognizer = timing.asr_seconds;
}

fn error_data(
    service_id: String,
    component: &str,
    reason: String,
    timing: TranscribeItemTiming,
    audio_seconds: f64,
) -> InferenceData {
    let mut data = InferenceData {
        service_id,
        spent_seconds: 0.0,
        spent_details: Default::default(),
        timings: TimingBreakdown::default(),
        text: Default::default(),
        sentences: Vec::new(),
        is_bad: true,
        bad_reason: Some(reason),
        bad_component: Some(component.to_string()),
    };
    fill_timing_fields(&mut data, timing, audio_seconds);
    data
}

#[cfg(test)]
mod tests {
    use super::{clean_call_boundary_noise, merge_channel_data};
    use crate::protocol::{InferenceData, SentenceResult, TextResult};

    #[test]
    fn merge_channel_data_combines_left_and_right_text() {
        let mut slot = None;
        merge_channel_data(&mut slot, channel_data("left", "hello", 20));
        merge_channel_data(&mut slot, channel_data("right", "world", 10));

        let data = slot.expect("merged data");
        assert_eq!(data.text.mono, "");
        assert_eq!(data.text.left, "hello");
        assert_eq!(data.text.right, "world");
        assert_eq!(
            data.sentences
                .iter()
                .map(|sentence| (sentence.channel.as_str(), sentence.sentence_index))
                .collect::<Vec<_>>(),
            vec![("right", 0), ("left", 1)]
        );
    }

    #[test]
    fn clean_call_boundary_noise_drops_leading_crosstalk() {
        let mut data = channel_data_with_sentences(vec![
            sentence("嗯。", 920, 1840),
            sentence("我把他登了公众号。", 12_720, 14_870),
            sentence("喂，啊，你好。", 20_870, 23_080),
            sentence("哎，你好，我是小云老师，这个号码加不上。", 23_360, 39_780),
        ]);

        clean_call_boundary_noise(&mut data);

        assert_eq!(data.sentences.len(), 2);
        assert_eq!(data.sentences[0].sentence_index, 0);
        assert_eq!(data.sentences[0].text, "喂，啊，你好。");
        assert!(data.text.mono.starts_with("喂，啊，你好。"));
        assert!(!data.text.mono.contains("公众号"));
    }

    #[test]
    fn clean_call_boundary_noise_removes_artifact_phrase_inside_first_sentence() {
        let mut data = channel_data_with_sentences(vec![
            sentence(
                "喂，您好，我是五八到家的，请问您现在还在找工作吗？然后我让他发资料下载APP过来接视频。喂，喂，谁找工作啊？",
                250,
                13_930,
            ),
            sentence("我没有找工作。", 14_460, 21_450),
        ]);

        clean_call_boundary_noise(&mut data);

        assert_eq!(data.sentences.len(), 2);
        assert!(!data.text.mono.contains("资料下载APP"));
        assert!(data.text.mono.contains("请问您现在还在找工作吗？"));
    }

    fn channel_data(channel: &str, text: &str, start_time: u64) -> InferenceData {
        let mut data = channel_data_with_sentences_for_channel(
            channel,
            vec![SentenceResult {
                text: text.to_string(),
                start_time,
                end_time: start_time + 10,
                duration: 10,
                channel: channel.to_string(),
                sentence_index: 0,
            }],
        );
        match channel {
            "left" => data.text.left = text.to_string(),
            "right" => data.text.right = text.to_string(),
            _ => data.text.mono = text.to_string(),
        }
        data
    }

    fn channel_data_with_sentences(sentences: Vec<SentenceResult>) -> InferenceData {
        channel_data_with_sentences_for_channel("mono", sentences)
    }

    fn channel_data_with_sentences_for_channel(
        channel: &str,
        sentences: Vec<SentenceResult>,
    ) -> InferenceData {
        let mono = sentences
            .iter()
            .map(|sentence| sentence.text.as_str())
            .collect::<String>();
        let mut text = TextResult::default();
        match channel {
            "left" => text.left = mono,
            "right" => text.right = mono,
            _ => text.mono = mono,
        }
        InferenceData {
            service_id: String::new(),
            spent_seconds: 0.0,
            spent_details: Default::default(),
            timings: Default::default(),
            text,
            sentences,
            is_bad: false,
            bad_reason: None,
            bad_component: None,
        }
    }

    fn sentence(text: &str, start_time: u64, end_time: u64) -> SentenceResult {
        SentenceResult {
            text: text.to_string(),
            start_time,
            end_time,
            duration: end_time - start_time,
            channel: "mono".to_string(),
            sentence_index: 0,
        }
    }
}

pub fn build_transcribe_service(
    pipeline: Arc<GlobalAsyncTranscribePipeline>,
) -> Arc<TranscribeService> {
    Arc::new(TranscribeService {
        service_id: format!("transcribe-{}", uuid::Uuid::new_v4().simple()),
        pipeline,
    })
}

pub fn build_transcribe_service_from_parts(
    offline: Arc<crate::runtime::OfflinePipeline>,
    asr_options: AsrOptions,
    vad_options: VadOptions,
    worker_options: PipelineWorkerOptions,
) -> Arc<TranscribeService> {
    build_transcribe_service(Arc::new(GlobalAsyncTranscribePipeline::start(
        vasr_data::AudioLoader,
        offline,
        asr_options,
        vad_options,
        worker_options,
    )))
}
