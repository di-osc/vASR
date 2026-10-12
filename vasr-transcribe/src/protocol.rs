use anyhow::{Context, Result};
use asr_data::{Audio, AudioChannel, AudioInfo, TimeRange, Timeline};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelScopeSourceDto {
    pub repo_id: String,
    pub file_path: String,
    #[serde(default)]
    pub revision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInputDto {
    /// Optional identity and probe metadata used to preserve an asr-data Audio record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub info: Option<AudioInfo>,
    pub url: Option<String>,
    pub b64_str: Option<String>,
    /// ModelScope 数据集里的单个音频文件。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modelscope: Option<ModelScopeSourceDto>,
    #[serde(default, deserialize_with = "deserialize_optional_bool")]
    pub mono: Option<bool>,
    pub hotword: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeRequest {
    pub inputs: Vec<AudioInputDto>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextResult {
    pub mono: String,
    pub left: String,
    pub right: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SentenceResult {
    pub text: String,
    pub start_time: u64,
    pub end_time: u64,
    pub duration: u64,
    pub channel: String,
    pub sentence_index: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpentDetails {
    #[serde(default)]
    pub loader: f64,
    #[serde(default)]
    pub detector: f64,
    #[serde(default)]
    pub recognizer: f64,
    #[serde(default)]
    pub sentencizer: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimingBreakdown {
    #[serde(default)]
    pub total: f64,
    #[serde(default)]
    pub loader: f64,
    #[serde(default)]
    pub pipeline: f64,
    #[serde(default)]
    pub audio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceData {
    #[serde(rename = "serviceId")]
    pub service_id: String,
    #[serde(rename = "spentSeconds")]
    pub spent_seconds: f64,
    #[serde(rename = "spentDetails")]
    #[serde(default)]
    pub spent_details: SpentDetails,
    #[serde(default)]
    pub timings: TimingBreakdown,
    pub text: TextResult,
    pub sentences: Vec<SentenceResult>,
    pub is_bad: bool,
    pub bad_reason: Option<String>,
    pub bad_component: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePerformance {
    pub batch_wall_seconds: f64,
    pub num_items: usize,
    pub throughput_items_per_second: f64,
    pub total_audio_duration_seconds: f64,
    pub speedup: f64,
    pub rtf: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeResponse {
    pub data: Vec<InferenceData>,
    pub inference_performance: InferencePerformance,
}

const TRANSCRIBE_ERROR_METADATA_KEY: &str = "vasr-transcribe-error";

/// Remove metadata emitted by earlier transcribe response versions. Audio metadata
/// otherwise remains untouched because it belongs to the asr-data record.
pub fn clear_legacy_transcribe_metadata(audio: &mut Audio) {
    for key in [
        "vasr-transcribe",
        "vasr-transcribe-source-error",
        "vasr-transcribe-probe-warning",
    ] {
        audio.metadata.remove(key);
    }
}

/// Preserve item-level failures in Audio's extensible metadata without adding
/// fields from the former transcribe response envelope.
pub fn mark_transcription_failed(audio: &mut Audio, component: &str, reason: &str) {
    audio.metadata.insert(
        TRANSCRIBE_ERROR_METADATA_KEY.to_string(),
        serde_json::json!({
            "component": component,
            "reason": reason,
        }),
    );
}

pub fn clear_transcription_error(audio: &mut Audio) {
    audio.metadata.remove(TRANSCRIBE_ERROR_METADATA_KEY);
}

/// Decode Audio JSONL into the legacy in-process shape used by eval/perf/stress reports.
/// The wire format remains plain `asr_data::Audio`; timing values are measured by the client.
pub fn parse_transcribe_jsonl(body: &str, batch_wall_seconds: f64) -> Result<TranscribeResponse> {
    let mut audios = Vec::new();
    for (line_index, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        audios.push(serde_json::from_str::<Audio>(line).with_context(|| {
            format!("failed to decode transcribe JSONL line {}", line_index + 1)
        })?);
    }

    let total_audio_duration_seconds = audios
        .iter()
        .map(|audio| audio.info.duration_ms() as f64 / 1_000.0)
        .sum::<f64>();
    let num_items = audios.len();
    let batch_wall_seconds = batch_wall_seconds.max(0.0);
    let data = audios.iter().map(inference_from_audio).collect();
    Ok(TranscribeResponse {
        data,
        inference_performance: InferencePerformance {
            batch_wall_seconds,
            num_items,
            throughput_items_per_second: num_items as f64 / batch_wall_seconds.max(f64::EPSILON),
            total_audio_duration_seconds,
            speedup: total_audio_duration_seconds / batch_wall_seconds.max(f64::EPSILON),
            rtf: batch_wall_seconds / total_audio_duration_seconds.max(f64::EPSILON),
        },
    })
}

fn inference_from_audio(audio: &Audio) -> InferenceData {
    let audio_seconds = audio.info.duration_ms() as f64 / 1_000.0;
    let transcription_error = audio.metadata.get(TRANSCRIBE_ERROR_METADATA_KEY);
    let mut inference = InferenceData {
        service_id: audio.id.clone(),
        spent_seconds: 0.0,
        spent_details: SpentDetails::default(),
        timings: TimingBreakdown {
            audio: audio_seconds,
            ..TimingBreakdown::default()
        },
        text: TextResult::default(),
        sentences: Vec::new(),
        is_bad: transcription_error.is_some(),
        bad_reason: transcription_error
            .and_then(|error| error.get("reason"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        bad_component: transcription_error
            .and_then(|error| error.get("component"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
    };
    for (channel, channel_name) in [
        (AudioChannel::Mono, "mono"),
        (AudioChannel::Left, "left"),
        (AudioChannel::Right, "right"),
    ] {
        let Ok(Some(timeline)) = audio.timeline(channel) else {
            continue;
        };
        let channel_data =
            InferenceData::from_timeline_channel(audio.id.clone(), timeline, channel_name);
        match channel_name {
            "left" => inference.text.left = channel_data.text.left,
            "right" => inference.text.right = channel_data.text.right,
            _ => inference.text.mono = channel_data.text.mono,
        }
        inference.sentences.extend(channel_data.sentences);
    }
    inference
        .sentences
        .sort_by_key(|sentence| (sentence.start_time, sentence.channel.clone()));
    for (index, sentence) in inference.sentences.iter_mut().enumerate() {
        sentence.sentence_index = index;
    }
    inference
}

impl InferenceData {
    pub fn from_timeline(service_id: impl Into<String>, timeline: &Timeline) -> Self {
        Self::from_timeline_channel(service_id, timeline, "mono")
    }

    pub fn from_timeline_channel(
        service_id: impl Into<String>,
        timeline: &Timeline,
        channel: &str,
    ) -> Self {
        let transcript = vasr::prediction_text(timeline);
        let mut sentences = Vec::new();
        for event in &timeline.prediction {
            let Some(transcription) = event.transcription() else {
                continue;
            };
            if transcription.sentences.is_empty() {
                let range = event.range();
                sentences.push(sentence_result(
                    transcription.text.clone(),
                    range,
                    channel,
                    sentences.len(),
                ));
            } else {
                for sentence in &transcription.sentences {
                    sentences.push(sentence_result(
                        sentence.text.clone(),
                        sentence.range,
                        channel,
                        sentences.len(),
                    ));
                }
            }
        }
        let mut text = TextResult::default();
        match channel {
            "left" => text.left = transcript,
            "right" => text.right = transcript,
            _ => text.mono = transcript,
        }
        Self {
            service_id: service_id.into(),
            spent_seconds: 0.0,
            spent_details: SpentDetails::default(),
            timings: TimingBreakdown::default(),
            text,
            sentences,
            is_bad: false,
            bad_reason: None,
            bad_component: None,
        }
    }
}

fn sentence_result(
    text: String,
    range: TimeRange,
    channel: &str,
    sentence_index: usize,
) -> SentenceResult {
    SentenceResult {
        text,
        start_time: range.start_ms as u64,
        end_time: range.end_ms as u64,
        duration: range.duration() as u64,
        channel: channel.to_string(),
        sentence_index,
    }
}

fn deserialize_optional_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct OptionalBoolVisitor;

    impl<'de> serde::de::Visitor<'de> for OptionalBoolVisitor {
        type Value = Option<bool>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a boolean or \"true\"/\"false\" string")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_any(self)
        }

        fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
            Ok(Some(value))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            match value.trim() {
                "true" => Ok(Some(true)),
                "false" => Ok(Some(false)),
                _ => Err(E::invalid_value(serde::de::Unexpected::Str(value), &self)),
            }
        }
    }

    deserializer.deserialize_option(OptionalBoolVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_input_mono_accepts_bool_and_string() -> serde_json::Result<()> {
        let bool_value: AudioInputDto = serde_json::from_str(r#"{"mono":true}"#)?;
        let string_value: AudioInputDto = serde_json::from_str(r#"{"mono":"false"}"#)?;
        let missing_value: AudioInputDto = serde_json::from_str(r#"{}"#)?;

        assert_eq!(bool_value.mono, Some(true));
        assert_eq!(string_value.mono, Some(false));
        assert_eq!(missing_value.mono, None);
        Ok(())
    }
}
