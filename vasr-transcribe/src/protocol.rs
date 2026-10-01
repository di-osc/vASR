use serde::{Deserialize, Serialize};
use asr_data::{TimeRange, Timeline};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelScopeSourceDto {
    pub repo_id: String,
    pub file_path: String,
    #[serde(default)]
    pub revision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInputDto {
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
