use anyhow::{Context, Result};
use asr_data::{Audio, AudioChannel, AudioInfo};
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

/// 该 Audio 的整段音频时长（秒）。
pub(crate) fn audio_seconds(audio: &Audio) -> f64 {
    audio.info.duration_ms() / 1_000.0
}

/// 该 Audio 是否带 item 级失败标记。
pub(crate) fn audio_is_bad(audio: &Audio) -> bool {
    audio.metadata.get(TRANSCRIBE_ERROR_METADATA_KEY).is_some()
}

/// item 级失败原因，来自 `mark_transcription_failed` 写入的 metadata。
pub(crate) fn audio_bad_reason(audio: &Audio) -> Option<String> {
    audio
        .metadata
        .get(TRANSCRIBE_ERROR_METADATA_KEY)
        .and_then(|error| error.get("reason"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// item 级失败组件，来自 `mark_transcription_failed` 写入的 metadata。
pub(crate) fn audio_bad_component(audio: &Audio) -> Option<String> {
    audio
        .metadata
        .get(TRANSCRIBE_ERROR_METADATA_KEY)
        .and_then(|error| error.get("component"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

/// 指定声道的预测文本；没有该时间轴时返回空串。
pub(crate) fn audio_channel_text(audio: &Audio, channel: AudioChannel) -> String {
    audio
        .timeline(channel)
        .ok()
        .flatten()
        .map(vasr::prediction_text)
        .unwrap_or_default()
}

/// 从服务端 Audio JSONL 响应里解出 `asr_data::Audio` 记录。
///
/// wire format 本身就是 `Audio`；这里不做任何信息丢失的转换。
pub(crate) fn parse_transcribe_audios(body: &str) -> Result<Vec<Audio>> {
    let mut audios = Vec::new();
    for (line_index, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        audios.push(serde_json::from_str::<Audio>(line).with_context(|| {
            format!("failed to decode transcribe JSONL line {}", line_index + 1)
        })?);
    }
    Ok(audios)
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
