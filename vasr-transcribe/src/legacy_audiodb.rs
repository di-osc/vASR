use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use rusqlite::{Connection, OpenFlags};
use serde::Deserialize;
use asr_data::{
    Audio, AudioChannel, AudioEncoding, AudioEvent, AudioFormat, AudioInfo, AudioSource,
    MAX_QUERY_LIMIT, Sentence, Speaker, Speech, TimeRange, Token, Transcription,
};

const APPLICATION_ID: i64 = 0x5641_5352; // "VASR"
const LEGACY_SCHEMA_V1: i64 = 1;
const LEGACY_SCHEMA_V2: i64 = 2;

#[derive(Debug, Deserialize)]
struct LegacyTimeline {
    #[serde(default)]
    id: String,
    #[serde(default)]
    audio_id: String,
    #[serde(default)]
    duration: Option<usize>,
    #[serde(default)]
    annotations: Vec<LegacyAnnotation>,
}

#[derive(Debug, Deserialize)]
struct LegacyAnnotation {
    id: String,
    range: TimeRange,
    status: LegacyStatus,
    source: LegacySource,
    confidence: Option<f32>,
    payload: LegacyPayload,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
enum LegacyStatus {
    Partial,
    Final,
    Revised,
    Deleted,
}

#[derive(Debug, Deserialize)]
enum LegacySource {
    User,
    Model(String),
    Stage(String),
    System,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
enum LegacyPayload {
    Speech,
    Silence,
    Token(Token),
    #[serde(alias = "Segment")]
    Transcription(LegacyTextSpan),
    Sentence(LegacyTextSpan),
    Speaker(String),
    Language(String),
    Hotword(serde_json::Value),
    AcousticEvent(serde_json::Value),
    Diagnostic(serde_json::Value),
}

#[derive(Debug, Deserialize)]
struct LegacyTextSpan {
    text: String,
    #[serde(default)]
    tokens: Vec<Token>,
    language: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LegacyFormat {
    channels: Option<u16>,
    encoding: Option<AudioEncoding>,
    sample_rate: Option<u32>,
}

pub fn read_legacy_audiodb(path: &Path, schema_version: i64) -> Result<Vec<Audio>> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("failed to open legacy AudioDb {}", path.display()))?;
    let application_id: i64 = connection
        .pragma_query_value(None, "application_id", |row| row.get(0))
        .context("failed to read AudioDb application id")?;
    if application_id != APPLICATION_ID {
        bail!("file is not an ASR AudioDb (application id {application_id:#x})");
    }

    let sql = match schema_version {
        LEGACY_SCHEMA_V1 => {
            "SELECT audio_id, source, timeline, metadata, duration_ms
             FROM audios
             ORDER BY audio_id
             LIMIT ?1"
        }
        LEGACY_SCHEMA_V2 => {
            "SELECT audios.audio_id, audio_sources.source, timelines.timeline, audios.metadata,
                    audios.duration_ms
             FROM audios
             JOIN audio_sources USING (audio_id)
             JOIN timelines USING (audio_id)
             ORDER BY audios.audio_id
             LIMIT ?1"
        }
        other => bail!("unsupported legacy audio database schema version {other}"),
    };

    let limit = i64::try_from(MAX_QUERY_LIMIT).unwrap_or(i64::MAX);
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map([limit], |row| {
        Ok(LegacyRow {
            audio_id: row.get(0)?,
            source: row.get(1)?,
            timeline: row.get(2)?,
            metadata: row.get(3)?,
            duration_ms: row.get(4)?,
        })
    })?;

    let mut audios = Vec::new();
    for row in rows {
        let row = row?;
        audios.push(
            convert_legacy_row(&row)
                .with_context(|| format!("failed to convert legacy audio {}", row.audio_id))?,
        );
    }
    Ok(audios)
}

struct LegacyRow {
    audio_id: String,
    source: Vec<u8>,
    timeline: Vec<u8>,
    metadata: String,
    duration_ms: Option<i64>,
}

fn convert_legacy_row(row: &LegacyRow) -> Result<Audio> {
    let source: AudioSource = rmp_serde::from_slice(&row.source).context("decode audio source")?;
    let timeline: LegacyTimeline =
        rmp_serde::from_slice(&row.timeline).context("decode audio timeline")?;
    let metadata: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&row.metadata).context("decode audio metadata")?;
    let duration_ms = row
        .duration_ms
        .and_then(|value| usize::try_from(value).ok())
        .or(timeline.duration)
        .unwrap_or(0);
    let info = audio_info_from_metadata(&metadata, duration_ms);
    let duration = info.timeline_duration_ms();
    let (reference, prediction) = convert_annotations(&timeline.annotations);

    let mut audio = Audio::with_id_from_info(row.audio_id.clone(), source, &info);
    audio.metadata = metadata;
    let audio_id = audio.id.clone();
    let converted = audio
        .ensure_timeline(AudioChannel::Mono, Some(duration))
        .map_err(|err| anyhow!("{err}"))?;
    if !timeline.id.trim().is_empty() {
        converted.id.clone_from(&timeline.id);
    }
    if !timeline.audio_id.trim().is_empty() {
        converted.audio_id = timeline.audio_id;
    } else {
        converted.audio_id = audio_id;
    }
    converted.duration = duration;
    converted.reference = reference;
    converted.prediction = prediction;
    Ok(audio)
}

fn audio_info_from_metadata(
    metadata: &BTreeMap<String, serde_json::Value>,
    duration_ms: usize,
) -> AudioInfo {
    let format = metadata
        .get("legacy_format")
        .cloned()
        .and_then(|value| serde_json::from_value::<LegacyFormat>(value).ok());
    let sample_rate = format
        .as_ref()
        .and_then(|format| format.sample_rate)
        .filter(|sample_rate| *sample_rate > 0)
        .unwrap_or(16_000);
    let channels = format
        .as_ref()
        .and_then(|format| format.channels)
        .filter(|channels| *channels > 0)
        .unwrap_or(1);
    let encoding = format
        .as_ref()
        .and_then(|format| format.encoding.clone())
        .unwrap_or(AudioEncoding::Unknown);
    let frame_count = u64::from(sample_rate).saturating_mul(duration_ms as u64) / 1_000;
    AudioInfo {
        sample_rate,
        channels,
        frame_count,
        source_format: AudioFormat {
            encoding,
            sample_rate,
            channels,
        },
    }
}

fn convert_annotations(annotations: &[LegacyAnnotation]) -> (Vec<AudioEvent>, Vec<AudioEvent>) {
    let mut reference = Vec::new();
    let mut prediction = Vec::new();
    for annotation in annotations {
        if annotation.status == LegacyStatus::Deleted {
            continue;
        }
        let is_reference = matches!(annotation.source, LegacySource::User)
            && matches!(
                annotation.status,
                LegacyStatus::Final | LegacyStatus::Revised
            );
        let source = if is_reference {
            None
        } else {
            Some(prediction_source(&annotation.source))
        };
        let id = if annotation.id.trim().is_empty() {
            String::new()
        } else {
            annotation.id.clone()
        };
        let Some(event) = convert_payload(annotation, id, source) else {
            continue;
        };
        if is_reference {
            reference.push(event);
        } else {
            prediction.push(event);
        }
    }
    (reference, prediction)
}

fn prediction_source(source: &LegacySource) -> String {
    match source {
        LegacySource::User => "user".to_string(),
        LegacySource::Model(name) | LegacySource::Stage(name) => {
            let trimmed = name.trim();
            if trimmed.is_empty() {
                "unknown".to_string()
            } else {
                trimmed.to_string()
            }
        }
        LegacySource::System => "system".to_string(),
    }
}

/// 把一条旧标注转成新事件模型里的 [`AudioEvent`]。
///
/// 旧模型把 token / 句子 / 语种 / 说话人分散在不同 span 里，新模型统一收进
/// [`Speech`]。这里做尽力而为的映射：token 作为句子挂到转写上，语种挂到语音上，
/// 说话人挂到语音上。
fn convert_payload(
    annotation: &LegacyAnnotation,
    id: String,
    source: Option<String>,
) -> Option<AudioEvent> {
    let range = annotation.range;
    match &annotation.payload {
        LegacyPayload::Speech => Some(AudioEvent::Speech(Speech {
            id,
            range,
            source,
            confidence: annotation.confidence,
            language: None,
            speaker: None,
            transcription: None,
        })),
        LegacyPayload::Silence => Some(AudioEvent::Base {
            id,
            range,
            source,
            name: "silence".to_string(),
            confidence: annotation.confidence,
        }),
        LegacyPayload::Token(token) => {
            let sentence = Sentence {
                text: token.text.clone(),
                range: token.range.unwrap_or(range),
                tokens: vec![token.clone()],
            };
            let transcription = Transcription {
                text: token.text.clone(),
                sentences: vec![sentence],
                confidence: token.confidence,
            };
            Some(speech_with_transcription(id, range, source, None, transcription))
        }
        LegacyPayload::Transcription(span) => {
            let transcription = legacy_transcription(span, range, annotation.confidence);
            Some(speech_with_transcription(
                id,
                range,
                source,
                span.language.clone(),
                transcription,
            ))
        }
        LegacyPayload::Sentence(span) => {
            let sentence = Sentence {
                text: span.text.clone(),
                range,
                tokens: span.tokens.clone(),
            };
            let transcription = Transcription {
                text: span.text.clone(),
                sentences: vec![sentence],
                confidence: None,
            };
            Some(speech_with_transcription(
                id,
                range,
                source,
                span.language.clone(),
                transcription,
            ))
        }
        LegacyPayload::Speaker(name) => Some(AudioEvent::Speech(Speech {
            id,
            range,
            source,
            confidence: annotation.confidence,
            language: None,
            speaker: Some(Speaker::new(name.clone())),
            transcription: None,
        })),
        LegacyPayload::Language(tag) => Some(AudioEvent::Speech(Speech {
            id,
            range,
            source,
            confidence: annotation.confidence,
            language: Some(tag.clone()),
            speaker: None,
            transcription: None,
        })),
        LegacyPayload::Hotword(_)
        | LegacyPayload::AcousticEvent(_)
        | LegacyPayload::Diagnostic(_) => None,
    }
}

fn legacy_transcription(
    span: &LegacyTextSpan,
    range: TimeRange,
    confidence: Option<f32>,
) -> Transcription {
    let sentences = if span.tokens.is_empty() {
        Vec::new()
    } else {
        vec![Sentence {
            text: span.text.clone(),
            range,
            tokens: span.tokens.clone(),
        }]
    };
    Transcription {
        text: span.text.clone(),
        sentences,
        confidence,
    }
}

fn speech_with_transcription(
    id: String,
    range: TimeRange,
    source: Option<String>,
    language: Option<String>,
    transcription: Transcription,
) -> AudioEvent {
    AudioEvent::Speech(Speech {
        id,
        range,
        source,
        confidence: None,
        language,
        speaker: None,
        transcription: Some(transcription),
    })
}
