use std::path::Path;

use anyhow::{Context, Result};
use base64::Engine;
use vasr_data::{
    Audio, AudioDb, AudioDbError, AudioDbMode, AudioQuery, AudioSource, MAX_QUERY_LIMIT,
};

use crate::AudioInputDto;
use crate::legacy_audiodb::read_legacy_audiodb;

pub fn is_vasr_data_file(path: &Path) -> bool {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if file_name.ends_with(".vasr.msgpack") {
        return true;
    }
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "vasr" | "msgpack"))
        .unwrap_or(false)
}

pub fn read_vasr_audios(input: &Path) -> Result<Vec<Audio>> {
    match AudioDb::open(input, AudioDbMode::ReadOnly) {
        Ok(db) => db
            .query(&AudioQuery {
                limit: MAX_QUERY_LIMIT,
                ..AudioQuery::default()
            })
            .with_context(|| format!("failed to read vASR AudioDb from {}", input.display())),
        Err(AudioDbError::UnsupportedSchema {
            found: version @ (1 | 2),
            expected: _,
        }) => read_legacy_audiodb(input, version)
            .with_context(|| format!("failed to read vASR AudioDb from {}", input.display())),
        Err(error) => Err(error)
            .with_context(|| format!("failed to read vASR AudioDb from {}", input.display())),
    }
}

pub fn audio_input_from_source(
    source: &AudioSource,
    mono: Option<bool>,
    hotword: Option<String>,
) -> AudioInputDto {
    match source {
        AudioSource::Path(path) => AudioInputDto {
            url: Some(path.display().to_string()),
            b64_str: None,
            mono,
            hotword,
        },
        AudioSource::Url(url) => AudioInputDto {
            url: Some(url.clone()),
            b64_str: None,
            mono,
            hotword,
        },
        AudioSource::Base64(data) => AudioInputDto {
            url: None,
            b64_str: Some(data.clone()),
            mono,
            hotword,
        },
        AudioSource::EncodedBytes(bytes) | AudioSource::PcmS16Le { bytes, .. } => AudioInputDto {
            url: None,
            b64_str: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
            mono,
            hotword,
        },
    }
}

pub fn read_audio_input_dtos(path: &Path, mono: Option<bool>) -> Result<Vec<AudioInputDto>> {
    if is_vasr_data_file(path) {
        let audios = read_vasr_audios(path)?;
        return Ok(audios
            .iter()
            .map(|audio| audio_input_from_source(&audio.source, mono, None))
            .collect());
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| AudioInputDto {
            url: Some(line.to_string()),
            b64_str: None,
            mono,
            hotword: None,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{read_audio_input_dtos, read_vasr_audios};
    use base64::Engine;
    use rusqlite::Connection;
    use serde::Serialize;
    use std::path::PathBuf;
    use vasr_data::{AudioSource, TimeRange};

    const APPLICATION_ID: i64 = 0x5641_5352;

    #[derive(Serialize)]
    struct LegacyTimeline {
        id: String,
        audio_id: String,
        duration: u64,
        annotations: Vec<LegacyAnnotation>,
    }

    #[derive(Serialize)]
    struct LegacyAnnotation {
        id: String,
        range: TimeRange,
        status: &'static str,
        source: &'static str,
        confidence: Option<f32>,
        payload: LegacyPayload,
    }

    #[derive(Serialize)]
    enum LegacyPayload {
        Transcription(LegacyTextSpan),
    }

    #[derive(Serialize)]
    struct LegacyTextSpan {
        text: String,
        tokens: Vec<()>,
        language: Option<String>,
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "vasr-dataset-{name}-{}-{}.vasr",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ))
    }

    fn write_schema2_db(path: &std::path::Path, source: &AudioSource, text: &str) {
        let source_blob = rmp_serde::to_vec_named(source).expect("encode source");
        let timeline = LegacyTimeline {
            id: "tl_test".to_string(),
            audio_id: "call-1".to_string(),
            duration: 1_000,
            annotations: vec![LegacyAnnotation {
                id: "ann_test".to_string(),
                range: TimeRange::new(0, 1_000),
                status: "Final",
                source: "User",
                confidence: None,
                payload: LegacyPayload::Transcription(LegacyTextSpan {
                    text: text.to_string(),
                    tokens: Vec::new(),
                    language: None,
                }),
            }],
        };
        let timeline_blob = rmp_serde::to_vec_named(&timeline).expect("encode timeline");
        let metadata = serde_json::json!({
            "legacy_format": {
                "channels": 1,
                "encoding": "Wav",
                "sample_rate": 16_000
            }
        })
        .to_string();

        let connection = Connection::open(path).expect("open sqlite");
        connection
            .pragma_update(None, "application_id", APPLICATION_ID)
            .expect("application id");
        connection
            .pragma_update(None, "user_version", 2)
            .expect("schema version");
        connection
            .execute_batch(
                "CREATE TABLE audios (
                     audio_id TEXT PRIMARY KEY NOT NULL,
                     metadata TEXT NOT NULL,
                     duration_ms INTEGER
                 );
                 CREATE TABLE audio_sources (
                     audio_id TEXT PRIMARY KEY NOT NULL,
                     source BLOB NOT NULL
                 );
                 CREATE TABLE timelines (
                     audio_id TEXT PRIMARY KEY NOT NULL,
                     timeline BLOB NOT NULL
                 );",
            )
            .expect("create tables");
        connection
            .execute(
                "INSERT INTO audios(audio_id, metadata, duration_ms) VALUES (?1, ?2, ?3)",
                rusqlite::params!["call-1", metadata, 1_000i64],
            )
            .expect("insert audio");
        connection
            .execute(
                "INSERT INTO audio_sources(audio_id, source) VALUES (?1, ?2)",
                rusqlite::params!["call-1", source_blob],
            )
            .expect("insert source");
        connection
            .execute(
                "INSERT INTO timelines(audio_id, timeline) VALUES (?1, ?2)",
                rusqlite::params!["call-1", timeline_blob],
            )
            .expect("insert timeline");
    }

    #[test]
    fn reads_schema2_audiodb_sources_and_reference_transcript() {
        let path = temp_path("schema2");
        write_schema2_db(
            &path,
            &AudioSource::from_url("https://example.com/call.wav"),
            "你好世界",
        );

        let audios = read_vasr_audios(&path).expect("read schema 2");
        std::fs::remove_file(&path).ok();

        assert_eq!(audios.len(), 1);
        assert_eq!(audios[0].audio_id(), "call-1");
        assert_eq!(
            audios[0].source,
            AudioSource::from_url("https://example.com/call.wav")
        );
        let transcript = audios[0]
            .mono_timeline()
            .expect("mono timeline")
            .reference_transcript();
        assert_eq!(transcript.text, "你好世界");
    }

    #[test]
    fn read_audio_input_dtos_accepts_schema2_vasr() {
        let path = temp_path("schema2-dtos");
        write_schema2_db(
            &path,
            &AudioSource::from_encoded_bytes(b"fake-mp3".to_vec()),
            "转写文本",
        );

        let dtos = read_audio_input_dtos(&path, Some(true)).expect("read dtos");
        std::fs::remove_file(&path).ok();

        assert_eq!(dtos.len(), 1);
        assert!(dtos[0].url.is_none());
        assert_eq!(
            dtos[0].b64_str,
            Some(base64::engine::general_purpose::STANDARD.encode(b"fake-mp3"))
        );
        assert_eq!(dtos[0].mono, Some(true));
    }

    #[test]
    fn reads_repo_schema2_dataset_if_present() {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../datasets/lbg_400call-200.vasr");
        if !path.exists() {
            return;
        }
        let audios = read_vasr_audios(&path).expect("read repo schema 2 dataset");
        assert!(!audios.is_empty());
        assert!(matches!(audios[0].source, AudioSource::EncodedBytes(_)));
        let text = audios[0]
            .mono_timeline()
            .expect("mono timeline")
            .reference_transcript()
            .text;
        assert!(!text.trim().is_empty());
    }
}
