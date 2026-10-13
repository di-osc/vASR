//! `-i` 输入收集：把单个输入字符串（音频文件 / 目录 / URL / vASR 数据文件）
//! 展开成音频来源，并在需要时保留 audiodb 记录里的参考文本和 metadata。

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use asr_data::{
    Audio, AudioDb, AudioDbError, AudioDbMode, AudioQuery, AudioSource, MAX_QUERY_LIMIT,
};
use base64::Engine;

use crate::legacy_audiodb::read_legacy_audiodb;
use crate::{AudioInputDto, ModelScopeSourceDto};

/// 判断是否是 vASR 数据文件（`.db` / `.vasr` / `.msgpack`）。
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
        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "db" | "vasr" | "msgpack"))
        .unwrap_or(false)
}

/// 读取 vASR 数据文件里的全部音频文档。
///
/// # Errors
///
/// 文件打不开、schema 不支持或解码失败时返回错误。
pub fn read_vasr_audios(input: &Path) -> Result<Vec<Audio>> {
    let db_path = input;
    match AudioDb::open(db_path, AudioDbMode::ReadOnly) {
        Ok(db) => db
            .query(&AudioQuery {
                limit: MAX_QUERY_LIMIT,
                ..AudioQuery::default()
            })
            .with_context(|| format!("failed to read vASR AudioDb from {}", db_path.display())),
        Err(AudioDbError::UnsupportedSchema {
            found: version @ (1 | 2),
            expected: _,
        }) => read_legacy_audiodb(db_path, version)
            .with_context(|| format!("failed to read vASR AudioDb from {}", db_path.display())),
        Err(error) => Err(error)
            .with_context(|| format!("failed to read vASR AudioDb from {}", db_path.display())),
    }
}

pub fn audio_input_from_source(
    source: &AudioSource,
    mono: Option<bool>,
    hotword: Option<String>,
) -> AudioInputDto {
    match source {
        AudioSource::Path(path) => AudioInputDto {
            id: None,
            info: None,
            url: Some(path.display().to_string()),
            b64_str: None,
            modelscope: None,
            mono,
            hotword,
        },
        AudioSource::Url(url) => AudioInputDto {
            id: None,
            info: None,
            url: Some(url.clone()),
            b64_str: None,
            modelscope: None,
            mono,
            hotword,
        },
        AudioSource::Base64(data) => AudioInputDto {
            id: None,
            info: None,
            url: None,
            b64_str: Some(data.clone()),
            modelscope: None,
            mono,
            hotword,
        },
        AudioSource::EncodedBytes(bytes) | AudioSource::PcmS16Le { bytes, .. } => AudioInputDto {
            id: None,
            info: None,
            url: None,
            b64_str: Some(base64::engine::general_purpose::STANDARD.encode(bytes)),
            modelscope: None,
            mono,
            hotword,
        },
        AudioSource::ModelScope {
            repo_id,
            file_path,
            revision,
        } => AudioInputDto {
            id: None,
            info: None,
            url: None,
            b64_str: None,
            modelscope: Some(ModelScopeSourceDto {
                repo_id: repo_id.clone(),
                file_path: file_path.clone(),
                revision: Some(revision.clone()),
            }),
            mono,
            hotword,
        },
    }
}

pub fn audio_input_from_audio(audio: &Audio, mono: Option<bool>) -> AudioInputDto {
    let mut input = audio_input_from_source(&audio.source, mono, hotword_from_record(audio));
    input.id = Some(audio.id.clone());
    input.info = Some(audio.info.clone());
    input
}

/// 从 Audio 记录 metadata 里读取热词。
pub fn hotword_from_record(record: &Audio) -> Option<String> {
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

/// 单个 `-i` 输入展开出的一条音频来源。
///
/// `record` 只在输入是 vASR 数据文件（audiodb）时存在，用来保留参考文本、metadata
/// 和 probe 信息；目录、单个音频文件和 URL 都只有 `source`。
#[derive(Debug, Clone)]
pub struct CollectedInput {
    pub source: AudioSource,
    pub record: Option<Audio>,
}

/// 判断 `-i` 字符串是否是远程 URL。
pub fn is_url_input(value: &str) -> bool {
    value.contains("://")
}

/// 音频文件扩展名。
pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aac" | "opus" | "webm"
            )
        })
        .unwrap_or(false)
}

/// 把单个 `-i` 输入展开成来源列表。
///
/// 支持四种输入：
///
/// - 远程 URL（含 `://`）→ 一条 [`AudioSource::Url`]
/// - vASR 数据文件（`.db` / `.vasr` / `.vasr.msgpack`）→ 每条记录一条来源
/// - 目录 → 目录下的音频文件，每条一个 [`AudioSource::Path`]
/// - 单个音频文件 → 一条 [`AudioSource::Path`]
///
/// # Errors
///
/// 路径不存在、扩展名不是音频、数据文件读取失败或目录里没有音频文件时返回错误。
pub fn collect_inputs(input: &str, recursive: bool) -> Result<Vec<CollectedInput>> {
    if is_url_input(input) {
        return Ok(vec![CollectedInput {
            source: AudioSource::Url(input.to_string()),
            record: None,
        }]);
    }

    let path = Path::new(input);
    if is_vasr_data_file(path) {
        let audios = read_vasr_audios(path)?;
        if audios.is_empty() {
            bail!("vASR record list is empty: {}", path.display());
        }
        return Ok(audios
            .into_iter()
            .map(|audio| CollectedInput {
                source: audio.source.clone(),
                record: Some(audio),
            })
            .collect());
    }

    if path.is_dir() {
        let mut files = Vec::new();
        collect_audio_files_in_dir(path, recursive, &mut files)?;
        files.sort();
        if files.is_empty() {
            bail!("no audio files found under {}", path.display());
        }
        return Ok(files
            .into_iter()
            .map(|file| CollectedInput {
                source: AudioSource::Path(file),
                record: None,
            })
            .collect());
    }

    if path.is_file() {
        if !is_audio_file(path) {
            bail!("unsupported audio file extension: {}", path.display());
        }
        return Ok(vec![CollectedInput {
            source: AudioSource::Path(path.to_path_buf()),
            record: None,
        }]);
    }

    bail!("input path does not exist: {}", path.display())
}

fn collect_audio_files_in_dir(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("failed to read directory {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                collect_audio_files_in_dir(&path, true, out)?;
            }
            continue;
        }
        if is_audio_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

/// 把一条收集到的输入转成 HTTP 请求项。
pub fn audio_input_from_collected(
    input: &CollectedInput,
    mono: Option<bool>,
    index: usize,
) -> AudioInputDto {
    match &input.record {
        Some(record) => audio_input_from_audio(record, mono),
        None => {
            let mut dto = audio_input_from_source(&input.source, mono, None);
            dto.id = Some(Audio::id_for_index(index));
            dto
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sqlite_database_file_is_a_dataset_input() {
        let path = std::path::Path::new("datasets/example1.db");
        assert!(super::is_vasr_data_file(path));
    }
    use super::{audio_input_from_collected, collect_inputs, read_vasr_audios};
    use asr_data::{AudioSource, TimeRange};
    use base64::Engine;
    use rusqlite::Connection;
    use serde::Serialize;
    use std::path::PathBuf;

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
    fn collect_inputs_accepts_schema2_vasr() {
        let path = temp_path("schema2-inputs");
        write_schema2_db(
            &path,
            &AudioSource::from_encoded_bytes(b"fake-mp3".to_vec()),
            "转写文本",
        );

        let collected = collect_inputs(&path.to_string_lossy(), false).expect("collect inputs");
        let dtos = collected
            .iter()
            .enumerate()
            .map(|(index, input)| audio_input_from_collected(input, Some(true), index))
            .collect::<Vec<_>>();
        std::fs::remove_file(&path).ok();

        assert_eq!(collected.len(), 1);
        assert!(collected[0].record.is_some());
        assert!(dtos[0].url.is_none());
        assert_eq!(
            dtos[0].b64_str,
            Some(base64::engine::general_purpose::STANDARD.encode(b"fake-mp3"))
        );
        assert_eq!(dtos[0].mono, Some(true));
    }

    #[test]
    fn collect_inputs_treats_http_url_as_single_source() {
        let collected =
            collect_inputs("https://example.com/a.wav", false).expect("collect url input");
        assert_eq!(collected.len(), 1);
        assert!(collected[0].record.is_none());
        assert_eq!(
            collected[0].source,
            AudioSource::from_url("https://example.com/a.wav")
        );
    }

    #[test]
    fn collect_inputs_expands_directory_audio_files() {
        let dir = std::env::temp_dir().join(format!(
            "vasr-dataset-dir-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        std::fs::write(dir.join("b.wav"), b"RIFF").expect("write b");
        std::fs::write(dir.join("a.mp3"), b"id3").expect("write a");
        std::fs::write(dir.join("notes.txt"), b"ignore").expect("write txt");

        let collected = collect_inputs(&dir.to_string_lossy(), false).expect("collect dir");
        let names = collected
            .iter()
            .map(|input| match &input.source {
                AudioSource::Path(path) => path.file_name().unwrap().to_string_lossy().to_string(),
                other => panic!("expected path source, got {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["a.mp3", "b.wav"]);
        assert!(collected.iter().all(|input| input.record.is_none()));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn collect_inputs_rejects_missing_path() {
        let missing = std::env::temp_dir().join(format!(
            "vasr-dataset-missing-{}-{}.wav",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let error = collect_inputs(&missing.to_string_lossy(), false).expect_err("missing path");
        assert!(error.to_string().contains("does not exist"));
    }

    #[test]
    fn reads_repo_call_100_dataset_if_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../datasets/call-100.db");
        if !path.exists() {
            return;
        }
        let audios = read_vasr_audios(&path).expect("read repo call-100 dataset");
        assert!(!audios.is_empty());
        // 数据集音频来自 ModelScope 仓库，库里只存仓库身份。
        assert!(matches!(audios[0].source, AudioSource::ModelScope { .. }));
        let text = audios[0]
            .mono_timeline()
            .expect("mono timeline")
            .reference_transcript()
            .text;
        assert!(!text.trim().is_empty());
    }
}
