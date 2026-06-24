use std::fs;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioAsset, AudioEncoding,
    DurationMs, PersistedAudioFormat, TextSpan, TimeRange, Timeline, VasrRecord, VasrRecordList,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FasrConvertSummary {
    pub records: usize,
}

pub fn convert_fasr_audiolist_to_msgpack(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> Result<FasrConvertSummary> {
    let input = input.as_ref();
    let output = output.as_ref();
    let records = read_fasr_audio_list(input)
        .with_context(|| format!("failed to read FASR AudioList from {}", input.display()))?;
    let count = records.len();
    if count == 0 {
        bail!("FASR AudioList is empty: {}", input.display());
    }

    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }

    VasrRecordList::new(records)
        .with_metadata_value("source_format", serde_json::json!("fasr-audiolist"))
        .with_metadata_value(
            "source_path",
            serde_json::json!(input.display().to_string()),
        )
        .write_msgpack(output)
        .with_context(|| format!("failed to write vASR MessagePack to {}", output.display()))?;

    Ok(FasrConvertSummary { records: count })
}

pub fn read_fasr_audio_list(input: impl AsRef<Path>) -> Result<Vec<VasrRecord>> {
    let file = fs::File::open(input.as_ref())?;
    let mut reader = BufReader::new(file);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if !matches!(&magic, b"FASRAL01" | b"FASRAL02") {
        bail!(
            "unsupported FASR magic: {}",
            String::from_utf8_lossy(&magic)
        );
    }

    let header_len = read_u64_be(&mut reader)?;
    let header_bytes = read_exact_len(&mut reader, header_len, "FASR header")?;
    let header: Value = serde_json::from_slice(&header_bytes)?;
    if header.get("format").and_then(Value::as_str) != Some("fasr-audiolist") {
        bail!("not a FASR AudioList file");
    }
    let expected_records = header
        .get("audios")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);

    let mut records = Vec::with_capacity(expected_records);
    for index in 0..expected_records {
        let meta_len = read_u64_be(&mut reader)
            .with_context(|| format!("failed to read metadata length for record {index}"))?;
        let meta_bytes = read_exact_len(&mut reader, meta_len, "FASR record metadata")
            .with_context(|| format!("failed to read metadata for record {index}"))?;
        let meta: Value = serde_json::from_slice(&meta_bytes)
            .with_context(|| format!("failed to parse metadata JSON for record {index}"))?;

        let audio_len = read_u64_be(&mut reader)
            .with_context(|| format!("failed to read audio length for record {index}"))?;
        let audio_bytes = read_exact_len(&mut reader, audio_len, "FASR audio bytes")
            .with_context(|| format!("failed to read audio bytes for record {index}"))?;

        records.push(fasr_record_to_vasr(index, meta, audio_bytes)?);
    }

    Ok(records)
}

fn fasr_record_to_vasr(index: usize, meta: Value, audio_bytes: Vec<u8>) -> Result<VasrRecord> {
    let id = value_str(&meta, "id")
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| VasrRecord::id_for_index(index));
    let source = meta.get("source").unwrap_or(&Value::Null);
    let audio = meta.get("audio").unwrap_or(&Value::Null);
    let text = meta
        .pointer("/text/reference")
        .and_then(Value::as_str)
        .or_else(|| value_str(&meta, "text"))
        .unwrap_or("");

    let duration_ms = audio
        .get("duration_s")
        .and_then(Value::as_f64)
        .or_else(|| meta.get("duration").and_then(Value::as_f64))
        .map(|seconds| DurationMs((seconds * 1000.0).round().max(0.0) as u64));
    let sample_rate = audio
        .get("sample_rate")
        .and_then(Value::as_u64)
        .or_else(|| meta.get("sample_rate").and_then(Value::as_u64))
        .or_else(|| source.get("sample_rate").and_then(Value::as_u64))
        .and_then(|value| u32::try_from(value).ok());
    let channels = audio
        .get("channels")
        .and_then(Value::as_u64)
        .or_else(|| meta.get("channel_count").and_then(Value::as_u64))
        .and_then(|value| u16::try_from(value).ok());
    let encoding = source
        .get("format")
        .and_then(Value::as_str)
        .map(audio_encoding_from_fasr)
        .unwrap_or(AudioEncoding::Other("unknown".to_string()));

    let mut timeline = Timeline::new(id.clone());
    timeline.duration = duration_ms;
    if !text.trim().is_empty() {
        timeline.push(Annotation::new(
            TimeRange::new(DurationMs(0), duration_ms.unwrap_or_default()),
            AnnotationPayload::Transcription(TextSpan::new(text)),
            AnnotationSource::User,
            AnnotationStatus::Final,
        ));
    }

    let media = AudioAsset::Embedded {
        bytes: audio_bytes,
        format: PersistedAudioFormat {
            sample_rate,
            channels,
            encoding,
        },
        duration: duration_ms,
        sha256: source
            .get("sha256")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    };

    let mut record =
        VasrRecord::new(media, timeline).with_metadata_value("fasr_id", serde_json::json!(id));
    if let Some(url) = value_str(&meta, "url") {
        record = record.with_metadata_value("source_url", serde_json::json!(url));
    }
    for key in ["source", "storage", "extra", "channels"] {
        if let Some(value) = meta.get(key)
            && !value.is_null()
        {
            record = record.with_metadata_value(format!("fasr_{key}"), value.clone());
        }
    }
    Ok(record)
}

fn value_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn audio_encoding_from_fasr(format: &str) -> AudioEncoding {
    match format.to_ascii_lowercase().as_str() {
        "wav" | "wave" => AudioEncoding::Wav,
        "flac" => AudioEncoding::Flac,
        "mp3" | "mpeg" => AudioEncoding::Mp3,
        "ogg" | "opus" => AudioEncoding::Ogg,
        "pcm_s16le" | "s16le" | "pcm16" => AudioEncoding::PcmS16Le,
        other => AudioEncoding::Other(other.to_string()),
    }
}

fn read_u64_be(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_be_bytes(bytes))
}

fn read_exact_len(reader: &mut impl Read, len: u64, label: &str) -> Result<Vec<u8>> {
    let len = usize::try_from(len).with_context(|| format!("{label} length is too large"))?;
    let mut bytes = vec![0_u8; len];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}
