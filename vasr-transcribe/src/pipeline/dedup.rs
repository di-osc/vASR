use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use asr_data::AudioSource;

use super::{TranscribeInput, TranscribeItemOutcome};

#[derive(Debug)]
pub(super) struct DeduplicatedTranscribeInputs {
    pub(super) unique_inputs: Vec<TranscribeInput>,
    pub(super) duplicate_indices: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TranscribeInputDedupKey {
    source: AudioSourceDedupKey,
    mono: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AudioSourceDedupKey {
    Path(PathBuf),
    Url(String),
    Base64(String),
    EncodedBytes(Vec<u8>),
    LocalFileContent { len: u64, hash: u64 },
}

pub(super) fn deduplicate_transcribe_inputs(
    inputs: Vec<TranscribeInput>,
) -> DeduplicatedTranscribeInputs {
    let mut seen = HashMap::with_capacity(inputs.len());
    let mut unique_inputs = Vec::with_capacity(inputs.len());
    let mut duplicate_indices = Vec::new();

    for input in inputs {
        if let Some(source_key) = audio_source_dedup_key(&input.source) {
            let key = TranscribeInputDedupKey {
                source: source_key,
                mono: input.mono,
            };
            if let Some(&canonical_index) = seen.get(&key) {
                duplicate_indices.push((input.index, canonical_index));
                continue;
            }
            seen.insert(key, input.index);
        }
        unique_inputs.push(input);
    }

    DeduplicatedTranscribeInputs {
        unique_inputs,
        duplicate_indices,
    }
}

fn audio_source_dedup_key(source: &AudioSource) -> Option<AudioSourceDedupKey> {
    match source {
        AudioSource::Path(path) => Some(
            local_file_content_key(path).unwrap_or_else(|| AudioSourceDedupKey::Path(path.clone())),
        ),
        AudioSource::Url(url) => {
            if let Some(path) = local_path_from_url(url) {
                return Some(
                    local_file_content_key(&path)
                        .unwrap_or_else(|| AudioSourceDedupKey::Url(url.clone())),
                );
            }
            Some(AudioSourceDedupKey::Url(url.clone()))
        }
        AudioSource::Base64(encoded) => Some(AudioSourceDedupKey::Base64(encoded.clone())),
        AudioSource::EncodedBytes(bytes) => Some(AudioSourceDedupKey::EncodedBytes(bytes.clone())),
        AudioSource::PcmS16Le {
            bytes,
            sample_rate,
            channels,
        } => {
            let mut key = sample_rate.to_le_bytes().to_vec();
            key.extend_from_slice(&channels.to_le_bytes());
            key.extend_from_slice(bytes);
            Some(AudioSourceDedupKey::EncodedBytes(key))
        }
    }
}

fn local_file_content_key(path: &Path) -> Option<AudioSourceDedupKey> {
    let bytes = fs::read(path).ok()?;
    Some(bytes_content_key(&bytes))
}

fn bytes_content_key(bytes: &[u8]) -> AudioSourceDedupKey {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    AudioSourceDedupKey::LocalFileContent {
        len: bytes.len() as u64,
        hash: hasher.finish(),
    }
}

fn local_path_from_url(value: &str) -> Option<PathBuf> {
    value
        .strip_prefix("file://")
        .map(|rest| PathBuf::from(percent_decode_path(rest)))
}

fn percent_decode_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2]))
        {
            out.push((hi << 4) | lo);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub(super) fn expand_duplicate_outcomes(
    mut outcomes: Vec<TranscribeItemOutcome>,
    duplicate_indices: &[(usize, usize)],
) -> Vec<TranscribeItemOutcome> {
    if duplicate_indices.is_empty() {
        outcomes.sort_unstable_by_key(|outcome| outcome.index);
        return outcomes;
    }

    let duplicate_outcomes = duplicate_indices
        .iter()
        .flat_map(|&(duplicate_index, canonical_index)| {
            outcomes
                .iter()
                .filter(move |outcome| outcome.index == canonical_index)
                .map(move |outcome| duplicate_outcome_from(outcome, duplicate_index))
        })
        .collect::<Vec<_>>();
    outcomes.extend(duplicate_outcomes);
    outcomes.sort_unstable_by_key(|outcome| outcome.index);
    outcomes
}

fn duplicate_outcome_from(
    outcome: &TranscribeItemOutcome,
    duplicate_index: usize,
) -> TranscribeItemOutcome {
    let result = match &outcome.result {
        Ok(timeline) => Ok(timeline.clone()),
        Err(error) => Err(anyhow::anyhow!("{}", error)),
    };
    TranscribeItemOutcome {
        index: duplicate_index,
        channel: outcome.channel,
        result,
        audio_seconds: outcome.audio_seconds,
        timing: outcome.timing,
        spans: outcome.spans,
        bad_component: outcome.bad_component,
    }
}
