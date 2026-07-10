use std::any::Any;

use anyhow::Result;
use vasr_data::{Annotation, AudioChunk, Timeline, Waveform};

#[derive(Debug, Clone)]
pub struct AsrOptions {
    pub language: Option<String>,
    pub context: String,
    /// Maximum concurrently decoded sequences. `0` uses the recognizer default.
    pub decode_max_seqs: usize,
    /// Maximum total audio duration (seconds) per ASR pool. `0` disables duration grouping.
    pub max_batch_audio_sec: f32,
    pub chunk_max_sec: Option<f32>,
    pub decode_tokens_per_sec: f64,
    pub decode_min_tokens: usize,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            language: None,
            context: String::new(),
            decode_max_seqs: 0,
            max_batch_audio_sec: 60.0,
            chunk_max_sec: None,
            decode_tokens_per_sec: 4.25,
            decode_min_tokens: 128,
        }
    }
}

pub trait OfflineAsrModel: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn transcribe(&self, waveforms: &[Waveform], options: &AsrOptions) -> Result<Vec<Timeline>>;
}

pub trait StreamingAsrModel: Send + Sync {
    fn new_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrSession>>;
}

pub trait StreamingAsrSession: Send {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>>;

    fn finish(&mut self) -> Result<Vec<Annotation>>;
}
