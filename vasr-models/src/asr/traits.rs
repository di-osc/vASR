use anyhow::Result;
use vasr_data::{Annotation, AudioChunk, Timeline, Waveform};

#[derive(Debug, Clone)]
pub struct AsrOptions {
    pub language: Option<String>,
    pub context: String,
    pub max_new_tokens: usize,
    /// Maximum concurrently decoded sequences. `0` uses the recognizer default.
    pub decode_max_seqs: usize,
    /// Maximum total audio duration (seconds) per ASR pool. `0` disables duration grouping.
    pub max_batch_audio_sec: f32,
    pub chunk_max_sec: Option<f32>,
}

impl Default for AsrOptions {
    fn default() -> Self {
        Self {
            language: None,
            context: String::new(),
            max_new_tokens: 128,
            decode_max_seqs: 0,
            max_batch_audio_sec: 60.0,
            chunk_max_sec: None,
        }
    }
}

pub trait OfflineAsrModel: Send + Sync {
    fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline>;

    fn transcribe_batch(
        &self,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        waveforms
            .iter()
            .map(|waveform| self.transcribe(waveform, options))
            .collect()
    }
}

pub trait StreamingAsrModel: Send + Sync {
    fn new_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrSession>>;
}

pub trait StreamingAsrSession: Send {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>>;

    fn finish(&mut self) -> Result<Vec<Annotation>>;
}
