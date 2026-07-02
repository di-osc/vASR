use anyhow::Result;
use vasr_data::{Annotation, AudioChunk, TimeRange, Waveform};

#[derive(Debug, Clone)]
pub struct VadOptions {
    /// `speech_noise_thres` in funasr_onnx / fasr.
    pub threshold: f32,
    /// Kept for API compatibility; funasr E2E VAD uses window thresholds instead.
    pub min_speech_ms: u64,
    /// `max_end_silence_time` in funasr_onnx / fasr (default 500).
    pub min_silence_ms: u64,
    /// Maximum offline VAD segment duration before ASR slicing.
    pub max_segment_ms: u64,
    /// Symmetric padding added to detected speech segments before ASR.
    pub pad_ms: u64,
    /// CPU worker threads for VAD internals; 0 uses Rayon default.
    pub num_threads: usize,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            min_speech_ms: 250,
            min_silence_ms: 500,
            max_segment_ms: 30_000,
            pad_ms: 0,
            num_threads: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VadSegment {
    pub range: TimeRange,
    pub probability: f32,
}

pub trait VadModel: Send + Sync {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>>;

    fn new_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadSession>>;
}

pub trait StreamingVadSession: Send {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>>;

    fn finish(&mut self) -> Result<Vec<Annotation>>;
}

pub(crate) fn run_with_vad_threads<T>(num_threads: usize, f: impl FnOnce() -> T + Send) -> Result<T>
where
    T: Send,
{
    if num_threads == 0 {
        return Ok(f());
    }
    Ok(rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()?
        .install(f))
}
