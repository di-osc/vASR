use anyhow::Result;
use asr_data::{Annotation, AudioChunk, Waveform};

pub use vasr::{AsrOptions, StreamingAsrModel, StreamingAsrSession};
pub use vasr_vad::{VadOptions, VadSegment};

pub use vasr::StreamingAsrModel as AsrModel;

pub trait VadModel: Send + Sync {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>>;

    fn new_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadSession>>;
}

pub trait StreamingVadSession: Send {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>>;

    fn finish(&mut self) -> Result<Vec<Annotation>>;
}

#[cfg(test)]
mod tests {
    use super::VadOptions;

    #[test]
    fn vad_defaults_match_vasr() {
        let opts = VadOptions::default();
        assert_eq!(opts.threshold, 0.6);
        assert_eq!(opts.min_silence_ms, 500);
        assert_eq!(opts.max_segment_ms, 30_000);
        assert_eq!(opts.pad_ms, 0);
        assert_eq!(opts.num_threads, 0);
    }
}
