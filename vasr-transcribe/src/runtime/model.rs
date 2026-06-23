pub use vasr::{AsrOptions, OfflineAsrModel as AsrModel, StreamingAsrSession};
pub use vasr_vad::{StreamingVadSession, VadModel, VadOptions, VadSegment};

#[derive(Debug, Clone)]
pub struct ParallelTranscribeOptions {
    pub asr_options: AsrOptions,
    pub vad_options: VadOptions,
}

impl Default for ParallelTranscribeOptions {
    fn default() -> Self {
        Self {
            asr_options: AsrOptions::default(),
            vad_options: VadOptions::default(),
        }
    }
}

impl From<AsrOptions> for ParallelTranscribeOptions {
    fn from(asr_options: AsrOptions) -> Self {
        Self {
            asr_options,
            ..Self::default()
        }
    }
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
    }
}
