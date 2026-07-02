pub use vasr::{AsrOptions, StreamingAsrModel, StreamingAsrSession};
pub use vasr_vad::{StreamingVadSession, VadModel, VadOptions, VadSegment};

pub use vasr::StreamingAsrModel as AsrModel;

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
    }
}
