pub mod fsmn;
pub mod traits;

pub use fsmn::{FsmnVadDetection, FsmnVadModel, FsmnVadTiming};
pub use traits::{StreamingVadSession, VadModel, VadOptions, VadSegment};

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
