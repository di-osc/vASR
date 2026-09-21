//! Voice activity detection models and traits.

pub mod firered;
pub mod fsmn;
pub mod traits;

pub use firered::{FireRedVadDetection, FireRedVadModel, FireRedVadTiming};
pub use fsmn::{FsmnVadDetection, FsmnVadForwardTiming, FsmnVadModel, FsmnVadTiming};
pub use traits::{StreamingVadSession, VadEvent, VadEventStatus, VadModel, VadOptions, VadSegment};

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
