//! vASR model implementations and model capability traits.

pub mod aligner;
pub mod asr;
pub mod download;
pub mod error;
pub mod vad;

pub use vasr_data::audio;

/// Directory where ModelScope models are cached.
///
/// Respects the `VASR_MODEL_DIR` environment variable. When set, it is used
/// directly (no further path composition). Otherwise defaults to
/// `$HOME/.cache/vasr` (or `/tmp/.cache/vasr` when `$HOME` is unset).
pub fn modelscope_cache_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("VASR_MODEL_DIR") {
        return std::path::PathBuf::from(dir);
    }
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
        .join(".cache")
        .join("vasr")
}

pub use aligner::Qwen3Aligner;
pub use asr::{
    AsrOptions, OfflineAsrModel, StreamingAsrModel, StreamingAsrSession,
    qwen3asr::{qwen_outputs_to_timelines, qwen_transcribe_options},
};
pub use asr::{Qwen3ASR, Qwen3Asr};
#[cfg(feature = "paged-attn")]
pub use vasr_paged_attn;
pub use vasr_quant;

// Compatibility aliases while downstream crates migrate to `vasr_models::asr::qwen3asr`.
pub use asr::qwen3asr::{capabilities, config, inference, model, processor};

pub mod qwen3_asr {
    pub use crate::asr::qwen3asr::*;
}
