//! vASR model implementations and model capability traits.

pub mod error;
pub mod layers;
pub mod models;
pub mod traits;

pub use vasr_data::audio;

pub use models::qwen3asr::Qwen3Asr;
pub use models::qwen3asr::Qwen3Asr as Qwen3ASR;
pub use models::qwen3asr::{qwen_outputs_to_timelines, qwen_transcribe_options};
pub use traits::{AsrOptions, OfflineAsrModel, StreamingAsrModel, StreamingAsrSession};
pub use vasr_hub::{download, modelscope, modelscope_cache_dir};
#[cfg(feature = "paged-attn")]
pub use vasr_paged_attn;
pub use vasr_quant;
pub use vasr_vad as vad;

// Compatibility aliases while downstream crates migrate to `vasr::models::qwen3asr`.
pub use models::qwen3asr::{capabilities, config, inference, model, processor};

pub mod qwen3_asr {
    pub use crate::models::qwen3asr::*;
}
