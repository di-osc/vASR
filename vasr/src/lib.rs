//! vASR model implementations and model capability traits.

pub mod error;
pub mod layers;
pub mod model_builder;
pub mod models;
pub mod traits;
pub mod utils;

pub use asr_data::audio;

pub use model_builder::{
    AsrQuantizationConfig, OfflineAsrBuildOutput, OfflineAsrBuildRequest, OfflineAsrBuildStats,
    OfflineAsrModelBuilder, build_offline_asr_model,
};
#[cfg(feature = "paged-attn")]
pub use model_builder::{AsrRuntimeConfig, KvCacheSizingStats};
pub use modelhub::{download, modelscope, modelscope_cache_dir, set_cache_dir};
pub use models::qwen3asr::{Qwen3ASR, Qwen3Asr};
pub use models::qwen3asr::{qwen_outputs_to_timelines, qwen_transcribe_options};
pub use traits::{AsrOptions, OfflineAsrModel, StreamingAsrModel, StreamingAsrSession};
#[cfg(feature = "paged-attn")]
pub use vasr_paged_attn;
pub use vasr_quant;
pub use vasr_vad as vad;

// Compatibility aliases while downstream crates migrate to `vasr::models::qwen3asr`.
pub use models::qwen3asr::{capabilities, config, inference, model, processor};

pub mod qwen3_asr {
    pub use crate::models::qwen3asr::*;
}
