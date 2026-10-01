//! vASR model implementations and model capability traits.

pub mod error;
pub mod layers;
pub mod model_builder;
pub mod models;
pub mod profiling;
pub mod timeline;
pub mod traits;
pub mod utils;

pub mod audio;

pub use audio::{AudioInput, AudioLoader, SAMPLE_RATE_HZ, normalize_audio_input};
#[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
pub use model_builder::PagedWorkspaceStats;
pub use model_builder::{
    AsrQuantizationConfig, OfflineAsrBuildOutput, OfflineAsrBuildRequest, OfflineAsrBuildStats,
    OfflineAsrModelBuilder, build_offline_asr_model,
};
#[cfg(feature = "paged-attn")]
pub use model_builder::{AsrRuntimeConfig, KvCacheSizingStats};
pub use modelhub::{download, modelscope, modelscope_cache_dir, set_cache_dir};
pub use models::qwen3asr::QWEN3_ASR_DECODE_TOKENS_PER_SEC;
pub use models::qwen3asr::{ExecutionPolicy, Qwen3ASR, Qwen3Asr};
pub use models::qwen3asr::{qwen_outputs_to_timelines, qwen_transcribe_options};
pub use models::sensevoice::{SenseVoiceLoadOptions, SenseVoiceOnline};
pub use timeline::{
    ASR_SOURCE, StreamEvent, VAD_SOURCE, activity_span, extend_predictions, offset_spans,
    prediction_text, reference_text, timeline_from_spans, transcription_span,
};
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
