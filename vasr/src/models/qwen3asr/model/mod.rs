//! Candle model implementation (audio tower + thinker LM + generation).

use std::path::PathBuf;

pub mod attention;
pub mod audio_encoder;
pub(crate) mod cuda_mem_debug;
pub mod generation;
pub mod kv_cache;
pub mod name_map;
#[cfg(feature = "paged-attn")]
pub mod paged_batch_engine;
pub mod thinker;
pub mod thinker_text;
pub mod weights;

pub use vasr_quant::Q8FastPathStats;
pub use vasr_quant::isq_linear;

pub fn reset_q8_fast_path_stats() {
    vasr_quant::reset_q8_fast_path_stats();
}

pub fn q8_fast_path_stats() -> Q8FastPathStats {
    vasr_quant::q8_fast_path_stats()
}

pub fn set_q8_prefill_lm_head_mmq_enabled(enabled: bool) {
    vasr_quant::set_q8_prefill_lm_head_mmq_enabled(enabled);
}

pub fn set_q8_prefill_lm_head_mmq_max_logits_bytes(bytes: usize) {
    vasr_quant::set_q8_prefill_lm_head_mmq_max_logits_bytes(bytes);
}

#[cfg(feature = "paged-attn")]
pub mod paged_kv_cache {
    pub use vasr_paged_attn::paged_kv_cache::*;
}

#[cfg(feature = "paged-attn")]
pub mod paged_cache_runtime {
    pub use vasr_paged_attn::paged_cache_runtime::*;
}

#[derive(Debug)]
pub(crate) struct LoadedQwen3Asr {
    pub model_dir: PathBuf,
    pub weights_paths: Vec<PathBuf>,
    pub thinker: thinker::ThinkerForConditionalGeneration,
}
