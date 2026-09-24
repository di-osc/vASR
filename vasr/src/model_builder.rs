use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device};

use crate::traits::OfflineAsrModel;
use crate::utils::progress::Progress;

#[derive(Debug, Clone, Default)]
pub struct AsrQuantizationConfig {
    pub isq: Option<String>,
    #[cfg(feature = "cuda")]
    pub q8_prefill_lm_head_mmq: bool,
    #[cfg(feature = "cuda")]
    pub q8_prefill_lm_head_mmq_max_logits_bytes: usize,
}

#[cfg(feature = "paged-attn")]
#[derive(Debug, Clone)]
pub struct AsrRuntimeConfig {
    pub language: Option<String>,
    pub max_num_seqs: usize,
    pub max_segment_sec: f32,
    pub max_hotword_tokens: usize,
    pub kv_peak_ratio: f32,
    pub decode_tokens_per_sec: f64,
    pub decode_min_tokens: usize,
    /// 权重加载之后，KV 池占用总显存的比例。仅 CUDA 上的离线服务使用。
    pub gpu_memory_utilization: f32,
    #[cfg(feature = "cuda-graph")]
    pub cuda_graph_chunked_argmax: bool,
    #[cfg(feature = "cuda-graph")]
    pub cuda_graph_chunked_argmax_size: usize,
}

#[derive(Debug, Clone)]
pub struct OfflineAsrBuildRequest {
    pub model: String,
    pub cache_dir: PathBuf,
    pub device: Device,
    pub dtype: DType,
    pub progress: Progress,
    pub quantization: AsrQuantizationConfig,
    #[cfg(feature = "paged-attn")]
    pub runtime: AsrRuntimeConfig,
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    pub cuda_graph_max_batch: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct OfflineAsrBuildStats {
    pub model_kind: &'static str,
    pub build_seconds: f64,
    pub parameter_bytes: Option<usize>,
    pub quantize_seconds: Option<f64>,
    pub quantized_layer_counts: Option<(usize, usize, usize)>,
    pub kv_cache_bytes: Option<usize>,
    pub kv_cache_tokens: Option<usize>,
    #[cfg(feature = "paged-attn")]
    pub kv_cache_sizing: Option<KvCacheSizingStats>,
    pub cuda_graph_bytes: Option<usize>,
    pub cuda_graph_seqs: Option<usize>,
    pub cuda_graph_context_tokens: Option<usize>,
    /// paged-attn v2 workspace reserved while capturing CUDA decode graphs.
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    pub paged_workspace: Option<PagedWorkspaceStats>,
}

/// Compact paged-attn workspace figures for the ASR load log.
#[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
#[derive(Debug, Clone, Copy)]
pub struct PagedWorkspaceStats {
    pub bytes: usize,
    pub floor_bytes: usize,
    pub block_table_cols: usize,
    pub batches: usize,
}

#[cfg(feature = "paged-attn")]
#[derive(Debug, Clone, Default)]
pub struct KvCacheSizingStats {
    pub max_num_seqs: usize,
    pub base_segment_sec: f32,
    pub max_segment_sec: f32,
    pub peak_ratio: f32,
    pub max_hotword_tokens: usize,
    pub decode_tokens_per_sec: f64,
    pub decode_min_tokens: usize,
    pub planning_decode_min_tokens: usize,
    pub fixed_prompt_tokens: usize,
    pub base_audio_prompt_tokens: usize,
    pub base_decode_tokens: usize,
    pub base_tokens_per_seq: usize,
    pub base_blocks_per_seq: usize,
    pub peak_audio_prompt_tokens: usize,
    pub peak_decode_tokens: usize,
    pub peak_tokens_per_seq: usize,
    pub peak_blocks_per_seq: usize,
    pub base_blocks: usize,
    pub peak_extra_blocks: usize,
    pub total_blocks: usize,
    pub block_size: usize,
}

pub struct OfflineAsrBuildOutput {
    pub model: Arc<dyn OfflineAsrModel>,
    pub stats: OfflineAsrBuildStats,
}

pub trait OfflineAsrModelBuilder: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, model: &str) -> bool;
    fn build(&self, req: OfflineAsrBuildRequest) -> Result<OfflineAsrBuildOutput>;
}

pub fn build_offline_asr_model(req: OfflineAsrBuildRequest) -> Result<OfflineAsrBuildOutput> {
    let builders: [&dyn OfflineAsrModelBuilder; 1] =
        [&crate::models::qwen3asr::builder::Qwen3AsrBuilder];
    let builder = builders
        .iter()
        .copied()
        .find(|builder| builder.supports(&req.model))
        .unwrap_or(builders[0]);
    tracing::debug!(
        target: "vasr::model_builder",
        "selected ASR model builder `{}` for model `{}`",
        builder.name(),
        req.model
    );
    builder.build(req)
}
