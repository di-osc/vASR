//! PagedAttention KV cache, block manager, and CUDA decode graph helpers.

#[cfg(all(feature = "cuda-paged-attn", target_family = "unix"))]
mod cuda;
#[cfg(feature = "cuda-graph")]
pub mod cuda_graph;
pub mod decode_forward;
pub mod flash;
#[cfg(feature = "metal-paged-attn")]
pub mod metal;
pub mod paged_cache_runtime;
pub mod paged_kv_cache;

pub use decode_forward::PagedCudaDecodeForward;
pub use flash::{FlashKMeta, FlashParams};
pub use paged_cache_runtime::{
    ModelExecutionLock, PagedBlockManager, PagedCacheConfig, PagedCacheMemory, PagedCacheRuntime,
    PagedCacheStats, SharedPagedCacheRuntime, bytes_per_paged_block,
};
pub use paged_kv_cache::{PAD_SLOT_ID, PagedInputMetadata, PagedKvCache};

#[cfg(feature = "cuda-graph")]
pub use cuda_graph::{
    CUDA_DECODE_GRAPH_CONTEXT_TOKENS, CUDA_DECODE_GRAPH_MAX_BATCH, DecodeGraphOutputKind,
};

#[cfg(all(feature = "cuda-paged-attn", target_family = "unix"))]
pub use cuda::{gather_kv_cache, paged_attention, reshape_and_cache};
#[cfg(all(
    feature = "metal-paged-attn",
    not(all(feature = "cuda-paged-attn", target_family = "unix"))
))]
pub use metal::paged::{gather_kv_cache, paged_attention, reshape_and_cache};

#[cfg(all(feature = "cuda-paged-attn", target_family = "unix"))]
pub fn reserve_paged_attention_v2_workspace(
    device: &candle_core::Device,
    bytes: usize,
) -> candle_core::Result<()> {
    cuda::reserve_paged_attention_v2_workspace(device, bytes)
}

#[cfg(not(all(feature = "cuda-paged-attn", target_family = "unix")))]
pub fn reserve_paged_attention_v2_workspace(
    _device: &candle_core::Device,
    _bytes: usize,
) -> candle_core::Result<()> {
    Ok(())
}

#[cfg(feature = "metal-paged-attn")]
pub use metal::{
    dense_attention_varlen_prefill, pack_kv_for_varlen_prefill, pack_query_for_varlen_prefill,
    paged_attention_varlen_prefill, supports_metal_varlen_prefill, unpack_varlen_attn_to_batched,
};
