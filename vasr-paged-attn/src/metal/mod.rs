mod compile;
pub mod kernels;
pub(crate) mod paged;
pub mod varlen_prefill;

pub use paged::pipeline_cache_stats;

pub use varlen_prefill::{
    dense_attention_varlen_prefill, pack_kv_for_varlen_prefill, pack_query_for_varlen_prefill,
    paged_attention_varlen_prefill, supports_metal_varlen_cache_dtype,
    supports_metal_varlen_prefill, unpack_varlen_attn_to_batched,
};

pub mod conv;
pub mod pointwise;
pub mod residual_norm;
pub mod rotary;

pub mod profile;
