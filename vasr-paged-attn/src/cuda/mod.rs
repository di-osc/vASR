pub const USE_FP8: bool = cfg!(has_fp8);

mod backend;
mod ffi;

pub use backend::{
    gather_kv_cache, paged_attention, reserve_paged_attention_v2_workspace, reshape_and_cache,
};
