#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod flash;

#[cfg(feature = "cuda")]
pub use flash::{flash_attn, flash_attn_varlen};
