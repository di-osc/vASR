//! High-level inference API: transcribe, streaming, output parsing.

#[cfg(feature = "paged-attn")]
pub mod batch_scheduler;
pub mod execution;
pub mod speech_rate;
pub mod streaming;
pub mod transcribe;
pub mod types;
pub mod utils;
