#![allow(dead_code, unused_imports)]

pub mod device;
pub mod model;
pub mod models;
pub mod pipeline;
pub mod scheduler;

pub use device::{auto_device, auto_dtype, device_label, resolve_device};
pub use model::{
    AsrModel, AsrOptions, ParallelTranscribeOptions, StreamingAsrSession, StreamingVadSession,
    VadModel, VadOptions, VadSegment,
};
pub use models::{qwen3_asr_adapter::Qwen3AsrAdapter, vad::FsmnVadModel};
pub use pipeline::OfflinePipeline;
pub use scheduler::{InferenceScheduler, ScheduledAsrModel};
