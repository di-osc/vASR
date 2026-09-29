// 测试代码里 `unwrap` / `expect` 是表达断言失败的常规方式；
// 工作区对非测试代码依然 deny 这两个 lint。
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

mod dataset;
mod device;
pub mod eval;
mod legacy_audiodb;
pub mod perf;
pub mod pipeline;
pub mod protocol;
mod quality;
pub mod run;
mod scheduler;
pub mod serve;
mod serve_tuning;
pub mod server;
pub mod stress;
mod table;

pub use eval::{EvalArgs, run_eval};
pub use perf::{PerfArgs, run_perf};
pub use pipeline::{
    AsyncTranscribePipeline, GlobalAsyncTranscribePipeline, TranscribeInput, TranscribeItemOutcome,
};
pub use protocol::*;
pub use run::{RunTranscribeArgs, run_local};
pub use serve::{
    CommonModelArgs, ServeTranscribeArgs, TranscribeArgs, TranscribePipelineArgs, VadCliArgs,
    build_async_transcribe_pipeline, init_logging, run_transcribe, validate_pipeline,
};
pub use server::{
    TranscribeService, build_transcribe_service, build_transcribe_service_from_parts,
    transcribe_router,
};
pub use stress::{StressArgs, run_stress};
