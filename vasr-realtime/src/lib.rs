// 测试代码里 `unwrap` / `expect` 是表达断言失败的常规方式；
// 工作区对非测试代码依然 deny 这两个 lint。
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod protocol;
mod runtime;
pub mod serve;
pub mod server;

pub use protocol::*;
pub use serve::{CommonModelArgs, RealtimeArgs, init_logging, run_realtime};
pub use server::{RealtimeService, RealtimeSession, realtime_router};
