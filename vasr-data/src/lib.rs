//! vASR 数据模型：音频文档、时间轴标注、评估指标和 SQLite 存储。
//!
//! 从 `asr-data` 0.1.2 迁入。不包含 Python 绑定，也不包含依赖 modelhub 的数据集下载。
#![allow(clippy::expect_used, clippy::unwrap_used)]

pub mod audio;
#[cfg(feature = "db")]
mod db;
mod doc;
#[cfg(all(feature = "db", feature = "metrics"))]
mod evaluation;
#[cfg(feature = "metrics")]
mod metrics;
mod timeline;
mod utils;

pub use audio::{AudioChannel, AudioEncoding, AudioFormat, AudioInfo, AudioSource};
pub use audio::{AudioChunk, AudioError, StreamingResampler, Waveform};
#[cfg(feature = "db")]
pub use db::{
    AudioDb, AudioDbError, AudioDbInfo, AudioDbMode, AudioQuery, DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT, read_audio_db_info,
};
pub use doc::{Audio, AudioChannelError, AudioStream, AudioTimelineError, AudioValidationError};
#[cfg(all(feature = "db", feature = "metrics"))]
pub use evaluation::{
    DatasetActivityEvaluation, DatasetActivityEventEvaluation, DatasetEvalError, DatasetEvaluation,
    DatasetEvaluator, DatasetSpeakerEvaluation, DatasetTranscriptionEvaluation, evaluate_dataset,
};
#[cfg(feature = "metrics")]
pub use metrics::{
    CerStats, ChineseTextNormalizationOptions, TextNormalizationError, compute_cer,
    normalize_for_cer, normalize_zh, normalize_zh_with_options,
};
#[cfg(feature = "metrics")]
pub use timeline::{
    ActivityEvaluation, ActivityEventEvaluation, TimelineEvalConfig, TimelineEvalError,
    TimelineEvaluation, TranscriptionEvaluation, TranscriptionNormalization,
};
pub use timeline::{
    Annotation, AudioActivity, AudioId, LanguageTag, Sentence, Speaker, SpeakerId, TimeSpan,
    TimeSpanConflictKind, TimeSpanId, TimeSpanOverlap, Timeline, TimelineId, TimelineSpanError,
    Token, Transcript, Transcription,
};
pub use utils::TimeRange;
