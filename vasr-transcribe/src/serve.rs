use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::device::{auto_dtype, device_label, resolve_device};
use crate::scheduler::{InferenceScheduler, ScheduledAsrModel};
use anyhow::{Result, bail};
use axum::{Json, Router, routing::get};
use candle_core::{DType, Device};
use clap::{ArgAction, Args};
use indicatif::{ProgressBar, ProgressStyle};
use vasr::AudioLoader;
use vasr::utils::progress::{Progress, ProgressEvent, ProgressHandle, ProgressSink, ProgressTask};
use vasr::{AsrOptions, OfflineAsrModel as AsrModel};
use vasr_vad::{FireRedVadModel, FsmnVadModel, VadModel, VadOptions};

use crate::pipeline::{
    AsyncTranscribePipeline, DEFAULT_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS,
    DEFAULT_ASR_ENCODER_BATCH_MAX_SEGMENTS, GlobalAsyncTranscribePipeline, PipelineWorkerOptions,
};
use crate::serve_tuning::{ResolvedServeTuning, ServeTuningOverrides, resolve_tuning};
use crate::server::build_transcribe_service;

const DEFAULT_MAX_NUM_SEQS: usize = 512;
const DEFAULT_ASR_MODEL: &str = "Qwen/Qwen3-ASR-0.6B";
const DEFAULT_MAX_HOTWORD_TOKENS: usize = 128;
const DEFAULT_KV_PEAK_RATIO: f32 = 0.25;
const DEFAULT_GPU_MEMORY_UTILIZATION: f32 = 0.9;
const DEFAULT_PROFILE_WINDOW_SECS: f64 = 5.0;
#[cfg(feature = "cuda-graph")]
const DEFAULT_CUDA_GRAPH_CHUNKED_ARGMAX_SIZE: usize = 4096;
#[cfg(feature = "cuda")]
const DEFAULT_Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_MIB: usize = 256;

struct IndicatifProgressSink;

impl ProgressSink for IndicatifProgressSink {
    fn start(&self, event: ProgressEvent) -> ProgressHandle {
        let total = event.total.unwrap_or(0);
        if total == 0 {
            return ProgressHandle::disabled();
        }
        let progress = ProgressBar::new(total);
        let style = ProgressStyle::with_template(
            "{msg}: [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("##-");
        progress.set_style(style);
        progress.set_message(progress_message(&event));
        ProgressHandle::new(Box::new(IndicatifProgressTask { progress }))
    }
}

struct IndicatifProgressTask {
    progress: ProgressBar,
}

impl ProgressTask for IndicatifProgressTask {
    fn inc(&self, n: u64) {
        self.progress.inc(n);
    }

    fn finish(&self) {
        self.progress.finish_and_clear();
    }
}

fn progress_message(event: &ProgressEvent) -> String {
    match (event.model, event.phase) {
        ("qwen3-asr", "load.audio_layers") => "Loading Qwen3-ASR audio layers".to_string(),
        ("qwen3-asr", "load.text_layers") => "Loading Qwen3-ASR repeating layers".to_string(),
        ("qwen3-asr", "quantize.audio_layers") => {
            "Loading Qwen3-ASR audio layers for quantization".to_string()
        }
        ("qwen3-asr", "quantize.text_layers") => {
            "Loading Qwen3-ASR repeating layers for quantization".to_string()
        }
        _ => format!("{} {}", event.model, event.phase),
    }
}
#[derive(Debug, Clone, Args)]
pub struct CommonModelArgs {
    /// Qwen3-ASR model: `Qwen3-ASR-0.6B` / `Qwen3-ASR-1.7B` auto-map to their repo ids, or pass a repo id / local directory directly.
    #[arg(long, default_value = DEFAULT_ASR_MODEL, env = "VASR_MODEL")]
    pub model: String,

    /// vASR model cache directory.
    #[arg(long = "cache-dir", default_value_os_t = vasr::modelscope_cache_dir())]
    pub cache_dir: PathBuf,

    /// Device: auto, cpu, metal, cuda.
    #[arg(long, default_value = "auto", env = "VASR_DEVICE")]
    pub device: String,

    /// Enable in-situ quantization, for example 8 or q8_0.
    #[arg(long, env = "VASR_ISQ")]
    pub isq: Option<String>,

    /// Maximum concurrent decode sequences. Also controls CUDA decode graph max seqs and KV-cache block sizing.
    #[arg(
        long = "max-num-seqs",
        default_value_t = DEFAULT_MAX_NUM_SEQS,
        env = "VASR_MAX_NUM_SEQS"
    )]
    pub max_num_seqs: usize,

    /// Maximum hotword/context tokens reserved in every ASR KV-cache sequence.
    #[arg(
        long = "max-hotword-tokens",
        default_value_t = DEFAULT_MAX_HOTWORD_TOKENS,
        env = "VASR_MAX_HOTWORD_TOKENS"
    )]
    pub max_hotword_tokens: usize,
    /// Fraction of decode sequences expected to grow from base segment length to max segment length.
    #[arg(
        long = "kv-peak-ratio",
        default_value_t = DEFAULT_KV_PEAK_RATIO,
        env = "VASR_KV_PEAK_RATIO"
    )]
    pub kv_peak_ratio: f32,

    /// Fraction of total GPU memory for weights plus KV. Lower values leave more room for audio encoder workspaces.
    #[arg(
        long = "gpu-memory-utilization",
        default_value_t = DEFAULT_GPU_MEMORY_UTILIZATION,
        env = "VASR_GPU_MEMORY_UTILIZATION"
    )]
    pub gpu_memory_utilization: f32,

    /// Capture CUDA decode graphs with chunked Q8 lm_head next-id output. Automatically enabled for --isq 8.
    #[cfg(feature = "cuda-graph")]
    #[arg(long = "cuda-graph-chunked-argmax", default_value_t = false)]
    pub cuda_graph_chunked_argmax: bool,

    /// Vocabulary chunk size used by CUDA graph chunked lm_head argmax.
    #[cfg(feature = "cuda-graph")]
    #[arg(long = "cuda-graph-chunked-argmax-size", default_value_t = 4096)]
    pub cuda_graph_chunked_argmax_size: usize,

    /// Use MMQ for large-batch Q8 prefill lm_head next-id. Disable to compare with chunked candidates.
    #[cfg(feature = "cuda")]
    #[arg(
        long = "q8-prefill-lm-head-mmq",
        default_value_t = true,
        action = ArgAction::Set
    )]
    pub q8_prefill_lm_head_mmq: bool,

    /// Maximum temporary logits MiB for large-batch Q8 prefill lm_head MMQ. 0 disables the limit.
    #[cfg(feature = "cuda")]
    #[arg(
        long = "q8-prefill-lm-head-mmq-max-logits-mib",
        default_value_t = DEFAULT_Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_MIB
    )]
    pub q8_prefill_lm_head_mmq_max_logits_mib: usize,
}

#[derive(Debug, Clone, Args)]
pub struct VadCliArgs {
    /// VAD model: firered or fsmn.
    #[arg(long = "vad-model", default_value = "fsmn", env = "VASR_VAD_MODEL")]
    pub vad_model: String,

    /// VAD device: auto, cpu, metal, cuda.
    #[arg(long = "vad-device", default_value = "cpu", env = "VASR_VAD_DEVICE")]
    pub vad_device: String,

    /// VAD speech probability threshold in (0, 1].
    #[arg(long, default_value_t = 0.6, env = "VASR_VAD_THRESHOLD")]
    pub vad_threshold: f32,

    /// Minimum speech duration in milliseconds before a segment is confirmed; must be > 0.
    #[arg(long, default_value_t = 250, env = "VASR_VAD_MIN_SPEECH_MS")]
    pub vad_min_speech_ms: u64,

    /// Minimum trailing silence in milliseconds before ending a speech segment; must be > 0.
    #[arg(long, default_value_t = 800, env = "VASR_VAD_MIN_SILENCE_MS")]
    pub vad_min_silence_ms: u64,

    /// Symmetric padding in milliseconds added to detected speech segments before ASR; 0 disables padding.
    #[arg(long = "vad-pad-ms", default_value_t = 0, env = "VASR_VAD_PAD_MS")]
    pub vad_pad_ms: u64,

    /// CPU threads used inside VAD model kernels; 0 uses all available Rayon worker threads.
    #[arg(
        long = "vad-num-threads",
        default_value_t = 0,
        env = "VASR_VAD_NUM_THREADS"
    )]
    pub vad_num_threads: usize,

    /// Maximum offline VAD segment duration in seconds before ASR; must be finite and > 0.
    #[arg(
        long = "vad-max-segment-sec",
        default_value_t = 30.0,
        env = "VASR_VAD_MAX_SEGMENT_SEC"
    )]
    pub vad_max_segment_sec: f32,
}

#[derive(Debug, Clone, Args)]
pub struct TranscribePipelineArgs {
    #[command(flatten)]
    pub model: CommonModelArgs,

    /// Number of audio load workers. 0 uses all available CPU cores.
    #[arg(
        long = "num-load-workers",
        default_value_t = 1,
        env = "VASR_NUM_LOAD_WORKERS"
    )]
    pub num_load_workers: usize,

    /// Number of VAD workers. 0 uses all available CPU cores.
    #[arg(
        long = "num-vad-workers",
        default_value_t = 1,
        env = "VASR_NUM_VAD_WORKERS"
    )]
    pub num_vad_workers: usize,

    /// Enable periodic service pipeline performance table.
    #[arg(
        long = "profile",
        default_value_t = false,
        action = ArgAction::SetTrue,
        env = "VASR_PROFILE"
    )]
    pub profile: bool,

    /// Service pipeline performance table window in seconds.
    #[arg(
        long = "profile-window",
        default_value_t = DEFAULT_PROFILE_WINDOW_SECS,
        env = "VASR_PROFILE_WINDOW"
    )]
    pub profile_window: f64,

    /// Maximum combined audio duration sent to one ASR audio-encoder batch.
    #[arg(
        long = "encoder-batch-max-audio-seconds",
        default_value_t = DEFAULT_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS,
        env = "VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS"
    )]
    pub encoder_batch_max_audio_seconds: f64,

    /// Maximum number of VAD segments sent to one ASR audio-encoder batch.
    #[arg(
        long = "encoder-batch-max-segments",
        default_value_t = DEFAULT_ASR_ENCODER_BATCH_MAX_SEGMENTS,
        env = "VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS"
    )]
    pub encoder_batch_max_segments: usize,

    #[command(flatten)]
    pub vad: VadCliArgs,
}

#[derive(Debug, Clone, Args)]
pub struct ServeTranscribeArgs {
    #[command(flatten)]
    pub pipeline: ServePipelineArgs,

    /// ASR language hint.
    #[arg(long, env = "VASR_LANGUAGE", default_value = "Chinese")]
    pub language: Option<String>,

    /// Bind host.
    #[arg(long, default_value = "0.0.0.0", env = "VASR_HOST")]
    pub host: String,

    /// Bind port.
    #[arg(long, default_value_t = 8867, env = "VASR_PORT")]
    pub port: u16,
}

#[derive(Debug, Clone, Args)]
pub struct ServeModelArgs {
    /// Qwen3-ASR model: `Qwen3-ASR-0.6B` / `Qwen3-ASR-1.7B` auto-map to their repo ids, or pass a repo id / local directory directly.
    #[arg(long, default_value = DEFAULT_ASR_MODEL, env = "VASR_MODEL")]
    pub model: String,

    /// vASR model cache directory.
    #[arg(long = "cache-dir", default_value_os_t = vasr::modelscope_cache_dir())]
    pub cache_dir: PathBuf,

    /// Device: auto, cpu, metal, cuda.
    #[arg(long, default_value = "auto", env = "VASR_DEVICE")]
    pub device: String,

    /// Enable in-situ quantization, for example 8 or q8_0.
    #[arg(long, env = "VASR_ISQ")]
    pub isq: Option<String>,

    /// Maximum concurrent decode sequences. Also controls CUDA decode graph max seqs and KV-cache block sizing.
    #[arg(long = "max-num-seqs", env = "VASR_MAX_NUM_SEQS")]
    pub max_num_seqs: Option<usize>,

    /// Maximum hotword/context tokens reserved in every ASR KV-cache sequence.
    #[arg(long = "max-hotword-tokens", default_value_t = DEFAULT_MAX_HOTWORD_TOKENS, env = "VASR_MAX_HOTWORD_TOKENS")]
    pub max_hotword_tokens: usize,

    /// Fraction of total GPU memory for weights plus KV. Lower values leave more room for audio encoder workspaces.
    #[arg(long = "gpu-memory-utilization", env = "VASR_GPU_MEMORY_UTILIZATION")]
    pub gpu_memory_utilization: Option<f32>,
}

#[derive(Debug, Clone, Args)]
pub struct ServeVadArgs {
    /// VAD model: firered or fsmn.
    #[arg(long = "vad-model", default_value = "fsmn", env = "VASR_VAD_MODEL")]
    pub vad_model: String,

    /// VAD device: auto, cpu, metal, cuda.
    #[arg(long = "vad-device", default_value = "auto", env = "VASR_VAD_DEVICE")]
    pub vad_device: String,

    /// VAD speech probability threshold in (0, 1].
    #[arg(long, default_value_t = 0.6, env = "VASR_VAD_THRESHOLD")]
    pub vad_threshold: f32,

    /// Minimum speech duration in milliseconds before a segment is confirmed; must be > 0.
    #[arg(long, default_value_t = 250, env = "VASR_VAD_MIN_SPEECH_MS")]
    pub vad_min_speech_ms: u64,

    /// Minimum trailing silence in milliseconds before ending a speech segment; must be > 0.
    #[arg(long, default_value_t = 800, env = "VASR_VAD_MIN_SILENCE_MS")]
    pub vad_min_silence_ms: u64,

    /// Symmetric padding in milliseconds added to detected speech segments before ASR; 0 disables padding.
    #[arg(long = "vad-pad-ms", default_value_t = 0, env = "VASR_VAD_PAD_MS")]
    pub vad_pad_ms: u64,

    /// Maximum offline VAD segment duration in seconds before ASR; must be finite and > 0.
    #[arg(
        long = "vad-max-segment-sec",
        default_value_t = 30.0,
        env = "VASR_VAD_MAX_SEGMENT_SEC"
    )]
    pub vad_max_segment_sec: f32,
}

#[derive(Debug, Clone, Args)]
pub struct ServePipelineArgs {
    #[command(flatten)]
    pub model: ServeModelArgs,

    /// Number of audio load workers. 0 uses all available CPU cores.
    #[arg(
        long = "num-load-workers",
        default_value_t = 2,
        env = "VASR_NUM_LOAD_WORKERS"
    )]
    pub num_load_workers: usize,

    /// Number of VAD workers. 0 uses all available CPU cores.
    #[arg(
        long = "num-vad-workers",
        default_value_t = 2,
        env = "VASR_NUM_VAD_WORKERS"
    )]
    pub num_vad_workers: usize,

    /// Enable periodic service pipeline performance table.
    #[arg(long, default_value_t = false, action = ArgAction::SetTrue, env = "VASR_PROFILE")]
    pub profile: bool,

    /// Maximum combined audio duration sent to one ASR audio-encoder batch.
    #[arg(
        long = "encoder-batch-max-audio-seconds",
        env = "VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS"
    )]
    pub encoder_batch_max_audio_seconds: Option<f64>,

    #[command(flatten)]
    pub vad: ServeVadArgs,
}

/// Backward-compatible alias for the HTTP serve entry point.
pub type TranscribeArgs = ServeTranscribeArgs;

pub fn init_logging(verbose: bool) {
    let default = if verbose {
        "warn,vasr_transcribe=debug,vasr::models::qwen3asr=debug"
    } else {
        "warn,vasr_transcribe=info"
    };
    let filter = std::env::var("VASR_LOG").unwrap_or_else(|_| default.to_string());

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
        .with_timer(tracing_subscriber::fmt::time::SystemTime)
        .with_target(true)
        .compact()
        .init();
}

pub async fn run_transcribe(args: ServeTranscribeArgs) -> Result<()> {
    validate_serve_pipeline(&args.pipeline)?;
    vasr::set_cache_dir(args.pipeline.model.cache_dir.clone());
    log_transcribe_start();
    let vad_args = internal_vad_args(&args.pipeline.vad)?;
    let vad_model = build_vad_model(&vad_args)?;
    let device = resolve_device(&args.pipeline.model.device)?;
    let hardware = match crate::serve_tuning::probe_hardware(&device) {
        Ok(hardware) => Some(hardware),
        Err(error) => {
            tracing::warn!(target: "vasr_transcribe::serve", "automatic capacity probe unavailable ({error}); using baseline fallback values");
            None
        }
    };
    let model_profile = crate::serve_tuning::estimate_model_memory_profile(
        &args.pipeline.model.model,
        args.pipeline.model.isq.as_deref(),
        if device.is_cuda() {
            crate::serve_tuning::Backend::Cuda
        } else if device.is_metal() {
            crate::serve_tuning::Backend::Metal
        } else {
            crate::serve_tuning::Backend::Cpu
        },
    );
    if model_profile.is_none() {
        tracing::warn!(target: "vasr_transcribe::serve", "model memory profile unavailable for {:?}; using baseline fallback values", args.pipeline.model.model);
    }
    let overrides = serve_tuning_overrides(&args.pipeline)?;
    let tuning = resolve_tuning(
        hardware,
        model_profile,
        f64::from(args.pipeline.vad.vad_max_segment_sec),
        args.pipeline.vad.vad_pad_ms,
        args.pipeline.model.max_hotword_tokens,
        env_parse("VASR_KV_PEAK_RATIO")?.unwrap_or(DEFAULT_KV_PEAK_RATIO),
        overrides,
    )?;
    let pipeline = resolve_serve_pipeline(&args.pipeline, tuning)?;
    validate_pipeline(&pipeline)?;
    log_transcribe_tuning(
        &pipeline,
        if tuning.used_fallback {
            "baseline"
        } else {
            "hardware"
        },
        tuning_value_source(
            args.pipeline.model.max_num_seqs.is_some(),
            "--max-num-seqs",
            "VASR_MAX_NUM_SEQS",
        ),
        tuning_value_source(
            args.pipeline.model.gpu_memory_utilization.is_some(),
            "--gpu-memory-utilization",
            "VASR_GPU_MEMORY_UTILIZATION",
        ),
        tuning_value_source(
            args.pipeline.encoder_batch_max_audio_seconds.is_some(),
            "--encoder-batch-max-audio-seconds",
            "VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS",
        ),
    );
    let use_runtime_capacity =
        args.pipeline.model.max_num_seqs.is_none() && (device.is_cuda() || device.is_metal());
    let mut asr_options = asr_options_from_pipeline(&pipeline, args.language.clone());
    if use_runtime_capacity {
        // The paged cache knows the post-weight allocation capacity; let its scheduler use that
        // runtime value instead of treating the pre-load estimate as a hard concurrency limit.
        asr_options.decode_max_seqs = 0;
    }
    let pipeline = Arc::new(GlobalAsyncTranscribePipeline::start(
        AudioLoader,
        vad_model,
        load_asr_model_for_pipeline_on_device(
            &pipeline,
            args.language.as_deref(),
            cuda_graph_max_batch(&pipeline.model),
            false,
            device,
            tuning.reserved_gpu_bytes,
        )?,
        asr_options,
        vad_options_from_pipeline(&pipeline),
        worker_options_from_pipeline(&pipeline),
    ));
    let fatal_errors = pipeline.subscribe_fatal_errors();
    let transcribe_service = build_transcribe_service(pipeline);

    let app = Router::new()
        .route("/health", get(health))
        .merge(crate::server::transcribe_router(transcribe_service));
    let addr = bind_addr(&args.host, args.port)?;
    serve_app(app, addr, fatal_errors).await
}

fn validate_serve_pipeline(pipeline: &ServePipelineArgs) -> Result<()> {
    if pipeline.model.model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if pipeline.model.cache_dir.as_os_str().is_empty() {
        bail!("--cache-dir must not be empty");
    }
    validate_vad_args(&internal_vad_args(&pipeline.vad)?)?;
    Ok(())
}

fn resolve_serve_pipeline(
    args: &ServePipelineArgs,
    tuning: ResolvedServeTuning,
) -> Result<TranscribePipelineArgs> {
    let model = CommonModelArgs {
        model: args.model.model.clone(),
        cache_dir: args.model.cache_dir.clone(),
        device: args.model.device.clone(),
        isq: args.model.isq.clone(),
        max_num_seqs: tuning.max_num_seqs,
        max_hotword_tokens: args.model.max_hotword_tokens,
        kv_peak_ratio: env_parse("VASR_KV_PEAK_RATIO")?.unwrap_or(DEFAULT_KV_PEAK_RATIO),
        gpu_memory_utilization: tuning.gpu_memory_utilization,
        #[cfg(feature = "cuda-graph")]
        cuda_graph_chunked_argmax: false,
        #[cfg(feature = "cuda-graph")]
        cuda_graph_chunked_argmax_size: DEFAULT_CUDA_GRAPH_CHUNKED_ARGMAX_SIZE,
        #[cfg(feature = "cuda")]
        q8_prefill_lm_head_mmq: true,
        #[cfg(feature = "cuda")]
        q8_prefill_lm_head_mmq_max_logits_mib: DEFAULT_Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_MIB,
    };
    let vad = internal_vad_args(&args.vad)?;
    Ok(TranscribePipelineArgs {
        model,
        num_load_workers: args.num_load_workers,
        num_vad_workers: args.num_vad_workers,
        profile: args.profile,
        profile_window: env_parse("VASR_PROFILE_WINDOW")?.unwrap_or(DEFAULT_PROFILE_WINDOW_SECS),
        encoder_batch_max_audio_seconds: tuning.encoder_batch_max_audio_seconds,
        encoder_batch_max_segments: tuning.encoder_batch_max_segments,
        vad,
    })
}

fn serve_tuning_overrides(args: &ServePipelineArgs) -> Result<ServeTuningOverrides> {
    Ok(ServeTuningOverrides {
        max_num_seqs: args.model.max_num_seqs,
        gpu_memory_utilization: args.model.gpu_memory_utilization,
        encoder_batch_max_audio_seconds: args.encoder_batch_max_audio_seconds,
        encoder_batch_max_segments: env_parse("VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS")?,
    })
}

fn internal_vad_args(args: &ServeVadArgs) -> Result<VadCliArgs> {
    Ok(VadCliArgs {
        vad_model: args.vad_model.clone(),
        vad_device: args.vad_device.clone(),
        vad_threshold: args.vad_threshold,
        vad_min_speech_ms: args.vad_min_speech_ms,
        vad_min_silence_ms: args.vad_min_silence_ms,
        vad_pad_ms: args.vad_pad_ms,
        vad_num_threads: env_parse("VASR_VAD_NUM_THREADS")?.unwrap_or(0),
        vad_max_segment_sec: args.vad_max_segment_sec,
    })
}

fn env_parse<T: std::str::FromStr>(name: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<T>()
                .map_err(|error| anyhow::anyhow!("invalid {name} value {value:?}: {error}"))
        })
        .transpose()
}

fn tuning_value_source(has_value: bool, flag: &str, env_key: &str) -> &'static str {
    let cli_value = std::env::args_os().any(|argument| {
        let argument = argument.to_string_lossy();
        argument == flag || argument.starts_with(&format!("{flag}="))
    });
    if cli_value {
        "cli"
    } else if std::env::var_os(env_key).is_some() {
        "env"
    } else if has_value {
        "configured"
    } else {
        "auto"
    }
}

pub fn build_async_transcribe_pipeline(
    pipeline: &TranscribePipelineArgs,
    language: Option<String>,
) -> Result<AsyncTranscribePipeline> {
    validate_pipeline(pipeline)?;
    configure_model_cache_dir(pipeline);
    log_transcribe_start();
    let vad = build_vad_model(&pipeline.vad)?;
    log_transcribe_tuning(
        pipeline,
        "not-run",
        "configured",
        "configured",
        "configured",
    );
    let asr = load_asr_model_for_pipeline(
        pipeline,
        language.as_deref(),
        cuda_graph_max_batch(&pipeline.model),
        false,
    )?;
    Ok(AsyncTranscribePipeline::new(
        AudioLoader,
        vad,
        asr,
        asr_options_from_pipeline(pipeline, language),
    )
    .with_vad_options(vad_options_from_pipeline(pipeline))
    .with_worker_options(worker_options_from_pipeline(pipeline)))
}

pub fn build_global_transcribe_pipeline(
    pipeline: &TranscribePipelineArgs,
    language: Option<String>,
) -> Result<Arc<GlobalAsyncTranscribePipeline>> {
    validate_pipeline(pipeline)?;
    configure_model_cache_dir(pipeline);
    log_transcribe_start();
    let vad = build_vad_model(&pipeline.vad)?;
    log_transcribe_tuning(
        pipeline,
        "not-run",
        "configured",
        "configured",
        "configured",
    );
    let asr = load_asr_model_for_pipeline(
        pipeline,
        language.as_deref(),
        cuda_graph_max_batch(&pipeline.model),
        false,
    )?;
    Ok(Arc::new(GlobalAsyncTranscribePipeline::start(
        AudioLoader,
        vad,
        asr,
        asr_options_from_pipeline(pipeline, language),
        vad_options_from_pipeline(pipeline),
        worker_options_from_pipeline(pipeline),
    )))
}

pub fn validate_pipeline(pipeline: &TranscribePipelineArgs) -> Result<()> {
    validate_model_args(&pipeline.model)?;
    if pipeline.profile_window <= 0.0 || !pipeline.profile_window.is_finite() {
        bail!("--profile-window must be a positive finite number");
    }
    if !pipeline.encoder_batch_max_audio_seconds.is_finite()
        || pipeline.encoder_batch_max_audio_seconds <= 0.0
    {
        bail!("--encoder-batch-max-audio-seconds must be finite and greater than zero");
    }
    if pipeline.encoder_batch_max_segments == 0 {
        bail!("--encoder-batch-max-segments must be greater than zero");
    }
    validate_vad_args(&pipeline.vad)?;
    let padded_vad_segment_seconds =
        f64::from(pipeline.vad.vad_max_segment_sec) + 2.0 * pipeline.vad.vad_pad_ms as f64 / 1000.0;
    if padded_vad_segment_seconds > pipeline.encoder_batch_max_audio_seconds {
        bail!(
            "--vad-max-segment-sec plus --vad-pad-ms exceeds --encoder-batch-max-audio-seconds; lower the VAD segment limit or raise the encoder batch duration"
        );
    }
    Ok(())
}

fn configure_model_cache_dir(pipeline: &TranscribePipelineArgs) {
    vasr::set_cache_dir(pipeline.model.cache_dir.clone());
}

fn build_vad_model(vad_args: &VadCliArgs) -> Result<Option<Arc<dyn VadModel>>> {
    let vad_load_start = Instant::now();
    let vad_device = resolve_vad_device(vad_args.vad_device.as_str())?;
    let vad_model = vad_args.vad_model.trim().to_ascii_lowercase();
    let vad: Arc<dyn VadModel> = match vad_model.as_str() {
        "fsmn" => {
            let vad = load_fsmn_vad(&vad_device)?;
            Arc::new(vad) as Arc<dyn VadModel>
        }
        "firered" => {
            let vad = load_firered_vad(&vad_device)?;
            Arc::new(vad) as Arc<dyn VadModel>
        }
        other => bail!("--vad-model must be fsmn or firered; got {other:?}"),
    };
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{}",
        startup_log_line(
            "vad",
            &[
                ("model", vad_model),
                (
                    "load",
                    format_load_seconds(vad_load_start.elapsed().as_secs_f64()),
                ),
                ("device", device_label(&vad_device).to_string()),
            ],
        )
    );
    Ok(Some(vad))
}

fn asr_options_from_pipeline(
    pipeline: &TranscribePipelineArgs,
    language: Option<String>,
) -> AsrOptions {
    AsrOptions {
        language,
        decode_max_seqs: pipeline.model.max_num_seqs,
        chunk_max_sec: None,
        ..AsrOptions::default()
    }
}

fn worker_options_from_pipeline(pipeline: &TranscribePipelineArgs) -> PipelineWorkerOptions {
    PipelineWorkerOptions {
        num_load_workers: pipeline.num_load_workers,
        num_vad_workers: pipeline.num_vad_workers,
        profile: pipeline.profile,
        profile_window: std::time::Duration::from_secs_f64(pipeline.profile_window),
        encoder_batch_max_segments: pipeline.encoder_batch_max_segments,
        encoder_batch_max_audio_seconds: pipeline.encoder_batch_max_audio_seconds,
    }
}

fn vad_options_from_args(vad: &VadCliArgs) -> VadOptions {
    VadOptions {
        threshold: vad.vad_threshold,
        min_speech_ms: vad.vad_min_speech_ms,
        min_silence_ms: vad.vad_min_silence_ms,
        max_segment_ms: (vad.vad_max_segment_sec * 1000.0).round() as u64,
        pad_ms: vad.vad_pad_ms,
        num_threads: vad.vad_num_threads,
    }
}

fn vad_options_from_pipeline(pipeline: &TranscribePipelineArgs) -> VadOptions {
    vad_options_from_args(&pipeline.vad)
}

fn validate_model_args(args: &CommonModelArgs) -> Result<()> {
    if args.model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if args.cache_dir.as_os_str().is_empty() {
        bail!("--cache-dir must not be empty");
    }
    if args.max_num_seqs == 0 {
        bail!("--max-num-seqs must be greater than zero");
    }
    if !args.kv_peak_ratio.is_finite() || !(0.0..=1.0).contains(&args.kv_peak_ratio) {
        bail!("--kv-peak-ratio must be finite and between 0.0 and 1.0");
    }
    if !args.gpu_memory_utilization.is_finite()
        || !(0.0..=1.0).contains(&args.gpu_memory_utilization)
        || args.gpu_memory_utilization == 0.0
    {
        bail!("--gpu-memory-utilization must be finite and in (0.0, 1.0]");
    }
    #[cfg(feature = "cuda-graph")]
    if args.cuda_graph_chunked_argmax_size == 0 {
        bail!("--cuda-graph-chunked-argmax-size must be greater than zero");
    }
    Ok(())
}

fn bytes_to_gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// 在加载 VAD 和 ASR 之前打印统一的推理启动日志。
fn log_transcribe_start() {
    tracing::info!(target: "vasr_transcribe::serve", "starting transcription");
}

fn format_transcribe_tuning_log(
    pipeline: &TranscribePipelineArgs,
    probe: &str,
    max_num_seqs_source: &str,
    gpu_memory_utilization_source: &str,
    encoder_batch_audio_seconds_source: &str,
) -> String {
    format!(
        "transcribe tuning | probe={probe} max_num_seqs={}({max_num_seqs_source}) gpu_memory_budget_fraction={:.3}({gpu_memory_utilization_source}) encoder_batch_max_audio_seconds={:.1}({encoder_batch_audio_seconds_source}) encoder_batch_max_segments={} workers=load:{}/vad:{}",
        pipeline.model.max_num_seqs,
        pipeline.model.gpu_memory_utilization,
        pipeline.encoder_batch_max_audio_seconds,
        pipeline.encoder_batch_max_segments,
        pipeline.num_load_workers,
        pipeline.num_vad_workers,
    )
}

fn log_transcribe_tuning(
    pipeline: &TranscribePipelineArgs,
    probe: &str,
    max_num_seqs_source: &str,
    gpu_memory_utilization_source: &str,
    encoder_batch_audio_seconds_source: &str,
) {
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{}",
        format_transcribe_tuning_log(
            pipeline,
            probe,
            max_num_seqs_source,
            gpu_memory_utilization_source,
            encoder_batch_audio_seconds_source,
        )
    );
}

/// 把加载耗时格式化成秒，不带单位后缀。
fn format_load_seconds(seconds: f64) -> String {
    format!("{seconds:.3}")
}

/// 把字节数格式化成 GiB，保留两位小数。
fn format_gib(bytes: usize) -> String {
    format!("{:.2}GiB", bytes_to_gib(bytes))
}

/// 拼一条启动日志：主体后只有一个竖线，属性用空格分隔。
fn startup_log_line(subject: &str, fields: &[(&str, String)]) -> String {
    let mut line = String::from(subject);
    if fields.is_empty() {
        return line;
    }
    line.push_str(" |");
    for (key, value) in fields {
        line.push(' ');
        line.push_str(key);
        line.push('=');
        line.push_str(value);
    }
    line
}

fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::U8 => "u8",
        DType::U32 => "u32",
        DType::I64 => "i64",
        _ => "unknown",
    }
}

/// 打印 ASR 加载结果和显存占用。
///
/// 显存行只保留整卡占用、参数、KV 和 CUDA graph。paged attention workspace 不写入启动日志。
fn log_model_loaded(
    model: &str,
    dtype: DType,
    stats: &vasr::OfflineAsrBuildStats,
    device: &Device,
) {
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{}",
        startup_log_line(
            "asr",
            &[
                ("model", model.to_string()),
                ("load", format_load_seconds(stats.build_seconds)),
                ("device", device_label(device).to_string()),
                ("dtype", dtype_label(dtype).to_string()),
            ],
        )
    );

    let mut gpu_fields = Vec::new();
    // 没有 CUDA 显存快照时不写 used/total/free。
    #[cfg(all(feature = "cuda", feature = "paged-attn"))]
    if device.is_cuda()
        && let Some(memory) = vasr::vasr_paged_attn::cuda_graph::cuda_memory_snapshot(device)
    {
        gpu_fields.push(("used", format_gib(memory.used())));
        gpu_fields.push(("total", format_gib(memory.total)));
        gpu_fields.push(("free", format_gib(memory.free)));
    }
    gpu_fields.push((
        "params",
        stats
            .parameter_bytes
            .map(format_gib)
            .unwrap_or_else(|| "unknown".to_string()),
    ));
    if let Some(bytes) = stats.kv_cache_bytes {
        gpu_fields.push(("kv", format_gib(bytes)));
    }
    if let Some(bytes) = stats.cuda_graph_bytes {
        gpu_fields.push(("graph", format_gib(bytes)));
    }
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{}",
        startup_log_line("gpu", &gpu_fields)
    );
}

fn validate_vad_args(vad: &VadCliArgs) -> Result<()> {
    match vad.vad_model.trim().to_ascii_lowercase().as_str() {
        "fsmn" | "firered" => {}
        other => bail!("--vad-model must be fsmn or firered; got {other:?}"),
    }
    match vad.vad_device.trim().to_ascii_lowercase().as_str() {
        "auto" | "cpu" | "metal" | "cuda" | "gpu" => {}
        other => bail!("--vad-device must be auto, cpu, metal, or cuda; got {other:?}"),
    }
    if !vad.vad_threshold.is_finite() || vad.vad_threshold <= 0.0 || vad.vad_threshold > 1.0 {
        bail!("--vad-threshold must be finite and in (0, 1]");
    }
    if vad.vad_min_speech_ms == 0 {
        bail!("--vad-min-speech-ms must be greater than zero");
    }
    if vad.vad_min_silence_ms == 0 {
        bail!("--vad-min-silence-ms must be greater than zero");
    }
    if !vad.vad_max_segment_sec.is_finite() || vad.vad_max_segment_sec <= 0.0 {
        bail!("--vad-max-segment-sec must be finite and greater than zero");
    }
    Ok(())
}

#[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
fn cuda_graph_max_batch(args: &CommonModelArgs) -> Option<usize> {
    if std::env::var_os("VASR_DISABLE_CUDA_GRAPH").is_some() {
        tracing::info!(
            target: "vasr_transcribe::serve",
            "CUDA decode graph disabled by VASR_DISABLE_CUDA_GRAPH."
        );
        return None;
    }
    if args.max_num_seqs == 0 {
        tracing::info!(
            target: "vasr_transcribe::serve",
            "CUDA decode graph disabled by --max-num-seqs=0."
        );
        return None;
    }
    let graph_max_batch = std::env::var("VASR_CUDA_GRAPH_MAX_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(args.max_num_seqs)
        .min(args.max_num_seqs);
    Some(graph_max_batch)
}

#[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
fn cuda_graph_max_batch(_args: &CommonModelArgs) -> Option<usize> {
    None
}

fn load_asr_model_for_pipeline(
    pipeline: &TranscribePipelineArgs,
    #[cfg_attr(not(feature = "paged-attn"), allow(unused_variables))] language: Option<&str>,
    #[cfg_attr(
        not(all(feature = "cuda-graph", feature = "paged-attn")),
        allow(unused_variables)
    )]
    cuda_graph_max_batch: Option<usize>,
    scheduled: bool,
) -> Result<Arc<dyn AsrModel>> {
    let args = &pipeline.model;
    let device = resolve_device(&args.device)?;
    load_asr_model_for_pipeline_on_device(
        pipeline,
        language,
        cuda_graph_max_batch,
        scheduled,
        device,
        0,
    )
}

fn load_asr_model_for_pipeline_on_device(
    pipeline: &TranscribePipelineArgs,
    #[cfg_attr(not(feature = "paged-attn"), allow(unused_variables))] language: Option<&str>,
    #[cfg_attr(
        not(all(feature = "cuda-graph", feature = "paged-attn")),
        allow(unused_variables)
    )]
    cuda_graph_max_batch: Option<usize>,
    scheduled: bool,
    device: Device,
    #[cfg_attr(not(feature = "paged-attn"), allow(unused_variables))]
    gpu_memory_reserved_bytes: usize,
) -> Result<Arc<dyn AsrModel>> {
    let args = &pipeline.model;
    let dtype = auto_dtype(&device)?;

    let output = build_on_selected_device(&device, |selected_device| {
        vasr::build_offline_asr_model(vasr::OfflineAsrBuildRequest {
            model: args.model.clone(),
            cache_dir: args.cache_dir.clone(),
            device: selected_device.clone(),
            dtype,
            progress: Progress::new(Arc::new(IndicatifProgressSink)),
            quantization: vasr::AsrQuantizationConfig {
                isq: args.isq.clone(),
                #[cfg(feature = "cuda")]
                q8_prefill_lm_head_mmq: args.q8_prefill_lm_head_mmq,
                #[cfg(feature = "cuda")]
                q8_prefill_lm_head_mmq_max_logits_bytes: args
                    .q8_prefill_lm_head_mmq_max_logits_mib
                    .saturating_mul(1024 * 1024),
            },
            #[cfg(feature = "paged-attn")]
            runtime: vasr::AsrRuntimeConfig {
                language: language.map(str::to_string),
                max_num_seqs: args.max_num_seqs,
                max_segment_sec: pipeline.vad.vad_max_segment_sec,
                max_hotword_tokens: args.max_hotword_tokens,
                kv_peak_ratio: args.kv_peak_ratio,
                gpu_memory_utilization: args.gpu_memory_utilization,
                gpu_memory_reserved_bytes,
                #[cfg(feature = "cuda-graph")]
                cuda_graph_chunked_argmax: args.cuda_graph_chunked_argmax,
                #[cfg(feature = "cuda-graph")]
                cuda_graph_chunked_argmax_size: args.cuda_graph_chunked_argmax_size,
            },
            #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
            cuda_graph_max_batch,
        })
    })?;
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    configure_cuda_mempool_release_threshold(&device);
    log_model_loaded(&args.model, dtype, &output.stats, &device);
    let base_asr = output.model;
    if scheduled {
        let asr_scheduler = InferenceScheduler::start("asr")?;
        Ok(Arc::new(ScheduledAsrModel::new(
            Arc::clone(&base_asr),
            asr_scheduler.clone(),
        )))
    } else {
        Ok(base_asr)
    }
}

fn build_on_selected_device<T>(
    selected_device: &Device,
    build: impl FnOnce(&Device) -> Result<T>,
) -> Result<T> {
    build(selected_device)
}

#[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
fn configure_cuda_mempool_release_threshold(device: &Device) {
    if std::env::var_os("VASR_DISABLE_CUDA_MEMPOOL_RELEASE_THRESHOLD").is_some() {
        return;
    }
    let threshold_mib = std::env::var("VASR_CUDA_MEMPOOL_RELEASE_THRESHOLD_MIB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let threshold_bytes = threshold_mib.saturating_mul(1024 * 1024);
    vasr::vasr_paged_attn::cuda_graph::cuda_set_device_mem_pool_release_threshold(
        device,
        threshold_bytes,
    );
}

fn bind_addr(host: &str, port: u16) -> Result<SocketAddr> {
    Ok(format!("{host}:{port}").parse()?)
}

async fn serve_app(
    app: Router,
    addr: SocketAddr,
    mut fatal_errors: tokio::sync::watch::Receiver<Option<String>>,
) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        target: "vasr_transcribe::serve",
        "server ready | http://{addr} endpoints=health,transcribe"
    );
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    };
    tokio::pin!(server);

    tokio::select! {
        result = server.as_mut() => {
            result?;
            Ok(())
        }
        _ = tokio::signal::ctrl_c() => {
            let _ = shutdown_tx.send(());
            server.as_mut().await?;
            Ok(())
        }
        fatal = wait_for_fatal_error(&mut fatal_errors) => {
            let fatal = fatal?;
            tracing::error!(
                target: "vasr_transcribe::serve",
                "fatal pipeline failure; terminating service process: {fatal}"
            );
            let _ = shutdown_tx.send(());
            if tokio::time::timeout(Duration::from_secs(1), server.as_mut())
                .await
                .is_err()
            {
                tracing::warn!(
                    target: "vasr_transcribe::serve",
                    "graceful shutdown timed out after fatal pipeline failure"
                );
            }
            bail!("fatal pipeline failure: {fatal}")
        }
    }
}

async fn wait_for_fatal_error(
    fatal_errors: &mut tokio::sync::watch::Receiver<Option<String>>,
) -> Result<String> {
    loop {
        if let Some(error) = fatal_errors.borrow().clone() {
            return Ok(error);
        }
        fatal_errors
            .changed()
            .await
            .map_err(|_| anyhow::anyhow!("fatal pipeline error channel closed"))?;
    }
}

fn resolve_vad_device(value: &str) -> Result<Device> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => {
            #[cfg(feature = "cuda")]
            {
                match Device::new_cuda(0) {
                    Ok(device) => return Ok(device),
                    Err(error) => {
                        tracing::warn!(target: "vasr_transcribe::serve", "VAD CUDA initialization failed; trying CPU: {error}")
                    }
                }
            }
            #[cfg(feature = "metal")]
            {
                match Device::new_metal(0) {
                    Ok(device) => return Ok(device),
                    Err(error) => {
                        tracing::warn!(target: "vasr_transcribe::serve", "VAD Metal initialization failed; trying CPU: {error}")
                    }
                }
            }
            tracing::info!(target: "vasr_transcribe::serve", "VAD auto device selected CPU");
            Ok(Device::Cpu)
        }
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(|err| {
                    anyhow::anyhow!("failed to create Metal device 0 for VAD: {err}")
                })
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("--vad-device metal requires building with `--features metal`")
            }
        }
        "cuda" | "gpu" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0)
                    .map_err(|err| anyhow::anyhow!("failed to create CUDA device 0 for VAD: {err}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("--vad-device cuda requires building with `--features cuda`")
            }
        }
        other => bail!("--vad-device must be auto, cpu, metal, or cuda; got {other:?}"),
    }
}

fn load_fsmn_vad(device: &Device) -> Result<FsmnVadModel> {
    FsmnVadModel::from_pretrained_on_device(None::<&std::path::Path>, device)
}

fn load_firered_vad(device: &Device) -> Result<FireRedVadModel> {
    FireRedVadModel::from_pretrained_on_device(None::<&std::path::Path>, device)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Debug, Parser)]
    struct TestServeCli {
        #[command(flatten)]
        args: ServeTranscribeArgs,
    }

    #[derive(Debug, Parser)]
    struct TestRunCli {
        #[command(flatten)]
        args: crate::run::RunTranscribeArgs,
    }

    #[test]
    fn serve_help_hides_internal_tuning_flags() {
        let help = TestServeCli::command().render_long_help().to_string();
        for retained in [
            "--max-hotword-tokens",
            "--num-load-workers",
            "--num-vad-workers",
            "--encoder-batch-max-audio-seconds",
            "--vad-model",
            "--vad-threshold",
            "--vad-min-speech-ms",
            "--vad-min-silence-ms",
            "--vad-pad-ms",
            "--vad-max-segment-sec",
        ] {
            assert!(help.contains(retained), "serve help lost {retained}");
        }
        for flag in [
            "--decode-tokens-per-sec",
            "--decode-min-tokens",
            "--kv-peak-ratio",
            "--cuda-graph-chunked-argmax",
            "--cuda-graph-chunked-argmax-size",
            "--q8-prefill-lm-head-mmq",
            "--q8-prefill-lm-head-mmq-max-logits-mib",
            "--profile-window",
            "--vad-num-threads",
            "--encoder-batch-max-segments",
        ] {
            assert!(
                !help.contains(flag),
                "serve help unexpectedly contains {flag}"
            );
        }
    }

    #[test]
    fn run_defaults_and_other_tuning_options_remain_unchanged() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let help = TestRunCli::command().render_long_help().to_string();
        for retained in [
            "--kv-peak-ratio",
            "--profile-window",
            "--vad-num-threads",
            "--encoder-batch-max-segments",
        ] {
            assert!(help.contains(retained), "run help lost {retained}");
        }
        for removed in ["--decode-tokens-per-sec", "--decode-min-tokens"] {
            assert!(
                !help.contains(removed),
                "run help unexpectedly contains {removed}"
            );
        }

        let args = TestRunCli::try_parse_from(["test", "--input", "input.wav"])
            .expect("run defaults should parse")
            .args;
        assert_eq!(args.pipeline.model.max_num_seqs, DEFAULT_MAX_NUM_SEQS);
        assert_eq!(
            args.pipeline.model.gpu_memory_utilization,
            DEFAULT_GPU_MEMORY_UTILIZATION
        );
        assert_eq!(args.pipeline.num_load_workers, 1);
        assert_eq!(args.pipeline.num_vad_workers, 1);
        assert_eq!(args.pipeline.encoder_batch_max_audio_seconds, 30.0);
        assert_eq!(args.pipeline.encoder_batch_max_segments, 32);
        assert_eq!(args.pipeline.vad.vad_device, "cpu");
    }

    #[test]
    fn serve_capacity_defaults_are_auto_and_workers_default_to_two() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let args = TestServeCli::try_parse_from(["test"])
            .expect("serve defaults should parse")
            .args;
        assert_eq!(args.pipeline.model.max_num_seqs, None);
        assert_eq!(args.pipeline.model.gpu_memory_utilization, None);
        assert_eq!(args.pipeline.model.max_hotword_tokens, 128);
        assert_eq!(args.pipeline.encoder_batch_max_audio_seconds, None);
        assert_eq!(args.pipeline.num_load_workers, 2);
        assert_eq!(args.pipeline.num_vad_workers, 2);
        assert_eq!(args.pipeline.vad.vad_device, "auto");
        assert_eq!(args.pipeline.model.device, "auto");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn serve_accepts_metal_vad_device() -> Result<()> {
        let args = TestServeCli::try_parse_from(["test", "--vad-device", "metal"])
            .expect("Metal VAD device should parse");
        validate_vad_args(&internal_vad_args(&args.args.pipeline.vad)?)?;
        assert!(resolve_vad_device("metal")?.is_metal());
        Ok(())
    }

    #[test]
    fn selected_device_oom_is_returned_without_retrying_another_backend() {
        let calls = std::cell::Cell::new(0);
        let error = build_on_selected_device(&Device::Cpu, |_| -> Result<()> {
            calls.set(calls.get() + 1);
            Err(anyhow::anyhow!("CUDA out of memory"))
        })
        .expect_err("the selected device OOM must propagate");

        assert_eq!(calls.get(), 1, "device loading must not be retried");
        assert!(error.to_string().contains("CUDA out of memory"));
    }

    #[test]
    fn serve_tuning_environment_values_reach_runtime_and_cli_wins() {
        let _guard = ENV_LOCK.lock().expect("env lock should not be poisoned");
        let env_values = [
            ("VASR_MAX_NUM_SEQS", "17"),
            ("VASR_GPU_MEMORY_UTILIZATION", "0.73"),
            ("VASR_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS", "45"),
            ("VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS", "6"),
            ("VASR_KV_PEAK_RATIO", "0.4"),
            ("VASR_PROFILE_WINDOW", "7"),
            ("VASR_VAD_NUM_THREADS", "3"),
        ];
        let previous = env_values.map(|(key, _)| (key, std::env::var_os(key)));
        unsafe {
            for (key, value) in env_values {
                std::env::set_var(key, value);
            }
        }

        let parsed = TestServeCli::try_parse_from(["test"])
            .expect("environment-backed serve tuning should parse");
        let overrides = serve_tuning_overrides(&parsed.args.pipeline)
            .expect("internal environment tuning should parse");
        assert_eq!(overrides.max_num_seqs, Some(17));
        assert_eq!(overrides.gpu_memory_utilization, Some(0.73));
        assert_eq!(overrides.encoder_batch_max_audio_seconds, Some(45.0));
        assert_eq!(overrides.encoder_batch_max_segments, Some(6));

        let tuning = resolve_tuning(None, None, 30.0, 0, 128, 0.4, overrides)
            .expect("manual environment values should be retained by fallback tuning");
        let pipeline = resolve_serve_pipeline(&parsed.args.pipeline, tuning)
            .expect("internal environment values should enter the pipeline");
        assert_eq!(pipeline.model.kv_peak_ratio, 0.4);
        assert_eq!(pipeline.profile_window, 7.0);
        assert_eq!(pipeline.vad.vad_num_threads, 3);
        assert_eq!(pipeline.encoder_batch_max_segments, 6);

        let cli = TestServeCli::try_parse_from([
            "test",
            "--max-num-seqs",
            "9",
            "--gpu-memory-utilization",
            "0.6",
            "--encoder-batch-max-audio-seconds",
            "50",
        ])
        .expect("CLI tuning values should parse");
        assert_eq!(cli.args.pipeline.model.max_num_seqs, Some(9));
        assert_eq!(cli.args.pipeline.model.gpu_memory_utilization, Some(0.6));
        assert_eq!(
            cli.args.pipeline.encoder_batch_max_audio_seconds,
            Some(50.0)
        );

        unsafe {
            for (key, old_value) in previous {
                if let Some(value) = old_value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }

    fn default_model_args() -> CommonModelArgs {
        CommonModelArgs {
            model: DEFAULT_ASR_MODEL.to_string(),
            cache_dir: vasr::modelscope_cache_dir(),
            device: "auto".to_string(),
            isq: None,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            max_hotword_tokens: DEFAULT_MAX_HOTWORD_TOKENS,
            kv_peak_ratio: DEFAULT_KV_PEAK_RATIO,
            gpu_memory_utilization: DEFAULT_GPU_MEMORY_UTILIZATION,
            #[cfg(feature = "cuda-graph")]
            cuda_graph_chunked_argmax: false,
            #[cfg(feature = "cuda-graph")]
            cuda_graph_chunked_argmax_size: 4096,
            #[cfg(feature = "cuda")]
            q8_prefill_lm_head_mmq: true,
            #[cfg(feature = "cuda")]
            q8_prefill_lm_head_mmq_max_logits_mib: DEFAULT_Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_MIB,
        }
    }

    fn default_pipeline_args() -> TranscribePipelineArgs {
        TranscribePipelineArgs {
            model: default_model_args(),
            num_load_workers: 1,
            num_vad_workers: 1,
            profile: false,
            profile_window: DEFAULT_PROFILE_WINDOW_SECS,
            encoder_batch_max_segments: DEFAULT_ASR_ENCODER_BATCH_MAX_SEGMENTS,
            encoder_batch_max_audio_seconds: DEFAULT_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS,
            vad: VadCliArgs {
                vad_model: "fsmn".to_string(),
                vad_device: "cpu".to_string(),
                vad_threshold: 0.6,
                vad_min_speech_ms: 250,
                vad_min_silence_ms: 800,
                vad_pad_ms: 0,
                vad_num_threads: 0,
                vad_max_segment_sec: 30.0,
            },
        }
    }

    #[test]
    fn pipeline_defaults_match_validated_vad_batch_bounds() {
        let pipeline = default_pipeline_args();

        let opts = asr_options_from_pipeline(&pipeline, None);
        assert!(opts.context.is_empty());
    }

    #[test]
    fn max_num_seqs_must_be_positive_for_kv_sizing() {
        let args = CommonModelArgs {
            max_num_seqs: 0,
            ..default_model_args()
        };

        let error = validate_model_args(&args).expect_err("max_num_seqs=0 should fail");
        assert!(error.to_string().contains("--max-num-seqs"));
    }

    #[test]
    fn kv_peak_ratio_must_be_valid() {
        let args = CommonModelArgs {
            kv_peak_ratio: 1.5,
            ..default_model_args()
        };

        let error = validate_model_args(&args).expect_err("kv_peak_ratio=1.5 should fail");
        assert!(error.to_string().contains("--kv-peak-ratio"));
    }

    #[tokio::test]
    async fn fatal_error_subscription_delivers_the_failure_reason() {
        let (tx, mut rx) = tokio::sync::watch::channel(None);
        tx.send_replace(Some("CUDA out of memory".to_string()));

        let error = wait_for_fatal_error(&mut rx)
            .await
            .expect("fatal error should be delivered");

        assert_eq!(error, "CUDA out of memory");
    }

    #[tokio::test]
    async fn fatal_pipeline_error_stops_http_server_with_error() {
        let (tx, rx) = tokio::sync::watch::channel(None);
        tx.send_replace(Some("CUDA out of memory".to_string()));

        let error = serve_app(
            Router::new(),
            "127.0.0.1:0".parse().expect("valid test address"),
            rx,
        )
        .await
        .expect_err("fatal pipeline error must stop the HTTP server");

        assert!(error.to_string().contains("CUDA out of memory"), "{error}");
    }

    #[test]
    fn startup_log_line_keeps_one_separator() {
        let line = startup_log_line(
            "vad",
            &[
                ("model", "fsmn".to_string()),
                ("load", "0.009".to_string()),
                ("device", "cuda".to_string()),
            ],
        );
        assert_eq!(line, "vad | model=fsmn load=0.009 device=cuda");

        let gpu = startup_log_line(
            "gpu",
            &[
                ("used", "19.45GiB".to_string()),
                ("total", "23.52GiB".to_string()),
                ("free", "4.07GiB".to_string()),
                ("params", "1.75GiB".to_string()),
                ("kv", "17.06GiB".to_string()),
                ("graph", "0.516GiB".to_string()),
            ],
        );
        assert_eq!(
            gpu,
            "gpu | used=19.45GiB total=23.52GiB free=4.07GiB params=1.75GiB kv=17.06GiB graph=0.516GiB"
        );
        assert_eq!(gpu.matches('|').count(), 1);
    }

    #[test]
    fn transcribe_tuning_log_uses_the_same_schema_for_configured_and_auto_values() {
        let mut pipeline = default_pipeline_args();
        pipeline.model.max_num_seqs = 107;
        pipeline.model.gpu_memory_utilization = 1.0;
        pipeline.encoder_batch_max_audio_seconds = 66.0;
        pipeline.encoder_batch_max_segments = 20;
        pipeline.num_load_workers = 2;
        pipeline.num_vad_workers = 2;

        let auto = format_transcribe_tuning_log(&pipeline, "hardware", "auto", "auto", "auto");
        assert_eq!(
            auto,
            "transcribe tuning | probe=hardware max_num_seqs=107(auto) gpu_memory_budget_fraction=1.000(auto) encoder_batch_max_audio_seconds=66.0(auto) encoder_batch_max_segments=20 workers=load:2/vad:2"
        );

        let configured = format_transcribe_tuning_log(
            &pipeline,
            "not-run",
            "configured",
            "configured",
            "configured",
        );
        assert_eq!(
            configured,
            "transcribe tuning | probe=not-run max_num_seqs=107(configured) gpu_memory_budget_fraction=1.000(configured) encoder_batch_max_audio_seconds=66.0(configured) encoder_batch_max_segments=20 workers=load:2/vad:2"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn q8_prefill_lm_head_mmq_defaults_enabled() {
        let args = default_model_args();
        assert!(args.q8_prefill_lm_head_mmq);
        assert_eq!(
            args.q8_prefill_lm_head_mmq_max_logits_mib,
            DEFAULT_Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_MIB
        );
    }
}
