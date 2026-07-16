use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::device::{auto_dtype, device_label, resolve_device};
use crate::scheduler::{InferenceScheduler, ScheduledAsrModel};
use anyhow::{Result, bail};
use asr_data::AudioLoader;
use axum::{Json, Router, routing::get};
use candle_core::{DType, Device};
use clap::{ArgAction, Args};
use indicatif::{ProgressBar, ProgressStyle};
use vasr::utils::progress::{Progress, ProgressEvent, ProgressHandle, ProgressSink, ProgressTask};
use vasr::{AsrOptions, OfflineAsrModel as AsrModel};
use vasr_vad::{FireRedVadModel, FsmnVadModel, VadModel, VadOptions};

use crate::pipeline::{
    AsyncTranscribePipeline, GlobalAsyncTranscribePipeline, PipelineWorkerOptions,
};
use crate::server::build_transcribe_service;

const DEFAULT_MAX_NUM_SEQS: usize = 256;
const DEFAULT_ASR_MODEL: &str = "Qwen/Qwen3-ASR-0.6B";
const DEFAULT_MAX_HOTWORD_TOKENS: usize = 128;
const DEFAULT_KV_PEAK_RATIO: f32 = 0.25;
const DEFAULT_DECODE_TOKENS_PER_SEC: f64 = 4.25;
const DEFAULT_DECODE_MIN_TOKENS: usize = 128;
const DEFAULT_PROFILE_WINDOW_SECS: f64 = 5.0;
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
        default_value_t = DEFAULT_MAX_NUM_SEQS
    )]
    pub max_num_seqs: usize,

    /// Maximum hotword/context tokens reserved in every ASR KV-cache sequence.
    #[arg(
        long = "max-hotword-tokens",
        default_value_t = DEFAULT_MAX_HOTWORD_TOKENS,
        env = "VASR_MAX_HOTWORD_TOKENS"
    )]
    pub max_hotword_tokens: usize,

    /// Estimated ASR decode tokens generated per second of segment audio.
    #[arg(long = "decode-tokens-per-sec", default_value_t = DEFAULT_DECODE_TOKENS_PER_SEC)]
    pub decode_tokens_per_sec: f64,

    /// Minimum runtime decode tokens reserved per ASR segment.
    #[arg(long = "decode-min-tokens", default_value_t = DEFAULT_DECODE_MIN_TOKENS)]
    pub decode_min_tokens: usize,

    /// Fraction of decode sequences expected to grow from base segment length to max segment length.
    #[arg(
        long = "kv-peak-ratio",
        default_value_t = DEFAULT_KV_PEAK_RATIO,
        env = "VASR_KV_PEAK_RATIO"
    )]
    pub kv_peak_ratio: f32,

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

    /// VAD device: auto, cpu, cuda.
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

    #[command(flatten)]
    pub vad: VadCliArgs,
}

#[derive(Debug, Clone, Args)]
pub struct ServeTranscribeArgs {
    #[command(flatten)]
    pub pipeline: TranscribePipelineArgs,

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
    validate_pipeline(&args.pipeline)?;
    configure_model_cache_dir(&args.pipeline);
    let pipeline = Arc::new(GlobalAsyncTranscribePipeline::start(
        AudioLoader,
        build_vad_model(&args.pipeline.vad)?,
        load_asr_model_for_pipeline(
            &args.pipeline,
            args.language.as_deref(),
            cuda_graph_max_batch(&args.pipeline.model),
            false,
        )?,
        asr_options_from_pipeline(&args.pipeline, args.language.clone()),
        vad_options_from_pipeline(&args.pipeline),
        worker_options_from_pipeline(&args.pipeline),
    ));
    let fatal_errors = pipeline.subscribe_fatal_errors();
    let transcribe_service = build_transcribe_service(pipeline);

    let app = Router::new()
        .route("/health", get(health))
        .merge(crate::server::transcribe_router(transcribe_service));
    let addr = bind_addr(&args.host, args.port)?;
    serve_app(app, addr, fatal_errors).await
}

pub fn build_async_transcribe_pipeline(
    pipeline: &TranscribePipelineArgs,
    language: Option<String>,
) -> Result<AsyncTranscribePipeline> {
    validate_pipeline(pipeline)?;
    configure_model_cache_dir(pipeline);
    let asr = load_asr_model_for_pipeline(
        pipeline,
        language.as_deref(),
        cuda_graph_max_batch(&pipeline.model),
        false,
    )?;
    Ok(AsyncTranscribePipeline::new(
        AudioLoader,
        build_vad_model(&pipeline.vad)?,
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
    let asr = load_asr_model_for_pipeline(
        pipeline,
        language.as_deref(),
        cuda_graph_max_batch(&pipeline.model),
        false,
    )?;
    Ok(Arc::new(GlobalAsyncTranscribePipeline::start(
        asr_data::AudioLoader,
        build_vad_model(&pipeline.vad)?,
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
    validate_vad_args(&pipeline.vad)
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
        "loaded vad | {:.3}s model={} device={}",
        vad_load_start.elapsed().as_secs_f64(),
        vad_model,
        device_label(&vad_device),
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
        decode_tokens_per_sec: pipeline.model.decode_tokens_per_sec,
        decode_min_tokens: pipeline.model.decode_min_tokens,
        ..AsrOptions::default()
    }
}

fn worker_options_from_pipeline(pipeline: &TranscribePipelineArgs) -> PipelineWorkerOptions {
    PipelineWorkerOptions {
        num_load_workers: pipeline.num_load_workers,
        num_vad_workers: pipeline.num_vad_workers,
        profile: pipeline.profile,
        profile_window: std::time::Duration::from_secs_f64(pipeline.profile_window),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model_args() -> CommonModelArgs {
        CommonModelArgs {
            model: DEFAULT_ASR_MODEL.to_string(),
            cache_dir: vasr::modelscope_cache_dir(),
            device: "auto".to_string(),
            isq: None,
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            max_hotword_tokens: DEFAULT_MAX_HOTWORD_TOKENS,
            decode_tokens_per_sec: DEFAULT_DECODE_TOKENS_PER_SEC,
            decode_min_tokens: DEFAULT_DECODE_MIN_TOKENS,
            kv_peak_ratio: DEFAULT_KV_PEAK_RATIO,
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

        assert!(error.to_string().contains("CUDA out of memory"));
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
    if !args.decode_tokens_per_sec.is_finite() || args.decode_tokens_per_sec <= 0.0 {
        bail!("--decode-tokens-per-sec must be finite and greater than zero");
    }
    if args.decode_min_tokens == 0 {
        bail!("--decode-min-tokens must be greater than zero");
    }
    if !args.kv_peak_ratio.is_finite() || !(0.0..=1.0).contains(&args.kv_peak_ratio) {
        bail!("--kv-peak-ratio must be finite and between 0.0 and 1.0");
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

fn log_model_loaded(
    stats: &vasr::OfflineAsrBuildStats,
    #[allow(unused_variables)] device: &Device,
) {
    let params = stats
        .parameter_bytes
        .map(|bytes| format!("{:.2}GiB", bytes_to_gib(bytes)))
        .unwrap_or_else(|| "unknown".to_string());
    let kv_cache = stats
        .kv_cache_bytes
        .map(|bytes| {
            format!(
                "{:.2}GiB/{}tok",
                bytes_to_gib(bytes),
                stats.kv_cache_tokens.unwrap_or(0)
            )
        })
        .unwrap_or_else(|| "0.00GiB/0tok".to_string());
    let cuda_graph = stats
        .cuda_graph_bytes
        .map(|bytes| {
            format!(
                "{:.2}GiB/{}seqs/{}tok",
                bytes_to_gib(bytes),
                stats.cuda_graph_seqs.unwrap_or(0),
                stats.cuda_graph_context_tokens.unwrap_or(0)
            )
        })
        .unwrap_or_else(|| "0.00GiB/0seqs/0tok".to_string());
    let isq_quant = stats
        .quantize_seconds
        .map(|seconds| format!(" isq_quant={seconds:.3}s"))
        .unwrap_or_default();
    let isq_layers = stats
        .quantized_layer_counts
        .map(|(audio, text, other)| format!(" isq_layers=audio:{audio},text:{text},other:{other}"))
        .unwrap_or_default();
    #[cfg(all(feature = "cuda", feature = "paged-attn"))]
    if device.is_cuda() {
        if let Some(memory) = vasr::vasr_paged_attn::cuda_graph::cuda_memory_snapshot(device) {
            let gpu_used = memory.used();
            tracing::info!(
                target: "vasr_transcribe::serve",
                "loaded model | {elapsed_seconds:.3}s{isq_quant}{isq_layers} params={params} kv={kv_cache} graph={cuda_graph} gpu={:.2}/{:.2}GiB free={:.2}GiB",
                bytes_to_gib(gpu_used),
                bytes_to_gib(memory.total),
                bytes_to_gib(memory.free),
                elapsed_seconds = stats.build_seconds,
            );
            return;
        }
    }

    tracing::info!(
        target: "vasr_transcribe::serve",
        "loaded model | {:.3}s{isq_quant}{isq_layers} params={params} kv={kv_cache} graph={cuda_graph}",
        stats.build_seconds
    );
}

fn validate_vad_args(vad: &VadCliArgs) -> Result<()> {
    match vad.vad_model.trim().to_ascii_lowercase().as_str() {
        "fsmn" | "firered" => {}
        other => bail!("--vad-model must be fsmn or firered; got {other:?}"),
    }
    match vad.vad_device.trim().to_ascii_lowercase().as_str() {
        "auto" | "cpu" | "cuda" | "gpu" => {}
        other => bail!("--vad-device must be auto, cpu, or cuda; got {other:?}"),
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
    let dtype = auto_dtype(&device)?;

    tracing::info!(
        target: "vasr_transcribe::serve",
        "starting service | model={} device={} dtype={}",
        args.model,
        device_label(&device),
        dtype_label(dtype),
    );
    let output = vasr::build_offline_asr_model(vasr::OfflineAsrBuildRequest {
        model: args.model.clone(),
        cache_dir: args.cache_dir.clone(),
        device: device.clone(),
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
            decode_tokens_per_sec: args.decode_tokens_per_sec,
            decode_min_tokens: args.decode_min_tokens,
            #[cfg(feature = "cuda-graph")]
            cuda_graph_chunked_argmax: args.cuda_graph_chunked_argmax,
            #[cfg(feature = "cuda-graph")]
            cuda_graph_chunked_argmax_size: args.cuda_graph_chunked_argmax_size,
        },
        #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
        cuda_graph_max_batch,
    })?;
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    configure_cuda_mempool_release_threshold(&device);
    log_model_loaded(&output.stats, &device);
    let base_asr = output.model;
    if scheduled {
        let asr_scheduler = InferenceScheduler::start("asr");
        Ok(Arc::new(ScheduledAsrModel::new(
            Arc::clone(&base_asr),
            asr_scheduler.clone(),
        )))
    } else {
        Ok(base_asr)
    }
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
        "server ready | http://{addr} endpoints=health,transcribe,inference"
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
                if let Ok(device) = Device::new_cuda(0) {
                    return Ok(device);
                }
            }
            Ok(Device::Cpu)
        }
        "cpu" => Ok(Device::Cpu),
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
        other => bail!("--vad-device must be auto, cpu, or cuda; got {other:?}"),
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
