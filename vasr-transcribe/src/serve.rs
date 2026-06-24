use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::runtime::{
    AsrModel, AsrOptions, FsmnVadModel, InferenceScheduler, OfflinePipeline, ScheduledAsrModel,
    VadModel, VadOptions, auto_dtype, device_label, resolve_device,
};
use anyhow::{Result, bail};
use axum::{Json, Router, routing::get};
use candle_core::Device;
use clap::Args;
use vasr::models::qwen3asr::model::isq_linear::resolve_isq_display;
#[cfg(any(feature = "metal", feature = "cuda"))]
use vasr::models::qwen3asr::model::paged_cache_runtime::{
    PagedCacheConfig, PagedCacheMemory, PagedCacheStats,
};
use vasr::models::qwen3asr::{LoadOptions, Qwen3ASR};
use vasr_data::AudioLoader;

use crate::pipeline::AsyncTranscribePipeline;
use crate::server::build_transcribe_service_from_parts;

const DEFAULT_MAX_NUM_SEQS: usize = 256;
const DEFAULT_ASR_MODEL: &str = "Qwen/Qwen3-ASR-0.6B";

#[derive(Debug, Clone, Args)]
pub struct CommonModelArgs {
    /// Qwen3-ASR model: `Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B`, or a local directory.
    #[arg(long, default_value = DEFAULT_ASR_MODEL, env = "VASR_MODEL")]
    pub model: String,

    /// vASR model cache directory.
    #[arg(long = "cache-dir", default_value_os_t = vasr::modelscope_cache_dir())]
    pub cache_dir: PathBuf,

    /// Device: auto, cpu, metal, cuda.
    #[arg(long, default_value = "auto", env = "VASR_DEVICE")]
    pub device: String,

    /// Enable in-situ quantization, for example q8_0.
    #[arg(long, env = "VASR_ISQ")]
    pub isq: Option<String>,

    /// Maximum decode tokens per request.
    #[arg(long, default_value_t = 4096, env = "VASR_MAX_NEW_TOKENS")]
    pub max_new_tokens: usize,

    /// Target GPU memory fraction for model parameters and PagedAttention KV cache.
    #[arg(
        long = "gpu-memory-utilization",
        default_value_t = 0.65,
        env = "VASR_GPU_MEMORY_UTILIZATION"
    )]
    pub gpu_memory_utilization: f32,

    /// Maximum concurrent decode sequences. Also controls CUDA decode graph max seqs; use 0 to disable CUDA graph prewarm/replay.
    #[arg(
        long = "max-num-seqs",
        default_value_t = DEFAULT_MAX_NUM_SEQS
    )]
    pub max_num_seqs: usize,
}

#[derive(Debug, Clone, Args)]
pub struct VadCliArgs {
    /// FSMN VAD speech probability threshold (`speech_noise_thres`).
    #[arg(long, default_value_t = 0.6, env = "VASR_VAD_THRESHOLD")]
    pub vad_threshold: f32,

    /// Minimum speech duration in milliseconds before a segment is confirmed.
    #[arg(long, default_value_t = 250, env = "VASR_VAD_MIN_SPEECH_MS")]
    pub vad_min_speech_ms: u64,

    /// Minimum trailing silence in milliseconds before ending a speech segment.
    #[arg(long, default_value_t = 500, env = "VASR_VAD_MIN_SILENCE_MS")]
    pub vad_min_silence_ms: u64,

    /// Maximum offline VAD segment duration in seconds before ASR.
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

    /// Maximum total segment audio seconds per ASR micro-batch. Set 0 to disable.
    #[arg(long, default_value_t = 60.0, env = "VASR_MAX_BATCH_AUDIO_SEC")]
    pub max_batch_audio_sec: f32,

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
        "warn,vasr_transcribe=debug,vasr=debug"
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
    let asr = load_asr_model(
        &args.pipeline.model,
        cuda_graph_max_batch(&args.pipeline.model),
        false,
    )?;
    let offline = build_offline_pipeline(asr)?;
    let transcribe_service = build_transcribe_service_from_parts(
        offline,
        asr_options_from_pipeline(&args.pipeline, args.language.clone()),
        vad_options_from_pipeline(&args.pipeline),
    );

    let app = Router::new()
        .route("/health", get(health))
        .merge(crate::server::transcribe_router(transcribe_service));
    let addr = bind_addr(&args.host, args.port)?;
    serve_app(app, addr).await
}

pub fn build_async_transcribe_pipeline(
    pipeline: &TranscribePipelineArgs,
    language: Option<String>,
) -> Result<AsyncTranscribePipeline> {
    validate_pipeline(pipeline)?;
    let asr = load_asr_model(
        &pipeline.model,
        cuda_graph_max_batch(&pipeline.model),
        false,
    )?;
    let offline = build_offline_pipeline(asr)?;
    Ok(AsyncTranscribePipeline::new(
        AudioLoader,
        offline,
        asr_options_from_pipeline(pipeline, language),
    )
    .with_vad_options(vad_options_from_pipeline(pipeline)))
}

pub fn validate_pipeline(pipeline: &TranscribePipelineArgs) -> Result<()> {
    validate_model_args(&pipeline.model)?;
    validate_vad_args(&pipeline.vad)
}

fn build_offline_pipeline(asr: Arc<dyn AsrModel>) -> Result<Arc<OfflinePipeline>> {
    let vad_load_start = Instant::now();
    let vad = load_fsmn_vad()?;
    tracing::info!(
        target: "vasr_transcribe::serve",
        "FSMN VAD loaded from `{}` in {:.3}s.",
        vad.model_dir().display(),
        vad_load_start.elapsed().as_secs_f64()
    );
    Ok(Arc::new(OfflinePipeline {
        vad: Some(Arc::new(vad) as Arc<dyn VadModel>),
        asr,
    }))
}

fn asr_options_from_pipeline(
    pipeline: &TranscribePipelineArgs,
    language: Option<String>,
) -> AsrOptions {
    AsrOptions {
        language,
        max_new_tokens: pipeline.model.max_new_tokens,
        max_batch_audio_sec: pipeline.max_batch_audio_sec,
        chunk_max_sec: None,
        ..AsrOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_defaults_match_validated_vad_batch_bounds() {
        let pipeline = TranscribePipelineArgs {
            model: CommonModelArgs {
                model: DEFAULT_ASR_MODEL.to_string(),
                cache_dir: vasr::modelscope_cache_dir(),
                device: "auto".to_string(),
                isq: None,
                max_new_tokens: 4096,
                gpu_memory_utilization: 0.65,
                max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            },
            max_batch_audio_sec: AsrOptions::default().max_batch_audio_sec,
            vad: VadCliArgs {
                vad_threshold: 0.6,
                vad_min_speech_ms: 250,
                vad_min_silence_ms: 500,
                vad_max_segment_sec: 30.0,
            },
        };

        let opts = asr_options_from_pipeline(&pipeline, None);
        assert_eq!(opts.max_new_tokens, 4096);
        assert_eq!(opts.max_batch_audio_sec, 60.0);
    }
}

fn vad_options_from_args(vad: &VadCliArgs) -> VadOptions {
    VadOptions {
        threshold: vad.vad_threshold,
        min_speech_ms: vad.vad_min_speech_ms,
        min_silence_ms: vad.vad_min_silence_ms,
        max_segment_ms: (vad.vad_max_segment_sec * 1000.0).round() as u64,
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
    if args.max_new_tokens == 0 {
        bail!("--max-new-tokens must be greater than zero");
    }
    if !(args.gpu_memory_utilization > 0.0 && args.gpu_memory_utilization <= 1.0) {
        bail!("--gpu-memory-utilization must be in (0.0, 1.0]");
    }
    Ok(())
}

fn resolve_asr_model_id(model: &str) -> &str {
    match model.trim() {
        "Qwen3-ASR-0.6B" => "Qwen/Qwen3-ASR-0.6B",
        "Qwen3-ASR-1.7B" => "Qwen/Qwen3-ASR-1.7B",
        other => other,
    }
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn build_paged_cache_config(
    device: &Device,
    gpu_memory_utilization: f32,
) -> Result<PagedCacheConfig> {
    let memory = if device.is_cuda() {
        PagedCacheMemory::GpuMemoryFraction {
            fraction: gpu_memory_utilization,
            reserved_bytes: 0,
        }
    } else {
        PagedCacheMemory::ContextSize(4096)
    };
    Ok(PagedCacheConfig {
        block_size: 32,
        memory,
    })
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn log_paged_cache_stats(stats: &PagedCacheStats, config: &PagedCacheConfig) {
    let sizing = match config.memory {
        PagedCacheMemory::ContextSize(tokens) => format!("context_tokens={tokens}"),
        PagedCacheMemory::Blocks(blocks) => format!("blocks={blocks}"),
        PagedCacheMemory::GpuMemoryFraction {
            fraction,
            reserved_bytes: _,
        } => format!("gpu_memory_utilization={fraction}"),
    };
    tracing::debug!(
        target: "vasr_transcribe::serve",
        "PagedAttention KV cache: {sizing}, block_size={}, blocks={}, free_blocks={}, max_context_tokens={}, memory={:.2}GiB.",
        stats.block_size,
        stats.num_blocks,
        stats.free_blocks,
        stats.max_context_tokens,
        bytes_to_gib(stats.bytes)
    );
}

fn bytes_to_gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn log_model_loaded(
    elapsed_seconds: f64,
    params_bytes: Option<usize>,
    kv_cache_bytes: Option<usize>,
    kv_cache_tokens: Option<usize>,
    cuda_graph_bytes: Option<usize>,
    cuda_graph_seqs: Option<usize>,
    cuda_graph_context_tokens: Option<usize>,
    #[allow(unused_variables)] device: &Device,
) {
    let params = params_bytes
        .map(|bytes| format!("{:.2}GiB", bytes_to_gib(bytes)))
        .unwrap_or_else(|| "unknown".to_string());
    let kv_cache = kv_cache_bytes
        .map(|bytes| {
            format!(
                "{:.2}GiB/{}tok",
                bytes_to_gib(bytes),
                kv_cache_tokens.unwrap_or(0)
            )
        })
        .unwrap_or_else(|| "0.00GiB/0tok".to_string());
    let cuda_graph = cuda_graph_bytes
        .map(|bytes| {
            format!(
                "{:.2}GiB/{}seqs/{}tok",
                bytes_to_gib(bytes),
                cuda_graph_seqs.unwrap_or(0),
                cuda_graph_context_tokens.unwrap_or(0)
            )
        })
        .unwrap_or_else(|| "0.00GiB/0seqs/0tok".to_string());
    let fixed_bytes = params_bytes
        .unwrap_or(0)
        .saturating_add(kv_cache_bytes.unwrap_or(0))
        .saturating_add(cuda_graph_bytes.unwrap_or(0));

    #[cfg(all(feature = "cuda", feature = "paged-attn"))]
    if device.is_cuda() {
        if let Some(memory) = vasr::vasr_paged_attn::cuda_graph::cuda_memory_snapshot(device) {
            let fixed_pct = percentage(fixed_bytes, memory.total);
            let gpu_used = memory.used();
            let gpu_used_pct = percentage(gpu_used, memory.total);
            tracing::info!(
                target: "vasr_transcribe::serve",
                "Model loaded in {elapsed_seconds:.3}s | params={params} kv_cache={kv_cache} cuda_graph={cuda_graph} fixed={:.2}GiB/{fixed_pct:.1}% gpu_used={:.2}GiB/{gpu_used_pct:.1}% free={:.2}GiB total={:.2}GiB.",
                bytes_to_gib(fixed_bytes),
                bytes_to_gib(gpu_used),
                bytes_to_gib(memory.free),
                bytes_to_gib(memory.total),
            );
            return;
        }
    }

    tracing::info!(
        target: "vasr_transcribe::serve",
        "Model loaded in {elapsed_seconds:.3}s | params={params} kv_cache={kv_cache} cuda_graph={cuda_graph} fixed={:.2}GiB.",
        bytes_to_gib(fixed_bytes)
    );
}

#[cfg(all(feature = "cuda", feature = "paged-attn"))]
fn percentage(bytes: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    bytes as f64 * 100.0 / total as f64
}

fn validate_vad_args(vad: &VadCliArgs) -> Result<()> {
    if !(0.0..=1.0).contains(&vad.vad_threshold) {
        bail!("--vad-threshold must be between 0.0 and 1.0");
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
    Some(args.max_num_seqs)
}

#[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
fn cuda_graph_max_batch(_args: &CommonModelArgs) -> Option<usize> {
    None
}

fn load_asr_model(
    args: &CommonModelArgs,
    #[cfg_attr(
        not(all(feature = "cuda-graph", feature = "paged-attn")),
        allow(unused_variables)
    )]
    cuda_graph_max_batch: Option<usize>,
    scheduled: bool,
) -> Result<Arc<dyn AsrModel>> {
    vasr::set_cache_dir(args.cache_dir.clone());
    let device = resolve_device(&args.device)?;
    let dtype = auto_dtype(&device)?;
    #[cfg(feature = "paged-attn")]
    let paged_cache_config = {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            build_paged_cache_config(&device, args.gpu_memory_utilization)?
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            vasr::PagedCacheConfig {
                block_size: 32,
                memory: vasr::PagedCacheMemory::ContextSize(100_000),
            }
        }
    };
    let load_options = LoadOptions {
        dtype,
        use_flash_attn: device.is_cuda(),
        isq: args.isq.clone(),
        #[cfg(feature = "paged-attn")]
        paged_cache: Some(paged_cache_config),
    };

    tracing::info!(
        target: "vasr_transcribe::serve",
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
        cfg!(target_feature = "f16c")
    );
    tracing::info!(
        target: "vasr_transcribe::serve",
        "Model kind is: qwen3-asr (no adapters)"
    );
    tracing::info!(
        target: "vasr_transcribe::serve",
        "Auto-selected DType {:?} for {}.",
        dtype,
        device_label(&device)
    );
    if let Some(isq) = args.isq.as_deref() {
        tracing::info!(
            target: "vasr_transcribe::serve",
            "ISQ selected is {} (requested={}, backend={}).",
            resolve_isq_display(isq, &device)?,
            isq,
            device_label(&device)
        );
    }
    tracing::info!(
        target: "vasr_transcribe::serve",
        "Loading Qwen3-ASR model `{}` from cache `{}` on {} (isq={:?}).",
        resolve_asr_model_id(&args.model),
        args.cache_dir.display(),
        device_label(&device),
        args.isq
    );
    let model_load_start = Instant::now();
    let qwen3_asr =
        Qwen3ASR::from_pretrained(resolve_asr_model_id(&args.model), &device, &load_options)?;
    let params_bytes = match qwen3_asr.weights_memory_stats(dtype) {
        Ok(stats) => Some(stats.parameter_bytes),
        Err(err) => {
            tracing::debug!(
                target: "vasr_transcribe::serve",
                "model parameter memory stats unavailable: {err}"
            );
            None
        }
    };
    log_model_config(&qwen3_asr);
    #[cfg(feature = "paged-attn")]
    let paged_cache_stats = qwen3_asr.paged_cache_stats();
    #[cfg(feature = "paged-attn")]
    let kv_cache_bytes = paged_cache_stats.as_ref().map(|stats| stats.bytes);
    #[cfg(feature = "paged-attn")]
    let kv_cache_tokens = paged_cache_stats
        .as_ref()
        .map(|stats| stats.max_context_tokens);
    #[cfg(not(feature = "paged-attn"))]
    let kv_cache_bytes = None;
    #[cfg(not(feature = "paged-attn"))]
    let kv_cache_tokens = None;
    #[cfg(any(feature = "metal", feature = "cuda"))]
    if let Some(stats) = paged_cache_stats.as_ref() {
        log_paged_cache_stats(stats, &paged_cache_config);
    }
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    let mut cuda_graph_bytes = None;
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    let mut cuda_graph_seqs = None;
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    let mut cuda_graph_context_tokens = None;
    #[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
    let cuda_graph_bytes = None;
    #[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
    let cuda_graph_seqs = None;
    #[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
    let cuda_graph_context_tokens = None;
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    if let Some(max_batch) = cuda_graph_max_batch.filter(|&n| n > 0) {
        let prewarm_start = Instant::now();
        let stats = qwen3_asr.prewarm_cuda_decode_graphs(max_batch)?;
        if stats.graphs > 0 {
            cuda_graph_seqs = Some(stats.max_batch);
            cuda_graph_context_tokens = Some(stats.context_tokens);
        }
        if stats.graphs > 0 {
            if let (Some(before), Some(after)) = (stats.memory_before, stats.memory_after) {
                let used_before = before.used();
                let used_after = after.used();
                let mem_delta = used_after.saturating_sub(used_before);
                cuda_graph_bytes = Some(mem_delta);
                tracing::debug!(
                    target: "vasr_transcribe::serve",
                    "CUDA decode graph prewarm: graphs={} max_batch={} context_tokens={} requested_max_batch={max_batch} elapsed={:.3}s mem_delta={:.2}GiB used_before={:.2}GiB used_after={:.2}GiB free_after={:.2}GiB total={:.2}GiB.",
                    stats.graphs,
                    stats.max_batch,
                    stats.context_tokens,
                    prewarm_start.elapsed().as_secs_f64(),
                    bytes_to_gib(mem_delta),
                    bytes_to_gib(used_before),
                    bytes_to_gib(used_after),
                    bytes_to_gib(after.free),
                    bytes_to_gib(after.total),
                );
            } else {
                tracing::debug!(
                    target: "vasr_transcribe::serve",
                    "CUDA decode graph prewarm: graphs={} max_batch={} context_tokens={} requested_max_batch={max_batch} elapsed={:.3}s.",
                    stats.graphs,
                    stats.max_batch,
                    stats.context_tokens,
                    prewarm_start.elapsed().as_secs_f64()
                );
            }
        } else {
            tracing::debug!(
                target: "vasr_transcribe::serve",
                "CUDA decode graph prewarm disabled after failed capture; requested_max_batch={max_batch} elapsed={:.3}s.",
                prewarm_start.elapsed().as_secs_f64()
            );
        }
    }
    log_model_loaded(
        model_load_start.elapsed().as_secs_f64(),
        params_bytes,
        kv_cache_bytes,
        kv_cache_tokens,
        cuda_graph_bytes,
        cuda_graph_seqs,
        cuda_graph_context_tokens,
        &device,
    );
    let base_asr: Arc<dyn AsrModel> = Arc::new(qwen3_asr);
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

fn bind_addr(host: &str, port: u16) -> Result<SocketAddr> {
    Ok(format!("{host}:{port}").parse()?)
}

async fn serve_app(app: Router, addr: SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        target: "vasr_transcribe::serve",
        "HTTP endpoints: GET /health, POST /transcribe, POST /inference"
    );
    tracing::info!(
        target: "vasr_transcribe::serve",
        "vASR transcribe service listening on http://{addr}"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn load_fsmn_vad() -> Result<FsmnVadModel> {
    FsmnVadModel::from_pretrained(None::<&std::path::Path>)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn log_model_config(model: &Qwen3ASR) {
    let config = model.config();
    let text = &config.thinker_config.text_config;
    let audio = &config.thinker_config.audio_config;
    tracing::info!(
        target: "vasr_transcribe::serve",
        "Model config: model_type={:?}, thinker_type={:?}, text_layers={}, hidden_size={}, kv_heads={}, audio_layers={}, audio_dim={}",
        config.model_type,
        config.thinker_config.model_type,
        text.num_hidden_layers,
        text.hidden_size,
        text.num_key_value_heads,
        audio.encoder_layers,
        audio.d_model
    );
}
