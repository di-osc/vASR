use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use crate::runtime::{
    AsrModel, AsrOptions, FsmnVadModel, InferenceScheduler, OfflinePipeline, Qwen3AsrModel,
    ScheduledAsrModel, VadModel, VadOptions, auto_dtype, device_label, resolve_device,
};
use anyhow::{Result, bail};
use axum::{Json, Router, routing::get};
#[cfg(any(feature = "metal", feature = "cuda"))]
use candle_core::Device;
use clap::Args;
use vasr_data::AudioLoader;
use vasr_models::asr::qwen3asr::LoadOptions;
use vasr_models::asr::qwen3asr::model::isq_linear::resolve_isq_display;
#[cfg(any(feature = "metal", feature = "cuda"))]
use vasr_models::asr::qwen3asr::model::paged_cache_runtime::{
    PagedCacheConfig, PagedCacheMemory, PagedCacheStats,
};

use crate::pipeline::AsyncTranscribePipeline;
use crate::server::build_transcribe_service_from_parts;

#[derive(Debug, Clone, Args)]
pub struct CommonModelArgs {
    /// Qwen3-ASR model: `Qwen/Qwen3-ASR-0.6B` or `Qwen/Qwen3-ASR-1.7B`, or a local directory.
    #[arg(long, default_value = "Qwen/Qwen3-ASR-0.6B", env = "VASR_MODEL")]
    pub model: String,

    /// Device: auto, cpu, metal, cuda.
    #[arg(long, default_value = "auto", env = "VASR_DEVICE")]
    pub device: String,

    /// Enable in-situ quantization, for example q8_0.
    #[arg(long, env = "VASR_ISQ")]
    pub isq: Option<String>,

    /// Maximum decode tokens per request.
    #[arg(long, default_value_t = 128, env = "VASR_MAX_NEW_TOKENS")]
    pub max_new_tokens: usize,

    /// Target fraction of total GPU memory usable by the process. PagedAttention KV cache uses the remaining budget after loaded model memory and an internal margin.
    #[arg(
        long = "gpu-memory-utilization",
        default_value_t = 0.65,
        env = "VASR_GPU_MEMORY_UTILIZATION"
    )]
    pub gpu_memory_utilization: f32,
}

#[derive(Debug, Clone, Args)]
pub struct VadCliArgs {
    /// FunASR FSMN VAD model directory. If omitted, vASR downloads/uses `funasr/fsmn-vad`.
    #[arg(long, env = "VASR_VAD_MODEL")]
    pub vad_model: Option<String>,

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

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1", env = "VASR_HOST")]
    pub host: String,

    /// Bind port.
    #[arg(long, default_value_t = 8000, env = "VASR_PORT")]
    pub port: u16,
}

/// Backward-compatible alias for the HTTP serve entry point.
pub type TranscribeArgs = ServeTranscribeArgs;

pub fn init_logging(verbose: bool) {
    let default = if verbose {
        "warn,vasr_transcribe=debug,vasr_models=debug"
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
    let asr = load_asr_model(&args.pipeline.model, default_cuda_graph_max_batch(), true)?;
    let offline = build_offline_pipeline(&args.pipeline, asr)?;
    let transcribe_service = build_transcribe_service_from_parts(
        offline,
        asr_options_from_pipeline(&args.pipeline, None),
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
    let asr = load_asr_model(&pipeline.model, default_cuda_graph_max_batch(), false)?;
    let offline = build_offline_pipeline(pipeline, asr)?;
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

fn build_offline_pipeline(
    pipeline: &TranscribePipelineArgs,
    asr: Arc<dyn AsrModel>,
) -> Result<Arc<OfflinePipeline>> {
    let vad_load_start = Instant::now();
    let vad = load_fsmn_vad(&pipeline.vad.vad_model)?;
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
                model: "Qwen/Qwen3-ASR-0.6B".to_string(),
                device: "auto".to_string(),
                isq: None,
                max_new_tokens: 128,
                gpu_memory_utilization: 0.65,
            },
            max_batch_audio_sec: AsrOptions::default().max_batch_audio_sec,
            vad: VadCliArgs {
                vad_model: None,
                vad_threshold: 0.6,
                vad_min_speech_ms: 250,
                vad_min_silence_ms: 500,
                vad_max_segment_sec: 30.0,
            },
        };

        let opts = asr_options_from_pipeline(&pipeline, None);
        assert_eq!(opts.max_new_tokens, 128);
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
    if args.max_new_tokens == 0 {
        bail!("--max-new-tokens must be greater than zero");
    }
    if !(args.gpu_memory_utilization > 0.0 && args.gpu_memory_utilization <= 1.0) {
        bail!("--gpu-memory-utilization must be in (0.0, 1.0]");
    }
    Ok(())
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn build_paged_cache_config(
    device: &Device,
    gpu_memory_utilization: f32,
) -> Result<PagedCacheConfig> {
    let memory = if device.is_cuda() {
        PagedCacheMemory::GpuMemoryFraction(gpu_memory_utilization)
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
        PagedCacheMemory::GpuMemoryFraction(fraction) => {
            format!("gpu_memory_utilization={fraction}")
        }
    };
    tracing::info!(
        target: "vasr_transcribe::serve",
        "PagedAttention KV cache: {sizing}, block_size={}, blocks={}, free_blocks={}, max_context_tokens={}, bytes={:.2} MiB.",
        stats.block_size,
        stats.num_blocks,
        stats.free_blocks,
        stats.max_context_tokens,
        stats.bytes as f64 / (1024.0 * 1024.0)
    );
}

#[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
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
fn default_cuda_graph_max_batch() -> Option<usize> {
    if std::env::var_os("VASR_DISABLE_CUDA_GRAPH").is_some() {
        tracing::info!(
            target: "vasr_transcribe::serve",
            "CUDA decode graph disabled by VASR_DISABLE_CUDA_GRAPH."
        );
        return None;
    }
    Some(vasr_models::vasr_paged_attn::CUDA_DECODE_GRAPH_MAX_BATCH)
}

#[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
fn default_cuda_graph_max_batch() -> Option<usize> {
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
            vasr_models::PagedCacheConfig {
                block_size: 32,
                memory: vasr_models::PagedCacheMemory::ContextSize(100_000),
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
        "Loading Qwen3-ASR model `{}` on {} (isq={:?}).",
        args.model,
        device_label(&device),
        args.isq
    );
    let model_load_start = Instant::now();
    let qwen3_asr = Qwen3AsrModel::from_pretrained(&args.model, &device, &load_options)?;
    log_model_config(&qwen3_asr);
    #[cfg(any(feature = "metal", feature = "cuda"))]
    if let Some(stats) = qwen3_asr.inner().paged_cache_stats() {
        log_paged_cache_stats(&stats, &paged_cache_config);
    }
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    if let Some(max_batch) = cuda_graph_max_batch.filter(|&n| n > 0) {
        let prewarm_start = Instant::now();
        let stats = qwen3_asr.inner().prewarm_cuda_decode_graphs(max_batch)?;
        if let (Some(before), Some(after)) = (stats.memory_before, stats.memory_after) {
            let used_before = before.used();
            let used_after = after.used();
            tracing::info!(
                target: "vasr_transcribe::serve",
                "CUDA decode graph prewarm: graphs={} max_batch={max_batch} elapsed={:.3}s mem_delta={:.2}MiB used_before={:.2}MiB used_after={:.2}MiB free_after={:.2}MiB total={:.2}MiB.",
                stats.graphs,
                prewarm_start.elapsed().as_secs_f64(),
                bytes_to_mib(used_after.saturating_sub(used_before)),
                bytes_to_mib(used_before),
                bytes_to_mib(used_after),
                bytes_to_mib(after.free),
                bytes_to_mib(after.total),
            );
        } else {
            tracing::info!(
                target: "vasr_transcribe::serve",
                "CUDA decode graph prewarm: graphs={} max_batch={max_batch} elapsed={:.3}s.",
                stats.graphs,
                prewarm_start.elapsed().as_secs_f64()
            );
        }
    }
    tracing::info!(
        target: "vasr_transcribe::serve",
        "Model loaded in {:.3}s.",
        model_load_start.elapsed().as_secs_f64()
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

fn load_fsmn_vad(path: &Option<String>) -> Result<FsmnVadModel> {
    FsmnVadModel::from_pretrained(path.as_deref())
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn log_model_config(model: &Qwen3AsrModel) {
    let config = model.inner().config();
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
