use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use crate::runtime::{
    AsrModel, AsrOptions, FsmnVadModel, Qwen3AsrModel, RealtimePipeline, VadModel, VadOptions,
    auto_dtype, device_label, resolve_device,
};
use anyhow::{Result, bail};
use axum::{Json, Router, routing::get};
#[cfg(any(feature = "metal", feature = "cuda"))]
use candle_core::Device;
use clap::Args;
use vasr::models::qwen3asr::LoadOptions;
use vasr::models::qwen3asr::model::isq_linear::resolve_isq_display;
#[cfg(any(feature = "metal", feature = "cuda"))]
use vasr::models::qwen3asr::model::paged_cache_runtime::{
    PagedCacheConfig, PagedCacheMemory, PagedCacheStats,
};

use crate::server::{RealtimeService, RealtimeSession};

#[cfg(feature = "cuda-graph")]
const DEFAULT_CUDA_GRAPH_MAX_SEQS: usize = vasr::vasr_paged_attn::CUDA_DECODE_GRAPH_MAX_BATCH;
#[cfg(not(feature = "cuda-graph"))]
const DEFAULT_CUDA_GRAPH_MAX_SEQS: usize = 512;

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

    /// Target GPU memory fraction for model parameters and PagedAttention KV cache.
    #[arg(
        long = "gpu-memory-utilization",
        default_value_t = 0.65,
        env = "VASR_GPU_MEMORY_UTILIZATION"
    )]
    pub gpu_memory_utilization: f32,

    /// Maximum CUDA decode graph batch width. Use 0 to disable CUDA graph prewarm/replay.
    #[arg(
        long = "cuda-graph-max-seqs",
        default_value_t = DEFAULT_CUDA_GRAPH_MAX_SEQS,
        env = "VASR_CUDA_GRAPH_MAX_SEQS"
    )]
    pub cuda_graph_max_seqs: usize,
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
}

#[derive(Debug, Clone, Args)]
pub struct RealtimeArgs {
    #[command(flatten)]
    pub model: CommonModelArgs,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1", env = "VASR_HOST")]
    pub host: String,

    /// Bind port.
    #[arg(long, default_value_t = 8000, env = "VASR_PORT")]
    pub port: u16,

    #[command(flatten)]
    pub vad: VadCliArgs,
}

pub fn init_logging(verbose: bool) {
    let default = if verbose {
        "warn,vasr_realtime=info"
    } else {
        "warn,vasr_realtime=info"
    };
    let filter = std::env::var("VASR_LOG").unwrap_or_else(|_| default.to_string());

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
        .with_timer(tracing_subscriber::fmt::time::SystemTime)
        .with_target(true)
        .compact()
        .init();
}

fn vad_options_from_args(vad: &VadCliArgs) -> VadOptions {
    VadOptions {
        threshold: vad.vad_threshold,
        min_speech_ms: vad.vad_min_speech_ms,
        min_silence_ms: vad.vad_min_silence_ms,
        ..VadOptions::default()
    }
}

pub async fn run_realtime(args: RealtimeArgs) -> Result<()> {
    validate_model_args(&args.model)?;
    validate_vad_args(&args.vad)?;
    let asr = load_asr_model(&args.model, cuda_graph_max_batch(&args.model))?;

    let realtime_asr = Arc::clone(&asr);
    let vad_model = args.vad.vad_model.clone();
    let vad_options = vad_options_from_args(&args.vad);
    let realtime_service = Arc::new(RealtimeService {
        make_session: Arc::new(move || {
            let vad = load_fsmn_vad(&vad_model)?.new_stream(&vad_options)?;
            let asr_stream = realtime_asr.new_stream(&AsrOptions {
                ..AsrOptions::default()
            })?;
            Ok(RealtimeSession::new(
                16_000,
                RealtimePipeline {
                    vad,
                    asr: asr_stream,
                },
            ))
        }),
    });

    let app = Router::new()
        .route("/health", get(health))
        .merge(crate::server::realtime_router(realtime_service));
    let addr = bind_addr(&args.host, args.port)?;
    serve_app(app, addr).await
}

fn validate_model_args(args: &CommonModelArgs) -> Result<()> {
    if args.model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if !(args.gpu_memory_utilization > 0.0 && args.gpu_memory_utilization <= 1.0) {
        bail!("--gpu-memory-utilization must be in (0.0, 1.0]");
    }
    if args.cuda_graph_max_seqs > DEFAULT_CUDA_GRAPH_MAX_SEQS {
        bail!(
            "--cuda-graph-max-seqs must be <= {} (use 0 to disable)",
            DEFAULT_CUDA_GRAPH_MAX_SEQS
        );
    }
    Ok(())
}

#[cfg(feature = "cuda-graph")]
fn cuda_graph_max_batch(args: &CommonModelArgs) -> Option<usize> {
    if std::env::var_os("VASR_DISABLE_CUDA_GRAPH").is_some() {
        tracing::info!(
            target: "vasr_realtime::serve",
            "CUDA decode graph disabled by VASR_DISABLE_CUDA_GRAPH."
        );
        return None;
    }
    if args.cuda_graph_max_seqs == 0 {
        tracing::info!(
            target: "vasr_realtime::serve",
            "CUDA decode graph disabled by --cuda-graph-max-seqs=0."
        );
        return None;
    }
    Some(args.cuda_graph_max_seqs)
}

#[cfg(not(feature = "cuda-graph"))]
fn cuda_graph_max_batch(_args: &CommonModelArgs) -> Option<usize> {
    None
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
        ..PagedCacheConfig::default()
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
    tracing::info!(
        target: "vasr_realtime::serve",
        "PagedAttention KV cache: {sizing}, block_size={}, blocks={}, free_blocks={}, max_context_tokens={}, memory={:.2}GiB.",
        stats.block_size,
        stats.num_blocks,
        stats.free_blocks,
        stats.max_context_tokens,
        bytes_to_gib(stats.bytes)
    );
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn bytes_to_gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
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
    Ok(())
}

fn load_asr_model(
    args: &CommonModelArgs,
    #[cfg_attr(not(feature = "cuda-graph"), allow(unused_variables))] cuda_graph_max_batch: Option<
        usize,
    >,
) -> Result<Arc<dyn AsrModel>> {
    let device = resolve_device(&args.device)?;
    let dtype = auto_dtype(&device)?;
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let paged_cache_config = build_paged_cache_config(&device, args.gpu_memory_utilization)?;
    let load_options = LoadOptions {
        dtype,
        use_flash_attn: device.is_cuda(),
        isq: args.isq.clone(),
        #[cfg(any(feature = "metal", feature = "cuda"))]
        paged_cache: Some(paged_cache_config),
        ..LoadOptions::default()
    };

    tracing::info!(
        target: "vasr_realtime::serve",
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
        cfg!(target_feature = "f16c")
    );
    tracing::info!(
        target: "vasr_realtime::serve",
        "Model kind is: qwen3-asr (no adapters)"
    );
    tracing::info!(
        target: "vasr_realtime::serve",
        "Auto-selected DType {:?} for {}.",
        dtype,
        device_label(&device)
    );
    if let Some(isq) = args.isq.as_deref() {
        tracing::info!(
            target: "vasr_realtime::serve",
            "ISQ selected is {} (requested={}, backend={}).",
            resolve_isq_display(isq, &device)?,
            isq,
            device_label(&device)
        );
    }
    tracing::info!(
        target: "vasr_realtime::serve",
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
    #[cfg(feature = "cuda-graph")]
    if let Some(max_batch) = cuda_graph_max_batch.filter(|&n| n > 0) {
        let prewarm_start = Instant::now();
        let stats = qwen3_asr.inner().prewarm_cuda_decode_graphs(max_batch)?;
        if let (Some(before), Some(after)) = (stats.memory_before, stats.memory_after) {
            let used_before = before.used();
            let used_after = after.used();
            tracing::info!(
                target: "vasr_realtime::serve",
                "CUDA decode graph prewarm: graphs={} max_batch={} requested_max_batch={max_batch} elapsed={:.3}s mem_delta={:.2}MiB used_before={:.2}MiB used_after={:.2}MiB free_after={:.2}MiB total={:.2}MiB.",
                stats.graphs,
                stats.max_batch,
                prewarm_start.elapsed().as_secs_f64(),
                bytes_to_mib(used_after.saturating_sub(used_before)),
                bytes_to_mib(used_before),
                bytes_to_mib(used_after),
                bytes_to_mib(after.free),
                bytes_to_mib(after.total),
            );
        } else {
            tracing::info!(
                target: "vasr_realtime::serve",
                "CUDA decode graph prewarm: graphs={} max_batch={} requested_max_batch={max_batch} elapsed={:.3}s.",
                stats.graphs,
                stats.max_batch,
                prewarm_start.elapsed().as_secs_f64()
            );
        }
    }
    tracing::info!(
        target: "vasr_realtime::serve",
        "Model loaded in {:.3}s.",
        model_load_start.elapsed().as_secs_f64()
    );
    Ok(Arc::new(qwen3_asr))
}

fn bind_addr(host: &str, port: u16) -> Result<SocketAddr> {
    Ok(format!("{host}:{port}").parse()?)
}

async fn serve_app(app: Router, addr: SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        target: "vasr_realtime::serve",
        "WebSocket endpoints: /v1/realtime, /api-ws/v1/realtime"
    );
    tracing::info!(
        target: "vasr_realtime::serve",
        "vASR realtime service listening on http://{addr}"
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
        target: "vasr_realtime::serve",
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
