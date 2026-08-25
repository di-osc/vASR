use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::config::{AsrModelConfig, VadModelConfig, VasrConfig};
use crate::runtime::{
    AsrModel, AsrOptions, FireRedVadModel, Qwen3AsrModel, VadModel, VadOptions, auto_dtype,
    device_label, resolve_device,
};
use anyhow::{Result, bail};
use axum::{Json, Router, routing::get};
#[cfg(any(feature = "metal", feature = "cuda"))]
use candle_core::Device;
use clap::Args;
use vasr::OfflineAsrModel;
use vasr::models::qwen3asr::LoadOptions;
use vasr::models::qwen3asr::model::isq_linear::resolve_isq_display;
#[cfg(any(feature = "metal", feature = "cuda"))]
use vasr::models::qwen3asr::model::paged_cache_runtime::{
    PagedCacheConfig, PagedCacheMemory, PagedCacheStats,
};
use vasr::models::sensevoice::{SenseVoiceLoadOptions, SenseVoiceOnline};

use crate::server::{RealtimeService, RealtimeSession};

#[cfg(feature = "cuda-graph")]
const DEFAULT_CUDA_GRAPH_MAX_SEQS: usize = vasr::vasr_paged_attn::CUDA_DECODE_GRAPH_MAX_BATCH;
#[cfg(not(feature = "cuda-graph"))]
const DEFAULT_CUDA_GRAPH_MAX_SEQS: usize = 512;

#[derive(Debug, Clone, Args)]
pub struct CommonModelArgs {
    /// Model type: auto, qwen3-asr, or sensevoice-online.
    #[arg(long, default_value = "auto", env = "VASR_MODEL_KIND")]
    pub model_kind: String,

    /// Qwen3-ASR model: `Qwen/Qwen3-ASR-0.6B` or `Qwen/Qwen3-ASR-1.7B`, or a local directory.
    #[arg(long, default_value = "Qwen/Qwen3-ASR-0.6B", env = "VASR_MODEL")]
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

    /// Recognition language. Qwen auto-detects when omitted; SenseVoice defaults to zh.
    #[arg(long, env = "VASR_LANGUAGE")]
    pub language: Option<String>,

    /// Newly available 60ms LFR frames required for a SenseVoice decode.
    #[arg(long, default_value_t = 4, env = "VASR_SENSEVOICE_CHUNK_SIZE")]
    pub sensevoice_chunk_size: usize,

    /// Maximum retained SenseVoice LFR frames. Zero keeps the full history.
    #[arg(long, default_value_t = 0, env = "VASR_SENSEVOICE_MAX_HISTORY")]
    pub sensevoice_max_history: usize,

    /// Enable the SenseVoice inverse-text-normalization query.
    #[arg(long, env = "VASR_SENSEVOICE_TEXTNORM")]
    pub sensevoice_textnorm: bool,

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
pub struct RealtimeArgs {
    #[command(flatten)]
    pub model: CommonModelArgs,

    /// Bind host.
    #[arg(long, default_value = "127.0.0.1", env = "VASR_HOST")]
    pub host: String,

    /// Bind port.
    #[arg(long, default_value_t = 8000, env = "VASR_PORT")]
    pub port: u16,
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

fn realtime_vad_options() -> VadOptions {
    VadOptions {
        threshold: 0.5,
        min_speech_ms: 80,
        min_silence_ms: 500,
        max_segment_ms: 20_000,
        pad_ms: 50,
        num_threads: 0,
    }
}

pub async fn run_realtime(args: RealtimeArgs) -> Result<()> {
    validate_model_args(&args.model)?;
    vasr::set_cache_dir(args.model.cache_dir.clone());
    let asr = load_asr_model(&args.model, cuda_graph_max_batch(&args.model))?;
    let vad: Arc<dyn VadModel> = Arc::new(load_firered_vad()?);

    let realtime_asr = Arc::clone(&asr);
    let realtime_vad = Arc::clone(&vad);
    let asr_language = args.model.language.clone();
    let vad_options = realtime_vad_options();
    let realtime_service = Arc::new(RealtimeService {
        make_session: Arc::new(move || {
            RealtimeSession::from_models(
                16_000,
                Arc::clone(&realtime_vad),
                Arc::clone(&realtime_asr),
                vad_options.clone(),
                AsrOptions {
                    language: asr_language.clone(),
                    ..AsrOptions::default()
                },
            )
        }),
    });

    let app = Router::new()
        .route("/health", get(health))
        .merge(crate::server::realtime_router(realtime_service));
    let addr = bind_addr(&args.host, args.port)?;
    serve_app(app, addr).await
}

pub async fn run_realtime_config(config: VasrConfig) -> Result<()> {
    if config.version != 1 {
        bail!(
            "unsupported config version {}; expected version = 1",
            config.version
        );
    }
    let cache_dir = config
        .models
        .cache_dir
        .clone()
        .unwrap_or_else(vasr::modelscope_cache_dir);
    if cache_dir.as_os_str().is_empty() {
        bail!("models.cache_dir must not be empty");
    }
    vasr::set_cache_dir(cache_dir.clone());
    validate_vad_config(&config.models.vad)?;
    let online_args = model_args_from_config(&config.models.online, cache_dir.clone());
    let offline_args = model_args_from_config(&config.models.offline, cache_dir);
    validate_model_args(&online_args)?;
    validate_model_args(&offline_args)?;
    if resolve_model_kind(&offline_args)? != RealtimeModelKind::Qwen3Asr {
        bail!(
            "models.offline.kind {:?} does not support offline transcription; expected qwen3-asr",
            offline_args.model_kind
        );
    }

    let vad: Arc<dyn VadModel> = Arc::new(load_firered_vad_config(&config.models.vad)?);
    let online_asr = load_asr_model(&online_args, cuda_graph_max_batch(&online_args))?;
    let offline_asr = load_offline_asr_model(&offline_args, cuda_graph_max_batch(&offline_args))?;
    let vad_options = vad_options_from_config(&config.models.vad);
    let online_options = AsrOptions {
        language: config.models.online.language.clone(),
        ..AsrOptions::default()
    };
    let offline_options = AsrOptions {
        language: config.models.offline.language.clone(),
        ..AsrOptions::default()
    };

    tracing::info!(
        target: "vasr_realtime::serve",
        "Starting realtime 2-pass service: vad={}, online={}, offline={}.",
        config.models.vad.kind,
        config.models.online.kind,
        config.models.offline.kind,
    );
    let service = Arc::new(RealtimeService {
        make_session: Arc::new(move || {
            RealtimeSession::from_two_pass_models(
                16_000,
                Arc::clone(&vad),
                Arc::clone(&online_asr),
                Some(Arc::clone(&offline_asr)),
                vad_options.clone(),
                online_options.clone(),
                offline_options.clone(),
            )
        }),
    });
    let app = Router::new()
        .route("/health", get(health))
        .merge(crate::server::realtime_router(service));
    let addr = bind_addr(&config.service.host, config.service.port)?;
    serve_app(app, addr).await
}

fn model_args_from_config(config: &AsrModelConfig, cache_dir: PathBuf) -> CommonModelArgs {
    CommonModelArgs {
        model_kind: config.kind.clone(),
        model: config.checkpoint.clone(),
        cache_dir,
        device: config.device.clone(),
        isq: config.isq.clone(),
        language: config.language.clone(),
        sensevoice_chunk_size: config.chunk_size,
        sensevoice_max_history: config.max_history,
        sensevoice_textnorm: config.textnorm,
        gpu_memory_utilization: config.gpu_memory_utilization,
        cuda_graph_max_seqs: config.cuda_graph_max_seqs,
    }
}

fn validate_vad_config(config: &VadModelConfig) -> Result<()> {
    if !matches!(
        config.kind.trim().to_ascii_lowercase().as_str(),
        "firered" | "firered-online" | "firered_online"
    ) {
        bail!(
            "unsupported models.vad.kind {:?}; expected firered",
            config.kind
        );
    }
    if !(0.0..=1.0).contains(&config.threshold) {
        bail!("models.vad.threshold must be in [0.0, 1.0]");
    }
    if config.min_speech_ms == 0 || config.min_silence_ms == 0 || config.max_segment_ms == 0 {
        bail!("models.vad durations must be greater than zero");
    }
    Ok(())
}

fn vad_options_from_config(config: &VadModelConfig) -> VadOptions {
    VadOptions {
        threshold: config.threshold,
        min_speech_ms: config.min_speech_ms,
        min_silence_ms: config.min_silence_ms,
        max_segment_ms: config.max_segment_ms,
        pad_ms: config.prefix_padding_ms,
        num_threads: config.num_threads,
    }
}

fn validate_model_args(args: &CommonModelArgs) -> Result<()> {
    if args.model.trim().is_empty() {
        bail!("--model must not be empty");
    }
    if args.cache_dir.as_os_str().is_empty() {
        bail!("--cache-dir must not be empty");
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
    resolve_model_kind(args)?;
    if args.sensevoice_chunk_size == 0 {
        bail!("--sensevoice-chunk-size must be greater than zero");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RealtimeModelKind {
    Qwen3Asr,
    SenseVoiceOnline,
}

fn resolve_model_kind(args: &CommonModelArgs) -> Result<RealtimeModelKind> {
    match args.model_kind.trim().to_ascii_lowercase().as_str() {
        "qwen3-asr" | "qwen3_asr" | "qwen" => Ok(RealtimeModelKind::Qwen3Asr),
        "sensevoice-online" | "sensevoice_online" | "sensevoice" => {
            Ok(RealtimeModelKind::SenseVoiceOnline)
        }
        "auto" => {
            let path = std::path::Path::new(&args.model);
            if args.model.to_ascii_lowercase().contains("sensevoice")
                || (path.is_dir() && path.join("tokens.json").is_file())
            {
                Ok(RealtimeModelKind::SenseVoiceOnline)
            } else {
                Ok(RealtimeModelKind::Qwen3Asr)
            }
        }
        value => bail!(
            "unsupported --model-kind {value:?}; expected auto, qwen3-asr, or sensevoice-online"
        ),
    }
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

fn load_asr_model(
    args: &CommonModelArgs,
    #[cfg_attr(not(feature = "cuda-graph"), allow(unused_variables))] cuda_graph_max_batch: Option<
        usize,
    >,
) -> Result<Arc<dyn AsrModel>> {
    let device = resolve_device(&args.device)?;
    let dtype = auto_dtype(&device)?;
    if resolve_model_kind(args)? == RealtimeModelKind::SenseVoiceOnline {
        if args.isq.is_some() {
            bail!("--isq is not supported by sensevoice-online");
        }
        let model_id = if args.model == "Qwen/Qwen3-ASR-0.6B" {
            "iic/SenseVoiceSmall"
        } else {
            args.model.as_str()
        };
        let options = SenseVoiceLoadOptions {
            dtype,
            chunk_size: args.sensevoice_chunk_size,
            max_history: args.sensevoice_max_history,
            language: args.language.clone().unwrap_or_else(|| "zh".to_string()),
            textnorm: args.sensevoice_textnorm,
        };
        tracing::info!(
            target: "vasr_realtime::serve",
            "Loading SenseVoice online model {model_id} on {} (dtype={dtype:?}, chunk_size={}, max_history={}, language={}, textnorm={}).",
            device_label(&device),
            options.chunk_size,
            options.max_history,
            options.language,
            options.textnorm
        );
        let model_load_start = Instant::now();
        let model = SenseVoiceOnline::from_pretrained(model_id, &device, &options)?;
        tracing::info!(
            target: "vasr_realtime::serve",
            "SenseVoice online model loaded in {:.3}s.",
            model_load_start.elapsed().as_secs_f64()
        );
        return Ok(Arc::new(model));
    }
    Ok(Arc::new(load_qwen3_asr_model(args, cuda_graph_max_batch)?))
}

fn load_offline_asr_model(
    args: &CommonModelArgs,
    cuda_graph_max_batch: Option<usize>,
) -> Result<Arc<dyn OfflineAsrModel>> {
    if resolve_model_kind(args)? != RealtimeModelKind::Qwen3Asr {
        bail!(
            "models.offline.kind {:?} does not support offline transcription; expected qwen3-asr",
            args.model_kind
        );
    }
    Ok(Arc::new(load_qwen3_asr_model(args, cuda_graph_max_batch)?))
}

fn load_qwen3_asr_model(
    args: &CommonModelArgs,
    #[cfg_attr(not(feature = "cuda-graph"), allow(unused_variables))] cuda_graph_max_batch: Option<
        usize,
    >,
) -> Result<Qwen3AsrModel> {
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
    Ok(qwen3_asr)
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

fn load_firered_vad() -> Result<FireRedVadModel> {
    FireRedVadModel::from_pretrained(None::<&std::path::Path>)
}

fn load_firered_vad_config(config: &VadModelConfig) -> Result<FireRedVadModel> {
    let device = resolve_device(&config.device)?;
    FireRedVadModel::from_pretrained_on_device(config.checkpoint.as_deref(), &device)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn config_rejects_online_only_model_in_offline_slot_before_loading() {
        let mut config = VasrConfig::default();
        config.models.offline = AsrModelConfig::sensevoice_online_default();
        let error = run_realtime_config(config).await.unwrap_err();
        assert!(error.to_string().contains("models.offline.kind"));
    }

    #[tokio::test]
    async fn config_rejects_unknown_version_before_loading() {
        let mut config = VasrConfig::default();
        config.version = 99;
        let error = run_realtime_config(config).await.unwrap_err();
        assert!(error.to_string().contains("unsupported config version"));
    }

    #[test]
    fn config_maps_vad_defaults_to_runtime_options() {
        let options = vad_options_from_config(&VadModelConfig::default());
        assert_eq!(options.threshold, 0.5);
        assert_eq!(options.min_speech_ms, 80);
        assert_eq!(options.min_silence_ms, 500);
        assert_eq!(options.max_segment_ms, 20_000);
        assert_eq!(options.pad_ms, 50);
    }
}
