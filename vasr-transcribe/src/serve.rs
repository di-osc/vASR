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
use candle_core::{DType, Device};
use clap::{ArgAction, Args};
use vasr::models::qwen3asr::model::isq_linear::{
    isq_layer_counts, isq_quantize_time_us, reset_isq_layer_counts, reset_isq_quantize_time,
    resolve_isq_display, resolve_isq_dtype,
};
#[cfg(any(feature = "metal", feature = "cuda"))]
use vasr::models::qwen3asr::model::paged_cache_runtime::{
    PagedCacheConfig, PagedCacheMemory, PagedCacheStats,
};
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::processor::{chat_template, feat_lengths, tokenizer::Tokenizer};
use vasr::models::qwen3asr::{LoadOptions, Qwen3ASR};
use vasr_data::AudioLoader;

use crate::pipeline::{
    AsyncTranscribePipeline, GlobalAsyncTranscribePipeline, PipelineWorkerOptions,
};
use crate::server::build_transcribe_service_from_parts;

const DEFAULT_MAX_NUM_SEQS: usize = 256;
const DEFAULT_ASR_MODEL: &str = "Qwen/Qwen3-ASR-0.6B";
#[cfg(feature = "paged-attn")]
const KV_BLOCK_SIZE: usize = 32;
const DEFAULT_MAX_HOTWORD_TOKENS: usize = 128;
const DEFAULT_KV_PEAK_RATIO: f32 = 0.25;
#[cfg(feature = "paged-attn")]
const DEFAULT_DECODE_TOKENS_PER_SEC: f64 = 4.25;
const DEFAULT_DECODE_MIN_TOKENS: usize = 128;
#[cfg(feature = "cuda")]
const DEFAULT_Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_MIB: usize = 256;
#[cfg(feature = "paged-attn")]
const ASR_SAMPLE_RATE_HZ: f64 = 16_000.0;
#[cfg(feature = "paged-attn")]
const ASR_FEATURE_HOP_LENGTH: usize = 160;
#[cfg(feature = "paged-attn")]
const KV_BASE_SEGMENT_SEC: f32 = 1.0;

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
    /// FSMN VAD device: auto, cpu, cuda.
    #[arg(long = "vad-device", default_value = "cpu", env = "VASR_VAD_DEVICE")]
    pub vad_device: String,

    /// FSMN VAD speech probability threshold (`speech_noise_thres`).
    #[arg(long, default_value_t = 0.6, env = "VASR_VAD_THRESHOLD")]
    pub vad_threshold: f32,

    /// Minimum speech duration in milliseconds before a segment is confirmed.
    #[arg(long, default_value_t = 250, env = "VASR_VAD_MIN_SPEECH_MS")]
    pub vad_min_speech_ms: u64,

    /// Minimum trailing silence in milliseconds before ending a speech segment.
    #[arg(long, default_value_t = 800, env = "VASR_VAD_MIN_SILENCE_MS")]
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

    /// Number of audio load workers. 0 uses all available CPU cores.
    #[arg(
        long = "num-load-workers",
        default_value_t = 0,
        env = "VASR_NUM_LOAD_WORKERS"
    )]
    pub num_load_workers: usize,

    /// Number of VAD workers. 0 uses all available CPU cores.
    #[arg(
        long = "num-vad-workers",
        default_value_t = 0,
        env = "VASR_NUM_VAD_WORKERS"
    )]
    pub num_vad_workers: usize,

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
    let transcribe_service = build_transcribe_service_from_parts(
        build_offline_pipeline(
            load_asr_model_for_pipeline(
                &args.pipeline,
                args.language.as_deref(),
                cuda_graph_max_batch(&args.pipeline.model),
                false,
            )?,
            &args.pipeline.vad,
        )?,
        asr_options_from_pipeline(&args.pipeline, args.language.clone()),
        vad_options_from_pipeline(&args.pipeline),
        worker_options_from_pipeline(&args.pipeline),
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
    let asr = load_asr_model_for_pipeline(
        pipeline,
        language.as_deref(),
        cuda_graph_max_batch(&pipeline.model),
        false,
    )?;
    let offline = build_offline_pipeline(asr, &pipeline.vad)?;
    Ok(AsyncTranscribePipeline::new(
        AudioLoader,
        offline,
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
    let asr = load_asr_model_for_pipeline(
        pipeline,
        language.as_deref(),
        cuda_graph_max_batch(&pipeline.model),
        false,
    )?;
    let offline = build_offline_pipeline(asr, &pipeline.vad)?;
    Ok(Arc::new(GlobalAsyncTranscribePipeline::start(
        vasr_data::AudioLoader,
        offline,
        asr_options_from_pipeline(pipeline, language),
        vad_options_from_pipeline(pipeline),
        worker_options_from_pipeline(pipeline),
    )))
}

pub fn validate_pipeline(pipeline: &TranscribePipelineArgs) -> Result<()> {
    validate_model_args(&pipeline.model)?;
    validate_vad_args(&pipeline.vad)
}

fn build_offline_pipeline(
    asr: Arc<dyn AsrModel>,
    vad_args: &VadCliArgs,
) -> Result<Arc<OfflinePipeline>> {
    let vad_load_start = Instant::now();
    let vad_device = resolve_vad_device(vad_args.vad_device.as_str())?;
    let vad = load_fsmn_vad(&vad_device)?;
    tracing::info!(
        target: "vasr_transcribe::serve",
        "loaded vad | {:.3}s model=fsmn device={}",
        vad_load_start.elapsed().as_secs_f64(),
        device_label(&vad_device),
    );
    tracing::debug!(
        target: "vasr_transcribe::serve",
        "FSMN VAD model loaded from `{}`.",
        vad.model_dir().display()
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
        max_new_tokens: 0,
        decode_max_seqs: pipeline.model.max_num_seqs,
        max_batch_audio_sec: pipeline.max_batch_audio_sec,
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
            max_batch_audio_sec: AsrOptions::default().max_batch_audio_sec,
            num_load_workers: 0,
            num_vad_workers: 0,
            vad: VadCliArgs {
                vad_device: "cpu".to_string(),
                vad_threshold: 0.6,
                vad_min_speech_ms: 250,
                vad_min_silence_ms: 800,
                vad_max_segment_sec: 30.0,
            },
        }
    }

    #[test]
    fn pipeline_defaults_match_validated_vad_batch_bounds() {
        let pipeline = default_pipeline_args();

        let opts = asr_options_from_pipeline(&pipeline, None);
        assert_eq!(opts.max_new_tokens, 0);
        assert_eq!(opts.max_batch_audio_sec, 60.0);
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

    #[cfg(feature = "cuda-graph")]
    #[test]
    fn isq8_auto_enables_chunked_argmax_detection() {
        assert!(!isq_resolves_to_q8(None));
        assert!(isq_resolves_to_q8(Some("8")));
        assert!(isq_resolves_to_q8(Some("auto")));
        assert!(!isq_resolves_to_q8(Some("q8_0;skip=lm_head")));
        assert!(!isq_resolves_to_q8(Some("q6_k")));
        assert!(!isq_resolves_to_q8(Some("q4_k")));
    }

    #[cfg(all(feature = "paged-attn", feature = "cuda-graph"))]
    #[test]
    fn isq8_auto_enables_chunked_argmax_in_paged_cache_config() -> anyhow::Result<()> {
        let mut pipeline = default_pipeline_args();
        pipeline.model.isq = Some("q8_0".to_string());
        let (config, _) = build_paged_cache_config(&pipeline, Some("Chinese"))?;
        assert!(config.cuda_graph_chunked_argmax);

        pipeline.model.isq = Some("q6_k".to_string());
        let (config, _) = build_paged_cache_config(&pipeline, Some("Chinese"))?;
        assert!(!config.cuda_graph_chunked_argmax);
        Ok(())
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn generated_token_budget_matches_scheduler_defaults() {
        assert_eq!(
            generated_token_budget(0.1, 4.25, KV_BLOCK_SIZE),
            KV_BLOCK_SIZE
        );
        assert_eq!(generated_token_budget(5.0, 4.25, KV_BLOCK_SIZE), 40);
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn max_audio_prompt_tokens_uses_feature_length_math() {
        // 30s at 16kHz with hop=160 gives 3000 frames; Qwen3-ASR's projector
        // length formula maps that to 390 audio placeholder tokens.
        assert_eq!(max_audio_prompt_tokens(30.0), 390);
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn kv_peak_ratio_reduces_peak_only_blocks() -> anyhow::Result<()> {
        let fixed_prompt_tokens = 18;
        let hotword_tokens = DEFAULT_MAX_HOTWORD_TOKENS;
        let base = kv_tokens_for_segment(
            fixed_prompt_tokens,
            hotword_tokens,
            1.0,
            DEFAULT_DECODE_TOKENS_PER_SEC,
        )?;
        let peak = kv_tokens_for_segment(
            fixed_prompt_tokens,
            hotword_tokens,
            30.0,
            DEFAULT_DECODE_TOKENS_PER_SEC,
        )?;
        let base_blocks = DEFAULT_MAX_NUM_SEQS * base.blocks_per_seq;
        let peak_extra_blocks = ((DEFAULT_MAX_NUM_SEQS
            * (peak.blocks_per_seq - base.blocks_per_seq)) as f64
            * f64::from(DEFAULT_KV_PEAK_RATIO))
        .ceil() as usize;

        assert_eq!(base.tokens_per_seq, 191);
        assert_eq!(base.blocks_per_seq, 6);
        assert_eq!(peak.tokens_per_seq, 776);
        assert_eq!(peak.blocks_per_seq, 25);
        assert_eq!(base_blocks + peak_extra_blocks, 2752);
        Ok(())
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

#[cfg(feature = "cuda-graph")]
fn isq_resolves_to_q8(isq: Option<&str>) -> bool {
    let Some(isq) = isq else {
        return false;
    };
    if isq
        .split([';', ':', ','])
        .skip(1)
        .any(|part| part.to_ascii_lowercase().contains("lm_head"))
    {
        return false;
    }
    resolve_isq_dtype(isq, &Device::Cpu)
        .map(|resolved| resolved.dtype() == candle_core::quantized::GgmlDType::Q8_0)
        .unwrap_or(false)
}

fn resolve_asr_model_id(model: &str) -> &str {
    match model.trim() {
        "Qwen3-ASR-0.6B" => "Qwen/Qwen3-ASR-0.6B",
        "Qwen3-ASR-1.7B" => "Qwen/Qwen3-ASR-1.7B",
        other => other,
    }
}

#[cfg(feature = "paged-attn")]
#[derive(Debug, Clone)]
struct KvCacheSizing {
    max_num_seqs: usize,
    base_segment_sec: f32,
    max_segment_sec: f32,
    peak_ratio: f32,
    max_hotword_tokens: usize,
    decode_tokens_per_sec: f64,
    decode_min_tokens: usize,
    planning_decode_min_tokens: usize,
    fixed_prompt_tokens: usize,
    base_audio_prompt_tokens: usize,
    base_decode_tokens: usize,
    base_tokens_per_seq: usize,
    base_blocks_per_seq: usize,
    peak_audio_prompt_tokens: usize,
    peak_decode_tokens: usize,
    peak_tokens_per_seq: usize,
    peak_blocks_per_seq: usize,
    base_blocks: usize,
    peak_extra_blocks: usize,
    total_blocks: usize,
}

#[cfg(feature = "paged-attn")]
fn build_paged_cache_config(
    pipeline: &TranscribePipelineArgs,
    language: Option<&str>,
) -> Result<(PagedCacheConfig, KvCacheSizing)> {
    let sizing = estimate_kv_cache_sizing(pipeline, language)?;
    Ok((
        PagedCacheConfig {
            block_size: KV_BLOCK_SIZE,
            memory: PagedCacheMemory::Blocks(sizing.total_blocks),
            #[cfg(feature = "cuda-graph")]
            cuda_graph_context_tokens: Some(cuda_graph_context_tokens(
                sizing.peak_blocks_per_seq * KV_BLOCK_SIZE,
            )),
            #[cfg(feature = "cuda-graph")]
            cuda_graph_chunked_argmax: pipeline.model.cuda_graph_chunked_argmax
                || isq_resolves_to_q8(pipeline.model.isq.as_deref()),
            #[cfg(feature = "cuda-graph")]
            cuda_graph_chunked_argmax_size: pipeline.model.cuda_graph_chunked_argmax_size,
        },
        sizing,
    ))
}

#[cfg(all(feature = "paged-attn", feature = "cuda-graph"))]
fn cuda_graph_context_tokens(default_tokens: usize) -> usize {
    std::env::var("VASR_CUDA_GRAPH_CONTEXT_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_tokens)
}

#[cfg(feature = "paged-attn")]
fn estimate_kv_cache_sizing(
    pipeline: &TranscribePipelineArgs,
    language: Option<&str>,
) -> Result<KvCacheSizing> {
    let model_id = resolve_asr_model_id(&pipeline.model.model);
    let tokenizer = Tokenizer::from_pretrained(model_id)?;
    let fixed_prompt = chat_template::build_prompt("", language);
    let audio_pad_id = tokenizer.token_to_id(chat_template::AUDIO_PAD)?;
    let base_prompt_tokens = tokenizer.encode(&fixed_prompt)?;
    let fixed_prompt_tokens = base_prompt_tokens
        .iter()
        .filter(|&&id| id != audio_pad_id)
        .count();

    let base = kv_tokens_for_segment(
        fixed_prompt_tokens,
        pipeline.model.max_hotword_tokens,
        KV_BASE_SEGMENT_SEC,
        pipeline.model.decode_tokens_per_sec,
    )?;
    let peak = kv_tokens_for_segment(
        fixed_prompt_tokens,
        pipeline.model.max_hotword_tokens,
        pipeline.vad.vad_max_segment_sec,
        pipeline.model.decode_tokens_per_sec,
    )?;
    let base_blocks = pipeline
        .model
        .max_num_seqs
        .checked_mul(base.blocks_per_seq)
        .ok_or_else(|| anyhow::anyhow!("KV cache block estimate overflow"))?;
    let extra_blocks_per_seq = peak.blocks_per_seq.saturating_sub(base.blocks_per_seq);
    let peak_extra_blocks = ((pipeline.model.max_num_seqs * extra_blocks_per_seq) as f64
        * f64::from(pipeline.model.kv_peak_ratio))
    .ceil() as usize;
    let total_blocks = base_blocks
        .checked_add(peak_extra_blocks)
        .ok_or_else(|| anyhow::anyhow!("KV cache block estimate overflow"))?;

    Ok(KvCacheSizing {
        max_num_seqs: pipeline.model.max_num_seqs,
        base_segment_sec: KV_BASE_SEGMENT_SEC,
        max_segment_sec: pipeline.vad.vad_max_segment_sec,
        peak_ratio: pipeline.model.kv_peak_ratio,
        max_hotword_tokens: pipeline.model.max_hotword_tokens,
        decode_tokens_per_sec: pipeline.model.decode_tokens_per_sec,
        decode_min_tokens: pipeline.model.decode_min_tokens,
        planning_decode_min_tokens: KV_BLOCK_SIZE,
        fixed_prompt_tokens,
        base_audio_prompt_tokens: base.audio_prompt_tokens,
        base_decode_tokens: base.decode_tokens,
        base_tokens_per_seq: base.tokens_per_seq,
        base_blocks_per_seq: base.blocks_per_seq,
        peak_audio_prompt_tokens: peak.audio_prompt_tokens,
        peak_decode_tokens: peak.decode_tokens,
        peak_tokens_per_seq: peak.tokens_per_seq,
        peak_blocks_per_seq: peak.blocks_per_seq,
        base_blocks,
        peak_extra_blocks,
        total_blocks,
    })
}

#[cfg(feature = "paged-attn")]
#[derive(Debug, Clone, Copy)]
struct KvSegmentTokens {
    audio_prompt_tokens: usize,
    decode_tokens: usize,
    tokens_per_seq: usize,
    blocks_per_seq: usize,
}

#[cfg(feature = "paged-attn")]
fn kv_tokens_for_segment(
    fixed_prompt_tokens: usize,
    max_hotword_tokens: usize,
    segment_sec: f32,
    decode_tokens_per_sec: f64,
) -> Result<KvSegmentTokens> {
    let audio_prompt_tokens = max_audio_prompt_tokens(segment_sec);
    let decode_tokens = generated_token_budget(segment_sec, decode_tokens_per_sec, KV_BLOCK_SIZE);
    let tokens_per_seq = fixed_prompt_tokens
        .checked_add(audio_prompt_tokens)
        .and_then(|n| n.checked_add(max_hotword_tokens))
        .and_then(|n| n.checked_add(decode_tokens))
        .ok_or_else(|| anyhow::anyhow!("KV cache token estimate overflow"))?;
    Ok(KvSegmentTokens {
        audio_prompt_tokens,
        decode_tokens,
        tokens_per_seq,
        blocks_per_seq: tokens_per_seq.div_ceil(KV_BLOCK_SIZE),
    })
}

#[cfg(feature = "paged-attn")]
fn max_audio_prompt_tokens(max_segment_sec: f32) -> usize {
    let samples = (f64::from(max_segment_sec) * ASR_SAMPLE_RATE_HZ).ceil() as usize;
    let frames = samples.div_ceil(ASR_FEATURE_HOP_LENGTH);
    feat_lengths::feat_extract_output_length(frames)
}

#[cfg(feature = "paged-attn")]
fn generated_token_budget(max_segment_sec: f32, tokens_per_sec: f64, min_tokens: usize) -> usize {
    let estimated = if max_segment_sec.is_finite() && max_segment_sec > 0.0 {
        (f64::from(max_segment_sec) * tokens_per_sec).ceil() as usize
    } else {
        0
    };
    estimated.max(min_tokens)
}

#[cfg(feature = "paged-attn")]
fn log_kv_cache_sizing(sizing: &KvCacheSizing) {
    let long_segments =
        ((sizing.max_num_seqs as f64) * f64::from(sizing.peak_ratio)).ceil() as usize;
    tracing::info!(
        target: "vasr_transcribe::serve",
        "planning paged kv cache | max_num_seqs={} vad_max_segment_sec={:.1}s kv_peak_ratio={:.2} max_hotword_tokens={}tok decode={:.2}tok/s runtime_min={}tok planning_min={}tok",
        sizing.max_num_seqs,
        sizing.max_segment_sec,
        sizing.peak_ratio,
        sizing.max_hotword_tokens,
        sizing.decode_tokens_per_sec,
        sizing.decode_min_tokens,
        sizing.planning_decode_min_tokens,
    );
    tracing::info!(
        target: "vasr_transcribe::serve",
        "planned paged kv cache | {} concurrent segments, {} may grow to {:.1}s | total_blocks={} capacity={}tok block={} context={}tok",
        sizing.max_num_seqs,
        long_segments,
        sizing.max_segment_sec,
        sizing.total_blocks,
        sizing.total_blocks.saturating_sub(1) * KV_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        sizing.peak_blocks_per_seq * KV_BLOCK_SIZE
    );
    tracing::debug!(
        target: "vasr_transcribe::serve",
        "KV cache details | peak_ratio={:.2} base={:.1}s/{}tok/{}blk peak={:.1}s/{}tok/{}blk base_blocks={} peak_extra_blocks={} prompt=fixed:{} base_audio:{} peak_audio:{} hotword:{} base_decode:{} peak_decode:{}",
        sizing.peak_ratio,
        sizing.base_segment_sec,
        sizing.base_tokens_per_seq,
        sizing.base_blocks_per_seq,
        sizing.max_segment_sec,
        sizing.peak_tokens_per_seq,
        sizing.peak_blocks_per_seq,
        sizing.base_blocks,
        sizing.peak_extra_blocks,
        sizing.fixed_prompt_tokens,
        sizing.base_audio_prompt_tokens,
        sizing.peak_audio_prompt_tokens,
        sizing.max_hotword_tokens,
        sizing.base_decode_tokens,
        sizing.peak_decode_tokens
    );
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn log_paged_cache_stats(stats: &PagedCacheStats, config: &PagedCacheConfig) {
    let sizing = match config.memory {
        PagedCacheMemory::ContextSize(tokens) => format!("context_tokens={tokens}"),
        PagedCacheMemory::Blocks(blocks) => format!("blocks={blocks}"),
        PagedCacheMemory::GpuMemoryFraction {
            fraction,
            reserved_bytes: _,
        } => format!("gpu_memory_fraction={fraction}"),
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
    elapsed_seconds: f64,
    isq_quant_seconds: Option<f64>,
    isq_layer_counts: Option<(usize, usize, usize)>,
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
    let isq_quant = isq_quant_seconds
        .map(|seconds| format!(" isq_quant={seconds:.3}s"))
        .unwrap_or_default();
    let isq_layers = isq_layer_counts
        .map(|(audio, text, other)| format!(" isq_layers=audio:{audio},text:{text},other:{other}"))
        .unwrap_or_default();
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
                "loaded model | {elapsed_seconds:.3}s{isq_quant}{isq_layers} params={params} kv={kv_cache} graph={cuda_graph} gpu={:.2}/{:.2}GiB free={:.2}GiB",
                bytes_to_gib(gpu_used),
                bytes_to_gib(memory.total),
                bytes_to_gib(memory.free),
            );
            tracing::debug!(
                target: "vasr_transcribe::serve",
                "model memory details | fixed={:.2}GiB/{fixed_pct:.1}% gpu_used={:.2}GiB/{gpu_used_pct:.1}%",
                bytes_to_gib(fixed_bytes),
                bytes_to_gib(gpu_used),
            );
            return;
        }
    }

    tracing::info!(
        target: "vasr_transcribe::serve",
        "loaded model | {elapsed_seconds:.3}s{isq_quant}{isq_layers} params={params} kv={kv_cache} graph={cuda_graph}"
    );
    tracing::debug!(
        target: "vasr_transcribe::serve",
        "model memory details | fixed={:.2}GiB",
        bytes_to_gib(fixed_bytes),
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
    match vad.vad_device.trim().to_ascii_lowercase().as_str() {
        "auto" | "cpu" | "cuda" | "gpu" => {}
        other => bail!("--vad-device must be auto, cpu, or cuda; got {other:?}"),
    }
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
    vasr::set_cache_dir(args.cache_dir.clone());
    let device = resolve_device(&args.device)?;
    let dtype = auto_dtype(&device)?;
    #[cfg(feature = "cuda")]
    {
        let max_logits_bytes = args
            .q8_prefill_lm_head_mmq_max_logits_mib
            .saturating_mul(1024 * 1024);
        vasr::models::qwen3asr::model::set_q8_prefill_lm_head_mmq_enabled(
            args.q8_prefill_lm_head_mmq,
        );
        vasr::models::qwen3asr::model::set_q8_prefill_lm_head_mmq_max_logits_bytes(
            max_logits_bytes,
        );
        tracing::info!(
            target: "vasr_transcribe::serve",
            "Q8 prefill lm_head MMQ path {} | max_logits={}MiB",
            if args.q8_prefill_lm_head_mmq {
                "enabled"
            } else {
                "disabled"
            },
            args.q8_prefill_lm_head_mmq_max_logits_mib,
        );
    }
    #[cfg(feature = "paged-attn")]
    let (paged_cache_config, kv_cache_sizing) = build_paged_cache_config(pipeline, language)?;
    let load_options = LoadOptions {
        dtype,
        use_flash_attn: device.is_cuda(),
        isq: args.isq.clone(),
        #[cfg(feature = "paged-attn")]
        paged_cache: Some(paged_cache_config),
    };

    tracing::debug!(
        target: "vasr_transcribe::serve",
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
        cfg!(target_feature = "f16c")
    );
    tracing::info!(
        target: "vasr_transcribe::serve",
        "starting service | model={} device={} dtype={}",
        args.model,
        device_label(&device),
        dtype_label(dtype),
    );
    tracing::debug!(target: "vasr_transcribe::serve", "model kind=qwen3-asr adapters=none");
    if let Some(isq) = args.isq.as_deref() {
        tracing::debug!(
            target: "vasr_transcribe::serve",
            "ISQ selected is {} (requested={}, backend={}).",
            resolve_isq_display(isq, &device)?,
            isq,
            device_label(&device)
        );
    }
    #[cfg(feature = "paged-attn")]
    log_kv_cache_sizing(&kv_cache_sizing);
    tracing::debug!(
        target: "vasr_transcribe::serve",
        "Loading Qwen3-ASR model `{}` from cache `{}` on {} (isq={:?}).",
        resolve_asr_model_id(&args.model),
        args.cache_dir.display(),
        device_label(&device),
        args.isq
    );
    let model_load_start = Instant::now();
    reset_isq_layer_counts();
    reset_isq_quantize_time();
    let qwen3_asr =
        Qwen3ASR::from_pretrained(resolve_asr_model_id(&args.model), &device, &load_options)?;
    let isq_quant_seconds = args
        .isq
        .as_ref()
        .map(|_| isq_quantize_time_us() as f64 / 1_000_000.0);
    let quantized_layer_counts = args.isq.as_ref().map(|_| isq_layer_counts());
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
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    configure_cuda_mempool_release_threshold(&device);
    log_model_loaded(
        model_load_start.elapsed().as_secs_f64(),
        isq_quant_seconds,
        quantized_layer_counts,
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

#[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
fn configure_cuda_mempool_release_threshold(device: &Device) {
    if std::env::var_os("VASR_DISABLE_CUDA_MEMPOOL_RELEASE_THRESHOLD").is_some() {
        tracing::debug!(
            target: "vasr_transcribe::serve",
            "CUDA mempool release threshold left unchanged"
        );
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
    tracing::info!(
        target: "vasr_transcribe::serve",
        "CUDA mempool release threshold set | threshold={threshold_mib}MiB"
    );
    vasr::vasr_paged_attn::cuda_graph::cuda_trim_device_mem_pools(device);
}

fn bind_addr(host: &str, port: u16) -> Result<SocketAddr> {
    Ok(format!("{host}:{port}").parse()?)
}

async fn serve_app(app: Router, addr: SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(
        target: "vasr_transcribe::serve",
        "server ready | http://{addr} endpoints=health,transcribe,inference"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
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
    tracing::debug!(
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
