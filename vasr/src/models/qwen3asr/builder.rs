use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

#[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
use crate::model_builder::PagedWorkspaceStats;
use crate::model_builder::{
    OfflineAsrBuildOutput, OfflineAsrBuildRequest, OfflineAsrBuildStats, OfflineAsrModelBuilder,
};

use super::{LoadOptions, Qwen3Asr};

#[cfg(feature = "paged-attn")]
const KV_BLOCK_SIZE: usize = 32;
#[cfg(feature = "paged-attn")]
const ASR_SAMPLE_RATE_HZ: f64 = 16_000.0;
#[cfg(feature = "paged-attn")]
const ASR_FEATURE_HOP_LENGTH: usize = 160;
#[cfg(feature = "paged-attn")]
const KV_BASE_SEGMENT_SEC: f32 = 1.0;

pub struct Qwen3AsrBuilder;

impl OfflineAsrModelBuilder for Qwen3AsrBuilder {
    fn name(&self) -> &'static str {
        "qwen3-asr"
    }

    fn supports(&self, model: &str) -> bool {
        resolve_qwen3_asr_model_id(model)
            .to_ascii_lowercase()
            .contains("qwen3-asr")
    }

    fn build(&self, req: OfflineAsrBuildRequest) -> Result<OfflineAsrBuildOutput> {
        build_qwen3_asr(req)
    }
}

pub fn resolve_qwen3_asr_model_id(model: &str) -> &str {
    match model.trim() {
        "Qwen3-ASR-0.6B" => "Qwen/Qwen3-ASR-0.6B",
        "Qwen3-ASR-1.7B" => "Qwen/Qwen3-ASR-1.7B",
        other => other,
    }
}

fn build_qwen3_asr(req: OfflineAsrBuildRequest) -> Result<OfflineAsrBuildOutput> {
    crate::set_cache_dir(req.cache_dir.clone());
    let model_id = resolve_qwen3_asr_model_id(&req.model);
    log_model_load_start(model_id, &req)?;

    #[cfg(feature = "cuda")]
    {
        super::model::set_q8_prefill_lm_head_mmq_enabled(req.quantization.q8_prefill_lm_head_mmq);
        super::model::set_q8_prefill_lm_head_mmq_max_logits_bytes(
            req.quantization.q8_prefill_lm_head_mmq_max_logits_bytes,
        );
    }

    #[cfg(feature = "paged-attn")]
    let (paged_cache, kv_cache_sizing) = build_paged_cache_config(model_id, &req)?;

    let load_options = LoadOptions {
        dtype: req.dtype,
        use_flash_attn: req.device.is_cuda(),
        isq: req.quantization.isq.clone(),
        progress: req.progress.clone(),
        #[cfg(feature = "paged-attn")]
        paged_cache: Some(paged_cache),
    };

    let build_start = Instant::now();
    super::model::isq_linear::reset_isq_layer_counts();
    #[cfg(feature = "timing")]
    super::model::isq_linear::reset_isq_quantize_time();
    let qwen3_asr = Qwen3Asr::from_pretrained(model_id, &req.device, &load_options)?;
    #[cfg(feature = "timing")]
    let quantize_seconds = req
        .quantization
        .isq
        .as_ref()
        .map(|_| super::model::isq_linear::isq_quantize_time_us() as f64 / 1_000_000.0);
    #[cfg(not(feature = "timing"))]
    let quantize_seconds = None;
    let quantized_layer_counts = req
        .quantization
        .isq
        .as_ref()
        .map(|_| super::model::isq_linear::isq_layer_counts());

    let parameter_bytes = match qwen3_asr.weights_memory_stats(req.dtype) {
        Ok(stats) => Some(stats.parameter_bytes),
        Err(err) => {
            tracing::debug!(
                target: "vasr::model_builder",
                "model parameter memory stats unavailable: {err}"
            );
            None
        }
    };

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

    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    let mut cuda_graph_bytes = None;
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    let mut cuda_graph_seqs = None;
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    let mut cuda_graph_context_tokens = None;
    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    let mut paged_workspace = None;
    #[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
    let cuda_graph_bytes = None;
    #[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
    let cuda_graph_seqs = None;
    #[cfg(not(all(feature = "cuda-graph", feature = "paged-attn")))]
    let cuda_graph_context_tokens = None;

    #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
    if let Some(max_batch) = req.cuda_graph_max_batch.filter(|&n| n > 0) {
        let prewarm_start = Instant::now();
        let stats = qwen3_asr.prewarm_cuda_decode_graphs(max_batch)?;
        paged_workspace = stats.paged_workspace.map(|workspace| PagedWorkspaceStats {
            bytes: workspace.bytes,
            floor_bytes: workspace.floor_bytes,
            block_table_cols: workspace.block_table_cols,
            batches: workspace.batches,
        });
        if stats.graphs > 0 {
            cuda_graph_seqs = Some(stats.max_batch);
            cuda_graph_context_tokens = Some(stats.context_tokens);
            if let (Some(before), Some(after)) = (stats.memory_before, stats.memory_after) {
                let used_before = before.used();
                let used_after = after.used();
                let mem_delta = used_after.saturating_sub(used_before);
                cuda_graph_bytes = Some(mem_delta);
                tracing::debug!(
                    target: "vasr::model_builder",
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
                    target: "vasr::model_builder",
                    "CUDA decode graph prewarm: graphs={} max_batch={} context_tokens={} requested_max_batch={max_batch} elapsed={:.3}s.",
                    stats.graphs,
                    stats.max_batch,
                    stats.context_tokens,
                    prewarm_start.elapsed().as_secs_f64()
                );
            }
        } else {
            tracing::debug!(
                target: "vasr::model_builder",
                "CUDA decode graph prewarm disabled after failed capture; requested_max_batch={max_batch} elapsed={:.3}s.",
                prewarm_start.elapsed().as_secs_f64()
            );
        }
    }

    Ok(OfflineAsrBuildOutput {
        model: Arc::new(qwen3_asr),
        stats: OfflineAsrBuildStats {
            model_kind: "qwen3-asr",
            build_seconds: build_start.elapsed().as_secs_f64(),
            parameter_bytes,
            quantize_seconds,
            quantized_layer_counts,
            kv_cache_bytes,
            kv_cache_tokens,
            #[cfg(feature = "paged-attn")]
            kv_cache_sizing: Some(kv_cache_sizing),
            cuda_graph_bytes,
            cuda_graph_seqs,
            cuda_graph_context_tokens,
            #[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
            paged_workspace,
        },
    })
}

#[cfg(feature = "paged-attn")]
fn build_paged_cache_config(
    model_id: &str,
    req: &OfflineAsrBuildRequest,
) -> Result<(
    super::PagedCacheConfig,
    crate::model_builder::KvCacheSizingStats,
)> {
    let sizing = estimate_kv_cache_sizing(model_id, req)?;
    log_kv_cache_sizing(&sizing);
    Ok((
        super::PagedCacheConfig {
            block_size: KV_BLOCK_SIZE,
            memory: if req.device.is_cuda() {
                // 权重已经先加载。预算 = 比例 × 总显存 − 当前占用，把剩下的显存划给 KV。
                super::PagedCacheMemory::GpuMemoryFraction {
                    fraction: req.runtime.gpu_memory_utilization,
                    reserved_bytes: 0,
                }
            } else {
                super::PagedCacheMemory::Blocks(sizing.total_blocks)
            },
            #[cfg(feature = "cuda-graph")]
            cuda_graph_context_tokens: Some(cuda_graph_context_tokens(
                sizing.peak_blocks_per_seq * KV_BLOCK_SIZE,
            )),
            #[cfg(feature = "cuda-graph")]
            cuda_graph_chunked_argmax: req.runtime.cuda_graph_chunked_argmax
                || isq_resolves_to_q8(req.quantization.isq.as_deref()),
            #[cfg(feature = "cuda-graph")]
            cuda_graph_chunked_argmax_size: req.runtime.cuda_graph_chunked_argmax_size,
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
    model_id: &str,
    req: &OfflineAsrBuildRequest,
) -> Result<crate::model_builder::KvCacheSizingStats> {
    let tokenizer = super::processor::tokenizer::Tokenizer::from_pretrained(model_id)?;
    let fixed_prompt =
        super::processor::chat_template::build_prompt("", req.runtime.language.as_deref());
    let audio_pad_id = tokenizer.token_to_id(super::processor::chat_template::AUDIO_PAD)?;
    let base_prompt_tokens = tokenizer.encode(&fixed_prompt)?;
    let fixed_prompt_tokens = base_prompt_tokens
        .iter()
        .filter(|&&id| id != audio_pad_id)
        .count();

    let base = kv_tokens_for_segment(
        fixed_prompt_tokens,
        req.runtime.max_hotword_tokens,
        KV_BASE_SEGMENT_SEC,
        req.runtime.decode_tokens_per_sec,
    )?;
    let peak = kv_tokens_for_segment(
        fixed_prompt_tokens,
        req.runtime.max_hotword_tokens,
        req.runtime.max_segment_sec,
        req.runtime.decode_tokens_per_sec,
    )?;
    let base_blocks = req
        .runtime
        .max_num_seqs
        .checked_mul(base.blocks_per_seq)
        .ok_or_else(|| anyhow::anyhow!("KV cache block estimate overflow"))?;
    let extra_blocks_per_seq = peak.blocks_per_seq.saturating_sub(base.blocks_per_seq);
    let peak_extra_blocks = ((req.runtime.max_num_seqs * extra_blocks_per_seq) as f64
        * f64::from(req.runtime.kv_peak_ratio))
    .ceil() as usize;
    let total_blocks = base_blocks
        .checked_add(peak_extra_blocks)
        .ok_or_else(|| anyhow::anyhow!("KV cache block estimate overflow"))?;

    Ok(crate::model_builder::KvCacheSizingStats {
        max_num_seqs: req.runtime.max_num_seqs,
        base_segment_sec: KV_BASE_SEGMENT_SEC,
        max_segment_sec: req.runtime.max_segment_sec,
        peak_ratio: req.runtime.kv_peak_ratio,
        max_hotword_tokens: req.runtime.max_hotword_tokens,
        decode_tokens_per_sec: req.runtime.decode_tokens_per_sec,
        decode_min_tokens: req.runtime.decode_min_tokens,
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
        block_size: KV_BLOCK_SIZE,
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
    super::processor::feat_lengths::feat_extract_output_length(frames)
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
fn log_kv_cache_sizing(sizing: &crate::model_builder::KvCacheSizingStats) {
    let long_segments =
        ((sizing.max_num_seqs as f64) * f64::from(sizing.peak_ratio)).ceil() as usize;
    tracing::info!(
        target: "vasr::model_builder",
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
        target: "vasr::model_builder",
        "planned paged kv cache | {} concurrent segments, {} may grow to {:.1}s | total_blocks={} capacity={}tok block={} context={}tok",
        sizing.max_num_seqs,
        long_segments,
        sizing.max_segment_sec,
        sizing.total_blocks,
        sizing.total_blocks.saturating_sub(1) * sizing.block_size,
        sizing.block_size,
        sizing.peak_blocks_per_seq * sizing.block_size
    );
    tracing::debug!(
        target: "vasr::model_builder",
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
    super::model::isq_linear::resolve_isq_dtype(isq, &candle_core::Device::Cpu)
        .map(|resolved| resolved.dtype() == candle_core::quantized::GgmlDType::Q8_0)
        .unwrap_or(false)
}

fn log_model_load_start(model_id: &str, req: &OfflineAsrBuildRequest) -> Result<()> {
    tracing::info!(
        target: "vasr::model_builder",
        "Loading Qwen3-ASR model `{}` from cache `{}` on {:?}.",
        model_id,
        req.cache_dir.display(),
        req.device
    );

    if let Some(isq) = req.quantization.isq.as_deref() {
        let isq_display = super::model::isq_linear::resolve_isq_display(isq, &req.device)?;
        tracing::info!(
            target: "vasr::model_builder",
            "Quantizing model weights to {isq_display}."
        );
        tracing::info!(
            target: "vasr::model_builder",
            "Loading model weights for quantization."
        );
    } else {
        tracing::info!(target: "vasr::model_builder", "Loading model weights.");
    }

    Ok(())
}

#[cfg(all(feature = "cuda-graph", feature = "paged-attn"))]
fn bytes_to_gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "paged-attn")]
    use super::*;

    #[cfg(feature = "cuda-graph")]
    #[test]
    fn isq8_auto_enables_chunked_argmax_detection() {
        assert!(!super::isq_resolves_to_q8(None));
        assert!(super::isq_resolves_to_q8(Some("8")));
        assert!(super::isq_resolves_to_q8(Some("auto")));
        assert!(!super::isq_resolves_to_q8(Some("q8_0;skip=lm_head")));
        assert!(!super::isq_resolves_to_q8(Some("q6_k")));
        assert!(!super::isq_resolves_to_q8(Some("q4_k")));
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn generated_token_budget_matches_scheduler_defaults() {
        assert_eq!(
            generated_token_budget(0.1, 4.25, KV_BLOCK_SIZE),
            KV_BLOCK_SIZE
        );
        assert_eq!(generated_token_budget(5.0, 4.25, KV_BLOCK_SIZE), 32);
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn max_audio_prompt_tokens_uses_feature_length_math() {
        assert_eq!(max_audio_prompt_tokens(30.0), 390);
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn kv_peak_ratio_reduces_peak_only_blocks() -> anyhow::Result<()> {
        let fixed_prompt_tokens = 18;
        let hotword_tokens = 128;
        let base = kv_tokens_for_segment(fixed_prompt_tokens, hotword_tokens, 1.0, 4.25)?;
        let peak = kv_tokens_for_segment(fixed_prompt_tokens, hotword_tokens, 30.0, 4.25)?;
        let base_blocks = 256 * base.blocks_per_seq;
        let peak_extra_blocks =
            ((256 * (peak.blocks_per_seq - base.blocks_per_seq)) as f64 * 0.25).ceil() as usize;

        assert_eq!(base.tokens_per_seq, 191);
        assert_eq!(base.blocks_per_seq, 6);
        assert_eq!(peak.tokens_per_seq, 664);
        assert_eq!(peak.blocks_per_seq, 21);
        assert_eq!(base_blocks + peak_extra_blocks, 2496);
        Ok(())
    }
}
