//! Audio tower (conv downsample + encoder transformer).
//!
//! This is a faithful port of `Qwen3ASRAudioEncoder` from the official Python
//! implementation:
//! `Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`.
//!
//! Note: the official code uses `cu_seqlens` to run windowed self-attention
//! inside the encoder (block-diagonal attention). This Rust port implements the
//! same segmentation for correctness; a FlashAttention2 varlen fast-path is
//! added separately (behind `--features flash-attn` and load options).

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Module, VarBuilder, conv2d, layer_norm};

use crate::config::AudioEncoderConfig;
use crate::model::cuda_mem_debug::log_cuda_mem;
use crate::model::isq_linear::{self, IsqLinear};
use crate::processor::feat_lengths::feat_extract_output_length;
use crate::utils::progress::{Progress, ProgressEvent};

#[cfg(feature = "flash-attn")]
use vasr_flash_attn::flash_attn_varlen;

const LAYER_NORM_EPS: f64 = 1e-5;

fn conv_forward(conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal-paged-attn")]
    if x.device().is_metal() {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| {
            std::env::var("VASR_METAL_CONV_FLATTEN")
                .ok()
                .is_some_and(|v| {
                    matches!(
                        v.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
        });
        let cfg = conv.config();
        if enabled && cfg.groups == 1 {
            let output = vasr_paged_attn::metal::conv::conv2d_flattened(
                x,
                conv.weight(),
                cfg.padding,
                cfg.stride,
                cfg.dilation,
            )?;
            return if let Some(bias) = conv.bias() {
                output.broadcast_add(&bias.reshape((1, bias.elem_count(), 1, 1))?)
            } else {
                Ok(output)
            };
        }
    }
    conv.forward(x)
}

#[derive(Debug, Clone, Default)]
struct ConvExecutionPolicy {
    auto: bool,
    nhwc_override: Option<bool>,
    im2col32_override: Option<bool>,
    bias_gelu_override: Option<bool>,
    native_chunks_override: Option<bool>,
    #[cfg(feature = "metal-paged-attn")]
    gemm_tile: Option<vasr_quant::metal_gemm::GemmTile>,
}

impl ConvExecutionPolicy {
    fn from_values(nhwc: Option<&str>, im2col32: Option<&str>) -> Result<Self> {
        Ok(Self {
            auto: false,
            nhwc_override: Self::parse_override(nhwc, "VASR_METAL_CONV_NHWC")?,
            im2col32_override: Self::parse_override(im2col32, "VASR_METAL_CONV_IM2COL32")?,
            bias_gelu_override: None,
            native_chunks_override: None,
            #[cfg(feature = "metal-paged-attn")]
            gemm_tile: None,
        })
    }
    fn parse_override(value: Option<&str>, name: &str) -> Result<Option<bool>> {
        match value.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None => Ok(None),
            Some("1" | "true" | "yes" | "on") => Ok(Some(true)),
            Some("0" | "false" | "no" | "off") => Ok(Some(false)),
            Some(_) => candle_core::bail!("{name} must be 0 or 1"),
        }
    }
    fn fused_bias_gelu(&self) -> bool {
        self.resolve().0 && self.bias_gelu_override.unwrap_or(self.auto)
    }
    fn native_chunks(&self) -> bool {
        self.native_chunks_override.unwrap_or(false)
    }
    fn resolve(&self) -> (bool, bool) {
        let nhwc = self.nhwc_override.unwrap_or(self.auto);
        (nhwc, nhwc && self.im2col32_override.unwrap_or(self.auto))
    }
}

fn conv_gelu_forward(conv: &Conv2d, x: &Tensor, policy: &ConvExecutionPolicy) -> Result<Tensor> {
    let _ = policy;
    #[cfg(feature = "metal-paged-attn")]
    if x.device().is_metal() {
        let (enabled, fast) = policy.resolve();
        let cfg = conv.config();
        if enabled && cfg.groups == 1 {
            let output = vasr_paged_attn::metal::conv::conv2d_nhwc_with_gemm_tile(
                x,
                conv.weight(),
                cfg.padding,
                cfg.stride,
                cfg.dilation,
                fast,
                policy.gemm_tile,
            )?;
            if policy.fused_bias_gelu() {
                if let Some(bias) = conv.bias() {
                    return vasr_paged_attn::metal::conv::bias_gelu_nhwc(&output, bias)?
                        .permute((0, 3, 1, 2));
                }
            }
            let output = if let Some(bias) = conv.bias() {
                output.broadcast_add(&bias.reshape((1, 1, 1, bias.elem_count()))?)?
            } else {
                output
            };
            return gelu(&output)?.permute((0, 3, 1, 2));
        }
    }
    gelu(&conv_forward(conv, x)?)
}

fn maybe_contiguous_for_metal(x: Tensor) -> Result<Tensor> {
    if x.device().is_metal() || x.device().is_cuda() {
        x.contiguous()
    } else {
        Ok(x)
    }
}

fn cast_input_features_to_weight_dtype(
    input_features: &Tensor,
    want_dtype: DType,
) -> Result<Tensor> {
    if input_features.dtype() == want_dtype {
        return Ok(input_features.clone());
    }

    // On Metal, avoid GPU→CPU→GPU round-trips for BF16 conversions.
    // Instead, convert via F32 as an intermediate (two on-device casts are
    // much cheaper than one CPU transfer because the GPU pipeline stays intact).
    if input_features.device().is_metal() && want_dtype == DType::BF16 {
        return input_features.to_dtype(DType::F32)?.to_dtype(want_dtype);
    }

    input_features.to_dtype(want_dtype)
}

fn pad_audio_chunk(
    chunk: &Tensor,
    dim: usize,
    left: usize,
    right: usize,
    native: bool,
) -> Result<Tensor> {
    if !native
        && chunk.device().is_metal()
        && chunk.dtype() == DType::BF16
        && (left != 0 || right != 0)
    {
        // Convert to F32 for padding (Metal has full F32 support), then back to BF16.
        return chunk
            .to_dtype(DType::F32)?
            .contiguous()?
            .pad_with_zeros(dim, left, right)?
            .to_dtype(chunk.dtype());
    }

    chunk.pad_with_zeros(dim, left, right)
}

fn stack_audio_chunks(chunks: &[Tensor], native: bool) -> Result<Tensor> {
    let first = chunks
        .first()
        .ok_or_else(|| candle_core::Error::Msg("missing audio chunks".to_string()))?;

    if !native && first.device().is_metal() && first.dtype() == DType::BF16 {
        // Convert each chunk to F32 for stacking, stack, then convert back.
        // Keeps all work on-device instead of GPU→CPU→GPU round-trip.
        let f32_chunks: Vec<Tensor> = chunks
            .iter()
            .map(|chunk| chunk.to_dtype(DType::F32)?.contiguous())
            .collect::<Result<Vec<_>>>()?;
        let f32_refs: Vec<&Tensor> = f32_chunks.iter().collect();
        let stacked = Tensor::stack(&f32_refs, 0)?;
        return stacked.to_dtype(first.dtype());
    }

    Tensor::stack(&chunks.iter().collect::<Vec<_>>(), 0)
}

fn cu_seqlens_from_aftercnn_len(
    aftercnn_len: usize,
    max_chunk_len_aftercnn: usize,
    window: usize,
    window_infer: usize,
) -> Result<Vec<usize>> {
    if aftercnn_len == 0 {
        return Ok(vec![0usize]);
    }
    if window == 0 {
        candle_core::bail!("audio_config window must be > 0");
    }
    let ratio = window_infer / window;
    if ratio == 0 {
        candle_core::bail!(
            "audio_config.n_window_infer must be >= audio_config.n_window*2: n_window_infer={window_infer} window={window}"
        );
    }

    let window_aftercnn =
        max_chunk_len_aftercnn
            .checked_mul(ratio)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "window_aftercnn overflow: max_chunk_len_aftercnn={max_chunk_len_aftercnn} ratio={ratio}"
                ))
            })?;
    if window_aftercnn == 0 {
        candle_core::bail!(
            "invalid window_aftercnn=0: max_chunk_len_aftercnn={max_chunk_len_aftercnn} ratio={ratio}"
        );
    }

    let mut cu: Vec<usize> = Vec::new();
    cu.push(0usize);

    let mut offset = 0usize;
    while offset < aftercnn_len {
        let remaining = aftercnn_len.saturating_sub(offset);
        let step = remaining.min(window_aftercnn);
        offset = offset
            .checked_add(step)
            .ok_or_else(|| candle_core::Error::Msg("cu_seqlens offset overflow".to_string()))?;
        cu.push(offset);
    }

    if cu.last().copied().is_none_or(|last| last != aftercnn_len) {
        candle_core::bail!(
            "internal error: cu_seqlens did not reach aftercnn_len: last={:?} aftercnn_len={aftercnn_len}",
            cu.last().copied()
        );
    }

    Ok(cu)
}

fn feature_lens_from_mask(mask: &Tensor, frames: usize) -> Result<Vec<usize>> {
    let (batch, t2) = mask.dims2()?;
    if t2 != frames {
        candle_core::bail!("feature_attention_mask frames mismatch: expected={frames}, got={t2}");
    }

    // Avoid transferring the full mask back to the host. We only need per-sample
    // lengths, so compute them on-device and materialize a (batch,) vector.
    let mask_f32 = mask.to_dtype(DType::F32)?;
    let nonzero = mask_f32.ne(0f32)?.to_dtype(DType::F32)?;
    let lens = nonzero.sum(1)?; // (batch,)
    let lens = lens.to_vec1::<f32>()?;
    if lens.len() != batch {
        candle_core::bail!(
            "internal error: feature_attention_mask lens mismatch: expected={batch}, got={}",
            lens.len()
        );
    }

    let mut out: Vec<usize> = Vec::with_capacity(batch);
    for (i, v) in lens.into_iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            candle_core::bail!("invalid feature_attention_mask length at {i}: {v}");
        }
        if v > frames as f32 + 0.01 {
            candle_core::bail!(
                "feature_attention_mask length at {i} exceeds frames: len={v} frames={frames}"
            );
        }
        let len = v.round() as usize;
        if (len as f32 - v).abs() > 0.01 {
            candle_core::bail!("feature_attention_mask length at {i} is not integral: {v}");
        }
        out.push(len);
    }
    Ok(out)
}

fn sinusoidal_positional_embedding(
    length: usize,
    channels: usize,
    device: &Device,
) -> Result<Tensor> {
    if !channels.is_multiple_of(2) {
        candle_core::bail!(
            "sinusoidal positional embedding requires even channels, got {channels}"
        );
    }
    if length == 0 {
        return Tensor::zeros((0usize, channels), DType::F32, device);
    }

    let half = channels / 2;
    let max_timescale = 10_000.0f64;
    let denom = (half as f64) - 1.0;
    if denom <= 0.0 {
        candle_core::bail!("invalid channels for sinusoidal embedding: {channels}");
    }
    let log_timescale_increment = max_timescale.ln() / denom;

    let mut inv_timescales = Vec::with_capacity(half);
    for i in 0..half {
        inv_timescales.push((-log_timescale_increment * (i as f64)).exp());
    }

    let mut data = Vec::with_capacity(length * channels);
    for t in 0..length {
        let t = t as f64;
        for &inv in &inv_timescales {
            data.push((t * inv).sin() as f32);
        }
        for &inv in &inv_timescales {
            data.push((t * inv).cos() as f32);
        }
    }

    Tensor::from_vec(data, (length, channels), device)
}

fn gelu(xs: &Tensor) -> Result<Tensor> {
    xs.gelu_erf()
}

fn audio_seq_bucket_size() -> Option<usize> {
    match std::env::var("VASR_QWEN3_ASR_AUDIO_SEQ_BUCKET") {
        Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") => None,
        Ok(v) => v.parse::<usize>().ok().filter(|&n| n > 1),
        Err(_) => None,
    }
}

fn round_up_to_multiple(value: usize, multiple: usize) -> Result<usize> {
    if multiple == 0 {
        candle_core::bail!("round_up_to_multiple requires multiple > 0");
    }
    if value == 0 {
        return Ok(0);
    }
    let rem = value % multiple;
    if rem == 0 {
        return Ok(value);
    }
    value
        .checked_add(multiple - rem)
        .ok_or_else(|| candle_core::Error::Msg("bucket size overflow".to_string()))
}

// Bound each grouped score tensor to at most eight windows, regardless of
// how many samples/windows are packed into the encoder batch.
const AUDIO_ATTN_GROUP_MAX_WINDOWS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioAttentionMode {
    Reference,
    ProjectOnce,
    Grouped,
    GroupedNative,
}

impl AudioAttentionMode {
    fn from_env() -> Result<Self> {
        match std::env::var("VASR_QWEN3_ASR_AUDIO_ATTN") {
            Err(std::env::VarError::NotPresent) => Ok(Self::Reference),
            Ok(value) => match value.as_str() {
                "reference" => Ok(Self::Reference),
                "project-once" | "project_once" => Ok(Self::ProjectOnce),
                "grouped" => Ok(Self::Grouped),
                "grouped-native" => Ok(Self::GroupedNative),
                _ => candle_core::bail!(
                    "invalid VASR_QWEN3_ASR_AUDIO_ATTN={value:?}; expected reference, project-once, grouped, or grouped-native"
                ),
            },
            Err(error) => candle_core::bail!("invalid VASR_QWEN3_ASR_AUDIO_ATTN: {error}"),
        }
    }
}

#[derive(Debug, Clone)]
struct AudioAttention {
    mode: AudioAttentionMode,
    use_flash_attn: bool,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    q_proj: IsqLinear,
    k_proj: IsqLinear,
    v_proj: IsqLinear,
    out_proj: IsqLinear,
}

impl AudioAttention {
    fn load(
        cfg: &AudioEncoderConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
        isq: Option<&str>,
    ) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let num_heads = cfg.encoder_attention_heads;
        if num_heads == 0 {
            candle_core::bail!("encoder_attention_heads must be > 0");
        }
        if !embed_dim.is_multiple_of(num_heads) {
            candle_core::bail!("d_model must be divisible by num_heads");
        }
        let head_dim = embed_dim / num_heads;
        let q_proj = isq_linear::linear_b(embed_dim, embed_dim, true, vb.pp("q_proj"), isq)?;
        let k_proj = isq_linear::linear_b(embed_dim, embed_dim, true, vb.pp("k_proj"), isq)?;
        let v_proj = isq_linear::linear_b(embed_dim, embed_dim, true, vb.pp("v_proj"), isq)?;
        let out_proj = isq_linear::linear_b(embed_dim, embed_dim, true, vb.pp("out_proj"), isq)?;
        Ok(Self {
            mode: AudioAttentionMode::from_env()?,
            use_flash_attn,
            num_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    fn project_qkv(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        {
            if let Some(qkv) = isq_linear::try_fused_q8_qkv(
                &self.q_proj,
                &self.k_proj,
                &self.v_proj,
                hidden_states,
            )? {
                return Ok(qkv);
            }
        }

        Ok((
            self.q_proj.forward(hidden_states)?,
            self.k_proj.forward(hidden_states)?,
            self.v_proj.forward(hidden_states)?,
        ))
    }

    fn forward(&self, hidden_states: Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let (seq_len, embed_dim) = hidden_states.dims2()?;
        let dtype = hidden_states.dtype();
        let device = hidden_states.device().clone();
        if seq_len == 0 {
            return Tensor::zeros((0usize, embed_dim), dtype, &device);
        }

        let (q, k, v) = {
            let hidden_states = hidden_states;
            crate::profiling::measure("encoder_qkv", &device, || self.project_qkv(&hidden_states))?
        };
        let q = q
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .detach();

        let k = k
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .detach();

        let v = v
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .detach();

        if self.use_flash_attn {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = cu_seqlens;
                candle_core::bail!("flash-attn support is not enabled in this build");
            }

            #[cfg(feature = "flash-attn")]
            {
                if cu_seqlens.len() < 2 {
                    candle_core::bail!(
                        "cu_seqlens must have at least 2 elements, got {}",
                        cu_seqlens.len()
                    );
                }
                if cu_seqlens.first().copied().unwrap_or(1) != 0 {
                    candle_core::bail!("cu_seqlens must start at 0, got {:?}", cu_seqlens.first());
                }
                if cu_seqlens.last().copied().unwrap_or_default() != seq_len {
                    candle_core::bail!(
                        "cu_seqlens last must equal seq_len: last={:?} seq_len={seq_len}",
                        cu_seqlens.last().copied()
                    );
                }
                for w in cu_seqlens.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    if b < a {
                        candle_core::bail!("cu_seqlens must be non-decreasing: {cu_seqlens:?}");
                    }
                }

                if !device.is_cuda() {
                    candle_core::bail!(
                        "flash-attn requires a CUDA device, got {:?}",
                        device.location()
                    );
                }
                match dtype {
                    DType::BF16 => {}
                    other => {
                        candle_core::bail!("vasr-flash-attn requires dtype bf16, got {other:?}")
                    }
                }

                let mut max_len = 0usize;
                for w in cu_seqlens.windows(2) {
                    let len = w[1].saturating_sub(w[0]);
                    max_len = max_len.max(len);
                }
                if max_len == 0 {
                    candle_core::bail!("invalid cu_seqlens: max_len=0 for seq_len={seq_len}");
                }

                let mut cu_u32: Vec<u32> = Vec::with_capacity(cu_seqlens.len());
                for (i, &v) in cu_seqlens.iter().enumerate() {
                    cu_u32.push(u32::try_from(v).map_err(|_| {
                        candle_core::Error::Msg(format!(
                            "cu_seqlens overflows u32 at index {i}: value={v}"
                        ))
                    })?);
                }
                let cu = Tensor::from_vec(cu_u32, (cu_seqlens.len(),), &device)?;

                let softmax_scale = self.scaling as f32;
                let attn = {
                    let q = q;
                    let k = k;
                    let v = v;
                    flash_attn_varlen(&q, &k, &v, &cu, &cu, max_len, max_len, softmax_scale, false)?
                        .reshape((seq_len, embed_dim))?
                };
                return Ok(self.out_proj.forward(&attn)?.detach());
            }
        }

        if cu_seqlens.len() < 2 {
            candle_core::bail!(
                "cu_seqlens must have at least 2 elements, got {}",
                cu_seqlens.len()
            );
        }
        if cu_seqlens.first().copied().unwrap_or(1) != 0
            || cu_seqlens.last().copied().unwrap_or_default() != seq_len
        {
            candle_core::bail!("invalid cu_seqlens={cu_seqlens:?} for seq_len={seq_len}");
        }

        if self.mode != AudioAttentionMode::Reference {
            return self.forward_optimized(&q, &k, &v, cu_seqlens, embed_dim);
        }

        let outs = {
            let q = q;
            let k = k;
            let v = v;
            let mut outs: Vec<Tensor> = Vec::with_capacity(cu_seqlens.len().saturating_sub(1));
            for w in cu_seqlens.windows(2) {
                let (start, end) = (w[0], w[1]);
                if end < start {
                    candle_core::bail!("cu_seqlens must be non-decreasing: {cu_seqlens:?}");
                }
                let len = end.saturating_sub(start);
                if len == 0 {
                    continue;
                }
                let q_seg = q.narrow(0, start, len)?;
                let k_seg = k.narrow(0, start, len)?;
                let v_seg = v.narrow(0, start, len)?;

                let q_seg = q_seg.transpose(0, 1)?.unsqueeze(0)?; // (1, h, s, d)
                let k_seg = k_seg.transpose(0, 1)?.unsqueeze(0)?; // (1, h, s, d)
                let v_seg = v_seg.transpose(0, 1)?.unsqueeze(0)?; // (1, h, s, d)

                // GPU GEMM kernels only support contiguous (or simple transposed) layouts.
                // These tensors are produced via reshapes/transposes that yield strided views,
                // so materialize contiguous buffers when running on Metal or CUDA.
                let q_seg = maybe_contiguous_for_metal(q_seg)?;
                let v_seg = maybe_contiguous_for_metal(v_seg)?;

                let attn = crate::profiling::measure("encoder_attention", q_seg.device(), || {
                    let k_t = maybe_contiguous_for_metal(k_seg.transpose(2, 3)?)?;
                    let attn_weights = q_seg.matmul(&k_t)?.affine(self.scaling, 0.0)?;
                    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                    attn_weights.matmul(&v_seg)
                })?;
                let attn = attn.squeeze(0)?.transpose(0, 1)?; // (s, h, d)
                let attn = attn.reshape((len, embed_dim))?.detach();
                outs.push(
                    crate::profiling::measure("encoder_out_proj", &device, || {
                        self.out_proj.forward(&attn)
                    })?
                    .detach(),
                );
            }
            outs
        };
        if outs.is_empty() {
            return Tensor::zeros((0usize, embed_dim), dtype, &device);
        }
        let attn_refs: Vec<&Tensor> = outs.iter().collect();
        let out = Tensor::cat(&attn_refs, 0)?.detach();
        Ok(out)
    }

    fn attend_batched(&self, q: Tensor, k: Tensor, v: Tensor) -> Result<Tensor> {
        // Inputs (batch, time, heads, head_dim); no window is padded or merged.
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.transpose(2, 3)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let output = crate::profiling::measure("encoder_attention", q.device(), || {
            let weights = q.matmul(&k)?.affine(self.scaling, 0.0)?;
            let weights = candle_nn::ops::softmax_last_dim(&weights)?;
            weights.matmul(&v)
        })?;
        output.transpose(1, 2)?.contiguous()
    }

    fn forward_optimized(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cu: &[usize],
        embed_dim: usize,
    ) -> Result<Tensor> {
        let mut windows = Vec::new();
        for pair in cu.windows(2) {
            if pair[1] < pair[0] {
                candle_core::bail!("cu_seqlens must be non-decreasing: {cu:?}");
            }
            let len = pair[1] - pair[0];
            if len > 0 {
                windows.push((pair[0], len));
            }
        }
        let mut outputs: Vec<Option<Tensor>> = vec![None; windows.len()];
        if self.mode == AudioAttentionMode::ProjectOnce {
            for (index, &(start, len)) in windows.iter().enumerate() {
                let output = self.attend_batched(
                    q.narrow(0, start, len)?.unsqueeze(0)?,
                    k.narrow(0, start, len)?.unsqueeze(0)?,
                    v.narrow(0, start, len)?.unsqueeze(0)?,
                )?;
                outputs[index] = Some(output.reshape((len, embed_dim))?.detach());
            }
        } else {
            let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
            for (index, &(_, len)) in windows.iter().enumerate() {
                groups.entry(len).or_default().push(index);
            }
            for (len, indices) in groups {
                for chunk in indices.chunks(AUDIO_ATTN_GROUP_MAX_WINDOWS) {
                    let stack = |source: &Tensor| -> Result<Tensor> {
                        let segments = chunk
                            .iter()
                            .map(|&index| source.narrow(0, windows[index].0, len))
                            .collect::<Result<Vec<_>>>()?;
                        // Native BF16 copy is characterized against the compatibility
                        // path on pinned Candle; keep it an independent experiment.
                        stack_audio_chunks(
                            &segments,
                            self.mode == AudioAttentionMode::GroupedNative,
                        )
                    };
                    let output = self.attend_batched(stack(q)?, stack(k)?, stack(v)?)?;
                    for (batch_index, &window_index) in chunk.iter().enumerate() {
                        outputs[window_index] = Some(
                            output
                                .narrow(0, batch_index, 1)?
                                .reshape((len, embed_dim))?
                                .detach(),
                        );
                    }
                }
            }
        }
        let outputs = outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    candle_core::Error::Msg("missing grouped attention window".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let joined = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)?;
        Ok(self.out_proj.forward(&joined)?.detach())
    }
}

#[derive(Debug, Clone)]
struct AudioEncoderLayer {
    self_attn: AudioAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: IsqLinear,
    fc2: IsqLinear,
    final_layer_norm: LayerNorm,
    #[cfg(feature = "metal-paged-attn")]
    fused_fc1_gelu: bool,
}

impl AudioEncoderLayer {
    fn load(
        cfg: &AudioEncoderConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
        isq: Option<&str>,
    ) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let self_attn = AudioAttention::load(cfg, vb.pp("self_attn"), use_flash_attn, isq)?;
        let self_attn_layer_norm = layer_norm(
            embed_dim,
            candle_nn::LayerNormConfig::from(LAYER_NORM_EPS),
            vb.pp("self_attn_layer_norm"),
        )?;
        let final_layer_norm = layer_norm(
            embed_dim,
            candle_nn::LayerNormConfig::from(LAYER_NORM_EPS),
            vb.pp("final_layer_norm"),
        )?;
        let fc1 = isq_linear::linear_b(embed_dim, cfg.encoder_ffn_dim, true, vb.pp("fc1"), isq)?;
        let fc2 = isq_linear::linear_b(cfg.encoder_ffn_dim, embed_dim, true, vb.pp("fc2"), isq)?;
        #[cfg(feature = "metal-paged-attn")]
        let fused_fc1_gelu = vb.device().is_metal()
            && ConvExecutionPolicy::parse_override(
                std::env::var("VASR_METAL_ENCODER_BIAS_GELU")
                    .ok()
                    .as_deref(),
                "VASR_METAL_ENCODER_BIAS_GELU",
            )?
            .unwrap_or(false);
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            #[cfg(feature = "metal-paged-attn")]
            fused_fc1_gelu,
        })
    }

    fn fc1_gelu(&self, normed: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal-paged-attn")]
        if self.fused_fc1_gelu && normed.device().is_metal() {
            if let IsqLinear::Linear(linear) = &self.fc1 {
                if let Some(bias) = linear.bias() {
                    // Keep the original dense matmul; the fused operator explicitly
                    // rounds the bias addition to the input dtype before GELU.
                    let projected = candle_nn::Linear::new(linear.weight().clone(), None)
                        .forward(normed)?
                        .detach();
                    return vasr_paged_attn::metal::conv::bias_gelu_last_dim(&projected, bias)
                        .map(|x| x.detach());
                }
            }
        }
        gelu(&self.fc1.forward(normed)?.detach()).map(|x| x.detach())
    }

    fn forward(&self, hidden_states: Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
        let x = {
            let residual = hidden_states;
            let attn = {
                let normed = self.self_attn_layer_norm.forward(&residual)?.detach();
                self.self_attn.forward(normed, cu_seqlens)?.detach()
            };
            (&residual + &attn)?.detach()
        };

        let ffn_device = x.device().clone();
        let x = crate::profiling::measure("encoder_ffn", &ffn_device, || {
            let residual = x;
            let activated = {
                let normed = self.final_layer_norm.forward(&residual)?.detach();
                #[cfg(feature = "cuda")]
                {
                    if let Some(activated) = isq_linear::try_fused_q8_gelu_erf(&self.fc1, &normed)?
                    {
                        activated.detach()
                    } else {
                        self.fc1_gelu(&normed)?
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    self.fc1_gelu(&normed)?
                }
            };
            let fc2 = self.fc2.forward(&activated)?.detach();
            Ok::<_, candle_core::Error>((&residual + &fc2)?.detach())
        })?;

        let x = if x.dtype() == DType::F16 {
            // Match the official float16 clamp to prevent infs.
            let clamp = 65_504.0f32 - 1000.0;
            x.clamp(-clamp, clamp)?
        } else {
            x
        };
        Ok(x.detach())
    }
}

#[derive(Debug, Clone)]
pub struct AudioTower {
    config: AudioEncoderConfig,
    positional_embedding: Tensor,
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: IsqLinear,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    proj1: IsqLinear,
    proj2: IsqLinear,
    use_flash_attn: bool,
    conv_micro_batch: usize,
    conv_cross_sample: bool,
    conv_policy: ConvExecutionPolicy,
}

struct AudioConvSample {
    hidden_states: Tensor,
    aftercnn_total: usize,
    max_t: usize,
    chunks: usize,
}

impl AudioTower {
    pub(crate) fn set_auto_conv_layout(&mut self, enabled: bool) {
        self.conv_policy.auto = enabled;
        #[cfg(feature = "metal-paged-attn")]
        if enabled {
            if std::env::var_os("VASR_METAL_CONV_GEMM_TILE").is_none() {
                self.conv_policy.gemm_tile = Some(vasr_quant::metal_gemm::GemmTile::Narrow);
            }
            for layer in &mut self.layers {
                if std::env::var_os("VASR_QWEN3_ASR_AUDIO_ATTN").is_none() {
                    layer.self_attn.mode = AudioAttentionMode::GroupedNative;
                }
                if std::env::var_os("VASR_METAL_ENCODER_BIAS_GELU").is_none() {
                    layer.fused_fc1_gelu = true;
                }
            }
        }
        let (nhwc, im2col32) = self.conv_policy.resolve();
        let bias_gelu = self.conv_policy.fused_bias_gelu();
        let native_chunks = self.conv_policy.native_chunks();
        if enabled || nhwc {
            tracing::info!(target: "vasr::execution",
                "Metal CNN policy | nhwc={nhwc} im2col32={im2col32} bias_gelu={bias_gelu} native_chunks={native_chunks} explicit_overrides=preserved");
        }
        #[cfg(feature = "metal-paged-attn")]
        tracing::info!(target: "vasr::execution", "Metal encoder policy | conv_tile={:?} attention={:?} fc1_bias_gelu={} overrides=preserved", self.conv_policy.gemm_tile, self.layers.first().map(|layer| layer.self_attn.mode), self.layers.first().is_some_and(|layer| layer.fused_fc1_gelu));
    }

    pub fn load(
        cfg: &AudioEncoderConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
        isq: Option<&str>,
        progress: &Progress,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let conv_policy = if device.is_metal() {
            let nhwc = std::env::var("VASR_METAL_CONV_NHWC").ok();
            let im2col32 = std::env::var("VASR_METAL_CONV_IM2COL32").ok();
            let bias_gelu = std::env::var("VASR_METAL_CONV_BIAS_GELU").ok();
            let mut policy =
                ConvExecutionPolicy::from_values(nhwc.as_deref(), im2col32.as_deref())?;
            policy.bias_gelu_override = ConvExecutionPolicy::parse_override(
                bias_gelu.as_deref(),
                "VASR_METAL_CONV_BIAS_GELU",
            )?;
            policy.native_chunks_override = ConvExecutionPolicy::parse_override(
                std::env::var("VASR_METAL_CONV_NATIVE_CHUNKS")
                    .ok()
                    .as_deref(),
                "VASR_METAL_CONV_NATIVE_CHUNKS",
            )?;
            #[cfg(feature = "metal-paged-attn")]
            {
                policy.gemm_tile = std::env::var("VASR_METAL_CONV_GEMM_TILE")
                    .ok()
                    .map(|value| vasr_quant::metal_gemm::GemmTile::parse(&value))
                    .transpose()?
                    .flatten();
            }
            policy
        } else {
            ConvExecutionPolicy::default()
        };
        let conv_cross_sample = if device.is_metal() {
            match std::env::var("VASR_METAL_CONV_CROSS_SAMPLE") {
                Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "yes" | "on" => true,
                    "0" | "false" | "no" | "off" => false,
                    _ => candle_core::bail!("VASR_METAL_CONV_CROSS_SAMPLE must be 0 or 1"),
                },
                Err(std::env::VarError::NotPresent) => false,
                Err(error) => candle_core::bail!("invalid VASR_METAL_CONV_CROSS_SAMPLE: {error}"),
            }
        } else {
            false
        };
        let conv_micro_batch = if device.is_metal() {
            match std::env::var("VASR_METAL_CONV_CHUNKS") {
                Ok(value) => {
                    let value = value.parse::<usize>().map_err(|_| {
                        candle_core::Error::Msg("VASR_METAL_CONV_CHUNKS must be 1..=64".into())
                    })?;
                    if !(1..=64).contains(&value) {
                        candle_core::bail!("VASR_METAL_CONV_CHUNKS must be 1..=64");
                    }
                    value
                }
                Err(std::env::VarError::NotPresent) => cfg.conv_chunksize.max(1),
                Err(error) => candle_core::bail!("invalid VASR_METAL_CONV_CHUNKS: {error}"),
            }
        } else {
            cfg.conv_chunksize.max(1)
        };

        let positional_embedding =
            sinusoidal_positional_embedding(cfg.max_source_positions, cfg.d_model, &device)?;

        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            ..Conv2dConfig::default()
        };
        let conv2d1 = conv2d(1, cfg.downsample_hidden_size, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2d2 = conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d2"),
        )?;
        let conv2d3 = conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d3"),
        )?;

        let mel_after = cfg.num_mel_bins.div_ceil(2).div_ceil(2).div_ceil(2);
        let conv_out_in = cfg.downsample_hidden_size * mel_after;
        let conv_out =
            isq_linear::linear_no_bias(conv_out_in, cfg.d_model, vb.pp("conv_out"), isq)?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        let phase = if isq.is_some() {
            "quantize.audio_layers"
        } else {
            "load.audio_layers"
        };
        let progress = progress.start(ProgressEvent::layers(
            "qwen3-asr",
            phase,
            cfg.encoder_layers,
        ));
        for i in 0..cfg.encoder_layers {
            layers.push(AudioEncoderLayer::load(
                cfg,
                vb.pp("layers").pp(i),
                use_flash_attn,
                isq,
            )?);
            progress.inc(1);
        }
        progress.finish();

        let ln_post = layer_norm(
            cfg.d_model,
            candle_nn::LayerNormConfig::from(LAYER_NORM_EPS),
            vb.pp("ln_post"),
        )?;
        let proj1 = isq_linear::linear_b(cfg.d_model, cfg.d_model, true, vb.pp("proj1"), isq)?;
        let proj2 = isq_linear::linear_b(cfg.d_model, cfg.output_dim, true, vb.pp("proj2"), isq)?;

        Ok(Self {
            config: cfg.clone(),
            conv_micro_batch,
            conv_cross_sample,
            conv_policy,
            positional_embedding,
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj2,
            use_flash_attn,
        })
    }

    pub fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }

    pub fn uses_flash_attn(&self) -> bool {
        self.use_flash_attn
    }

    pub fn forward(
        &self,
        input_features: &Tensor,
        feature_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, _mel, frames) = input_features.dims3()?;
        let feature_lens = match feature_attention_mask {
            None => vec![frames; batch],
            Some(mask) => {
                let (b2, t2) = mask.dims2()?;
                if (b2, t2) != (batch, frames) {
                    candle_core::bail!(
                        "feature_attention_mask dims mismatch: expected=({batch},{frames}), got=({b2},{t2})"
                    );
                }
                feature_lens_from_mask(mask, frames)?
            }
        };
        self.forward_with_lens(input_features, feature_lens.as_slice())
    }

    pub fn forward_with_lens(
        &self,
        input_features: &Tensor,
        feature_lens: &[usize],
    ) -> Result<Tensor> {
        let want_dtype = self.conv2d1.weight().dtype();
        let input_features = cast_input_features_to_weight_dtype(input_features, want_dtype)?;

        let (batch, mel, frames) = input_features.dims3()?;
        if mel != self.config.num_mel_bins {
            candle_core::bail!(
                "input_features mel bins mismatch: expected={}, got={mel}",
                self.config.num_mel_bins
            );
        }
        if feature_lens.len() != batch {
            candle_core::bail!(
                "feature_lens batch mismatch: expected={batch}, got={}",
                feature_lens.len()
            );
        }
        for (i, &len) in feature_lens.iter().enumerate() {
            if len > frames {
                candle_core::bail!("feature_lens[{i}] exceeds frames: len={len} frames={frames}");
            }
        }

        log_cuda_mem(
            input_features.device(),
            &format!("audio_encoder_batch_start(batch={batch},frames={frames})"),
        );

        let window = self
            .config
            .n_window
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::Msg("n_window overflow".to_string()))?;
        if window == 0 {
            candle_core::bail!("audio_config.n_window must be > 0");
        }

        let mut sample_hiddens: Vec<Tensor> = Vec::with_capacity(batch);
        let mut sample_aftercnn_totals: Vec<usize> = Vec::with_capacity(batch);
        let mut sample_max_ts: Vec<usize> = Vec::with_capacity(batch);
        let mut total_chunks = 0usize;

        if self.conv_cross_sample {
            let samples = crate::profiling::measure(
                "encoder_conv_cross_sample",
                input_features.device(),
                || self.forward_conv_cross_sample(&input_features, feature_lens, window),
            )?;
            for sample in samples {
                total_chunks = total_chunks.saturating_add(sample.chunks);
                sample_aftercnn_totals.push(sample.aftercnn_total);
                sample_max_ts.push(sample.max_t);
                if sample.aftercnn_total > 0 {
                    sample_hiddens.push(sample.hidden_states);
                }
            }
        } else {
            for (sample_idx, &sample_len) in feature_lens.iter().enumerate() {
                if sample_len == 0 {
                    sample_aftercnn_totals.push(0);
                    sample_max_ts.push(0);
                    continue;
                }

                let input_feature = input_features.narrow(0, sample_idx, 1)?.squeeze(0)?;
                let input_feature = input_feature.narrow(1, 0, sample_len)?;
                let sample = crate::profiling::measure(
                    "encoder_conv_sample",
                    input_feature.device(),
                    || self.forward_conv_sample(&input_feature, window),
                )?;
                total_chunks = total_chunks.saturating_add(sample.chunks);
                sample_aftercnn_totals.push(sample.aftercnn_total);
                sample_max_ts.push(sample.max_t);
                sample_hiddens.push(sample.hidden_states);
            }
        }

        if sample_hiddens.is_empty() {
            return Tensor::zeros(
                (0usize, self.config.output_dim),
                input_features.dtype(),
                input_features.device(),
            );
        }

        let idx_len: usize = sample_aftercnn_totals.iter().sum();
        let mut hidden_states = {
            let sample_hiddens = sample_hiddens;
            let sample_refs: Vec<&Tensor> = sample_hiddens.iter().collect();
            Tensor::cat(sample_refs.as_slice(), 0)?
        };
        let padded_idx_len = match audio_seq_bucket_size() {
            Some(bucket) => round_up_to_multiple(idx_len, bucket)?,
            None => idx_len,
        };
        hidden_states = if padded_idx_len > idx_len {
            hidden_states.pad_with_zeros(0, 0, padded_idx_len - idx_len)?
        } else {
            hidden_states
        };
        log_cuda_mem(
            input_features.device(),
            &format!(
                "audio_encoder_after_conv_samples(batch={batch},chunks={total_chunks},seq={idx_len},padded_seq={padded_idx_len})"
            ),
        );

        let mut cu_seqlens = Vec::new();
        cu_seqlens.push(0usize);
        let mut seq_offset = 0usize;
        for (sample_idx, (&aftercnn_total, &max_t)) in sample_aftercnn_totals
            .iter()
            .zip(sample_max_ts.iter())
            .enumerate()
        {
            if aftercnn_total == 0 {
                continue;
            }
            let local = cu_seqlens_from_aftercnn_len(
                aftercnn_total,
                max_t,
                window,
                self.config.n_window_infer,
            )?;
            for &end in local.iter().skip(1) {
                cu_seqlens.push(seq_offset.checked_add(end).ok_or_else(|| {
                    candle_core::Error::Msg("cu_seqlens offset overflow".to_string())
                })?);
            }
            seq_offset = seq_offset.checked_add(aftercnn_total).ok_or_else(|| {
                candle_core::Error::Msg("cu_seqlens sample offset overflow".to_string())
            })?;
            if cu_seqlens.last().copied().unwrap_or_default() != seq_offset {
                candle_core::bail!(
                    "cu_seqlens mismatch after sample={sample_idx}: last={:?} offset={seq_offset}",
                    cu_seqlens.last().copied()
                );
            }
        }
        if seq_offset != idx_len {
            candle_core::bail!(
                "batch aftercnn length mismatch: expected={seq_offset}, got={idx_len}"
            );
        }
        if padded_idx_len > idx_len {
            cu_seqlens.push(padded_idx_len);
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer
                .forward(hidden_states, cu_seqlens.as_slice())?
                .detach();
            if std::env::var_os("VASR_DEBUG_AUDIO_ENCODER_LAYERS").is_some() {
                log_cuda_mem(
                    input_features.device(),
                    &format!("audio_encoder_after_layer_{layer_idx}"),
                );
            }
        }
        log_cuda_mem(
            input_features.device(),
            &format!(
                "audio_encoder_after_layers(batch={batch},seq={idx_len},padded_seq={padded_idx_len})"
            ),
        );

        hidden_states = self.ln_post.forward(&hidden_states)?;
        hidden_states = self.proj1.forward(&hidden_states)?;
        hidden_states = gelu(&hidden_states)?;
        hidden_states = self.proj2.forward(&hidden_states)?;
        log_cuda_mem(
            input_features.device(),
            &format!(
                "audio_encoder_after_proj(batch={batch},seq={idx_len},padded_seq={padded_idx_len})"
            ),
        );
        Ok(hidden_states.narrow(0, 0, idx_len)?.detach())
    }

    fn forward_conv_layers(&self, x: &Tensor) -> Result<Tensor> {
        let x = crate::profiling::measure("encoder_conv1", x.device(), || {
            conv_gelu_forward(&self.conv2d1, x, &self.conv_policy)
        })?;
        let x = crate::profiling::measure("encoder_conv2", x.device(), || {
            conv_gelu_forward(&self.conv2d2, &x, &self.conv_policy)
        })?;
        let x = crate::profiling::measure("encoder_conv3", x.device(), || {
            conv_gelu_forward(&self.conv2d3, &x, &self.conv_policy)
        })?;
        Ok(x)
    }

    fn project_conv_chunks(&self, mut x: Tensor) -> Result<Tensor> {
        let (b, c, f, t) = x.dims4()?;
        x = x.transpose(1, 3)?.transpose(2, 3)?.contiguous()?;
        x = x.reshape((b, t, c * f))?;
        x = self.conv_out.forward(&x)?;

        let pe = self.positional_embedding.narrow(0, 0, t)?.unsqueeze(0)?;
        let pe = pe.to_dtype(x.dtype())?;
        x = x.broadcast_add(&pe)?;

        Ok(x)
    }

    /// Stream padded chunks in sample order; only this CNN chunk dimension is
    /// batched across samples. Position ids still restart independently per chunk.
    fn forward_conv_cross_sample(
        &self,
        features: &Tensor,
        lens: &[usize],
        window: usize,
    ) -> Result<Vec<AudioConvSample>> {
        let (batch, mel, frames) = features.dims3()?;
        if batch != lens.len()
            || mel != self.config.num_mel_bins
            || window == 0
            || lens.iter().any(|&len| len > frames)
        {
            candle_core::bail!("invalid cross-sample CNN input dimensions/lengths/window");
        }
        let limit = self.conv_micro_batch.clamp(1, 64);
        let mut chunks = Vec::with_capacity(limit);
        let mut metadata = Vec::with_capacity(limit);
        let mut outputs: Vec<Vec<Tensor>> = vec![Vec::new(); batch];
        let mut totals = vec![0usize; batch];
        let mut max_ts = vec![0usize; batch];
        let mut counts = vec![0usize; batch];
        for (sample_index, &len) in lens.iter().enumerate() {
            if len == 0 {
                continue;
            }
            let input = features
                .narrow(0, sample_index, 1)?
                .squeeze(0)?
                .narrow(1, 0, len)?
                .transpose(0, 1)?;
            for start in (0..len).step_by(window) {
                let chunk_len = (len - start).min(window);
                let aftercnn_len = feat_extract_output_length(chunk_len);
                chunks.push(pad_audio_chunk(
                    &input.narrow(0, start, chunk_len)?,
                    0,
                    0,
                    window - chunk_len,
                    self.conv_policy.native_chunks(),
                )?);
                metadata.push((sample_index, aftercnn_len));
                totals[sample_index] =
                    totals[sample_index]
                        .checked_add(aftercnn_len)
                        .ok_or_else(|| {
                            candle_core::Error::Msg("cross-sample aftercnn length overflow".into())
                        })?;
                counts[sample_index] += 1;
                if chunks.len() == limit {
                    self.flush_conv_chunks(&chunks, &metadata, &mut outputs, &mut max_ts)?;
                    chunks.clear();
                    metadata.clear();
                }
            }
            if totals[sample_index] != feat_extract_output_length(len) {
                candle_core::bail!(
                    "cross-sample aftercnn length mismatch for sample={sample_index}"
                );
            }
        }
        if !chunks.is_empty() {
            self.flush_conv_chunks(&chunks, &metadata, &mut outputs, &mut max_ts)?;
        }
        let mut samples = Vec::with_capacity(batch);
        for (index, parts) in outputs.into_iter().enumerate() {
            let hidden_states = if parts.is_empty() {
                Tensor::zeros(
                    (0usize, self.config.d_model),
                    features.dtype(),
                    features.device(),
                )?
            } else {
                Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0)?.detach()
            };
            if hidden_states.dim(0)? != totals[index] {
                candle_core::bail!("cross-sample output length mismatch for sample={index}");
            }
            samples.push(AudioConvSample {
                hidden_states,
                aftercnn_total: totals[index],
                max_t: max_ts[index],
                chunks: counts[index],
            });
        }
        Ok(samples)
    }

    fn flush_conv_chunks(
        &self,
        chunks: &[Tensor],
        metadata: &[(usize, usize)],
        outputs: &mut [Vec<Tensor>],
        max_ts: &mut [usize],
    ) -> Result<()> {
        let packed = stack_audio_chunks(chunks, self.conv_policy.native_chunks())?
            .transpose(1, 2)?
            .unsqueeze(1)?;
        let x = self.project_conv_chunks(self.forward_conv_layers(&packed)?)?;
        let (_, max_t, _) = x.dims3()?;
        for (chunk_index, &(sample_index, valid_t)) in metadata.iter().enumerate() {
            if valid_t > max_t {
                candle_core::bail!("cross-sample aftercnn_len={valid_t} exceeds max_t={max_t}");
            }
            max_ts[sample_index] = max_t;
            outputs[sample_index].push(
                x.narrow(0, chunk_index, 1)?
                    .squeeze(0)?
                    .narrow(0, 0, valid_t)?
                    .detach(),
            );
        }
        Ok(())
    }

    fn forward_conv_sample(
        &self,
        input_feature: &Tensor,
        window: usize,
    ) -> Result<AudioConvSample> {
        let (mel, frames) = input_feature.dims2()?;
        if mel != self.config.num_mel_bins {
            candle_core::bail!(
                "input_feature mel bins mismatch: expected={}, got={mel}",
                self.config.num_mel_bins
            );
        }
        if frames == 0 {
            return Ok(AudioConvSample {
                hidden_states: Tensor::zeros(
                    (0usize, self.config.d_model),
                    input_feature.dtype(),
                    input_feature.device(),
                )?,
                aftercnn_total: 0,
                max_t: 0,
                chunks: 0,
            });
        }

        let input_t = input_feature.transpose(0, 1)?;
        let chunk_num = frames.div_ceil(window);
        let mut padded_chunks: Vec<Tensor> = Vec::with_capacity(chunk_num);
        let mut aftercnn_lens: Vec<usize> = Vec::with_capacity(chunk_num);
        let mut aftercnn_total = 0usize;
        let mut offset = 0usize;

        for chunk_idx in 0..chunk_num {
            let remaining = frames.saturating_sub(offset);
            let chunk_len = remaining.min(window);
            if chunk_len == 0 {
                candle_core::bail!("invalid zero chunk length: chunk={chunk_idx}");
            }
            let chunk = input_t.narrow(0, offset, chunk_len)?;
            let pad = window.saturating_sub(chunk_len);
            padded_chunks.push(pad_audio_chunk(
                &chunk,
                0,
                0,
                pad,
                self.conv_policy.native_chunks(),
            )?);
            let aftercnn_len = feat_extract_output_length(chunk_len);
            aftercnn_lens.push(aftercnn_len);
            aftercnn_total = aftercnn_total.saturating_add(aftercnn_len);
            offset = offset.saturating_add(chunk_len);
        }
        if offset != frames {
            candle_core::bail!(
                "chunk split did not consume full sequence: offset={offset} frames={frames}"
            );
        }
        let expected_aftercnn_total = feat_extract_output_length(frames);
        if aftercnn_total != expected_aftercnn_total {
            candle_core::bail!(
                "aftercnn length mismatch: expected={expected_aftercnn_total}, got={aftercnn_total}"
            );
        }

        let padded_feature = {
            let padded_chunks = padded_chunks;
            stack_audio_chunks(&padded_chunks, self.conv_policy.native_chunks())?
                .transpose(1, 2)?
                .unsqueeze(1)?
        };

        let conv_outs = {
            let padded_feature = padded_feature;
            let mut conv_outs: Vec<Tensor> = Vec::new();
            let mut start = 0usize;
            while start < chunk_num {
                let len = (chunk_num - start).min(self.conv_micro_batch);
                let x = padded_feature.narrow(0, start, len)?;
                let x = self.forward_conv_layers(&x)?;
                conv_outs.push(x.detach());
                start = start.saturating_add(len);
            }
            conv_outs
        };

        let mut x = {
            let conv_outs = conv_outs;
            if conv_outs.len() == 1 {
                conv_outs
                    .first()
                    .ok_or_else(|| candle_core::Error::Msg("missing conv output".to_string()))?
                    .clone()
            } else {
                Tensor::cat(&conv_outs.iter().collect::<Vec<_>>(), 0)?
            }
        };

        x = self.project_conv_chunks(x)?;
        let (b, t, _) = x.dims3()?;

        let max_t = t;
        let hidden_states = {
            let x = x;
            let flat = x.reshape((b * max_t, self.config.d_model))?;
            let mut idxs: Vec<u32> = Vec::with_capacity(aftercnn_total);
            for (chunk_idx, &len) in aftercnn_lens.iter().enumerate() {
                if len > max_t {
                    candle_core::bail!("aftercnn_len={len} exceeds max_t={max_t}");
                }
                let base = chunk_idx
                    .checked_mul(max_t)
                    .ok_or_else(|| candle_core::Error::Msg("index overflow".to_string()))?;
                for j in 0..len {
                    let pos = base
                        .checked_add(j)
                        .ok_or_else(|| candle_core::Error::Msg("index overflow".to_string()))?;
                    let pos_u32 = u32::try_from(pos)
                        .map_err(|_| candle_core::Error::Msg("index overflow".to_string()))?;
                    idxs.push(pos_u32);
                }
            }
            let idx_len = idxs.len();
            if idx_len != aftercnn_total {
                candle_core::bail!(
                    "sample aftercnn length mismatch: expected={aftercnn_total}, got={idx_len}"
                );
            }
            let idx = Tensor::from_vec(idxs, (idx_len,), input_feature.device())?;
            flat.index_select(&idx, 0)?.detach()
        };

        Ok(AudioConvSample {
            hidden_states,
            aftercnn_total,
            max_t,
            chunks: chunk_num,
        })
    }

    #[cfg(test)]
    fn forward_one(&self, input_feature: &Tensor) -> Result<Tensor> {
        let (mel, frames) = input_feature.dims2()?;
        log_cuda_mem(
            input_feature.device(),
            &format!("audio_encoder_one_start(frames={frames})"),
        );
        if mel != self.config.num_mel_bins {
            candle_core::bail!(
                "input_feature mel bins mismatch: expected={}, got={mel}",
                self.config.num_mel_bins
            );
        }

        let window = self
            .config
            .n_window
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::Msg("n_window overflow".to_string()))?;
        if window == 0 {
            candle_core::bail!("audio_config.n_window must be > 0");
        }

        let sample = self.forward_conv_sample(input_feature, window)?;
        if sample.chunks == 0 {
            return Tensor::zeros(
                (0usize, self.config.output_dim),
                input_feature.dtype(),
                input_feature.device(),
            );
        }
        let idx_len = sample.aftercnn_total;
        let max_t = sample.max_t;
        let mut hidden_states = sample.hidden_states;
        let padded_idx_len = match audio_seq_bucket_size() {
            Some(bucket) => round_up_to_multiple(idx_len, bucket)?,
            None => idx_len,
        };
        hidden_states = if padded_idx_len > idx_len {
            hidden_states.pad_with_zeros(0, 0, padded_idx_len - idx_len)?
        } else {
            hidden_states
        };
        log_cuda_mem(
            input_feature.device(),
            &format!(
                "audio_encoder_after_conv_sample(frames={frames},chunks={},seq={idx_len},padded_seq={padded_idx_len})",
                sample.chunks
            ),
        );

        // Encoder transformer layers.
        let aftercnn_total = feat_extract_output_length(frames);
        if aftercnn_total != idx_len {
            candle_core::bail!(
                "aftercnn length mismatch: expected={aftercnn_total}, got={idx_len}"
            );
        }
        let mut cu_seqlens = cu_seqlens_from_aftercnn_len(
            aftercnn_total,
            max_t,
            window,
            self.config.n_window_infer,
        )?;
        if padded_idx_len > idx_len {
            cu_seqlens.push(padded_idx_len);
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer
                .forward(hidden_states, cu_seqlens.as_slice())?
                .detach();
            if std::env::var_os("VASR_DEBUG_AUDIO_ENCODER_LAYERS").is_some() {
                log_cuda_mem(
                    input_feature.device(),
                    &format!("audio_encoder_after_layer_{layer_idx}"),
                );
            }
        }
        log_cuda_mem(
            input_feature.device(),
            &format!(
                "audio_encoder_after_layers(frames={frames},seq={idx_len},padded_seq={padded_idx_len})"
            ),
        );

        hidden_states = self.ln_post.forward(&hidden_states)?;
        hidden_states = self.proj1.forward(&hidden_states)?;
        hidden_states = gelu(&hidden_states)?;
        hidden_states = self.proj2.forward(&hidden_states)?;
        log_cuda_mem(
            input_feature.device(),
            &format!(
                "audio_encoder_after_proj(frames={frames},seq={idx_len},padded_seq={padded_idx_len})"
            ),
        );
        Ok(hidden_states.narrow(0, 0, idx_len)?.detach())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AudioAttention, AudioTower, cu_seqlens_from_aftercnn_len, feature_lens_from_mask,
        sinusoidal_positional_embedding,
    };
    use crate::config::AudioEncoderConfig;
    use crate::utils::progress::Progress;
    use candle_nn::Linear;
    use vasr_quant::isq_linear::IsqLinear;

    /// 确定性伪随机权重后端，用于验证多段 encoder 在非零权重下仍然 batch 不变。
    ///
    /// `VarBuilder::zeros` 会把所有中间量压成 0，padding / 窗口边界写错也看不出来；
    /// 这里按张量名做稳定哈希，生成非零、可复现且幅度适中的权重。
    struct SeededBackend;

    /// FNV-1a，把张量名映射成一个稳定的 64 位种子。
    fn fingerprint(name: &str) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in name.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    #[cfg(feature = "metal-paged-attn")]
    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn encoder_fc1_fusion_keeps_dense_projection_values() -> anyhow::Result<()> {
        use candle_core::{DType, Device, Tensor};
        let device = Device::new_metal(0)?;
        let cfg = test_config();
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let vb =
                candle_nn::VarBuilder::from_backend(Box::new(SeededBackend), dtype, device.clone());
            let mut layer = super::AudioEncoderLayer::load(&cfg, vb, false, None)?;
            for rows in [1, 17, 129] {
                let x = Tensor::from_vec(
                    (0..rows * cfg.d_model)
                        .map(|i| ((i * 13 % 193) as f32 - 96.) / 23.)
                        .collect::<Vec<_>>(),
                    (rows, cfg.d_model),
                    &device,
                )?
                .to_dtype(dtype)?;
                layer.fused_fc1_gelu = false;
                let expected = layer.fc1_gelu(&x)?;
                layer.fused_fc1_gelu = true;
                let actual = layer.fc1_gelu(&x)?;
                assert_tensor_close(&actual, &expected, 0.)?;
            }
        }
        Ok(())
    }

    impl candle_nn::var_builder::SimpleBackend for SeededBackend {
        fn get(
            &self,
            shape: candle_core::Shape,
            name: &str,
            _init: candle_nn::Init,
            dtype: candle_core::DType,
            device: &candle_core::Device,
        ) -> candle_core::Result<candle_core::Tensor> {
            let count = shape.elem_count();
            let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ fingerprint(name);
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                // xorshift64*：稳定、快速，不引入外部 RNG 依赖。
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let x = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
                // 映射到 [-0.5, 0.5)，避免随机权重把激活值推到溢出。
                values.push(((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32);
            }
            candle_core::Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
        }

        fn get_unchecked(
            &self,
            _name: &str,
            _dtype: candle_core::DType,
            _device: &candle_core::Device,
        ) -> candle_core::Result<candle_core::Tensor> {
            candle_core::bail!("SeededBackend 需要形状，请使用 get 而不是 get_unchecked")
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    /// 构造一个用于数值对比的测试配置。
    fn test_config() -> AudioEncoderConfig {
        AudioEncoderConfig {
            num_mel_bins: 4,
            encoder_layers: 1,
            encoder_attention_heads: 1,
            encoder_ffn_dim: 8,
            d_model: 4,
            dropout: 0.0,
            attention_dropout: 0.0,
            activation_function: "gelu".to_string(),
            activation_dropout: 0.0,
            scale_embedding: false,
            initializer_range: 0.02,
            max_source_positions: 64,
            n_window: 5,
            output_dim: 6,
            n_window_infer: 20,
            conv_chunksize: 8,
            downsample_hidden_size: 2,
        }
    }

    /// 用给定权重后端加载 tower，并按 batch-vs-per-sample 方式比较输出。
    fn assert_batched_matches_per_sample(
        device: &candle_core::Device,
        dtype: candle_core::DType,
        zeros: bool,
        tolerance: f32,
    ) -> anyhow::Result<()> {
        let cfg = test_config();
        let vb = if zeros {
            candle_nn::VarBuilder::zeros(dtype, device)
        } else {
            candle_nn::VarBuilder::from_backend(Box::new(SeededBackend), dtype, device.clone())
        };
        let tower = AudioTower::load(&cfg, vb, false, None, &Progress::disabled())?;

        let batch = 3usize;
        let frames = 10usize;
        let lens = [10usize, 7usize, 4usize];
        let feats: Vec<f32> = (0..batch * cfg.num_mel_bins * frames)
            .map(|i| (i as f32) / 100.0)
            .collect();
        let input_features =
            candle_core::Tensor::from_vec(feats, (batch, cfg.num_mel_bins, frames), device)?;

        let batched = tower.forward_with_lens(&input_features, &lens)?;
        let weight_dtype = tower.conv2d1.weight().dtype();
        let mut per_sample = Vec::with_capacity(batch);
        for (idx, &len) in lens.iter().enumerate() {
            let input = input_features
                .narrow(0, idx, 1)?
                .squeeze(0)?
                .narrow(1, 0, len)?;
            // `forward_one` 是测试辅助路径，不负责 dtype 转换，这里手动对齐权重 dtype。
            let input = super::cast_input_features_to_weight_dtype(&input, weight_dtype)?;
            per_sample.push(tower.forward_one(&input)?);
        }
        let expected = candle_core::Tensor::cat(&per_sample.iter().collect::<Vec<_>>(), 0)?;

        if batched.dims() != expected.dims() {
            anyhow::bail!(
                "batched dims mismatch: batched={:?} expected={:?}",
                batched.dims(),
                expected.dims()
            );
        }
        let got = batched
            .to_dtype(candle_core::DType::F32)?
            .to_vec2::<f32>()?;
        let exp = expected
            .to_dtype(candle_core::DType::F32)?
            .to_vec2::<f32>()?;
        let mut max_abs = 0f32;
        for (row_g, row_e) in got.iter().zip(exp.iter()) {
            for (&a, &b) in row_g.iter().zip(row_e.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        if max_abs > tolerance {
            anyhow::bail!("batched forward mismatch: max_abs={max_abs} tolerance={tolerance}");
        }
        Ok(())
    }

    #[test]
    fn fused_conv_policy_is_automatic_and_respects_explicit_off() -> anyhow::Result<()> {
        let mut enabled = super::ConvExecutionPolicy::from_values(Some("1"), None)?;
        enabled.bias_gelu_override = Some(true);
        assert!(enabled.fused_bias_gelu());
        enabled.nhwc_override = Some(false);
        enabled.auto = true;
        assert!(!enabled.fused_bias_gelu());
        let mut automatic = super::ConvExecutionPolicy::from_values(None, None)?;
        automatic.auto = true;
        assert!(automatic.fused_bias_gelu());
        automatic.bias_gelu_override = Some(false);
        assert!(!automatic.fused_bias_gelu());
        Ok(())
    }

    #[test]
    fn conv_execution_policy_preserves_explicit_reference_override() -> anyhow::Result<()> {
        let mut policy = super::ConvExecutionPolicy::from_values(None, None)?;
        assert_eq!(policy.resolve(), (false, false));
        policy.auto = true;
        assert_eq!(policy.resolve(), (true, true));
        let mut disabled = super::ConvExecutionPolicy::from_values(Some("0"), Some("1"))?;
        disabled.auto = true;
        assert_eq!(disabled.resolve(), (false, false));
        let mut layout_only = super::ConvExecutionPolicy::from_values(Some("1"), Some("0"))?;
        layout_only.auto = true;
        assert_eq!(layout_only.resolve(), (true, false));
        assert!(super::ConvExecutionPolicy::from_values(Some("invalid"), None).is_err());
        Ok(())
    }

    fn assert_cross_sample_conv_matches_reference(
        device: &candle_core::Device,
        dtype: candle_core::DType,
        tolerance: f32,
    ) -> anyhow::Result<()> {
        let mut cfg = test_config();
        cfg.n_window = 50;
        cfg.n_window_infer = 200;
        let vb =
            candle_nn::VarBuilder::from_backend(Box::new(SeededBackend), dtype, device.clone());
        let mut tower = AudioTower::load(&cfg, vb, false, None, &Progress::disabled())?;
        // Force multiple microbatches, including a boundary in the middle of a sample.
        tower.conv_micro_batch = 2;
        let lens = [0usize, 1, 99, 100, 101, 399, 663, 0];
        let frames = 663;
        let values = (0..lens.len() * cfg.num_mel_bins * frames)
            .map(|i| ((i * 19 % 113) as f32 - 56.0) / 61.0)
            .collect::<Vec<_>>();
        let features =
            candle_core::Tensor::from_vec(values, (lens.len(), cfg.num_mel_bins, frames), device)?;
        let features = super::cast_input_features_to_weight_dtype(&features, dtype)?;
        let window = cfg.n_window * 2;
        let got = tower.forward_conv_cross_sample(&features, &lens, window)?;
        assert_eq!(got.len(), lens.len());
        for (index, (&len, got)) in lens.iter().zip(&got).enumerate() {
            let input = features
                .narrow(0, index, 1)?
                .squeeze(0)?
                .narrow(1, 0, len)?;
            let reference = tower.forward_conv_sample(&input, window)?;
            assert_eq!(
                (got.aftercnn_total, got.max_t, got.chunks),
                (reference.aftercnn_total, reference.max_t, reference.chunks)
            );
            assert_tensor_close(&got.hidden_states, &reference.hidden_states, tolerance)?;
            if got.hidden_states.elem_count() > 0 {
                let values = got
                    .hidden_states
                    .to_dtype(candle_core::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(values.iter().all(|value| value.is_finite()));
            }
        }
        tower.conv_cross_sample = false;
        let reference = tower.forward_with_lens(&features, &lens)?;
        tower.conv_cross_sample = true;
        let routed = tower.forward_with_lens(&features, &lens)?;
        let encoder_tolerance = if dtype == candle_core::DType::BF16 {
            0.1
        } else {
            tolerance
        };
        assert_tensor_close(&routed, &reference, encoder_tolerance)?;
        let empty = tower.forward_conv_cross_sample(&features, &[0; 8], window)?;
        assert!(empty.iter().all(|sample| sample.aftercnn_total == 0
            && sample.max_t == 0
            && sample.chunks == 0
            && sample.hidden_states.dims() == [0, cfg.d_model]));
        Ok(())
    }

    #[test]
    fn test_cross_sample_conv_nonzero_cpu() -> anyhow::Result<()> {
        assert_cross_sample_conv_matches_reference(
            &candle_core::Device::Cpu,
            candle_core::DType::F32,
            1e-4,
        )
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires exclusive Metal GPU; run after stopping soak/benchmark server"]
    fn test_cross_sample_conv_nonzero_metal_f32() -> anyhow::Result<()> {
        assert_cross_sample_conv_matches_reference(
            &candle_core::Device::new_metal(0)?,
            candle_core::DType::F32,
            1e-3,
        )
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires exclusive Metal GPU; run after stopping soak/benchmark server"]
    fn test_cross_sample_conv_nonzero_metal_bf16() -> anyhow::Result<()> {
        assert_cross_sample_conv_matches_reference(
            &candle_core::Device::new_metal(0)?,
            candle_core::DType::BF16,
            0.025,
        )
    }

    fn assert_attention_modes(
        device: &candle_core::Device,
        dtype: candle_core::DType,
        tolerance: f32,
    ) -> anyhow::Result<()> {
        use super::AudioAttentionMode;
        let mut cfg = test_config();
        cfg.encoder_attention_heads = 2;
        let vb =
            candle_nn::VarBuilder::from_backend(Box::new(SeededBackend), dtype, device.clone());
        let mut attention = AudioAttention::load(&cfg, vb, false, None)?;
        attention.mode = AudioAttentionMode::Reference;
        // Unequal tails, repeated lengths, empty window, and multiple microbatches.
        let lengths = [3usize, 1, 3, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 2];
        let mut cu = vec![0];
        for len in lengths {
            cu.push(cu.last().copied().unwrap_or(0) + len);
        }
        let seq = cu.last().copied().unwrap_or(0);
        let values: Vec<f32> = (0..seq * cfg.d_model)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 29.0)
            .collect();
        let xs =
            candle_core::Tensor::from_vec(values, (seq, cfg.d_model), device)?.to_dtype(dtype)?;
        let expected = attention.forward(xs.clone(), &cu)?;
        for mode in [
            AudioAttentionMode::ProjectOnce,
            AudioAttentionMode::Grouped,
            AudioAttentionMode::GroupedNative,
        ] {
            attention.mode = mode;
            let got = attention.forward(xs.clone(), &cu)?;
            assert_tensor_close(&got, &expected, tolerance)?;
            let mut isolated = Vec::new();
            for window in cu.windows(2) {
                let len = window[1] - window[0];
                if len > 0 {
                    isolated.push(attention.forward(xs.narrow(0, window[0], len)?, &[0, len])?);
                }
            }
            let isolated = candle_core::Tensor::cat(&isolated.iter().collect::<Vec<_>>(), 0)?;
            assert_tensor_close(&got, &isolated, tolerance)?;
        }
        Ok(())
    }

    fn assert_tensor_close(
        got: &candle_core::Tensor,
        expected: &candle_core::Tensor,
        tolerance: f32,
    ) -> anyhow::Result<()> {
        assert_eq!(got.dims(), expected.dims());
        // Empty shapes already establish equality. Candle Metal cannot dispatch
        // a zero-element cast (linear_split divides by zero while encoding).
        if got.elem_count() == 0 {
            return Ok(());
        }
        let got = got
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            got.iter().chain(&expected).all(|value| value.is_finite()),
            "non-finite attention values"
        );
        let max_abs = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= tolerance,
            "attention mismatch: max_abs={max_abs} tolerance={tolerance}"
        );
        Ok(())
    }

    #[test]
    fn test_attention_optimized_modes_preserve_values_and_isolation_cpu() -> anyhow::Result<()> {
        assert_attention_modes(&candle_core::Device::Cpu, candle_core::DType::F32, 1e-5)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_attention_optimized_modes_preserve_values_and_isolation_metal_f32() -> anyhow::Result<()>
    {
        assert_attention_modes(
            &candle_core::Device::new_metal(0)?,
            candle_core::DType::F32,
            1e-3,
        )
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_attention_optimized_modes_preserve_values_and_isolation_metal_bf16()
    -> anyhow::Result<()> {
        assert_attention_modes(
            &candle_core::Device::new_metal(0)?,
            candle_core::DType::BF16,
            0.025,
        )
    }

    #[test]
    fn test_sinusoidal_positional_embedding_shape() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let pe = sinusoidal_positional_embedding(10, 8, &device)?;
        if pe.dims() != vec![10, 8] {
            anyhow::bail!("unexpected pe dims: {:?}", pe.dims());
        }
        Ok(())
    }

    #[test]
    fn test_feature_lens_from_mask_counts_nonzero() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let mask = candle_core::Tensor::from_vec(
            vec![1u32, 1, 0, 0, 1, 1, 1, 0],
            (2usize, 4usize),
            &device,
        )?;
        let lens = feature_lens_from_mask(&mask, 4)?;
        if lens != vec![2usize, 3usize] {
            anyhow::bail!("unexpected lens: {lens:?}");
        }
        Ok(())
    }

    #[test]
    fn test_audio_tower_forward_with_lens_matches_mask() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cfg = AudioEncoderConfig {
            num_mel_bins: 4,
            encoder_layers: 1,
            encoder_attention_heads: 1,
            encoder_ffn_dim: 8,
            d_model: 4,
            dropout: 0.0,
            attention_dropout: 0.0,
            activation_function: "gelu".to_string(),
            activation_dropout: 0.0,
            scale_embedding: false,
            initializer_range: 0.02,
            max_source_positions: 64,
            n_window: 5,
            output_dim: 6,
            n_window_infer: 20,
            conv_chunksize: 8,
            downsample_hidden_size: 2,
        };
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let tower = AudioTower::load(&cfg, vb, false, None, &Progress::disabled())?;

        let batch = 2usize;
        let mel = cfg.num_mel_bins;
        let frames = 10usize;
        let feats: Vec<f32> = (0..batch * mel * frames).map(|i| i as f32).collect();
        let input_features = candle_core::Tensor::from_vec(feats, (batch, mel, frames), &device)?;

        let len1 = 6usize;
        let mut mask: Vec<u32> = Vec::with_capacity(batch * frames);
        mask.extend(std::iter::repeat_n(1u32, frames));
        mask.extend(std::iter::repeat_n(1u32, len1));
        mask.extend(std::iter::repeat_n(0u32, frames - len1));
        let mask = candle_core::Tensor::from_vec(mask, (batch, frames), &device)?;

        let out_mask = tower.forward(&input_features, Some(&mask))?;
        let out_lens = tower.forward_with_lens(&input_features, &[frames, len1])?;

        if out_mask.dims() != out_lens.dims() {
            anyhow::bail!(
                "output dims mismatch: mask={:?} lens={:?}",
                out_mask.dims(),
                out_lens.dims()
            );
        }

        let got = out_mask.to_vec2::<f32>()?;
        let exp = out_lens.to_vec2::<f32>()?;
        let mut max_abs = 0f32;
        for (row_g, row_e) in got.iter().zip(exp.iter()) {
            for (&a, &b) in row_g.iter().zip(row_e.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        if max_abs > 1e-6 {
            anyhow::bail!("forward mismatch: max_abs={max_abs}");
        }
        Ok(())
    }

    #[test]
    fn test_audio_tower_batched_forward_matches_per_sample_forward() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cfg = AudioEncoderConfig {
            num_mel_bins: 4,
            encoder_layers: 1,
            encoder_attention_heads: 1,
            encoder_ffn_dim: 8,
            d_model: 4,
            dropout: 0.0,
            attention_dropout: 0.0,
            activation_function: "gelu".to_string(),
            activation_dropout: 0.0,
            scale_embedding: false,
            initializer_range: 0.02,
            max_source_positions: 64,
            n_window: 5,
            output_dim: 6,
            n_window_infer: 20,
            conv_chunksize: 8,
            downsample_hidden_size: 2,
        };
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let tower = AudioTower::load(&cfg, vb, false, None, &Progress::disabled())?;

        let batch = 3usize;
        let mel = cfg.num_mel_bins;
        let frames = 10usize;
        let lens = [10usize, 7usize, 4usize];
        let feats: Vec<f32> = (0..batch * mel * frames)
            .map(|i| (i as f32) / 100.0)
            .collect();
        let input_features = candle_core::Tensor::from_vec(feats, (batch, mel, frames), &device)?;

        let batched = tower.forward_with_lens(&input_features, &lens)?;
        let weight_dtype = tower.conv2d1.weight().dtype();
        let mut per_sample = Vec::with_capacity(batch);
        for (idx, &len) in lens.iter().enumerate() {
            let input = input_features
                .narrow(0, idx, 1)?
                .squeeze(0)?
                .narrow(1, 0, len)?;
            // `forward_one` 是测试辅助路径，不负责 dtype 转换，这里手动对齐权重 dtype。
            let input = super::cast_input_features_to_weight_dtype(&input, weight_dtype)?;
            per_sample.push(tower.forward_one(&input)?);
        }
        let expected = candle_core::Tensor::cat(&per_sample.iter().collect::<Vec<_>>(), 0)?;

        if batched.dims() != expected.dims() {
            anyhow::bail!(
                "batched dims mismatch: batched={:?} expected={:?}",
                batched.dims(),
                expected.dims()
            );
        }

        let got = batched.to_vec2::<f32>()?;
        let exp = expected.to_vec2::<f32>()?;
        let mut max_abs = 0f32;
        for (row_g, row_e) in got.iter().zip(exp.iter()) {
            for (&a, &b) in row_g.iter().zip(row_e.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        if max_abs > 1e-5 {
            anyhow::bail!("batched forward mismatch: max_abs={max_abs}");
        }
        Ok(())
    }

    #[test]
    fn test_nonzero_weights_batched_forward_matches_per_sample() -> anyhow::Result<()> {
        // 零权重测试无法暴露 padding / 窗口边界错误；这里用非零权重复验多段拼接。
        assert_batched_matches_per_sample(
            &candle_core::Device::Cpu,
            candle_core::DType::F32,
            false,
            1e-4,
        )
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_nonzero_weights_batched_forward_matches_per_sample_f32() -> anyhow::Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        assert_batched_matches_per_sample(&device, candle_core::DType::F32, false, 1e-3)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_nonzero_weights_batched_forward_matches_per_sample_bf16() -> anyhow::Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        // BF16 精度较低，放宽到 1e-1；重点验证多段 padding / 边界是否一致。
        assert_batched_matches_per_sample(&device, candle_core::DType::BF16, false, 1e-1)
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_input_features_can_be_cast_to_bf16() -> anyhow::Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        let input =
            candle_core::Tensor::from_vec(vec![1f32, 2.0], (1usize, 1usize, 2usize), &device)?;

        let got = super::cast_input_features_to_weight_dtype(&input, candle_core::DType::BF16)?;

        assert_eq!(got.dtype(), candle_core::DType::BF16);
        assert!(got.device().is_metal());
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_bf16_transpose_can_be_materialized_contiguous() -> anyhow::Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        let input =
            candle_core::Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2usize, 2usize), &device)?;
        let input = super::cast_input_features_to_weight_dtype(&input, candle_core::DType::BF16)?;

        let got = input.transpose(0, 1)?.contiguous()?;

        assert_eq!(got.dtype(), candle_core::DType::BF16);
        assert!(got.device().is_metal());
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "exclusive Metal chunk preparation diagnostic"]
    fn diagnose_native_bf16_chunk_preparation() -> anyhow::Result<()> {
        use candle_core::{DType, Device, Tensor};
        let device = Device::new_metal(0)?;
        for frames in [5usize, 105, 805, 3005] {
            let mel = 128;
            let input = Tensor::from_vec(
                (0..mel * (frames + 7))
                    .map(|i| ((i % 193) as f32 - 96.) / 11.)
                    .collect::<Vec<_>>(),
                (mel, frames + 7),
                &device,
            )?
            .to_dtype(DType::BF16)?
            .narrow(1, 7, frames)?
            .transpose(0, 1)?;
            let mut reference = Vec::new();
            let mut native = Vec::new();
            for start in (0..frames).step_by(100) {
                let len = (frames - start).min(100);
                let chunk = input.narrow(0, start, len)?;
                reference.push(super::pad_audio_chunk(&chunk, 0, 0, 100 - len, false)?);
                native.push(super::pad_audio_chunk(&chunk, 0, 0, 100 - len, true)?);
            }
            let ref_stack = super::stack_audio_chunks(&reference, false)?;
            let native_stack = super::stack_audio_chunks(&native, true)?;
            assert_tensor_close(&native_stack, &ref_stack, 0.)?;
            for phase in ["compat", "native"] {
                let mut times = Vec::new();
                for iteration in 0..24 {
                    device.synchronize()?;
                    let start_time = std::time::Instant::now();
                    let chunks = (0..frames)
                        .step_by(100)
                        .map(|start| {
                            let len = (frames - start).min(100);
                            let chunk = input.narrow(0, start, len)?;
                            if phase == "compat" {
                                super::pad_audio_chunk(&chunk, 0, 0, 100 - len, false)
                            } else {
                                chunk.pad_with_zeros(0, 0, 100 - len)
                            }
                        })
                        .collect::<candle_core::Result<Vec<_>>>()?;
                    let output = if phase == "compat" {
                        super::stack_audio_chunks(&chunks, false)?
                    } else {
                        Tensor::stack(&chunks.iter().collect::<Vec<_>>(), 0)?
                    };
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if iteration >= 4 {
                        times.push(start_time.elapsed().as_secs_f64() * 1e6);
                    }
                }
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "chunk_prep frames={frames} phase={phase} median_us={:.3} min_us={:.3} max_us={:.3}",
                    (times[9] + times[10]) / 2.,
                    times[0],
                    times[19]
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_bf16_audio_chunk_padding_avoids_strided_copy_kernel() -> anyhow::Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        let mel = 128usize;
        let frames = 1505usize;
        let input = candle_core::Tensor::from_vec(
            (0..mel * frames).map(|i| i as f32).collect::<Vec<_>>(),
            (1usize, mel, frames),
            &device,
        )?;
        let input = super::cast_input_features_to_weight_dtype(&input, candle_core::DType::BF16)?;
        let input = input.narrow(0, 0, 1)?.squeeze(0)?;

        let input_t = input.transpose(0, 1)?;
        let chunk = input_t.narrow(0, 1500, 5)?;
        let padded = super::pad_audio_chunk(&chunk, 0, 0, 95, false)?;

        assert_eq!(padded.dtype(), candle_core::DType::BF16);
        assert!(padded.device().is_metal());
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_bf16_audio_chunks_can_be_stacked() -> anyhow::Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        let mel = 128usize;
        let frames = 1505usize;
        let input = candle_core::Tensor::from_vec(
            (0..mel * frames).map(|i| i as f32).collect::<Vec<_>>(),
            (1usize, mel, frames),
            &device,
        )?;
        let input = super::cast_input_features_to_weight_dtype(&input, candle_core::DType::BF16)?;
        let input = input.narrow(0, 0, 1)?.squeeze(0)?;

        let input_t = input.transpose(0, 1)?;
        let mut chunks = Vec::new();
        for offset in (0..frames).step_by(100) {
            let len = (frames - offset).min(100);
            let chunk = input_t.narrow(0, offset, len)?;
            chunks.push(super::pad_audio_chunk(&chunk, 0, 0, 100 - len, false)?);
        }
        let stacked = super::stack_audio_chunks(&chunks, false)?;

        assert_eq!(stacked.dtype(), candle_core::DType::BF16);
        assert!(stacked.device().is_metal());
        Ok(())
    }

    #[test]
    fn test_cu_seqlens_from_aftercnn_len_splits_into_windows() -> anyhow::Result<()> {
        // Mirrors the official computation:
        // window_aftercnn = max_chunk_len_aftercnn * (n_window_infer / (n_window*2))
        let aftercnn_len = 100usize;
        let max_chunk_len_aftercnn = 26usize;
        let window = 200usize; // n_window*2
        let n_window_infer = 400usize;
        let cu = cu_seqlens_from_aftercnn_len(
            aftercnn_len,
            max_chunk_len_aftercnn,
            window,
            n_window_infer,
        )?;
        if cu != vec![0usize, 52, 100] {
            anyhow::bail!("unexpected cu_seqlens: {cu:?}");
        }
        Ok(())
    }

    #[test]
    fn test_audio_attention_respects_cu_seqlens_when_eager() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let embed_dim = 4usize;
        let seq_len = 5usize;

        let mut eye = vec![0f32; embed_dim * embed_dim];
        for i in 0..embed_dim {
            eye[i * embed_dim + i] = 1.0;
        }
        let w = candle_core::Tensor::from_vec(eye, (embed_dim, embed_dim), &device)?;
        let q_proj = IsqLinear::new(Linear::new(w.clone(), None), None, &device)?;
        let k_proj = IsqLinear::new(Linear::new(w.clone(), None), None, &device)?;
        let v_proj = IsqLinear::new(Linear::new(w.clone(), None), None, &device)?;
        let out_proj = IsqLinear::new(Linear::new(w, None), None, &device)?;

        let attn = AudioAttention {
            mode: super::AudioAttentionMode::Reference,
            use_flash_attn: false,
            num_heads: 1,
            head_dim: embed_dim,
            scaling: (embed_dim as f64).powf(-0.5),
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        };

        let xs: Vec<f32> = (0..seq_len * embed_dim).map(|i| i as f32).collect();
        let hidden_states = candle_core::Tensor::from_vec(xs, (seq_len, embed_dim), &device)?;

        let out_blocked = attn.forward(hidden_states.clone(), &[0usize, 2, 5])?;
        let out_full = attn.forward(hidden_states, &[0usize, 5])?;

        let got = out_blocked.to_vec2::<f32>()?;
        let exp = out_full.to_vec2::<f32>()?;
        let mut max_abs = 0f32;
        for (row_g, row_e) in got.iter().zip(exp.iter()) {
            for (&a, &b) in row_g.iter().zip(row_e.iter()) {
                max_abs = max_abs.max((a - b).abs());
            }
        }
        if max_abs <= 1e-4 {
            anyhow::bail!("segmented attention unexpectedly matched full attention");
        }
        Ok(())
    }
}
