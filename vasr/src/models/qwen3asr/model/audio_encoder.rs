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

fn pad_audio_chunk(chunk: &Tensor, dim: usize, left: usize, right: usize) -> Result<Tensor> {
    if chunk.device().is_metal() && chunk.dtype() == DType::BF16 && (left != 0 || right != 0) {
        // Convert to F32 for padding (Metal has full F32 support), then back to BF16.
        return chunk
            .to_dtype(DType::F32)?
            .contiguous()?
            .pad_with_zeros(dim, left, right)?
            .to_dtype(chunk.dtype());
    }

    chunk.pad_with_zeros(dim, left, right)
}

fn stack_audio_chunks(chunks: &[Tensor]) -> Result<Tensor> {
    let first = chunks
        .first()
        .ok_or_else(|| candle_core::Error::Msg("missing audio chunks".to_string()))?;

    if first.device().is_metal() && first.dtype() == DType::BF16 {
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

#[derive(Debug, Clone)]
struct AudioAttention {
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
            self.project_qkv(&hidden_states)?
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

                let attn = {
                    let k_t = maybe_contiguous_for_metal(k_seg.transpose(2, 3)?)?; // (1, h, d, s)
                    let attn_weights = q_seg.matmul(&k_t)?.affine(self.scaling, 0.0)?;
                    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                    attn_weights.matmul(&v_seg)? // (1, h, s, d)
                };
                let attn = attn.squeeze(0)?.transpose(0, 1)?; // (s, h, d)
                let attn = attn.reshape((len, embed_dim))?.detach();
                outs.push(self.out_proj.forward(&attn)?.detach());
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
}

#[derive(Debug, Clone)]
struct AudioEncoderLayer {
    self_attn: AudioAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: IsqLinear,
    fc2: IsqLinear,
    final_layer_norm: LayerNorm,
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
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
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

        let x = {
            let residual = x;
            let activated = {
                let normed = self.final_layer_norm.forward(&residual)?.detach();
                #[cfg(feature = "cuda")]
                {
                    if let Some(activated) = isq_linear::try_fused_q8_gelu_erf(&self.fc1, &normed)?
                    {
                        activated.detach()
                    } else {
                        gelu(&self.fc1.forward(&normed)?.detach())?.detach()
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    gelu(&self.fc1.forward(&normed)?.detach())?.detach()
                }
            };
            let fc2 = self.fc2.forward(&activated)?.detach();
            (&residual + &fc2)?.detach()
        };

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
}

struct AudioConvSample {
    hidden_states: Tensor,
    aftercnn_total: usize,
    max_t: usize,
    chunks: usize,
}

impl AudioTower {
    pub fn load(
        cfg: &AudioEncoderConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
        isq: Option<&str>,
        progress: &Progress,
    ) -> Result<Self> {
        let device = vb.device().clone();

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

        for (sample_idx, &sample_len) in feature_lens.iter().enumerate() {
            if sample_len == 0 {
                sample_aftercnn_totals.push(0);
                sample_max_ts.push(0);
                continue;
            }

            let input_feature = input_features.narrow(0, sample_idx, 1)?.squeeze(0)?;
            let input_feature = input_feature.narrow(1, 0, sample_len)?;
            let sample = self.forward_conv_sample(&input_feature, window)?;
            total_chunks = total_chunks.saturating_add(sample.chunks);
            sample_aftercnn_totals.push(sample.aftercnn_total);
            sample_max_ts.push(sample.max_t);
            sample_hiddens.push(sample.hidden_states);
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
            padded_chunks.push(pad_audio_chunk(&chunk, 0, 0, pad)?);
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
            stack_audio_chunks(&padded_chunks)?
                .transpose(1, 2)?
                .unsqueeze(1)?
        };

        let conv_outs = {
            let padded_feature = padded_feature;
            let mut conv_outs: Vec<Tensor> = Vec::new();
            let mut start = 0usize;
            while start < chunk_num {
                let len = (chunk_num - start).min(self.config.conv_chunksize.max(1));
                let x = padded_feature.narrow(0, start, len)?;
                let x = gelu(&self.conv2d1.forward(&x)?)?;
                let x = gelu(&self.conv2d2.forward(&x)?)?;
                let x = gelu(&self.conv2d3.forward(&x)?)?;
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

        let (b, c, f, t) = x.dims4()?;
        x = x.transpose(1, 3)?.transpose(2, 3)?.contiguous()?;
        x = x.reshape((b, t, c * f))?;
        x = self.conv_out.forward(&x)?;

        let pe = self.positional_embedding.narrow(0, 0, t)?.unsqueeze(0)?;
        let pe = pe.to_dtype(x.dtype())?;
        x = x.broadcast_add(&pe)?;

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
        let padded = super::pad_audio_chunk(&chunk, 0, 0, 95)?;

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
            chunks.push(super::pad_audio_chunk(&chunk, 0, 0, 100 - len)?);
        }
        let stacked = super::stack_audio_chunks(&chunks)?;

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
