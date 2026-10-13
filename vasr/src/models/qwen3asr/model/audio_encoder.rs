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

mod attention;
pub(crate) mod policy;
use crate::model::operator_policy::OperatorEnvironment;
use attention::AudioAttention;
use policy::{AudioAttentionMode, AudioOperatorPolicy, ConvExecutionPolicy};

const LAYER_NORM_EPS: f64 = 1e-5;

fn conv_forward(conv: &Conv2d, x: &Tensor, flatten: bool) -> Result<Tensor> {
    #[cfg(feature = "metal-paged-attn")]
    if x.device().is_metal() {
        let cfg = conv.config();
        if flatten && cfg.groups == 1 {
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
    let _ = flatten;
    conv.forward(x)
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
    gelu(&conv_forward(conv, x, policy.flatten)?)
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
    #[cfg(all(test, feature = "metal-paged-attn"))]
    fn load(
        cfg: &AudioEncoderConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
        isq: Option<&str>,
    ) -> Result<Self> {
        let env = OperatorEnvironment::capture();
        let policy = AudioOperatorPolicy::resolve(
            &env,
            cfg,
            vb.device().is_metal(),
            vb.device().is_cuda(),
            false,
        )?;
        Self::load_with_policy(cfg, vb, use_flash_attn, isq, &policy)
    }

    fn load_with_policy(
        cfg: &AudioEncoderConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
        isq: Option<&str>,
        policy: &AudioOperatorPolicy,
    ) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let self_attn =
            AudioAttention::load_with_policy(cfg, vb.pp("self_attn"), use_flash_attn, isq, policy)?;
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
            #[cfg(feature = "metal-paged-attn")]
            fused_fc1_gelu: policy.fused_fc1_gelu,
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
    seq_bucket: Option<usize>,
    debug_layers: bool,
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
        let env = OperatorEnvironment::capture();
        let policy = AudioOperatorPolicy::resolve(
            &env,
            cfg,
            vb.device().is_metal(),
            vb.device().is_cuda(),
            false,
        )?;
        Self::load_with_policy(cfg, vb, use_flash_attn, isq, progress, &policy)
    }

    pub(crate) fn load_with_policy(
        cfg: &AudioEncoderConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
        isq: Option<&str>,
        progress: &Progress,
        policy: &AudioOperatorPolicy,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let conv_policy = policy.conv.clone();
        let conv_cross_sample = policy.cross_sample;
        let conv_micro_batch = policy.micro_batch;
        let (nhwc, im2col32) = conv_policy.resolve();
        let bias_gelu = conv_policy.fused_bias_gelu();
        let native_chunks = conv_policy.native_chunks();
        if conv_policy.auto || nhwc {
            tracing::debug!(target: "vasr::execution", "Metal CNN policy | nhwc={nhwc} im2col32={im2col32} bias_gelu={bias_gelu} native_chunks={native_chunks} explicit_overrides=preserved");
        }
        #[cfg(feature = "metal-paged-attn")]
        tracing::debug!(target: "vasr::execution", "Metal encoder policy | conv_tile={:?} attention={:?} fc1_bias_gelu={} overrides=preserved", conv_policy.gemm_tile, policy.attention, policy.fused_fc1_gelu);

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
            layers.push(AudioEncoderLayer::load_with_policy(
                cfg,
                vb.pp("layers").pp(i),
                use_flash_attn,
                isq,
                policy,
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
            seq_bucket: policy.seq_bucket,
            debug_layers: policy.debug_layers,
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
        let padded_idx_len = match self.seq_bucket {
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
            if self.debug_layers {
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
        let limit = self
            .conv_micro_batch
            .clamp(1, if features.device().is_cuda() { 256 } else { 64 });
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
        let padded_idx_len = match self.seq_bucket {
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
            if self.debug_layers {
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
mod tests;
