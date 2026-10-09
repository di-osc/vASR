//! Thinker text model (decoder-only transformer with mRoPE).
//!
//! This is a faithful port of the "thinker.model" module in the official
//! Qwen3-ASR implementation:
//! `Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`.

use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder, embedding, rms_norm};

use crate::config::TextConfig;
use crate::layers::MultimodalRope;
#[cfg(feature = "paged-attn")]
use crate::layers::mrope::PreparedMultimodalRope;
use crate::model::attention;
use crate::model::isq_linear::{self, IsqLinear};
use crate::model::kv_cache::KVCache;
use crate::utils::progress::{Progress, ProgressEvent};

#[cfg(feature = "paged-attn")]
use crate::model::paged_kv_cache::{PagedInputMetadata, PagedKvCache};
#[cfg(all(
    feature = "paged-attn",
    any(feature = "cuda-paged-attn", feature = "metal-paged-attn")
))]
use vasr_paged_attn::{gather_kv_cache, paged_attention, reshape_and_cache};

#[cfg(feature = "paged-attn")]
#[derive(Clone)]
struct PagedAttentionHandle {
    k_scale: Tensor,
    v_scale: Tensor,
}

#[cfg(feature = "paged-attn")]
impl std::fmt::Debug for PagedAttentionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedAttentionHandle").finish()
    }
}

#[cfg(feature = "flash-attn")]
use candle_core::{DType, Device};

#[cfg(feature = "flash-attn")]
use vasr_flash_attn::{flash_attn, flash_attn_varlen};

const DEFAULT_MROPE_SECTION: [usize; 3] = [24, 20, 20];

fn load_multimodal_rope(cfg: &TextConfig, device: &candle_core::Device) -> Result<MultimodalRope> {
    let (mrope_section, interleaved) = match cfg.rope_scaling.as_ref() {
        None => (DEFAULT_MROPE_SECTION.to_vec(), false),
        Some(scaling) => {
            match scaling
                .rope_type
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                None | Some("default") => {}
                Some(other) => candle_core::bail!(
                    "unsupported Qwen3-ASR mRoPE rope_type `{other}`; only `default` is supported"
                ),
            }
            let section = if scaling.mrope_section.is_empty() {
                DEFAULT_MROPE_SECTION.to_vec()
            } else {
                scaling.mrope_section.clone()
            };
            (section, scaling.mrope_interleaved || scaling.interleaved)
        }
    };

    MultimodalRope::new(
        cfg.head_dim,
        cfg.max_position_embeddings,
        cfg.rope_theta,
        mrope_section,
        interleaved,
        device,
    )
}

fn cache_input_is_packed(tensor: &Tensor) -> Result<bool> {
    if tensor.dims().len() != 3 {
        return Ok(false);
    }
    let (_, heads, head_size) = tensor.dims3()?;
    let stride = tensor.stride();
    Ok(stride[2] == 1 && stride[1] == head_size && stride[0] == heads * head_size)
}

fn maybe_contiguous_for_accelerator(x: Tensor) -> Result<Tensor> {
    if x.device().is_metal() || x.device().is_cuda() {
        if cache_input_is_packed(&x)? {
            Ok(x)
        } else {
            x.contiguous()
        }
    } else {
        Ok(x)
    }
}

fn initial_hidden_states_for_paged_decode(inputs_embeds: &Tensor) -> Result<Tensor> {
    let seq_len = inputs_embeds.dim(1)?;
    if seq_len == 1 {
        if inputs_embeds.device().is_metal() {
            return inputs_embeds.affine(1.0, 0.0);
        }
        #[cfg(feature = "cuda-graph")]
        if inputs_embeds.device().is_cuda() {
            return inputs_embeds.affine(1.0, 0.0);
        }
    }
    Ok(inputs_embeds.clone())
}

#[cfg(feature = "paged-attn")]
fn unpack_gathered_kv_for_attention(
    packed: &Tensor,
    kv_lens: &[usize],
    num_kv_heads: usize,
    _num_kv_groups: usize,
    head_size: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if kv_lens.is_empty() {
        candle_core::bail!("unpack_gathered_kv_for_attention requires at least one sequence");
    }

    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
    let batch = kv_lens.len();
    if max_kv == 0 {
        return Tensor::zeros(
            (batch, num_kv_heads, 0usize, head_size),
            packed.dtype(),
            device,
        );
    }

    if kv_lens.iter().all(|&len| len == max_kv) {
        let total = batch
            .checked_mul(max_kv)
            .ok_or_else(|| candle_core::Error::Msg("gathered kv token count overflow".into()))?;
        return packed
            .narrow(0, 0, total)?
            .reshape((batch, max_kv, num_kv_heads, head_size))?
            .transpose(1, 2);
    }

    let mut start = 0usize;
    let mut unpacked: Vec<Tensor> = Vec::with_capacity(batch);

    for &kv_len in kv_lens {
        let mut seq = packed
            .narrow(0, start, kv_len)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        if kv_len < max_kv {
            let pad = Tensor::zeros(
                (1usize, num_kv_heads, max_kv - kv_len, head_size),
                packed.dtype(),
                device,
            )?;
            seq = Tensor::cat(&[&seq, &pad], 2)?;
        }
        unpacked.push(seq);
        start = start.saturating_add(kv_len);
    }

    Tensor::cat(&unpacked, 0)
}

#[cfg(feature = "paged-attn")]
fn paged_prefill_attention_mask(
    input_metadata: &PagedInputMetadata,
    kv_lens: &[usize],
    batch: usize,
    seq_len: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> Result<attention::AttentionMask> {
    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
    if input_metadata.prefill_causal_only {
        return Ok(attention::AttentionMask::None);
    }
    let mask = if let Some(mask) = input_metadata.prefill_attention_mask.as_ref() {
        mask.clone()
    } else {
        attention::make_causal_mask(
            input_metadata.token_attention_mask.as_ref(),
            batch,
            seq_len,
            dtype,
            device,
        )?
    };
    Ok(attention::AttentionMask::Custom(attention::adjust_kv_mask(
        &mask, max_kv,
    )?))
}

#[cfg(feature = "paged-attn")]
fn metal_hybrid_paged_prefill_enabled() -> bool {
    crate::model::generation::metal_paged_attention_fallback_active()
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "cuda-paged-attn", feature = "metal-paged-attn"))]
fn run_paged_prefill_attention_fallback(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    input_metadata: &PagedInputMetadata,
    kv_lens: &[usize],
    paged_attn: &PagedAttentionHandle,
    batch: usize,
    seq_len: usize,
    hidden_states: &Tensor,
    scaling: f32,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let cu_kv = input_metadata
        .cu_seqlens_kv
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("paged metadata missing cu_seqlens_kv".into()))?;
    let (k_gathered, v_gathered) = gather_kv_cache(
        key_cache,
        value_cache,
        Some(&paged_attn.k_scale),
        Some(&paged_attn.v_scale),
        &input_metadata.block_tables,
        cu_kv,
        hidden_states.dtype(),
    )?;
    let mask = paged_prefill_attention_mask(
        input_metadata,
        kv_lens,
        batch,
        seq_len,
        q.dtype(),
        q.device(),
    )?;
    let flash_params = input_metadata.prefill_flash_params();
    let avoid_packed_varlen = q.device().is_metal() && metal_hybrid_paged_prefill_enabled();
    if !avoid_packed_varlen && attention::supports_packed_varlen_sdpa(q) {
        let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
        let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;
        let packed_mask = if q.device().is_cuda() {
            // CUDA packed varlen relies on cu_seqlens + causal flash-attn. A custom
            // padded mask disables flash-attn and cannot be applied to batch=1 packed K/V.
            attention::AttentionMask::None
        } else {
            mask
        };
        attention::run_attention(
            q,
            &k_4d,
            &v_4d,
            &packed_mask,
            flash_params,
            scaling,
            true,
            num_key_value_groups,
        )
    } else {
        let k = unpack_gathered_kv_for_attention(
            &k_gathered,
            kv_lens,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            hidden_states.device(),
        )?;
        let v = unpack_gathered_kv_for_attention(
            &v_gathered,
            kv_lens,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            hidden_states.device(),
        )?;
        attention::run_attention(
            q,
            &k,
            &v,
            &mask,
            flash_params,
            scaling,
            true,
            num_key_value_groups,
        )
    }
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
#[cfg(not(any(feature = "cuda-paged-attn", feature = "metal-paged-attn")))]
fn run_paged_prefill_attention_fallback(
    _q: &Tensor,
    _key_cache: &Tensor,
    _value_cache: &Tensor,
    _input_metadata: &PagedInputMetadata,
    _kv_lens: &[usize],
    _paged_attn: &PagedAttentionHandle,
    _batch: usize,
    _seq_len: usize,
    _hidden_states: &Tensor,
    _scaling: f32,
    _num_key_value_heads: usize,
    _num_key_value_groups: usize,
    _head_dim: usize,
) -> Result<Tensor> {
    candle_core::bail!(
        "paged-attn backend not enabled: build with cuda-paged-attn or metal-paged-attn"
    )
}

#[cfg(feature = "flash-attn")]
fn seqlens_from_attention_mask(mask: &Tensor, seq_len: usize) -> Result<Vec<usize>> {
    let (batch, t2) = mask.dims2()?;
    if t2 != seq_len {
        candle_core::bail!("attention_mask seq_len mismatch: expected={seq_len}, got={t2}");
    }

    let mask_u8 = mask.ne(0u32)?;
    let lens_f32 = mask_u8.to_dtype(DType::F32)?.sum(1)?;
    let lens = lens_f32.to_vec1::<f32>()?;
    if lens.len() != batch {
        candle_core::bail!(
            "internal error: attention_mask lens mismatch: expected={batch}, got={}",
            lens.len()
        );
    }

    let mut out: Vec<usize> = Vec::with_capacity(batch);
    for (i, v) in lens.into_iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            candle_core::bail!("invalid attention_mask length at {i}: {v}");
        }
        if v > seq_len as f32 + 0.01 {
            candle_core::bail!(
                "attention_mask length at {i} exceeds seq_len: len={v} seq_len={seq_len}"
            );
        }
        let len = v.round() as usize;
        if (len as f32 - v).abs() > 0.01 {
            candle_core::bail!("attention_mask length at {i} is not integral: {v}");
        }
        out.push(len);
    }
    Ok(out)
}

#[cfg(feature = "flash-attn")]
fn cu_seqlens_u32(lengths: &[usize], device: &Device) -> Result<(Tensor, usize, u32)> {
    let mut cu: Vec<u32> = Vec::with_capacity(lengths.len().saturating_add(1));
    cu.push(0);

    let mut total: u32 = 0;
    let mut max_len: usize = 0;
    for (i, &len) in lengths.iter().enumerate() {
        let len_u32 = u32::try_from(len).map_err(|_| {
            candle_core::Error::Msg(format!(
                "sequence length overflows u32 at index {i}: len={len}"
            ))
        })?;
        total = total.checked_add(len_u32).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "cumulative sequence length overflows u32 at index {i}: total={total} len={len}"
            ))
        })?;
        cu.push(total);
        max_len = max_len.max(len);
    }

    let cu_t = Tensor::from_vec(cu, (lengths.len().saturating_add(1),), device)?;
    Ok((cu_t, max_len, total))
}

#[cfg(feature = "flash-attn")]
fn mask_nonzero_indices_u32(mask: &Tensor, seq_len: usize) -> Result<Vec<u32>> {
    let (batch, t2) = mask.dims2()?;
    if t2 != seq_len {
        candle_core::bail!("attention_mask seq_len mismatch: expected={seq_len}, got={t2}");
    }

    let rows = mask.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
    if rows.len() != batch {
        candle_core::bail!(
            "internal error: attention_mask row mismatch: expected={batch}, got={}",
            rows.len()
        );
    }

    let mut idxs = Vec::new();
    for (b, row) in rows.iter().enumerate() {
        if row.len() != seq_len {
            candle_core::bail!(
                "attention_mask row len mismatch at batch index {b}: expected={seq_len}, got={}",
                row.len()
            );
        }
        let base = b
            .checked_mul(seq_len)
            .ok_or_else(|| candle_core::Error::Msg("index overflow".to_string()))?;
        for (j, &value) in row.iter().enumerate() {
            if value == 0 {
                continue;
            }
            let pos = base
                .checked_add(j)
                .ok_or_else(|| candle_core::Error::Msg("index overflow".to_string()))?;
            idxs.push(
                u32::try_from(pos).map_err(|_| {
                    candle_core::Error::Msg(format!("index overflows u32: pos={pos}"))
                })?,
            );
        }
    }

    if idxs.is_empty() && batch > 0 {
        candle_core::bail!("attention_mask contains no valid tokens");
    }

    Ok(idxs)
}

fn metal_dense_projection_fusion_enabled() -> Result<bool> {
    match std::env::var("VASR_METAL_DENSE_PROJECTION_FUSION") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) => match value.as_str() {
            "1" | "true" | "on" => Ok(true),
            "0" | "false" | "off" => Ok(false),
            _ => candle_core::bail!(
                "invalid VASR_METAL_DENSE_PROJECTION_FUSION={value:?}; expected 0 or 1"
            ),
        },
        Err(error) => candle_core::bail!("invalid VASR_METAL_DENSE_PROJECTION_FUSION: {error}"),
    }
}

fn parse_prefill_gate_up_policy(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("0" | "false" | "off") => Ok(false),
        Some("1" | "true" | "on") => Ok(true),
        _ => candle_core::bail!("VASR_METAL_PREFILL_GATE_UP must be 0 or 1"),
    }
}

fn parse_prepared_mrope_policy(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("0" | "false" | "off") => Ok(false),
        Some("1" | "true" | "on") => Ok(true),
        _ => candle_core::bail!("VASR_METAL_PREPARED_MROPE must be 0 or 1"),
    }
}

fn parse_packed_rope_policy(value: Option<&str>) -> Result<bool> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("0" | "false" | "off") => Ok(false),
        Some("1" | "true" | "on") => Ok(true),
        _ => candle_core::bail!("VASR_METAL_PREFILL_ROPE_KERNEL must be 0 or 1"),
    }
}

fn resolve_rope_defaults(
    prepared: Option<bool>,
    packed: Option<bool>,
    validated: bool,
) -> (bool, bool) {
    let packed = packed.unwrap_or(validated);
    let prepared = prepared.unwrap_or(packed);
    (prepared, packed && prepared)
}

fn read_rope_override(name: &str, parse: fn(Option<&str>) -> Result<bool>) -> Result<Option<bool>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(parse(Some(&value))?)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => candle_core::bail!("invalid {name}; expected 0 or 1"),
    }
}

/// One dense GEMM for independent projections sharing the same input.
/// Quantized weights and other devices retain their existing paths.
#[derive(Debug, Clone)]
struct MetalDenseProjection {
    linear: candle_nn::Linear,
    output_dims: Vec<usize>,
}

impl MetalDenseProjection {
    fn new(projections: &[&IsqLinear], enabled: bool) -> Result<Option<Self>> {
        if !enabled || projections.is_empty() {
            return Ok(None);
        }
        let mut linears = Vec::with_capacity(projections.len());
        for projection in projections {
            let IsqLinear::Linear(linear) = projection else {
                return Ok(None);
            };
            linears.push(linear);
        }
        let first = linears[0].weight();
        if !first.device().is_metal() {
            return Ok(None);
        }
        let input_dim = first.dim(1)?;
        let dtype = first.dtype();
        let has_bias = linears[0].bias().is_some();
        let mut weights = Vec::with_capacity(linears.len());
        let mut biases = Vec::with_capacity(linears.len());
        let mut output_dims = Vec::with_capacity(linears.len());
        for linear in linears {
            let weight = linear.weight();
            if weight.dim(1)? != input_dim
                || weight.dtype() != dtype
                || !weight.device().same_device(first.device())
                || linear.bias().is_some() != has_bias
            {
                return Ok(None);
            }
            output_dims.push(weight.dim(0)?);
            weights.push(weight);
            if let Some(bias) = linear.bias() {
                if bias.dtype() != dtype || !bias.device().same_device(first.device()) {
                    return Ok(None);
                }
                biases.push(bias);
            }
        }
        // Metal's BF16 copy kernels do not support every strided cat layout.
        // Concatenate via F32 on-device only during load; inference stays BF16.
        let concat = |tensors: &[&Tensor]| -> Result<Tensor> {
            if dtype == candle_core::DType::BF16 {
                let tensors = tensors
                    .iter()
                    .map(|tensor| tensor.to_dtype(candle_core::DType::F32)?.contiguous())
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&tensors.iter().collect::<Vec<_>>(), 0)?.to_dtype(dtype)
            } else {
                Tensor::cat(tensors, 0)
            }
        };
        let weight = concat(&weights)?;
        let bias = if has_bias {
            Some(concat(&biases)?)
        } else {
            None
        };
        Ok(Some(Self {
            linear: candle_nn::Linear::new(weight, bias),
            output_dims,
        }))
    }

    fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        self.forward_views(xs)?
            .into_iter()
            .map(|view| view.contiguous())
            .collect()
    }

    fn forward_views(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let joined = self.linear.forward(&xs.contiguous()?)?;
        self.split_output(&joined)
    }

    /// Apply the current prefill GEMM policy before returning strided views.
    fn forward_prefill_views(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let joined = IsqLinear::Linear(self.linear.clone()).forward(&xs.contiguous()?)?;
        self.split_output(&joined)
    }

    fn split_output(&self, joined: &Tensor) -> Result<Vec<Tensor>> {
        let last_dim = joined.rank() - 1;
        let mut start = 0;
        let mut outputs = Vec::with_capacity(self.output_dims.len());
        for &dim in &self.output_dims {
            outputs.push(joined.narrow(last_dim, start, dim)?);
            start += dim;
        }
        Ok(outputs)
    }

    /// Gate and up keep contiguous slices of the joined allocation, including
    /// their nonzero offsets. Decode need not retain another copy of each weight.
    fn shared_linears(&self) -> Result<Vec<IsqLinear>> {
        let mut start = 0;
        self.output_dims
            .iter()
            .map(|&dim| {
                let weight = self.linear.weight().narrow(0, start, dim)?;
                let bias = self
                    .linear
                    .bias()
                    .map(|bias| bias.narrow(0, start, dim))
                    .transpose()?;
                start += dim;
                Ok(IsqLinear::Linear(candle_nn::Linear::new(weight, bias)))
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextMlp {
    fused_gate_up: Option<MetalDenseProjection>,
    prefill_gate_up: Option<MetalDenseProjection>,
    prefill_fused_silu_mul: bool,
    fused_silu_mul: bool,
    gate_proj: IsqLinear,
    up_proj: IsqLinear,
    down_proj: IsqLinear,
    hidden_act: String,
}

impl ThinkerTextMlp {
    fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        isq: Option<&str>,
        dense_fusion: bool,
        prefill_gate_up: bool,
    ) -> Result<Self> {
        let mut gate_proj = isq_linear::linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("gate_proj"),
            isq,
        )?;
        let mut up_proj = isq_linear::linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("up_proj"),
            isq,
        )?;
        let down_proj = isq_linear::linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("down_proj"),
            isq,
        )?;
        let fused_gate_up = MetalDenseProjection::new(&[&gate_proj, &up_proj], dense_fusion)?;
        let prefill_gate_up = if prefill_gate_up {
            match fused_gate_up.clone() {
                Some(projection) => Some(projection),
                None => MetalDenseProjection::new(&[&gate_proj, &up_proj], true)?,
            }
        } else {
            None
        };
        if let Some(projection) = &prefill_gate_up {
            let mut shared = projection.shared_linears()?.into_iter();
            gate_proj = shared
                .next()
                .ok_or_else(|| candle_core::Error::Msg("missing shared gate projection".into()))?;
            up_proj = shared
                .next()
                .ok_or_else(|| candle_core::Error::Msg("missing shared up projection".into()))?;
        }
        let fused_silu_mul = cfg!(feature = "metal-paged-attn")
            && vb.device().is_metal()
            && match std::env::var("VASR_METAL_SILU_MUL") {
                Err(std::env::VarError::NotPresent) => false,
                Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "on" => true,
                    "0" | "false" | "off" => false,
                    _ => candle_core::bail!("VASR_METAL_SILU_MUL must be 0 or 1"),
                },
                Err(error) => candle_core::bail!("invalid VASR_METAL_SILU_MUL: {error}"),
            };
        Ok(Self {
            fused_gate_up,
            prefill_gate_up,
            // An explicit old activation override still takes precedence. The
            // new path otherwise reads joined views without materializing them.
            prefill_fused_silu_mul: if std::env::var_os("VASR_METAL_SILU_MUL").is_some() {
                fused_silu_mul
            } else {
                true
            },
            fused_silu_mul,
            gate_proj,
            up_proj,
            down_proj,
            hidden_act: cfg.hidden_act.clone(),
        })
    }

    fn prefill_projection(&self) -> Option<&MetalDenseProjection> {
        if vasr_quant::isq_linear::linear_is_prefill() {
            self.prefill_gate_up.as_ref()
        } else {
            None
        }
    }
}

impl Module for ThinkerTextMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal-paged-attn")]
        if let Some(projection) = self.prefill_projection() {
            if !matches!(self.hidden_act.as_str(), "silu" | "swish") {
                candle_core::bail!("unsupported hidden_act={:?}", self.hidden_act);
            }
            let views = projection.forward_prefill_views(xs)?;
            let hidden = if self.prefill_fused_silu_mul {
                vasr_paged_attn::metal::pointwise::silu_mul(&views[0], &views[1])?
            } else {
                candle_nn::ops::silu(&views[0].contiguous()?)?
                    .broadcast_mul(&views[1].contiguous()?)?
            };
            return self.down_proj.forward(&hidden);
        }
        #[cfg(feature = "cuda")]
        if matches!(self.hidden_act.as_str(), "silu" | "swish")
            && let Some(hidden) =
                isq_linear::try_fused_q8_silu_gate_up(&self.gate_proj, &self.up_proj, xs)?
        {
            return self.down_proj.forward(&hidden);
        }
        let (gate, up) = if let Some(fused) = &self.fused_gate_up {
            let outputs = if self.fused_silu_mul {
                fused.forward_views(xs)?
            } else {
                fused.forward(xs)?
            };
            (outputs[0].clone(), outputs[1].clone())
        } else {
            (self.gate_proj.forward(xs)?, self.up_proj.forward(xs)?)
        };
        #[cfg(feature = "metal-paged-attn")]
        if self.fused_silu_mul && xs.device().is_metal() {
            if !matches!(self.hidden_act.as_str(), "silu" | "swish") {
                candle_core::bail!("unsupported hidden_act={:?}", self.hidden_act);
            }
            let hidden = vasr_paged_attn::metal::pointwise::silu_mul(&gate, &up)?;
            return self.down_proj.forward(&hidden);
        }
        let gate = match self.hidden_act.as_str() {
            "silu" | "swish" => candle_nn::ops::silu(&gate)?,
            other => candle_core::bail!("unsupported hidden_act={other:?}"),
        };
        let hidden = gate.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextAttention {
    #[cfg(feature = "paged-attn")]
    metal_auto_packed_paged: bool,
    fused_qkv: Option<MetalDenseProjection>,
    use_flash_attn: bool,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
    scaling: f64,
    q_proj: IsqLinear,
    k_proj: IsqLinear,
    v_proj: IsqLinear,
    o_proj: IsqLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    #[cfg(feature = "paged-attn")]
    paged_attn: Option<PagedAttentionHandle>,
}

impl ThinkerTextAttention {
    fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
        use_flash_attn: bool,
        isq: Option<&str>,
        dense_fusion: bool,
    ) -> Result<Self> {
        #[cfg(not(feature = "paged-attn"))]
        let _ = device;

        let head_dim = cfg.head_dim;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let num_key_value_groups = num_attention_heads / num_key_value_heads;

        let q_out = num_attention_heads * head_dim;
        let kv_out = num_key_value_heads * head_dim;

        let q_proj = isq_linear::linear_b(
            cfg.hidden_size,
            q_out,
            cfg.attention_bias,
            vb.pp("q_proj"),
            isq,
        )?;
        let k_proj = isq_linear::linear_b(
            cfg.hidden_size,
            kv_out,
            cfg.attention_bias,
            vb.pp("k_proj"),
            isq,
        )?;
        let v_proj = isq_linear::linear_b(
            cfg.hidden_size,
            kv_out,
            cfg.attention_bias,
            vb.pp("v_proj"),
            isq,
        )?;
        let o_proj = isq_linear::linear_b(
            q_out,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
            isq,
        )?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        #[cfg(feature = "paged-attn")]
        let paged_attn = Some(PagedAttentionHandle {
            k_scale: Tensor::new(1f32, device)?,
            v_scale: Tensor::new(1f32, device)?,
        });

        let fused_qkv = MetalDenseProjection::new(&[&q_proj, &k_proj, &v_proj], dense_fusion)?;
        Ok(Self {
            #[cfg(feature = "paged-attn")]
            metal_auto_packed_paged: false,
            fused_qkv,
            use_flash_attn,
            num_attention_heads,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            #[cfg(feature = "paged-attn")]
            paged_attn,
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

        if let Some(fused) = &self.fused_qkv {
            let outputs = fused.forward(hidden_states)?;
            return Ok((outputs[0].clone(), outputs[1].clone(), outputs[2].clone()));
        }

        Ok((
            self.q_proj.forward(hidden_states)?,
            self.k_proj.forward(hidden_states)?,
            self.v_proj.forward(hidden_states)?,
        ))
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &MultimodalRope,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;

        let (q, k, v) = self.project_qkv(hidden_states)?;
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((batch, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?, v)
        } else {
            let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
            let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
            let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
            let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
            let v = v
                .reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        };

        let (cos, sin) = position_embeddings;
        let (q, k) = rope_scaling.forward(&q, &k, cos, sin)?;

        if self.use_flash_attn && attention_mask.is_none() {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = token_attention_mask;
                candle_core::bail!("flash-attn support is not enabled in this build");
            }
            #[cfg(feature = "flash-attn")]
            {
                let softmax_scale = self.scaling as f32;

                let q4 = q.transpose(1, 2)?.contiguous()?; // (b, s, h, d)
                let k4 = k.transpose(1, 2)?.contiguous()?; // (b, s, kv, d)
                let v4 = v.transpose(1, 2)?.contiguous()?; // (b, s, kv, d)

                let Some(tok_mask) = token_attention_mask else {
                    let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                    let attn =
                        attn.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&attn);
                };

                let seqlens = seqlens_from_attention_mask(tok_mask, seq_len)?;
                let (cu, max_len, total_u32) =
                    cu_seqlens_u32(seqlens.as_slice(), hidden_states.device())?;
                let total = usize::try_from(total_u32).map_err(|_| {
                    candle_core::Error::Msg(format!(
                        "total sequence length overflows usize: total={total_u32}"
                    ))
                })?;

                let flat_total = batch.checked_mul(seq_len).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "batch*seq_len overflow: batch={batch} seq_len={seq_len}"
                    ))
                })?;

                if total == flat_total {
                    let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                    let attn =
                        attn.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&attn);
                }

                let idxs = mask_nonzero_indices_u32(tok_mask, seq_len)?;
                if idxs.len() != total {
                    candle_core::bail!(
                        "internal error: index len mismatch: idxs={} total={total}",
                        idxs.len()
                    );
                }

                let idx = Tensor::from_vec(idxs, (total,), hidden_states.device())?;

                let q_flat = q4.reshape((flat_total, self.num_attention_heads, self.head_dim))?;
                let k_flat = k4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;
                let v_flat = v4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;

                let q_unpad = q_flat.index_select(&idx, 0)?.contiguous()?;
                let k_unpad = k_flat.index_select(&idx, 0)?.contiguous()?;
                let v_unpad = v_flat.index_select(&idx, 0)?.contiguous()?;

                let out_unpad = flash_attn_varlen(
                    &q_unpad,
                    &k_unpad,
                    &v_unpad,
                    &cu,
                    &cu,
                    max_len,
                    max_len,
                    softmax_scale,
                    true,
                )?;

                let out = {
                    let zeros = Tensor::zeros(
                        (flat_total, self.num_attention_heads, self.head_dim),
                        out_unpad.dtype(),
                        out_unpad.device(),
                    )?;
                    let flat = zeros.index_add(&idx, &out_unpad, 0)?;
                    flat.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?
                };

                let out =
                    out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                return self.o_proj.forward(&out);
            }
        }

        let mask = match attention_mask {
            Some(mask) => attention::AttentionMask::Custom(mask.clone()),
            None => attention::AttentionMask::None,
        };
        let flash_params = if seq_len > 1 {
            let query_lens = vec![seq_len; batch];
            attention::make_flash_params(
                query_lens.as_slice(),
                query_lens.as_slice(),
                hidden_states.device(),
                true,
            )?
        } else {
            None
        };
        let q = maybe_contiguous_for_accelerator(q)?;
        let k = maybe_contiguous_for_accelerator(k)?;
        let v = maybe_contiguous_for_accelerator(v)?;
        let attn_output = attention::run_attention(
            &q,
            &k,
            &v,
            &mask,
            flash_params.as_ref(),
            self.scaling as f32,
            true,
            self.num_key_value_groups,
        )?;
        let attn_output = attn_output.transpose(1, 2)?; // (b, s, h, d)
        let attn_output =
            attn_output.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &MultimodalRope,
        layer_cache: (&mut KVCache, usize),
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;
        let (kv_cache, layer_idx) = layer_cache;

        let (q, k, v) = self.project_qkv(hidden_states)?;
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((batch, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?, v)
        } else {
            let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
            let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
            let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
            let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
            let v = v
                .reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        };

        let (cos, sin) = position_embeddings;
        let (q, k) = rope_scaling.forward(&q, &k, cos, sin)?;

        // Cache rotated keys to match the attention math.
        let (k, v) = kv_cache.update(layer_idx, &k, &v)?;

        if self.use_flash_attn && attention_mask.is_none() {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = token_attention_mask;
                candle_core::bail!("flash-attn support is not enabled in this build");
            }
            #[cfg(feature = "flash-attn")]
            {
                let softmax_scale = self.scaling as f32;
                let Some(token_attention_mask) = token_attention_mask else {
                    candle_core::bail!(
                        "flash-attn cached decode without a token attention mask is unsupported"
                    );
                };

                let (_b2, total_len) = token_attention_mask.dims2()?;
                if _b2 != batch {
                    candle_core::bail!(
                        "token_attention_mask batch mismatch: expected={batch}, got={_b2}"
                    );
                }
                let cache_len = total_len.checked_sub(seq_len).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "attention mask shorter than new seq_len: total_len={total_len} new_len={seq_len}"
                    ))
                })?;

                let q4 = q.transpose(1, 2)?.contiguous()?; // (b, s_q, h, d)
                let k4 = k.transpose(1, 2)?.contiguous()?; // (b, s_k, kv, d)
                let v4 = v.transpose(1, 2)?.contiguous()?; // (b, s_k, kv, d)

                if cache_len == 0 {
                    let seqlens = seqlens_from_attention_mask(token_attention_mask, seq_len)?;
                    let (cu, max_len, total_u32) =
                        cu_seqlens_u32(seqlens.as_slice(), hidden_states.device())?;
                    let total = usize::try_from(total_u32).map_err(|_| {
                        candle_core::Error::Msg(format!(
                            "total sequence length overflows usize: total={total_u32}"
                        ))
                    })?;

                    let flat_total = batch.checked_mul(seq_len).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "batch*seq_len overflow: batch={batch} seq_len={seq_len}"
                        ))
                    })?;

                    if total == flat_total {
                        let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                        let attn = attn.reshape((
                            batch,
                            seq_len,
                            self.num_attention_heads * self.head_dim,
                        ))?;
                        return self.o_proj.forward(&attn);
                    }

                    let idxs = mask_nonzero_indices_u32(token_attention_mask, seq_len)?;
                    if idxs.len() != total {
                        candle_core::bail!(
                            "internal error: index len mismatch: idxs={} total={total}",
                            idxs.len()
                        );
                    }
                    let idx = Tensor::from_vec(idxs, (total,), hidden_states.device())?;

                    let q_flat =
                        q4.reshape((flat_total, self.num_attention_heads, self.head_dim))?;
                    let k_flat =
                        k4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;
                    let v_flat =
                        v4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;

                    let q_unpad = q_flat.index_select(&idx, 0)?.contiguous()?;
                    let k_unpad = k_flat.index_select(&idx, 0)?.contiguous()?;
                    let v_unpad = v_flat.index_select(&idx, 0)?.contiguous()?;

                    let out_unpad = flash_attn_varlen(
                        &q_unpad,
                        &k_unpad,
                        &v_unpad,
                        &cu,
                        &cu,
                        max_len,
                        max_len,
                        softmax_scale,
                        true,
                    )?;

                    let out = {
                        let zeros = Tensor::zeros(
                            (flat_total, self.num_attention_heads, self.head_dim),
                            out_unpad.dtype(),
                            out_unpad.device(),
                        )?;
                        let flat = zeros.index_add(&idx, &out_unpad, 0)?;
                        flat.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?
                    };

                    let out =
                        out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&out);
                }

                // Cached decode step: q has no padding, but k/v include left padding.
                let k_total = batch
                    .checked_mul(total_len)
                    .ok_or_else(|| candle_core::Error::Msg("k/v size overflow".to_string()))?;
                let q_total = batch
                    .checked_mul(seq_len)
                    .ok_or_else(|| candle_core::Error::Msg("q size overflow".to_string()))?;

                let q3 = q4.reshape((q_total, self.num_attention_heads, self.head_dim))?;

                let seqlens_k = seqlens_from_attention_mask(token_attention_mask, total_len)?;
                let (cu_k, max_k, total_k_u32) =
                    cu_seqlens_u32(seqlens_k.as_slice(), hidden_states.device())?;
                let total_k = usize::try_from(total_k_u32).map_err(|_| {
                    candle_core::Error::Msg(format!(
                        "total_k overflows usize: total_k={total_k_u32}"
                    ))
                })?;

                let seqlens_q = vec![seq_len; batch];
                let (cu_q, max_q, total_q_u32) =
                    cu_seqlens_u32(seqlens_q.as_slice(), hidden_states.device())?;
                let total_q = usize::try_from(total_q_u32).map_err(|_| {
                    candle_core::Error::Msg(format!(
                        "total_q overflows usize: total_q={total_q_u32}"
                    ))
                })?;

                if total_q != q_total {
                    candle_core::bail!(
                        "internal error: q total mismatch: expected={q_total}, got={total_q}"
                    );
                }

                let idxs_k = mask_nonzero_indices_u32(token_attention_mask, total_len)?;
                if idxs_k.len() != total_k {
                    candle_core::bail!(
                        "internal error: k index len mismatch: idxs={} total_k={total_k}",
                        idxs_k.len()
                    );
                }
                let idx_k = Tensor::from_vec(idxs_k, (total_k,), hidden_states.device())?;

                let k3 = k4
                    .reshape((k_total, self.num_key_value_heads, self.head_dim))?
                    .index_select(&idx_k, 0)?
                    .contiguous()?;
                let v3 = v4
                    .reshape((k_total, self.num_key_value_heads, self.head_dim))?
                    .index_select(&idx_k, 0)?
                    .contiguous()?;

                let out = flash_attn_varlen(
                    &q3,
                    &k3,
                    &v3,
                    &cu_q,
                    &cu_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    true,
                )?;
                let out =
                    out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                return self.o_proj.forward(&out);
            }
        }

        let mask = match attention_mask {
            Some(mask) => attention::AttentionMask::Custom(mask.clone()),
            None => attention::AttentionMask::None,
        };
        let flash_params = if seq_len > 1 {
            let cache_len = kv_cache.seq_len();
            let query_lens = vec![seq_len; batch];
            let kv_lens = vec![cache_len.saturating_add(seq_len); batch];
            attention::make_flash_params(
                query_lens.as_slice(),
                kv_lens.as_slice(),
                hidden_states.device(),
                true,
            )?
        } else {
            None
        };
        let q = maybe_contiguous_for_accelerator(q)?;
        let k = maybe_contiguous_for_accelerator(k)?;
        let v = maybe_contiguous_for_accelerator(v)?;
        let attn_output = attention::run_attention(
            &q,
            &k,
            &v,
            &mask,
            flash_params.as_ref(),
            self.scaling as f32,
            true,
            self.num_key_value_groups,
        )?;
        let attn_output = attn_output.transpose(1, 2)?; // (b, s, h, d)
        let attn_output =
            attn_output.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    #[cfg(feature = "paged-attn")]
    fn forward_with_paged_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        input_metadata: &PagedInputMetadata,
        rope_scaling: &MultimodalRope,
        key_cache: &Tensor,
        value_cache: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;
        let paged_attn = self
            .paged_attn
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("paged attention is not initialized".into()))?;

        let (q, k, v) = self.project_qkv(hidden_states)?;
        let (q, k, v) = if seq_len == 1 {
            let q = q.reshape((batch, self.num_attention_heads, seq_len, self.head_dim))?;
            let k = k.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            let v = v.reshape((batch, self.num_key_value_heads, seq_len, self.head_dim))?;
            (self.q_norm.forward(&q)?, self.k_norm.forward(&k)?, v)
        } else {
            let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
            let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
            let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
            let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
            let v = v
                .reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        };

        let (cos, sin) = position_embeddings;
        let (q, k) = rope_scaling.forward(&q, &k, cos, sin)?;

        let flat_tokens = batch.checked_mul(seq_len).ok_or_else(|| {
            candle_core::Error::Msg("paged attention token count overflow".into())
        })?;
        let k_flat = if seq_len == 1 {
            maybe_contiguous_for_accelerator(k.reshape((
                flat_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
        } else {
            maybe_contiguous_for_accelerator(k.transpose(1, 2)?.reshape((
                flat_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
        }
        .detach();
        let v_flat = if seq_len == 1 {
            maybe_contiguous_for_accelerator(v.reshape((
                flat_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
        } else {
            maybe_contiguous_for_accelerator(v.transpose(1, 2)?.reshape((
                flat_tokens,
                self.num_key_value_heads,
                self.head_dim,
            ))?)?
        }
        .detach();
        #[cfg(any(feature = "cuda-paged-attn", feature = "metal-paged-attn"))]
        {
            reshape_and_cache(
                &k_flat,
                &v_flat,
                Some(&paged_attn.k_scale),
                Some(&paged_attn.v_scale),
                key_cache,
                value_cache,
                &input_metadata.slot_mapping,
            )?;
        }
        #[cfg(not(any(feature = "cuda-paged-attn", feature = "metal-paged-attn")))]
        {
            return candle_core::bail!(
                "paged-attn backend not enabled: build with cuda-paged-attn or metal-paged-attn"
            );
        }

        let attn = if seq_len == 1 {
            #[cfg(any(feature = "cuda-paged-attn", feature = "metal-paged-attn"))]
            {
                let q_flat = maybe_contiguous_for_accelerator(q.reshape((
                    batch,
                    self.num_attention_heads,
                    self.head_dim,
                ))?)?;
                crate::profiling::measure("decode_attention", hidden_states.device(), || {
                    paged_attention(
                        &q_flat,
                        Some(&paged_attn.k_scale),
                        Some(&paged_attn.v_scale),
                        key_cache,
                        value_cache,
                        &input_metadata.block_tables,
                        &input_metadata.context_lens,
                        None,
                        input_metadata.max_context_len,
                        self.scaling as f32,
                        1.0,
                        None,
                    )
                })?
            }
            #[cfg(not(any(feature = "cuda-paged-attn", feature = "metal-paged-attn")))]
            {
                return candle_core::bail!(
                    "paged-attn backend not enabled: build with cuda-paged-attn or metal-paged-attn"
                );
            }
        } else {
            let kv_lens = input_metadata
                .kv_lens
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("paged metadata missing kv_lens".into()))?;
            let attn_output = {
                #[cfg(feature = "metal-paged-attn")]
                {
                    let block_size = key_cache.dim(3)?;
                    if !metal_hybrid_paged_prefill_enabled()
                        && vasr_paged_attn::supports_metal_varlen_cache_dtype(
                            q.dtype(),
                            key_cache.dtype(),
                            value_cache.dtype(),
                        )
                        && vasr_paged_attn::supports_metal_varlen_prefill(
                            q.device(),
                            self.head_dim,
                            block_size,
                        )
                    {
                        if let (Some(cu_q), Some(query_lens)) = (
                            input_metadata.cu_seqlens_q.as_ref(),
                            input_metadata.query_lens.as_ref(),
                        ) {
                            let q_contig = maybe_contiguous_for_accelerator(q.clone())?;
                            let q_packed = vasr_paged_attn::pack_query_for_varlen_prefill(
                                &q_contig, query_lens,
                            )?;
                            let attn_packed = vasr_paged_attn::paged_attention_varlen_prefill(
                                &q_packed,
                                key_cache,
                                value_cache,
                                &input_metadata.block_tables,
                                &input_metadata.context_lens,
                                cu_q,
                                Some(&paged_attn.k_scale),
                                Some(&paged_attn.v_scale),
                                self.scaling as f32,
                                1.0,
                            )?;
                            vasr_paged_attn::unpack_varlen_attn_to_batched(
                                &attn_packed,
                                query_lens,
                                batch,
                                seq_len,
                            )?
                        } else {
                            run_paged_prefill_attention_fallback(
                                &q,
                                key_cache,
                                value_cache,
                                input_metadata,
                                kv_lens.as_slice(),
                                paged_attn,
                                batch,
                                seq_len,
                                hidden_states,
                                self.scaling as f32,
                                self.num_key_value_heads,
                                self.num_key_value_groups,
                                self.head_dim,
                            )?
                        }
                    } else {
                        run_paged_prefill_attention_fallback(
                            &q,
                            key_cache,
                            value_cache,
                            input_metadata,
                            kv_lens.as_slice(),
                            paged_attn,
                            batch,
                            seq_len,
                            hidden_states,
                            self.scaling as f32,
                            self.num_key_value_heads,
                            self.num_key_value_groups,
                            self.head_dim,
                        )?
                    }
                }
                #[cfg(not(feature = "metal-paged-attn"))]
                {
                    run_paged_prefill_attention_fallback(
                        &q,
                        key_cache,
                        value_cache,
                        input_metadata,
                        kv_lens.as_slice(),
                        paged_attn,
                        batch,
                        seq_len,
                        hidden_states,
                        self.scaling as f32,
                        self.num_key_value_heads,
                        self.num_key_value_groups,
                        self.head_dim,
                    )?
                }
            };
            attn_output.transpose(1, 2)?.reshape((
                batch,
                seq_len,
                self.num_attention_heads * self.head_dim,
            ))?
        };
        let attn = attn.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }

    #[cfg(feature = "paged-attn")]
    fn forward_packed_prefill_with_paged_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        prepared_position_embeddings: Option<&PreparedMultimodalRope>,
        input_metadata: &PagedInputMetadata,
        rope_scaling: &MultimodalRope,
        key_cache: &Tensor,
        value_cache: &Tensor,
    ) -> Result<Tensor> {
        let (packed_batch, total_tokens, _hidden) = hidden_states.dims3()?;
        if packed_batch != 1 {
            candle_core::bail!(
                "packed paged prefill expects batch=1 packed tokens, got batch={packed_batch}"
            );
        }
        let query_lens = input_metadata
            .query_lens
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("packed prefill missing query_lens".into()))?;
        let expected_total: usize = query_lens.iter().sum();
        if expected_total != total_tokens {
            candle_core::bail!(
                "packed prefill token mismatch: hidden={total_tokens} query_lens_total={expected_total}"
            );
        }
        let paged_attn = self
            .paged_attn
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("paged attention is not initialized".into()))?;

        let (q, k, v) = self.project_qkv(hidden_states)?;
        let q = q
            .reshape((total_tokens, self.num_attention_heads, self.head_dim))?
            .unsqueeze(0)?
            .transpose(1, 2)?;
        let k = k
            .reshape((total_tokens, self.num_key_value_heads, self.head_dim))?
            .unsqueeze(0)?
            .transpose(1, 2)?;
        let v = v
            .reshape((total_tokens, self.num_key_value_heads, self.head_dim))?
            .unsqueeze(0)?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (cos, sin) = position_embeddings;
        let (q, k) = match prepared_position_embeddings {
            Some(prepared) => prepared.forward(&q, &k)?,
            None => rope_scaling.forward(&q, &k, cos, sin)?,
        };

        let k_flat = maybe_contiguous_for_accelerator(k.transpose(1, 2)?.reshape((
            total_tokens,
            self.num_key_value_heads,
            self.head_dim,
        ))?)?
        .detach();
        let v_flat = maybe_contiguous_for_accelerator(v.transpose(1, 2)?.reshape((
            total_tokens,
            self.num_key_value_heads,
            self.head_dim,
        ))?)?
        .detach();
        #[cfg(any(feature = "cuda-paged-attn", feature = "metal-paged-attn"))]
        {
            reshape_and_cache(
                &k_flat,
                &v_flat,
                Some(&paged_attn.k_scale),
                Some(&paged_attn.v_scale),
                key_cache,
                value_cache,
                &input_metadata.slot_mapping,
            )?;
        }
        #[cfg(not(any(feature = "cuda-paged-attn", feature = "metal-paged-attn")))]
        {
            return candle_core::bail!(
                "paged-attn backend not enabled: build with cuda-paged-attn or metal-paged-attn"
            );
        }

        #[cfg(feature = "metal-paged-attn")]
        {
            static DIRECT_PAGED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
            let direct = DIRECT_PAGED
                .get_or_init(|| {
                    std::env::var("VASR_METAL_PACKED_PAGED_ATTN").ok().map(|v| {
                        matches!(
                            v.trim().to_ascii_lowercase().as_str(),
                            "1" | "true" | "yes" | "on"
                        )
                    })
                })
                .unwrap_or(self.metal_auto_packed_paged);
            if direct
                && vasr_paged_attn::supports_metal_varlen_cache_dtype(
                    q.dtype(),
                    key_cache.dtype(),
                    value_cache.dtype(),
                )
                && q.device().is_metal()
                && vasr_paged_attn::supports_metal_varlen_prefill(
                    q.device(),
                    self.head_dim,
                    key_cache.dim(3)?,
                )
            {
                let cu_q = input_metadata.cu_seqlens_q.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("packed prefill missing cu_seqlens_q".into())
                })?;
                let packed_query = q.transpose(1, 2)?.contiguous()?.reshape((
                    total_tokens,
                    self.num_attention_heads,
                    self.head_dim,
                ))?;
                let packed_output =
                    crate::profiling::measure("prefill_attention", q.device(), || {
                        vasr_paged_attn::paged_attention_varlen_prefill(
                            &packed_query,
                            key_cache,
                            value_cache,
                            &input_metadata.block_tables,
                            &input_metadata.context_lens,
                            cu_q,
                            Some(&paged_attn.k_scale),
                            Some(&paged_attn.v_scale),
                            self.scaling as f32,
                            1.0,
                        )
                    })?;
                let output = packed_output.reshape((
                    1usize,
                    total_tokens,
                    self.num_attention_heads * self.head_dim,
                ))?;
                return self.o_proj.forward(&output);
            }
        }

        let cu_kv = input_metadata.cu_seqlens_kv.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("packed prefill missing cu_seqlens_kv".into())
        })?;
        let (k_gathered, v_gathered) = gather_kv_cache(
            key_cache,
            value_cache,
            Some(&paged_attn.k_scale),
            Some(&paged_attn.v_scale),
            &input_metadata.block_tables,
            cu_kv,
            hidden_states.dtype(),
        )?;
        let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
        let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;
        let q = maybe_contiguous_for_accelerator(q)?;
        let attn_output = if q.device().is_metal() {
            let kv_lens = input_metadata.kv_lens.as_deref().ok_or_else(|| {
                candle_core::Error::Msg("packed Metal prefill missing host kv_lens".into())
            })?;
            attention::run_packed_prefill_attention(
                &q,
                &k_4d,
                &v_4d,
                query_lens,
                kv_lens,
                self.scaling as f32,
                true,
                self.num_key_value_groups,
            )?
        } else {
            attention::run_attention(
                &q,
                &k_4d,
                &v_4d,
                &attention::AttentionMask::None,
                input_metadata.prefill_flash_params(),
                self.scaling as f32,
                true,
                self.num_key_value_groups,
            )?
        };
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            1usize,
            total_tokens,
            self.num_attention_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextDecoderLayer {
    self_attn: ThinkerTextAttention,
    mlp: ThinkerTextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    fused_residual_norm: bool,
}

impl ThinkerTextDecoderLayer {
    fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
        use_flash_attn: bool,
        isq: Option<&str>,
        dense_fusion: bool,
        fused_residual_norm: bool,
        prefill_gate_up: bool,
    ) -> Result<Self> {
        let self_attn = ThinkerTextAttention::load(
            cfg,
            vb.pp("self_attn"),
            device,
            use_flash_attn,
            isq,
            dense_fusion,
        )?;
        let mlp = ThinkerTextMlp::load(cfg, vb.pp("mlp"), isq, dense_fusion, prefill_gate_up)?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            fused_residual_norm,
        })
    }

    fn post_attention_norm(
        &self,
        residual: &Tensor,
        attention: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "metal-paged-attn")]
        if self.fused_residual_norm
            && let Some(output) = vasr_paged_attn::metal::residual_norm::try_add_rms_norm(
                residual,
                attention,
                self.post_attention_layernorm.weight(),
                self.post_attention_layernorm.eps() as f32,
            )?
        {
            return Ok(output);
        }
        let sum = (residual + attention)?;
        let normalized = self.post_attention_layernorm.forward(&sum)?;
        Ok((sum, normalized))
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &MultimodalRope,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward(
            &x,
            position_embeddings,
            attention_mask,
            token_attention_mask,
            rope_scaling,
        )?;
        let (residual, x) = self.post_attention_norm(&residual, &x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }

    fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &MultimodalRope,
        layer_cache: (&mut KVCache, usize),
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward_with_kv_cache(
            &x,
            position_embeddings,
            attention_mask,
            token_attention_mask,
            rope_scaling,
            layer_cache,
        )?;
        let (residual, x) = self.post_attention_norm(&residual, &x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }

    #[cfg(feature = "paged-attn")]
    fn forward_with_paged_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        input_metadata: &PagedInputMetadata,
        rope_scaling: &MultimodalRope,
        key_cache: &Tensor,
        value_cache: &Tensor,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward_with_paged_cache(
            &x,
            position_embeddings,
            input_metadata,
            rope_scaling,
            key_cache,
            value_cache,
        )?;
        let (residual, x) = self.post_attention_norm(&residual, &x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }

    #[cfg(feature = "paged-attn")]
    fn forward_packed_prefill_with_paged_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        prepared_position_embeddings: Option<&PreparedMultimodalRope>,
        input_metadata: &PagedInputMetadata,
        rope_scaling: &MultimodalRope,
        key_cache: &Tensor,
        value_cache: &Tensor,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward_packed_prefill_with_paged_cache(
            &x,
            position_embeddings,
            prepared_position_embeddings,
            input_metadata,
            rope_scaling,
            key_cache,
            value_cache,
        )?;
        let (residual, x) = self.post_attention_norm(&residual, &x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }
}

/// Text part of Qwen3-ASR thinker (decoder-only transformer).
#[derive(Debug, Clone)]
pub struct ThinkerTextModel {
    embed_tokens: Embedding,
    layers: Vec<ThinkerTextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: MultimodalRope,
    prepared_mrope: bool,
    packed_prefill_rope: bool,
    #[cfg(feature = "paged-attn")]
    prepared_mrope_override: Option<bool>,
    #[cfg(feature = "paged-attn")]
    packed_rope_override: Option<bool>,
    hidden_size: usize,
    use_flash_attn: bool,
}

impl ThinkerTextModel {
    pub fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
        use_flash_attn: bool,
        isq: Option<&str>,
        progress: &Progress,
    ) -> Result<Self> {
        let dense_fusion = metal_dense_projection_fusion_enabled()?;
        // Parse once. The outer model supplies measured hardware/model/dtype
        // eligibility after loading these layers; explicit disables take priority.
        let (prepared_mrope_override, packed_rope_override) =
            if cfg!(feature = "metal-paged-attn") && device.is_metal() {
                (
                    read_rope_override("VASR_METAL_PREPARED_MROPE", parse_prepared_mrope_policy)?,
                    read_rope_override("VASR_METAL_PREFILL_ROPE_KERNEL", parse_packed_rope_policy)?,
                )
            } else {
                (Some(false), Some(false))
            };
        let (prepared_mrope, packed_prefill_rope) =
            resolve_rope_defaults(prepared_mrope_override, packed_rope_override, false);
        let prefill_gate_up = cfg!(feature = "metal-paged-attn")
            && device.is_metal()
            && parse_prefill_gate_up_policy(
                std::env::var("VASR_METAL_PREFILL_GATE_UP").ok().as_deref(),
            )?;
        let fused_residual_norm = cfg!(feature = "metal-paged-attn")
            && device.is_metal()
            && match std::env::var("VASR_METAL_RESIDUAL_RMSNORM") {
                Err(std::env::VarError::NotPresent) => false,
                Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "on" => true,
                    "0" | "false" | "off" => false,
                    _ => candle_core::bail!("VASR_METAL_RESIDUAL_RMSNORM must be 0 or 1"),
                },
                Err(error) => candle_core::bail!("invalid VASR_METAL_RESIDUAL_RMSNORM: {error}"),
            };
        if device.is_metal() {
            tracing::info!(target: "vasr::execution", fused_residual_norm, "Metal residual RMSNorm policy");
        }
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let phase = if isq.is_some() {
            "quantize.text_layers"
        } else {
            "load.text_layers"
        };
        let progress = progress.start(ProgressEvent::layers(
            "qwen3-asr",
            phase,
            cfg.num_hidden_layers,
        ));
        for idx in 0..cfg.num_hidden_layers {
            layers.push(ThinkerTextDecoderLayer::load(
                cfg,
                vb.pp("layers").pp(idx.to_string()),
                device,
                use_flash_attn,
                isq,
                dense_fusion,
                fused_residual_norm,
                prefill_gate_up,
            )?);
            progress.inc(1);
        }
        progress.finish();
        if device.is_metal() {
            let eligible_layers = layers
                .iter()
                .filter(|layer| layer.mlp.prefill_gate_up.is_some())
                .count();
            tracing::info!(target: "vasr::execution", requested=prefill_gate_up, eligible_layers,
                "Metal prefill-only gate/up candidate policy");
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = load_multimodal_rope(cfg, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            prepared_mrope,
            packed_prefill_rope,
            #[cfg(feature = "paged-attn")]
            prepared_mrope_override,
            #[cfg(feature = "paged-attn")]
            packed_rope_override,
            hidden_size: cfg.hidden_size,
            use_flash_attn,
        })
    }

    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    pub fn embed_tokens_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    #[cfg(feature = "paged-attn")]
    pub(crate) fn set_auto_packed_paged_prefill(&mut self, enabled: bool) {
        for layer in &mut self.layers {
            layer.self_attn.metal_auto_packed_paged = enabled;
        }
        (self.prepared_mrope, self.packed_prefill_rope) = resolve_rope_defaults(
            self.prepared_mrope_override,
            self.packed_rope_override,
            enabled,
        );
        if self.embed_tokens.embeddings().device().is_metal() {
            tracing::info!(target: "vasr::execution", override_value=?self.prepared_mrope_override,
                enabled=self.prepared_mrope, "Metal prefill position preparation policy");
            tracing::info!(target: "vasr::execution", override_value=?self.packed_rope_override,
                enabled=self.packed_prefill_rope, "Metal packed prefill RoPE policy");
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    #[cfg(feature = "paged-attn")]
    pub fn num_key_value_heads(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.self_attn.num_key_value_heads)
            .unwrap_or(0)
    }

    #[cfg(feature = "paged-attn")]
    pub fn head_dim(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.self_attn.head_dim)
            .unwrap_or(0)
    }

    pub fn forward(
        &self,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }

        let device = inputs_embeds.device();
        let dtype = inputs_embeds.dtype();
        let causal_mask = if self.use_flash_attn {
            None
        } else {
            Some(attention::make_causal_mask(
                attention_mask,
                batch,
                seq_len,
                dtype,
                device,
            )?)
        };

        // Shape is used only for dtype/device in the rope kernel, matching the official behavior.
        let (cos, sin) = self.rotary_emb.build_cache(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                position_embeddings,
                causal_mask.as_ref(),
                attention_mask,
                &self.rotary_emb,
            )?;
        }

        self.norm.forward(&hidden_states)
    }

    pub fn forward_with_kv_cache(
        &self,
        attention_mask: &Tensor,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }

        let (b2, total_len) = attention_mask.dims2()?;
        if b2 != batch {
            candle_core::bail!("attention_mask batch mismatch: expected={batch}, got={b2}");
        }

        let cache_len = kv_cache.seq_len();
        if total_len != cache_len.saturating_add(seq_len) {
            candle_core::bail!(
                "attention_mask total_len mismatch vs cache: total_len={total_len} cache_len={cache_len} new_len={seq_len}"
            );
        }

        let device = inputs_embeds.device();
        let dtype = inputs_embeds.dtype();
        let causal_mask = if self.use_flash_attn {
            None
        } else {
            Some(attention::make_causal_mask_cached(
                Some(attention_mask),
                batch,
                cache_len,
                seq_len,
                dtype,
                device,
            )?)
        };

        let (cos, sin) = self.rotary_emb.build_cache(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = initial_hidden_states_for_paged_decode(inputs_embeds)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_with_kv_cache(
                &hidden_states,
                position_embeddings,
                causal_mask.as_ref(),
                Some(attention_mask),
                &self.rotary_emb,
                (&mut *kv_cache, layer_idx),
            )?;
        }

        self.norm.forward(&hidden_states)
    }

    #[cfg(feature = "paged-attn")]
    pub fn forward_with_paged_cache(
        &self,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        paged_cache: &PagedKvCache,
        input_metadata: &PagedInputMetadata,
    ) -> Result<Tensor> {
        let (_batch, _seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }

        let (cos, sin) = self.rotary_emb.build_cache(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = initial_hidden_states_for_paged_decode(inputs_embeds)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (key_cache, value_cache) = paged_cache.key_value_cache(layer_idx)?;
            hidden_states = layer.forward_with_paged_cache(
                &hidden_states,
                position_embeddings,
                input_metadata,
                &self.rotary_emb,
                key_cache,
                value_cache,
            )?;
        }

        self.norm.forward(&hidden_states)
    }

    #[cfg(feature = "paged-attn")]
    pub fn forward_packed_prefill_with_paged_cache(
        &self,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        paged_cache: &PagedKvCache,
        input_metadata: &PagedInputMetadata,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if batch != 1 {
            candle_core::bail!("packed prefill expects inputs_embeds batch=1, got {batch}");
        }
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }

        let (cos, sin) = self.rotary_emb.build_cache(inputs_embeds, position_ids)?;
        let mut prepared = if self.prepared_mrope {
            self.rotary_emb.prepare_prefill_cache(&cos, &sin, seq_len)?
        } else {
            None
        };
        #[cfg(feature = "metal-paged-attn")]
        if self.packed_prefill_rope
            && let Some(prepared) = &mut prepared
        {
            prepared.enable_packed_metal();
        }
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = inputs_embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (key_cache, value_cache) = paged_cache.key_value_cache(layer_idx)?;
            hidden_states = layer.forward_packed_prefill_with_paged_cache(
                &hidden_states,
                position_embeddings,
                prepared.as_ref(),
                input_metadata,
                &self.rotary_emb,
                key_cache,
                value_cache,
            )?;
        }

        self.norm.forward(&hidden_states)
    }

    pub fn forward_decode_one_without_padding(
        &self,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle_core::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }
        if seq_len != 1 {
            candle_core::bail!("decode-one fast path expects seq_len=1, got={seq_len}");
        }

        let token_attention_mask = if self.use_flash_attn {
            let total_len = kv_cache.seq_len().saturating_add(seq_len);
            Some(Tensor::ones(
                (batch, total_len),
                candle_core::DType::U32,
                inputs_embeds.device(),
            )?)
        } else {
            None
        };

        // seq_len==1: only the first (temporal) modality is needed for mRoPE
        let (cos, sin) = self
            .rotary_emb
            .build_cache_first_modality(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = inputs_embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_with_kv_cache(
                &hidden_states,
                position_embeddings,
                None,
                token_attention_mask.as_ref(),
                &self.rotary_emb,
                (&mut *kv_cache, layer_idx),
            )?;
        }

        self.norm.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::load_multimodal_rope;
    #[cfg(feature = "paged-attn")]
    use super::unpack_gathered_kv_for_attention;

    fn dense_fixture(
        out: usize,
        input: usize,
        bias: bool,
        seed: usize,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> candle_core::Result<super::IsqLinear> {
        let values = (0..out * input)
            .map(|i| (((i * 17 + seed * 23) % 97) as f32 - 48.0) / 193.0)
            .collect::<Vec<_>>();
        let weight =
            candle_core::Tensor::from_vec(values, (out, input), device)?.to_dtype(dtype)?;
        let bias = if bias {
            Some(
                candle_core::Tensor::from_vec(
                    (0..out)
                        .map(|i| (i as f32 - 3.0) / 53.0)
                        .collect::<Vec<_>>(),
                    out,
                    device,
                )?
                .to_dtype(dtype)?,
            )
        } else {
            None
        };
        Ok(super::IsqLinear::Linear(candle_nn::Linear::new(
            weight, bias,
        )))
    }

    #[test]
    fn test_metal_dense_projection_fusion_leaves_cpu_unchanged() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let gate = dense_fixture(12, 8, false, 1, candle_core::DType::F32, &device)?;
        let up = dense_fixture(12, 8, false, 2, candle_core::DType::F32, &device)?;
        assert!(super::MetalDenseProjection::new(&[&gate, &up], true)?.is_none());
        Ok(())
    }

    #[test]
    fn measured_rope_defaults_require_validated_profile_and_keep_overrides() {
        use super::resolve_rope_defaults as resolve;
        assert_eq!(resolve(None, None, false), (false, false));
        assert_eq!(resolve(None, None, true), (true, true));
        assert_eq!(resolve(Some(false), None, true), (false, false));
        assert_eq!(resolve(None, Some(false), true), (false, false));
        assert_eq!(resolve(Some(false), Some(true), true), (false, false));
        assert_eq!(resolve(Some(true), Some(false), true), (true, false));
        assert_eq!(resolve(None, Some(true), false), (true, true));
        assert_eq!(resolve(Some(true), None, false), (true, false));
    }

    #[test]
    fn packed_rope_policy_requires_explicit_boolean() -> anyhow::Result<()> {
        assert!(!super::parse_packed_rope_policy(None)?);
        assert!(super::parse_packed_rope_policy(Some(" TRUE "))?);
        assert!(!super::parse_packed_rope_policy(Some("off"))?);
        assert!(super::parse_packed_rope_policy(Some("invalid")).is_err());
        Ok(())
    }

    #[test]
    fn prepared_mrope_candidate_requires_explicit_opt_in() -> anyhow::Result<()> {
        assert!(!super::parse_prepared_mrope_policy(None)?);
        for value in ["0", "false", "off", " FALSE "] {
            assert!(!super::parse_prepared_mrope_policy(Some(value))?);
        }
        for value in ["1", "true", "on", " TRUE "] {
            assert!(super::parse_prepared_mrope_policy(Some(value))?);
        }
        assert!(super::parse_prepared_mrope_policy(Some("invalid")).is_err());
        Ok(())
    }

    #[test]
    fn prefill_gate_up_candidate_requires_explicit_opt_in() -> anyhow::Result<()> {
        assert!(!super::parse_prefill_gate_up_policy(None)?);
        for value in ["0", "false", "off"] {
            assert!(!super::parse_prefill_gate_up_policy(Some(value))?);
        }
        for value in ["1", "true", "on"] {
            assert!(super::parse_prefill_gate_up_policy(Some(value))?);
        }
        assert!(super::parse_prefill_gate_up_policy(Some("invalid")).is_err());
        Ok(())
    }

    #[cfg(feature = "metal-paged-attn")]
    #[test]
    #[ignore = "exclusive Metal prefill-only MLP numerical comparison"]
    fn prefill_gate_up_candidate_preserves_decode_and_shared_weights() -> anyhow::Result<()> {
        use candle_core::{DType, Device, Tensor};
        use candle_nn::Module;
        use vasr_quant::{
            isq_linear::set_linear_is_prefill,
            metal_gemm::{GemmTile, set_prefill_tile},
        };
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let gate = dense_fixture(96, 32, false, 1, dtype, &device)?;
            let up = dense_fixture(96, 32, false, 2, dtype, &device)?;
            let down = dense_fixture(32, 96, false, 3, dtype, &device)?;
            let projection = super::MetalDenseProjection::new(&[&gate, &up], true)?.unwrap();
            let mut shared = projection.shared_linears()?.into_iter();
            let shared_gate = shared.next().unwrap();
            let shared_up = shared.next().unwrap();
            for (index, linear) in [&shared_gate, &shared_up].into_iter().enumerate() {
                let super::IsqLinear::Linear(linear) = linear else {
                    panic!("dense view expected")
                };
                assert!(linear.weight().is_contiguous());
                assert_eq!(linear.weight().dims(), [96, 32]);
                let (_storage, layout) = linear.weight().storage_and_layout();
                assert_eq!(layout.start_offset(), index * 96 * 32);
            }
            let reference = super::ThinkerTextMlp {
                fused_gate_up: None,
                prefill_gate_up: None,
                prefill_fused_silu_mul: false,
                fused_silu_mul: false,
                gate_proj: gate,
                up_proj: up,
                down_proj: down.clone(),
                hidden_act: "silu".into(),
            };
            let candidate = super::ThinkerTextMlp {
                fused_gate_up: None,
                prefill_gate_up: Some(projection),
                prefill_fused_silu_mul: true,
                fused_silu_mul: false,
                gate_proj: shared_gate,
                up_proj: shared_up,
                down_proj: down,
                hidden_act: "silu".into(),
            };
            for rows in [1, 17, 65, 257] {
                let input = Tensor::from_vec(
                    (0..(rows + 1) * 32)
                        .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                        .collect::<Vec<_>>(),
                    (1, rows + 1, 32),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(1, 1, rows)?;
                let _tiles = set_prefill_tile(Some(GemmTile::Adaptive));
                {
                    let _decode = set_linear_is_prefill(false);
                    assert!(candidate.prefill_projection().is_none());
                    assert_dense_close(
                        &candidate.forward(&input)?,
                        &reference.forward(&input)?,
                        0.,
                    )?;
                }
                let _prefill = set_linear_is_prefill(true);
                assert!(candidate.prefill_projection().is_some());
                let tolerance = match dtype {
                    DType::F32 => 1e-4,
                    DType::F16 => 0.005,
                    _ => 0.025,
                };
                assert_dense_close(
                    &candidate.forward(&input)?,
                    &reference.forward(&input)?,
                    tolerance,
                )?;
                let mut disabled_activation = candidate.clone();
                disabled_activation.prefill_fused_silu_mul = false;
                assert_dense_close(
                    &disabled_activation.forward(&input)?,
                    &reference.forward(&input)?,
                    tolerance,
                )?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal-paged-attn")]
    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn test_metal_joined_gate_up_views_preserve_values() -> anyhow::Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        for dtype in [candle_core::DType::F32, candle_core::DType::BF16] {
            let gate = dense_fixture(12, 8, false, 1, dtype, &device)?;
            let up = dense_fixture(12, 8, false, 2, dtype, &device)?;
            let projection = super::MetalDenseProjection::new(&[&gate, &up], true)?.unwrap();
            for (batch, seq) in [(1usize, 1usize), (32, 1), (3, 17)] {
                let x = candle_core::Tensor::from_vec(
                    (0..batch * seq * 8)
                        .map(|i| ((i * 11 % 83) as f32 - 41.) / 71.)
                        .collect::<Vec<_>>(),
                    (batch, seq, 8),
                    &device,
                )?
                .to_dtype(dtype)?;
                let copied = projection.forward(&x)?;
                let views = projection.forward_views(&x)?;
                assert_eq!(views.len(), 2);
                for (view, copied) in views.iter().zip(&copied) {
                    if batch * seq > 1 {
                        assert!(!view.is_contiguous());
                    }
                    assert_dense_close(view, copied, 0.)?;
                }
                let hidden = vasr_paged_attn::metal::pointwise::silu_mul(&views[0], &views[1])?;
                let reference = copied[0].silu()?.mul(&copied[1])?;
                assert_dense_close(&hidden, &reference, 0.)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn assert_metal_dense_projection_fusion(
        dtype: candle_core::DType,
        tolerance: f32,
    ) -> anyhow::Result<()> {
        use candle_nn::Module;
        let device = candle_core::Device::new_metal(0)?;
        let gate = dense_fixture(12, 8, false, 1, dtype, &device)?;
        let up = dense_fixture(12, 8, false, 2, dtype, &device)?;
        let down = dense_fixture(8, 12, false, 3, dtype, &device)?;
        let reference = super::ThinkerTextMlp {
            prefill_gate_up: None,
            prefill_fused_silu_mul: false,
            fused_silu_mul: false,
            gate_proj: gate.clone(),
            up_proj: up.clone(),
            down_proj: down.clone(),
            hidden_act: "silu".into(),
            fused_gate_up: None,
        };
        let fused = super::ThinkerTextMlp {
            prefill_gate_up: None,
            prefill_fused_silu_mul: false,
            fused_silu_mul: true,
            gate_proj: gate.clone(),
            up_proj: up.clone(),
            down_proj: down,
            hidden_act: "silu".into(),
            fused_gate_up: super::MetalDenseProjection::new(&[&gate, &up], true)?,
        };
        assert!(fused.fused_gate_up.is_some());
        assert!(super::MetalDenseProjection::new(&[&gate, &up], false)?.is_none());
        for batch in [1usize, 8, 32] {
            for seq in [1usize, 5] {
                let xs = candle_core::Tensor::from_vec(
                    (0..batch * seq * 8)
                        .map(|i| ((i * 11 % 83) as f32 - 41.0) / 71.0)
                        .collect::<Vec<_>>(),
                    (batch, seq, 8),
                    &device,
                )?
                .to_dtype(dtype)?;
                assert_dense_close(&fused.forward(&xs)?, &reference.forward(&xs)?, tolerance)?;
                // Q has more output channels than K/V; exercise bias and no bias.
                for bias in [false, true] {
                    let q = dense_fixture(8, 8, bias, 4, dtype, &device)?;
                    let k = dense_fixture(4, 8, bias, 5, dtype, &device)?;
                    let v = dense_fixture(4, 8, bias, 6, dtype, &device)?;
                    let projection = super::MetalDenseProjection::new(&[&q, &k, &v], true)?
                        .ok_or_else(|| anyhow::anyhow!("missing fused dense QKV"))?;
                    let got = projection.forward(&xs)?;
                    for (output, reference) in got.iter().zip([&q, &k, &v]) {
                        assert_dense_close(output, &reference.forward(&xs)?, tolerance)?;
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn assert_dense_close(
        got: &candle_core::Tensor,
        expected: &candle_core::Tensor,
        tolerance: f32,
    ) -> anyhow::Result<()> {
        assert_eq!(got.dims(), expected.dims());
        let got = got
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(got.iter().chain(&expected).all(|x| x.is_finite()));
        let max_abs = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs <= tolerance,
            "dense fusion mismatch max_abs={max_abs}, tolerance={tolerance}"
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires exclusive Metal GPU; run explicitly after stopping benchmark server"]
    fn test_metal_dense_projection_fusion_f32() -> anyhow::Result<()> {
        assert_metal_dense_projection_fusion(candle_core::DType::F32, 1e-4)
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires exclusive Metal GPU; run explicitly after stopping benchmark server"]
    fn test_metal_dense_projection_fusion_bf16() -> anyhow::Result<()> {
        assert_metal_dense_projection_fusion(candle_core::DType::BF16, 0.025)
    }

    #[cfg(feature = "metal-paged-attn")]
    #[test]
    #[ignore = "exclusive Metal prefill gate/up projection microbenchmark"]
    fn diagnose_prefill_gate_up_projection() -> anyhow::Result<()> {
        use candle_core::{DType, Device, Tensor};
        use vasr_quant::metal_gemm::{GemmTile, matmul_nt, set_prefill_tile};
        let device = Device::new_metal(0)?;
        let gate = dense_fixture(3072, 1024, false, 1, DType::BF16, &device)?;
        let up = dense_fixture(3072, 1024, false, 2, DType::BF16, &device)?;
        let joined = super::MetalDenseProjection::new(&[&gate, &up], true)?.unwrap();
        let _policy = set_prefill_tile(Some(GemmTile::Adaptive));
        for rows in [65, 256, 640, 1000, 2000] {
            let input = Tensor::from_vec(
                (0..rows * 1024)
                    .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                    .collect::<Vec<_>>(),
                (1, rows, 1024),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let run = |index| -> candle_core::Result<Tensor> {
                match index {
                    0 => {
                        let gate = candle_nn::ops::silu(&gate.forward(&input)?)?;
                        gate.broadcast_mul(&up.forward(&input)?)
                    }
                    1 => {
                        let outputs = joined.forward(&input)?;
                        candle_nn::ops::silu(&outputs[0])?.broadcast_mul(&outputs[1])
                    }
                    2 => {
                        let views = joined.forward_views(&input)?;
                        vasr_paged_attn::metal::pointwise::silu_mul(&views[0], &views[1])
                    }
                    _ => {
                        let joined = matmul_nt(&input, joined.linear.weight(), GemmTile::Adaptive)?;
                        vasr_paged_attn::metal::pointwise::silu_mul(
                            &joined.narrow(2, 0, 3072)?,
                            &joined.narrow(2, 3072, 3072)?,
                        )
                    }
                }
            };
            let reference = run(0)?;
            for index in 1..4 {
                assert_dense_close(&run(index)?, &reference, 0.125)?;
            }
            let mut times: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
            for iteration in 0..60 {
                for offset in 0..4 {
                    let index = if iteration % 2 == 0 {
                        (iteration + offset) % 4
                    } else {
                        (iteration + 4 - offset) % 4
                    };
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = run(index)?;
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if iteration >= 12 {
                        times[index].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for (mode, times) in [
                "independent_adaptive",
                "joined_candle_copy",
                "joined_candle_views",
                "joined_adaptive_views",
            ]
            .into_iter()
            .zip(&mut times)
            {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "gate_up rows={rows} mode={mode} median_us={:.3} min_us={:.3} max_us={:.3}",
                    (times[23] + times[24]) / 2.,
                    times[0],
                    times[47]
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal-paged-attn")]
    #[test]
    #[ignore = "exclusive Metal Qwen3-ASR-0.6B QKV projection microbenchmark"]
    fn diagnose_prefill_qkv_projection() -> anyhow::Result<()> {
        use candle_core::{DType, Device, Tensor};
        use vasr_quant::isq_linear::set_linear_is_prefill;
        use vasr_quant::metal_gemm::{GemmTile, set_prefill_tile};
        let device = Device::new_metal(0)?;
        // Actual 0.6B geometry: 16 query heads / 8 KV heads, head_dim=128.
        // Materialize each joined slice, as project_qkv consumers reshape it.
        let q = dense_fixture(2048, 1024, false, 4, DType::BF16, &device)?;
        let k = dense_fixture(1024, 1024, false, 5, DType::BF16, &device)?;
        let v = dense_fixture(1024, 1024, false, 6, DType::BF16, &device)?;
        let joined = super::MetalDenseProjection::new(&[&q, &k, &v], true)?.unwrap();
        let _policy = set_prefill_tile(Some(GemmTile::Adaptive));
        let _prefill = set_linear_is_prefill(true);
        for rows in [65, 256, 640, 1000, 2000] {
            let input = Tensor::from_vec(
                (0..(rows + 1) * 1024)
                    .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                    .collect::<Vec<_>>(),
                (1, rows + 1, 1024),
                &device,
            )?
            .to_dtype(DType::BF16)?
            .narrow(1, 1, rows)?;
            let run = |index| -> candle_core::Result<Vec<Tensor>> {
                match index {
                    0 => Ok(vec![
                        q.forward(&input)?,
                        k.forward(&input)?,
                        v.forward(&input)?,
                    ]),
                    1 => joined.forward(&input),
                    _ => joined
                        .forward_prefill_views(&input)?
                        .into_iter()
                        .map(|view| view.contiguous())
                        .collect(),
                }
            };
            let reference = run(0)?;
            for index in 1..3 {
                for (got, expected) in run(index)?.iter().zip(&reference) {
                    assert!(got.is_contiguous());
                    assert_dense_close(got, expected, 0.125)?;
                }
            }
            let mut times: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for iteration in 0..60 {
                for offset in 0..3 {
                    let index = if iteration % 2 == 0 {
                        (iteration + offset) % 3
                    } else {
                        (iteration + 3 - offset) % 3
                    };
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = run(index)?;
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if iteration >= 12 {
                        times[index].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for (mode, times) in [
                "independent_adaptive",
                "joined_candle_copy",
                "joined_adaptive_copy",
            ]
            .into_iter()
            .zip(&mut times)
            {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "qkv rows={rows} mode={mode} median_us={:.3} min_us={:.3} max_us={:.3}",
                    (times[23] + times[24]) / 2.,
                    times[0],
                    times[47]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_rotary_embedding_loads_without_rope_scaling() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let cfg = crate::config::TextConfig::default();
        let _ = load_multimodal_rope(&cfg, &device)?;
        Ok(())
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn test_unpack_gathered_kv_for_attention_repeats_groups() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let packed = candle_core::Tensor::from_vec(
            vec![1f32, 10.0, 2.0, 20.0, 3.0, 30.0],
            (3usize, 2usize, 1usize),
            &device,
        )?;
        let out = unpack_gathered_kv_for_attention(&packed, &[2, 1], 2, 2, 1, &device)?;
        let got = out.flatten_all()?.to_vec1::<f32>()?;
        let expected = vec![1.0, 2.0, 10.0, 20.0, 3.0, 0.0, 30.0, 0.0];
        if got != expected {
            anyhow::bail!("unexpected unpacked kv: expected={expected:?} got={got:?}");
        }
        Ok(())
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn test_unpack_gathered_kv_for_attention_equal_length_fast_path() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let packed = candle_core::Tensor::from_vec(
            vec![1f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            (4usize, 2usize, 1usize),
            &device,
        )?;
        let out = unpack_gathered_kv_for_attention(&packed, &[2, 2], 2, 2, 1, &device)?;
        let got = out.flatten_all()?.to_vec1::<f32>()?;
        let expected = vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        if got != expected {
            anyhow::bail!("unexpected equal-length unpacked kv: expected={expected:?} got={got:?}");
        }
        Ok(())
    }
}
