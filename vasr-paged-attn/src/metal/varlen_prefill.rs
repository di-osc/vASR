//! Metal fused varlen prefill attention over paged KV cache (attention.rs kernel).

use std::sync::OnceLock;

use candle_core::{DType, Device, MetalStorage, Result, Storage, Tensor};

use crate::metal::kernels::{self, Kernels, VarlenPrefillDType};

static KERNELS: OnceLock<Kernels> = OnceLock::new();

fn kernels() -> &'static Kernels {
    KERNELS.get_or_init(Kernels::new)
}

const SUPPORTED_HEAD_DIMS: [usize; 5] = [64, 96, 128, 192, 256];
const SUPPORTED_BLOCK_SIZES: [usize; 2] = [32, 64];

pub fn supports_metal_varlen_prefill(device: &Device, head_dim: usize, block_size: usize) -> bool {
    if std::env::var_os("VASR_DISABLE_METAL_VARLEN_PREFILL").is_some() {
        return false;
    }
    device.is_metal()
        && SUPPORTED_HEAD_DIMS.contains(&head_dim)
        && SUPPORTED_BLOCK_SIZES.contains(&block_size)
}

pub fn pack_query_for_varlen_prefill(q: &Tensor, query_lens: &[usize]) -> Result<Tensor> {
    let (batch, heads, seq_len, head_dim) = q.dims4()?;
    if query_lens.len() != batch {
        candle_core::bail!(
            "query_lens batch mismatch: expected={batch}, got={}",
            query_lens.len()
        );
    }
    let mut chunks = Vec::with_capacity(batch);
    for (row, &len) in query_lens.iter().enumerate() {
        if len > seq_len {
            candle_core::bail!("query_len exceeds padded seq_len: len={len} seq_len={seq_len}");
        }
        if len == 0 {
            candle_core::bail!("query_len must be non-zero for row {row}");
        }
        let flat = q
            .narrow(0, row, 1)?
            .narrow(2, 0, len)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((len, heads, head_dim))?;
        chunks.push(flat);
    }
    Tensor::cat(chunks.as_slice(), 0)
}

fn cumulative_seqlens_from_lengths(lengths: &[usize], device: &Device) -> Result<Tensor> {
    let mut out = Vec::with_capacity(lengths.len().saturating_add(1));
    out.push(0u32);
    let mut total = 0u32;
    for &len in lengths {
        let len_u32 = u32::try_from(len).map_err(|_| {
            candle_core::Error::Msg(format!("sequence length overflows u32: {len}"))
        })?;
        total = total.checked_add(len_u32).ok_or_else(|| {
            candle_core::Error::Msg("cumulative sequence length overflow".to_string())
        })?;
        out.push(total);
    }
    Tensor::from_vec(out, (lengths.len() + 1,), device)
}

pub fn pack_kv_for_varlen_prefill(
    k: &Tensor,
    v: &Tensor,
    query_lens: &[usize],
) -> Result<(Tensor, Tensor)> {
    let (batch, num_kv_heads, seq_len, head_dim) = k.dims4()?;
    let (v_batch, v_num_kv_heads, v_seq_len, v_head_dim) = v.dims4()?;
    if query_lens.len() != batch
        || v_batch != batch
        || v_num_kv_heads != num_kv_heads
        || v_seq_len != seq_len
        || v_head_dim != head_dim
    {
        candle_core::bail!(
            "packed kv shape mismatch: k={:?} v={:?} query_lens={}",
            k.shape(),
            v.shape(),
            query_lens.len()
        );
    }
    let mut k_chunks = Vec::with_capacity(batch);
    let mut v_chunks = Vec::with_capacity(batch);
    for (row, &len) in query_lens.iter().enumerate() {
        if len > seq_len {
            candle_core::bail!("query_len exceeds padded seq_len: len={len} seq_len={seq_len}");
        }
        if len == 0 {
            candle_core::bail!("query_len must be non-zero for row {row}");
        }
        k_chunks.push(
            k.narrow(0, row, 1)?
                .narrow(2, 0, len)?
                .transpose(1, 2)?
                .reshape((len, num_kv_heads, head_dim))?,
        );
        v_chunks.push(
            v.narrow(0, row, 1)?
                .narrow(2, 0, len)?
                .transpose(1, 2)?
                .reshape((len, num_kv_heads, head_dim))?,
        );
    }
    Ok((
        Tensor::cat(k_chunks.as_slice(), 0)?,
        Tensor::cat(v_chunks.as_slice(), 0)?,
    ))
}

pub fn unpack_varlen_attn_to_batched(
    attn: &Tensor,
    query_lens: &[usize],
    batch: usize,
    seq_len: usize,
) -> Result<Tensor> {
    let (total_tokens, heads, head_dim) = attn.dims3()?;
    if query_lens.len() != batch {
        candle_core::bail!(
            "query_lens batch mismatch: expected={batch}, got={}",
            query_lens.len()
        );
    }
    let expected_total: usize = query_lens.iter().sum();
    if expected_total != total_tokens {
        candle_core::bail!(
            "varlen attn token mismatch: expected={expected_total}, got={total_tokens}"
        );
    }

    let mut rows = Vec::with_capacity(batch);
    let mut offset = 0usize;
    for &len in query_lens {
        let slice = attn
            .narrow(0, offset, len)?
            .reshape((1, len, heads, head_dim))?
            .transpose(1, 2)?;
        let row_out = if len < seq_len {
            let pad = Tensor::zeros(
                (1, heads, seq_len - len, head_dim),
                attn.dtype(),
                attn.device(),
            )?;
            Tensor::cat(&[slice, pad], 2)?
        } else {
            slice
        };
        rows.push(row_out);
        offset += len;
    }
    Tensor::cat(rows.as_slice(), 0)
}

pub fn dense_attention_varlen_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    query_lens: &[usize],
    scale: f32,
    softcapping: f32,
    block_size: usize,
) -> Result<Tensor> {
    if !q.device().is_metal() {
        candle_core::bail!("metal dense varlen prefill requires a Metal device");
    }
    let (batch, num_q_heads, seq_len, head_dim) = q.dims4()?;
    if query_lens.len() != batch {
        candle_core::bail!(
            "query_lens batch mismatch: expected={batch}, got={}",
            query_lens.len()
        );
    }
    if !supports_metal_varlen_prefill(q.device(), head_dim, block_size) {
        candle_core::bail!(
            "unsupported metal dense varlen prefill configuration: head_dim={head_dim} block_size={block_size}"
        );
    }

    let q_packed = pack_query_for_varlen_prefill(q, query_lens)?;
    let (k_packed, v_packed) = pack_kv_for_varlen_prefill(k, v, query_lens)?;
    let cu_seqlens_q = cumulative_seqlens_from_lengths(query_lens, q.device())?;
    let seq_lens = Tensor::from_vec(
        query_lens
            .iter()
            .map(|&len| {
                u32::try_from(len).map_err(|_| {
                    candle_core::Error::Msg(format!("sequence length overflows u32: {len}"))
                })
            })
            .collect::<Result<Vec<_>>>()?,
        (batch,),
        q.device(),
    )?;

    let ty = VarlenPrefillDType::from_candle(q.dtype())
        .map_err(|e| candle_core::Error::Msg(format!("dense varlen prefill dtype: {e:?}")))?;
    let dev = q.device().as_metal_device()?;

    let (q_s, q_l) = q_packed.storage_and_layout();
    let Storage::Metal(q_s) = &*q_s else {
        candle_core::bail!("query must be a metal tensor");
    };
    let (k_s, k_l) = k_packed.storage_and_layout();
    let Storage::Metal(k_s) = &*k_s else {
        candle_core::bail!("key must be a metal tensor");
    };
    let (v_s, v_l) = v_packed.storage_and_layout();
    let Storage::Metal(v_s) = &*v_s else {
        candle_core::bail!("value must be a metal tensor");
    };
    let (seq_s, seq_l) = seq_lens.storage_and_layout();
    let Storage::Metal(seq_s) = &*seq_s else {
        candle_core::bail!("seq_lens must be a metal tensor");
    };
    let (cu_s, cu_l) = cu_seqlens_q.storage_and_layout();
    let Storage::Metal(cu_s) = &*cu_s else {
        candle_core::bail!("cu_seqlens_q must be a metal tensor");
    };

    let encoder = dev.command_encoder()?;
    encoder.set_label("vasr-dense-varlen-prefill");
    let elem_count = q_packed.elem_count();
    let out_buf = dev.new_buffer(elem_count, q.dtype(), "vasr-dense-varlen-prefill-out")?;

    kernels::call_dense_varlen_prefill(
        dev.device(),
        &encoder,
        kernels(),
        ty,
        &out_buf,
        q_s.buffer(),
        q_l.start_offset() * q.dtype().size_in_bytes(),
        k_s.buffer(),
        k_l.start_offset() * k.dtype().size_in_bytes(),
        v_s.buffer(),
        v_l.start_offset() * v.dtype().size_in_bytes(),
        seq_s.buffer(),
        seq_l.start_offset() * seq_lens.dtype().size_in_bytes(),
        cu_s.buffer(),
        cu_l.start_offset() * cu_seqlens_q.dtype().size_in_bytes(),
        k.dim(1)? as i32,
        scale,
        batch as i32,
        num_q_heads as i32,
        q_packed.dim(0)? as i32,
        head_dim as i32,
        block_size as i32,
        softcapping,
        q_l.stride()[0] as i32,
    )
    .map_err(|e| candle_core::Error::Msg(format!("vasr dense varlen prefill kernel: {e}")))?;

    // Unpacking submits GPU work; release the command encoder guard first.
    drop(encoder);
    let packed_out = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            dev.clone(),
            elem_count,
            q.dtype(),
        )),
        q_l.shape().clone(),
    ));
    unpack_varlen_attn_to_batched(&packed_out, query_lens, batch, seq_len)
}

/// Kernel reads dense KV with query's scalar type; FP8 is a separate specialization.
pub fn supports_metal_varlen_cache_dtype(query: DType, key: DType, value: DType) -> bool {
    matches!(query, DType::F16 | DType::BF16 | DType::F32)
        && key == value
        && (key == query || key == DType::F8E4M3)
}

fn validate_paged_varlen_layout(q: &Tensor, key: &Tensor, value: &Tensor) -> Result<()> {
    let (_, _, head_dim) = q.dims3()?;
    let (_, _, _, _, x) = key.dims5()?;
    if !q.is_contiguous()
        || !key.is_contiguous()
        || !value.is_contiguous()
        || x != 16 / key.dtype().size_in_bytes()
        || head_dim % x != 0
    {
        candle_core::bail!("unsupported varlen layout");
    }
    Ok(())
}

fn validate_fp8_scale(scale: &Tensor, device: &candle_core::Device) -> Result<()> {
    let (_storage, layout) = scale.storage_and_layout();
    if scale.dtype() != DType::F32
        || scale.elem_count() != 1
        || !scale.is_contiguous()
        || layout.start_offset() != 0
        || !scale.device().same_device(device)
    {
        candle_core::bail!("fp8 scale must be a same-device contiguous F32 scalar at offset zero");
    }
    Ok(())
}

fn resolve_fp8_scales(
    key_cache: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<Option<(Tensor, Tensor)>> {
    if key_cache.dtype() != DType::F8E4M3 {
        return Ok(None);
    }
    let (Some(k_scale), Some(v_scale)) = (k_scale, v_scale) else {
        candle_core::bail!("fp8 kv cache requires k_scale and v_scale");
    };
    Ok(Some((k_scale.clone(), v_scale.clone())))
}

#[allow(clippy::too_many_arguments)]
pub fn paged_attention_varlen_prefill(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    cu_seqlens_q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    if !q.device().is_metal() {
        candle_core::bail!("metal varlen prefill requires a Metal device");
    }
    if !supports_metal_varlen_cache_dtype(q.dtype(), key_cache.dtype(), value_cache.dtype()) {
        candle_core::bail!(
            "varlen prefill requires matching dense query/K/V dtypes or two FP8 caches"
        );
    }
    for tensor in [
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
    ] {
        if !q.device().same_device(tensor.device()) {
            candle_core::bail!("varlen prefill tensors must share the query device");
        }
    }
    for tensor in [block_tables, context_lens, cu_seqlens_q] {
        if tensor.dtype() != DType::U32 || !tensor.is_contiguous() {
            candle_core::bail!("varlen prefill metadata must be contiguous U32 tensors");
        }
    }
    let (total_tokens, num_q_heads, head_dim) = q.dims3()?;
    let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = key_cache.dims5()?;
    let (num_blocks_v, num_kv_heads_v, head_size_v, block_size_v) = value_cache.dims4()?;
    validate_paged_varlen_layout(q, key_cache, value_cache)?;
    if x == 0 || head_dim % x != 0 || num_kv_heads == 0 || num_q_heads % num_kv_heads != 0 {
        candle_core::bail!("invalid varlen prefill head/cache geometry");
    }
    if num_blocks != num_blocks_v
        || num_kv_heads != num_kv_heads_v
        || head_size_v != head_dim
        || block_size != block_size_v
        || head_size_kc != head_dim / x
    {
        candle_core::bail!(
            "paged cache shape mismatch for varlen prefill: key={:?} value={:?} head_dim={head_dim}",
            key_cache.shape(),
            value_cache.shape()
        );
    }
    if !supports_metal_varlen_prefill(q.device(), head_dim, block_size) {
        candle_core::bail!(
            "unsupported metal varlen prefill configuration: head_dim={head_dim} block_size={block_size}"
        );
    }

    let cu_len = cu_seqlens_q.dim(0)?;
    if cu_len < 2 || total_tokens == 0 {
        candle_core::bail!("varlen prefill requires at least one nonempty query sequence");
    }
    let num_seqs = cu_len - 1;
    let (num_seqs_bt, max_num_blocks_per_seq) = block_tables.dims2()?;
    if num_seqs_bt != num_seqs || context_lens.dim(0)? != num_seqs {
        candle_core::bail!(
            "varlen metadata batch mismatch: num_seqs={num_seqs} block_tables={num_seqs_bt} context_lens={}",
            context_lens.dim(0)?
        );
    }

    let ty = VarlenPrefillDType::from_candle(q.dtype())
        .map_err(|e| candle_core::Error::Msg(format!("varlen prefill dtype: {e:?}")))?;
    let fp8_scales = resolve_fp8_scales(key_cache, k_scale, v_scale)?;
    if let Some((k, v)) = &fp8_scales {
        validate_fp8_scale(k, q.device())?;
        validate_fp8_scale(v, q.device())?;
    }
    let quantized_cache = fp8_scales.is_some();

    let dev = q.device().as_metal_device()?;

    let (q_s, q_l) = q.storage_and_layout();
    let Storage::Metal(q_s) = &*q_s else {
        candle_core::bail!("query must be a metal tensor");
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let Storage::Metal(kc_s) = &*kc_s else {
        candle_core::bail!("key_cache must be a metal tensor");
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let Storage::Metal(vc_s) = &*vc_s else {
        candle_core::bail!("value_cache must be a metal tensor");
    };
    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let Storage::Metal(bt_s) = &*bt_s else {
        candle_core::bail!("block_tables must be a metal tensor");
    };
    let (cl_s, cl_l) = context_lens.storage_and_layout();
    let Storage::Metal(cl_s) = &*cl_s else {
        candle_core::bail!("context_lens must be a metal tensor");
    };
    let (cu_s, cu_l) = cu_seqlens_q.storage_and_layout();
    let Storage::Metal(cu_s) = &*cu_s else {
        candle_core::bail!("cu_seqlens_q must be a metal tensor");
    };
    let (o_s, o_l) = q.storage_and_layout();
    let Storage::Metal(_o_s) = &*o_s else {
        candle_core::bail!("query must be a metal tensor");
    };

    let k_scale_buf = fp8_scales
        .as_ref()
        .map(|(k, _)| metal_buffer(k))
        .transpose()?;
    let v_scale_buf = fp8_scales
        .as_ref()
        .map(|(_, v)| metal_buffer(v))
        .transpose()?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("vasr-varlen-prefill");
    let elem_count = q.elem_count();
    let out_buf = dev.new_buffer(elem_count, q.dtype(), "vasr-varlen-prefill-out")?;

    kernels::call_varlen_prefill(
        dev.device(),
        &encoder,
        kernels(),
        ty,
        quantized_cache,
        &out_buf,
        q_s.buffer(),
        q_l.start_offset() * q.dtype().size_in_bytes(),
        kc_s.buffer(),
        kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
        vc_s.buffer(),
        vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
        bt_s.buffer(),
        bt_l.start_offset() * block_tables.dtype().size_in_bytes(),
        cl_s.buffer(),
        cl_l.start_offset() * context_lens.dtype().size_in_bytes(),
        cu_s.buffer(),
        cu_l.start_offset() * cu_seqlens_q.dtype().size_in_bytes(),
        k_scale_buf.as_ref(),
        v_scale_buf.as_ref(),
        num_kv_heads as i32,
        scale,
        max_num_blocks_per_seq as i32,
        num_seqs as i32,
        num_q_heads as i32,
        total_tokens as i32,
        head_dim as i32,
        block_size as i32,
        softcapping,
        q_l.stride()[0] as i32,
        0,
        num_blocks as i32,
        kc_l.stride()[0] as i32,
        kc_l.stride()[1] as i32,
    )
    .map_err(|e| candle_core::Error::Msg(format!("vasr varlen prefill kernel: {e}")))?;

    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            dev.clone(),
            elem_count,
            q.dtype(),
        )),
        o_l.shape().clone(),
    )))
}

fn metal_buffer(t: &Tensor) -> Result<candle_metal_kernels::metal::Buffer> {
    let (storage, _) = t.storage_and_layout();
    match &*storage {
        Storage::Metal(s) => Ok(s.buffer().clone()),
        _ => candle_core::bail!("expected metal tensor"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn varlen_layout_rejects_token_gaps_and_wrong_cache_packing() -> anyhow::Result<()> {
        use candle_core::{DType, Device, Tensor};
        let q = Tensor::zeros((2, 2, 64), DType::F32, &Device::Cpu)?;
        let key = Tensor::zeros((2, 1, 16, 32, 4), DType::F32, &Device::Cpu)?;
        let value = Tensor::zeros((2, 1, 64, 32), DType::F32, &Device::Cpu)?;
        assert!(super::validate_paged_varlen_layout(&q, &key, &value).is_ok());
        let gapped = Tensor::zeros((2, 3, 64), DType::F32, &Device::Cpu)?.narrow(1, 0, 2)?;
        assert!(super::validate_paged_varlen_layout(&gapped, &key, &value).is_err());
        let wrong_x = Tensor::zeros((2, 1, 8, 32, 8), DType::F32, &Device::Cpu)?;
        assert!(super::validate_paged_varlen_layout(&q, &wrong_x, &value).is_err());
        Ok(())
    }
    #[test]
    fn paged_cache_dtype_eligibility_is_unconditional() {
        use candle_core::DType;
        assert!(super::supports_metal_varlen_cache_dtype(
            DType::BF16,
            DType::BF16,
            DType::BF16
        ));
        assert!(!super::supports_metal_varlen_cache_dtype(
            DType::BF16,
            DType::F16,
            DType::F16
        ));
        assert!(!super::supports_metal_varlen_cache_dtype(
            DType::BF16,
            DType::BF16,
            DType::F16
        ));
        assert!(!super::supports_metal_varlen_cache_dtype(
            DType::BF16,
            DType::F8E4M3,
            DType::BF16
        ));
        assert!(super::supports_metal_varlen_cache_dtype(
            DType::BF16,
            DType::F8E4M3,
            DType::F8E4M3
        ));
        assert!(!super::supports_metal_varlen_cache_dtype(
            DType::U32,
            DType::U32,
            DType::U32
        ));
    }

    #[test]
    fn fp8_scales_reject_wrong_dtype_shape_and_device() -> anyhow::Result<()> {
        use candle_core::{DType, Device, Tensor};
        let scale = Tensor::ones((), DType::F32, &Device::Cpu)?;
        assert!(super::validate_fp8_scale(&scale, &Device::Cpu).is_ok());
        assert!(super::validate_fp8_scale(&scale.to_dtype(DType::BF16)?, &Device::Cpu).is_err());
        assert!(
            super::validate_fp8_scale(&Tensor::ones(2, DType::F32, &Device::Cpu)?, &Device::Cpu)
                .is_err()
        );
        let offset = Tensor::ones(2, DType::F32, &Device::Cpu)?.narrow(0, 1, 1)?;
        assert!(super::validate_fp8_scale(&offset, &Device::Cpu).is_err());
        Ok(())
    }
    use super::{pack_query_for_varlen_prefill, unpack_varlen_attn_to_batched};
    use candle_core::{Device, Tensor};

    #[test]
    fn pack_unpack_roundtrip() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1f32, (2, 4, 5, 8), &device)?;
        let lens = vec![3usize, 2];
        let packed = pack_query_for_varlen_prefill(&q, &lens)?;
        assert_eq!(packed.dims3()?, (5, 4, 8));
        let restored = unpack_varlen_attn_to_batched(&packed, &lens, 2, 5)?;
        let row0 = q.narrow(0, 0, 1)?.narrow(2, 0, 3)?;
        let got0 = restored.narrow(0, 0, 1)?.narrow(2, 0, 3)?;
        let diff = (row0 - got0)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn packed_query_preserves_token_head_coordinates() -> anyhow::Result<()> {
        let q = Tensor::from_vec(
            (0..2 * 3 * 5 * 2).map(|i| i as f32).collect::<Vec<_>>(),
            (2, 3, 5, 2),
            &Device::Cpu,
        )?;
        let packed = pack_query_for_varlen_prefill(&q, &[3, 2])?.to_vec3::<f32>()?;
        let mut t = 0;
        for (row, len) in [3, 2].into_iter().enumerate() {
            for token in 0..len {
                for head in 0..3 {
                    let index = ((row * 3 + head) * 5 + token) * 2;
                    assert_eq!(packed[t][head], vec![index as f32, (index + 1) as f32]);
                }
                t += 1;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires isolated Metal GPU"]
    fn varlen_dense_and_paged_match_causal_reference_mixed_lengths() -> anyhow::Result<()> {
        use super::*;
        use candle_core::DType;
        let device = Device::new_metal(0)?;
        let lens = [3usize, 70, 13];
        for d in [64, 128] {
            let (b, h, kv, seq) = (3, 4, 2, 70);
            for dtype in [DType::F32, DType::BF16] {
                let tensor = |heads: usize, seed: usize| -> candle_core::Result<Tensor> {
                    Tensor::from_vec(
                        (0..b * heads * seq * d)
                            .map(|i| (((i * seed + 17) % 101) as f32 - 50.) / 100.)
                            .collect::<Vec<_>>(),
                        (b, heads, seq, d),
                        &device,
                    )?
                    .to_dtype(dtype)
                };
                let q = tensor(h, 7)?;
                let k = tensor(kv, 11)?;
                let v = tensor(kv, 13)?;
                let scale = 1. / (d as f32).sqrt();
                let dense = dense_attention_varlen_prefill(&q, &k, &v, &lens, scale, 1., 32)?;
                let cache = crate::PagedKvCache::new_batched(1, kv, d, 32, b, seq, dtype, &device)?;
                let tables = vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8]];
                let meta = cache.input_metadata_from_query_lens(&tables, &lens, &device)?;
                let (kc, vc) = cache.key_value_cache(0)?;
                let (pk, pv) = pack_kv_for_varlen_prefill(&k, &v, &lens)?;
                crate::reshape_and_cache(
                    &pk.contiguous()?,
                    &pv.contiguous()?,
                    None,
                    None,
                    kc,
                    vc,
                    &meta.slot_mapping,
                )?;
                let pq = pack_query_for_varlen_prefill(&q, &lens)?.contiguous()?;
                let paged = paged_attention_varlen_prefill(
                    &pq,
                    kc,
                    vc,
                    &meta.block_tables,
                    &meta.context_lens,
                    meta.cu_seqlens_q.as_ref().unwrap(),
                    None,
                    None,
                    scale,
                    1.,
                )?;
                let paged = unpack_varlen_attn_to_batched(&paged, &lens, b, seq)?;
                for (row, &len) in lens.iter().enumerate() {
                    let qr = q.narrow(0, row, 1)?.narrow(2, 0, len)?.contiguous()?;
                    let repeat = |t: &Tensor| -> candle_core::Result<Tensor> {
                        t.narrow(0, row, 1)?
                            .narrow(2, 0, len)?
                            .unsqueeze(2)?
                            .broadcast_as((1, kv, h / kv, len, d))?
                            .contiguous()?
                            .reshape((1, h, len, d))
                    };
                    let kr = repeat(&k)?;
                    let vr = repeat(&v)?;
                    let mask = Tensor::from_vec(
                        (0..len * len)
                            .map(|i| {
                                if i % len > i / len {
                                    f32::NEG_INFINITY
                                } else {
                                    0.
                                }
                            })
                            .collect::<Vec<_>>(),
                        (1, 1, len, len),
                        &device,
                    )?
                    .to_dtype(dtype)?;
                    let scores = qr
                        .matmul(&kr.transpose(2, 3)?.contiguous()?)?
                        .affine(scale as f64, 0.)?
                        .broadcast_add(&mask)?;
                    // Equivalent row-wise softmax, avoiding candle-nn dependency.
                    let max = scores.max_keepdim(3)?;
                    let exp = scores.broadcast_sub(&max)?.exp()?;
                    let weights = exp.broadcast_div(&exp.sum_keepdim(3)?)?;
                    let expected = weights
                        .matmul(&vr)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for (kind, actual) in [&dense, &paged].into_iter().enumerate() {
                        let got = actual
                            .narrow(0, row, 1)?
                            .narrow(2, 0, len)?
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        let tol = if dtype == DType::F32 { 1e-4 } else { 0.015 };
                        for (a, &e) in got.iter().zip(&expected) {
                            assert!(a.is_finite() && e.is_finite());
                            assert!(
                                (a - e).abs() <= tol + tol * e.abs(),
                                "kind={kind} {dtype:?} row={row} {a} != {e}"
                            );
                        }
                    }
                }
                // Unequal Q/KV lengths must align causal positions to the cache tail.
                let short_lens = [2usize, 7, 1];
                let mut short_meta =
                    cache.input_metadata_from_query_lens(&tables, &short_lens, &device)?;
                short_meta.context_lens = Tensor::from_vec(
                    lens.iter().map(|&n| n as u32).collect::<Vec<_>>(),
                    (b,),
                    &device,
                )?;
                let short_q = pack_query_for_varlen_prefill(&q, &short_lens)?.contiguous()?;
                let short_out = paged_attention_varlen_prefill(
                    &short_q,
                    kc,
                    vc,
                    &short_meta.block_tables,
                    &short_meta.context_lens,
                    short_meta.cu_seqlens_q.as_ref().unwrap(),
                    None,
                    None,
                    scale,
                    1.,
                )?;
                let short_out = unpack_varlen_attn_to_batched(&short_out, &short_lens, b, seq)?;
                for row in 0..b {
                    let (ql, kl) = (short_lens[row], lens[row]);
                    let qr = q.narrow(0, row, 1)?.narrow(2, 0, ql)?.contiguous()?;
                    let repeat = |t: &Tensor| -> candle_core::Result<Tensor> {
                        t.narrow(0, row, 1)?
                            .narrow(2, 0, kl)?
                            .unsqueeze(2)?
                            .broadcast_as((1, kv, h / kv, kl, d))?
                            .contiguous()?
                            .reshape((1, h, kl, d))
                    };
                    let kr = repeat(&k)?;
                    let vr = repeat(&v)?;
                    let mask = Tensor::from_vec(
                        (0..ql * kl)
                            .map(|i| {
                                if i % kl > kl - ql + i / kl {
                                    f32::NEG_INFINITY
                                } else {
                                    0.
                                }
                            })
                            .collect::<Vec<_>>(),
                        (1, 1, ql, kl),
                        &device,
                    )?
                    .to_dtype(dtype)?;
                    let scores = qr
                        .matmul(&kr.transpose(2, 3)?.contiguous()?)?
                        .affine(scale as f64, 0.)?
                        .broadcast_add(&mask)?;
                    let exp = scores.broadcast_sub(&scores.max_keepdim(3)?)?.exp()?;
                    let expected = exp
                        .broadcast_div(&exp.sum_keepdim(3)?)?
                        .matmul(&vr)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let got = short_out
                        .narrow(0, row, 1)?
                        .narrow(2, 0, ql)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let tol = if dtype == DType::F32 { 1e-4 } else { 0.015 };
                    for (a, e) in got.iter().zip(expected) {
                        assert!(a.is_finite() && e.is_finite());
                        assert!(
                            (a - e).abs() <= tol + tol * e.abs(),
                            "cached {dtype:?} row={row} {a} != {e}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
