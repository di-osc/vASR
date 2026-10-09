//! Reusable multimodal RoPE (mRoPE) layer.

use candle_core::{DType, Device, IndexOp, Result, Tensor};

/// Multimodal rotary embedding generator for 3D positions.
///
/// `position_ids` is shaped `(3, batch, seq_len)` where the leading dimension
/// corresponds to `[temporal, height, width]`.
#[derive(Debug, Clone)]
pub struct MultimodalRope {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
    interleaved: bool,
}

/// Position data shared by layers within one prefill forward, never across requests.
/// Query and key projections are still rotated separately for every layer.
#[derive(Debug, Clone)]
pub struct PreparedMultimodalRope {
    cos: Tensor,
    sin: Tensor,
    #[cfg(feature = "metal-paged-attn")]
    packed_metal: bool,
}

impl PreparedMultimodalRope {
    #[cfg(feature = "metal-paged-attn")]
    pub(crate) fn enable_packed_metal(&mut self) {
        self.packed_metal = true;
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, _, seq, head_dim) = q.dims4()?;
        let (kv_batch, _, kv_seq, kv_dim) = k.dims4()?;
        if seq <= 1
            || (batch, seq, head_dim) != (kv_batch, kv_seq, kv_dim)
            || self.cos.dims() != [batch, 1, seq, head_dim]
            || self.sin.dims() != self.cos.dims()
        {
            candle_core::bail!(
                "prepared mRoPE requires matching multi-token q/k and position data"
            );
        }
        #[cfg(feature = "metal-paged-attn")]
        if self.packed_metal {
            let q = vasr_paged_attn::metal::rotary::try_packed_rope(q, &self.cos, &self.sin)?;
            let k = vasr_paged_attn::metal::rotary::try_packed_rope(k, &self.cos, &self.sin)?;
            if let (Some(q), Some(k)) = (q, k) {
                return Ok((q, k));
            }
        }
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        Ok((
            apply_rope_full_cache(&q, &self.cos, &self.sin)?,
            apply_rope_full_cache(&k, &self.cos, &self.sin)?,
        ))
    }
}

impl MultimodalRope {
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        mrope_section: Vec<usize>,
        interleaved: bool,
        device: &Device,
    ) -> Result<Self> {
        let _ = max_position_embeddings;
        let inv_freq = compute_default_inv_freq(head_dim, rope_theta, device)?;
        Ok(Self {
            inv_freq,
            mrope_section,
            interleaved,
        })
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        apply_multimodal_rotary_pos_emb(
            q,
            k,
            cos,
            sin,
            self.mrope_section.as_slice(),
            self.interleaved,
        )
    }

    /// Preserve the original single-token/standard paths. Interleaved prefill
    /// selects modalities and expands the half-head cache only once per forward.
    pub fn prepare_prefill_cache(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        seq_len: usize,
    ) -> Result<Option<PreparedMultimodalRope>> {
        if !self.interleaved || seq_len <= 1 {
            return Ok(None);
        }
        if cos.dims() != sin.dims()
            || cos.dim(2)? != seq_len
            || cos.dtype() != sin.dtype()
            || !cos.device().same_device(sin.device())
        {
            candle_core::bail!("prepared mRoPE position cache mismatch");
        }
        let (cos_half, sin_half) = prepare_interleaved_half_cache(cos, sin, &self.mrope_section)?;
        let cos = Tensor::cat(&[&cos_half, &cos_half], candle_core::D::Minus1)?.unsqueeze(1)?;
        let sin = Tensor::cat(&[&sin_half, &sin_half], candle_core::D::Minus1)?.unsqueeze(1)?;
        Ok(Some(PreparedMultimodalRope {
            cos,
            sin,
            #[cfg(feature = "metal-paged-attn")]
            packed_metal: false,
        }))
    }

    /// Build cos and sin cache for multimodal positions.
    ///
    /// Returns half-dim embeddings of shape `(3, batch, seq_len, head_dim/2)`.
    pub fn build_cache(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = x.dtype();
        let seq_len = position_ids.dim(2)?;

        let _ = seq_len;
        let inv_freq = self.inv_freq.clone();

        // inv_freq: (half_dim,) -> (1, 1, half_dim, 1)
        let inv_freq = inv_freq
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(3)?
            .to_dtype(DType::F32)?;

        // position_ids: (3, batch, seq_len) -> (3, batch, 1, seq_len)
        let position_ids = position_ids.unsqueeze(2)?.to_dtype(DType::F32)?;

        // freqs: (3, batch, half_dim, seq_len) -> (3, batch, seq_len, half_dim)
        let freqs = inv_freq.broadcast_mul(&position_ids)?;
        let freqs = freqs.transpose(2, 3)?.contiguous()?;

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok((cos, sin))
    }

    /// Build cos/sin cache for only the first modality (temporal), used in seq_len==1 decode.
    ///
    /// Returns `(cos, sin)` each of shape `(batch, seq_len, head_dim/2)`.
    pub fn build_cache_first_modality(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let dtype = x.dtype();
        let seq_len = position_ids.dim(2)?;

        let _ = seq_len;
        let inv_freq = self.inv_freq.clone();

        // Use only the first (temporal) position row: (1, batch, seq_len)
        let pos_first = position_ids.i(0)?;

        // inv_freq: (half_dim,) -> (1, 1, half_dim, 1)
        let inv_freq = inv_freq
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(3)?
            .to_dtype(DType::F32)?;

        // pos_first: (batch, seq_len) -> (batch, 1, 1, seq_len)
        let pos_first = pos_first.unsqueeze(1)?.unsqueeze(1)?.to_dtype(DType::F32)?;

        // freqs: (batch, 1, half_dim, seq_len) -> (batch, 1, seq_len, half_dim)
        let freqs = inv_freq.broadcast_mul(&pos_first)?;
        let freqs = freqs.transpose(2, 3)?.contiguous()?;

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        // Squeeze the modality dimension: (batch, 1, seq_len, half_dim) -> (batch, seq_len, half_dim)
        let cos = cos.squeeze(1)?;
        let sin = sin.squeeze(1)?;
        Ok((cos, sin))
    }
}

fn compute_default_inv_freq(head_dim: usize, rope_theta: f64, device: &Device) -> Result<Tensor> {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32))
        .collect();
    Tensor::from_vec(inv_freq, half_dim, device)
}

/// Apply multimodal rotary position embedding for 3D positions.
///
fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
    interleaved: bool,
) -> Result<(Tensor, Tensor)> {
    // seq_len==1 fast path: only the first modality is needed for decode.
    // Keep first-modality semantics instead of falling through to the full
    // 3-modality prefill path.
    if q.dim(2).unwrap_or(0) == 1 {
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            return apply_multimodal_rotary_pos_emb_seq_one_cuda(q, k, cos, sin, interleaved);
        }
        return apply_rope_batched_generic(q, k, cos, sin);
    }

    if interleaved {
        apply_multimodal_rotary_pos_emb_interleaved(q, k, cos, sin, mrope_section)
    } else {
        apply_multimodal_rotary_pos_emb_standard(q, k, cos, sin, mrope_section)
    }
}

#[cfg(feature = "cuda")]
fn apply_multimodal_rotary_pos_emb_seq_one_cuda(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    interleaved: bool,
) -> Result<(Tensor, Tensor)> {
    let (cos_first, sin_first) = if cos.rank() == 4 && cos.dim(0)? == 3 {
        (cos.i(0)?.squeeze(1)?, sin.i(0)?.squeeze(1)?)
    } else {
        (cos.squeeze(1)?, sin.squeeze(1)?)
    };

    let _ = interleaved;
    vasr_quant::apply_rotary_qk(q, k, &cos_first, &sin_first, true)
}

fn apply_rope_batched_generic(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Simplified non-accelerated path: take first modality and keep
    // (batch, seq_len, half_dim) for the generic batched RoPE helper.
    let (cos_first, sin_first) = if cos.rank() == 4 && cos.dim(0)? == 3 {
        (cos.i(0)?, sin.i(0)?)
    } else {
        (cos.clone(), sin.clone())
    };
    let q_embed = apply_rope_batched(q, &cos_first, &sin_first)?;
    let k_embed = apply_rope_batched(k, &cos_first, &sin_first)?;
    Ok((q_embed, k_embed))
}

fn apply_multimodal_rotary_pos_emb_standard(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let mut cos_parts: Vec<Tensor> = Vec::new();
    let mut sin_parts: Vec<Tensor> = Vec::new();
    let mut offset = 0usize;

    for (i, &section_size) in mrope_section.iter().enumerate() {
        let cos_modality = cos.i(i)?;
        let sin_modality = sin.i(i)?;

        let cos_section = cos_modality.narrow(candle_core::D::Minus1, offset, section_size)?;
        let sin_section = sin_modality.narrow(candle_core::D::Minus1, offset, section_size)?;

        cos_parts.push(cos_section);
        sin_parts.push(sin_section);
        offset = offset.saturating_add(section_size);
    }

    let cos_half = Tensor::cat(
        &cos_parts.iter().collect::<Vec<_>>(),
        candle_core::D::Minus1,
    )?
    .contiguous()?;
    let sin_half = Tensor::cat(
        &sin_parts.iter().collect::<Vec<_>>(),
        candle_core::D::Minus1,
    )?
    .contiguous()?;

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_embed = apply_rope_batched(&q, &cos_half, &sin_half)?;
    let k_embed = apply_rope_batched(&k, &cos_half, &sin_half)?;
    Ok((q_embed, k_embed))
}

fn apply_multimodal_rotary_pos_emb_interleaved(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let (cos_half, sin_half) = prepare_interleaved_half_cache(cos, sin, mrope_section)?;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_embed = apply_rope_batched(&q, &cos_half, &sin_half)?;
    let k_embed = apply_rope_batched(&k, &cos_half, &sin_half)?;
    Ok((q_embed, k_embed))
}

fn prepare_interleaved_half_cache(
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor)> {
    let (_modalities, _batch, _seq_len, half_dim) = cos.dims4()?;
    let modality_num = mrope_section.len();

    let original_dtype = cos.dtype();
    let cos_half = cos.contiguous()?.to_dtype(DType::F32)?;
    let sin_half = sin.contiguous()?.to_dtype(DType::F32)?;

    let m1_end = if mrope_section.len() > 1 {
        (mrope_section[1] * modality_num).min(half_dim)
    } else {
        0
    };
    let m2_end = if mrope_section.len() > 2 {
        (mrope_section[2] * modality_num).min(half_dim)
    } else {
        0
    };

    let cos_m0 = cos_half.i(0)?.contiguous()?;
    let sin_m0 = sin_half.i(0)?.contiguous()?;
    let cos_m1 = cos_half.i(1)?.contiguous()?;
    let sin_m1 = sin_half.i(1)?.contiguous()?;
    let cos_m2 = cos_half.i(2)?.contiguous()?;
    let sin_m2 = sin_half.i(2)?.contiguous()?;

    let mut cos_parts: Vec<Tensor> = Vec::with_capacity(half_dim);
    let mut sin_parts: Vec<Tensor> = Vec::with_capacity(half_dim);

    for pos in 0..half_dim {
        let modality = if modality_num >= 3 && mrope_section.len() >= 3 {
            if pos >= 1 && pos < m1_end && (pos - 1) % modality_num == 0 {
                1
            } else if pos >= 2 && pos < m2_end && (pos - 2) % modality_num == 0 {
                2
            } else {
                0
            }
        } else {
            0
        };

        let (cos_src, sin_src) = if modality == 0 {
            (&cos_m0, &sin_m0)
        } else if modality == 1 {
            (&cos_m1, &sin_m1)
        } else if modality == 2 {
            (&cos_m2, &sin_m2)
        } else {
            candle_core::bail!("invalid modality={modality} for interleaved mRoPE")
        };

        let cos_col = cos_src.narrow(2, pos, 1)?;
        let sin_col = sin_src.narrow(2, pos, 1)?;
        cos_parts.push(cos_col);
        sin_parts.push(sin_col);
    }

    let cos_half = Tensor::cat(&cos_parts.iter().collect::<Vec<_>>(), 2)?;
    let sin_half = Tensor::cat(&sin_parts.iter().collect::<Vec<_>>(), 2)?;

    let cos_half = cos_half.to_dtype(original_dtype)?.contiguous()?;
    let sin_half = sin_half.to_dtype(original_dtype)?.contiguous()?;

    Ok((cos_half, sin_half))
}

fn apply_rope_batched(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
    let (batch, _heads, seq_len, head_dim) = x.dims4()?;
    let (cos_batch, cos_seq, cos_half_dim) = cos_half.dims3()?;
    let (sin_batch, sin_seq, sin_half_dim) = sin_half.dims3()?;
    if cos_batch != batch || sin_batch != batch {
        candle_core::bail!("mRoPE batch mismatch: x={batch}, cos={cos_batch}, sin={sin_batch}");
    }
    if cos_seq < seq_len || sin_seq < seq_len {
        candle_core::bail!("mRoPE seq mismatch: x={seq_len}, cos={cos_seq}, sin={sin_seq}");
    }
    if cos_half_dim * 2 != head_dim || sin_half_dim * 2 != head_dim {
        candle_core::bail!(
            "mRoPE dim mismatch: x={head_dim}, cos_half={cos_half_dim}, sin_half={sin_half_dim}"
        );
    }

    let cos = Tensor::cat(&[cos_half, cos_half], candle_core::D::Minus1)?
        .narrow(1, 0, seq_len)?
        .unsqueeze(1)?;
    let sin = Tensor::cat(&[sin_half, sin_half], candle_core::D::Minus1)?
        .narrow(1, 0, seq_len)?
        .unsqueeze(1)?;
    apply_rope_full_cache(x, &cos, &sin)
}

fn apply_rope_full_cache(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // Keep the dtype rounding at each multiply and add; no arithmetic fusion.
    (x.broadcast_mul(cos)? + rotate_half(x)?.broadcast_mul(sin)?)?.contiguous()
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(candle_core::D::Minus1)?;
    let x1 = x.narrow(candle_core::D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(
        candle_core::D::Minus1,
        last_dim / 2,
        last_dim - last_dim / 2,
    )?;
    Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::{DType, Device, MultimodalRope, Result, Tensor};

    #[test]
    fn test_multimodal_rope_shapes() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;

        // head_dim=128 => half_dim=64, mrope_section sums to 64.
        let head_dim = 128usize;
        let rope = MultimodalRope::new(head_dim, 1024, 10000.0, vec![24, 20, 20], false, &device)?;

        let batch = 2usize;
        let seq_len = 7usize;

        // Dummy x: only dtype/device are used by forward.
        let x = candle_core::Tensor::zeros(
            (batch, 1, seq_len, head_dim),
            candle_core::DType::F32,
            &device,
        )?;

        // position_ids: (3, batch, seq_len)
        let pos1 = candle_core::Tensor::arange(0i64, seq_len as i64, &device)?.unsqueeze(0)?;
        let pos1 = pos1.broadcast_as((batch, seq_len))?;
        let position_ids = candle_core::Tensor::stack(&[&pos1, &pos1, &pos1], 0)?;

        let (cos, sin) = rope.build_cache(&x, &position_ids)?;
        let (m, b, s, half) = cos.dims4()?;
        if (m, b, s, half) != (3, batch, seq_len, head_dim / 2) {
            anyhow::bail!("unexpected cos dims: {:?}", cos.dims());
        }
        let (m, b, s, half) = sin.dims4()?;
        if (m, b, s, half) != (3, batch, seq_len, head_dim / 2) {
            anyhow::bail!("unexpected sin dims: {:?}", sin.dims());
        }

        // Apply mRoPE to q/k (shape checks only).
        let q = candle_core::Tensor::zeros(
            (batch, 4, seq_len, head_dim),
            candle_core::DType::F32,
            &device,
        )?;
        let k = candle_core::Tensor::zeros(
            (batch, 4, seq_len, head_dim),
            candle_core::DType::F32,
            &device,
        )?;
        let (q1, k1) = rope.forward(&q, &k, &cos, &sin)?;
        if q1.dims() != q.dims() || k1.dims() != k.dims() {
            anyhow::bail!("unexpected output dims for standard mRoPE");
        }

        let interleaved =
            MultimodalRope::new(head_dim, 1024, 10000.0, vec![24, 20, 20], true, &device)?;
        let (q2, k2) = interleaved.forward(&q, &k, &cos, &sin)?;
        if q2.dims() != q.dims() || k2.dims() != k.dims() {
            anyhow::bail!("unexpected output dims for interleaved mRoPE");
        }

        Ok(())
    }

    #[test]
    fn prepared_prefill_rope_preserves_unequal_modalities_and_offset_inputs() -> anyhow::Result<()>
    {
        let device = Device::Cpu;
        let rope = MultimodalRope::new(128, 4096, 1e6, vec![24, 20, 20], true, &device)?;
        for (batch, seq) in [(1, 2), (2, 7), (1, 65)] {
            let make_input = |heads, seed| -> Result<Tensor> {
                Tensor::from_vec(
                    (0..batch * (seq + 1) * heads * 128)
                        .map(|i| ((i * seed % 193) as f32 - 96.) / 151.)
                        .collect::<Vec<_>>(),
                    (batch, seq + 1, heads, 128),
                    &device,
                )?
                .narrow(1, 1, seq)?
                .transpose(1, 2)
            };
            let q = make_input(4, 17)?;
            let k = make_input(2, 13)?;
            // Temporal, height and width deliberately differ. Audio-only equal
            // positions would hide an incorrect modality selection.
            let positions = Tensor::from_vec(
                (0..3 * batch * seq)
                    .map(|i| (i / (batch * seq) * 101 + i % (batch * seq) * 3) as i64)
                    .collect::<Vec<_>>(),
                (3, batch, seq),
                &device,
            )?;
            let (cos, sin) = rope.build_cache(&q, &positions)?;
            let prepared = rope.prepare_prefill_cache(&cos, &sin, seq)?.unwrap();
            // Each layer must still apply the rotation to its own q/k values.
            for layer in 0..3 {
                let layer_q = q.affine(1., layer as f64 / 7.)?;
                let expected = rope.forward(&layer_q, &k, &cos, &sin)?;
                let got = prepared.forward(&layer_q, &k)?;
                for (got, expected) in [(&got.0, &expected.0), (&got.1, &expected.1)] {
                    assert_eq!(got.dims(), expected.dims());
                    assert_eq!(
                        got.flatten_all()?.to_vec1::<f32>()?,
                        expected.flatten_all()?.to_vec1::<f32>()?
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn prepared_prefill_rope_keeps_decode_and_standard_reference_paths() -> anyhow::Result<()> {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 4, 1, 128), DType::F32, &device)?;
        let positions = Tensor::from_vec(vec![7i64, 101, 203], (3, 1, 1), &device)?;
        for interleaved in [false, true] {
            let rope = MultimodalRope::new(128, 4096, 1e6, vec![24, 20, 20], interleaved, &device)?;
            let (cos, sin) = rope.build_cache(&q, &positions)?;
            assert!(rope.prepare_prefill_cache(&cos, &sin, 1)?.is_none());
            if !interleaved {
                assert!(rope.prepare_prefill_cache(&cos, &sin, 2)?.is_none());
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn prepared_prefill_rope_matches_metal_reference() -> anyhow::Result<()> {
        check_prepared_rope_metal(false)
    }

    #[cfg(feature = "metal-paged-attn")]
    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn packed_prefill_rope_matches_materialized_rounding_and_layout() -> anyhow::Result<()> {
        check_prepared_rope_metal(true)
    }

    #[cfg(feature = "metal")]
    fn check_prepared_rope_metal(packed: bool) -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        let rope = MultimodalRope::new(128, 4096, 1e6, vec![24, 20, 20], true, &device)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for seq in [2, 65, 641] {
                let input = |heads, seed| -> Result<Tensor> {
                    Tensor::from_vec(
                        (0..(seq + 1) * heads * 128)
                            .map(|i| ((i * seed % 193) as f32 - 96.) / 151.)
                            .collect::<Vec<_>>(),
                        (1, seq + 1, heads, 128),
                        &device,
                    )?
                    .to_dtype(dtype)?
                    .narrow(1, 1, seq)?
                    .transpose(1, 2)
                };
                let q = input(16, 17)?;
                let k = input(8, 13)?;
                let positions = Tensor::from_vec(
                    (0..3 * seq)
                        .map(|i| (i / seq * 101 + i % seq * 3) as i64)
                        .collect::<Vec<_>>(),
                    (3, 1, seq),
                    &device,
                )?;
                let (cos, sin) = rope.build_cache(&q, &positions)?;
                #[allow(unused_mut)]
                let mut prepared = rope.prepare_prefill_cache(&cos, &sin, seq)?.unwrap();
                #[cfg(feature = "metal-paged-attn")]
                if packed {
                    prepared.enable_packed_metal();
                }
                let expected = rope.forward(&q, &k, &cos, &sin)?;
                let got = prepared.forward(&q, &k)?;
                if packed {
                    assert!(got.0.transpose(1, 2)?.is_contiguous());
                    assert!(got.1.transpose(1, 2)?.is_contiguous());
                }
                for (got, expected) in [(&got.0, &expected.0), (&got.1, &expected.1)] {
                    assert_eq!(got.dims(), expected.dims());
                    let a = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let b = expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert!(a.iter().chain(&b).all(|value| value.is_finite()));
                    assert!(
                        a == b,
                        "prepared mRoPE changed values: dtype={dtype:?} seq={seq}"
                    );
                }
                assert!(
                    prepared
                        .forward(&q.narrow(2, 0, 1)?, &k.narrow(2, 0, 1)?)
                        .is_err()
                );
                eprintln!(
                    "prepared mRoPE dtype={dtype:?} seq={seq} q_heads=16 kv_heads=8 packed={packed} exact=true"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal-paged-attn")]
    #[test]
    #[ignore = "exclusive Metal 28-layer mRoPE microbenchmark"]
    fn diagnose_prepared_prefill_rope() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        let rope = MultimodalRope::new(128, 4096, 1e6, vec![24, 20, 20], true, &device)?;
        for seq in [256, 640, 2000] {
            let input = |heads, seed| -> Result<Tensor> {
                Tensor::from_vec(
                    (0..seq * heads * 128)
                        .map(|i| ((i * seed % 193) as f32 - 96.) / 151.)
                        .collect::<Vec<_>>(),
                    (1, seq, heads, 128),
                    &device,
                )?
                .to_dtype(DType::BF16)?
                .transpose(1, 2)
            };
            let q = input(16, 17)?;
            let k = input(8, 13)?;
            let positions = Tensor::from_vec(
                (0..3 * seq)
                    .map(|i| (i / seq * 101 + i % seq * 3) as i64)
                    .collect::<Vec<_>>(),
                (3, 1, seq),
                &device,
            )?;
            let (cos, sin) = rope.build_cache(&q, &positions)?;
            let mut times: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for iteration in 0..60 {
                for offset in 0..3 {
                    let mode = if iteration % 2 == 0 {
                        (iteration + offset) % 3
                    } else {
                        (iteration + 3 - offset) % 3
                    };
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    // Include the once-per-forward preparation in candidate timing.
                    let mut prepared = if mode > 0 {
                        rope.prepare_prefill_cache(&cos, &sin, seq)?
                    } else {
                        None
                    };
                    if mode == 2 {
                        prepared.as_mut().unwrap().enable_packed_metal();
                    }
                    for _ in 0..28 {
                        let output = match &prepared {
                            Some(prepared) => prepared.forward(&q, &k)?,
                            None => rope.forward(&q, &k, &cos, &sin)?,
                        };
                        // Both real consumers need B,S,H,D contiguous storage.
                        // The fused output is already stored in this order.
                        std::hint::black_box((
                            output.0.transpose(1, 2)?.contiguous()?,
                            output.1.transpose(1, 2)?.contiguous()?,
                        ));
                    }
                    device.synchronize()?;
                    if iteration >= 12 {
                        times[mode].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for (mode, samples) in [
                "per_layer_reference",
                "per_forward_prepared",
                "packed_kernel",
            ]
            .into_iter()
            .zip(&mut times)
            {
                samples.sort_by(f64::total_cmp);
                eprintln!(
                    "rope seq={seq} layers=28 mode={mode} median_us={:.3} min_us={:.3} max_us={:.3}",
                    (samples[23] + samples[24]) / 2.,
                    samples[0],
                    samples[47]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_seq_one_first_modality_matches_standard_when_positions_match() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let head_dim = 128usize;
        let rope = MultimodalRope::new(head_dim, 1024, 10000.0, vec![24, 20, 20], false, &device)?;

        let batch = 2usize;
        let seq_len = 1usize;
        let x = candle_core::Tensor::zeros(
            (batch, 1, seq_len, head_dim),
            candle_core::DType::F32,
            &device,
        )?;
        let pos = candle_core::Tensor::from_vec(vec![70i64, 71], (batch, seq_len), &device)?;
        let position_ids = candle_core::Tensor::stack(&[&pos, &pos, &pos], 0)?;
        let (cos_full, sin_full) = rope.build_cache(&x, &position_ids)?;
        let (cos_one, sin_one) = rope.build_cache_first_modality(&x, &position_ids)?;

        let q =
            candle_core::Tensor::randn(0f32, 0.02f32, (batch, 4usize, seq_len, head_dim), &device)?;
        let k =
            candle_core::Tensor::randn(0f32, 0.02f32, (batch, 4usize, seq_len, head_dim), &device)?;
        let mrope_section = &[24usize, 20, 20];

        let (q_standard, k_standard) = super::apply_multimodal_rotary_pos_emb_standard(
            &q,
            &k,
            &cos_full,
            &sin_full,
            mrope_section,
        )?;
        let (q_one, k_one) = super::apply_rope_batched_generic(&q, &k, &cos_one, &sin_one)?;

        let q_diff = (q_standard - q_one)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let k_diff = (k_standard - k_one)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(q_diff < 1e-6, "q diff {q_diff}");
        assert!(k_diff < 1e-6, "k diff {k_diff}");
        Ok(())
    }
}
