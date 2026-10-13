//! Window-isolated audio attention implementations.
use super::*;

// Bound each grouped score tensor to at most eight windows, regardless of
// how many samples/windows are packed into the encoder batch.
const AUDIO_ATTN_GROUP_MAX_WINDOWS: usize = 8;

#[derive(Debug, Clone)]
pub(super) struct AudioAttention {
    pub(super) mode: AudioAttentionMode,
    pub(super) use_flash_attn: bool,
    pub(super) num_heads: usize,
    pub(super) head_dim: usize,
    pub(super) scaling: f64,
    pub(super) q_proj: IsqLinear,
    pub(super) k_proj: IsqLinear,
    pub(super) v_proj: IsqLinear,
    pub(super) out_proj: IsqLinear,
}

impl AudioAttention {
    #[cfg(test)]
    pub(super) fn load(
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

    pub(super) fn load_with_policy(
        cfg: &AudioEncoderConfig,
        vb: VarBuilder,
        use_flash_attn: bool,
        isq: Option<&str>,
        policy: &AudioOperatorPolicy,
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
            mode: policy.attention,
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

    pub(super) fn forward(&self, hidden_states: Tensor, cu_seqlens: &[usize]) -> Result<Tensor> {
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
