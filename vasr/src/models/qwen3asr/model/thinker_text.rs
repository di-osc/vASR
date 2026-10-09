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
#[cfg(test)]
use crate::model::isq_linear::IsqLinear;
use crate::model::kv_cache::KVCache;
use crate::utils::progress::{Progress, ProgressEvent};

#[cfg(feature = "paged-attn")]
use crate::model::paged_kv_cache::{PagedInputMetadata, PagedKvCache};

mod dense_projection;
mod mlp;
pub(crate) mod policy;
#[cfg(test)]
mod tests;
mod text_attention;

use crate::model::operator_policy::OperatorEnvironment;
#[cfg(test)]
use dense_projection::MetalDenseProjection;
use mlp::ThinkerTextMlp;
use policy::TextOperatorPolicy;
#[cfg(test)]
use policy::{
    parse_packed_rope_policy, parse_prefill_gate_up_policy, parse_prepared_mrope_policy,
    resolve_rope_defaults,
};
#[cfg(all(test, feature = "paged-attn"))]
use text_attention::unpack_gathered_kv_for_attention;
use text_attention::{ThinkerTextAttention, initial_hidden_states_for_paged_decode};

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
        policy: &TextOperatorPolicy,
    ) -> Result<Self> {
        let self_attn = ThinkerTextAttention::load(
            cfg,
            vb.pp("self_attn"),
            device,
            use_flash_attn,
            isq,
            policy,
        )?;
        let mlp = ThinkerTextMlp::load(cfg, vb.pp("mlp"), isq, policy)?;
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
            fused_residual_norm: policy.fused_residual_norm,
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
        let env = OperatorEnvironment::capture();
        let policy = TextOperatorPolicy::resolve(
            &env,
            cfg!(feature = "metal-paged-attn") && device.is_metal(),
            false,
        )?;
        Self::load_with_policy(cfg, vb, device, use_flash_attn, isq, progress, &policy)
    }

    pub(crate) fn load_with_policy(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
        use_flash_attn: bool,
        isq: Option<&str>,
        progress: &Progress,
        policy: &TextOperatorPolicy,
    ) -> Result<Self> {
        if device.is_metal() {
            tracing::info!(target: "vasr::execution", fused_residual_norm=policy.fused_residual_norm, "Metal residual RMSNorm policy");
        }
        #[cfg(feature = "paged-attn")]
        if device.is_metal() {
            tracing::info!(target: "vasr::execution", override_value=?policy.prepared_override,
                enabled=policy.prepared_mrope, "Metal prefill position preparation policy");
            tracing::info!(target: "vasr::execution", override_value=?policy.packed_rope_override,
                enabled=policy.packed_prefill_rope, "Metal packed prefill RoPE policy");
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
                policy,
            )?);
            progress.inc(1);
        }
        progress.finish();
        if device.is_metal() {
            let eligible_layers = layers
                .iter()
                .filter(|layer| layer.mlp.prefill_gate_up.is_some())
                .count();
            tracing::info!(target: "vasr::execution", requested=policy.prefill_gate_up, eligible_layers,
                "Metal prefill-only gate/up candidate policy");
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = load_multimodal_rope(cfg, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            prepared_mrope: policy.prepared_mrope,
            packed_prefill_rope: policy.packed_prefill_rope,
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
