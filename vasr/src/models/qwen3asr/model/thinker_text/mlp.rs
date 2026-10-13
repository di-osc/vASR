//! Decoder MLP, including the measured and experimental Metal branches.
use super::dense_projection::MetalDenseProjection;
use super::policy::TextOperatorPolicy;
use crate::config::TextConfig;
use crate::model::isq_linear::{self, IsqLinear};
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

#[derive(Debug, Clone)]
pub(super) struct ThinkerTextMlp {
    pub(super) fused_gate_up: Option<MetalDenseProjection>,
    pub(super) prefill_gate_up: Option<MetalDenseProjection>,
    #[cfg(feature = "metal-paged-attn")]
    pub(super) prefill_fused_silu_mul: bool,
    pub(super) fused_silu_mul: bool,
    pub(super) gate_proj: IsqLinear,
    pub(super) up_proj: IsqLinear,
    pub(super) down_proj: IsqLinear,
    pub(super) hidden_act: String,
}

impl ThinkerTextMlp {
    pub(super) fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        isq: Option<&str>,
        policy: &TextOperatorPolicy,
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
        let fused_gate_up =
            MetalDenseProjection::new(&[&gate_proj, &up_proj], policy.dense_fusion)?;
        let prefill_gate_up = if policy.prefill_gate_up {
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
        Ok(Self {
            fused_gate_up,
            prefill_gate_up,
            #[cfg(feature = "metal-paged-attn")]
            prefill_fused_silu_mul: policy.prefill_fused_silu_mul,
            fused_silu_mul: policy.fused_silu_mul,
            gate_proj,
            up_proj,
            down_proj,
            hidden_act: cfg.hidden_act.clone(),
        })
    }

    #[cfg(feature = "metal-paged-attn")]
    pub(super) fn prefill_projection(&self) -> Option<&MetalDenseProjection> {
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
