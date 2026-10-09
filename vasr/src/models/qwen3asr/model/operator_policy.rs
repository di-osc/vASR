//! Immutable model-load configuration and the measured Metal profile boundary.
use std::collections::BTreeMap;
use std::ffi::OsString;

use crate::config::asr_config::ThinkerConfig;
use candle_core::{DType, Result};

/// The checkpoint fields that qualify the measured Qwen3-ASR 0.6B profile.
/// Service callers additionally retain their canonical model-ID gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelGeometry {
    pub model_type: Option<String>,
    pub audio_layers: usize,
    pub audio_hidden: usize,
    pub audio_output: usize,
    pub text_layers: usize,
    pub text_hidden: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub mel_bins: usize,
    pub downsample_hidden: usize,
    pub audio_window: usize,
}
impl ModelGeometry {
    pub fn from_config(cfg: &ThinkerConfig) -> Self {
        Self {
            model_type: cfg.model_type.clone(),
            audio_layers: cfg.audio_config.encoder_layers,
            audio_hidden: cfg.audio_config.d_model,
            audio_output: cfg.audio_config.output_dim,
            text_layers: cfg.text_config.num_hidden_layers,
            text_hidden: cfg.text_config.hidden_size,
            attention_heads: cfg.text_config.num_attention_heads,
            kv_heads: cfg.text_config.num_key_value_heads,
            head_dim: cfg.text_config.head_dim,
            mel_bins: cfg.audio_config.num_mel_bins,
            downsample_hidden: cfg.audio_config.downsample_hidden_size,
            audio_window: cfg.audio_config.n_window,
        }
    }
}

/// Pure qualification shared by model loading and service tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedMetalProfile {
    pub packed_prefill: bool,
    pub conv_layout: bool,
}
impl ValidatedMetalProfile {
    pub fn resolve(
        geometry: &ModelGeometry,
        dtype: DType,
        quantized: bool,
        hardware: bool,
    ) -> Self {
        let packed_prefill = hardware
            && dtype == DType::BF16
            && !quantized
            && matches!(
                geometry.model_type.as_deref(),
                Some("qwen3_asr" | "qwen3_asr_thinker")
            )
            && geometry.audio_layers == 18
            && geometry.audio_hidden == 896
            && geometry.audio_output == 1024
            && geometry.text_layers == 28
            && geometry.text_hidden == 1024
            && geometry.attention_heads == 16
            && geometry.kv_heads == 8
            && geometry.head_dim == 128;
        Self {
            packed_prefill,
            conv_layout: packed_prefill
                && geometry.mel_bins == 128
                && geometry.downsample_hidden == 480
                && geometry.audio_window == 50,
        }
    }
}

// Every model operator override is read exactly once at the construction boundary.
// Keeping OsString preserves legacy `var` versus `var_os` behavior for non-Unicode values.
const OPERATOR_KEYS: &[&str] = &[
    "VASR_METAL_DECODE_GEMM_TILE",
    "VASR_METAL_PREFILL_GEMM_TILE",
    "VASR_QWEN3_ASR_AUDIO_FLASH_ATTN",
    "VASR_METAL_DENSE_PROJECTION_FUSION",
    "VASR_METAL_PREPARED_MROPE",
    "VASR_METAL_PREFILL_ROPE_KERNEL",
    "VASR_METAL_PREFILL_GATE_UP",
    "VASR_METAL_RESIDUAL_RMSNORM",
    "VASR_METAL_SILU_MUL",
    "VASR_METAL_PACKED_PAGED_ATTN",
    "VASR_METAL_CONV_FLATTEN",
    "VASR_METAL_CONV_NHWC",
    "VASR_METAL_CONV_IM2COL32",
    "VASR_METAL_CONV_BIAS_GELU",
    "VASR_METAL_CONV_NATIVE_CHUNKS",
    "VASR_METAL_CONV_GEMM_TILE",
    "VASR_METAL_CONV_CROSS_SAMPLE",
    "VASR_METAL_CONV_CHUNKS",
    "VASR_QWEN3_ASR_AUDIO_ATTN",
    "VASR_METAL_ENCODER_BIAS_GELU",
    "VASR_QWEN3_ASR_AUDIO_SEQ_BUCKET",
    "VASR_DEBUG_AUDIO_ENCODER_LAYERS",
];
#[derive(Debug, Clone, Default)]
pub(crate) struct OperatorEnvironment(BTreeMap<&'static str, OsString>);
impl OperatorEnvironment {
    pub(crate) fn capture() -> Self {
        Self(
            OPERATOR_KEYS
                .iter()
                .filter_map(|&name| std::env::var_os(name).map(|value| (name, value)))
                .collect(),
        )
    }
    pub(crate) fn var(&self, name: &str) -> std::result::Result<&str, std::env::VarError> {
        match self.0.get(name) {
            None => Err(std::env::VarError::NotPresent),
            Some(value) => value
                .to_str()
                .ok_or_else(|| std::env::VarError::NotUnicode(value.clone())),
        }
    }
    pub(crate) fn present(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }
    #[cfg(test)]
    pub(crate) fn from_values(values: &[(&'static str, OsString)]) -> Self {
        Self(values.iter().cloned().collect())
    }
}

/// Final model/operator policy resolved before constructing either tower.
#[derive(Debug, Clone)]
pub(crate) struct ModelOperatorPolicy {
    pub(crate) profile: ValidatedMetalProfile,
    pub(crate) text: super::thinker_text::policy::TextOperatorPolicy,
    pub(crate) audio: super::audio_encoder::policy::AudioOperatorPolicy,
    pub(crate) audio_flash: bool,
    #[cfg(feature = "metal-paged-attn")]
    pub(crate) decode_gemm_tile: Option<vasr_quant::metal_gemm::GemmTile>,
    #[cfg(feature = "metal-paged-attn")]
    pub(crate) prefill_gemm_tile: Option<vasr_quant::metal_gemm::GemmTile>,
}
impl ModelOperatorPolicy {
    pub(crate) fn capture(
        cfg: &ThinkerConfig,
        vb: &candle_nn::VarBuilder,
        device: &candle_core::Device,
        use_flash_attn: bool,
        quantized: bool,
    ) -> Result<Self> {
        let env = OperatorEnvironment::capture();
        #[cfg(feature = "metal-paged-attn")]
        let hardware = vasr_paged_attn::metal::profile::validated_m4_pro_48gb(device);
        #[cfg(not(feature = "metal-paged-attn"))]
        let hardware = false;
        let profile = ValidatedMetalProfile::resolve(
            &ModelGeometry::from_config(cfg),
            vb.dtype(),
            quantized,
            hardware,
        );
        #[cfg(feature = "metal-paged-attn")]
        let gemm = |name: &str| -> Result<Option<vasr_quant::metal_gemm::GemmTile>> {
            if !device.is_metal() {
                return Ok(None);
            }
            match env.var(name).ok() {
                Some(value) => vasr_quant::metal_gemm::GemmTile::parse(value),
                None => Ok(profile
                    .packed_prefill
                    .then_some(vasr_quant::metal_gemm::GemmTile::Adaptive)),
            }
        };
        Ok(Self {
            text: super::thinker_text::policy::TextOperatorPolicy::resolve(
                &env,
                cfg!(feature = "metal-paged-attn") && device.is_metal(),
                profile.packed_prefill,
            )?,
            audio: super::audio_encoder::policy::AudioOperatorPolicy::resolve(
                &env,
                &cfg.audio_config,
                vb.device().is_metal(),
                profile.conv_layout,
            )?,
            audio_flash: super::audio_encoder::policy::audio_flash_attention(&env, use_flash_attn),
            #[cfg(feature = "metal-paged-attn")]
            decode_gemm_tile: gemm("VASR_METAL_DECODE_GEMM_TILE")?,
            #[cfg(feature = "metal-paged-attn")]
            prefill_gemm_tile: gemm("VASR_METAL_PREFILL_GEMM_TILE")?,
            profile,
        })
    }
}

/// Strict operators historically accept lowercase values without trimming.
pub(crate) fn parse_strict_bool(value: Option<&str>, name: &str) -> Result<bool> {
    match value {
        None | Some("0" | "false" | "off") => Ok(false),
        Some("1" | "true" | "on") => Ok(true),
        Some(value) => candle_core::bail!("invalid {name}={value:?}; expected 0 or 1"),
    }
}
/// Normalized legacy overrides; only convolution accepts yes/no aliases.
pub(crate) fn parse_normalized_override(
    value: Option<&str>,
    name: &str,
    yes_no: bool,
) -> Result<Option<bool>> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None => Ok(None),
        Some("1" | "true" | "on") => Ok(Some(true)),
        Some("0" | "false" | "off") => Ok(Some(false)),
        Some("yes") if yes_no => Ok(Some(true)),
        Some("no") if yes_no => Ok(Some(false)),
        _ => candle_core::bail!("{name} must be 0 or 1"),
    }
}
pub(crate) fn legacy_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_and_normalized_parsers_keep_distinct_legacy_contracts() -> Result<()> {
        assert!(!parse_strict_bool(None, "x")?);
        for value in ["1", "true", "on"] {
            assert!(parse_strict_bool(Some(value), "x")?);
        }
        for value in ["0", "false", "off"] {
            assert!(!parse_strict_bool(Some(value), "x")?);
        }
        for value in [" TRUE ", "yes", "no", "", "invalid"] {
            assert!(parse_strict_bool(Some(value), "x").is_err());
        }
        assert_eq!(parse_normalized_override(None, "x", true)?, None);
        assert_eq!(
            parse_normalized_override(Some(" OFF "), "x", true)?,
            Some(false)
        );
        assert_eq!(
            parse_normalized_override(Some(" YeS "), "x", true)?,
            Some(true)
        );
        assert!(parse_normalized_override(Some("yes"), "x", false).is_err());
        Ok(())
    }
    #[test]
    fn snapshots_are_instance_owned_and_preserve_presence() {
        let first = OperatorEnvironment::from_values(&[("x", "false".into())]);
        let second = OperatorEnvironment::from_values(&[("x", "true".into())]);
        assert!(first.present("x"));
        assert_eq!(first.var("x"), Ok("false"));
        assert_eq!(second.var("x"), Ok("true"));
        assert_eq!(first.var("missing"), Err(std::env::VarError::NotPresent));
    }
    #[cfg(unix)]
    #[test]
    fn non_unicode_presence_does_not_become_a_unicode_value() {
        use std::os::unix::ffi::OsStringExt;
        let env = OperatorEnvironment::from_values(&[("x", OsString::from_vec(vec![0xff]))]);
        assert!(env.present("x"));
        assert!(matches!(
            env.var("x"),
            Err(std::env::VarError::NotUnicode(_))
        ));
    }
    #[test]
    fn qualification_requires_the_actual_checkpoint_geometry() {
        let mut cfg = ThinkerConfig::default();
        cfg.model_type = Some("qwen3_asr".into());
        cfg.audio_config.encoder_layers = 18;
        cfg.audio_config.d_model = 896;
        cfg.audio_config.output_dim = 1024;
        cfg.text_config.num_hidden_layers = 28;
        cfg.text_config.hidden_size = 1024;
        cfg.text_config.num_attention_heads = 16;
        cfg.text_config.num_key_value_heads = 8;
        cfg.text_config.head_dim = 128;
        cfg.audio_config.n_window = 50;
        let geometry = ModelGeometry::from_config(&cfg);
        let profile = ValidatedMetalProfile::resolve(&geometry, DType::BF16, false, true);
        assert!(profile.packed_prefill && profile.conv_layout);
        for (dtype, quantized, hardware) in [
            (DType::F16, false, true),
            (DType::BF16, true, true),
            (DType::BF16, false, false),
        ] {
            assert!(
                !ValidatedMetalProfile::resolve(&geometry, dtype, quantized, hardware)
                    .packed_prefill
            );
        }
        for field in 0..9 {
            let mut invalid = geometry.clone();
            match field {
                0 => invalid.audio_layers += 1,
                1 => invalid.audio_hidden += 1,
                2 => invalid.audio_output += 1,
                3 => invalid.text_layers += 1,
                4 => invalid.text_hidden += 1,
                5 => invalid.attention_heads += 1,
                6 => invalid.kv_heads += 1,
                7 => invalid.head_dim += 1,
                _ => invalid.model_type = Some("qwen3_asr_forced_aligner".into()),
            }
            assert!(
                !ValidatedMetalProfile::resolve(&invalid, DType::BF16, false, true).packed_prefill
            );
        }
    }
}
