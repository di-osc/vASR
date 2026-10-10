//! Audio operator configuration captured once for the whole tower.
use crate::config::AudioEncoderConfig;
use crate::model::operator_policy::{
    OperatorEnvironment, legacy_truthy, parse_normalized_override,
};
use candle_core::Result;

#[derive(Debug, Clone, Default)]
pub(super) struct ConvExecutionPolicy {
    pub(super) auto: bool,
    pub(super) nhwc_override: Option<bool>,
    pub(super) im2col32_override: Option<bool>,
    pub(super) bias_gelu_override: Option<bool>,
    pub(super) native_chunks_override: Option<bool>,
    pub(super) flatten: bool,
    #[cfg(feature = "metal-paged-attn")]
    pub(super) gemm_tile: Option<vasr_quant::metal_gemm::GemmTile>,
}
impl ConvExecutionPolicy {
    #[cfg(test)]
    pub(super) fn from_values(nhwc: Option<&str>, im2col32: Option<&str>) -> Result<Self> {
        Ok(Self {
            nhwc_override: Self::parse_override(nhwc, "VASR_METAL_CONV_NHWC")?,
            im2col32_override: Self::parse_override(im2col32, "VASR_METAL_CONV_IM2COL32")?,
            ..Self::default()
        })
    }
    pub(super) fn parse_override(value: Option<&str>, name: &str) -> Result<Option<bool>> {
        parse_normalized_override(value, name, true)
    }
    pub(super) fn fused_bias_gelu(&self) -> bool {
        self.resolve().0 && self.bias_gelu_override.unwrap_or(self.auto)
    }
    pub(super) fn native_chunks(&self) -> bool {
        self.native_chunks_override.unwrap_or(false)
    }
    pub(super) fn resolve(&self) -> (bool, bool) {
        let nhwc = self.nhwc_override.unwrap_or(self.auto);
        (nhwc, nhwc && self.im2col32_override.unwrap_or(self.auto))
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AudioAttentionMode {
    Reference,
    ProjectOnce,
    Grouped,
    GroupedNative,
}
impl AudioAttentionMode {
    fn resolve(env: &OperatorEnvironment) -> Result<Self> {
        match env.var("VASR_QWEN3_ASR_AUDIO_ATTN") {
            Err(std::env::VarError::NotPresent) => Ok(Self::Reference),
            Ok(value) => match value {
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
pub(crate) struct AudioOperatorPolicy {
    pub(super) conv: ConvExecutionPolicy,
    pub(super) cross_sample: bool,
    pub(super) micro_batch: usize,
    pub(super) attention: AudioAttentionMode,
    #[cfg(feature = "metal-paged-attn")]
    pub(super) fused_fc1_gelu: bool,
    pub(super) seq_bucket: Option<usize>,
    pub(super) debug_layers: bool,
}
impl AudioOperatorPolicy {
    pub(crate) fn resolve(
        env: &OperatorEnvironment,
        cfg: &AudioEncoderConfig,
        is_metal: bool,
        validated: bool,
    ) -> Result<Self> {
        let mut conv = ConvExecutionPolicy::default();
        if is_metal {
            conv.auto = validated;
            for (name, output) in [
                ("VASR_METAL_CONV_NHWC", &mut conv.nhwc_override),
                ("VASR_METAL_CONV_IM2COL32", &mut conv.im2col32_override),
                ("VASR_METAL_CONV_BIAS_GELU", &mut conv.bias_gelu_override),
                (
                    "VASR_METAL_CONV_NATIVE_CHUNKS",
                    &mut conv.native_chunks_override,
                ),
            ] {
                *output = ConvExecutionPolicy::parse_override(env.var(name).ok(), name)?;
            }
            conv.flatten = env
                .var("VASR_METAL_CONV_FLATTEN")
                .ok()
                .is_some_and(legacy_truthy);
            #[cfg(feature = "metal-paged-attn")]
            {
                conv.gemm_tile = env
                    .var("VASR_METAL_CONV_GEMM_TILE")
                    .ok()
                    .map(vasr_quant::metal_gemm::GemmTile::parse)
                    .transpose()?
                    .flatten();
                if validated && !env.present("VASR_METAL_CONV_GEMM_TILE") {
                    conv.gemm_tile = Some(vasr_quant::metal_gemm::GemmTile::Narrow);
                }
            }
        }
        let cross_sample = is_metal
            && match env.var("VASR_METAL_CONV_CROSS_SAMPLE") {
                Ok(value) => ConvExecutionPolicy::parse_override(
                    Some(value),
                    "VASR_METAL_CONV_CROSS_SAMPLE",
                )?
                .unwrap_or(false),
                Err(std::env::VarError::NotPresent) => false,
                Err(error) => candle_core::bail!("invalid VASR_METAL_CONV_CROSS_SAMPLE: {error}"),
            };
        let micro_batch = if is_metal {
            match env.var("VASR_METAL_CONV_CHUNKS") {
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
        let mut attention = AudioAttentionMode::resolve(env)?;
        let _fused_fc1_gelu = if cfg!(feature = "metal-paged-attn") && is_metal {
            let value = ConvExecutionPolicy::parse_override(
                env.var("VASR_METAL_ENCODER_BIAS_GELU").ok(),
                "VASR_METAL_ENCODER_BIAS_GELU",
            )?
            .unwrap_or(false);
            if validated && !env.present("VASR_QWEN3_ASR_AUDIO_ATTN") {
                attention = AudioAttentionMode::GroupedNative;
            }
            value || (validated && !env.present("VASR_METAL_ENCODER_BIAS_GELU"))
        } else {
            false
        };
        let seq_bucket = env
            .var("VASR_QWEN3_ASR_AUDIO_SEQ_BUCKET")
            .ok()
            .and_then(|value| {
                if value == "0" || value.eq_ignore_ascii_case("off") {
                    None
                } else {
                    value.parse::<usize>().ok().filter(|&n| n > 1)
                }
            });
        Ok(Self {
            conv,
            cross_sample,
            micro_batch,
            attention,
            #[cfg(feature = "metal-paged-attn")]
            fused_fc1_gelu: _fused_fc1_gelu,
            seq_bucket,
            debug_layers: env.present("VASR_DEBUG_AUDIO_ENCODER_LAYERS"),
        })
    }
}

/// This override historically warns and uses the caller's choice on invalid values.
/// It may disable flash attention, but never enables it beyond the load option.
pub(crate) fn audio_flash_attention(env: &OperatorEnvironment, requested: bool) -> bool {
    match env.var("VASR_QWEN3_ASR_AUDIO_FLASH_ATTN") {
        Ok("0" | "false" | "False" | "FALSE" | "off" | "OFF") => false,
        Ok("1" | "true" | "True" | "TRUE" | "on" | "ON") => requested,
        Ok(other) => {
            tracing::warn!(target: "vasr::models::qwen3asr", "ignoring invalid VASR_QWEN3_ASR_AUDIO_FLASH_ATTN={other:?}; expected 0/1");
            requested
        }
        Err(_) => requested,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn audio_flash_override_preserves_warning_fallback_and_never_widens() {
        for value in ["true", "FALSE", " Off ", "invalid", ""] {
            let env = OperatorEnvironment::from_values(&[(
                "VASR_QWEN3_ASR_AUDIO_FLASH_ATTN",
                value.into(),
            )]);
            assert!(!audio_flash_attention(&env, false));
            assert_eq!(audio_flash_attention(&env, true), value != "FALSE");
        }
    }
    #[test]
    fn audio_snapshots_preserve_ranges_presence_and_instance_independence() -> Result<()> {
        let cfg = AudioEncoderConfig::default();
        let first =
            AudioOperatorPolicy::resolve(&OperatorEnvironment::default(), &cfg, true, true)?;
        assert_eq!(first.conv.resolve(), (true, true));
        let off = OperatorEnvironment::from_values(&[
            ("VASR_METAL_CONV_NHWC", "false".into()),
            ("VASR_METAL_CONV_CROSS_SAMPLE", "0".into()),
            ("VASR_DEBUG_AUDIO_ENCODER_LAYERS", "false".into()),
            ("VASR_QWEN3_ASR_AUDIO_SEQ_BUCKET", "16".into()),
        ]);
        let off = AudioOperatorPolicy::resolve(&off, &cfg, true, true)?;
        assert_eq!(off.conv.resolve(), (false, false));
        assert!(!off.cross_sample);
        assert!(off.debug_layers);
        assert_eq!(off.seq_bucket, Some(16));
        assert_eq!(first.conv.resolve(), (true, true));
        for value in ["0", "65", "invalid", " 8 "] {
            let env = OperatorEnvironment::from_values(&[("VASR_METAL_CONV_CHUNKS", value.into())]);
            assert!(AudioOperatorPolicy::resolve(&env, &cfg, true, false).is_err());
            assert!(AudioOperatorPolicy::resolve(&env, &cfg, false, false).is_ok());
        }
        for value in ["REFERENCE", "invalid"] {
            let env =
                OperatorEnvironment::from_values(&[("VASR_QWEN3_ASR_AUDIO_ATTN", value.into())]);
            assert!(AudioOperatorPolicy::resolve(&env, &cfg, false, false).is_err());
        }
        Ok(())
    }
    #[cfg(unix)]
    #[test]
    fn non_unicode_conv_override_keeps_legacy_fallback_while_required_values_error() -> Result<()> {
        use std::os::unix::ffi::OsStringExt;
        let invalid = std::ffi::OsString::from_vec(vec![0xff]);
        let cfg = AudioEncoderConfig::default();
        let conv = OperatorEnvironment::from_values(&[("VASR_METAL_CONV_NHWC", invalid.clone())]);
        assert_eq!(
            AudioOperatorPolicy::resolve(&conv, &cfg, true, true)?
                .conv
                .resolve(),
            (true, true)
        );
        let required =
            OperatorEnvironment::from_values(&[("VASR_METAL_CONV_CROSS_SAMPLE", invalid)]);
        assert!(AudioOperatorPolicy::resolve(&required, &cfg, true, false).is_err());
        Ok(())
    }
}
