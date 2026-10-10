//! Resolve text operator overrides once before allocating decoder layers.
use crate::model::operator_policy::{
    OperatorEnvironment, legacy_truthy, parse_normalized_override, parse_strict_bool,
};
use candle_core::Result;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TextOperatorPolicy {
    pub dense_fusion: bool,
    pub prefill_gate_up: bool,
    pub fused_silu_mul: bool,
    pub prefill_fused_silu_mul: bool,
    pub fused_residual_norm: bool,
    pub direct_paged_attention: bool,
    pub prepared_mrope: bool,
    pub packed_prefill_rope: bool,
    #[cfg(feature = "paged-attn")]
    pub prepared_override: Option<bool>,
    #[cfg(feature = "paged-attn")]
    pub packed_rope_override: Option<bool>,
}
impl TextOperatorPolicy {
    pub(crate) fn resolve(
        env: &OperatorEnvironment,
        metal_operators: bool,
        validated: bool,
    ) -> Result<Self> {
        // Dense fusion has historically been validated on every backend.
        let dense_fusion = match env.var("VASR_METAL_DENSE_PROJECTION_FUSION") {
            Ok(value) => parse_strict_bool(Some(value), "VASR_METAL_DENSE_PROJECTION_FUSION")?,
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => candle_core::bail!("invalid VASR_METAL_DENSE_PROJECTION_FUSION: {error}"),
        };
        let (prepared_override, packed_rope_override) = if metal_operators {
            (
                read_rope_override(
                    env,
                    "VASR_METAL_PREPARED_MROPE",
                    parse_prepared_mrope_policy,
                )?,
                read_rope_override(
                    env,
                    "VASR_METAL_PREFILL_ROPE_KERNEL",
                    parse_packed_rope_policy,
                )?,
            )
        } else {
            (Some(false), Some(false))
        };
        let (prepared_mrope, packed_prefill_rope) =
            resolve_rope_defaults(prepared_override, packed_rope_override, validated);
        let prefill_gate_up = metal_operators
            && parse_prefill_gate_up_policy(env.var("VASR_METAL_PREFILL_GATE_UP").ok())?;
        let fused_silu_mul =
            metal_operators && required_normalized_bool(env, "VASR_METAL_SILU_MUL")?;
        let fused_residual_norm =
            metal_operators && required_normalized_bool(env, "VASR_METAL_RESIDUAL_RMSNORM")?;
        Ok(Self {
            dense_fusion,
            prefill_gate_up,
            fused_silu_mul,
            fused_residual_norm,
            prefill_fused_silu_mul: if env.present("VASR_METAL_SILU_MUL") {
                fused_silu_mul
            } else {
                true
            },
            // Unknown Unicode values historically disable this experimental switch.
            direct_paged_attention: env
                .var("VASR_METAL_PACKED_PAGED_ATTN")
                .ok()
                .map(legacy_truthy)
                .unwrap_or(validated),
            prepared_mrope,
            packed_prefill_rope,
            #[cfg(feature = "paged-attn")]
            prepared_override,
            #[cfg(feature = "paged-attn")]
            packed_rope_override,
        })
    }
}
fn required_normalized_bool(env: &OperatorEnvironment, name: &str) -> Result<bool> {
    match env.var(name) {
        Ok(value) => Ok(parse_normalized_override(Some(value), name, false)?.unwrap_or(false)),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => candle_core::bail!("invalid {name}: {error}"),
    }
}
pub(super) fn parse_prefill_gate_up_policy(value: Option<&str>) -> Result<bool> {
    Ok(parse_normalized_override(value, "VASR_METAL_PREFILL_GATE_UP", false)?.unwrap_or(false))
}
pub(super) fn parse_prepared_mrope_policy(value: Option<&str>) -> Result<bool> {
    Ok(parse_normalized_override(value, "VASR_METAL_PREPARED_MROPE", false)?.unwrap_or(false))
}
pub(super) fn parse_packed_rope_policy(value: Option<&str>) -> Result<bool> {
    Ok(parse_normalized_override(value, "VASR_METAL_PREFILL_ROPE_KERNEL", false)?.unwrap_or(false))
}
pub(super) fn resolve_rope_defaults(
    prepared: Option<bool>,
    packed: Option<bool>,
    validated: bool,
) -> (bool, bool) {
    let packed = packed.unwrap_or(validated);
    let prepared = prepared.unwrap_or(packed);
    (prepared, packed && prepared)
}
fn read_rope_override(
    env: &OperatorEnvironment,
    name: &str,
    parse: fn(Option<&str>) -> Result<bool>,
) -> Result<Option<bool>> {
    match env.var(name) {
        Ok(value) => Ok(Some(parse(Some(value))?)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => candle_core::bail!("invalid {name}; expected 0 or 1"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_defaults_explicit_off_and_multiple_instances_remain_independent() -> Result<()> {
        let automatic = TextOperatorPolicy::resolve(&OperatorEnvironment::default(), true, true)?;
        assert!(
            automatic.direct_paged_attention
                && automatic.prepared_mrope
                && automatic.packed_prefill_rope
        );
        let off_env = OperatorEnvironment::from_values(&[
            ("VASR_METAL_PACKED_PAGED_ATTN", "0".into()),
            ("VASR_METAL_PREPARED_MROPE", "off".into()),
        ]);
        let off = TextOperatorPolicy::resolve(&off_env, true, true)?;
        assert!(!off.direct_paged_attention && !off.prepared_mrope && !off.packed_prefill_rope);
        let other = TextOperatorPolicy::resolve(&OperatorEnvironment::default(), true, false)?;
        assert!(
            !other.direct_paged_attention && !other.prepared_mrope && !other.packed_prefill_rope
        );
        assert!(automatic.direct_paged_attention);
        Ok(())
    }
    #[test]
    fn activation_presence_and_invalid_inputs_keep_legacy_behavior() -> Result<()> {
        let unset = TextOperatorPolicy::resolve(&OperatorEnvironment::default(), true, false)?;
        assert!(!unset.fused_silu_mul && unset.prefill_fused_silu_mul);
        let off = OperatorEnvironment::from_values(&[("VASR_METAL_SILU_MUL", "false".into())]);
        let off = TextOperatorPolicy::resolve(&off, true, false)?;
        assert!(!off.fused_silu_mul && !off.prefill_fused_silu_mul);
        let invalid =
            OperatorEnvironment::from_values(&[("VASR_METAL_SILU_MUL", "invalid".into())]);
        assert!(TextOperatorPolicy::resolve(&invalid, true, false).is_err());
        assert!(TextOperatorPolicy::resolve(&invalid, false, false).is_ok());
        let strict = OperatorEnvironment::from_values(&[(
            "VASR_METAL_DENSE_PROJECTION_FUSION",
            "TRUE".into(),
        )]);
        assert!(TextOperatorPolicy::resolve(&strict, false, false).is_err());
        let lenient =
            OperatorEnvironment::from_values(&[("VASR_METAL_PACKED_PAGED_ATTN", "invalid".into())]);
        assert!(!TextOperatorPolicy::resolve(&lenient, true, true)?.direct_paged_attention);
        Ok(())
    }
    #[cfg(unix)]
    #[test]
    fn non_unicode_rope_errors_but_prefill_gate_up_falls_back() -> Result<()> {
        use std::os::unix::ffi::OsStringExt;
        let invalid = std::ffi::OsString::from_vec(vec![0xff]);
        let rope =
            OperatorEnvironment::from_values(&[("VASR_METAL_PREPARED_MROPE", invalid.clone())]);
        assert!(TextOperatorPolicy::resolve(&rope, true, false).is_err());
        let gate = OperatorEnvironment::from_values(&[("VASR_METAL_PREFILL_GATE_UP", invalid)]);
        assert!(!TextOperatorPolicy::resolve(&gate, true, false)?.prefill_gate_up);
        Ok(())
    }
}
