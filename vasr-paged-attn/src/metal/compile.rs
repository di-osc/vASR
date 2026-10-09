//! Shared compilation and cache identity for the rounding-preserving decoder kernels.

use candle_metal_kernels::metal::{ComputePipeline, ConstantValues, Device, Value};
use objc2::rc::Retained;
use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(super) enum KernelSource {
    Pointwise,
    ResidualNorm,
    Rotary,
}

impl KernelSource {
    // Each stable identity names exactly one embedded source in this binary.
    fn source(self) -> &'static str {
        match self {
            Self::Pointwise => include_str!("pointwise.metal"),
            Self::ResidualNorm => include_str!("residual_norm.metal"),
            Self::Rotary => include_str!("rotary.metal"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum MathOptions {
    Modern {
        mode: MTLMathMode,
        functions: MTLMathFloatingPointFunctions,
    },
    Legacy {
        fast_math: bool,
    },
}

impl MathOptions {
    fn for_os(fast_math: bool, modern_api: bool) -> Self {
        if modern_api {
            Self::Modern {
                mode: if fast_math {
                    MTLMathMode::Fast
                } else {
                    MTLMathMode::Relaxed
                },
                functions: if fast_math {
                    MTLMathFloatingPointFunctions::Fast
                } else {
                    MTLMathFloatingPointFunctions::Precise
                },
            }
        } else {
            Self::Legacy { fast_math }
        }
    }

    fn to_mtl(self) -> Retained<MTLCompileOptions> {
        // Match Candle 5404348 exactly, including the floating-point functions
        // used by exp() and the legacy API on older operating systems.
        let options = MTLCompileOptions::new();
        match self {
            Self::Modern { mode, functions } => {
                options.setMathMode(mode);
                options.setMathFloatingPointFunctions(functions);
            }
            Self::Legacy { fast_math } => {
                #[allow(deprecated)]
                options.setFastMathEnabled(fast_math);
            }
        }
        options
    }
}

fn fast_math_from_env(value: Option<&str>) -> bool {
    value
        .map(|value| matches!(value, "1" | "true" | "t" | "yes" | "y"))
        .unwrap_or(true)
}

// Candle 5404348 compares F32 constants as floats but hashes their bits.
// Use one bitwise identity for both equality and hashing, retaining type tags.
#[derive(Debug, Eq, PartialEq, Hash)]
enum ConstantValueKey {
    USize(usize),
    Bool(bool),
    F32(u32),
    U16(u16),
}

impl From<&Value> for ConstantValueKey {
    fn from(value: &Value) -> Self {
        match value {
            Value::USize(value) => Self::USize(*value),
            Value::Bool(value) => Self::Bool(*value),
            Value::F32(value) => Self::F32(value.to_bits()),
            Value::U16(value) => Self::U16(*value),
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash)]
struct PipelineKey {
    device_id: u64,
    source: KernelSource,
    function: &'static str,
    constants: Option<Vec<(usize, ConstantValueKey)>>,
    math: MathOptions,
}

impl PipelineKey {
    fn new(
        device_id: u64,
        source: KernelSource,
        function: &'static str,
        constants: Option<&[(usize, Value)]>,
        math: MathOptions,
    ) -> Self {
        Self {
            device_id,
            source,
            function,
            // Do not sort or deduplicate: Metal sets these in order, so the
            // last setter at a repeated index is significant.
            constants: constants.map(|values| {
                values
                    .iter()
                    .map(|(index, value)| (*index, ConstantValueKey::from(value)))
                    .collect()
            }),
            math,
        }
    }
}

pub(super) fn load_pipeline(
    device: &Device,
    source: KernelSource,
    function: &'static str,
    constants: Option<Vec<(usize, Value)>>,
) -> candle_core::Result<ComputePipeline> {
    let value = std::env::var("CANDLE_METAL_ENABLE_FAST_MATH").ok();
    let math = MathOptions::for_os(
        fast_math_from_env(value.as_deref()),
        objc2::available!(macos = 15, ios = 18),
    );
    load_pipeline_with_options(device, source, function, constants, math)
}

fn load_pipeline_with_options(
    device: &Device,
    source: KernelSource,
    function: &'static str,
    constants: Option<Vec<(usize, Value)>>,
    math: MathOptions,
) -> candle_core::Result<ComputePipeline> {
    static CACHE: OnceLock<Mutex<HashMap<PipelineKey, ComputePipeline>>> = OnceLock::new();
    let key = PipelineKey::new(
        device.registry_id(),
        source,
        function,
        constants.as_deref(),
        math,
    );
    let mut cache = CACHE
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    if let Some(pipeline) = cache.get(&key) {
        return Ok(pipeline.clone());
    }
    let options = math.to_mtl();
    let library = device
        .new_library_with_source(source.source(), Some(&options))
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    // Only the cache key is transformed. Pass the original types, float bits
    // and setter ordering to Candle/Metal unchanged.
    let constants = constants.map(ConstantValues::new);
    let function = library
        .get_function(key.function, constants.as_ref())
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
    cache.insert(key, pipeline.clone());
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn environment_values_keep_candle_fast_math_semantics() {
        assert!(fast_math_from_env(None));
        for value in ["1", "true", "t", "yes", "y"] {
            assert!(fast_math_from_env(Some(value)), "value={value:?}");
        }
        for value in ["", "0", "false", "no", "TRUE", "Yes", " true", "true ", "2"] {
            assert!(!fast_math_from_env(Some(value)), "value={value:?}");
        }
    }

    #[test]
    fn modern_precision_matches_candle_5404348() {
        assert_eq!(
            MathOptions::for_os(true, true),
            MathOptions::Modern {
                mode: MTLMathMode::Fast,
                functions: MTLMathFloatingPointFunctions::Fast,
            }
        );
        assert_eq!(
            MathOptions::for_os(false, true),
            MathOptions::Modern {
                mode: MTLMathMode::Relaxed,
                functions: MTLMathFloatingPointFunctions::Precise,
            }
        );
    }

    #[test]
    fn legacy_precision_matches_candle_5404348() {
        for fast_math in [false, true] {
            assert_eq!(
                MathOptions::for_os(fast_math, false),
                MathOptions::Legacy { fast_math }
            );
        }
    }

    #[test]
    fn modern_mtl_options_apply_both_precision_settings() {
        if !objc2::available!(macos = 15, ios = 18) {
            return;
        }
        for fast_math in [false, true] {
            let options = MathOptions::for_os(fast_math, true).to_mtl();
            assert_eq!(
                options.mathMode(),
                if fast_math {
                    MTLMathMode::Fast
                } else {
                    MTLMathMode::Relaxed
                }
            );
            assert_eq!(
                options.mathFloatingPointFunctions(),
                if fast_math {
                    MTLMathFloatingPointFunctions::Fast
                } else {
                    MTLMathFloatingPointFunctions::Precise
                }
            );
        }
    }

    #[test]
    fn legacy_mtl_options_apply_fast_math_setting() {
        for fast_math in [false, true] {
            let options = MathOptions::Legacy { fast_math }.to_mtl();
            #[allow(deprecated)]
            let actual = options.fastMathEnabled();
            assert_eq!(actual, fast_math);
        }
    }

    #[test]
    fn cache_key_separates_device_source_function_constants_and_precision() {
        let math = MathOptions::Modern {
            mode: MTLMathMode::Fast,
            functions: MTLMathFloatingPointFunctions::Fast,
        };
        let make_key = |device, source, function, constants: Option<Vec<(usize, Value)>>, math| {
            PipelineKey::new(device, source, function, constants.as_deref(), math)
        };
        let mut cache = HashMap::new();
        let base = make_key(7, KernelSource::Pointwise, "same_name", None, math);
        cache.insert(base, "base");
        assert_eq!(
            cache.get(&make_key(
                7,
                KernelSource::Pointwise,
                "same_name",
                None,
                math
            )),
            Some(&"base")
        );
        let variants = [
            make_key(8, KernelSource::Pointwise, "same_name", None, math),
            make_key(7, KernelSource::ResidualNorm, "same_name", None, math),
            make_key(7, KernelSource::Rotary, "same_name", None, math),
            make_key(7, KernelSource::Pointwise, "other_name", None, math),
            make_key(
                7,
                KernelSource::Pointwise,
                "same_name",
                Some(vec![(0, Value::Bool(true))]),
                math,
            ),
            make_key(
                7,
                KernelSource::Pointwise,
                "same_name",
                Some(vec![(0, Value::Bool(false))]),
                math,
            ),
            make_key(
                7,
                KernelSource::Pointwise,
                "same_name",
                None,
                MathOptions::Modern {
                    mode: MTLMathMode::Relaxed,
                    functions: MTLMathFloatingPointFunctions::Fast,
                },
            ),
            make_key(
                7,
                KernelSource::Pointwise,
                "same_name",
                None,
                MathOptions::Modern {
                    mode: MTLMathMode::Fast,
                    functions: MTLMathFloatingPointFunctions::Precise,
                },
            ),
            make_key(
                7,
                KernelSource::Pointwise,
                "same_name",
                None,
                MathOptions::Legacy { fast_math: true },
            ),
            make_key(
                7,
                KernelSource::Pointwise,
                "same_name",
                None,
                MathOptions::Legacy { fast_math: false },
            ),
        ];
        for key in variants {
            assert!(
                cache.get(&key).is_none(),
                "key aliases another pipeline: {key:?}"
            );
            cache.insert(key, "variant");
        }
        assert_eq!(cache.len(), 11);
    }

    fn key_for_constants(values: Vec<(usize, Value)>) -> PipelineKey {
        PipelineKey::new(
            7,
            KernelSource::Pointwise,
            "same_name",
            Some(&values),
            MathOptions::for_os(true, true),
        )
    }

    #[test]
    fn cache_reuses_identical_nan_constants() {
        let mut cache = HashMap::new();
        for bits in [0x7fc0_0000, 0x7fc0_0001, 0xffc0_0000] {
            let make_key = || key_for_constants(vec![(3, Value::F32(f32::from_bits(bits)))]);
            cache.insert(make_key(), bits);
            assert_eq!(cache.get(&make_key()), Some(&bits), "NaN bits={bits:#x}");
            cache.insert(make_key(), bits);
        }
        assert_eq!(
            cache.len(),
            3,
            "identical NaN payloads must reuse the cache key"
        );
    }

    #[test]
    fn cache_distinguishes_positive_and_negative_zero_constants() {
        let positive = || key_for_constants(vec![(3, Value::F32(0.0))]);
        let negative = || key_for_constants(vec![(3, Value::F32(-0.0))]);
        assert_ne!(
            positive(),
            negative(),
            "signed zeros have different Metal bit patterns"
        );
        let mut cache = HashMap::new();
        cache.insert(positive(), "positive");
        cache.insert(negative(), "negative");
        assert_eq!(cache.get(&positive()), Some(&"positive"));
        assert_eq!(cache.get(&negative()), Some(&"negative"));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn constant_key_preserves_setter_order_and_duplicate_indices() {
        let first = || key_for_constants(vec![(3, Value::U16(1)), (3, Value::U16(2))]);
        let reversed = key_for_constants(vec![(3, Value::U16(2)), (3, Value::U16(1))]);
        assert_eq!(first(), first());
        assert_ne!(
            first(),
            reversed,
            "last setter at a repeated index changes the value"
        );
        assert_ne!(first(), key_for_constants(vec![(3, Value::U16(2))]));
    }

    #[test]
    fn constant_key_preserves_value_types_and_indices() {
        let values = [
            Value::USize(1),
            Value::Bool(true),
            Value::F32(1.0),
            Value::U16(1),
        ];
        let mut cache = HashMap::new();
        for (i, value) in values.into_iter().enumerate() {
            assert!(
                cache
                    .insert(key_for_constants(vec![(3, value)]), i)
                    .is_none()
            );
        }
        assert!(
            cache
                .insert(key_for_constants(vec![(4, Value::U16(1))]), 4)
                .is_none()
        );
        assert_eq!(cache.len(), 5);
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn pipeline_cache_reuses_state_and_separates_precision() -> anyhow::Result<()> {
        let device = Device::system_default().expect("Metal device required");
        let modern_api = objc2::available!(macos = 15, ios = 18);
        let fast = MathOptions::for_os(true, modern_api);
        let precise = MathOptions::for_os(false, modern_api);
        let a = load_pipeline_with_options(
            &device,
            KernelSource::Pointwise,
            "silu_mul_f32",
            None,
            fast,
        )?;
        let b = load_pipeline_with_options(
            &device,
            KernelSource::Pointwise,
            "silu_mul_f32",
            None,
            fast,
        )?;
        let c = load_pipeline_with_options(
            &device,
            KernelSource::Pointwise,
            "silu_mul_f32",
            None,
            precise,
        )?;
        assert!(
            std::ptr::eq(a.as_ref(), b.as_ref()),
            "cache miss for identical compilation"
        );
        assert!(
            !std::ptr::eq(a.as_ref(), c.as_ref()),
            "different precision reused a pipeline"
        );
        Ok(())
    }
}
