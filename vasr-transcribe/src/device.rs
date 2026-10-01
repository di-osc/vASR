use anyhow::{Result, bail};
use candle_core::{DType, Device};

/// Resolve a device string to a candle Device.
pub fn resolve_device(value: &str) -> Result<Device> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => auto_device(),
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
                    .map_err(|err| anyhow::anyhow!("failed to create Metal device 0: {err}"))
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("metal requested but vasr was built without the metal feature")
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0)
                    .map_err(|err| anyhow::anyhow!("failed to create CUDA device 0: {err}"))
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!("cuda requested but vasr was built without the cuda feature")
            }
        }
        other => bail!("unknown device {other:?}; expected auto, cpu, metal, or cuda"),
    }
}

/// Auto-select the best available device.
pub fn auto_device() -> Result<Device> {
    #[allow(unused_mut)]
    let mut failures: Vec<String> = Vec::new();
    #[cfg(feature = "cuda")]
    {
        match Device::new_cuda(0) {
            Ok(device) => return Ok(device),
            Err(err) => failures.push(format!("CUDA: {err}")),
        }
    }
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => return Ok(device),
            Err(err) => failures.push(format!("Metal: {err}")),
        }
    }
    if failures.is_empty() {
        tracing::info!(target: "vasr_transcribe::device", "auto device selected CPU: no accelerator backend is enabled");
    } else {
        tracing::warn!(target: "vasr_transcribe::device", "accelerator initialization failed; falling back to CPU: {}", failures.join("; "));
    }
    Ok(Device::Cpu)
}

/// Parse the opt-in Metal ASR dtype without mutating process environment.
fn parse_metal_asr_dtype(value: Option<&str>) -> Result<DType> {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        None | Some("bf16") => Ok(DType::BF16),
        Some("f16") => Ok(DType::F16),
        Some("f32") => Ok(DType::F32),
        Some(other) => bail!("invalid VASR_METAL_ASR_DTYPE={other:?}; expected bf16, f16, or f32"),
    }
}

/// Auto-select dtype for the given device.
pub fn auto_dtype(device: &Device) -> Result<DType> {
    if device.is_metal() {
        let value = match std::env::var("VASR_METAL_ASR_DTYPE") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => bail!("invalid VASR_METAL_ASR_DTYPE: {error}"),
        };
        return parse_metal_asr_dtype(value.as_deref());
    }
    Ok(if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    })
}

/// Human-readable label for a device.
pub fn device_label(device: &Device) -> &'static str {
    if device.is_cpu() {
        "cpu"
    } else if device.is_metal() {
        "metal"
    } else if device.is_cuda() {
        "cuda"
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::{auto_dtype, parse_metal_asr_dtype};
    use candle_core::{DType, Device};

    #[test]
    fn test_parse_metal_asr_dtype_defaults_to_bf16() -> anyhow::Result<()> {
        assert_eq!(parse_metal_asr_dtype(None)?, DType::BF16);
        Ok(())
    }

    #[test]
    fn test_parse_metal_asr_dtype_accepts_normalized_names() -> anyhow::Result<()> {
        for (value, expected) in [
            ("bf16", DType::BF16),
            (" BF16 ", DType::BF16),
            ("f16", DType::F16),
            (" F16\n", DType::F16),
            ("f32", DType::F32),
            ("\tF32 ", DType::F32),
        ] {
            assert_eq!(parse_metal_asr_dtype(Some(value))?, expected);
        }
        Ok(())
    }

    #[test]
    fn test_parse_metal_asr_dtype_rejects_invalid_values() {
        for value in ["", " ", "f64", "b16", "auto", "invalid"] {
            let error = match parse_metal_asr_dtype(Some(value)) {
                Err(error) => error,
                Ok(dtype) => panic!("invalid dtype {value:?} unexpectedly parsed as {dtype:?}"),
            };
            assert!(error.to_string().contains("VASR_METAL_ASR_DTYPE"));
        }
    }

    #[test]
    fn test_auto_dtype_cpu_remains_f32() -> anyhow::Result<()> {
        assert_eq!(auto_dtype(&Device::Cpu)?, DType::F32);
        Ok(())
    }
}
