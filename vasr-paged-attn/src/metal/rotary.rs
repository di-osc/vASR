//! RoPE with the materialized multiply/add rounding and packed output storage.
use candle_core::{DType, Result, Tensor};

/// Return a B,H,S,D view of B,S,H,D storage, or None for an unsupported backend.
/// Full-head cos/sin are B,1,S,D; no position or attention semantics live here.
pub fn try_packed_rope(input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Option<Tensor>> {
    let (batch, heads, seq, width) = input.dims4()?;
    if cos.dims() != [batch, 1, seq, width]
        || sin.dims() != cos.dims()
        || input.dtype() != cos.dtype()
        || input.dtype() != sin.dtype()
        || !input.device().same_device(cos.device())
        || !input.device().same_device(sin.device())
    {
        candle_core::bail!(
            "packed RoPE requires matching full-head position data, dtype and device"
        );
    }
    if !input.device().is_metal()
        || !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16)
        || width == 0
        || width % 2 != 0
        || input.elem_count() == 0
        || input.elem_count() > u32::MAX as usize
        || !cos.is_contiguous()
        || !sin.is_contiguous()
    {
        return Ok(None);
    }
    let max_offset = input
        .dims()
        .iter()
        .zip(input.stride())
        .try_fold(0usize, |sum, (&dim, &stride)| {
            sum.checked_add((dim - 1).checked_mul(stride)?)
        });
    if max_offset.is_none_or(|offset| offset > u32::MAX as usize) {
        return Ok(None);
    }
    let values = [
        input.elem_count(),
        seq,
        heads,
        width,
        input.stride()[0],
        input.stride()[1],
        input.stride()[2],
        input.stride()[3],
    ];
    let mut params = [0u32; 8];
    for (param, value) in params.iter_mut().zip(values) {
        let Ok(value) = u32::try_from(value) else {
            return Ok(None);
        };
        *param = value;
    }
    let output = input.apply_op3_no_bwd(
        cos,
        sin,
        &PackedRope {
            params,
            shape: (batch, seq, heads, width).into(),
        },
    )?;
    Ok(Some(output.transpose(1, 2)?))
}

struct PackedRope {
    params: [u32; 8],
    shape: candle_core::Shape,
}
impl candle_core::CustomOp3 for PackedRope {
    fn name(&self) -> &'static str {
        "packed-rope"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("packed RoPE kernel requires Metal")
    }
    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        input_layout: &candle_core::Layout,
        cos: &candle_core::MetalStorage,
        cos_layout: &candle_core::Layout,
        sin: &candle_core::MetalStorage,
        sin_layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_metal_kernels::metal::ComputePipeline;
        use objc2_metal::{
            MTLCompileOptions, MTLDevice, MTLMathFloatingPointFunctions, MTLMathMode, MTLSize,
        };
        use std::{
            collections::HashMap,
            sync::{Mutex, OnceLock},
        };
        static CACHE: OnceLock<Mutex<HashMap<(u64, &'static str, bool), ComputePipeline>>> =
            OnceLock::new();
        let name = match input.dtype() {
            DType::F32 => "packed_rope_f32",
            DType::F16 => "packed_rope_f16",
            DType::BF16 => "packed_rope_bf16",
            other => candle_core::bail!("packed RoPE unsupported dtype {other:?}"),
        };
        let device = input.device();
        let fast_math = std::env::var("CANDLE_METAL_ENABLE_FAST_MATH")
            .map(|value| matches!(value.as_str(), "1" | "true" | "t" | "yes" | "y"))
            .unwrap_or(true);
        let key = (device.device().as_ref().registryID(), name, fast_math);
        let pipeline = {
            let mut cache = CACHE
                .get_or_init(Mutex::default)
                .lock()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if let Some(pipeline) = cache.get(&key) {
                pipeline.clone()
            } else {
                let options = MTLCompileOptions::new();
                if objc2::available!(macos = 15, ios = 18) {
                    options.setMathMode(if fast_math {
                        MTLMathMode::Fast
                    } else {
                        MTLMathMode::Relaxed
                    });
                    options.setMathFloatingPointFunctions(if fast_math {
                        MTLMathFloatingPointFunctions::Fast
                    } else {
                        MTLMathFloatingPointFunctions::Precise
                    });
                } else {
                    #[allow(deprecated)]
                    options.setFastMathEnabled(fast_math);
                }
                let library = device
                    .device()
                    .new_library_with_source(include_str!("rotary.metal"), Some(&options))
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                let function = library
                    .get_function(name, None)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                let pipeline = device
                    .device()
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
                cache.insert(key, pipeline.clone());
                pipeline
            }
        };
        let count = input_layout.shape().elem_count();
        let output = device.new_buffer(count, input.dtype(), "packed-rope")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        for (index, (storage, layout)) in
            [(input, input_layout), (cos, cos_layout), (sin, sin_layout)]
                .into_iter()
                .enumerate()
        {
            encoder.set_input_buffer(
                index,
                Some(storage.buffer()),
                layout.start_offset() * storage.dtype().size_in_bytes(),
            );
        }
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &self.params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: count.div_ceil(4).div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        drop(guard);
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), count, input.dtype()),
            self.shape.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn packed_rope_tail_stride_and_position_offsets() -> anyhow::Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (batch, heads, seq, width) in [(2, 3, 7, 130), (1, 1, 1, 14), (1, 5, 67, 128)] {
                let input = Tensor::from_vec(
                    (0..(batch + 2) * (seq + 2) * (heads + 1) * width)
                        .map(|i| ((i * 17 % 1009) as f32 - 504.) / 23.)
                        .collect::<Vec<_>>(),
                    (batch + 2, seq + 2, heads + 1, width),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, batch)?
                .narrow(1, 1, seq)?
                .narrow(2, 1, heads)?
                .transpose(1, 2)?;
                let positions = |seed| -> Result<Tensor> {
                    Tensor::from_vec(
                        (0..(batch + 2) * seq * width)
                            .map(|i| ((i * seed % 193) as f32 - 96.) / 151.)
                            .collect::<Vec<_>>(),
                        (batch + 2, 1, seq, width),
                        &device,
                    )?
                    .to_dtype(dtype)?
                    .narrow(0, 1, batch)
                };
                let cos = positions(13)?;
                let sin = positions(29)?;
                let contiguous = input.contiguous()?;
                let first = contiguous.narrow(3, 0, width / 2)?;
                let second = contiguous.narrow(3, width / 2, width / 2)?;
                let rotated = Tensor::cat(&[&second.neg()?, &first], 3)?;
                let expected = (contiguous.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
                let got = try_packed_rope(&input, &cos, &sin)?.unwrap();
                assert!(got.transpose(1, 2)?.is_contiguous());
                let a = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let b = expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(a.iter().chain(&b).all(|value| value.is_finite()));
                let differing = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(differing, 0, "dtype={dtype:?} shape={:?}", input.dims());
                eprintln!(
                    "packed rope dtype={dtype:?} batch={batch} heads={heads} seq={seq} width={width} offset=true bitwise=true"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn packed_rope_cpu_fallback_and_shape_validation() -> anyhow::Result<()> {
        let input = Tensor::ones((2, 4, 7, 128), DType::F32, &candle_core::Device::Cpu)?;
        let positions = Tensor::ones((2, 1, 7, 128), DType::F32, &candle_core::Device::Cpu)?;
        assert!(try_packed_rope(&input, &positions, &positions)?.is_none());
        assert!(try_packed_rope(&input, &positions.narrow(2, 0, 1)?, &positions).is_err());
        Ok(())
    }
}
