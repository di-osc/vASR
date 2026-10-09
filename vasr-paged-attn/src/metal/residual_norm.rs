//! Residual addition and RMSNorm with the materialized dtype boundaries preserved.

use candle_core::{DType, Layout, Result, Shape, Tensor};

/// Return `None` when the ordinary Candle add/RMSNorm path is required.
/// The first view is the rounded residual sum, the second its normalized value.
pub fn try_add_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<(Tensor, Tensor)>> {
    let count = input.elem_count();
    if !input.device().is_metal()
        || input.dims().last() != Some(&1024)
        || input.dims() != residual.dims()
        || weight.dims() != [1024]
        || input.dtype() != residual.dtype()
        || input.dtype() != weight.dtype()
        || !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16)
        || !input.device().same_device(residual.device())
        || !input.device().same_device(weight.device())
        || !input.is_contiguous()
        || !residual.is_contiguous()
        || !weight.is_contiguous()
        || count == 0
        || count > u32::MAX as usize / 2
    {
        return Ok(None);
    }
    let packed = input.apply_op3_no_bwd(residual, weight, &AddRmsNorm { eps })?;
    // Candle 5404348's F32 quantized matrix consumer assumes input offset zero.
    // Keep the normalized activation first; the ordinary residual add accepts offsets.
    let normalized = packed.narrow(0, 0, count)?.reshape(input.shape())?;
    let sum = packed.narrow(0, count, count)?.reshape(input.shape())?;
    Ok(Some((sum, normalized)))
}

struct AddRmsNorm {
    eps: f32,
}

impl candle_core::CustomOp3 for AddRmsNorm {
    fn name(&self) -> &'static str {
        "residual-add-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("residual add/RMSNorm requires Metal")
    }

    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        input_layout: &Layout,
        residual: &candle_core::MetalStorage,
        residual_layout: &Layout,
        weight: &candle_core::MetalStorage,
        weight_layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use super::compile::{KernelSource, load_pipeline};
        use candle_core::backend::BackendStorage;
        use objc2_metal::MTLSize;
        let name = match input.dtype() {
            DType::F32 => "residual_rmsnorm_f32",
            DType::F16 => "residual_rmsnorm_f16",
            DType::BF16 => "residual_rmsnorm_bf16",
            dtype => candle_core::bail!("unsupported residual RMSNorm dtype {dtype:?}"),
        };
        let device = input.device();
        let pipeline = load_pipeline(device.device(), KernelSource::ResidualNorm, name, None)?;
        let count = input_layout.shape().elem_count();
        let output = device.new_buffer(count * 2, input.dtype(), "residual-add-rmsnorm")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(input.buffer()),
            input_layout.start_offset() * input.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(residual.buffer()),
            residual_layout.start_offset() * residual.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            2,
            Some(weight.buffer()),
            weight_layout.start_offset() * weight.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(count as u32));
        encoder.set_bytes(5, &self.eps);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: count / 1024,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 512,
                height: 1,
                depth: 1,
            },
        );
        drop(guard);
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), count * 2, input.dtype()),
            Shape::from(vec![count * 2]),
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Module, RmsNorm};

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn residual_norm_matches_materialized_with_offsets() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for rows in [1usize, 8, 33, 64, 513] {
                let width = 1024;
                let x = Tensor::from_vec(
                    (0..(rows + 2) * width)
                        .map(|i| ((i * 17 % 1009) as f32 - 504.) / 23.)
                        .collect::<Vec<_>>(),
                    (1, rows + 2, width),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(1, 1, rows)?;
                let residual = x.affine(-0.73, 0.17)?;
                let weight = Tensor::from_vec(
                    (0..width + 1)
                        .map(|i| 0.5 + (i % 73) as f32 / 71.)
                        .collect::<Vec<_>>(),
                    width + 1,
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, width)?;
                let sum = (&x + &residual)?;
                let norm = RmsNorm::new(weight.clone(), 1e-6).forward(&sum)?;
                let (actual_sum, actual_norm) =
                    super::try_add_rms_norm(&x, &residual, &weight, 1e-6)?
                        .expect("supported layout must use the fused kernel");
                for (got, expected) in [(actual_sum, sum), (actual_norm, norm)] {
                    assert_eq!(got.dims(), expected.dims());
                    let a = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let b = expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let diff = a.iter().zip(&b).filter(|(a, b)| a != b).count();
                    let max_error = a
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert_eq!(diff, 0, "dtype={dtype:?} rows={rows} max_error={max_error}");
                    assert!(a.iter().all(|v| v.is_finite()));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn unsupported_residual_norm_keeps_reference_path() -> anyhow::Result<()> {
        let x = Tensor::zeros((2, 1024), DType::F32, &Device::Cpu)?;
        let weight = Tensor::ones(1024, DType::F32, &Device::Cpu)?;
        assert!(super::try_add_rms_norm(&x, &x, &weight, 1e-6)?.is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn residual_norm_layout_fallback_and_silence() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let x = Tensor::zeros((3, 1024), dtype, &device)?;
            let weight = Tensor::ones(1024, dtype, &device)?;
            let (sum, norm) = super::try_add_rms_norm(&x, &x, &weight, 1e-6)?.unwrap();
            for output in [sum, norm] {
                assert!(
                    output
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|v| *v == 0.)
                );
            }
            let strided = Tensor::zeros((3, 2048), dtype, &device)?.narrow(1, 0, 1024)?;
            assert!(super::try_add_rms_norm(&strided, &strided, &weight, 1e-6)?.is_none());
            let empty = x.narrow(0, 0, 0)?;
            assert!(super::try_add_rms_norm(&empty, &empty, &weight, 1e-6)?.is_none());
            let head = x.narrow(1, 0, 128)?.contiguous()?;
            assert!(
                super::try_add_rms_norm(&head, &head, &weight.narrow(0, 0, 128)?, 1e-6)?.is_none()
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn residual_norm_f32_quantized_consumer_matches_reference() -> anyhow::Result<()> {
        use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
        let device = Device::new_metal(0)?;
        let x = Tensor::from_vec(
            (0..8 * 1024)
                .map(|i| ((i * 17 % 1009) as f32 - 504.) / 23.)
                .collect::<Vec<_>>(),
            (1, 8, 1024),
            &device,
        )?;
        let residual = x.affine(-0.73, 0.17)?;
        let weight = Tensor::ones(1024, DType::F32, &device)?;
        let sum = (&x + &residual)?;
        let reference = RmsNorm::new(weight.clone(), 1e-6).forward(&sum)?;
        let (_, normalized) = super::try_add_rms_norm(&x, &residual, &weight, 1e-6)?.unwrap();
        let projection = Tensor::from_vec(
            (0..32 * 1024)
                .map(|i| ((i * 11 % 83) as f32 - 41.) / 71.)
                .collect::<Vec<_>>(),
            (32, 1024),
            &Device::Cpu,
        )?;
        let linear = QMatMul::from_qtensor(QTensor::quantize_onto(
            &projection,
            GgmlDType::Q8_0,
            &device,
        )?)?;
        assert_eq!(
            linear
                .forward(&normalized)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            linear
                .forward(&reference)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        assert_eq!(
            normalized.layout().start_offset(),
            0,
            "Candle quantized matrix consumer requires offset zero"
        );
        Ok(())
    }

    #[test]
    #[ignore = "synchronized Metal microbenchmark; requires exclusive GPU"]
    fn diagnose_residual_norm() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for rows in [1usize, 8, 32, 64, 512, 2048] {
            let width = 1024;
            let input = Tensor::from_vec(
                (0..rows * width)
                    .map(|i| ((i * 17 % 1009) as f32 - 504.) / 23.)
                    .collect::<Vec<_>>(),
                (rows, width),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let residual = input.affine(-0.73, 0.17)?;
            let weight = Tensor::ones(width, DType::BF16, &device)?;
            let norm = RmsNorm::new(weight.clone(), 1e-6);
            let mut times = [Vec::new(), Vec::new()];
            for trial in 0..36 {
                // Interleave and reverse order to avoid phased GPU timing bias.
                for phase in [trial % 2, (trial + 1) % 2] {
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = if phase == 0 {
                        let sum = (&input + &residual)?;
                        let normalized = norm.forward(&sum)?;
                        (sum, normalized)
                    } else {
                        super::try_add_rms_norm(&input, &residual, &weight, 1e-6)?.unwrap()
                    };
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if trial >= 4 {
                        times[phase].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for (phase, values) in times.iter_mut().enumerate() {
                values.sort_by(f64::total_cmp);
                eprintln!(
                    "residual_norm rows={rows} phase={} median_us={:.3} min_us={:.3} max_us={:.3}",
                    ["materialized", "fused"][phase],
                    (values[15] + values[16]) / 2.,
                    values[0],
                    values[31]
                );
            }
        }
        Ok(())
    }
}
