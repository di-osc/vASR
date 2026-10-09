//! Pointwise decoder candidates, compared against the materialized Candle path.

use candle_core::{DType, Result, Tensor};

/// Preserve the SiLU store/load rounding boundary before multiplying by `up`.
/// Both ordinary projection outputs and views of a joined gate/up GEMM are supported.
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.dims() != up.dims()
        || gate.dtype() != up.dtype()
        || !gate.device().same_device(up.device())
    {
        candle_core::bail!("SiLU/mul requires matching shapes, dtype and device");
    }
    let count = gate.elem_count();
    if gate.device().is_metal()
        && matches!(gate.dtype(), DType::F32 | DType::F16 | DType::BF16)
        && count > 0
        && count <= u32::MAX as usize
        && let (Some(gate_stride), Some(up_stride)) = (row_stride(gate), row_stride(up))
    {
        let width = *gate.dims().last().unwrap();
        return gate.apply_op2_no_bwd(
            up,
            &SiluMul {
                params: [count as u32, width as u32, gate_stride, up_stride],
            },
        );
    }
    gate.silu()?.mul(up)
}

fn row_stride(tensor: &Tensor) -> Option<u32> {
    let (&width, prefix) = tensor.dims().split_last()?;
    let (&column_stride, strides) = tensor.stride().split_last()?;
    if width == 0 || column_stride != 1 {
        return None;
    }
    let row_stride = strides.last().copied().unwrap_or(width);
    let mut expected = row_stride;
    for (&dim, &stride) in prefix.iter().zip(strides).rev() {
        if dim > 1 && stride != expected {
            return None;
        }
        expected = expected.checked_mul(dim)?;
    }
    let max_offset = tensor
        .dims()
        .iter()
        .zip(tensor.stride())
        .try_fold(0usize, |sum, (&dim, &stride)| {
            sum.checked_add((dim - 1).checked_mul(stride)?)
        })?;
    if max_offset > u32::MAX as usize {
        return None;
    }
    u32::try_from(row_stride).ok()
}

struct SiluMul {
    params: [u32; 4],
}
impl candle_core::CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "silu-mul"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("SiLU/mul kernel requires Metal")
    }
    fn metal_fwd(
        &self,
        gate: &candle_core::MetalStorage,
        gate_layout: &candle_core::Layout,
        up: &candle_core::MetalStorage,
        up_layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use super::compile::{KernelSource, load_pipeline};
        use candle_core::backend::BackendStorage;
        use objc2_metal::MTLSize;
        let name = match gate.dtype() {
            DType::F32 => "silu_mul_f32",
            DType::F16 => "silu_mul_f16",
            DType::BF16 => "silu_mul_bf16",
            other => candle_core::bail!("SiLU/mul unsupported dtype {other:?}"),
        };
        let device = gate.device();
        let pipeline = load_pipeline(device.device(), KernelSource::Pointwise, name, None)?;
        let count = gate_layout.shape().elem_count();
        let output = device.new_buffer(count, gate.dtype(), "silu-mul")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(gate.buffer()),
            gate_layout.start_offset() * gate.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(up.buffer()),
            up_layout.start_offset() * up.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &self.params);
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
            candle_core::MetalStorage::new(output, device.clone(), count, gate.dtype()),
            gate_layout.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn silu_mul_matches_materialized_with_views_and_offsets() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (rows, width) in [(1usize, 13usize), (32, 3072), (137, 3072)] {
                let count = (rows + 2) * width * 2;
                let joined = Tensor::from_vec(
                    (0..count)
                        .map(|i| ((i * 17 % 1009) as f32 - 504.) / 23.)
                        .collect::<Vec<_>>(),
                    (rows + 2, width * 2),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, rows)?;
                let gate = joined.narrow(1, 0, width)?;
                let up = joined.narrow(1, width, width)?;
                // Match both separate contiguous projections and slices of one joined GEMM.
                for (gate, up) in [
                    (gate.clone(), up.clone()),
                    (gate.contiguous()?, up.contiguous()?),
                ] {
                    let expected = gate.contiguous()?.silu()?.mul(&up.contiguous()?)?;
                    let got = super::silu_mul(&gate, &up)?;
                    assert_eq!(got.dims(), expected.dims());
                    let a = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let b = expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let differing = a.iter().zip(&b).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differing, 0,
                        "SiLU rounding changed: dtype={dtype:?} rows={rows} width={width}"
                    );
                    assert!(a.iter().all(|v| v.is_finite()));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn silu_mul_cpu_fallback_and_shape_validation() -> anyhow::Result<()> {
        let gate = Tensor::from_vec(vec![-4f32, -1., 0., 1., 4., 9.], (2, 3), &Device::Cpu)?;
        let up = gate.affine(0.5, 1.)?;
        let got = super::silu_mul(&gate.transpose(0, 1)?, &up.transpose(0, 1)?)?;
        let expected = gate.transpose(0, 1)?.silu()?.mul(&up.transpose(0, 1)?)?;
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        assert!(super::silu_mul(&gate, &up.flatten_all()?).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "synchronized Metal microbenchmark; requires exclusive GPU"]
    fn diagnose_silu_mul() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for rows in [1usize, 8, 32, 512, 1024] {
            let width = 3072;
            let joined = Tensor::from_vec(
                (0..rows * width * 2)
                    .map(|i| ((i % 97) as f32 - 48.) / 23.)
                    .collect::<Vec<_>>(),
                (rows, width * 2),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let gate = joined.narrow(1, 0, width)?;
            let up = joined.narrow(1, width, width)?;
            let gate_copy = gate.contiguous()?;
            let up_copy = up.contiguous()?;
            for phase in ["separate", "fused", "joined_materialized", "joined_fused"] {
                let mut times = Vec::new();
                for i in 0..24 {
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = match phase {
                        "separate" => gate_copy.silu()?.mul(&up_copy)?,
                        "fused" => super::silu_mul(&gate_copy, &up_copy)?,
                        "joined_materialized" => {
                            gate.contiguous()?.silu()?.mul(&up.contiguous()?)?
                        }
                        _ => super::silu_mul(&gate, &up)?,
                    };
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if i >= 4 {
                        times.push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "silu_mul rows={rows} width={width} phase={phase} median_us={:.3} min_us={:.3} max_us={:.3}",
                    (times[9] + times[10]) / 2.,
                    times[0],
                    times[19]
                );
            }
        }
        Ok(())
    }
}
