//! Test-only Metal 4 matrix candidate; not reachable from model inference.
use candle_core::{DType, Result, Tensor};

fn matmul_nt(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    matmul_nt_with_tile(input, weight, 32)
}

fn matmul_nt_with_tile(input: &Tensor, weight: &Tensor, tile_columns: usize) -> Result<Tensor> {
    matmul_nt_with_options(input, weight, tile_columns, false)
}

fn matmul_nt_with_options(
    input: &Tensor,
    weight: &Tensor,
    tile_columns: usize,
    relaxed_precision: bool,
) -> Result<Tensor> {
    if !matches!(tile_columns, 32 | 64) {
        candle_core::bail!("invalid MPP prototype tile");
    }
    let (out, columns) = weight.dims2()?;
    if !input.device().is_metal()
        || !input.device().same_device(weight.device())
        || input.rank() < 2
        || input.dims().last() != Some(&columns)
        || input.dtype() != weight.dtype()
        || !input.is_contiguous()
        || !weight.is_contiguous()
        || columns == 0
        || out == 0
        || input.elem_count() == 0
        || !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        candle_core::bail!(
            "MPP prototype requires nonempty contiguous Metal matrices with matching dtype"
        );
    }
    let rows = input.elem_count() / columns;
    if [
        rows,
        out,
        columns,
        rows.saturating_mul(out),
        input.elem_count(),
        weight.elem_count(),
    ]
    .iter()
    .any(|&value| value > i32::MAX as usize)
    {
        candle_core::bail!("MPP prototype matrix exceeds i32 indexing");
    }
    let mut shape = input.dims().to_vec();
    *shape.last_mut().unwrap() = out;
    input.apply_op2_no_bwd(
        weight,
        &MppNt {
            rows,
            out,
            columns,
            tile_columns,
            relaxed_precision,
            shape: shape.into(),
        },
    )
}

struct MppNt {
    rows: usize,
    out: usize,
    columns: usize,
    tile_columns: usize,
    relaxed_precision: bool,
    shape: candle_core::Shape,
}
impl candle_core::CustomOp2 for MppNt {
    fn name(&self) -> &'static str {
        "metal4-mpp-nt-prototype"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("MPP prototype requires Metal");
    }
    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        input_layout: &candle_core::Layout,
        weight: &candle_core::MetalStorage,
        weight_layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_metal_kernels::metal::ComputePipeline;
        use objc2_metal::{MTLCompileOptions, MTLDevice, MTLLanguageVersion, MTLSize};
        use std::{
            collections::HashMap,
            sync::{Mutex, OnceLock},
        };
        static CACHE: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> =
            OnceLock::new();
        let name = match (input.dtype(), self.tile_columns, self.relaxed_precision) {
            (DType::F32, 32, false) => "mpp_nt_f32_32",
            (DType::F32, 64, false) => "mpp_nt_f32_64",
            (DType::F16, 32, false) => "mpp_nt_f16_32",
            (DType::F16, 64, false) => "mpp_nt_f16_64",
            (DType::F16, 32, true) => "mpp_nt_f16_32_relaxed",
            (DType::F16, 64, true) => "mpp_nt_f16_64_relaxed",
            (DType::BF16, 32, false) => "mpp_nt_bf16_32",
            (DType::BF16, 64, false) => "mpp_nt_bf16_64",
            (DType::BF16, 32, true) => "mpp_nt_bf16_32_relaxed",
            (DType::BF16, 64, true) => "mpp_nt_bf16_64_relaxed",
            _ => candle_core::bail!("unsupported MPP dtype"),
        };
        let device = input.device();
        let key = (device.device().as_ref().registryID(), name);
        let pipeline = {
            let mut cache = CACHE
                .get_or_init(Mutex::default)
                .lock()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            if let Some(pipeline) = cache.get(&key) {
                pipeline.clone()
            } else {
                let options = MTLCompileOptions::new();
                options.setLanguageVersion(MTLLanguageVersion::Version4_0);
                let library = device
                    .device()
                    .new_library_with_source(include_str!("metal_mpp.metal"), Some(&options))
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let function = library
                    .get_function(name, None)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                let pipeline = device
                    .device()
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                cache.insert(key, pipeline.clone());
                pipeline
            }
        };
        let count = self.rows * self.out;
        let output = device.new_buffer(count, input.dtype(), "mpp-nt-prototype")?;
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
            Some(weight.buffer()),
            weight_layout.start_offset() * weight.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &[self.rows as u32, self.out as u32, self.columns as u32]);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: self.out.div_ceil(self.tile_columns),
                height: self.rows.div_ceil(64),
                depth: 1,
            },
            MTLSize {
                width: 128,
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
    use candle_core::Device;

    #[test]
    #[ignore = "exclusive Metal 4 MPP matrix microbenchmark"]
    fn diagnose_mpp_prefill_matrices() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for (rows, out, columns) in [
            (256, 3072, 1024),
            (640, 3072, 1024),
            (640, 1024, 3072),
            (640, 1024, 1024),
            (1000, 3072, 1024),
            (2000, 3072, 1024),
            (64, 3072, 1024),
        ] {
            let input = Tensor::from_vec(
                (0..rows * columns)
                    .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                    .collect::<Vec<_>>(),
                (rows, columns),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..out * columns)
                    .map(|i| ((i * 13 % 211) as f32 - 105.) / 501.)
                    .collect::<Vec<_>>(),
                (out, columns),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight_half = weight.to_dtype(DType::F16)?;
            let mut times: [Vec<f64>; 8] = std::array::from_fn(|_| Vec::new());
            for iteration in 0..60 {
                for offset in 0..8 {
                    let index = if iteration % 2 == 0 {
                        (iteration + offset) % 8
                    } else {
                        (iteration + 8 - offset) % 8
                    };
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = match index {
                        0 => crate::metal_gemm::matmul_nt(
                            &input,
                            &weight,
                            crate::metal_gemm::GemmTile::Adaptive,
                        )?,
                        1 => matmul_nt(&input, &weight)?,
                        2 => matmul_nt_with_tile(&input, &weight, 64)?,
                        3 => matmul_nt_with_options(&input, &weight, 32, true)?,
                        4 => matmul_nt_with_options(&input, &weight, 64, true)?,
                        5 => matmul_nt_with_options(
                            &input.to_dtype(DType::F16)?,
                            &weight_half,
                            32,
                            false,
                        )?
                        .to_dtype(DType::BF16)?,
                        6 => matmul_nt_with_options(
                            &input.to_dtype(DType::F16)?,
                            &weight_half,
                            32,
                            true,
                        )?
                        .to_dtype(DType::BF16)?,
                        _ => matmul_nt_with_options(
                            &input.to_dtype(DType::F16)?,
                            &weight_half,
                            64,
                            true,
                        )?
                        .to_dtype(DType::BF16)?,
                    };
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if iteration >= 12 {
                        times[index].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for (mode, times) in [
                "adaptive",
                "mpp64x32",
                "mpp64x64",
                "mpp64x32_relaxed",
                "mpp64x64_relaxed",
                "half_operands32_precise",
                "half_operands32_relaxed",
                "half_operands64_relaxed",
            ]
            .into_iter()
            .zip(&mut times)
            {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "mpp rows={rows} out={out} columns={columns} mode={mode} median_us={:.3} min_us={:.3} max_us={:.3}",
                    (times[23] + times[24]) / 2.,
                    times[0],
                    times[47]
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "exclusive Metal 4 MPP matrix numerical comparison"]
    fn mpp_nt_matches_dense_with_tails_and_offsets() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (rows, out, columns) in [(3, 37, 19), (65, 129, 128), (641, 3072, 1024)] {
                let x = Tensor::from_vec(
                    (0..(rows + 2) * columns)
                        .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                        .collect::<Vec<_>>(),
                    (rows + 2, columns),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, rows)?;
                let weight = Tensor::from_vec(
                    (0..(out + 1) * columns)
                        .map(|i| ((i * 13 % 211) as f32 - 105.) / 501.)
                        .collect::<Vec<_>>(),
                    (out + 1, columns),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, out)?;
                let reference = x.matmul(&weight.t()?)?;
                let reference = reference
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let variants = if matches!(dtype, DType::BF16 | DType::F16) {
                    vec![(32, false), (64, false), (32, true), (64, true)]
                } else {
                    vec![(32, false), (64, false)]
                };
                for (tile, relaxed) in variants {
                    let candidate = super::matmul_nt_with_options(&x, &weight, tile, relaxed)?;
                    assert_eq!(candidate.dims(), [rows, out]);
                    let candidate = candidate
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert!(reference.iter().chain(&candidate).all(|v| v.is_finite()));
                    let max_error = reference
                        .iter()
                        .zip(&candidate)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let tolerance = match dtype {
                        DType::F32 => 1e-4,
                        DType::F16 => 0.015,
                        _ => 0.125,
                    };
                    assert!(
                        max_error <= tolerance,
                        "dtype={dtype:?} rows={rows} out={out} columns={columns} tile={tile} relaxed={relaxed} error={max_error}"
                    );
                    eprintln!(
                        "mpp dtype={dtype:?} rows={rows} out={out} columns={columns} tile={tile} relaxed={relaxed} max_error={max_error}"
                    );
                }
            }
        }
        Ok(())
    }
}
