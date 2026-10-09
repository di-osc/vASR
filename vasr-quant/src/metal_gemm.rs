//! Decode GEMM tile candidates using the pinned Candle MLX kernels.

use candle_core::{DType, Result, Tensor};
use std::cell::Cell;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmTile {
    Small,
    SmallSwizzled,
    Adaptive,
    Balanced,
    Wide,
    Square4,
    Narrow,
}
impl GemmTile {
    pub const ALL: [Self; 7] = [
        Self::Small,
        Self::SmallSwizzled,
        Self::Adaptive,
        Self::Balanced,
        Self::Wide,
        Self::Square4,
        Self::Narrow,
    ];
    pub fn parse(value: &str) -> Result<Option<Self>> {
        match value.trim().to_ascii_lowercase().as_str() {
            "0" | "off" => Ok(None),
            "32x32" => Ok(Some(Self::Small)),
            "32x32-swizzle" => Ok(Some(Self::SmallSwizzled)),
            "adaptive" => Ok(Some(Self::Adaptive)),
            "32x64" => Ok(Some(Self::Balanced)),
            "64x64" => Ok(Some(Self::Wide)),
            "64x64-simd4" => Ok(Some(Self::Square4)),
            "64x32" => Ok(Some(Self::Narrow)),
            _ => candle_core::bail!(
                "Metal GEMM tile must be 0, 32x32, 32x32-swizzle, adaptive, 32x64, 64x64, 64x64-simd4 or 64x32"
            ),
        }
    }
    fn dimensions(self) -> (usize, usize, usize, usize, usize) {
        match self {
            Self::Small | Self::SmallSwizzled | Self::Adaptive => (32, 32, 16, 2, 2),
            Self::Balanced => (32, 64, 16, 1, 2),
            Self::Wide => (64, 64, 16, 1, 2),
            Self::Square4 => (64, 64, 16, 2, 2),
            Self::Narrow => (64, 32, 32, 2, 2),
        }
    }
}

#[cfg(test)]
thread_local! { static CALLS: Cell<usize> = const { Cell::new(0) };
static LAST_TILE: Cell<Option<GemmTile>> = const { Cell::new(None) }; }

thread_local! { static TILE: Cell<(Option<GemmTile>, bool)> = const { Cell::new((None, false)) }; }
pub struct GemmGuard {
    previous: (Option<GemmTile>, bool),
    // The guard must restore the same thread-local slot that it changed.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for GemmGuard {
    fn drop(&mut self) {
        TILE.with(|tile| tile.set(self.previous));
    }
}
pub fn set_tile(tile: Option<GemmTile>) -> GemmGuard {
    GemmGuard {
        previous: TILE.with(|state| state.replace((tile, false))),
        _thread: std::marker::PhantomData,
    }
}
/// Select an independent dense prefill policy for this forward, including
/// last-token logits. Quantized layers and other devices do not use it.
pub fn set_prefill_tile(tile: Option<GemmTile>) -> GemmGuard {
    GemmGuard {
        previous: TILE.with(|state| state.replace((tile, true))),
        _thread: std::marker::PhantomData,
    }
}
#[cfg(test)]
pub(crate) fn current_tile() -> Option<GemmTile> {
    TILE.with(|state| state.get().0)
}
pub(crate) fn tile_for_linear(rows: usize, is_prefill: bool) -> Option<GemmTile> {
    let (tile, prefill_scope) = TILE.with(Cell::get);
    tile.filter(|_| rows >= 2 && (prefill_scope || (!is_prefill && rows <= 128)))
}

pub fn matmul_nt(input: &Tensor, weight: &Tensor, tile: GemmTile) -> Result<Tensor> {
    let (out, columns) = weight.dims2()?;
    if input.rank() < 2
        || input.dims().last() != Some(&columns)
        || input.dtype() != weight.dtype()
        || !input.device().same_device(weight.device())
    {
        candle_core::bail!(
            "tiled GEMM requires matching dense input/weight columns, dtype and device"
        );
    }
    let rows = input
        .elem_count()
        .checked_div(columns)
        .filter(|_| columns > 0)
        .ok_or_else(|| candle_core::Error::Msg("tiled GEMM requires nonzero columns".into()))?;
    let tile = if tile == GemmTile::Adaptive {
        if rows <= 32 {
            GemmTile::Small
        } else {
            GemmTile::Narrow
        }
    } else {
        tile
    };
    let mut shape = input.dims().to_vec();
    *shape.last_mut().unwrap() = out;
    if !input.device().is_metal()
        || rows <= 1
        || out == 0
        || !input.is_contiguous()
        || !weight.is_contiguous()
        || !matches!(input.dtype(), DType::F32 | DType::F16 | DType::BF16)
        || [
            rows,
            out,
            columns,
            input.elem_count(),
            weight.elem_count(),
            rows.saturating_mul(out),
        ]
        .iter()
        .any(|&v| v > i32::MAX as usize)
    {
        return input
            .contiguous()?
            .reshape((rows, columns))?
            .matmul(&weight.contiguous()?.t()?)?
            .reshape(shape);
    }
    #[cfg(test)]
    {
        CALLS.with(|count| count.set(count.get() + 1));
        LAST_TILE.with(|last| last.set(Some(tile)));
    }
    tracing::trace!(target: "vasr_quant::metal_gemm", rows, out, columns, ?tile,
        prefill_scope=TILE.with(|state| state.get().1), "tiled dense GEMM dispatch");
    input.apply_op2_no_bwd(
        weight,
        &TiledNt {
            rows,
            out,
            columns,
            tile,
            shape: shape.into(),
        },
    )
}

struct TiledNt {
    rows: usize,
    out: usize,
    columns: usize,
    tile: GemmTile,
    shape: candle_core::Shape,
}
impl candle_core::CustomOp2 for TiledNt {
    fn name(&self) -> &'static str {
        "tiled-dense-nt"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("tiled GEMM kernel requires Metal")
    }
    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        input_layout: &candle_core::Layout,
        weight: &candle_core::MetalStorage,
        weight_layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_metal_kernels::{
            metal::{ConstantValues, Value},
            source::Source,
        };
        use objc2_metal::MTLSize;
        let (bm, bn, bk, wm, wn) = self.tile.dimensions();
        let dtype = match input.dtype() {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            other => candle_core::bail!("unsupported tiled GEMM dtype {other:?}"),
        };
        let device = input.device();
        let constants = ConstantValues::new(vec![
            (10, Value::Bool(false)),
            (100, Value::Bool(false)),
            (110, Value::Bool(false)),
            (200, Value::Bool(self.rows % bm == 0)),
            (201, Value::Bool(self.out % bn == 0)),
            (202, Value::Bool(self.columns % bk == 0)),
            (300, Value::Bool(false)),
        ]);
        let pipeline = device
            .kernels()
            .load_pipeline_with_constants(
                device.device(),
                Source::Gemm,
                format!("gemm_nt_{dtype}_{dtype}_{bm}_{bn}_{bk}_{wm}_{wn}"),
                Some(constants),
            )
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let count = self.rows * self.out;
        let output = device.new_buffer(count, input.dtype(), "tiled-dense-nt")?;
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
        encoder.set_output_buffer(3, Some(&output), 0);
        let (tn, tm) = (self.out.div_ceil(bn), self.rows.div_ceil(bm));
        let swizzle_log = u32::from(self.tile == GemmTile::SmallSwizzled);
        let swizzle = 1usize << swizzle_log;
        // Exact Candle 5404348 GemmParams ABI (72 bytes). Three i64 batch strides
        // are zero for one flattened matrix. Tile counts stay logical; only
        // the dispatch grid is swizzled, so odd M tile counts are bounded.
        // The final word is alignment padding.
        let params = [
            self.rows as i32,
            self.out as i32,
            self.columns as i32,
            self.columns as i32,
            self.columns as i32,
            self.out as i32,
            tn as i32,
            tm as i32,
            0,
            0,
            0,
            0,
            0,
            0,
            swizzle_log as i32,
            (self.columns / bk) as i32,
            1,
            0,
        ];
        encoder.set_bytes(4, &params);
        encoder.set_bytes(6, &[1i32]);
        encoder.set_bytes(7, &[0i64, 0i64]);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: tn * swizzle,
                height: tm.div_ceil(swizzle),
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: wn,
                depth: wm,
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
    use super::{CALLS, LAST_TILE};
    use candle_core::{DType, Device, Tensor};
    use std::cell::Cell;

    #[test]
    #[ignore = "exclusive Metal tile microbenchmark"]
    fn diagnose_decode_gemm_tiles() -> anyhow::Result<()> {
        diagnose_gemm_shapes(&[
            (8usize, 1024usize, 1024usize),
            (16, 3072, 1024),
            (32, 1024, 1024),
            (32, 3072, 1024),
            (32, 1024, 3072),
            (64, 3072, 1024),
            (3200, 480, 4320),
            (64, 151936, 1024),
        ])
    }

    #[test]
    #[ignore = "exclusive Metal encoder tile microbenchmark"]
    fn diagnose_encoder_gemm_tiles() -> anyhow::Result<()> {
        diagnose_gemm_shapes(&[
            (128, 3584, 896),
            (1000, 3584, 896),
            (1000, 896, 3584),
            (1000, 896, 896),
            (128, 151936, 1024),
        ])
    }

    #[test]
    #[ignore = "exclusive Metal prefill tile microbenchmark"]
    fn diagnose_prefill_gemm_tiles() -> anyhow::Result<()> {
        diagnose_gemm_shapes(&[
            (1000, 3072, 1024),
            (1000, 1024, 3072),
            (1000, 1024, 1024),
            (2000, 3072, 1024),
        ])
    }

    #[test]
    #[ignore = "exclusive Metal prefill weight-layout microbenchmark"]
    fn diagnose_prefill_weight_layouts() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for (rows, out, input) in [
            (640, 3072, 1024),
            (640, 1024, 3072),
            (640, 1024, 1024),
            (1000, 3072, 1024),
            (2000, 3072, 1024),
            (64, 3072, 1024),
            (64, 1024, 3072),
        ] {
            let x = Tensor::from_vec(
                (0..rows * input)
                    .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                    .collect::<Vec<_>>(),
                (rows, input),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let weight = Tensor::from_vec(
                (0..out * input)
                    .map(|i| ((i * 13 % 211) as f32 - 105.) / 501.)
                    .collect::<Vec<_>>(),
                (out, input),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            // Transpose only once outside the timed loop. This tests a physical
            // weight layout, without quantization or different input values.
            let weight_kn = weight.t()?.contiguous()?;
            let expected = x.matmul(&weight.t()?)?;
            let got = x.matmul(&weight_kn)?;
            let a = expected
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let b = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            let max_error = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_error <= 0.125,
                "rows={rows} out={out} in={input} max_error={max_error}"
            );
            let mut times: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
            for iteration in 0..60 {
                for offset in 0..3 {
                    let index = if iteration % 2 == 0 {
                        (iteration + offset) % 3
                    } else {
                        (iteration + 3 - offset) % 3
                    };
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = match index {
                        0 => super::matmul_nt(&x, &weight, super::GemmTile::Adaptive)?,
                        1 => x.matmul(&weight.t()?)?,
                        _ => x.matmul(&weight_kn)?,
                    };
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if iteration >= 12 {
                        times[index].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for (name, times) in ["adaptive_nt", "candle_nt", "candle_nn"]
                .into_iter()
                .zip(&mut times)
            {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "layout rows={rows} out={out} in={input} mode={name} median_us={:.3} min_us={:.3} max_us={:.3} max_error={max_error}",
                    (times[23] + times[24]) / 2.,
                    times[0],
                    times[47]
                );
            }
        }
        Ok(())
    }

    fn diagnose_gemm_shapes(shapes: &[(usize, usize, usize)]) -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        eprintln!(
            "architecture={} device_type={:?}",
            device.as_metal_device()?.device().architecture_name(),
            device.as_metal_device()?.device().device_type()
        );
        for &(rows, out, input) in shapes {
            let x = Tensor::from_vec(
                (0..rows * input)
                    .map(|i| ((i % 193) as f32 - 96.) / 151.)
                    .collect::<Vec<_>>(),
                (rows, input),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let w = Tensor::from_vec(
                (0..out * input)
                    .map(|i| ((i % 211) as f32 - 105.) / 501.)
                    .collect::<Vec<_>>(),
                (out, input),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let tiles = [
                None,
                Some(super::GemmTile::Small),
                Some(super::GemmTile::Balanced),
                Some(super::GemmTile::Wide),
                Some(super::GemmTile::Narrow),
                Some(super::GemmTile::SmallSwizzled),
                Some(super::GemmTile::Square4),
            ];
            let mut times: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::new());
            // Rotate/reverse the order so clocks and buffer-pool warmup cannot
            // systematically favor the last tile. Each alternative is warmed.
            for iteration in 0..60 {
                for j in 0..tiles.len() {
                    let index = if iteration % 2 == 0 {
                        (iteration + j) % tiles.len()
                    } else {
                        (iteration + tiles.len() - j) % tiles.len()
                    };
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = match tiles[index] {
                        None => x.matmul(&w.t()?)?,
                        Some(tile) => super::matmul_nt(&x, &w, tile)?,
                    };
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if iteration >= 12 {
                        times[index].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for (tile, times) in tiles.into_iter().zip(&mut times) {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "gemm rows={rows} out={out} in={input} tile={tile:?} median_us={:.3} min_us={:.3} max_us={:.3}",
                    (times[23] + times[24]) / 2.,
                    times[0],
                    times[47]
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn selectable_four_simd_tile_preserves_dense_linear_tails() -> anyhow::Result<()> {
        use candle_core::Module;
        let device = Device::new_metal(0)?;
        let selected = super::GemmTile::parse("64x64-simd4")?
            .ok_or_else(|| anyhow::anyhow!("four SIMD tile must be selectable"))?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for rows in [31, 33, 65] {
                let hidden = 1024;
                let out = 1001;
                let input = Tensor::from_vec(
                    (0..(rows + 2) * hidden)
                        .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                        .collect::<Vec<_>>(),
                    (rows + 2, 1, hidden),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, rows)?;
                let weight = Tensor::from_vec(
                    (0..(out + 2) * hidden)
                        .map(|i| ((i * 13 % 211) as f32 - 105.) / 501.)
                        .collect::<Vec<_>>(),
                    (out + 2, hidden),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, out)?;
                let dense = candle_nn::Linear::new(weight, None);
                let expected = dense.forward(&input)?;
                let linear = crate::isq_linear::LinearX::new(dense, None, &device)?;
                CALLS.with(|calls| calls.set(0));
                let got = {
                    let _guard = super::set_tile(Some(selected));
                    linear.forward(&input)?
                };
                assert_eq!(CALLS.with(Cell::get), 1);
                assert_eq!(LAST_TILE.with(Cell::get), Some(selected));
                assert_eq!(got.dims(), expected.dims());
                let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let expected = expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let max_error = got
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(got.iter().all(|value| value.is_finite()));
                let tolerance = match dtype {
                    DType::F32 => 1e-4,
                    DType::F16 => 0.015,
                    _ => 0.125,
                };
                assert!(
                    max_error <= tolerance,
                    "four SIMD dense LinearX dtype={dtype:?} rows={rows} max_error={max_error}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn tiled_nt_matches_candle_with_tails_and_offsets() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (rows, out, input) in [
                (3usize, 37usize, 19usize),
                (8, 1024, 1024),
                (32, 3072, 1024),
                (64, 1024, 3072),
                (128, 1024, 3072),
                (129, 1024, 3072),
                (1001, 3072, 1024),
            ] {
                let x = Tensor::from_vec(
                    (0..(rows + 2) * input)
                        .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                        .collect::<Vec<_>>(),
                    (rows + 2, input),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, rows)?;
                let w = Tensor::from_vec(
                    (0..(out + 1) * input)
                        .map(|i| ((i * 13 % 211) as f32 - 105.) / 501.)
                        .collect::<Vec<_>>(),
                    (out + 1, input),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, out)?;
                let expected = x.matmul(&w.t()?)?;
                for tile in super::GemmTile::ALL {
                    let got =
                        super::matmul_nt(&x.reshape((rows, 1, input))?, &w, tile)?.squeeze(1)?;
                    assert_eq!(got.dims(), expected.dims());
                    let a = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let b = expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert!(a.iter().chain(&b).all(|v| v.is_finite()));
                    let max = a
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let tolerance = match dtype {
                        DType::F32 => 1e-4,
                        DType::F16 => 0.015,
                        _ => 0.125,
                    };
                    assert!(
                        max <= tolerance,
                        "tile={tile:?} dtype={dtype:?} rows={rows} out={out} in={input} max={max}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn dense_linear_uses_decode_tile_and_keeps_prefill_reference() -> anyhow::Result<()> {
        use crate::isq_linear::{LinearX, set_linear_is_prefill};
        use candle_core::Module;
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::BF16] {
            let weight = Tensor::from_vec(
                (0..37 * 19)
                    .map(|i| ((i * 13 % 211) as f32 - 105.) / 501.)
                    .collect::<Vec<_>>(),
                (37, 19),
                &device,
            )?
            .to_dtype(dtype)?;
            let bias = Tensor::from_vec(
                (0..37).map(|i| i as f32 / 127.).collect::<Vec<_>>(),
                37,
                &device,
            )?
            .to_dtype(dtype)?;
            let dense = candle_nn::Linear::new(weight, Some(bias));
            let linear = LinearX::Linear(dense.clone());
            let x = Tensor::from_vec(
                (0..3 * 19)
                    .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                    .collect::<Vec<_>>(),
                (3, 1, 19),
                &device,
            )?
            .to_dtype(dtype)?;
            let reference = dense.forward(&x)?;
            let _tile = super::set_tile(Some(super::GemmTile::Small));
            CALLS.with(|count| count.set(0));
            let got = linear.forward(&x)?;
            assert_eq!(
                CALLS.with(Cell::get),
                1,
                "decode must use the candidate kernel"
            );
            let delta = (got.to_dtype(DType::F32)? - reference.to_dtype(DType::F32)?)?
                .abs()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                delta
                    .iter()
                    .all(|v| *v <= if dtype == DType::F32 { 1e-4 } else { 0.015 })
            );
            {
                let _prefill = set_linear_is_prefill(true);
                let got = linear.forward(&x)?;
                assert_eq!(
                    CALLS.with(Cell::get),
                    1,
                    "prefill must retain its reference path"
                );
                assert_eq!(
                    got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    reference
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn adaptive_decode_tiles_select_by_actual_rows() -> anyhow::Result<()> {
        use crate::isq_linear::LinearX;
        use candle_core::Module;
        let device = Device::new_metal(0)?;
        let weight = Tensor::randn(0f32, 0.03, (1024, 1024), &device)?.to_dtype(DType::BF16)?;
        let dense = candle_nn::Linear::new(weight, None);
        let linear = LinearX::Linear(dense.clone());
        let _tile = super::set_tile(Some(super::GemmTile::Adaptive));
        for rows in [32, 33, 64] {
            let x = Tensor::randn(0f32, 0.3, (rows, 1, 1024), &device)?.to_dtype(DType::BF16)?;
            let _got = linear.forward(&x)?;
            assert_eq!(
                LAST_TILE.with(Cell::get),
                Some(if rows <= 32 {
                    super::GemmTile::Small
                } else {
                    super::GemmTile::Narrow
                })
            );
            // Baseline is bit-identical at 33/64 rows; 32 rows is covered by
            // the nonzero F32/F16/BF16 tile comparison test.
            if rows > 32 {
                assert_eq!(
                    _got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    dense
                        .forward(&x)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn swizzled_small_tiles_keep_logical_tail_rows() -> anyhow::Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for rows in [3, 32, 33, 64, 65] {
                let x = Tensor::randn(0f32, 0.3, (rows, 19), &device)?.to_dtype(dtype)?;
                let w = Tensor::randn(0f32, 0.03, (37, 19), &device)?.to_dtype(dtype)?;
                let expected = super::matmul_nt(&x, &w, super::GemmTile::Small)?;
                let actual = super::matmul_nt(&x, &w, super::GemmTile::SmallSwizzled)?;
                assert_eq!(
                    actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    "swizzle must only change tile scheduling, including odd tile counts"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn tile_parser_and_nested_context() -> anyhow::Result<()> {
        assert_eq!(super::GemmTile::parse("0")?, None);
        assert!(super::GemmTile::parse("unknown").is_err());
        assert_eq!(super::current_tile(), None);
        let outer = super::set_tile(Some(super::GemmTile::Small));
        {
            let _inner = super::set_tile(Some(super::GemmTile::Balanced));
            assert_eq!(super::current_tile(), Some(super::GemmTile::Balanced));
        }
        assert_eq!(super::current_tile(), Some(super::GemmTile::Small));
        drop(outer);
        assert_eq!(super::current_tile(), None);
        Ok(())
    }

    #[test]
    fn prefill_tile_scope_keeps_decode_limits_and_restores_on_exit() {
        let outer = super::set_tile(Some(super::GemmTile::Small));
        assert_eq!(
            super::tile_for_linear(128, false),
            Some(super::GemmTile::Small)
        );
        assert_eq!(super::tile_for_linear(129, false), None);
        assert_eq!(super::tile_for_linear(8, true), None);
        {
            let _prefill = super::set_prefill_tile(Some(super::GemmTile::Narrow));
            assert_eq!(
                super::tile_for_linear(1000, true),
                Some(super::GemmTile::Narrow)
            );
            assert_eq!(
                super::tile_for_linear(8, false),
                Some(super::GemmTile::Narrow)
            );
            assert_eq!(super::tile_for_linear(1, true), None);
        }
        assert_eq!(super::tile_for_linear(8, true), None);
        drop(outer);
        assert_eq!(super::tile_for_linear(8, false), None);
    }
}
