//! Metal convolution candidate: reuse im2col, combine chunk rows in one GEMM.
use candle_core::{Result, Tensor};

#[cfg(test)]
static FAST_DISPATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn conv2d_flattened(
    input: &Tensor,
    weight: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    // The legacy flatten experiment remains a reference im2col route; the
    // independent NHWC fast override must not bypass a model's explicit off.
    conv2d_nhwc_with_im2col32(input, weight, padding, stride, dilation, false)?
        .permute((0, 3, 1, 2))?
        .contiguous()
}

/// Return spatial-major output so pointwise operations can execute contiguously
/// before exposing an NCHW view to the next strided im2col operation.
pub fn conv2d_nhwc(
    input: &Tensor,
    weight: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Result<Tensor> {
    let fast = std::env::var("VASR_METAL_CONV_IM2COL32")
        .ok()
        .is_some_and(|v| matches!(v.trim(), "1" | "true" | "on"));
    conv2d_nhwc_with_im2col32(input, weight, padding, stride, dilation, fast)
}

pub fn conv2d_nhwc_with_im2col32(
    input: &Tensor,
    weight: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    use_fast: bool,
) -> Result<Tensor> {
    conv2d_nhwc_with_gemm_tile(input, weight, padding, stride, dilation, use_fast, None)
}

pub fn conv2d_nhwc_with_gemm_tile(
    input: &Tensor,
    weight: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    use_fast: bool,
    gemm_tile: Option<vasr_quant::metal_gemm::GemmTile>,
) -> Result<Tensor> {
    let (batch, channels, height, width) = input.dims4()?;
    let (out_channels, kernel_channels, kh, kw) = weight.dims4()?;
    if !input.device().is_metal()
        || channels != kernel_channels
        || stride == 0
        || dilation == 0
        || kh == 0
        || kw == 0
    {
        candle_core::bail!("invalid flattened Metal conv2d inputs");
    }
    let output_size = |length: usize, kernel: usize| -> Result<usize> {
        let padded = length
            .checked_add(
                padding
                    .checked_mul(2)
                    .ok_or_else(|| candle_core::Error::Msg("conv padding overflow".into()))?,
            )
            .ok_or_else(|| candle_core::Error::Msg("conv length overflow".into()))?;
        let extent = dilation
            .checked_mul(kernel - 1)
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| candle_core::Error::Msg("conv kernel overflow".into()))?;
        Ok(padded
            .checked_sub(extent)
            .ok_or_else(|| candle_core::Error::Msg("conv kernel exceeds input".into()))?
            / stride
            + 1)
    };
    let oh = output_size(height, kh)?;
    let ow = output_size(width, kw)?;
    let rows = batch
        .checked_mul(oh)
        .and_then(|x| x.checked_mul(ow))
        .ok_or_else(|| candle_core::Error::Msg("conv rows overflow".into()))?;
    let cols = channels
        .checked_mul(kh)
        .and_then(|x| x.checked_mul(kw))
        .ok_or_else(|| candle_core::Error::Msg("conv columns overflow".into()))?;
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| candle_core::Error::Msg("conv allocation overflow".into()))?;
    let col = if use_fast
        && kh == 3
        && kw == 3
        && padding == 1
        && stride == 2
        && dilation == 1
        && im2col32_params(input).is_ok()
    {
        im2col3x3_stride2_32(input)?
    } else {
        input.apply_op1_no_bwd(&Im2Col {
            kh,
            kw,
            padding,
            stride,
            dilation,
            rows,
            cols,
            count,
        })?
    };
    let kernel = weight.contiguous()?.reshape((out_channels, cols))?;
    // The input layer has K=9; only the large-K CNN projections were profiled.
    let output = if let Some(tile) = gemm_tile.filter(|_| cols >= 1024) {
        vasr_quant::metal_gemm::matmul_nt(&col, &kernel, tile)?
    } else {
        col.matmul(&kernel.t()?)?
    };
    output.reshape((batch, oh, ow, out_channels))
}

fn im2col32_params(input: &Tensor) -> Result<([u32; 10], usize, usize)> {
    let (batch, channels, h, w) = input.dims4()?;
    if !input.device().is_metal()
        || batch == 0
        || channels == 0
        || h == 0
        || w == 0
        || h > i32::MAX as usize - 2
        || w > i32::MAX as usize - 2
    {
        candle_core::bail!("im2col32 requires nonempty bounded Metal input");
    }
    let rows = batch
        .checked_mul(h.div_ceil(2))
        .and_then(|v| v.checked_mul(w.div_ceil(2)))
        .ok_or_else(|| candle_core::Error::Msg("im2col32 row overflow".into()))?;
    let cols = channels
        .checked_mul(9)
        .ok_or_else(|| candle_core::Error::Msg("im2col32 columns overflow".into()))?;
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| candle_core::Error::Msg("im2col32 allocation overflow".into()))?;
    let strides = input.stride();
    let address = input
        .dims()
        .iter()
        .zip(strides)
        .try_fold(0usize, |sum, (&dim, &stride)| {
            (dim - 1)
                .checked_mul(stride)
                .and_then(|v| sum.checked_add(v))
        })
        .ok_or_else(|| candle_core::Error::Msg("im2col32 address overflow".into()))?;
    if count > u32::MAX as usize || address > u32::MAX as usize {
        candle_core::bail!("im2col32 input/output exceed 32-bit addressing");
    }
    let values = [
        channels,
        h,
        w,
        h.div_ceil(2),
        w.div_ceil(2),
        rows,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
    ];
    let mut p = [0u32; 10];
    for (dst, value) in p.iter_mut().zip(values) {
        *dst = u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg("im2col32 stride overflow".into()))?;
    }
    Ok((p, rows, cols))
}

/// Fuse the NHWC channel bias and existing GELU, preserving the intermediate
/// dtype rounding. Unsupported layouts retain the materialized reference.
pub fn bias_gelu_nhwc(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (_, _, _, channels) = input.dims4()?;
    if bias.dims1()? != channels
        || !input.device().same_device(bias.device())
        || input.dtype() != bias.dtype()
    {
        candle_core::bail!("NHWC bias requires matching channels, dtype and device");
    }
    let count = input.elem_count();
    if !input.device().is_metal()
        || !input.is_contiguous()
        || !bias.is_contiguous()
        || count == 0
        || count > u32::MAX as usize
        || !matches!(
            input.dtype(),
            candle_core::DType::F32 | candle_core::DType::F16 | candle_core::DType::BF16
        )
    {
        return input
            .broadcast_add(&bias.reshape((1, 1, 1, channels))?)?
            .gelu_erf();
    }
    input.apply_op2_no_bwd(
        bias,
        &BiasGelu {
            channels: channels as u32,
            store_boundary: false,
        },
    )
}

/// Fuse a last-dimension bias/GELU for dense encoder projections. The
/// original NHWC entry and its compiled kernel remain unchanged.
pub fn bias_gelu_last_dim(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let channels = *input
        .dims()
        .last()
        .ok_or_else(|| candle_core::Error::Msg("bias/GELU requires a last dimension".into()))?;
    if channels == 0
        || bias.dims1()? != channels
        || input.dtype() != bias.dtype()
        || !input.device().same_device(bias.device())
    {
        candle_core::bail!("bias/GELU requires matching nonzero width, dtype and device");
    }
    let fast_math = std::env::var("CANDLE_METAL_ENABLE_FAST_MATH")
        .map(|value| matches!(value.as_str(), "true" | "t" | "yes" | "y" | "1"))
        .unwrap_or(true);
    if !input.device().is_metal()
        || !input.is_contiguous()
        || !bias.is_contiguous()
        || input.elem_count() == 0
        || input.elem_count() > u32::MAX as usize
        || !fast_math
        || !matches!(
            input.dtype(),
            candle_core::DType::F32 | candle_core::DType::F16 | candle_core::DType::BF16
        )
    {
        return input.broadcast_add(bias)?.gelu_erf();
    }
    input.apply_op2_no_bwd(
        bias,
        &BiasGelu {
            channels: channels as u32,
            store_boundary: true,
        },
    )
}

struct BiasGelu {
    channels: u32,
    store_boundary: bool,
}
impl candle_core::CustomOp2 for BiasGelu {
    fn name(&self) -> &'static str {
        "audio-bias-gelu"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("fused bias/GELU requires Metal")
    }
    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        layout: &candle_core::Layout,
        bias: &candle_core::MetalStorage,
        bias_layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use super::compile::{CompilePolicy, KernelSource, load_pipeline_with_policy};
        use candle_core::{DType, backend::BackendStorage};
        use objc2_metal::MTLSize;
        let name = match (input.dtype(), self.store_boundary) {
            (DType::F32, false) => "bias_gelu_f32",
            (DType::F16, false) => "bias_gelu_f16",
            (DType::BF16, false) => "bias_gelu_bf16",
            (DType::F32, true) => "bias_gelu_last_f32",
            (DType::F16, true) => "bias_gelu_last_f16",
            (DType::BF16, true) => "bias_gelu_last_bf16",
            other => candle_core::bail!("bias/GELU unsupported dtype {other:?}"),
        };
        let device = input.device();
        let pipeline = load_pipeline_with_policy(
            device.device(),
            KernelSource::BiasGelu,
            name,
            None,
            if self.store_boundary {
                CompilePolicy::Fast
            } else {
                CompilePolicy::MetalDefault
            },
        )?;
        let count = layout.shape().elem_count();
        let output = device.new_buffer(count, input.dtype(), "audio-bias-gelu")?;
        let guard = device.command_encoder()?;
        let encoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(input.buffer()),
            layout.start_offset() * input.dtype().size_in_bytes(),
        );
        encoder.set_input_buffer(
            1,
            Some(bias.buffer()),
            bias_layout.start_offset() * bias.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &[count as u32, self.channels]);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: count.div_ceil(256),
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
            layout.shape().clone(),
        ))
    }
}

/// Exact specialization for the audio tower's 3x3, stride=2, padding=1 convolution.
pub fn im2col3x3_stride2_32(input: &Tensor) -> Result<Tensor> {
    let (params, rows, cols) = im2col32_params(input)?;
    input.apply_op1_no_bwd(&Im2Col32 { params, rows, cols })
}

struct Im2Col32 {
    params: [u32; 10],
    rows: usize,
    cols: usize,
}
impl candle_core::CustomOp1 for Im2Col32 {
    fn name(&self) -> &'static str {
        "audio-im2col32"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("im2col32 requires Metal storage")
    }
    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use super::compile::{CompilePolicy, KernelSource, load_pipeline_with_policy};
        use candle_core::{DType, backend::BackendStorage};
        use objc2_metal::MTLSize;
        let name = match input.dtype() {
            DType::F32 => "im2col32_f32",
            DType::F16 | DType::BF16 => "im2col32_16",
            other => candle_core::bail!("im2col32 unsupported dtype {other:?}"),
        };
        let device = input.device();
        let pipeline = load_pipeline_with_policy(
            device.device(),
            KernelSource::Im2Col32,
            name,
            None,
            CompilePolicy::MetalDefault,
        )?;
        let count = self.rows * self.cols;
        let output = device.new_buffer(count, input.dtype(), "audio-im2col32")?;
        let command_guard = device.command_encoder()?;
        let encoder = command_guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(input.buffer()),
            layout.start_offset() * input.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(1, Some(&output), 0);
        encoder.set_bytes(2, &self.params);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: (self.params[0] as usize).div_ceil(32),
                height: self.rows.div_ceil(4),
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        drop(command_guard);
        #[cfg(test)]
        FAST_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), count, input.dtype()),
            (self.rows, self.cols).into(),
        ))
    }
}

struct Im2Col {
    kh: usize,
    kw: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    rows: usize,
    cols: usize,
    count: usize,
}
impl candle_core::CustomOp1 for Im2Col {
    fn name(&self) -> &'static str {
        "flattened-conv-im2col"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("Metal im2col requires Metal storage")
    }
    fn metal_fwd(
        &self,
        input: &candle_core::MetalStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::{DType, backend::BackendStorage};
        let name = match input.dtype() {
            DType::F32 => "im2col_f32",
            DType::F16 => "im2col_f16",
            DType::BF16 => "im2col_bf16",
            other => candle_core::bail!("unsupported im2col dtype {other:?}"),
        };
        let device = input.device();
        let output = device.new_buffer(self.count, input.dtype(), "flattened-conv-im2col")?;
        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_im2col_strided(
            device.device(),
            &encoder,
            device.kernels(),
            name,
            layout.dims(),
            layout.stride(),
            (self.kh, self.kw, self.stride, self.padding, self.dilation),
            candle_metal_kernels::BufferOffset {
                buffer: input.buffer(),
                offset_in_bytes: layout.start_offset() * input.dtype().size_in_bytes(),
            },
            &output,
        )
        .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        drop(encoder);
        Ok((
            candle_core::MetalStorage::new(output, device.clone(), self.count, input.dtype()),
            (self.rows, self.cols).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    #[ignore = "requires Metal and VASR_METAL_CONV_IM2COL32=1; test serially"]
    fn legacy_flatten_keeps_reference_im2col() -> Result<()> {
        assert_eq!(
            std::env::var("VASR_METAL_CONV_IM2COL32").as_deref(),
            Ok("1")
        );
        let device = Device::new_metal(0)?;
        let x = Tensor::ones((1, 1, 7, 9), DType::BF16, &device)?;
        let w = Tensor::ones((3, 1, 3, 3), DType::BF16, &device)?;
        let before = FAST_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed);
        let _ = conv2d_flattened(&x, &w, 1, 2, 1)?;
        assert_eq!(
            FAST_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed),
            before,
            "reference flatten must not inherit the independent NHWC fast override"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires Metal GPU; run separately from benchmarks"]
    fn im2col32_matches_strided_reference_exactly() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (batch, channels, height, width) in
                [(2, 1, 17, 13), (8, 480, 64, 50), (1, 480, 32, 25)]
            {
                let x = Tensor::randn(
                    0f32,
                    0.3,
                    (batch + 1, channels, height + 2, width + 2),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, batch)?
                .narrow(2, 1, height)?
                .narrow(3, 1, width)?;
                let nhwc_view = x
                    .permute((0, 2, 3, 1))?
                    .contiguous()?
                    .permute((0, 3, 1, 2))?;
                for x in [x, nhwc_view] {
                    let rows = batch * height.div_ceil(2) * width.div_ceil(2);
                    let cols = channels * 9;
                    let reference = x.apply_op1_no_bwd(&Im2Col {
                        kh: 3,
                        kw: 3,
                        padding: 1,
                        stride: 2,
                        dilation: 1,
                        rows,
                        cols,
                        count: rows * cols,
                    })?;
                    let actual = im2col3x3_stride2_32(&x)?;
                    assert_eq!(actual.dims(), reference.dims());
                    assert_eq!(
                        actual
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        reference
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires Metal GPU; run separately from benchmarks"]
    fn nhwc_conv_chain_matches_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::BF16] {
            for (batch, h, w, chain) in [
                (2, 17, 13, [(1, 24), (24, 32), (32, 16)]),
                (8, 128, 100, [(1, 480), (480, 480), (480, 480)]),
            ] {
                // Narrowing preserves a nonzero storage offset and noncontiguous
                // spatial strides, as produced by feature chunk slicing.
                let mut actual = Tensor::randn(0f32, 0.3, (batch + 1, 1, h + 2, w + 2), &device)?
                    .to_dtype(dtype)?
                    .narrow(0, 1, batch)?
                    .narrow(2, 1, h)?
                    .narrow(3, 1, w)?;
                let mut expected = actual.clone();
                for (input, output) in chain {
                    let weight = Tensor::randn(0f32, 0.03, (output, input, 3, 3), &device)?
                        .to_dtype(dtype)?;
                    let bias = Tensor::randn(0f32, 0.01, output, &device)?.to_dtype(dtype)?;
                    expected = expected
                        .conv2d(&weight, 1, 2, 1, 1)?
                        .broadcast_add(&bias.reshape((1, output, 1, 1))?)?
                        .gelu_erf()?;
                    actual = conv2d_nhwc(&actual, &weight, 1, 2, 1)?
                        .broadcast_add(&bias.reshape((1, 1, 1, output))?)?
                        .gelu_erf()?
                        .permute((0, 3, 1, 2))?;
                    assert_eq!(actual.dims(), expected.dims());
                    let a = actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let b = expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let tolerance = if dtype == DType::F32 { 1e-5 } else { 0.01 };
                    for (a, b) in a.into_iter().zip(b) {
                        assert!(
                            (a - b).abs() <= tolerance + tolerance * b.abs(),
                            "{dtype:?}: {a} != {b}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn tiled_nhwc_conv_preserves_tail_chain() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::BF16] {
            let mut expected = Tensor::randn(0f32, 0.3, (3, 1, 19, 15), &device)?
                .to_dtype(dtype)?
                .narrow(0, 1, 2)?
                .narrow(2, 1, 17)?
                .narrow(3, 1, 13)?;
            let mut actual = expected.clone();
            for (input, output) in [(1, 480), (480, 480), (480, 480)] {
                let weight =
                    Tensor::randn(0f32, 0.03, (output, input, 3, 3), &device)?.to_dtype(dtype)?;
                expected = conv2d_nhwc_with_im2col32(&expected, &weight, 1, 2, 1, true)?
                    .gelu_erf()?
                    .permute((0, 3, 1, 2))?;
                actual = conv2d_nhwc_with_gemm_tile(
                    &actual,
                    &weight,
                    1,
                    2,
                    1,
                    true,
                    Some(vasr_quant::metal_gemm::GemmTile::Narrow),
                )?
                .gelu_erf()?
                .permute((0, 3, 1, 2))?;
                let a = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let b = expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(actual.dims(), expected.dims());
                let tolerance = if dtype == DType::F32 { 1e-5 } else { 0.01 };
                assert!(a.iter().zip(&b).all(|(a, b)| a.is_finite()
                    && b.is_finite()
                    && (a - b).abs() <= tolerance + tolerance * b.abs()));
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Metal GPU"]
    fn last_dim_bias_gelu_matches_materialized_exactly() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (rows, width) in [(1, 13), (17, 896), (257, 3584)] {
                let x = Tensor::from_vec(
                    (0..(rows + 1) * width)
                        .map(|i| ((i * 17 % 1009) as f32 - 504.) / 23.)
                        .collect::<Vec<_>>(),
                    (rows + 1, width),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, rows)?;
                let b = Tensor::from_vec(
                    (0..width + 1)
                        .map(|i| ((i * 13 % 127) as f32 - 63.) / 51.)
                        .collect::<Vec<_>>(),
                    width + 1,
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, width)?;
                for x in [
                    x.clone(),
                    x.reshape((1, rows, width))?,
                    x.t()?.contiguous()?.t()?,
                ] {
                    let expected = x.broadcast_add(&b)?.gelu_erf()?;
                    let actual = bias_gelu_last_dim(&x, &b)?;
                    assert_eq!(actual.dims(), expected.dims());
                    assert_eq!(
                        actual
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        expected
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        "must preserve bias rounding and GELU at {dtype:?}/{rows}/{width}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn last_dim_bias_gelu_cpu_fallback_and_validation() -> Result<()> {
        let input = Tensor::from_vec(vec![-4f32, -1., 0., 1., 2., 5.], (2, 3), &Device::Cpu)?;
        let bias = Tensor::from_vec(vec![0.125f32, -0.5, 1.], 3, &Device::Cpu)?;
        let expected = input.broadcast_add(&bias)?.gelu_erf()?;
        let actual = bias_gelu_last_dim(&input, &bias)?;
        assert_eq!(actual.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        assert!(bias_gelu_last_dim(&input, &bias.narrow(0, 0, 2)?).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "synchronized Metal microbenchmark; requires exclusive GPU"]
    fn diagnose_encoder_bias_gelu() -> Result<()> {
        let device = Device::new_metal(0)?;
        for rows in [128, 1000, 3000] {
            let x = Tensor::randn(0f32, 1., (rows, 3584), &device)?.to_dtype(DType::BF16)?;
            let bias = Tensor::randn(0f32, 0.1, 3584, &device)?.to_dtype(DType::BF16)?;
            let mut samples = [Vec::new(), Vec::new()];
            // Alternate order so clock and thermal drift affect both variants.
            for iteration in 0..48 {
                for phase in [iteration % 2, 1 - iteration % 2] {
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    let output = if phase == 0 {
                        x.broadcast_add(&bias)?.gelu_erf()?
                    } else {
                        bias_gelu_last_dim(&x, &bias)?
                    };
                    device.synchronize()?;
                    std::hint::black_box(output);
                    if iteration >= 8 {
                        samples[phase].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for (phase, mut times) in samples.into_iter().enumerate() {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "encoder_bias_gelu rows={rows} phase={phase} median_us={:.3}",
                    (times[19] + times[20]) / 2.
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires Metal GPU; run separately from benchmarks"]
    fn nhwc_bias_gelu_matches_materialized_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for (rows, channels) in [(37, 7), (25600, 480), (6400, 480), (1664, 480)] {
                // Wide positive/negative range and near-zero values exercise erf
                // and the materialized low-precision addition boundary.
                let input = Tensor::from_vec(
                    (0..(rows + 1) * channels)
                        .map(|i| ((i % 311) as f32 - 155.) / 23.)
                        .collect::<Vec<_>>(),
                    (rows + 1, channels),
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, rows)?;
                let bias = Tensor::from_vec(
                    (0..channels + 1)
                        .map(|i| ((i % 19) as f32 - 9.) / 37.)
                        .collect::<Vec<_>>(),
                    channels + 1,
                    &device,
                )?
                .to_dtype(dtype)?
                .narrow(0, 1, channels)?;
                for x in [
                    input.reshape((1, 1, rows, channels))?,
                    input
                        .t()?
                        .contiguous()?
                        .t()?
                        .reshape((1, 1, rows, channels))?,
                ] {
                    let expected = x
                        .broadcast_add(&bias.reshape((1, 1, 1, channels))?)?
                        .gelu_erf()?;
                    let actual = bias_gelu_nhwc(&x, &bias)?;
                    let a = actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let b = expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        a, b,
                        "bias/GELU must preserve materialized {dtype:?} arithmetic"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "synchronized Metal diagnosis; never run alongside throughput benchmarks"]
    fn diagnose_encoder_conv_stages() -> Result<()> {
        let device = Device::new_metal(0)?;
        for batch in [1, 8] {
            for (channels, height, width) in [(1, 128, 100), (480, 64, 50), (480, 32, 25)] {
                let x = Tensor::randn(0f32, 0.3, (batch, channels, height, width), &device)?
                    .to_dtype(DType::BF16)?;
                let w = Tensor::randn(0f32, 0.01, (480, channels, 3, 3), &device)?
                    .to_dtype(DType::BF16)?;
                let oh = (height + 1) / 2;
                let ow = (width + 1) / 2;
                let rows = batch * oh * ow;
                let cols = channels * 9;
                let op = Im2Col {
                    kh: 3,
                    kw: 3,
                    padding: 1,
                    stride: 2,
                    dilation: 1,
                    rows,
                    cols,
                    count: rows * cols,
                };
                for phase in [
                    "conv",
                    "im2col",
                    "im2col32",
                    "gemm",
                    "gemm_nn",
                    "restore",
                    "bias",
                    "gelu",
                    "bias_gelu",
                    "fused_bias_gelu",
                ] {
                    let col = x.apply_op1_no_bwd(&op)?;
                    let kernel = w.reshape((480, cols))?.t()?;
                    let kernel_nn = kernel.contiguous()?;
                    let raw = col.matmul(&kernel)?.reshape((batch, oh, ow, 480))?;
                    let y = raw.permute((0, 3, 1, 2))?;
                    let bias = Tensor::ones((1, 1, 1, 480), DType::BF16, &device)?;
                    let mut samples = Vec::new();
                    for iteration in 0..16 {
                        device.synchronize()?;
                        let start = std::time::Instant::now();
                        let result = match phase {
                            "conv" => x.conv2d(&w, 1, 2, 1, 1)?,
                            "im2col" => x.apply_op1_no_bwd(&op)?,
                            "im2col32" => im2col3x3_stride2_32(&x)?,
                            "gemm" => col.matmul(&kernel)?,
                            "gemm_nn" => col.matmul(&kernel_nn)?,
                            "restore" => y.contiguous()?,
                            "bias" => raw.broadcast_add(&bias)?,
                            "gelu" => raw.gelu_erf()?,
                            "bias_gelu" => raw.broadcast_add(&bias)?.gelu_erf()?,
                            "fused_bias_gelu" => bias_gelu_nhwc(&raw, &bias.flatten_all()?)?,
                            _ => unreachable!(),
                        };
                        device.synchronize()?;
                        if iteration >= 4 {
                            samples.push(start.elapsed().as_secs_f64() * 1000.);
                        }
                        drop(result);
                    }
                    samples.sort_by(f64::total_cmp);
                    println!(
                        "conv_diagnosis batch={batch} input={channels}x{height}x{width} phase={phase} median_ms={:.3} range_ms={:.3}..{:.3}",
                        (samples[5] + samples[6]) / 2.,
                        samples[0],
                        samples[11]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires Metal GPU; run separately from benchmarks"]
    fn flattened_conv_matches_nonzero_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::BF16] {
            for batch in [1, 8, 16] {
                for channels in [1, 32, 480] {
                    let (h, w, out) = if channels == 480 {
                        (40, 50, 480)
                    } else {
                        (13, 11, 24)
                    };
                    let x = Tensor::from_vec(
                        (0..batch * channels * h * w)
                            .map(|i| ((i % 97) as f32 - 48.) / 97.)
                            .collect::<Vec<_>>(),
                        (batch, channels, h, w),
                        &device,
                    )?
                    .to_dtype(dtype)?;
                    let weight = Tensor::from_vec(
                        (0..out * channels * 3 * 3)
                            .map(|i| ((i % 29) as f32 - 14.) / 300.)
                            .collect::<Vec<_>>(),
                        (out, channels, 3, 3),
                        &device,
                    )?
                    .to_dtype(dtype)?;
                    for input in [x.clone(), x.transpose(2, 3)?] {
                        let expected = input.conv2d(&weight, 1, 2, 1, 1)?;
                        let actual = conv2d_flattened(&input, &weight, 1, 2, 1)?;
                        assert_eq!(actual.dims(), expected.dims());
                        let a = actual
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        let b = expected
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        let tolerance = if dtype == DType::F32 { 1e-5 } else { 0.01 };
                        for (a, b) in a.into_iter().zip(b) {
                            assert!(a.is_finite() && b.is_finite());
                            assert!(
                                (a - b).abs() <= tolerance + tolerance * b.abs(),
                                "{dtype:?} batch={batch} channels={channels} {a} != {b}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
