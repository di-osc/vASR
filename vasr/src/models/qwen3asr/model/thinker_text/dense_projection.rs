//! Dense projections sharing an input allocation.
use crate::model::isq_linear::IsqLinear;
use candle_core::{Result, Tensor};
use candle_nn::Module;

/// One dense GEMM for independent projections sharing the same input.
/// Quantized weights and other devices retain their existing paths.
#[derive(Debug, Clone)]
pub(super) struct MetalDenseProjection {
    pub(super) linear: candle_nn::Linear,
    pub(super) output_dims: Vec<usize>,
}

impl MetalDenseProjection {
    pub(super) fn new(projections: &[&IsqLinear], enabled: bool) -> Result<Option<Self>> {
        if !enabled || projections.is_empty() {
            return Ok(None);
        }
        let mut linears = Vec::with_capacity(projections.len());
        for projection in projections {
            let IsqLinear::Linear(linear) = projection else {
                return Ok(None);
            };
            linears.push(linear);
        }
        let first = linears[0].weight();
        if !first.device().is_metal() {
            return Ok(None);
        }
        let input_dim = first.dim(1)?;
        let dtype = first.dtype();
        let has_bias = linears[0].bias().is_some();
        let mut weights = Vec::with_capacity(linears.len());
        let mut biases = Vec::with_capacity(linears.len());
        let mut output_dims = Vec::with_capacity(linears.len());
        for linear in linears {
            let weight = linear.weight();
            if weight.dim(1)? != input_dim
                || weight.dtype() != dtype
                || !weight.device().same_device(first.device())
                || linear.bias().is_some() != has_bias
            {
                return Ok(None);
            }
            output_dims.push(weight.dim(0)?);
            weights.push(weight);
            if let Some(bias) = linear.bias() {
                if bias.dtype() != dtype || !bias.device().same_device(first.device()) {
                    return Ok(None);
                }
                biases.push(bias);
            }
        }
        // Metal's BF16 copy kernels do not support every strided cat layout.
        // Concatenate via F32 on-device only during load; inference stays BF16.
        let concat = |tensors: &[&Tensor]| -> Result<Tensor> {
            if dtype == candle_core::DType::BF16 {
                let tensors = tensors
                    .iter()
                    .map(|tensor| tensor.to_dtype(candle_core::DType::F32)?.contiguous())
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&tensors.iter().collect::<Vec<_>>(), 0)?.to_dtype(dtype)
            } else {
                Tensor::cat(tensors, 0)
            }
        };
        let weight = concat(&weights)?;
        let bias = if has_bias {
            Some(concat(&biases)?)
        } else {
            None
        };
        Ok(Some(Self {
            linear: candle_nn::Linear::new(weight, bias),
            output_dims,
        }))
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        self.forward_views(xs)?
            .into_iter()
            .map(|view| view.contiguous())
            .collect()
    }

    pub(super) fn forward_views(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let joined = self.linear.forward(&xs.contiguous()?)?;
        self.split_output(&joined)
    }

    /// Apply the current prefill GEMM policy before returning strided views.
    #[cfg(any(feature = "metal-paged-attn", all(test, feature = "metal")))]
    pub(super) fn forward_prefill_views(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let joined = IsqLinear::Linear(self.linear.clone()).forward(&xs.contiguous()?)?;
        self.split_output(&joined)
    }

    pub(super) fn split_output(&self, joined: &Tensor) -> Result<Vec<Tensor>> {
        let last_dim = joined.rank() - 1;
        let mut start = 0;
        let mut outputs = Vec::with_capacity(self.output_dims.len());
        for &dim in &self.output_dims {
            outputs.push(joined.narrow(last_dim, start, dim)?);
            start += dim;
        }
        Ok(outputs)
    }

    /// Gate and up keep contiguous slices of the joined allocation, including
    /// their nonzero offsets. Decode need not retain another copy of each weight.
    pub(super) fn shared_linears(&self) -> Result<Vec<IsqLinear>> {
        let mut start = 0;
        self.output_dims
            .iter()
            .map(|&dim| {
                let weight = self.linear.weight().narrow(0, start, dim)?;
                let bias = self
                    .linear
                    .bias()
                    .map(|bias| bias.narrow(0, start, dim))
                    .transpose()?;
                start += dim;
                Ok(IsqLinear::Linear(candle_nn::Linear::new(weight, bias)))
            })
            .collect()
    }
}
