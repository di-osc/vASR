use crate::ffi;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::bf16;

fn round_multiple(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

struct FlashAttn {
    softmax_scale: f32,
    causal: bool,
}

impl FlashAttn {
    fn cuda_fwd_t(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let q = q.as_cuda_slice::<bf16>()?;
        let k = k.as_cuda_slice::<bf16>()?;
        let v = v.as_cuda_slice::<bf16>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle_core::bail!(
                "vasr-flash-attn expects rank-4 q/k/v tensors (q={q_rank}, k={k_rank}, v={v_rank})"
            )
        }
        if q_stride[q_rank - 1] != 1 || k_stride[k_rank - 1] != 1 || v_stride[v_rank - 1] != 1 {
            candle_core::bail!(
                "vasr-flash-attn expects contiguous last dimension (q={q_stride:?}, k={k_stride:?}, v={v_stride:?})"
            )
        }

        let (b_sz, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
        let (_b_sz, seqlen_k, num_heads_k, _head_size_og) = k_l.shape().dims4()?;
        let expected_kv = (b_sz, seqlen_k, num_heads_k, head_size_og);
        if expected_kv != k_l.shape().dims4()? || expected_kv != v_l.shape().dims4()? {
            candle_core::bail!(
                "shape mismatch q {:?}, k {:?}, v {:?}",
                q_l.shape(),
                k_l.shape(),
                v_l.shape()
            )
        }
        validate_supported(head_size_og, self.causal)?;
        if num_heads % num_heads_k != 0 {
            candle_core::bail!(
                "number of k/v heads {num_heads_k} must divide number of query heads {num_heads}"
            )
        }

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let mut dst = unsafe { dev.alloc::<bf16>(elem_count)? };
        let mut softmax_lse = dev.alloc_zeros::<f32>(b_sz * num_heads * seqlen_q)?;
        let stream = dev.cuda_stream();

        unsafe {
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (k_ptr, _guard) = k.device_ptr(&stream);
            let (v_ptr, _guard) = v.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            let (softmax_lse_ptr, _guard) = softmax_lse.device_ptr_mut(&stream);
            ffi::run_mha(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                dst_ptr as *const core::ffi::c_void,
                softmax_lse_ptr as *const core::ffi::c_void,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                q_stride[0] as u32,
                k_stride[0] as u32,
                v_stride[0] as u32,
                o_stride[0] as u32,
                0,
                q_stride[q_rank - 3] as u32,
                k_stride[k_rank - 3] as u32,
                v_stride[v_rank - 3] as u32,
                o_stride[o_rank - 3] as u32,
                q_stride[q_rank - 2] as u32,
                k_stride[k_rank - 2] as u32,
                v_stride[v_rank - 2] as u32,
                o_stride[o_rank - 2] as u32,
                b_sz as u32,
                num_heads as u32,
                num_heads_k as u32,
                head_size as u32,
                head_size_rounded as u32,
                self.softmax_scale,
                seqlen_q as u32,
                seqlen_k as u32,
                seqlen_q_rounded as u32,
                seqlen_k_rounded as u32,
                (b_sz * seqlen_q) as u32,
                1,
                i32::from(self.causal),
                0,
                -1,
                if self.causal { 0 } else { -1 },
                0.0,
                std::ptr::null(),
                0,
                -1,
                std::ptr::null(),
                0,
                0,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        }

        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle_core::CustomOp3 for FlashAttn {
    fn name(&self) -> &'static str {
        "vasr-flash-attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("vasr-flash-attn has no cpu support")
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match q.dtype() {
            DType::BF16 => self.cuda_fwd_t(q, q_l, k, k_l, v, v_l),
            dt => candle_core::bail!("vasr-flash-attn only supports bf16, got {dt:?}"),
        }
    }
}

struct FlashAttnVarLen {
    softmax_scale: f32,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    seqlens_q: Tensor,
    seqlens_k: Tensor,
    causal: bool,
}

impl FlashAttnVarLen {
    fn cuda_fwd_t(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let out_l = Layout::contiguous(&out_shape);

        let (seqlens_q, seqlens_q_layout) = self.seqlens_q.storage_and_layout();
        let seqlens_q = match &*seqlens_q {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle_core::bail!("seqlens_q must be a cuda tensor"),
        };
        let seqlens_q = match seqlens_q_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_q.slice(o1..o2),
            None => candle_core::bail!("seqlens_q must be contiguous"),
        };

        let (seqlens_k, seqlens_k_layout) = self.seqlens_k.storage_and_layout();
        let seqlens_k = match &*seqlens_k {
            candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle_core::bail!("seqlens_k must be a cuda tensor"),
        };
        let seqlens_k = match seqlens_k_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_k.slice(o1..o2),
            None => candle_core::bail!("seqlens_k must be contiguous"),
        };

        let q = q.as_cuda_slice::<bf16>()?;
        let k = k.as_cuda_slice::<bf16>()?;
        let v = v.as_cuda_slice::<bf16>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();
        let o_stride = out_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();
        let o_rank = o_stride.len();

        if q_rank != 3 || k_rank != 3 || v_rank != 3 {
            candle_core::bail!(
                "vasr-flash-attn-varlen expects rank-3 q/k/v tensors (q={q_rank}, k={k_rank}, v={v_rank})"
            )
        }
        if q_stride[q_rank - 1] != 1 || k_stride[k_rank - 1] != 1 || v_stride[v_rank - 1] != 1 {
            candle_core::bail!(
                "vasr-flash-attn-varlen expects contiguous last dimension (q={q_stride:?}, k={k_stride:?}, v={v_stride:?})"
            )
        }

        let (total_q, num_heads, head_size_og) = q_l.shape().dims3()?;
        let (total_k, num_heads_k, _head_size_og) = k_l.shape().dims3()?;
        let expected_kv = (total_k, num_heads_k, head_size_og);
        if expected_kv != k_l.shape().dims3()? || expected_kv != v_l.shape().dims3()? {
            candle_core::bail!(
                "shape mismatch q {:?}, k {:?}, v {:?}",
                q_l.shape(),
                k_l.shape(),
                v_l.shape()
            )
        }
        validate_supported(head_size_og, self.causal)?;
        if num_heads % num_heads_k != 0 {
            candle_core::bail!(
                "number of k/v heads {num_heads_k} must divide number of query heads {num_heads}"
            )
        }

        let nseqlens_q = seqlens_q_layout.shape().dims1()?;
        let nseqlens_k = seqlens_k_layout.shape().dims1()?;
        if nseqlens_q < 2 || nseqlens_k != nseqlens_q {
            candle_core::bail!(
                "seqlens_q and seqlens_k must have matching length >= 2 ({nseqlens_q} vs {nseqlens_k})"
            )
        }
        let batch_size = nseqlens_q - 1;

        let head_size = round_multiple(head_size_og, 8);
        let head_size_rounded = round_multiple(head_size, 32);
        let seqlen_q_rounded = round_multiple(self.max_seqlen_q, 128);
        let seqlen_k_rounded = round_multiple(self.max_seqlen_k, 128);

        let elem_count = out_shape.elem_count();
        let mut dst = unsafe { dev.alloc::<bf16>(elem_count)? };
        let mut softmax_lse = dev.alloc_zeros::<f32>(num_heads * total_q)?;
        let stream = dev.cuda_stream();

        unsafe {
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (k_ptr, _guard) = k.device_ptr(&stream);
            let (v_ptr, _guard) = v.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
            let (softmax_lse_ptr, _guard) = softmax_lse.device_ptr_mut(&stream);
            let (seqlens_q_ptr, _guard) = seqlens_q.device_ptr(&stream);
            let (seqlens_k_ptr, _guard) = seqlens_k.device_ptr(&stream);
            ffi::run_mha(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                dst_ptr as *const core::ffi::c_void,
                softmax_lse_ptr as *const core::ffi::c_void,
                std::ptr::null(),
                seqlens_q_ptr as *const i32,
                seqlens_k_ptr as *const i32,
                0,
                0,
                0,
                0,
                0,
                q_stride[q_rank - 3] as u32,
                k_stride[k_rank - 3] as u32,
                v_stride[v_rank - 3] as u32,
                o_stride[o_rank - 3] as u32,
                q_stride[q_rank - 2] as u32,
                k_stride[k_rank - 2] as u32,
                v_stride[v_rank - 2] as u32,
                o_stride[o_rank - 2] as u32,
                batch_size as u32,
                num_heads as u32,
                num_heads_k as u32,
                head_size as u32,
                head_size_rounded as u32,
                self.softmax_scale,
                self.max_seqlen_q as u32,
                self.max_seqlen_k as u32,
                seqlen_q_rounded as u32,
                seqlen_k_rounded as u32,
                total_q as u32,
                1,
                i32::from(self.causal),
                1,
                -1,
                if self.causal { 0 } else { -1 },
                0.0,
                std::ptr::null(),
                0,
                -1,
                std::ptr::null(),
                0,
                0,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        }

        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle_core::CustomOp3 for FlashAttnVarLen {
    fn name(&self) -> &'static str {
        "vasr-flash-attn-varlen"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("vasr-flash-attn has no cpu support")
    }

    fn cuda_fwd(
        &self,
        q: &candle_core::CudaStorage,
        q_l: &Layout,
        k: &candle_core::CudaStorage,
        k_l: &Layout,
        v: &candle_core::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match q.dtype() {
            DType::BF16 => self.cuda_fwd_t(q, q_l, k, k_l, v, v_l),
            dt => candle_core::bail!("vasr-flash-attn only supports bf16, got {dt:?}"),
        }
    }
}

fn validate_supported(head_dim: usize, causal: bool) -> Result<()> {
    match (head_dim, causal) {
        (64, false) | (128, true) => Ok(()),
        _ => candle_core::bail!(
            "vasr-flash-attn only includes bf16 Qwen3-ASR kernels: hdim64 non-causal and hdim128 causal; got hdim={head_dim} causal={causal}"
        ),
    }
}

pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    q.apply_op3(
        k,
        v,
        FlashAttn {
            softmax_scale,
            causal,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    q.apply_op3(
        k,
        v,
        FlashAttnVarLen {
            softmax_scale,
            max_seqlen_q,
            max_seqlen_k,
            seqlens_q: seqlens_q.clone(),
            seqlens_k: seqlens_k.clone(),
            causal,
        },
    )
}
