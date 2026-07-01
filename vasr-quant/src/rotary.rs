use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::{CudaStorage, DType, Result, Shape, Storage, Tensor};

unsafe extern "C" {
    fn vasr_rotary_embedding(
        query: *const core::ffi::c_void,
        key: *const core::ffi::c_void,
        cos_cache: *const core::ffi::c_void,
        sin_cache: *const core::ffi::c_void,
        query_out: *mut core::ffi::c_void,
        key_out: *mut core::ffi::c_void,
        is_neox: i32,
        head_size: i32,
        num_tokens: i64,
        rot_dim: i32,
        num_heads: i32,
        num_kv_heads: i32,
        query_stride: i64,
        key_stride: i64,
        dtype: u32,
        stream: i64,
    );
}

pub fn apply_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
) -> Result<(Tensor, Tensor)> {
    if !q.device().is_cuda() {
        candle_core::bail!("CUDA rotary requires CUDA tensors");
    }
    if q.dtype() != k.dtype() || q.dtype() != cos.dtype() || q.dtype() != sin.dtype() {
        candle_core::bail!(
            "RoPE dtype mismatch q={:?} k={:?} cos={:?} sin={:?}",
            q.dtype(),
            k.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }
    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
        candle_core::bail!("q/k RoPE shape mismatch {:?} {:?}", q.shape(), k.shape());
    }
    let (cache_rows, rot_dim) = match cos.dims() {
        [rows, dim] => (*rows, *dim),
        [cos_batch, cos_seq, dim] if *cos_batch == batch && *cos_seq == seq_len => {
            (batch * seq_len, *dim)
        }
        _ => candle_core::bail!("invalid RoPE cos shape {:?}", cos.shape()),
    };
    let (sin_rows, sin_dim) = match sin.dims() {
        [rows, dim] => (*rows, *dim),
        [sin_batch, sin_seq, dim] if *sin_batch == batch && *sin_seq == seq_len => {
            (batch * seq_len, *dim)
        }
        _ => candle_core::bail!("invalid RoPE sin shape {:?}", sin.shape()),
    };
    if (cache_rows, rot_dim) != (sin_rows, sin_dim) {
        candle_core::bail!(
            "RoPE cos/sin shape mismatch {:?} {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if cache_rows != seq_len && cache_rows != batch * seq_len {
        candle_core::bail!(
            "RoPE cache rows {cache_rows} are incompatible with batch {batch} and seq {seq_len}"
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        candle_core::bail!(
            "RoPE rot dim {} is incompatible with head dim {head_dim}",
            rot_dim * 2
        );
    }

    let q_embed = q.transpose(1, 2)?.flatten(0, 1)?.contiguous()?;
    let k_embed = k.transpose(1, 2)?.flatten(0, 1)?.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let dev = q.device().clone();
    let dev = match dev {
        candle_core::Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };
    let stream = dev.cuda_stream();
    let q_storage = q_embed.storage_and_layout().0;
    let k_storage = k_embed.storage_and_layout().0;
    let cos_storage = cos.storage_and_layout().0;
    let sin_storage = sin.storage_and_layout().0;
    let q_cuda = match &*q_storage {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("q must be CUDA"),
    };
    let k_cuda = match &*k_storage {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("k must be CUDA"),
    };
    let cos_cuda = match &*cos_storage {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("cos must be CUDA"),
    };
    let sin_cuda = match &*sin_storage {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("sin must be CUDA"),
    };

    macro_rules! run {
        ($ty:ty, $dtype:expr) => {{
            let mut q_out = dev.alloc_zeros::<$ty>(q_embed.elem_count())?;
            let mut k_out = dev.alloc_zeros::<$ty>(k_embed.elem_count())?;
            let q_slice = q_cuda.as_cuda_slice::<$ty>()?;
            let k_slice = k_cuda.as_cuda_slice::<$ty>()?;
            let cos_slice = cos_cuda.as_cuda_slice::<$ty>()?;
            let sin_slice = sin_cuda.as_cuda_slice::<$ty>()?;
            let (q_ptr, _q_guard) = q_slice.device_ptr(&stream);
            let (k_ptr, _k_guard) = k_slice.device_ptr(&stream);
            let (cos_ptr, _cos_guard) = cos_slice.device_ptr(&stream);
            let (sin_ptr, _sin_guard) = sin_slice.device_ptr(&stream);
            let (q_out_ptr, _q_out_guard) = q_out.device_ptr_mut(&stream);
            let (k_out_ptr, _k_out_guard) = k_out.device_ptr_mut(&stream);
            unsafe {
                vasr_rotary_embedding(
                    q_ptr as *const core::ffi::c_void,
                    k_ptr as *const core::ffi::c_void,
                    cos_ptr as *const core::ffi::c_void,
                    sin_ptr as *const core::ffi::c_void,
                    q_out_ptr as *mut core::ffi::c_void,
                    k_out_ptr as *mut core::ffi::c_void,
                    i32::from(is_neox),
                    head_dim as i32,
                    (batch * seq_len) as i64,
                    rot_dim as i32,
                    q_heads as i32,
                    k_heads as i32,
                    (q_heads * head_dim) as i64,
                    (k_heads * head_dim) as i64,
                    $dtype,
                    stream.cu_stream() as i64,
                );
            }
            drop(_q_guard);
            drop(_k_guard);
            drop(_cos_guard);
            drop(_sin_guard);
            drop(_q_out_guard);
            drop(_k_out_guard);
            let q_storage = CudaStorage::wrap_cuda_slice(q_out, dev.clone());
            let q_out = Tensor::from((Storage::Cuda(q_storage), Shape::from(q_embed.dims())));
            let q_out = q_out
                .reshape((batch, seq_len, q_heads, head_dim))?
                .transpose(1, 2)?;
            let k_storage = CudaStorage::wrap_cuda_slice(k_out, dev.clone());
            let k_out = Tensor::from((Storage::Cuda(k_storage), Shape::from(k_embed.dims())));
            let k_out = k_out
                .reshape((batch, seq_len, k_heads, head_dim))?
                .transpose(1, 2)?;
            return Ok((q_out, k_out));
        }};
    }

    match q.dtype() {
        DType::F16 => run!(half::f16, 0),
        DType::BF16 => run!(half::bf16, 1),
        DType::F32 => run!(f32, 2),
        other => candle_core::bail!("unsupported CUDA RoPE dtype {other:?}"),
    }
}
