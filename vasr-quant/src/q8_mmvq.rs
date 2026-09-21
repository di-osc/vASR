//! Q8_0 decode matvec fast path adapted from mistral.rs `fast_mmvq`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr};
use candle_core::cuda_backend::cudarc::driver::{CudaStream, SyncOnDrop};
use candle_core::cuda_backend::{CudaDType, DeviceId};
use candle_core::op::BackpropOp;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36;
const Q8_1_MMQ_TYPE_SIZE: usize = 144;
const MATRIX_ROW_PADDING: usize = 512;
const Q8_MMVQ_MAX_BATCH: usize = 8;
const DEFAULT_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_BYTES: usize = 256 * 1024 * 1024;
type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);
static PLAIN_PREFILL_CALLS: AtomicU64 = AtomicU64::new(0);
static PLAIN_DECODE_CALLS: AtomicU64 = AtomicU64::new(0);
static PLAIN_PREFILL_ROWS: AtomicU64 = AtomicU64::new(0);
static PLAIN_DECODE_ROWS: AtomicU64 = AtomicU64::new(0);
static PLAIN_AUDIO_CALLS: AtomicU64 = AtomicU64::new(0);
static PLAIN_AUDIO_ROWS: AtomicU64 = AtomicU64::new(0);
static PLAIN_LM_HEAD_CALLS: AtomicU64 = AtomicU64::new(0);
static PLAIN_LM_HEAD_ROWS: AtomicU64 = AtomicU64::new(0);
static MMQ_PREFILL_CALLS: AtomicU64 = AtomicU64::new(0);
static MMQ_PREFILL_ROWS: AtomicU64 = AtomicU64::new(0);
static MMQ_DECODE_CALLS: AtomicU64 = AtomicU64::new(0);
static MMQ_DECODE_ROWS: AtomicU64 = AtomicU64::new(0);
static MMQ_LM_HEAD_CALLS: AtomicU64 = AtomicU64::new(0);
static MMQ_LM_HEAD_ROWS: AtomicU64 = AtomicU64::new(0);
static PREFILL_LM_HEAD_MMQ_SKIP_CALLS: AtomicU64 = AtomicU64::new(0);
static PREFILL_LM_HEAD_MMQ_SKIP_ROWS: AtomicU64 = AtomicU64::new(0);
static PREFILL_NEXT_ID_CALLS: AtomicU64 = AtomicU64::new(0);
static PREFILL_NEXT_ID_ROWS: AtomicU64 = AtomicU64::new(0);
static DECODE_NEXT_ID_CALLS: AtomicU64 = AtomicU64::new(0);
static DECODE_NEXT_ID_ROWS: AtomicU64 = AtomicU64::new(0);
static FUSED_GLU_DECODE_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_GLU_DECODE_ROWS: AtomicU64 = AtomicU64::new(0);
static FUSED_ACT_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_ACT_ROWS: AtomicU64 = AtomicU64::new(0);
static FUSED_QKV_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_QKV_ROWS: AtomicU64 = AtomicU64::new(0);
static MMQ_FUSED_QKV_CALLS: AtomicU64 = AtomicU64::new(0);
static MMQ_FUSED_QKV_ROWS: AtomicU64 = AtomicU64::new(0);
static Q8_PREFILL_LM_HEAD_MMQ: AtomicBool = AtomicBool::new(true);
static Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_BYTES: AtomicUsize =
    AtomicUsize::new(DEFAULT_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_BYTES);

#[derive(Clone, Copy, Debug, Default)]
pub struct FastPathStats {
    pub plain_prefill_calls: u64,
    pub plain_decode_calls: u64,
    pub plain_prefill_rows: u64,
    pub plain_decode_rows: u64,
    pub plain_audio_calls: u64,
    pub plain_audio_rows: u64,
    pub plain_lm_head_calls: u64,
    pub plain_lm_head_rows: u64,
    pub mmq_prefill_calls: u64,
    pub mmq_prefill_rows: u64,
    pub mmq_decode_calls: u64,
    pub mmq_decode_rows: u64,
    pub mmq_lm_head_calls: u64,
    pub mmq_lm_head_rows: u64,
    pub prefill_lm_head_mmq_skip_calls: u64,
    pub prefill_lm_head_mmq_skip_rows: u64,
    pub prefill_next_id_calls: u64,
    pub prefill_next_id_rows: u64,
    pub decode_next_id_calls: u64,
    pub decode_next_id_rows: u64,
    pub fused_glu_decode_calls: u64,
    pub fused_glu_decode_rows: u64,
    pub fused_act_calls: u64,
    pub fused_act_rows: u64,
    pub fused_qkv_calls: u64,
    pub fused_qkv_rows: u64,
    pub mmq_fused_qkv_calls: u64,
    pub mmq_fused_qkv_rows: u64,
}

impl FastPathStats {
    pub fn saturating_sub(self, baseline: Self) -> Self {
        Self {
            plain_prefill_calls: self
                .plain_prefill_calls
                .saturating_sub(baseline.plain_prefill_calls),
            plain_decode_calls: self
                .plain_decode_calls
                .saturating_sub(baseline.plain_decode_calls),
            plain_prefill_rows: self
                .plain_prefill_rows
                .saturating_sub(baseline.plain_prefill_rows),
            plain_decode_rows: self
                .plain_decode_rows
                .saturating_sub(baseline.plain_decode_rows),
            plain_audio_calls: self
                .plain_audio_calls
                .saturating_sub(baseline.plain_audio_calls),
            plain_audio_rows: self
                .plain_audio_rows
                .saturating_sub(baseline.plain_audio_rows),
            plain_lm_head_calls: self
                .plain_lm_head_calls
                .saturating_sub(baseline.plain_lm_head_calls),
            plain_lm_head_rows: self
                .plain_lm_head_rows
                .saturating_sub(baseline.plain_lm_head_rows),
            mmq_prefill_calls: self
                .mmq_prefill_calls
                .saturating_sub(baseline.mmq_prefill_calls),
            mmq_prefill_rows: self
                .mmq_prefill_rows
                .saturating_sub(baseline.mmq_prefill_rows),
            mmq_decode_calls: self
                .mmq_decode_calls
                .saturating_sub(baseline.mmq_decode_calls),
            mmq_decode_rows: self
                .mmq_decode_rows
                .saturating_sub(baseline.mmq_decode_rows),
            mmq_lm_head_calls: self
                .mmq_lm_head_calls
                .saturating_sub(baseline.mmq_lm_head_calls),
            mmq_lm_head_rows: self
                .mmq_lm_head_rows
                .saturating_sub(baseline.mmq_lm_head_rows),
            prefill_lm_head_mmq_skip_calls: self
                .prefill_lm_head_mmq_skip_calls
                .saturating_sub(baseline.prefill_lm_head_mmq_skip_calls),
            prefill_lm_head_mmq_skip_rows: self
                .prefill_lm_head_mmq_skip_rows
                .saturating_sub(baseline.prefill_lm_head_mmq_skip_rows),
            prefill_next_id_calls: self
                .prefill_next_id_calls
                .saturating_sub(baseline.prefill_next_id_calls),
            prefill_next_id_rows: self
                .prefill_next_id_rows
                .saturating_sub(baseline.prefill_next_id_rows),
            decode_next_id_calls: self
                .decode_next_id_calls
                .saturating_sub(baseline.decode_next_id_calls),
            decode_next_id_rows: self
                .decode_next_id_rows
                .saturating_sub(baseline.decode_next_id_rows),
            fused_glu_decode_calls: self
                .fused_glu_decode_calls
                .saturating_sub(baseline.fused_glu_decode_calls),
            fused_glu_decode_rows: self
                .fused_glu_decode_rows
                .saturating_sub(baseline.fused_glu_decode_rows),
            fused_act_calls: self
                .fused_act_calls
                .saturating_sub(baseline.fused_act_calls),
            fused_act_rows: self.fused_act_rows.saturating_sub(baseline.fused_act_rows),
            fused_qkv_calls: self
                .fused_qkv_calls
                .saturating_sub(baseline.fused_qkv_calls),
            fused_qkv_rows: self.fused_qkv_rows.saturating_sub(baseline.fused_qkv_rows),
            mmq_fused_qkv_calls: self
                .mmq_fused_qkv_calls
                .saturating_sub(baseline.mmq_fused_qkv_calls),
            mmq_fused_qkv_rows: self
                .mmq_fused_qkv_rows
                .saturating_sub(baseline.mmq_fused_qkv_rows),
        }
    }
}

type FusedGluLauncher = unsafe extern "C" fn(
    vx_gate: *const std::ffi::c_void,
    vx_up: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    activation: i32,
    stream: *mut std::ffi::c_void,
);
type FusedActLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    bias: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

type FusedQkvLauncher = unsafe extern "C" fn(
    vx_q: *const std::ffi::c_void,
    vx_k: *const std::ffi::c_void,
    vx_v: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    q_bias: *const std::ffi::c_void,
    k_bias: *const std::ffi::c_void,
    v_bias: *const std::ffi::c_void,
    q_dst: *mut std::ffi::c_void,
    k_dst: *mut std::ffi::c_void,
    v_dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_q: i32,
    nrows_k: i32,
    nrows_v: i32,
    stride_col_y: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

unsafe extern "C" {
    fn launch_mmvq_gguf_q8_0_bf16_plain(
        vx: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f16_plain(
        vx: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f32_plain(
        vx: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_bf16_fused_glu(
        vx_gate: *const std::ffi::c_void,
        vx_up: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f16_fused_glu(
        vx_gate: *const std::ffi::c_void,
        vx_up: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f32_fused_glu(
        vx_gate: *const std::ffi::c_void,
        vx_up: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_bf16_fused_act(
        vx: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        bias: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        activation: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_bf16_fused_qkv(
        vx_q: *const std::ffi::c_void,
        vx_k: *const std::ffi::c_void,
        vx_v: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        q_bias: *const std::ffi::c_void,
        k_bias: *const std::ffi::c_void,
        v_bias: *const std::ffi::c_void,
        q_dst: *mut std::ffi::c_void,
        k_dst: *mut std::ffi::c_void,
        v_dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_q: i32,
        nrows_k: i32,
        nrows_v: i32,
        stride_col_y: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f16_fused_qkv(
        vx_q: *const std::ffi::c_void,
        vx_k: *const std::ffi::c_void,
        vx_v: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        q_bias: *const std::ffi::c_void,
        k_bias: *const std::ffi::c_void,
        v_bias: *const std::ffi::c_void,
        q_dst: *mut std::ffi::c_void,
        k_dst: *mut std::ffi::c_void,
        v_dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_q: i32,
        nrows_k: i32,
        nrows_v: i32,
        stride_col_y: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_q8_0_f32_fused_qkv(
        vx_q: *const std::ffi::c_void,
        vx_k: *const std::ffi::c_void,
        vx_v: *const std::ffi::c_void,
        vy: *const std::ffi::c_void,
        q_bias: *const std::ffi::c_void,
        k_bias: *const std::ffi::c_void,
        v_bias: *const std::ffi::c_void,
        q_dst: *mut std::ffi::c_void,
        k_dst: *mut std::ffi::c_void,
        v_dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_q: i32,
        nrows_k: i32,
        nrows_v: i32,
        stride_col_y: i32,
        b_size: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_quantize_q8_1_bf16(
        x: *const std::ffi::c_void,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_quantize_q8_1_f16(
        x: *const std::ffi::c_void,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mmvq_gguf_quantize_q8_1_f32(
        x: *const std::ffi::c_void,
        vy: *mut std::ffi::c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q8_0_lm_head_candidates(
        weight: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        batch_size: i32,
        chunk_size: i32,
        chunks: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q8_0_lm_head_candidates_reduce(
        candidates: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        batch_size: i32,
        chunks: i32,
        stream: *mut std::ffi::c_void,
    );
    fn vasr_launch_mmq_quantize_q8_1_D4(
        x: *const std::ffi::c_void,
        ids: *const i32,
        vy: *mut std::ffi::c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut std::ffi::c_void,
    );
    fn vasr_launch_mmq_gguf_q8_0(
        tmp_fixup: *mut std::ffi::c_void,
        x: *const std::ffi::c_void,
        y: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        type_dst: i32,
        stream: *mut std::ffi::c_void,
    );
}

struct ScratchSlot {
    device_id: DeviceId,
    bytes: usize,
    slice: CudaSlice<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Vec<ScratchSlot>> = const { RefCell::new(Vec::new()) };
}

fn with_scratch<R>(
    dev: &CudaDevice,
    bytes: usize,
    stream: &CudaStream,
    f: impl FnOnce(*mut std::ffi::c_void) -> Result<R>,
) -> Result<R> {
    let bytes = bytes.max(1);
    SCRATCH.with(|scratch| {
        let mut slots = scratch.borrow_mut();
        let device_id = dev.id();
        let slot_idx = if let Some(idx) = slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.device_id == device_id && slot.bytes >= bytes)
            .min_by_key(|(_, slot)| slot.bytes)
            .map(|(idx, _)| idx)
        {
            idx
        } else {
            // CUDA graphs capture raw workspace pointers. Keep older slots
            // alive instead of replacing them when a later request needs a
            // larger scratch buffer.
            let slice = unsafe { dev.alloc::<u8>(bytes)? };
            slots.push(ScratchSlot {
                device_id,
                bytes,
                slice,
            });
            slots.len() - 1
        };
        let (ptr, guard) = slots[slot_idx].slice.device_ptr_mut(stream);
        let result = f(ptr as *mut std::ffi::c_void);
        drop(guard);
        result
    })
}

fn with_temp_scratch<R>(
    dev: &CudaDevice,
    bytes: usize,
    stream: &CudaStream,
    f: impl FnOnce(*mut std::ffi::c_void) -> Result<R>,
) -> Result<R> {
    let mut slice = unsafe { dev.alloc::<u8>(bytes.max(1))? };
    let (ptr, guard) = slice.device_ptr_mut(stream);
    let result = f(ptr as *mut std::ffi::c_void);
    drop(guard);
    result
}

fn with_plain_scratch<R>(
    kind: PlainKind,
    dev: &CudaDevice,
    bytes: usize,
    stream: &CudaStream,
    f: impl FnOnce(*mut std::ffi::c_void) -> Result<R>,
) -> Result<R> {
    if kind.caches_scratch() {
        with_scratch(dev, bytes, stream, f)
    } else {
        with_temp_scratch(dev, bytes, stream, f)
    }
}

#[derive(Clone, Copy)]
struct DeviceInfo {
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
}

static DEVICE_INFO: OnceLock<Mutex<HashMap<DeviceId, DeviceInfo>>> = OnceLock::new();

fn get_device_info(dev: &CudaDevice) -> DeviceInfo {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};

    let map = DEVICE_INFO.get_or_init(|| Mutex::new(HashMap::new()));
    let key = dev.id();
    let mut guard = map
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(info) = guard.get(&key) {
        return *info;
    }

    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(8);
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    let nsm = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .unwrap_or(1);
    let smpbo = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
    }
    .unwrap_or(49152);
    let warp_size = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        )
    }
    .unwrap_or(32);
    let info = DeviceInfo {
        cc: major * 100 + minor * 10,
        nsm,
        smpbo: smpbo as i64,
        warp_size,
    };
    guard.insert(key, info);
    info
}

pub fn trim_cuda_workspace() {}

pub fn reset_fast_path_stats() {
    PLAIN_PREFILL_CALLS.store(0, Ordering::Relaxed);
    PLAIN_DECODE_CALLS.store(0, Ordering::Relaxed);
    PLAIN_PREFILL_ROWS.store(0, Ordering::Relaxed);
    PLAIN_DECODE_ROWS.store(0, Ordering::Relaxed);
    PLAIN_AUDIO_CALLS.store(0, Ordering::Relaxed);
    PLAIN_AUDIO_ROWS.store(0, Ordering::Relaxed);
    PLAIN_LM_HEAD_CALLS.store(0, Ordering::Relaxed);
    PLAIN_LM_HEAD_ROWS.store(0, Ordering::Relaxed);
    MMQ_PREFILL_CALLS.store(0, Ordering::Relaxed);
    MMQ_PREFILL_ROWS.store(0, Ordering::Relaxed);
    MMQ_DECODE_CALLS.store(0, Ordering::Relaxed);
    MMQ_DECODE_ROWS.store(0, Ordering::Relaxed);
    MMQ_LM_HEAD_CALLS.store(0, Ordering::Relaxed);
    MMQ_LM_HEAD_ROWS.store(0, Ordering::Relaxed);
    PREFILL_LM_HEAD_MMQ_SKIP_CALLS.store(0, Ordering::Relaxed);
    PREFILL_LM_HEAD_MMQ_SKIP_ROWS.store(0, Ordering::Relaxed);
    PREFILL_NEXT_ID_CALLS.store(0, Ordering::Relaxed);
    PREFILL_NEXT_ID_ROWS.store(0, Ordering::Relaxed);
    DECODE_NEXT_ID_CALLS.store(0, Ordering::Relaxed);
    DECODE_NEXT_ID_ROWS.store(0, Ordering::Relaxed);
    FUSED_GLU_DECODE_CALLS.store(0, Ordering::Relaxed);
    FUSED_GLU_DECODE_ROWS.store(0, Ordering::Relaxed);
    FUSED_ACT_CALLS.store(0, Ordering::Relaxed);
    FUSED_ACT_ROWS.store(0, Ordering::Relaxed);
    FUSED_QKV_CALLS.store(0, Ordering::Relaxed);
    FUSED_QKV_ROWS.store(0, Ordering::Relaxed);
    MMQ_FUSED_QKV_CALLS.store(0, Ordering::Relaxed);
    MMQ_FUSED_QKV_ROWS.store(0, Ordering::Relaxed);
}

pub fn fast_path_stats() -> FastPathStats {
    FastPathStats {
        plain_prefill_calls: PLAIN_PREFILL_CALLS.load(Ordering::Relaxed),
        plain_decode_calls: PLAIN_DECODE_CALLS.load(Ordering::Relaxed),
        plain_prefill_rows: PLAIN_PREFILL_ROWS.load(Ordering::Relaxed),
        plain_decode_rows: PLAIN_DECODE_ROWS.load(Ordering::Relaxed),
        plain_audio_calls: PLAIN_AUDIO_CALLS.load(Ordering::Relaxed),
        plain_audio_rows: PLAIN_AUDIO_ROWS.load(Ordering::Relaxed),
        plain_lm_head_calls: PLAIN_LM_HEAD_CALLS.load(Ordering::Relaxed),
        plain_lm_head_rows: PLAIN_LM_HEAD_ROWS.load(Ordering::Relaxed),
        mmq_prefill_calls: MMQ_PREFILL_CALLS.load(Ordering::Relaxed),
        mmq_prefill_rows: MMQ_PREFILL_ROWS.load(Ordering::Relaxed),
        mmq_decode_calls: MMQ_DECODE_CALLS.load(Ordering::Relaxed),
        mmq_decode_rows: MMQ_DECODE_ROWS.load(Ordering::Relaxed),
        mmq_lm_head_calls: MMQ_LM_HEAD_CALLS.load(Ordering::Relaxed),
        mmq_lm_head_rows: MMQ_LM_HEAD_ROWS.load(Ordering::Relaxed),
        prefill_lm_head_mmq_skip_calls: PREFILL_LM_HEAD_MMQ_SKIP_CALLS.load(Ordering::Relaxed),
        prefill_lm_head_mmq_skip_rows: PREFILL_LM_HEAD_MMQ_SKIP_ROWS.load(Ordering::Relaxed),
        prefill_next_id_calls: PREFILL_NEXT_ID_CALLS.load(Ordering::Relaxed),
        prefill_next_id_rows: PREFILL_NEXT_ID_ROWS.load(Ordering::Relaxed),
        decode_next_id_calls: DECODE_NEXT_ID_CALLS.load(Ordering::Relaxed),
        decode_next_id_rows: DECODE_NEXT_ID_ROWS.load(Ordering::Relaxed),
        fused_glu_decode_calls: FUSED_GLU_DECODE_CALLS.load(Ordering::Relaxed),
        fused_glu_decode_rows: FUSED_GLU_DECODE_ROWS.load(Ordering::Relaxed),
        fused_act_calls: FUSED_ACT_CALLS.load(Ordering::Relaxed),
        fused_act_rows: FUSED_ACT_ROWS.load(Ordering::Relaxed),
        fused_qkv_calls: FUSED_QKV_CALLS.load(Ordering::Relaxed),
        fused_qkv_rows: FUSED_QKV_ROWS.load(Ordering::Relaxed),
        mmq_fused_qkv_calls: MMQ_FUSED_QKV_CALLS.load(Ordering::Relaxed),
        mmq_fused_qkv_rows: MMQ_FUSED_QKV_ROWS.load(Ordering::Relaxed),
    }
}

pub fn set_prefill_lm_head_mmq_enabled(enabled: bool) {
    Q8_PREFILL_LM_HEAD_MMQ.store(enabled, Ordering::Relaxed);
}

pub fn set_prefill_lm_head_mmq_max_logits_bytes(bytes: usize) {
    Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_BYTES.store(bytes, Ordering::Relaxed);
}

fn prefill_lm_head_mmq_enabled() -> bool {
    Q8_PREFILL_LM_HEAD_MMQ.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrefillLmHeadMmqDecision {
    Allowed,
    Disabled,
    OverLimit,
    Unsupported,
}

fn prefill_lm_head_mmq_decision(
    b_size: usize,
    vocab: usize,
    dtype: DType,
) -> PrefillLmHeadMmqDecision {
    if !prefill_lm_head_mmq_enabled() {
        return PrefillLmHeadMmqDecision::Disabled;
    }
    let Some(elem_bytes) = dtype_size_bytes(dtype) else {
        return PrefillLmHeadMmqDecision::Unsupported;
    };
    let Some(logits_bytes) = b_size
        .checked_mul(vocab)
        .and_then(|values| values.checked_mul(elem_bytes))
    else {
        return PrefillLmHeadMmqDecision::OverLimit;
    };
    let max_bytes = Q8_PREFILL_LM_HEAD_MMQ_MAX_LOGITS_BYTES.load(Ordering::Relaxed);
    if max_bytes == 0 || logits_bytes <= max_bytes {
        PrefillLmHeadMmqDecision::Allowed
    } else {
        PrefillLmHeadMmqDecision::OverLimit
    }
}

fn dtype_size_bytes(dtype: DType) -> Option<usize> {
    match dtype {
        DType::BF16 | DType::F16 => Some(2),
        DType::F32 => Some(4),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PlainKind {
    Audio,
    Prefill,
    Decode,
    LmHead,
    LmHeadPrefill,
}

impl PlainKind {
    fn caches_scratch(self) -> bool {
        matches!(self, Self::Decode | Self::LmHead)
    }
}

fn record_plain_fast_path(kind: PlainKind, rows: usize) {
    let rows = rows.min(u64::MAX as usize) as u64;
    match kind {
        PlainKind::Audio => {
            PLAIN_AUDIO_CALLS.fetch_add(1, Ordering::Relaxed);
            PLAIN_AUDIO_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
        PlainKind::Prefill => {
            PLAIN_PREFILL_CALLS.fetch_add(1, Ordering::Relaxed);
            PLAIN_PREFILL_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
        PlainKind::Decode => {
            PLAIN_DECODE_CALLS.fetch_add(1, Ordering::Relaxed);
            PLAIN_DECODE_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
        PlainKind::LmHead | PlainKind::LmHeadPrefill => {
            PLAIN_LM_HEAD_CALLS.fetch_add(1, Ordering::Relaxed);
            PLAIN_LM_HEAD_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
    }
}

fn record_mmq_fast_path(kind: PlainKind, rows: usize) {
    let rows = rows.min(u64::MAX as usize) as u64;
    match kind {
        PlainKind::LmHead | PlainKind::LmHeadPrefill => {
            MMQ_LM_HEAD_CALLS.fetch_add(1, Ordering::Relaxed);
            MMQ_LM_HEAD_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
        PlainKind::Decode => {
            MMQ_DECODE_CALLS.fetch_add(1, Ordering::Relaxed);
            MMQ_DECODE_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
        _ => {
            MMQ_PREFILL_CALLS.fetch_add(1, Ordering::Relaxed);
            MMQ_PREFILL_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
    }
}

fn record_fused_glu_fast_path(rows: usize) {
    let rows = rows.min(u64::MAX as usize) as u64;
    FUSED_GLU_DECODE_CALLS.fetch_add(1, Ordering::Relaxed);
    FUSED_GLU_DECODE_ROWS.fetch_add(rows, Ordering::Relaxed);
}

fn record_fused_act_fast_path(rows: usize) {
    let rows = rows.min(u64::MAX as usize) as u64;
    FUSED_ACT_CALLS.fetch_add(1, Ordering::Relaxed);
    FUSED_ACT_ROWS.fetch_add(rows, Ordering::Relaxed);
}

fn record_fused_qkv_fast_path(rows: usize) {
    let rows = rows.min(u64::MAX as usize) as u64;
    FUSED_QKV_CALLS.fetch_add(1, Ordering::Relaxed);
    FUSED_QKV_ROWS.fetch_add(rows, Ordering::Relaxed);
}

fn record_mmq_fused_qkv_fast_path(rows: usize) {
    let rows = rows.min(u64::MAX as usize) as u64;
    MMQ_FUSED_QKV_CALLS.fetch_add(1, Ordering::Relaxed);
    MMQ_FUSED_QKV_ROWS.fetch_add(rows, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug)]
pub enum NextIdKind {
    Prefill,
    Decode,
}

fn record_next_id_fast_path(kind: NextIdKind, rows: usize) {
    let rows = rows.min(u64::MAX as usize) as u64;
    match kind {
        NextIdKind::Prefill => {
            PREFILL_NEXT_ID_CALLS.fetch_add(1, Ordering::Relaxed);
            PREFILL_NEXT_ID_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
        NextIdKind::Decode => {
            DECODE_NEXT_ID_CALLS.fetch_add(1, Ordering::Relaxed);
            DECODE_NEXT_ID_ROWS.fetch_add(rows, Ordering::Relaxed);
        }
    }
}

fn record_prefill_lm_head_mmq_skip(rows: usize) {
    let rows = rows.min(u64::MAX as usize) as u64;
    PREFILL_LM_HEAD_MMQ_SKIP_CALLS.fetch_add(1, Ordering::Relaxed);
    PREFILL_LM_HEAD_MMQ_SKIP_ROWS.fetch_add(rows, Ordering::Relaxed);
}

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

fn output_shape(xs: &Tensor, nrows: usize) -> Shape {
    let mut out_dims = xs.dims().to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = nrows;
    Shape::from(out_dims)
}

fn dtype_to_ggml_type(dtype: DType) -> Option<i32> {
    match dtype {
        DType::F32 => Some(0),
        DType::F16 => Some(1),
        DType::BF16 => Some(30),
        _ => None,
    }
}

fn launcher(dtype: DType) -> Result<PlainLauncher> {
    match dtype {
        DType::BF16 => Ok(launch_mmvq_gguf_q8_0_bf16_plain),
        DType::F16 => Ok(launch_mmvq_gguf_q8_0_f16_plain),
        DType::F32 => Ok(launch_mmvq_gguf_q8_0_f32_plain),
        other => candle_core::bail!("q8_mmvq: unsupported activation dtype {other:?}"),
    }
}

fn try_plain_mmq_q8_0(w: &QTensor, xs: &Tensor, kind: PlainKind) -> Result<Option<Tensor>> {
    if w.dtype() != GgmlDType::Q8_0 || !w.device().is_cuda() {
        return Ok(None);
    }
    let input_type = match dtype_to_ggml_type(xs.dtype()) {
        Some(t) => t,
        None => return Ok(None),
    };
    let Device::Cuda(dev) = w.device() else {
        return Ok(None);
    };
    let (nrows, ncols) = w.shape().dims2()?;
    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        _ => return Ok(None),
    };
    if b_size <= 8 || k != ncols || k % 32 != 0 {
        return Ok(None);
    }

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        return Ok(None);
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * Q8_1_BLOCK_SIZE);
    let blocks_per_row = k_padded / (4 * Q8_1_BLOCK_SIZE);
    let activation_bytes = b_size * blocks_per_row * Q8_1_MMQ_TYPE_SIZE;
    let activation_extra_bytes = 128 * Q8_1_MMQ_TYPE_SIZE;
    const MMQ_X_MAX: usize = 128;
    const MMQ_Y_MAX: usize = 128;
    const MAX_SMS: usize = 256;
    let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
    let scratch_bytes = activation_bytes + activation_extra_bytes + fixup_bytes;
    let (weight_ptr, _weight_guard) = w.device_ptr_with_guard(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;
    let di = get_device_info(&dev);
    let stride_row_x = (k / 32) as i64;
    let stride_col_dst = nrows as i64;

    macro_rules! run {
        ($ty:ty, $slice_fn:ident) => {{
            let slice = xs_cuda.$slice_fn::<$ty>()?;
            let mut out = unsafe { dev.alloc::<$ty>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            with_plain_scratch(kind, &dev, scratch_bytes, &stream, |scratch_ptr| {
                let fixup_ptr = unsafe {
                    (scratch_ptr as *mut u8).add(activation_bytes + activation_extra_bytes)
                        as *mut std::ffi::c_void
                };
                unsafe {
                    vasr_launch_mmq_quantize_q8_1_D4(
                        xs_ptr as *const std::ffi::c_void,
                        std::ptr::null(),
                        scratch_ptr,
                        input_type,
                        k as i64,
                        k as i64,
                        0,
                        0,
                        k_padded as i64,
                        b_size as i64,
                        1,
                        1,
                        stream_ptr,
                    );
                    vasr_launch_mmq_gguf_q8_0(
                        fixup_ptr,
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i64,
                        nrows as i64,
                        b_size as i64,
                        stride_row_x,
                        stride_col_dst,
                        di.cc,
                        di.nsm,
                        di.smpbo,
                        di.warp_size,
                        input_type,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(out_guard);
            drop(xs_guard);
            record_mmq_fast_path(kind, b_size);
            Ok(Some(tensor_from_cuda_slice(
                out,
                &dev,
                output_shape(&xs, nrows),
            )))
        }};
    }

    match xs.dtype() {
        DType::BF16 => run!(half::bf16, as_cuda_slice),
        DType::F16 => run!(half::f16, as_cuda_slice),
        DType::F32 => run!(f32, as_cuda_slice),
        _ => Ok(None),
    }
}

fn fused_glu_launcher(dtype: DType) -> Result<FusedGluLauncher> {
    match dtype {
        DType::BF16 => Ok(launch_mmvq_gguf_q8_0_bf16_fused_glu),
        DType::F16 => Ok(launch_mmvq_gguf_q8_0_f16_fused_glu),
        DType::F32 => Ok(launch_mmvq_gguf_q8_0_f32_fused_glu),
        other => candle_core::bail!("q8_mmvq fused_glu: unsupported activation dtype {other:?}"),
    }
}

fn fused_act_launcher(dtype: DType) -> Result<FusedActLauncher> {
    match dtype {
        DType::BF16 => Ok(launch_mmvq_gguf_q8_0_bf16_fused_act),
        other => candle_core::bail!("q8_mmvq fused_act: unsupported activation dtype {other:?}"),
    }
}

fn fused_qkv_launcher(dtype: DType) -> Result<FusedQkvLauncher> {
    match dtype {
        DType::BF16 => Ok(launch_mmvq_gguf_q8_0_bf16_fused_qkv),
        DType::F16 => Ok(launch_mmvq_gguf_q8_0_f16_fused_qkv),
        DType::F32 => Ok(launch_mmvq_gguf_q8_0_f32_fused_qkv),
        other => candle_core::bail!("q8_mmvq fused_qkv: unsupported activation dtype {other:?}"),
    }
}

fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    stream: &'a CudaStream,
    lo: usize,
) -> (u64, SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

fn slice_ptr_mut_on_stream<'a, T: DeviceRepr>(
    v: &'a mut CudaSlice<T>,
    stream: &'a CudaStream,
    lo: usize,
) -> (u64, SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

fn tensor_from_cuda_slice<T: DeviceRepr + CudaDType>(
    slice: CudaSlice<T>,
    dev: &CudaDevice,
    shape: Shape,
) -> Tensor {
    Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
        shape,
        BackpropOp::none(),
        false,
    )
}

fn prepare_fused_qkv_bias(
    bias: Option<&Tensor>,
    rows: usize,
    dtype: DType,
    device: &Device,
    name: &str,
) -> Result<Option<Tensor>> {
    let Some(bias) = bias else {
        return Ok(None);
    };
    if bias.elem_count() != rows {
        candle_core::bail!(
            "q8_mmvq fused_qkv: {name} bias has {} elements, expected {rows}",
            bias.elem_count()
        );
    }
    bias.to_dtype(dtype)?
        .to_device(device)?
        .contiguous()
        .map(Some)
}

fn add_projection_bias(
    ys: Tensor,
    bias: Option<&Tensor>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let Some(bias) = bias else {
        return Ok(ys);
    };
    ys.broadcast_add(&bias.to_dtype(dtype)?.to_device(device)?)
}

fn can_run_with_max_batch(w: &QTensor, xs: &Tensor, max_batch: usize) -> bool {
    if w.dtype() != GgmlDType::Q8_0 || !w.device().is_cuda() {
        return false;
    }
    if !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return false;
    }
    let flat_batch = xs.dims()[..xs.dims().len().saturating_sub(1)]
        .iter()
        .product::<usize>();
    flat_batch > 0 && flat_batch <= max_batch
}

pub fn can_run_plain(w: &QTensor, xs: &Tensor) -> bool {
    can_run_with_max_batch(w, xs, i32::MAX as usize)
}

pub fn can_run_decode(w: &QTensor, xs: &Tensor) -> bool {
    can_run_with_max_batch(w, xs, i32::MAX as usize)
}

pub fn lm_head_next_ids_q8_0(
    w: &QTensor,
    xs: &Tensor,
    chunk_size: usize,
    kind: NextIdKind,
) -> Result<Option<Tensor>> {
    lm_head_q8_0(w, xs, chunk_size, true, kind)
}

fn lm_head_q8_0(
    w: &QTensor,
    xs: &Tensor,
    chunk_size: usize,
    reduce_next_ids: bool,
    kind: NextIdKind,
) -> Result<Option<Tensor>> {
    if chunk_size == 0 || w.dtype() != GgmlDType::Q8_0 || !w.device().is_cuda() {
        return Ok(None);
    }
    if !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }
    let Device::Cuda(dev) = w.device() else {
        return Ok(None);
    };
    let (nrows, ncols) = w.shape().dims2()?;
    let (b_size, k) = match xs.dims() {
        [b, 1, k] => (*b, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if b_size == 0 || k != ncols || k % Q8_1_BLOCK_SIZE != 0 {
        return Ok(None);
    }
    if reduce_next_ids && b_size > 8 {
        if matches!(kind, NextIdKind::Decode) {
            if let Some(next_ids) = try_lm_head_next_ids_mmq_q8_0(w, xs, b_size, kind)? {
                record_next_id_fast_path(kind, b_size);
                return Ok(Some(next_ids));
            }
            candle_core::bail!(
                "q8 lm_head decode next-id requires MMQ for batch={b_size}; refusing to fall back to chunked candidates"
            );
        } else {
            match prefill_lm_head_mmq_decision(b_size, nrows, xs.dtype()) {
                PrefillLmHeadMmqDecision::Allowed => {
                    if let Some(next_ids) = try_lm_head_next_ids_mmq_q8_0(w, xs, b_size, kind)? {
                        record_next_id_fast_path(kind, b_size);
                        return Ok(Some(next_ids));
                    }
                }
                PrefillLmHeadMmqDecision::OverLimit => {
                    record_prefill_lm_head_mmq_skip(b_size);
                }
                PrefillLmHeadMmqDecision::Disabled | PrefillLmHeadMmqDecision::Unsupported => {}
            }
        }
    }
    let chunks = nrows.div_ceil(chunk_size);
    if chunks == 0
        || b_size > i32::MAX as usize
        || nrows > i32::MAX as usize
        || ncols > i32::MAX as usize
        || chunk_size > i32::MAX as usize
        || chunks > i32::MAX as usize
    {
        return Ok(None);
    }

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        return Ok(None);
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;
    let (weight_ptr, _weight_guard) = w.device_ptr_with_guard(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;

    macro_rules! run {
        ($ty:ty, $slice_fn:ident, $quantize:ident) => {{
            let slice = xs_cuda.$slice_fn::<$ty>()?;
            let candidates_len = 2 * b_size * chunks;
            let storage_len = if reduce_next_ids {
                candidates_len + b_size
            } else {
                candidates_len
            };
            let mut storage = unsafe { dev.alloc::<f32>(storage_len)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (storage_ptr, storage_guard) = slice_ptr_mut_on_stream(&mut storage, &stream, 0);
            let candidates_ptr = storage_ptr as *mut std::ffi::c_void;
            let next_ids_ptr = if reduce_next_ids {
                unsafe { (candidates_ptr as *mut f32).add(candidates_len) as *mut std::ffi::c_void }
            } else {
                std::ptr::null_mut()
            };
            let plain_kind = match kind {
                NextIdKind::Prefill => PlainKind::LmHeadPrefill,
                NextIdKind::Decode => PlainKind::LmHead,
            };
            with_plain_scratch(plain_kind, &dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    $quantize(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch_q8_0_lm_head_candidates(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        candidates_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        num_blocks_per_row as i32,
                        b_size as i32,
                        chunk_size as i32,
                        chunks as i32,
                        stream_ptr,
                    );
                    if reduce_next_ids {
                        launch_q8_0_lm_head_candidates_reduce(
                            candidates_ptr as *const std::ffi::c_void,
                            next_ids_ptr,
                            b_size as i32,
                            chunks as i32,
                            stream_ptr,
                        );
                    }
                }
                Ok(())
            })?;
            drop(storage_guard);
            drop(xs_guard);
            record_plain_fast_path(plain_kind, b_size);
            record_next_id_fast_path(kind, b_size);
            if reduce_next_ids {
                let full = tensor_from_cuda_slice(storage, &dev, Shape::from((storage_len,)));
                Ok(Some(full.narrow(0, candidates_len, b_size)?))
            } else {
                Ok(Some(tensor_from_cuda_slice(
                    storage,
                    &dev,
                    Shape::from((2usize, b_size, chunks)),
                )))
            }
        }};
    }

    match xs.dtype() {
        DType::BF16 => run!(
            half::bf16,
            as_cuda_slice,
            launch_mmvq_gguf_quantize_q8_1_bf16
        ),
        DType::F16 => run!(half::f16, as_cuda_slice, launch_mmvq_gguf_quantize_q8_1_f16),
        DType::F32 => run!(f32, as_cuda_slice, launch_mmvq_gguf_quantize_q8_1_f32),
        _ => Ok(None),
    }
}

fn try_lm_head_next_ids_mmq_q8_0(
    w: &QTensor,
    xs: &Tensor,
    b_size: usize,
    kind: NextIdKind,
) -> Result<Option<Tensor>> {
    let plain_kind = match kind {
        NextIdKind::Prefill => PlainKind::LmHeadPrefill,
        NextIdKind::Decode => PlainKind::LmHead,
    };
    let Some(logits) = try_plain_mmq_q8_0(w, xs, plain_kind)? else {
        return Ok(None);
    };
    let logits = match logits.dims() {
        [_, 1, _] => logits.squeeze(1)?,
        [_, _] => logits,
        _ => return Ok(None),
    };
    let next_ids = logits.argmax(1usize)?.to_dtype(DType::F32)?;
    if next_ids.elem_count() != b_size {
        candle_core::bail!(
            "q8 lm_head MMQ next-id output length mismatch: got {} expected {b_size}",
            next_ids.elem_count()
        );
    }
    Ok(Some(next_ids))
}

fn can_run_fused_glu_with_max_batch(
    gate_w: &QTensor,
    up_w: &QTensor,
    xs: &Tensor,
    max_batch: usize,
) -> bool {
    gate_w.dtype() == GgmlDType::Q8_0
        && up_w.dtype() == GgmlDType::Q8_0
        && gate_w.device().is_cuda()
        && up_w.device().is_cuda()
        && gate_w.device().same_device(&up_w.device())
        && gate_w.shape() == up_w.shape()
        && matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32)
        && {
            let flat_batch = xs.dims()[..xs.dims().len().saturating_sub(1)]
                .iter()
                .product::<usize>();
            flat_batch > 0 && flat_batch <= max_batch
        }
}

pub fn can_run_fused_glu_decode(gate_w: &QTensor, up_w: &QTensor, xs: &Tensor) -> bool {
    can_run_fused_glu_with_max_batch(gate_w, up_w, xs, Q8_MMVQ_MAX_BATCH)
}

pub fn can_run_fused_qkv(q_w: &QTensor, k_w: &QTensor, v_w: &QTensor, xs: &Tensor) -> bool {
    q_w.dtype() == GgmlDType::Q8_0
        && k_w.dtype() == GgmlDType::Q8_0
        && v_w.dtype() == GgmlDType::Q8_0
        && q_w.device().is_cuda()
        && k_w.device().is_cuda()
        && v_w.device().is_cuda()
        && q_w.device().same_device(&k_w.device())
        && q_w.device().same_device(&v_w.device())
        && matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32)
        && {
            let Ok((_, q_cols)) = q_w.shape().dims2() else {
                return false;
            };
            let Ok((_, k_cols)) = k_w.shape().dims2() else {
                return false;
            };
            let Ok((_, v_cols)) = v_w.shape().dims2() else {
                return false;
            };
            let dims = xs.dims();
            let flat_batch = dims[..dims.len().saturating_sub(1)]
                .iter()
                .product::<usize>();
            flat_batch > 0
                && dims.last().copied() == Some(q_cols)
                && q_cols == k_cols
                && q_cols == v_cols
        }
}

pub fn plain(w: &QTensor, xs: &Tensor, kind: PlainKind) -> Result<Tensor> {
    if !can_run_decode(w, xs) {
        candle_core::bail!("q8_mmvq: unsupported fast path input");
    }
    if let Some(out) = try_plain_mmq_q8_0(w, xs, kind)? {
        return Ok(out);
    }
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("q8_mmvq: weight must live on CUDA");
    };
    let (nrows, ncols) = w.shape().dims2()?;
    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("q8_mmvq: unexpected input rank {other:?}"),
    };
    if b_size > Q8_MMVQ_MAX_BATCH {
        candle_core::bail!(
            "q8_mmvq: large batch Q8_0 linear was not covered by MMQ path: kind={kind:?} weight=[{nrows},{ncols}] input_shape={:?} input_dtype={:?}",
            xs.dims(),
            xs.dtype()
        );
    }
    if k != ncols {
        candle_core::bail!("q8_mmvq: shape mismatch weight [{nrows}, {ncols}] input tail {k}");
    }
    record_plain_fast_path(kind, b_size);

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("q8_mmvq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;
    let (weight_ptr, _weight_guard) = w.device_ptr_with_guard(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;
    let launch = launcher(xs.dtype())?;

    match xs.dtype() {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            with_plain_scratch(kind, &dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch(
                        weight_ptr,
                        scratch_ptr,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        num_blocks_per_row as i32,
                        nrows as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(out_guard);
            drop(xs_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            with_plain_scratch(kind, &dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch(
                        weight_ptr,
                        scratch_ptr,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        num_blocks_per_row as i32,
                        nrows as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(out_guard);
            drop(xs_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            with_plain_scratch(kind, &dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch(
                        weight_ptr,
                        scratch_ptr,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        num_blocks_per_row as i32,
                        nrows as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(out_guard);
            drop(xs_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        _ => unreachable!(),
    }
}

pub fn fused_qkv(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: &QTensor,
    q_bias: Option<&Tensor>,
    k_bias: Option<&Tensor>,
    v_bias: Option<&Tensor>,
    xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    if !can_run_fused_qkv(q_w, k_w, v_w, xs) {
        candle_core::bail!("q8_mmvq fused_qkv: unsupported fast path input");
    }
    let Device::Cuda(dev) = q_w.device() else {
        candle_core::bail!("q8_mmvq fused_qkv: q weight must live on CUDA");
    };
    let (q_rows, q_cols) = q_w.shape().dims2()?;
    let (k_rows, k_cols) = k_w.shape().dims2()?;
    let (v_rows, v_cols) = v_w.shape().dims2()?;
    if q_cols != k_cols || q_cols != v_cols {
        candle_core::bail!(
            "q8_mmvq fused_qkv: q/k/v input dims mismatch {q_cols}/{k_cols}/{v_cols}"
        );
    }
    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("q8_mmvq fused_qkv: unexpected input rank {other:?}"),
    };
    if k != q_cols {
        candle_core::bail!("q8_mmvq fused_qkv: shape mismatch input tail {k} vs weight {q_cols}");
    }
    if b_size > Q8_MMVQ_MAX_BATCH {
        if let Some((q, k, v)) =
            try_fused_qkv_mmq_q8_0(q_w, k_w, v_w, q_bias, k_bias, v_bias, xs, b_size, k)?
        {
            record_fused_qkv_fast_path(b_size);
            return Ok((q, k, v));
        }
        return Ok((
            add_projection_bias(
                plain(q_w, xs, PlainKind::Decode)?,
                q_bias,
                xs.dtype(),
                xs.device(),
            )?,
            add_projection_bias(
                plain(k_w, xs, PlainKind::Decode)?,
                k_bias,
                xs.dtype(),
                xs.device(),
            )?,
            add_projection_bias(
                plain(v_w, xs, PlainKind::Decode)?,
                v_bias,
                xs.dtype(),
                xs.device(),
            )?,
        ));
    }
    record_fused_qkv_fast_path(b_size);

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let q_bias = prepare_fused_qkv_bias(q_bias, q_rows, xs.dtype(), xs.device(), "q")?;
    let k_bias = prepare_fused_qkv_bias(k_bias, k_rows, xs.dtype(), xs.device(), "k")?;
    let v_bias = prepare_fused_qkv_bias(v_bias, v_rows, xs.dtype(), xs.device(), "v")?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("q8_mmvq fused_qkv: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;
    let (q_ptr, _q_guard) = q_w.device_ptr_with_guard(&stream)?;
    let (k_ptr, _k_guard) = k_w.device_ptr_with_guard(&stream)?;
    let (v_ptr, _v_guard) = v_w.device_ptr_with_guard(&stream)?;
    let q_ptr = q_ptr as *const std::ffi::c_void;
    let k_ptr = k_ptr as *const std::ffi::c_void;
    let v_ptr = v_ptr as *const std::ffi::c_void;
    let launch = fused_qkv_launcher(xs.dtype())?;

    macro_rules! run {
        ($ty:ty, $slice_fn:ident, $quantize:ident) => {{
            let slice = xs_cuda.$slice_fn::<$ty>()?;
            let mut q_out = unsafe { dev.alloc::<$ty>(q_rows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<$ty>(k_rows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<$ty>(v_rows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (q_out_ptr, q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, &stream, 0);
            let (k_out_ptr, k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, &stream, 0);
            let (v_out_ptr, v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, &stream, 0);
            let q_bias_storage_layout = q_bias.as_ref().map(|bias| bias.storage_and_layout());
            let k_bias_storage_layout = k_bias.as_ref().map(|bias| bias.storage_and_layout());
            let v_bias_storage_layout = v_bias.as_ref().map(|bias| bias.storage_and_layout());

            let (q_bias_ptr, q_bias_guard) = if let Some((storage, layout)) = &q_bias_storage_layout
            {
                let Storage::Cuda(cuda) = &**storage else {
                    candle_core::bail!("q8_mmvq fused_qkv: q bias must live on CUDA");
                };
                let slice = cuda.as_cuda_slice::<$ty>()?;
                let (ptr, guard) = slice_ptr_on_stream(slice, &stream, layout.start_offset());
                (ptr as *const std::ffi::c_void, Some(guard))
            } else {
                (std::ptr::null(), None)
            };
            let (k_bias_ptr, k_bias_guard) = if let Some((storage, layout)) = &k_bias_storage_layout
            {
                let Storage::Cuda(cuda) = &**storage else {
                    candle_core::bail!("q8_mmvq fused_qkv: k bias must live on CUDA");
                };
                let slice = cuda.as_cuda_slice::<$ty>()?;
                let (ptr, guard) = slice_ptr_on_stream(slice, &stream, layout.start_offset());
                (ptr as *const std::ffi::c_void, Some(guard))
            } else {
                (std::ptr::null(), None)
            };
            let (v_bias_ptr, v_bias_guard) = if let Some((storage, layout)) = &v_bias_storage_layout
            {
                let Storage::Cuda(cuda) = &**storage else {
                    candle_core::bail!("q8_mmvq fused_qkv: v bias must live on CUDA");
                };
                let slice = cuda.as_cuda_slice::<$ty>()?;
                let (ptr, guard) = slice_ptr_on_stream(slice, &stream, layout.start_offset());
                (ptr as *const std::ffi::c_void, Some(guard))
            } else {
                (std::ptr::null(), None)
            };
            with_scratch(&dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    $quantize(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_bias_ptr,
                        k_bias_ptr,
                        v_bias_ptr,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_rows as i32,
                        k_rows as i32,
                        v_rows as i32,
                        num_blocks_per_row as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(v_bias_guard);
            drop(k_bias_guard);
            drop(q_bias_guard);
            drop(v_out_guard);
            drop(k_out_guard);
            drop(q_out_guard);
            drop(xs_guard);
            Ok((
                tensor_from_cuda_slice(q_out, &dev, output_shape(&xs, q_rows)),
                tensor_from_cuda_slice(k_out, &dev, output_shape(&xs, k_rows)),
                tensor_from_cuda_slice(v_out, &dev, output_shape(&xs, v_rows)),
            ))
        }};
    }

    match xs.dtype() {
        DType::BF16 => run!(
            half::bf16,
            as_cuda_slice,
            launch_mmvq_gguf_quantize_q8_1_bf16
        ),
        DType::F16 => run!(half::f16, as_cuda_slice, launch_mmvq_gguf_quantize_q8_1_f16),
        DType::F32 => run!(f32, as_cuda_slice, launch_mmvq_gguf_quantize_q8_1_f32),
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn try_fused_qkv_mmq_q8_0(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: &QTensor,
    q_bias: Option<&Tensor>,
    k_bias: Option<&Tensor>,
    v_bias: Option<&Tensor>,
    xs: &Tensor,
    b_size: usize,
    k: usize,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    if q_w.dtype() != GgmlDType::Q8_0
        || k_w.dtype() != GgmlDType::Q8_0
        || v_w.dtype() != GgmlDType::Q8_0
    {
        return Ok(None);
    }
    let input_type = match dtype_to_ggml_type(xs.dtype()) {
        Some(t) => t,
        None => return Ok(None),
    };
    if b_size <= Q8_MMVQ_MAX_BATCH || !k.is_multiple_of(Q8_1_BLOCK_SIZE) {
        return Ok(None);
    }
    let Device::Cuda(dev) = q_w.device() else {
        return Ok(None);
    };
    let (q_rows, q_cols) = q_w.shape().dims2()?;
    let (k_rows, k_cols) = k_w.shape().dims2()?;
    let (v_rows, v_cols) = v_w.shape().dims2()?;
    if q_cols != k_cols || q_cols != v_cols || k != q_cols {
        return Ok(None);
    }

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        return Ok(None);
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * Q8_1_BLOCK_SIZE);
    let blocks_per_row = k_padded / (4 * Q8_1_BLOCK_SIZE);
    let activation_bytes = b_size * blocks_per_row * Q8_1_MMQ_TYPE_SIZE;
    let activation_extra_bytes = 128 * Q8_1_MMQ_TYPE_SIZE;
    const MMQ_X_MAX: usize = 128;
    const MMQ_Y_MAX: usize = 128;
    const MAX_SMS: usize = 256;
    let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
    let scratch_bytes = activation_bytes + activation_extra_bytes + fixup_bytes;
    let (q_ptr, _q_guard) = q_w.device_ptr_with_guard(&stream)?;
    let (k_ptr, _k_guard) = k_w.device_ptr_with_guard(&stream)?;
    let (v_ptr, _v_guard) = v_w.device_ptr_with_guard(&stream)?;
    let q_ptr = q_ptr as *const std::ffi::c_void;
    let k_ptr = k_ptr as *const std::ffi::c_void;
    let v_ptr = v_ptr as *const std::ffi::c_void;
    let di = get_device_info(&dev);
    let stride_row_x = (k / Q8_1_BLOCK_SIZE) as i64;

    macro_rules! run {
        ($ty:ty, $slice_fn:ident) => {{
            let slice = xs_cuda.$slice_fn::<$ty>()?;
            let mut q_out = unsafe { dev.alloc::<$ty>(q_rows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<$ty>(k_rows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<$ty>(v_rows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (q_out_ptr, q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, &stream, 0);
            let (k_out_ptr, k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, &stream, 0);
            let (v_out_ptr, v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, &stream, 0);
            with_scratch(&dev, scratch_bytes, &stream, |scratch_ptr| {
                let fixup_ptr = unsafe {
                    (scratch_ptr as *mut u8).add(activation_bytes + activation_extra_bytes)
                        as *mut std::ffi::c_void
                };
                unsafe {
                    vasr_launch_mmq_quantize_q8_1_D4(
                        xs_ptr as *const std::ffi::c_void,
                        std::ptr::null(),
                        scratch_ptr,
                        input_type,
                        k as i64,
                        k as i64,
                        0,
                        0,
                        k_padded as i64,
                        b_size as i64,
                        1,
                        1,
                        stream_ptr,
                    );
                    vasr_launch_mmq_gguf_q8_0(
                        fixup_ptr,
                        q_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k as i64,
                        q_rows as i64,
                        b_size as i64,
                        stride_row_x,
                        q_rows as i64,
                        di.cc,
                        di.nsm,
                        di.smpbo,
                        di.warp_size,
                        input_type,
                        stream_ptr,
                    );
                    vasr_launch_mmq_gguf_q8_0(
                        fixup_ptr,
                        k_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        k as i64,
                        k_rows as i64,
                        b_size as i64,
                        stride_row_x,
                        k_rows as i64,
                        di.cc,
                        di.nsm,
                        di.smpbo,
                        di.warp_size,
                        input_type,
                        stream_ptr,
                    );
                    vasr_launch_mmq_gguf_q8_0(
                        fixup_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i64,
                        v_rows as i64,
                        b_size as i64,
                        stride_row_x,
                        v_rows as i64,
                        di.cc,
                        di.nsm,
                        di.smpbo,
                        di.warp_size,
                        input_type,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(v_out_guard);
            drop(k_out_guard);
            drop(q_out_guard);
            drop(xs_guard);
            record_mmq_fast_path(PlainKind::Decode, b_size);
            record_mmq_fast_path(PlainKind::Decode, b_size);
            record_mmq_fast_path(PlainKind::Decode, b_size);
            record_mmq_fused_qkv_fast_path(b_size);
            let q = tensor_from_cuda_slice(q_out, &dev, output_shape(&xs, q_rows));
            let k_tensor = tensor_from_cuda_slice(k_out, &dev, output_shape(&xs, k_rows));
            let v = tensor_from_cuda_slice(v_out, &dev, output_shape(&xs, v_rows));
            let q = add_projection_bias(q, q_bias, xs.dtype(), xs.device())?;
            let k_tensor = add_projection_bias(k_tensor, k_bias, xs.dtype(), xs.device())?;
            let v = add_projection_bias(v, v_bias, xs.dtype(), xs.device())?;
            Ok(Some((q, k_tensor, v)))
        }};
    }

    match xs.dtype() {
        DType::BF16 => run!(half::bf16, as_cuda_slice),
        DType::F16 => run!(half::f16, as_cuda_slice),
        DType::F32 => run!(f32, as_cuda_slice),
        _ => Ok(None),
    }
}

pub fn fused_gelu_erf(w: &QTensor, bias: Option<&Tensor>, xs: &Tensor) -> Result<Tensor> {
    if !can_run_plain(w, xs) {
        candle_core::bail!("q8_mmvq fused_act: unsupported fast path input");
    }
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("q8_mmvq fused_act: weight must live on CUDA");
    };
    let (nrows, ncols) = w.shape().dims2()?;
    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("q8_mmvq fused_act: unexpected input rank {other:?}"),
    };
    if k != ncols {
        candle_core::bail!(
            "q8_mmvq fused_act: shape mismatch weight [{nrows}, {ncols}] input tail {k}"
        );
    }
    record_fused_act_fast_path(b_size);

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let bias = bias
        .map(|b| {
            b.to_dtype(DType::BF16)?
                .to_device(xs.device())?
                .contiguous()
        })
        .transpose()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("q8_mmvq fused_act: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;
    let (weight_ptr, _weight_guard) = w.device_ptr_with_guard(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;
    let launch = fused_act_launcher(xs.dtype())?;
    const GLU_GELU_ERF: i32 = 3;

    match xs.dtype() {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let bias_storage_and_layout = bias.as_ref().map(|bias| bias.storage_and_layout());
            let (bias_ptr, bias_guard) =
                if let Some((bias_storage, bias_layout)) = bias_storage_and_layout.as_ref() {
                    let Storage::Cuda(bias_cuda) = &**bias_storage else {
                        candle_core::bail!("q8_mmvq fused_act: bias must live on CUDA");
                    };
                    if bias_layout.shape().dims() != [nrows] {
                        candle_core::bail!(
                            "q8_mmvq fused_act: bias shape mismatch: got {:?}, expected [{nrows}]",
                            bias_layout.shape().dims()
                        );
                    }
                    let bias_slice = bias_cuda.as_cuda_slice::<half::bf16>()?;
                    let (ptr, guard) =
                        slice_ptr_on_stream(bias_slice, &stream, bias_layout.start_offset());
                    (ptr as *const std::ffi::c_void, Some(guard))
                } else {
                    (std::ptr::null(), None)
                };
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            with_temp_scratch(&dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        bias_ptr,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        num_blocks_per_row as i32,
                        nrows as i32,
                        b_size as i32,
                        GLU_GELU_ERF,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(out_guard);
            drop(bias_guard);
            drop(xs_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        other => candle_core::bail!("q8_mmvq fused_act: unsupported input dtype {other:?}"),
    }
}

pub fn fused_glu_silu(gate_w: &QTensor, up_w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    if !can_run_fused_glu_decode(gate_w, up_w, xs) {
        candle_core::bail!("q8_mmvq fused_glu: unsupported fast path input");
    }
    let Device::Cuda(dev) = gate_w.device() else {
        candle_core::bail!("q8_mmvq fused_glu: gate weight must live on CUDA");
    };
    let Device::Cuda(up_dev) = up_w.device() else {
        candle_core::bail!("q8_mmvq fused_glu: up weight must live on CUDA");
    };
    if dev.id() != up_dev.id() {
        candle_core::bail!("q8_mmvq fused_glu: gate/up weights are on different CUDA devices");
    }

    let (nrows, ncols) = gate_w.shape().dims2()?;
    let (up_nrows, up_ncols) = up_w.shape().dims2()?;
    if (nrows, ncols) != (up_nrows, up_ncols) {
        candle_core::bail!(
            "q8_mmvq fused_glu: gate/up shape mismatch [{nrows}, {ncols}] vs [{up_nrows}, {up_ncols}]"
        );
    }
    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("q8_mmvq fused_glu: unexpected input rank {other:?}"),
    };
    if k != ncols {
        candle_core::bail!(
            "q8_mmvq fused_glu: shape mismatch weight [{nrows}, {ncols}] input tail {k}"
        );
    }
    record_fused_glu_fast_path(b_size);

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("q8_mmvq fused_glu: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;
    let (gate_ptr, _gate_guard) = gate_w.device_ptr_with_guard(&stream)?;
    let (up_ptr, _up_guard) = up_w.device_ptr_with_guard(&stream)?;
    let gate_ptr = gate_ptr as *const std::ffi::c_void;
    let up_ptr = up_ptr as *const std::ffi::c_void;
    let launch = fused_glu_launcher(xs.dtype())?;
    const GLU_SILU: i32 = 0;

    match xs.dtype() {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            with_scratch(&dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        num_blocks_per_row as i32,
                        nrows as i32,
                        b_size as i32,
                        GLU_SILU,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(out_guard);
            drop(xs_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            with_scratch(&dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        num_blocks_per_row as i32,
                        nrows as i32,
                        b_size as i32,
                        GLU_SILU,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(out_guard);
            drop(xs_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
            let (xs_ptr, xs_guard) = slice_ptr_on_stream(slice, &stream, xs_offset);
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, &stream, 0);
            with_scratch(&dev, scratch_bytes, &stream, |scratch_ptr| {
                unsafe {
                    launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launch(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        num_blocks_per_row as i32,
                        nrows as i32,
                        b_size as i32,
                        GLU_SILU,
                        stream_ptr,
                    );
                }
                Ok(())
            })?;
            drop(out_guard);
            drop(xs_guard);
            Ok(tensor_from_cuda_slice(out, &dev, output_shape(&xs, nrows)))
        }
        _ => unreachable!(),
    }
}
