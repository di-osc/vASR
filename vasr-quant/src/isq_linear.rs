//! X-infer style LinearX/QLinear layers with in-situ quantization.

use std::cell::Cell;
use std::sync::Arc;
#[cfg(feature = "timing")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[cfg(feature = "metal")]
use candle_core::quantized::k_quants;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

#[cfg(feature = "timing")]
static ISQ_QUANTIZE_US: AtomicU64 = AtomicU64::new(0);
static ISQ_AUDIO_LINEAR_COUNT: AtomicUsize = AtomicUsize::new(0);
static ISQ_TEXT_LINEAR_COUNT: AtomicUsize = AtomicUsize::new(0);
static ISQ_OTHER_LINEAR_COUNT: AtomicUsize = AtomicUsize::new(0);
static AFQ_FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "metal")]
static METAL_ISQ_DISABLED_WARNED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IsqModuleKind {
    Audio,
    Text,
    Other,
}

thread_local! {
    static LINEAR_IS_PREFILL: Cell<bool> = const { Cell::new(false) };
}

pub struct LinearPrefillGuard {
    prev: bool,
}

impl Drop for LinearPrefillGuard {
    fn drop(&mut self) {
        LINEAR_IS_PREFILL.with(|flag| flag.set(self.prev));
    }
}

pub fn set_linear_is_prefill(is_prefill: bool) -> LinearPrefillGuard {
    let prev = LINEAR_IS_PREFILL.with(|flag| {
        let prev = flag.get();
        flag.set(is_prefill);
        prev
    });
    LinearPrefillGuard { prev }
}

pub fn linear_is_prefill() -> bool {
    LINEAR_IS_PREFILL.with(|flag| flag.get())
}

#[cfg(feature = "timing")]
pub fn reset_isq_quantize_time() {
    ISQ_QUANTIZE_US.store(0, Ordering::Relaxed);
    reset_isq_layer_counts();
}

#[cfg(feature = "timing")]
pub fn isq_quantize_time_us() -> u64 {
    ISQ_QUANTIZE_US.load(Ordering::Relaxed)
}

pub fn reset_isq_layer_counts() {
    ISQ_AUDIO_LINEAR_COUNT.store(0, Ordering::Relaxed);
    ISQ_TEXT_LINEAR_COUNT.store(0, Ordering::Relaxed);
    ISQ_OTHER_LINEAR_COUNT.store(0, Ordering::Relaxed);
}

pub fn isq_layer_counts() -> (usize, usize, usize) {
    (
        ISQ_AUDIO_LINEAR_COUNT.load(Ordering::Relaxed),
        ISQ_TEXT_LINEAR_COUNT.load(Ordering::Relaxed),
        ISQ_OTHER_LINEAR_COUNT.load(Ordering::Relaxed),
    )
}

#[derive(Clone)]
pub enum LinearX {
    Linear(Linear),
    QLinear(QLinear),
}

pub type IsqLinear = LinearX;

/// Target bit width for automatic ISQ selection.
///
/// This mirrors mistral.rs: Metal prefers AFQ variants, while other backends
/// prefer GGML K-quants. vASR currently implements GGML ISQ only, so Metal AFQ
/// selections resolve to their GGML fallback until the AFQ backend lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsqBits {
    Two,
    Three,
    Four,
    Five,
    Six,
    Eight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedIsq {
    Ggml(GgmlDType),
    AfqFallback {
        requested: &'static str,
        fallback: GgmlDType,
    },
}

impl ResolvedIsq {
    pub fn dtype(self) -> GgmlDType {
        match self {
            Self::Ggml(dtype) => dtype,
            Self::AfqFallback { fallback, .. } => fallback,
        }
    }

    pub fn display_name(self) -> String {
        match self {
            Self::Ggml(dtype) => ggml_dtype_name(dtype).to_string(),
            Self::AfqFallback {
                requested,
                fallback,
            } => {
                format!("{requested} -> {}", ggml_dtype_name(fallback))
            }
        }
    }
}

impl IsqBits {
    pub fn resolve(self, device: &Device) -> ResolvedIsq {
        match (self, device.is_metal()) {
            (Self::Two, true) => ResolvedIsq::AfqFallback {
                requested: "afq2",
                fallback: GgmlDType::Q2K,
            },
            (Self::Two, false) => ResolvedIsq::Ggml(GgmlDType::Q2K),
            (Self::Three, true) => ResolvedIsq::AfqFallback {
                requested: "afq3",
                fallback: GgmlDType::Q3K,
            },
            (Self::Three, false) => ResolvedIsq::Ggml(GgmlDType::Q3K),
            (Self::Four, true) => ResolvedIsq::AfqFallback {
                requested: "afq4",
                fallback: GgmlDType::Q4K,
            },
            (Self::Four, false) => ResolvedIsq::Ggml(GgmlDType::Q4K),
            (Self::Five, _) => ResolvedIsq::Ggml(GgmlDType::Q5K),
            (Self::Six, true) => ResolvedIsq::AfqFallback {
                requested: "afq6",
                fallback: GgmlDType::Q6K,
            },
            (Self::Six, false) => ResolvedIsq::Ggml(GgmlDType::Q6K),
            (Self::Eight, true) => ResolvedIsq::AfqFallback {
                requested: "afq8",
                fallback: GgmlDType::Q8_0,
            },
            (Self::Eight, false) => ResolvedIsq::Ggml(GgmlDType::Q8_0),
        }
    }
}

impl TryFrom<&str> for IsqBits {
    type Error = ();

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value {
            "2" | "auto2" | "auto_2" | "auto-2" => Ok(Self::Two),
            "3" | "auto3" | "auto_3" | "auto-3" => Ok(Self::Three),
            "4" | "auto4" | "auto_4" | "auto-4" => Ok(Self::Four),
            "5" | "auto5" | "auto_5" | "auto-5" => Ok(Self::Five),
            "6" | "auto6" | "auto_6" | "auto-6" => Ok(Self::Six),
            "8" | "auto" | "auto8" | "auto_8" | "auto-8" => Ok(Self::Eight),
            _ => Err(()),
        }
    }
}

#[derive(Clone)]
pub struct QLinear {
    matmul: Arc<QMatMul>,
    bias: Option<Tensor>,
    output_dtype: DType,
    dtype: GgmlDType,
    #[cfg(feature = "cuda")]
    module_kind: IsqModuleKind,
    module_path: Arc<str>,
    #[cfg(feature = "metal")]
    metal_quant_batch_mm: bool,
}

impl std::fmt::Debug for LinearX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear(linear) => f.debug_tuple("Linear").field(linear).finish(),
            Self::QLinear(linear) => f.debug_tuple("QLinear").field(linear).finish(),
        }
    }
}

impl std::fmt::Debug for QLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QLinear")
            .field("dtype", &self.dtype)
            .field("output_dtype", &self.output_dtype)
            .field("module_path", &self.module_path)
            .finish_non_exhaustive()
    }
}

impl LinearX {
    pub fn new(linear: Linear, isq: Option<&str>, target_device: &Device) -> Result<Self> {
        Self::new_with_module_path(linear, isq, target_device, "")
    }

    pub fn new_with_module_path(
        linear: Linear,
        isq: Option<&str>,
        target_device: &Device,
        module_path: &str,
    ) -> Result<Self> {
        let Some(isq) = isq else {
            let linear = linear_to_device(linear, target_device)?;
            return Ok(Self::Linear(linear));
        };

        let spec = parse_isq_spec(isq, target_device)?;
        if should_skip_module(module_path, &spec.modules_to_not_convert) {
            let linear = linear_to_device(linear, target_device)?;
            return Ok(Self::Linear(linear));
        }

        if spec.disable_quantization {
            let linear = linear_to_device(linear, target_device)?;
            return Ok(Self::Linear(linear));
        }

        let requested = spec
            .dtype
            .ok_or_else(|| candle_core::Error::Msg("missing ISQ dtype".to_string()))?;
        let weight = linear.weight();
        let output_dtype = weight.dtype();
        let bias = linear
            .bias()
            .map(|b| b.to_dtype(output_dtype)?.to_device(target_device))
            .transpose()?;
        let dtype = compatible_dtype(weight, requested);

        let Some(dtype) = dtype else {
            candle_core::bail!(
                "ISQ `{}` cannot quantize linear `{}` with weight shape {:?}: input dimension is not divisible by block size {}",
                ggml_dtype_name(requested),
                module_path,
                weight.dims(),
                requested.block_size()
            );
        };

        #[cfg(feature = "timing")]
        let start_quantize = std::time::Instant::now();
        let qtensor = if weight.device().is_cpu() && !target_device.is_cpu() {
            QTensor::quantize_onto(weight, dtype, target_device)?
        } else {
            let qtensor = QTensor::quantize(weight, dtype)?;
            qtensor_to_device(qtensor, target_device)?
        };
        let matmul = QMatMul::from_qtensor(qtensor)?;
        #[cfg(feature = "timing")]
        ISQ_QUANTIZE_US.fetch_add(
            start_quantize
                .elapsed()
                .as_micros()
                .min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
        record_isq_linear(module_path);
        #[cfg(feature = "cuda")]
        let module_kind = isq_module_kind(module_path);
        Ok(Self::QLinear(QLinear {
            matmul: Arc::new(matmul),
            bias,
            output_dtype,
            dtype,
            #[cfg(feature = "cuda")]
            module_kind,
            module_path: Arc::from(module_path),
            #[cfg(feature = "metal")]
            metal_quant_batch_mm: metal_quant_batch_mm_flag(
                std::env::var("VASR_METAL_QUANT_BATCH_MM").ok().as_deref(),
            ),
        }))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(linear) => linear.forward(x),
            Self::QLinear(linear) => linear.forward(x),
        }
    }

    pub fn forward_chunked_argmax_next_ids(
        &self,
        x: &Tensor,
        chunk_size: usize,
    ) -> Result<Option<Tensor>> {
        #[cfg(not(feature = "cuda"))]
        let _ = x;
        if chunk_size == 0 {
            return Ok(None);
        }
        #[cfg(feature = "cuda")]
        if let Self::QLinear(qlinear) = self
            && let Some(next_ids) = qlinear.forward_q8_lm_head_next_ids(x, chunk_size)?
        {
            return Ok(Some(next_ids));
        }
        Ok(None)
    }

    pub fn supports_chunked_argmax_next_ids(&self) -> bool {
        #[cfg(feature = "cuda")]
        if let Self::QLinear(qlinear) = self {
            return qlinear.supports_q8_lm_head_next_ids();
        }
        false
    }
}

impl candle_core::Module for LinearX {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        LinearX::forward(self, x)
    }
}

#[cfg(feature = "metal")]
fn metal_quant_batch_mm_flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(feature = "metal")]
fn metal_quant_batch_mm_shape(
    enabled: bool,
    is_metal: bool,
    is_quantized: bool,
    is_prefill: bool,
    dims: &[usize],
) -> Option<(usize, usize)> {
    if !enabled || !is_metal || !is_quantized || is_prefill {
        return None;
    }
    match dims {
        &[batch, 1, hidden] if batch > 1 && hidden > 0 => Some((batch, hidden)),
        _ => None,
    }
}

impl QLinear {
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let QMatMul::QTensor(qtensor) = self.matmul.as_ref() {
            let kind = if self.module_path.as_ref().ends_with("lm_head") {
                if linear_is_prefill() {
                    crate::q8_mmvq::PlainKind::LmHeadPrefill
                } else {
                    crate::q8_mmvq::PlainKind::LmHead
                }
            } else {
                match self.module_kind {
                    IsqModuleKind::Audio => crate::q8_mmvq::PlainKind::Audio,
                    IsqModuleKind::Text | IsqModuleKind::Other if linear_is_prefill() => {
                        crate::q8_mmvq::PlainKind::Prefill
                    }
                    IsqModuleKind::Text | IsqModuleKind::Other => crate::q8_mmvq::PlainKind::Decode,
                }
            };
            let can_run_q8_mmvq = match kind {
                crate::q8_mmvq::PlainKind::Audio => crate::q8_mmvq::can_run_plain(qtensor, x),
                crate::q8_mmvq::PlainKind::Prefill => crate::q8_mmvq::can_run_plain(qtensor, x),
                crate::q8_mmvq::PlainKind::Decode => crate::q8_mmvq::can_run_decode(qtensor, x),
                crate::q8_mmvq::PlainKind::LmHead => crate::q8_mmvq::can_run_plain(qtensor, x),
                crate::q8_mmvq::PlainKind::LmHeadPrefill => {
                    crate::q8_mmvq::can_run_plain(qtensor, x)
                }
            };
            if can_run_q8_mmvq {
                let mut ys = crate::q8_mmvq::plain(qtensor, x, kind)?;
                if let Some(bias) = &self.bias {
                    ys = ys.broadcast_add(&bias.to_dtype(ys.dtype())?.to_device(x.device())?)?;
                }
                return Ok(ys);
            }
            if qtensor.dtype() == GgmlDType::Q8_0 && qtensor.device().is_cuda() {
                candle_core::bail!(
                    "CUDA Q8_0 linear is not covered by q8_mmvq fast path: module={} kind={kind:?} input_shape={:?} input_dtype={:?}",
                    self.module_path,
                    x.dims(),
                    x.dtype()
                );
            }
        }

        let xs = if x.dtype() == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        // Candle treats [batch, 1, hidden] as separate matrix-vector dispatches.
        // Opt in at layer construction to a single quantized matrix-matrix call.
        #[cfg(feature = "metal")]
        let batch_mm_shape = metal_quant_batch_mm_shape(
            self.metal_quant_batch_mm,
            xs.device().is_metal(),
            matches!(self.matmul.as_ref(), QMatMul::QTensor(weight) if weight.device().is_metal()),
            linear_is_prefill(),
            xs.dims(),
        );
        #[cfg(feature = "metal")]
        let mut ys = if let Some((batch, hidden)) = batch_mm_shape {
            let output = self
                .matmul
                .as_ref()
                .forward(&xs.reshape((batch, hidden))?)?;
            let (_, out_dim) = output.dims2()?;
            output.reshape((batch, 1, out_dim))?
        } else {
            self.matmul.as_ref().forward(&xs)?
        };
        #[cfg(not(feature = "metal"))]
        let mut ys = self.matmul.as_ref().forward(&xs)?;
        if let Some(bias) = &self.bias {
            ys = ys.broadcast_add(&bias.to_dtype(DType::F32)?.to_device(x.device())?)?;
        }
        if self.output_dtype == DType::F32 {
            Ok(ys)
        } else {
            ys.to_dtype(self.output_dtype)
        }
    }

    #[cfg(feature = "cuda")]
    fn forward_q8_lm_head_next_ids(&self, x: &Tensor, chunk_size: usize) -> Result<Option<Tensor>> {
        if !self.module_path.as_ref().ends_with("lm_head") || self.bias.is_some() {
            return Ok(None);
        }
        let QMatMul::QTensor(weight) = self.matmul.as_ref() else {
            return Ok(None);
        };
        let kind = if linear_is_prefill() {
            crate::q8_mmvq::NextIdKind::Prefill
        } else {
            crate::q8_mmvq::NextIdKind::Decode
        };
        crate::q8_mmvq::lm_head_next_ids_q8_0(weight, x, chunk_size, kind)
    }

    #[cfg(feature = "cuda")]
    fn supports_q8_lm_head_next_ids(&self) -> bool {
        if !self.module_path.as_ref().ends_with("lm_head") || self.bias.is_some() {
            return false;
        }
        let QMatMul::QTensor(weight) = self.matmul.as_ref() else {
            return false;
        };
        weight.dtype() == GgmlDType::Q8_0 && weight.device().is_cuda()
    }
}

#[cfg(feature = "cuda")]
pub fn try_fused_q8_silu_gate_up(
    gate: &IsqLinear,
    up: &IsqLinear,
    x: &Tensor,
) -> Result<Option<Tensor>> {
    if linear_is_prefill() {
        return Ok(None);
    }
    let (LinearX::QLinear(gate), LinearX::QLinear(up)) = (gate, up) else {
        return Ok(None);
    };
    if gate.bias.is_some() || up.bias.is_some() {
        return Ok(None);
    }
    let (QMatMul::QTensor(gate_q), QMatMul::QTensor(up_q)) =
        (gate.matmul.as_ref(), up.matmul.as_ref())
    else {
        return Ok(None);
    };
    if !crate::q8_mmvq::can_run_fused_glu_decode(gate_q, up_q, x) {
        return Ok(None);
    }
    Ok(Some(crate::q8_mmvq::fused_glu_silu(gate_q, up_q, x)?))
}

#[cfg(feature = "cuda")]
pub fn try_fused_q8_gelu_erf(linear: &IsqLinear, x: &Tensor) -> Result<Option<Tensor>> {
    let LinearX::QLinear(linear) = linear else {
        return Ok(None);
    };
    let QMatMul::QTensor(weight) = linear.matmul.as_ref() else {
        return Ok(None);
    };
    if !crate::q8_mmvq::can_run_plain(weight, x) || x.dtype() != DType::BF16 {
        return Ok(None);
    }
    Ok(Some(crate::q8_mmvq::fused_gelu_erf(
        weight,
        linear.bias.as_ref(),
        x,
    )?))
}

#[cfg(feature = "cuda")]
pub fn try_fused_q8_qkv(
    q: &IsqLinear,
    k: &IsqLinear,
    v: &IsqLinear,
    x: &Tensor,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    if q8_fused_qkv_disabled() {
        return Ok(None);
    }
    if linear_is_prefill() {
        return Ok(None);
    }
    let (LinearX::QLinear(q), LinearX::QLinear(k), LinearX::QLinear(v)) = (q, k, v) else {
        return Ok(None);
    };
    let (QMatMul::QTensor(q_w), QMatMul::QTensor(k_w), QMatMul::QTensor(v_w)) =
        (q.matmul.as_ref(), k.matmul.as_ref(), v.matmul.as_ref())
    else {
        return Ok(None);
    };
    if q.module_kind != k.module_kind || q.module_kind != v.module_kind {
        return Ok(None);
    }
    if matches!(q.module_kind, IsqModuleKind::Audio) {
        let flat_batch = x.dims()[..x.dims().len().saturating_sub(1)]
            .iter()
            .product::<usize>();
        if flat_batch > 8 {
            return Ok(None);
        }
    }
    if !crate::q8_mmvq::can_run_fused_qkv(q_w, k_w, v_w, x) {
        return Ok(None);
    }
    let (q_y, k_y, v_y) = crate::q8_mmvq::fused_qkv(
        q_w,
        k_w,
        v_w,
        q.bias.as_ref(),
        k.bias.as_ref(),
        v.bias.as_ref(),
        x,
    )?;
    Ok(Some((q_y, k_y, v_y)))
}

#[cfg(feature = "cuda")]
fn q8_fused_qkv_disabled() -> bool {
    matches!(
        std::env::var("VASR_Q8_DISABLE_FUSED_QKV").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("on") | Ok("ON")
    )
}

impl candle_core::Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        QLinear::forward(self, x)
    }
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: VarBuilder,
    isq: Option<&str>,
) -> Result<IsqLinear> {
    let target_device = vb.device().clone();
    let module_path = vb.prefix();
    let linear = if bias {
        candle_nn::linear(in_dim, out_dim, vb)?
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)?
    };
    LinearX::new_with_module_path(linear, isq, &target_device, &module_path)
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    isq: Option<&str>,
) -> Result<IsqLinear> {
    let target_device = vb.device().clone();
    let module_path = vb.prefix();
    let linear = candle_nn::linear_no_bias(in_dim, out_dim, vb)?;
    LinearX::new_with_module_path(linear, isq, &target_device, &module_path)
}

fn linear_to_device(linear: Linear, device: &Device) -> Result<Linear> {
    let weight = linear.weight().to_device(device)?;
    let bias = linear.bias().map(|b| b.to_device(device)).transpose()?;
    Ok(Linear::new(weight, bias))
}

fn qtensor_to_device(qtensor: QTensor, device: &Device) -> Result<QTensor> {
    match device {
        Device::Cpu => Ok(qtensor),
        #[cfg(feature = "metal")]
        Device::Metal(metal) if matches!(qtensor.device(), Device::Cpu) => {
            load_cpu_qtensor_to_metal(qtensor, metal)
        }
        _ => Ok(qtensor),
    }
}

#[cfg(feature = "metal")]
fn load_cpu_qtensor_to_metal(
    qtensor: QTensor,
    device: &candle_core::MetalDevice,
) -> Result<QTensor> {
    let dtype = qtensor.dtype();
    let shape = qtensor.shape().clone();
    let data = qtensor.data()?;

    macro_rules! load {
        ($block:ty) => {{
            let blocks = bytes_as_blocks::<$block>(data.as_ref())?;
            let storage = candle_core::quantized::metal::load_quantized(device, blocks)?;
            QTensor::new(storage, shape)
        }};
    }

    match dtype {
        GgmlDType::Q4_0 => load!(k_quants::BlockQ4_0),
        GgmlDType::Q4_1 => load!(k_quants::BlockQ4_1),
        GgmlDType::Q5_0 => load!(k_quants::BlockQ5_0),
        GgmlDType::Q5_1 => load!(k_quants::BlockQ5_1),
        GgmlDType::Q8_0 => load!(k_quants::BlockQ8_0),
        GgmlDType::Q8_1 => load!(k_quants::BlockQ8_1),
        GgmlDType::Q2K => load!(k_quants::BlockQ2K),
        GgmlDType::Q3K => load!(k_quants::BlockQ3K),
        GgmlDType::Q4K => load!(k_quants::BlockQ4K),
        GgmlDType::Q5K => load!(k_quants::BlockQ5K),
        GgmlDType::Q6K => load!(k_quants::BlockQ6K),
        GgmlDType::Q8K => load!(k_quants::BlockQ8K),
        other => candle_core::bail!("unsupported ISQ metal upload dtype: {other:?}"),
    }
}

#[cfg(feature = "metal")]
fn bytes_as_blocks<T>(bytes: &[u8]) -> Result<&[T]> {
    let block_size = std::mem::size_of::<T>();
    if block_size == 0 || bytes.len() % block_size != 0 {
        candle_core::bail!(
            "quantized byte length is not divisible by block size: bytes={} block_size={block_size}",
            bytes.len()
        );
    }

    let (prefix, blocks, suffix) = unsafe { bytes.align_to::<T>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        candle_core::bail!(
            "quantized bytes are not aligned for block upload: prefix={} suffix={}",
            prefix.len(),
            suffix.len()
        );
    }
    Ok(blocks)
}

pub fn resolve_isq_dtype(value: &str, device: &Device) -> Result<ResolvedIsq> {
    let dtype = value
        .split([';', ':', ','])
        .next()
        .unwrap_or(value)
        .trim()
        .to_ascii_lowercase();

    if let Ok(bits) = IsqBits::try_from(dtype.as_str()) {
        return Ok(bits.resolve(device));
    }

    match dtype.as_str() {
        "q40" | "q4_0" => Ok(ResolvedIsq::Ggml(GgmlDType::Q4_0)),
        "q4" | "q41" | "q4_1" => Ok(ResolvedIsq::Ggml(GgmlDType::Q4_1)),
        "q50" | "q5_0" => Ok(ResolvedIsq::Ggml(GgmlDType::Q5_0)),
        "q5" | "q51" | "q5_1" => Ok(ResolvedIsq::Ggml(GgmlDType::Q5_1)),
        "q8" | "q80" | "q8_0" => Ok(ResolvedIsq::Ggml(GgmlDType::Q8_0)),
        "q2k" | "q2_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q2K)),
        "q3k" | "q3_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q3K)),
        "q4k" | "q4_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q4K)),
        "q5k" | "q5_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q5K)),
        "q6k" | "q6_k" => Ok(ResolvedIsq::Ggml(GgmlDType::Q6K)),
        "afq2" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq2",
            fallback: GgmlDType::Q2K,
        }),
        "afq3" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq3",
            fallback: GgmlDType::Q3K,
        }),
        "afq4" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq4",
            fallback: GgmlDType::Q4K,
        }),
        "afq6" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq6",
            fallback: GgmlDType::Q6K,
        }),
        "afq8" => Ok(ResolvedIsq::AfqFallback {
            requested: "afq8",
            fallback: GgmlDType::Q8_0,
        }),
        other => candle_core::bail!(
            "unsupported isq dtype {other:?}; use auto/auto8/auto6/auto4/8/6/4/q8_0/q6_k/q5_k/q4_k/q3_k/q2_k"
        ),
    }
}

pub fn parse_isq_dtype(value: &str) -> Result<GgmlDType> {
    Ok(resolve_isq_dtype(value, &Device::Cpu)?.dtype())
}

pub fn resolve_isq_display(value: &str, device: &Device) -> Result<String> {
    Ok(resolve_isq_dtype(value, device)?.display_name())
}

#[derive(Debug, Clone)]
struct IsqSpec {
    dtype: Option<GgmlDType>,
    disable_quantization: bool,
    modules_to_not_convert: Vec<String>,
}

fn parse_isq_spec(value: &str, device: &Device) -> Result<IsqSpec> {
    let resolved = resolve_isq_dtype(value, device)?;
    let dtype = Some(resolved.dtype());
    #[cfg(not(feature = "metal"))]
    let disable_quantization = false;
    #[cfg(feature = "metal")]
    let mut disable_quantization = false;
    #[cfg(feature = "metal")]
    if device.is_metal() && std::env::var_os("VASR_DISABLE_METAL_ISQ").is_some() {
        disable_quantization = true;
        if !METAL_ISQ_DISABLED_WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "Metal ISQ was disabled by VASR_DISABLE_METAL_ISQ=1; using dense linear layers instead."
            );
        }
    }
    match resolved {
        ResolvedIsq::AfqFallback {
            requested,
            fallback,
        } if !AFQ_FALLBACK_WARNED.swap(true, Ordering::Relaxed) => {
            tracing::warn!(
                "ISQ `{}` resolved by auto selection, but AFQ is not implemented in this runtime yet; falling back to {}.",
                requested,
                ggml_dtype_name(fallback)
            );
        }
        _ => {}
    }
    let mut modules_to_not_convert = Vec::new();
    let mut reading_skip_values = false;

    for part in value.split([';', ',']) {
        let part = part.trim();
        if reading_skip_values && !part.contains('=') {
            modules_to_not_convert.extend(split_skip_values(part));
            continue;
        }
        let Some((key, values)) = part.split_once('=') else {
            reading_skip_values = false;
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        if matches!(
            key.as_str(),
            "skip" | "skip_modules" | "modules_to_not_convert"
        ) {
            reading_skip_values = true;
            modules_to_not_convert.extend(split_skip_values(values));
        } else {
            reading_skip_values = false;
        }
    }

    modules_to_not_convert.sort();
    modules_to_not_convert.dedup();
    Ok(IsqSpec {
        dtype,
        disable_quantization,
        modules_to_not_convert,
    })
}

fn split_skip_values(values: &str) -> impl Iterator<Item = String> + '_ {
    values
        .split(['|', '+', ','])
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
}

fn record_isq_linear(module_path: &str) {
    match isq_module_kind(module_path) {
        IsqModuleKind::Audio => {
            ISQ_AUDIO_LINEAR_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        IsqModuleKind::Text => {
            ISQ_TEXT_LINEAR_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        IsqModuleKind::Other => {
            ISQ_OTHER_LINEAR_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn isq_module_kind(module_path: &str) -> IsqModuleKind {
    if module_path.contains("audio_tower") {
        IsqModuleKind::Audio
    } else if module_path.contains(".model.") || module_path.ends_with("lm_head") {
        IsqModuleKind::Text
    } else {
        IsqModuleKind::Other
    }
}

fn ggml_dtype_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        GgmlDType::Q8K => "q8_k",
        _ => "unknown",
    }
}

fn should_skip_module(module_path: &str, modules_to_not_convert: &[String]) -> bool {
    if module_path.is_empty() {
        return false;
    }

    modules_to_not_convert
        .iter()
        .any(|item| module_path_matches_not_convert(module_path, item))
}

fn module_path_matches_not_convert(module_path: &str, item: &str) -> bool {
    let module_path = module_path.trim_end_matches(".weight");
    let item = item.trim().trim_end_matches(".weight");
    !item.is_empty()
        && (module_path == item
            || module_path.ends_with(item)
            || module_path.ends_with(&format!(".{item}"))
            || item.ends_with(module_path)
            || item.ends_with(&format!(".{module_path}")))
}

fn compatible_dtype(weight: &Tensor, requested: GgmlDType) -> Option<GgmlDType> {
    let last_dim = weight.dim(candle_core::D::Minus1).ok()?;
    if last_dim % requested.block_size() == 0 {
        Some(requested)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::try_fused_q8_silu_gate_up;
    use super::{
        IsqBits, IsqLinear, LinearX, ResolvedIsq, linear_is_prefill,
        module_path_matches_not_convert, parse_isq_dtype, resolve_isq_dtype, set_linear_is_prefill,
    };
    use candle_core::quantized::GgmlDType;

    #[cfg(feature = "metal")]
    #[test]
    fn metal_quant_batch_mm_only_accepts_opted_in_quantized_decode() {
        use super::metal_quant_batch_mm_shape;
        for batch in [1, 2, 8, 16, 32] {
            let expected = (batch > 1).then_some((batch, 1024));
            assert_eq!(
                metal_quant_batch_mm_shape(true, true, true, false, &[batch, 1, 1024]),
                expected
            );
        }
        for dims in [
            &[32, 1024][..],
            &[32, 2, 1024],
            &[0, 1, 1024],
            &[32, 1, 0],
            &[2, 16, 1, 1024],
        ] {
            assert_eq!(
                metal_quant_batch_mm_shape(true, true, true, false, dims),
                None
            );
        }
        for (enabled, metal, quantized, prefill) in [
            (false, true, true, false),
            (true, false, true, false),
            (true, true, false, false),
            (true, true, true, true),
        ] {
            assert_eq!(
                metal_quant_batch_mm_shape(enabled, metal, quantized, prefill, &[32, 1, 1024]),
                None
            );
        }
        assert!(!super::metal_quant_batch_mm_flag(None));
        for value in ["", "0", "false", "no", "garbage"] {
            assert!(!super::metal_quant_batch_mm_flag(Some(value)));
        }
        for value in ["1", "true", "yes", "on", " TRUE "] {
            assert!(super::metal_quant_batch_mm_flag(Some(value)));
        }
    }

    // Explicit GPU test: run only after concurrent throughput measurements finish.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires Metal GPU; compares quantized batch MM with row MV"]
    fn metal_quant_batch_mm_matches_row_mv_qwen06b_dimensions() -> anyhow::Result<()> {
        use candle_core::quantized::{QMatMul, QTensor};
        use candle_core::{DType, Device, Tensor};
        use std::sync::Arc;
        let device = Device::new_metal(0)?;
        // Local Qwen3-ASR-0.6B config: hidden=1024, q=16*128, kv=8*128, intermediate=3072.
        // Shapes are (out, in): q, k/v, o, gate/up, down, and quantized tied lm_head.
        for (out_dim, in_dim) in [
            (2048, 1024),
            (1024, 1024),
            (1024, 2048),
            (3072, 1024),
            (1024, 3072),
            (151936, 1024),
        ] {
            let weights: Vec<f32> = (0..out_dim * in_dim)
                .map(|i| (((i * 17 + i / in_dim * 13) % 127) as f32 - 63.0) / 1024.0)
                .collect();
            let weight = Tensor::from_vec(weights, (out_dim, in_dim), &Device::Cpu)?;
            for quant in [GgmlDType::Q8_0, GgmlDType::Q6K, GgmlDType::Q5_0, GgmlDType::Q4_0] {
                let matmul = Arc::new(QMatMul::from_qtensor(QTensor::quantize_onto(
                    &weight, quant, &device,
                )?)?);
                anyhow::ensure!(
                    matches!(matmul.as_ref(), QMatMul::QTensor(_)),
                    "dense dequantization env overrides must be disabled for this regression"
                );
                for dtype in [DType::F32, DType::BF16] {
                    for has_bias in [false, true] {
                        let bias = if has_bias {
                            Some(
                                Tensor::from_vec(
                                    (0..out_dim)
                                        .map(|i| (i % 11) as f32 / 128.0)
                                        .collect::<Vec<_>>(),
                                    out_dim,
                                    &device,
                                )?
                                .to_dtype(dtype)?,
                            )
                        } else {
                            None
                        };
                        let legacy = super::QLinear {
                            matmul: matmul.clone(),
                            bias,
                            output_dtype: dtype,
                            dtype: quant,
                            #[cfg(feature = "cuda")]
                            module_kind: super::IsqModuleKind::Text,
                            module_path: Arc::from("test.qwen06b"),
                            metal_quant_batch_mm: false,
                        };
                        let mut candidate = legacy.clone();
                        candidate.metal_quant_batch_mm = true;
                        let _decode_guard = set_linear_is_prefill(false);
                        for batch in [1, 2, 8, 16, 32] {
                            let input = Tensor::from_vec(
                                (0..batch * in_dim)
                                    .map(|i| {
                                        (((i * 7 + i / in_dim * 5) % 97) as f32 - 48.0) / 256.0
                                    })
                                    .collect::<Vec<_>>(),
                                (batch, 1, in_dim),
                                &device,
                            )?
                            .to_dtype(dtype)?;
                            let expected = legacy.forward(&input)?;
                            let actual = candidate.forward(&input)?;
                            assert_eq!(actual.dims(), &[batch, 1, out_dim]);
                            assert_eq!(actual.dtype(), dtype);
                            let expected = expected
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            let actual = actual
                                .to_dtype(DType::F32)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            for (index, (actual, expected)) in
                                actual.iter().zip(&expected).enumerate()
                            {
                                // Quantized MM uses a different reduction order/internal precision.
                                let tolerance = 0.01 + 0.01 * expected.abs();
                                assert!(
                                    actual.is_finite() && (actual - expected).abs() <= tolerance,
                                    "quant={quant:?} dtype={dtype:?} bias={has_bias} shape=({batch},1,{in_dim})->{out_dim} index={index} actual={actual} expected={expected}"
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_parse_isq_dtype_aliases() -> anyhow::Result<()> {
        assert_eq!(parse_isq_dtype("q4_k")?, GgmlDType::Q4K);
        assert_eq!(parse_isq_dtype("q4k")?, GgmlDType::Q4K);
        assert_eq!(parse_isq_dtype("q8")?, GgmlDType::Q8_0);
        assert_eq!(parse_isq_dtype("auto")?, GgmlDType::Q8_0);
        Ok(())
    }

    #[test]
    fn test_auto_isq_resolves_like_mistralrs_on_cpu() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        assert_eq!(
            IsqBits::Four.resolve(&device),
            ResolvedIsq::Ggml(GgmlDType::Q4K)
        );
        assert_eq!(
            resolve_isq_dtype("auto6", &device)?,
            ResolvedIsq::Ggml(GgmlDType::Q6K)
        );
        assert_eq!(
            resolve_isq_dtype("8", &device)?,
            ResolvedIsq::Ggml(GgmlDType::Q8_0)
        );
        Ok(())
    }

    #[test]
    fn test_isq8_quantizes_audio_text_and_lm_head_paths() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;

        for (module_path, want_kind) in [
            (
                "thinker.audio_tower.layers.0.self_attn.q_proj",
                super::IsqModuleKind::Audio,
            ),
            (
                "thinker.model.layers.0.self_attn.q_proj",
                super::IsqModuleKind::Text,
            ),
            ("thinker.lm_head", super::IsqModuleKind::Text),
        ] {
            assert_eq!(super::isq_module_kind(module_path), want_kind);
            let weight =
                candle_core::Tensor::zeros((4usize, 32usize), candle_core::DType::F32, &device)?;
            let linear = candle_nn::Linear::new(weight, None);
            let linear = IsqLinear::new_with_module_path(linear, Some("8"), &device, module_path)?;
            assert!(
                matches!(linear, LinearX::QLinear(_)),
                "{module_path} should be quantized for --isq 8"
            );
        }

        Ok(())
    }

    #[test]
    fn test_q8_prefill_lm_head_mmq_toggle_is_callable() {
        crate::set_q8_prefill_lm_head_mmq_enabled(false);
        crate::set_q8_prefill_lm_head_mmq_max_logits_bytes(128 * 1024 * 1024);
        crate::set_q8_prefill_lm_head_mmq_enabled(true);
        crate::set_q8_prefill_lm_head_mmq_max_logits_bytes(256 * 1024 * 1024);
    }

    #[test]
    fn test_isq_linear_preserves_shape_and_dtype() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let weight =
            candle_core::Tensor::zeros((4usize, 32usize), candle_core::DType::F32, &device)?;
        let linear = candle_nn::Linear::new(weight, None);
        let linear = IsqLinear::new(linear, Some("q8_0"), &device)?;
        let x = candle_core::Tensor::ones((2usize, 32usize), candle_core::DType::F32, &device)?;
        let y = linear.forward(&x)?;
        assert_eq!(y.dims(), &[2, 4]);
        assert_eq!(y.dtype(), candle_core::DType::F32);
        Ok(())
    }

    #[test]
    fn test_linear_prefill_guard_restores_previous_state() {
        assert!(!linear_is_prefill());
        {
            let _guard = set_linear_is_prefill(true);
            assert!(linear_is_prefill());
            {
                let _nested = set_linear_is_prefill(false);
                assert!(!linear_is_prefill());
            }
            assert!(linear_is_prefill());
        }
        assert!(!linear_is_prefill());
    }

    #[test]
    fn test_isq_skip_module_keeps_dense_linear() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let weight =
            candle_core::Tensor::zeros((4usize, 32usize), candle_core::DType::F32, &device)?;
        let linear = candle_nn::Linear::new(weight, None);
        let linear = IsqLinear::new_with_module_path(
            linear,
            Some("q8_0;skip=layers.0.self_attn.q_proj"),
            &device,
            "thinker.model.layers.0.self_attn.q_proj",
        )?;
        assert!(matches!(linear, LinearX::Linear(_)));
        assert!(module_path_matches_not_convert(
            "thinker.lm_head",
            "lm_head"
        ));
        Ok(())
    }

    #[test]
    fn test_isq_skip_accepts_comma_separated_values() -> anyhow::Result<()> {
        let device = candle_core::Device::Cpu;
        let weight =
            candle_core::Tensor::zeros((4usize, 32usize), candle_core::DType::F32, &device)?;
        let linear = IsqLinear::new_with_module_path(
            candle_nn::Linear::new(weight, None),
            Some("q8_0;skip=q_proj,k_proj"),
            &device,
            "thinker.model.layers.0.self_attn.k_proj",
        )?;
        assert!(matches!(linear, LinearX::Linear(_)));
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_q8_mmvq_fast_path_matches_fallback_on_cuda() -> anyhow::Result<()> {
        use candle_core::Module;

        let device = candle_core::Device::new_cuda(0)?;
        let weight = candle_core::Tensor::randn(
            0f32,
            0.02f32,
            (64usize, 128usize),
            &candle_core::Device::Cpu,
        )?
        .to_dtype(candle_core::DType::BF16)?;
        let linear = candle_nn::Linear::new(weight, None);
        let linear = IsqLinear::new_with_module_path(
            linear,
            Some("q8_0"),
            &device,
            "thinker.audio_tower.layers.0.fc1",
        )?;
        let LinearX::QLinear(qlinear) = linear else {
            anyhow::bail!("expected QLinear");
        };
        for rows in [1usize, 8, 9, 32, 128, 512] {
            super::super::q8_mmvq::reset_fast_path_stats();
            let x = candle_core::Tensor::randn(0f32, 0.02f32, (1usize, rows, 128usize), &device)?
                .to_dtype(candle_core::DType::BF16)?;

            let fast = qlinear.forward(&x)?.to_dtype(candle_core::DType::F32)?;
            let fallback = qlinear
                .matmul
                .as_ref()
                .forward(&x.to_dtype(candle_core::DType::F32)?)?
                .to_dtype(candle_core::DType::BF16)?
                .to_dtype(candle_core::DType::F32)?;
            let diff = (fast - fallback)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                diff < 0.05,
                "q8 mmvq fast path diverged from fallback for rows={rows}: max abs diff {diff}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_q8_mmvq_large_shapes_match_fallback_on_cuda() -> anyhow::Result<()> {
        use candle_core::Module;

        let device = candle_core::Device::new_cuda(0)?;
        for (out_dim, in_dim, rows) in [
            (2048usize, 2048usize, 512usize),
            (4096usize, 2048usize, 512usize),
            (2048usize, 4096usize, 512usize),
            (5120usize, 1280usize, 1024usize),
            (1280usize, 5120usize, 1024usize),
        ] {
            let weight = candle_core::Tensor::randn(0f32, 0.02f32, (out_dim, in_dim), &device)?
                .to_dtype(candle_core::DType::BF16)?;
            let linear = IsqLinear::new_with_module_path(
                candle_nn::Linear::new(weight, None),
                Some("q8_0"),
                &device,
                "thinker.model.layers.0.mlp.down_proj",
            )?;
            let LinearX::QLinear(qlinear) = linear else {
                anyhow::bail!("expected QLinear");
            };
            super::super::q8_mmvq::reset_fast_path_stats();
            let x = candle_core::Tensor::randn(0f32, 0.02f32, (1usize, rows, in_dim), &device)?
                .to_dtype(candle_core::DType::BF16)?;

            let fast = qlinear.forward(&x)?.to_dtype(candle_core::DType::F32)?;
            let fallback = qlinear
                .matmul
                .as_ref()
                .forward(&x.to_dtype(candle_core::DType::F32)?)?
                .to_dtype(candle_core::DType::BF16)?
                .to_dtype(candle_core::DType::F32)?;
            let diff = (fast - fallback)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                diff < 0.08,
                "q8 mmvq fast path diverged from fallback for shape out={out_dim} in={in_dim} rows={rows}: max abs diff {diff}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_q8_mmq_prefill_matches_fallback_on_cuda() -> anyhow::Result<()> {
        use candle_core::Module;

        let device = candle_core::Device::new_cuda(0)?;
        let weight = candle_core::Tensor::randn(
            0f32,
            0.02f32,
            (256usize, 256usize),
            &candle_core::Device::Cpu,
        )?
        .to_dtype(candle_core::DType::BF16)?;
        let linear = IsqLinear::new_with_module_path(
            candle_nn::Linear::new(weight, None),
            Some("q8_0"),
            &device,
            "thinker.model.layers.0.mlp.down_proj",
        )?;
        let LinearX::QLinear(qlinear) = linear else {
            anyhow::bail!("expected QLinear");
        };
        super::super::q8_mmvq::reset_fast_path_stats();
        let x = candle_core::Tensor::randn(0f32, 0.02f32, (1usize, 64usize, 256usize), &device)?
            .to_dtype(candle_core::DType::BF16)?;

        let _guard = set_linear_is_prefill(true);
        let fast = qlinear.forward(&x)?.to_dtype(candle_core::DType::F32)?;
        let stats = super::super::q8_mmvq::fast_path_stats();
        anyhow::ensure!(
            stats.mmq_prefill_calls > 0,
            "expected MMQ prefill path to run, got stats={stats:?}"
        );
        let fallback = qlinear
            .matmul
            .as_ref()
            .forward(&x.to_dtype(candle_core::DType::F32)?)?
            .to_dtype(candle_core::DType::BF16)?
            .to_dtype(candle_core::DType::F32)?;
        let diff = (fast - fallback)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 0.08,
            "q8 mmq prefill path diverged from fallback: max abs diff {diff}"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_q8_mmvq_fused_silu_gate_up_matches_unfused_on_cuda() -> anyhow::Result<()> {
        let device = candle_core::Device::new_cuda(0)?;
        let gate_weight = candle_core::Tensor::randn(0f32, 0.02f32, (64usize, 128usize), &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let up_weight = candle_core::Tensor::randn(0f32, 0.02f32, (64usize, 128usize), &device)?
            .to_dtype(candle_core::DType::BF16)?;
        let gate = IsqLinear::new(
            candle_nn::Linear::new(gate_weight, None),
            Some("q8_0"),
            &device,
        )?;
        let up = IsqLinear::new(
            candle_nn::Linear::new(up_weight, None),
            Some("q8_0"),
            &device,
        )?;
        for rows in [1usize, 8] {
            let x = candle_core::Tensor::randn(0f32, 0.02f32, (1usize, rows, 128usize), &device)?
                .to_dtype(candle_core::DType::BF16)?;

            let fused = try_fused_q8_silu_gate_up(&gate, &up, &x)?
                .ok_or_else(|| anyhow::anyhow!("expected fused q8 gate/up path"))?
                .to_dtype(candle_core::DType::F32)?;
            let gate_y = gate.forward(&x)?;
            let up_y = up.forward(&x)?;
            let unfused = candle_nn::ops::silu(&gate_y)?
                .broadcast_mul(&up_y)?
                .to_dtype(candle_core::DType::F32)?;

            let diff = (fused - unfused)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                diff < 0.05,
                "q8 fused gate/up path diverged from unfused for rows={rows}: max abs diff {diff}"
            );
        }
        for rows in [9usize, 32, 128, 512] {
            let x = candle_core::Tensor::randn(0f32, 0.02f32, (1usize, rows, 128usize), &device)?
                .to_dtype(candle_core::DType::BF16)?;
            assert!(
                try_fused_q8_silu_gate_up(&gate, &up, &x)?.is_none(),
                "q8 fused gate/up should stay on MMVQ-sized decode batches; rows={rows} should fall back to ordinary Q8 Linear/MMQ"
            );
        }
        Ok(())
    }
}
