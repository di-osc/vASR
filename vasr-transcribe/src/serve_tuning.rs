use anyhow::{Result, bail};
use candle_core::Device;
use vasr::ExecutionPolicy;

const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_MAX_NUM_SEQS: usize = 512;
const DEFAULT_GPU_MEMORY_UTILIZATION: f32 = 0.8;
const DEFAULT_ENCODER_MAX_AUDIO_SECONDS: f64 = 30.0;
const DEFAULT_ENCODER_MAX_SEGMENTS: usize = 32;
const SAFETY_MARGIN_RATIO: f64 = 0.10;
const ENCODER_MEMORY_RATIO: f64 = 0.25;
const MAX_AUTO_ENCODER_AUDIO_SECONDS: f64 = 120.0;
const MAX_AUTO_ENCODER_SEGMENTS: usize = 32;
/// Metal 自动模式下单批 encoder 的默认段数（完整数据集验证过的候选）。
const METAL_AUTO_ENCODER_SEGMENTS: usize = 8;
const MAX_AUTO_NUM_SEQS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Backend {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct HardwareProfile {
    pub backend: Backend,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub cpu_parallelism: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ModelMemoryProfile {
    /// Set only after checking the measured hardware, canonical model and dtype.
    pub validated_metal_100x: bool,
    pub weight_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub fixed_prompt_tokens: usize,
    pub encoder_bytes_per_audio_second: u64,
    pub cuda_graph_workspace_bytes: u64,
}

#[cfg(any(feature = "metal", test))]
pub(super) fn validated_metal_100x_profile(
    model: &str,
    dtype: candle_core::DType,
    quantized: bool,
    hardware: bool,
    geometry: &vasr::models::qwen3asr::ModelGeometry,
) -> bool {
    matches!(
        model.trim().to_ascii_lowercase().as_str(),
        "qwen3-asr-0.6b" | "qwen/qwen3-asr-0.6b"
    ) && vasr::models::qwen3asr::ValidatedMetalProfile::resolve(
        geometry, dtype, quantized, hardware,
    )
    .conv_layout
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ServeTuningOverrides {
    pub max_num_seqs: Option<usize>,
    pub gpu_memory_utilization: Option<f32>,
    pub encoder_batch_max_audio_seconds: Option<f64>,
    pub encoder_batch_max_segments: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ResolvedServeTuning {
    pub max_num_seqs: usize,
    pub gpu_memory_utilization: f32,
    pub encoder_batch_max_audio_seconds: f64,
    pub encoder_batch_max_segments: usize,
    pub reserved_gpu_bytes: usize,
    pub source_memory_bytes: usize,
    pub used_fallback: bool,
    /// 启动时解析一次的统一执行策略，随模型与 runtime 传递。
    pub execution: ExecutionPolicy,
}

pub(super) fn resolve_tuning(
    hardware: Option<HardwareProfile>,
    model: Option<ModelMemoryProfile>,
    vad_max_segment_seconds: f64,
    vad_pad_ms: u64,
    max_hotword_tokens: usize,
    kv_peak_ratio: f32,
    overrides: ServeTuningOverrides,
) -> Result<ResolvedServeTuning> {
    let is_metal = hardware.is_some_and(|hardware| hardware.backend == Backend::Metal);
    let Some((hardware, model)) = hardware.zip(model) else {
        return fallback_tuning(vad_max_segment_seconds, vad_pad_ms, is_metal, overrides);
    };
    if hardware.total_memory_bytes == 0
        || model.kv_bytes_per_token == 0
        || model.encoder_bytes_per_audio_second == 0
    {
        return fallback_tuning(vad_max_segment_seconds, vad_pad_ms, is_metal, overrides);
    }
    if hardware.available_memory_bytes == 0 {
        bail!("insufficient memory: selected device reports no available memory");
    }

    let vad_minimum_audio_seconds = vad_max_segment_seconds + 2.0 * vad_pad_ms as f64 / 1000.0;
    if !vad_minimum_audio_seconds.is_finite() || vad_minimum_audio_seconds <= 0.0 {
        bail!("maximum VAD segment duration must be finite and greater than zero");
    }

    let memory_after_weights = hardware
        .available_memory_bytes
        .saturating_sub(model.weight_bytes);
    let safety_margin_bytes = (memory_after_weights as f64 * SAFETY_MARGIN_RATIO).ceil() as u64;
    let encoder_budget_bytes = (memory_after_weights as f64 * ENCODER_MEMORY_RATIO).floor() as u64;
    // 多段 encoder 批量：此前 Metal 上出现 CER 退化的根因是 encoder/prefill/decode
    // 的 Metal forward 未串行化（见 docs/performance/m4-pro-qwen3-asr-2026-09-29.md），
    // 该问题已由共享 ModelExecutionLock 修复。这里放开到验证过的默认段数，
    // 并用 VASR_ASR_ENCODER_BATCH_MAX_SEGMENTS 继续做 4/8/16/32 的精度扫描。
    let auto_segments = if hardware.backend == Backend::Metal {
        if model.validated_metal_100x {
            16
        } else {
            METAL_AUTO_ENCODER_SEGMENTS
        }
    } else {
        (((memory_after_weights / BYTES_PER_GIB) as usize).saturating_mul(2))
            .clamp(1, MAX_AUTO_ENCODER_SEGMENTS)
    };
    let segments = overrides
        .encoder_batch_max_segments
        .unwrap_or(auto_segments);
    if segments == 0 {
        bail!("encoder batch max segments must be greater than zero");
    }

    let seconds_per_segment_scale = segments as f64 / MAX_AUTO_ENCODER_SEGMENTS as f64;
    let auto_audio_seconds = (encoder_budget_bytes as f64
        / model.encoder_bytes_per_audio_second as f64
        / seconds_per_segment_scale)
        .floor()
        .clamp(
            vad_minimum_audio_seconds,
            MAX_AUTO_ENCODER_AUDIO_SECONDS.max(vad_minimum_audio_seconds),
        );
    let audio_seconds = overrides
        .encoder_batch_max_audio_seconds
        .unwrap_or(auto_audio_seconds);
    if !audio_seconds.is_finite() || audio_seconds < vad_minimum_audio_seconds {
        bail!(
            "encoder batch duration must be finite and at least {:.3}s to hold the maximum VAD segment and padding",
            vad_minimum_audio_seconds
        );
    }

    let encoder_workspace_bytes =
        (model.encoder_bytes_per_audio_second as f64 * audio_seconds * seconds_per_segment_scale)
            .ceil() as u64;
    let graph_workspace_bytes = if hardware.backend == Backend::Cuda {
        model.cuda_graph_workspace_bytes
    } else {
        0
    };
    // CUDA request bodies and encoded audio live in host RAM, not device VRAM.
    let source_available = if hardware.backend == Backend::Cuda {
        probe_host_memory().ok().map(|(_, available)| available)
    } else {
        Some(hardware.available_memory_bytes)
    };
    let source_memory_bytes = crate::pipeline::source_memory_budget(source_available);
    let reserved_bytes = safety_margin_bytes
        .saturating_add(encoder_workspace_bytes)
        .saturating_add(graph_workspace_bytes)
        .saturating_add(if hardware.backend == Backend::Cuda {
            0
        } else {
            source_memory_bytes as u64
        });
    let mut kv_budget_bytes = memory_after_weights.saturating_sub(reserved_bytes);
    let utilization =
        overrides
            .gpu_memory_utilization
            .unwrap_or(if hardware.backend == Backend::Cuda {
                1.0
            } else {
                DEFAULT_GPU_MEMORY_UTILIZATION
            });
    if !utilization.is_finite() || utilization <= 0.0 || utilization > 1.0 {
        bail!("gpu memory utilization must be finite and in (0.0, 1.0]");
    }
    if hardware.backend == Backend::Cuda {
        let memory_used_after_weights = hardware
            .total_memory_bytes
            .saturating_sub(memory_after_weights);
        let fractional_budget =
            (hardware.total_memory_bytes as f64 * f64::from(utilization)) as u64;
        kv_budget_bytes = kv_budget_bytes.min(
            fractional_budget
                .saturating_sub(memory_used_after_weights)
                .saturating_sub(reserved_bytes),
        );
    }

    let platform_limit = match hardware.backend {
        Backend::Cpu => hardware.cpu_parallelism.saturating_mul(16).max(1),
        Backend::Metal | Backend::Cuda => MAX_AUTO_NUM_SEQS,
    };
    // Use exactly the same prompt/audio lengths, block rounding and peak-ratio
    // rounding as the model builder. Reserve one null block in the physical pool.
    let required_bytes = |sequences| -> Result<u64> {
        let tokens = vasr::models::qwen3asr::builder::kv_cache_tokens_for_sequences(
            model.fixed_prompt_tokens,
            max_hotword_tokens,
            vad_minimum_audio_seconds as f32,
            kv_peak_ratio,
            sequences,
        )?;
        model
            .kv_bytes_per_token
            .checked_mul(
                (tokens as u64)
                    .saturating_add(vasr::models::qwen3asr::builder::KV_BLOCK_SIZE as u64),
            )
            .ok_or_else(|| anyhow::anyhow!("KV byte estimate overflow"))
    };
    let mut low = 0usize;
    let mut high = platform_limit;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if required_bytes(middle)? <= kv_budget_bytes {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let capacity_limit = low;
    if capacity_limit == 0 {
        bail!(
            "insufficient memory for one ASR sequence after reserving encoder workspace and a 10% safety margin"
        );
    }
    let max_num_seqs = overrides.max_num_seqs.unwrap_or(capacity_limit);
    if max_num_seqs == 0 || max_num_seqs > capacity_limit {
        bail!(
            "--max-num-seqs={max_num_seqs} exceeds the estimated safe capacity of {capacity_limit} sequences"
        );
    }

    let reserved_gpu_bytes = if hardware.backend == Backend::Cuda {
        usize::try_from(reserved_bytes).unwrap_or(usize::MAX)
    } else {
        0
    };
    // 内存安全容量与执行批量分开：容量用于 KV 规划，执行批量按统一策略收窄。
    let (auto_slots, auto_wave) = if model.validated_metal_100x {
        (64, 16)
    } else {
        (
            vasr::models::qwen3asr::inference::execution::METAL_AUTO_MAX_ACTIVE_SLOTS,
            vasr::models::qwen3asr::inference::execution::METAL_AUTO_MAX_PREFILL_WAVE,
        )
    };
    let execution = ExecutionPolicy::resolve_backend_with_limits(
        hardware.backend == Backend::Metal,
        max_num_seqs,
        overrides.max_num_seqs.is_some(),
        auto_slots,
        auto_wave,
    );
    Ok(ResolvedServeTuning {
        max_num_seqs,
        gpu_memory_utilization: utilization,
        encoder_batch_max_audio_seconds: audio_seconds,
        encoder_batch_max_segments: segments,
        reserved_gpu_bytes,
        source_memory_bytes,
        used_fallback: false,
        execution,
    })
}

fn fallback_tuning(
    vad_max_segment_seconds: f64,
    vad_pad_ms: u64,
    is_metal: bool,
    overrides: ServeTuningOverrides,
) -> Result<ResolvedServeTuning> {
    let minimum_audio_seconds = vad_max_segment_seconds + 2.0 * vad_pad_ms as f64 / 1000.0;
    let audio_seconds = overrides
        .encoder_batch_max_audio_seconds
        .unwrap_or(DEFAULT_ENCODER_MAX_AUDIO_SECONDS.max(minimum_audio_seconds));
    if !audio_seconds.is_finite() || audio_seconds < minimum_audio_seconds {
        bail!(
            "encoder batch duration must be finite and at least {:.3}s to hold the maximum VAD segment and padding",
            minimum_audio_seconds
        );
    }
    let utilization = overrides
        .gpu_memory_utilization
        .unwrap_or(DEFAULT_GPU_MEMORY_UTILIZATION);
    if !utilization.is_finite() || utilization <= 0.0 || utilization > 1.0 {
        bail!("gpu memory utilization must be finite and in (0.0, 1.0]");
    }
    let max_num_seqs = overrides.max_num_seqs.unwrap_or(DEFAULT_MAX_NUM_SEQS);
    if max_num_seqs == 0 {
        bail!("--max-num-seqs must be greater than zero");
    }
    let segments = overrides
        .encoder_batch_max_segments
        .unwrap_or(DEFAULT_ENCODER_MAX_SEGMENTS);
    if segments == 0 {
        bail!("encoder batch max segments must be greater than zero");
    }
    Ok(ResolvedServeTuning {
        max_num_seqs,
        gpu_memory_utilization: utilization,
        encoder_batch_max_audio_seconds: audio_seconds,
        encoder_batch_max_segments: segments,
        reserved_gpu_bytes: 0,
        source_memory_bytes: crate::pipeline::source_memory_budget(None),
        used_fallback: true,
        execution: ExecutionPolicy::resolve_backend(
            is_metal,
            max_num_seqs,
            overrides.max_num_seqs.is_some(),
        ),
    })
}

pub(super) fn probe_hardware(device: &Device) -> Result<HardwareProfile> {
    let cpu_parallelism = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    if device.is_cuda() {
        #[cfg(all(feature = "cuda", feature = "cuda-graph", feature = "paged-attn"))]
        if let Some(memory) = vasr::vasr_paged_attn::cuda_graph::cuda_memory_snapshot(device) {
            return Ok(HardwareProfile {
                backend: Backend::Cuda,
                total_memory_bytes: memory.total as u64,
                available_memory_bytes: memory.free as u64,
                cpu_parallelism,
            });
        }
        bail!("CUDA memory probe is unavailable for the selected device");
    }

    let (total_memory_bytes, available_memory_bytes) = probe_host_memory()?;
    Ok(HardwareProfile {
        backend: if device.is_metal() {
            Backend::Metal
        } else {
            Backend::Cpu
        },
        total_memory_bytes,
        available_memory_bytes,
        cpu_parallelism,
    })
}

pub(super) fn estimate_model_memory_profile(
    model: &str,
    dtype: candle_core::DType,
    language: Option<&str>,
    backend: Backend,
) -> Result<ModelMemoryProfile> {
    use vasr::models::qwen3asr::{Qwen3ASR, resolve_qwen3_asr_model_id};
    let model_id = resolve_qwen3_asr_model_id(model);
    let config = Qwen3ASR::pretrained_config(model_id)?;
    let text = &config.thinker_config.text_config;
    let kv_bytes_per_token = [
        text.num_hidden_layers,
        text.num_key_value_heads,
        text.head_dim,
        2,
        dtype.size_in_bytes(),
    ]
    .into_iter()
    .try_fold(1u64, |product, n| product.checked_mul(n as u64))
    .ok_or_else(|| anyhow::anyhow!("KV bytes per token overflow"))?;
    Ok(ModelMemoryProfile {
        validated_metal_100x: false,
        // Quantized models keep some tensors dense; use the full dense checkpoint
        // as a conservative estimate until the loader can report resident ISQ bytes.
        weight_bytes: Qwen3ASR::pretrained_parameter_bytes(model_id, dtype)? as u64,
        kv_bytes_per_token,
        fixed_prompt_tokens: vasr::models::qwen3asr::builder::kv_cache_fixed_prompt_tokens(
            model_id, language,
        )?,
        encoder_bytes_per_audio_second: 64 * 1024 * 1024,
        cuda_graph_workspace_bytes: if backend == Backend::Cuda
            && cfg!(feature = "cuda-graph")
            && std::env::var_os("VASR_DISABLE_CUDA_GRAPH").is_none()
        {
            1024 * 1024 * 1024
        } else {
            0
        },
    })
}

#[cfg(target_os = "linux")]
fn probe_host_memory() -> Result<(u64, u64)> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let mut total = None;
    let mut available = None;
    for line in meminfo.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let kib = value
            .split_whitespace()
            .next()
            .and_then(|value| value.parse::<u64>().ok());
        match key {
            "MemTotal" => total = kib.map(|value| value.saturating_mul(1024)),
            "MemAvailable" => available = kib.map(|value| value.saturating_mul(1024)),
            _ => {}
        }
    }
    match (total, available) {
        (Some(total), Some(available)) if total > 0 && available > 0 => {
            if let Some((limit, cgroup_available)) = probe_cgroup_memory()
                && limit < total
            {
                return Ok((limit, available.min(cgroup_available).min(limit)));
            }
            Ok((total, available))
        }
        _ => bail!("/proc/meminfo does not report usable total and available memory"),
    }
}

#[cfg(target_os = "linux")]
fn probe_cgroup_memory() -> Option<(u64, u64)> {
    let limit_v2 = std::fs::read_to_string("/sys/fs/cgroup/memory.max").ok();
    let current_v2 = std::fs::read_to_string("/sys/fs/cgroup/memory.current").ok();
    if let (Some(limit), Some(current)) = (limit_v2, current_v2)
        && let (Ok(limit), Ok(current)) =
            (limit.trim().parse::<u64>(), current.trim().parse::<u64>())
        && limit < u64::MAX / 2
    {
        return Some((limit, limit.saturating_sub(current)));
    }

    let limit_v1 = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.limit_in_bytes").ok()?;
    let current_v1 = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.usage_in_bytes").ok()?;
    let limit = limit_v1.trim().parse::<u64>().ok()?;
    let current = current_v1.trim().parse::<u64>().ok()?;
    (limit < u64::MAX / 2).then_some((limit, limit.saturating_sub(current)))
}

#[cfg(target_os = "macos")]
fn probe_host_memory() -> Result<(u64, u64)> {
    use std::process::Command;

    let total_output = Command::new("sysctl").args(["-n", "hw.memsize"]).output()?;
    if !total_output.status.success() {
        bail!("sysctl could not read hw.memsize");
    }
    let total = String::from_utf8_lossy(&total_output.stdout)
        .trim()
        .parse::<u64>()?;
    let vm_output = Command::new("vm_stat").output()?;
    if !vm_output.status.success() {
        bail!("vm_stat could not report available memory");
    }
    let text = String::from_utf8_lossy(&vm_output.stdout);
    let page_size = text
        .lines()
        .next()
        .and_then(|line| line.split("page size of ").nth(1))
        .and_then(|value| value.split(" bytes").next())
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| anyhow::anyhow!("vm_stat did not report page size"))?;
    let mut available_pages = 0u64;
    for line in text.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if ["Pages free", "Pages inactive", "Pages speculative"].contains(&name.trim()) {
            let pages = value
                .trim()
                .trim_end_matches('.')
                .replace('.', "")
                .parse::<u64>()?;
            available_pages = available_pages.saturating_add(pages);
        }
    }
    let available = available_pages.saturating_mul(page_size);
    if total == 0 || available == 0 {
        bail!("vm_stat did not report usable available memory");
    }
    Ok((total, available.min(total)))
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn probe_host_memory() -> Result<(u64, u64)> {
    bail!("host memory probing is unsupported on this operating system")
}

#[cfg(test)]
mod tests {
    use super::{
        Backend, HardwareProfile, METAL_AUTO_ENCODER_SEGMENTS, ModelMemoryProfile,
        ServeTuningOverrides, resolve_tuning,
    };

    const GIB: u64 = 1024 * 1024 * 1024;

    fn input(memory_gib: u64) -> (HardwareProfile, ModelMemoryProfile, ServeTuningOverrides) {
        (
            HardwareProfile {
                backend: Backend::Cuda,
                total_memory_bytes: memory_gib * GIB,
                available_memory_bytes: memory_gib * GIB,
                cpu_parallelism: 16,
            },
            ModelMemoryProfile {
                validated_metal_100x: false,
                weight_bytes: GIB,
                kv_bytes_per_token: 112 * 1024,
                fixed_prompt_tokens: 128,
                encoder_bytes_per_audio_second: 64 * 1024 * 1024,
                cuda_graph_workspace_bytes: 1024 * 1024 * 1024,
            },
            ServeTuningOverrides::default(),
        )
    }

    #[test]
    #[ignore = "requires VASR_TEST_MODEL_DIR pointing to a complete Qwen3-ASR checkpoint"]
    fn checkpoint_profile_and_capacity_without_device_allocation() -> anyhow::Result<()> {
        let path = std::env::var("VASR_TEST_MODEL_DIR")?;
        let model = super::estimate_model_memory_profile(
            &path,
            candle_core::DType::BF16,
            None,
            Backend::Cuda,
        )?;
        let hardware = HardwareProfile {
            backend: Backend::Cuda,
            total_memory_bytes: 24 * GIB,
            available_memory_bytes: 24 * GIB,
            cpu_parallelism: 16,
        };
        let tuning = resolve_tuning(
            Some(hardware),
            Some(model),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides::default(),
        )?;
        assert!(model.weight_bytes > 0);
        assert!(model.fixed_prompt_tokens > 0);
        println!(
            "checkpoint weight_bytes={} kv_bytes_per_token={} fixed_prompt_tokens={}\nsimulated free 24 GiB: max_num_seqs={} reserved_gpu_bytes={} encoder_seconds={}",
            model.weight_bytes,
            model.kv_bytes_per_token,
            model.fixed_prompt_tokens,
            tuning.max_num_seqs,
            tuning.reserved_gpu_bytes,
            tuning.encoder_batch_max_audio_seconds
        );
        Ok(())
    }

    #[test]
    fn cuda_reserves_device_workspace_but_not_host_source_buffers() {
        let (hardware, model, overrides) = input(24);
        let tuning =
            resolve_tuning(Some(hardware), Some(model), 30.0, 0, 128, 0.25, overrides).unwrap();
        let safety =
            ((hardware.available_memory_bytes - model.weight_bytes) as f64 * 0.1).ceil() as u64;
        let encoder = (model.encoder_bytes_per_audio_second as f64
            * tuning.encoder_batch_max_audio_seconds
            * tuning.encoder_batch_max_segments as f64
            / 32.0)
            .ceil() as u64;
        assert_eq!(
            tuning.reserved_gpu_bytes as u64,
            safety + encoder + model.cuda_graph_workspace_bytes
        );
        assert!(tuning.source_memory_bytes > 0);
    }

    #[test]
    fn capacity_is_the_largest_count_that_fits_exact_block_budget() {
        let (hardware, model, overrides) = input(24);
        let tuning =
            resolve_tuning(Some(hardware), Some(model), 30.0, 0, 128, 0.25, overrides).unwrap();
        let budget =
            hardware.available_memory_bytes - model.weight_bytes - tuning.reserved_gpu_bytes as u64;
        let bytes = |n| {
            (vasr::models::qwen3asr::builder::kv_cache_tokens_for_sequences(
                model.fixed_prompt_tokens,
                128,
                30.0,
                0.25,
                n,
            )
            .unwrap() as u64
                + 32)
                * model.kv_bytes_per_token
        };
        assert!(bytes(tuning.max_num_seqs) <= budget);
        assert!(bytes(tuning.max_num_seqs + 1) > budget);
        let allowed = resolve_tuning(
            Some(hardware),
            Some(model),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides {
                max_num_seqs: Some(tuning.max_num_seqs),
                ..overrides
            },
        );
        assert!(allowed.is_ok());
        let rejected = resolve_tuning(
            Some(hardware),
            Some(model),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides {
                max_num_seqs: Some(tuning.max_num_seqs + 1),
                ..overrides
            },
        );
        assert!(rejected.is_err());
    }

    #[test]
    fn larger_memory_increases_automatic_capacity() {
        let (low_hw, profile, overrides) = input(8);
        let (high_hw, _, _) = input(32);
        let low = resolve_tuning(Some(low_hw), Some(profile), 30.0, 0, 128, 0.25, overrides)
            .expect("8 GiB profile should resolve");
        let high = resolve_tuning(Some(high_hw), Some(profile), 30.0, 0, 128, 0.25, overrides)
            .expect("32 GiB profile should resolve");

        assert!(high.max_num_seqs > low.max_num_seqs);
        assert!(high.encoder_batch_max_audio_seconds > low.encoder_batch_max_audio_seconds);
        assert!(high.encoder_batch_max_segments > low.encoder_batch_max_segments);
    }

    #[test]
    fn explicit_values_override_the_automatic_values() {
        let (hardware, profile, _) = input(32);
        let resolved = resolve_tuning(
            Some(hardware),
            Some(profile),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides {
                max_num_seqs: Some(8),
                gpu_memory_utilization: Some(0.75),
                encoder_batch_max_audio_seconds: Some(40.0),
                encoder_batch_max_segments: Some(7),
            },
        )
        .expect("overrides should fit a 32 GiB profile");

        assert_eq!(resolved.max_num_seqs, 8);
        assert_eq!(resolved.gpu_memory_utilization, 0.75);
        assert_eq!(resolved.encoder_batch_max_audio_seconds, 40.0);
        assert_eq!(resolved.encoder_batch_max_segments, 7);
    }

    #[test]
    fn cuda_reservation_includes_ten_percent_margin_encoder_and_graph_workspace() {
        let (hardware, profile, overrides) = input(32);
        let resolved = resolve_tuning(Some(hardware), Some(profile), 30.0, 0, 128, 0.25, overrides)
            .expect("large CUDA profile should resolve");
        let estimated_weight_remainder = hardware.available_memory_bytes - profile.weight_bytes;
        let ten_percent = (estimated_weight_remainder as f64 * 0.10).ceil() as usize;

        assert!(resolved.reserved_gpu_bytes >= ten_percent);
        assert!(resolved.reserved_gpu_bytes >= profile.cuda_graph_workspace_bytes as usize);
        assert!(resolved.reserved_gpu_bytes < estimated_weight_remainder as usize);
    }

    #[test]
    fn explicit_sequence_count_above_estimated_capacity_fails() {
        let (hardware, profile, _) = input(8);
        let error = resolve_tuning(
            Some(hardware),
            Some(profile),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides {
                max_num_seqs: Some(512),
                ..ServeTuningOverrides::default()
            },
        )
        .expect_err("manual concurrency above the safe profile should fail");
        assert!(error.to_string().contains("estimated safe capacity"));
    }

    #[test]
    fn capacity_below_one_vad_segment_fails_instead_of_clamping() {
        let (mut hardware, profile, overrides) = input(2);
        hardware.available_memory_bytes = 256 * 1024 * 1024;
        hardware.total_memory_bytes = 2 * GIB;

        let error = resolve_tuning(Some(hardware), Some(profile), 30.0, 0, 128, 0.25, overrides)
            .expect_err("too little memory should fail startup");
        assert!(error.to_string().contains("insufficient memory"));
    }

    #[test]
    fn known_zero_available_memory_is_an_error_not_probe_fallback() {
        let (mut hardware, profile, overrides) = input(8);
        hardware.available_memory_bytes = 0;
        let error = resolve_tuning(Some(hardware), Some(profile), 30.0, 0, 128, 0.25, overrides)
            .expect_err("known zero free memory must fail startup");
        assert!(error.to_string().contains("no available memory"));
    }

    #[test]
    fn probe_failure_uses_baseline_defaults_and_preserves_manual_values() {
        let resolved = resolve_tuning(
            None,
            None,
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides::default(),
        )
        .expect("probe failure should use fallback");
        assert_eq!(resolved.max_num_seqs, 512);
        assert_eq!(resolved.gpu_memory_utilization, 0.8);
        assert_eq!(resolved.encoder_batch_max_audio_seconds, 30.0);
        assert_eq!(resolved.encoder_batch_max_segments, 32);
        assert!(resolved.used_fallback);
    }

    #[test]
    fn encoder_duration_covers_vad_maximum_and_padding() {
        let (hardware, profile, _) = input(16);
        let resolved = resolve_tuning(
            Some(hardware),
            Some(profile),
            30.0,
            250,
            128,
            0.25,
            ServeTuningOverrides::default(),
        )
        .expect("automatic duration should cover padded VAD segment");
        assert!(resolved.encoder_batch_max_audio_seconds >= 30.5);

        let error = resolve_tuning(
            Some(hardware),
            Some(profile),
            30.0,
            250,
            128,
            0.25,
            ServeTuningOverrides {
                encoder_batch_max_audio_seconds: Some(30.0),
                ..ServeTuningOverrides::default()
            },
        )
        .expect_err("manual duration must not truncate the VAD segment");
        assert!(error.to_string().contains("at least 30.500s"));
    }

    #[test]
    fn long_vad_segments_raise_the_automatic_encoder_duration_cap() {
        let (hardware, profile, overrides) = input(128);
        let resolved = resolve_tuning(
            Some(hardware),
            Some(profile),
            150.0,
            0,
            128,
            0.25,
            overrides,
        )
        .expect("long VAD segments should be reflected in the automatic batch duration");
        assert!(resolved.encoder_batch_max_audio_seconds >= 150.0);
    }

    #[test]
    fn cpu_and_metal_profiles_resolve_with_safe_capacity_limits() {
        for backend in [Backend::Cpu, Backend::Metal] {
            let (mut low_hardware, profile, _) = input(8);
            let mut high_hardware = input(32).0;
            low_hardware.backend = backend;
            high_hardware.backend = backend;
            let low = resolve_tuning(
                Some(low_hardware),
                Some(profile),
                30.0,
                0,
                128,
                0.25,
                ServeTuningOverrides::default(),
            )
            .expect("lower CPU/Metal profile should resolve");
            let high = resolve_tuning(
                Some(high_hardware),
                Some(profile),
                30.0,
                0,
                128,
                0.25,
                ServeTuningOverrides::default(),
            )
            .expect("higher CPU/Metal profile should resolve");

            assert!(high.max_num_seqs > low.max_num_seqs);
            if backend == Backend::Cpu {
                assert!(high.encoder_batch_max_audio_seconds > low.encoder_batch_max_audio_seconds);
            } else {
                assert_eq!(high.encoder_batch_max_segments, METAL_AUTO_ENCODER_SEGMENTS);
            }
            assert!(high.max_num_seqs <= 512);
            assert_eq!(low.reserved_gpu_bytes, 0);
            assert_eq!(high.reserved_gpu_bytes, 0);
        }
    }

    #[test]
    fn measured_metal_profile_resolves_full_policy_and_honors_capacity() {
        let (mut hardware, mut profile, _) = input(48);
        hardware.backend = Backend::Metal;
        profile.validated_metal_100x = true;
        let tuning = resolve_tuning(
            Some(hardware),
            Some(profile),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides::default(),
        )
        .unwrap();
        assert_eq!(tuning.encoder_batch_max_segments, 16);
        assert_eq!(tuning.execution.active_slots, 64);
        assert_eq!(tuning.execution.decode_rows, 64);
        assert_eq!(tuning.execution.prefill_wave, 16);
        let limited = resolve_tuning(
            Some(hardware),
            Some(profile),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides {
                max_num_seqs: Some(4),
                encoder_batch_max_segments: Some(7),
                ..ServeTuningOverrides::default()
            },
        )
        .unwrap();
        assert_eq!(limited.encoder_batch_max_segments, 7);
        assert_eq!(limited.execution.active_slots, 4);
        assert_eq!(limited.execution.decode_rows, 4);
        assert_eq!(limited.execution.prefill_wave, 4);
    }

    fn measured_geometry() -> vasr::models::qwen3asr::ModelGeometry {
        vasr::models::qwen3asr::ModelGeometry {
            model_type: Some("qwen3_asr_thinker".into()),
            audio_layers: 18,
            audio_hidden: 896,
            audio_output: 1024,
            text_layers: 28,
            text_hidden: 1024,
            attention_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            mel_bins: 128,
            downsample_hidden: 480,
            audio_window: 50,
        }
    }

    #[test]
    fn canonical_name_does_not_qualify_a_different_checkpoint_geometry() {
        let mut geometry = measured_geometry();
        geometry.text_hidden = 777;
        assert!(!super::validated_metal_100x_profile(
            "Qwen3-ASR-0.6B",
            candle_core::DType::BF16,
            false,
            true,
            &geometry
        ));
        geometry = measured_geometry();
        geometry.audio_window = 100;
        assert!(!super::validated_metal_100x_profile(
            "Qwen3-ASR-0.6B",
            candle_core::DType::BF16,
            false,
            true,
            &geometry
        ));
    }

    #[test]
    fn measured_metal_profile_rejects_other_models_dtypes_and_quantization() {
        use super::validated_metal_100x_profile as eligible;
        use candle_core::DType;
        assert!(eligible(
            "Qwen3-ASR-0.6B",
            DType::BF16,
            false,
            true,
            &measured_geometry()
        ));
        assert!(eligible(
            "Qwen/Qwen3-ASR-0.6B",
            DType::BF16,
            false,
            true,
            &measured_geometry()
        ));
        for (model, dtype, quantized, hardware) in [
            ("Qwen3-ASR-1.7B", DType::BF16, false, true),
            ("Other/Qwen3-ASR-0.6B", DType::BF16, false, true),
            ("/tmp/Qwen3-ASR-0.6B", DType::BF16, false, true),
            ("Qwen3-ASR-0.6B", DType::F16, false, true),
            ("Qwen3-ASR-0.6B", DType::BF16, true, true),
            ("Qwen3-ASR-0.6B", DType::BF16, false, false),
        ] {
            assert!(!eligible(
                model,
                dtype,
                quantized,
                hardware,
                &measured_geometry()
            ));
        }
    }

    #[test]
    fn metal_encoder_batch_defaults_to_validated_segments_and_honors_override() {
        let (mut hardware, profile, _) = input(32);
        hardware.backend = Backend::Metal;
        let automatic = resolve_tuning(
            Some(hardware),
            Some(profile),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides::default(),
        )
        .expect("Metal profile should resolve");
        assert_eq!(
            automatic.encoder_batch_max_segments,
            METAL_AUTO_ENCODER_SEGMENTS
        );

        let overridden = resolve_tuning(
            Some(hardware),
            Some(profile),
            30.0,
            0,
            128,
            0.25,
            ServeTuningOverrides {
                encoder_batch_max_segments: Some(7),
                ..ServeTuningOverrides::default()
            },
        )
        .expect("explicit Metal encoder batching override should be retained");
        assert_eq!(overridden.encoder_batch_max_segments, 7);
    }
}
