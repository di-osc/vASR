use anyhow::{Result, bail};
use candle_core::Device;

const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;
const DEFAULT_MAX_NUM_SEQS: usize = 512;
const DEFAULT_GPU_MEMORY_UTILIZATION: f32 = 0.8;
const DEFAULT_ENCODER_MAX_AUDIO_SECONDS: f64 = 30.0;
const DEFAULT_ENCODER_MAX_SEGMENTS: usize = 32;
const SAFETY_MARGIN_RATIO: f64 = 0.10;
const ENCODER_MEMORY_RATIO: f64 = 0.25;
const MAX_AUTO_ENCODER_AUDIO_SECONDS: f64 = 120.0;
const MAX_AUTO_ENCODER_SEGMENTS: usize = 32;
const MAX_AUTO_NUM_SEQS: usize = 512;
const MIN_KV_DECODE_TOKENS: usize = 32;
const ESTIMATED_FIXED_PROMPT_TOKENS: usize = 128;

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
    pub weight_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub encoder_bytes_per_audio_second: u64,
    pub cuda_graph_workspace_bytes: u64,
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
    pub used_fallback: bool,
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
    let Some((hardware, model)) = hardware.zip(model) else {
        return fallback_tuning(vad_max_segment_seconds, vad_pad_ms, overrides);
    };
    if hardware.total_memory_bytes == 0
        || model.kv_bytes_per_token == 0
        || model.encoder_bytes_per_audio_second == 0
    {
        return fallback_tuning(vad_max_segment_seconds, vad_pad_ms, overrides);
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
    let auto_segments = (((memory_after_weights / BYTES_PER_GIB) as usize).saturating_mul(2))
        .clamp(1, MAX_AUTO_ENCODER_SEGMENTS);
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
    let reserved_bytes = safety_margin_bytes
        .saturating_add(encoder_workspace_bytes)
        .saturating_add(graph_workspace_bytes);
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

    let expected_tokens_per_sequence = expected_kv_tokens_per_sequence(
        vad_max_segment_seconds,
        max_hotword_tokens,
        kv_peak_ratio,
    )?;
    let bytes_per_sequence = model
        .kv_bytes_per_token
        .checked_mul(expected_tokens_per_sequence as u64)
        .ok_or_else(|| anyhow::anyhow!("KV bytes-per-sequence estimate overflow"))?;
    let memory_limited_sequences = (kv_budget_bytes / bytes_per_sequence.max(1)) as usize;
    let platform_limit = match hardware.backend {
        Backend::Cpu => hardware.cpu_parallelism.saturating_mul(16).max(1),
        Backend::Metal | Backend::Cuda => MAX_AUTO_NUM_SEQS,
    };
    let capacity_limit = memory_limited_sequences.min(platform_limit);
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
    Ok(ResolvedServeTuning {
        max_num_seqs,
        gpu_memory_utilization: utilization,
        encoder_batch_max_audio_seconds: audio_seconds,
        encoder_batch_max_segments: segments,
        reserved_gpu_bytes,
        used_fallback: false,
    })
}

fn expected_kv_tokens_per_sequence(
    max_segment_seconds: f64,
    max_hotword_tokens: usize,
    peak_ratio: f32,
) -> Result<u64> {
    if !peak_ratio.is_finite() || !(0.0..=1.0).contains(&peak_ratio) {
        bail!("KV peak ratio must be finite and between 0.0 and 1.0");
    }
    let base_tokens = ESTIMATED_FIXED_PROMPT_TOKENS as f64
        + max_hotword_tokens as f64
        + vasr::QWEN3_ASR_DECODE_TOKENS_PER_SEC.max(MIN_KV_DECODE_TOKENS as f64);
    let peak_tokens = ESTIMATED_FIXED_PROMPT_TOKENS as f64
        + max_hotword_tokens as f64
        + max_segment_seconds * 25.0
        + (max_segment_seconds * vasr::QWEN3_ASR_DECODE_TOKENS_PER_SEC)
            .ceil()
            .max(MIN_KV_DECODE_TOKENS as f64);
    let expected = base_tokens + (peak_tokens - base_tokens).max(0.0) * f64::from(peak_ratio);
    if !expected.is_finite() || expected <= 0.0 || expected > u64::MAX as f64 {
        bail!("KV token estimate is out of range");
    }
    Ok(expected.ceil() as u64)
}

fn fallback_tuning(
    vad_max_segment_seconds: f64,
    vad_pad_ms: u64,
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
        used_fallback: true,
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
    isq: Option<&str>,
    backend: Backend,
) -> Option<ModelMemoryProfile> {
    let lower = model.to_ascii_lowercase();
    let (parameters, kv_bytes_per_token) = if lower.contains("1.7b") {
        (1_700_000_000u64, 112 * 1024u64)
    } else if lower.contains("0.6b") {
        (600_000_000u64, 112 * 1024u64)
    } else {
        return None;
    };
    let bytes_per_parameter = match isq.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("4" | "q4_0" | "q4_k") => 0.65,
        Some("8" | "q8_0" | "q8") => 1.1,
        _ if backend == Backend::Cpu => 4.0,
        _ => 2.0,
    };
    Some(ModelMemoryProfile {
        weight_bytes: (parameters as f64 * bytes_per_parameter) as u64,
        kv_bytes_per_token,
        // At the 32-segment baseline this reserves roughly 2 GiB for a 30-second batch.
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
        Backend, HardwareProfile, ModelMemoryProfile, ServeTuningOverrides, resolve_tuning,
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
                weight_bytes: GIB,
                kv_bytes_per_token: 112 * 1024,
                encoder_bytes_per_audio_second: 64 * 1024 * 1024,
                cuda_graph_workspace_bytes: 1024 * 1024 * 1024,
            },
            ServeTuningOverrides::default(),
        )
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
            assert!(high.encoder_batch_max_audio_seconds > low.encoder_batch_max_audio_seconds);
            assert!(high.max_num_seqs <= 512);
            assert_eq!(low.reserved_gpu_bytes, 0);
            assert_eq!(high.reserved_gpu_bytes, 0);
        }
    }
}
