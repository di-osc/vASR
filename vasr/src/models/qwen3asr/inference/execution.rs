//! 统一执行策略：把“内存安全容量”“实际执行批量”和“设备执行保护”分开管理。
//!
//! 过去这些限制散落在服务侧和调度器内部，各自读取环境变量或退回内部默认值，
//! 导致自动模式在 Metal 上永远只跑 1 个活跃槽位、1 行 decode，并且把服务解析出的
//! 安全容量丢弃掉。这个模块把它们收敛成一个在启动时解析一次、随后沿模型与
//! paged runtime 传递的不可变策略。

use candle_core::Device;

/// Optional CUDA forward serialization for controlled throughput experiments.
/// Metal always uses the shared model gate for correctness.
pub(crate) fn model_execution_gate_enabled(device: &Device) -> bool {
    static CUDA_SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    device.is_metal()
        || (device.is_cuda()
            && *CUDA_SERIAL.get_or_init(|| {
                std::env::var("VASR_CUDA_SERIAL_FORWARD")
                    .ok()
                    .is_some_and(|value| {
                        matches!(
                            value.trim().to_ascii_lowercase().as_str(),
                            "1" | "true" | "on"
                        )
                    })
            }))
}

/// 未验证 Metal 组合的保守活跃槽位上限。
///
/// 2026-09-30 的算子与批量扫描选出 32 槽、wave 8、encoder 8：
/// `docs/performance/m4-pro-qwen3-asr-2026-09-30.md`。2026-10-08 已验证的
/// M4 Pro / 48 GB、Qwen3-ASR-0.6B、BF16 组合由服务传入 64 槽和 wave 16；
/// 通用入口继续使用此回退。显式执行参数优先，始终受安全容量约束。
pub const METAL_AUTO_MAX_ACTIVE_SLOTS: usize = 32;

/// 未验证 Metal 组合的单次 prefill wave 上限。
///
/// 容量和 prefill wave 是独立限制：即使 KV block 允许更多序列，也不让单次
/// prefill 一次执行过多行。已验证组合的 wave 16 也独立于安全容量。
pub const METAL_AUTO_MAX_PREFILL_WAVE: usize = 8;

/// 一次服务启动解析出的统一执行策略。
///
/// 策略在启动时解析一次，随后通过 [`crate::AsrOptions`] 和
/// [`crate::models::qwen3asr::inference::types::TranscribeOptions`] 传到调度器；
/// 调度器不再各自读取环境变量或退回内部默认值。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPolicy {
    /// 并发解码的活跃槽位数，是内存安全容量的子集。
    pub active_slots: usize,
    /// 单次 decode forward 的行数，始终不大于 `active_slots`。
    pub decode_rows: usize,
    /// 是否允许把处于不同 decode step 的槽位合并到同一次 forward。
    pub mixed_step: bool,
    /// 单次 prefill wave 允许打包的最大序列数。
    pub prefill_wave: usize,
}

impl ExecutionPolicy {
    /// 从内存安全容量与设备类型推导执行策略。
    ///
    /// * `capacity`：服务解析出的内存安全容量，同时用于 KV block 规划。
    /// * `explicit_capacity`：容量是否由用户显式指定。自动模式下 Metal 会把活跃
    ///   槽位收窄到已验证的默认上限；显式指定时按容量实际执行，方便容量扫描。
    ///
    /// 非 Metal 组合保持既有行为（容量即活跃槽位、prefill wave 等于容量），
    /// 即“未验证组合保留保守回退”。
    pub fn resolve(device: &Device, capacity: usize, explicit_capacity: bool) -> Self {
        Self::resolve_backend(device.is_metal(), capacity, explicit_capacity)
    }

    /// 与 [`Self::resolve`] 相同，但直接使用后端判断，方便启动阶段在拿到
    /// `Device` 之前（或只有后端枚举时）解析策略。
    pub fn resolve_backend(is_metal: bool, capacity: usize, explicit_capacity: bool) -> Self {
        Self::resolve_backend_with_limits(
            is_metal,
            capacity,
            explicit_capacity,
            METAL_AUTO_MAX_ACTIVE_SLOTS,
            METAL_AUTO_MAX_PREFILL_WAVE,
        )
    }

    /// Resolve a measured automatic batch policy without changing safe KV capacity
    /// or overriding explicit execution settings. Other backends ignore these limits.
    pub fn resolve_backend_with_limits(
        is_metal: bool,
        capacity: usize,
        explicit_capacity: bool,
        auto_slots: usize,
        auto_prefill_wave: usize,
    ) -> Self {
        ExecutionOverrides::capture().resolve_backend(
            is_metal,
            capacity,
            explicit_capacity,
            auto_slots,
            auto_prefill_wave,
        )
    }

    /// 把策略约束到给定的活跃槽位数以内，保证不越过内存安全容量。
    #[must_use]
    pub fn clamped_to(self, active_slots: usize) -> Self {
        let active_slots = active_slots.max(1);
        Self {
            active_slots: self.active_slots.min(active_slots).max(1),
            decode_rows: self.decode_rows.min(active_slots).max(1),
            mixed_step: self.mixed_step,
            prefill_wave: self.prefill_wave.min(active_slots).max(1),
        }
    }

    /// 解析一次并打印最终生效的策略，便于性能报告复现配置。
    pub fn log_resolved(&self, target: &str, device_label: &str, capacity: usize) {
        tracing::info!(
            target: "vasr::execution",
            "{} 执行策略 | device={} capacity={} active_slots={} decode_rows={} mixed_step={} prefill_wave={}",
            target,
            device_label,
            capacity,
            self.active_slots,
            self.decode_rows,
            self.mixed_step,
            self.prefill_wave,
        );
    }
}

/// Captured once by a scheduler config, so fallback execution never re-reads
/// environment variables during prefill or decode. Public policy literals stay compatible.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExecutionOverrides {
    active_slots: Option<usize>,
    decode_rows: Option<usize>,
    pub(crate) prefill_wave: Option<usize>,
    mixed_step: bool,
}
impl ExecutionOverrides {
    pub(crate) fn capture() -> Self {
        Self {
            active_slots: env_usize("VASR_METAL_MAX_ACTIVE_SEQS"),
            decode_rows: env_usize("VASR_METAL_PAGED_DECODE_BATCH_SIZE"),
            prefill_wave: env_usize("VASR_PREFILL_MAX_WAVE"),
            mixed_step: !env_flag("VASR_DISABLE_MIXED_STEP_DECODE"),
        }
    }
    pub(crate) fn resolve_backend(
        self,
        is_metal: bool,
        capacity: usize,
        explicit_capacity: bool,
        auto_slots: usize,
        auto_prefill_wave: usize,
    ) -> ExecutionPolicy {
        let capacity = capacity.max(1);
        if !is_metal {
            return ExecutionPolicy {
                active_slots: capacity,
                decode_rows: capacity,
                // 非 Metal 默认允许 mixed-step，只有显式禁用才关闭。
                mixed_step: self.mixed_step,
                prefill_wave: capacity,
            };
        }

        let auto_capacity = if explicit_capacity {
            capacity
        } else {
            capacity.min(auto_slots.max(1))
        };
        // 环境变量仍然最高优先级，但不会突破本次解析出的安全容量。
        let active_slots = self
            .active_slots
            .unwrap_or(auto_capacity)
            .clamp(1, capacity);
        let decode_rows = self
            .decode_rows
            .unwrap_or(active_slots)
            .clamp(1, active_slots);
        let prefill_wave = self
            .prefill_wave
            .unwrap_or(auto_capacity.min(auto_prefill_wave.max(1)))
            .clamp(1, active_slots);
        ExecutionPolicy {
            active_slots,
            decode_rows,
            // Metal 现在默认开启 mixed-step；显式禁用优先。
            mixed_step: self.mixed_step,
            prefill_wave,
        }
    }
}

/// 读取一个正整数环境变量，非正数或解析失败时返回 `None`。
fn env_usize(name: &str) -> Option<usize> {
    parse_positive_usize(std::env::var(name).ok().as_deref())
}

fn parse_positive_usize(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

/// 读取一个布尔开关，只要设置了该环境变量就视为开启。
fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::{ExecutionPolicy, METAL_AUTO_MAX_ACTIVE_SLOTS, METAL_AUTO_MAX_PREFILL_WAVE};

    #[test]
    fn captured_execution_overrides_preserve_priority_and_independent_defaults() {
        use super::ExecutionOverrides;
        let first = ExecutionOverrides {
            active_slots: Some(128),
            decode_rows: Some(96),
            prefill_wave: Some(32),
            mixed_step: false,
        };
        let resolved = first.resolve_backend(true, 64, false, 32, 8);
        assert_eq!(
            resolved,
            ExecutionPolicy {
                active_slots: 64,
                decode_rows: 64,
                prefill_wave: 32,
                mixed_step: false
            }
        );
        let other = ExecutionOverrides {
            active_slots: None,
            decode_rows: None,
            prefill_wave: None,
            mixed_step: true,
        };
        assert_eq!(
            other.resolve_backend(true, 128, false, 64, 16),
            ExecutionPolicy {
                active_slots: 64,
                decode_rows: 64,
                prefill_wave: 16,
                mixed_step: true
            }
        );
        assert_eq!(
            first.resolve_backend(false, 4, false, 32, 8),
            ExecutionPolicy {
                active_slots: 4,
                decode_rows: 4,
                prefill_wave: 4,
                mixed_step: false
            }
        );
        assert_eq!(first.resolve_backend(true, 1, false, 32, 8).active_slots, 1);
    }

    #[test]
    fn legacy_execution_integer_parser_ignores_invalid_and_zero_without_trimming() {
        use super::parse_positive_usize;
        assert_eq!(parse_positive_usize(None), None);
        for value in ["0", "false", "invalid", "-1", " 64 ", ""] {
            assert_eq!(parse_positive_usize(Some(value)), None);
        }
        assert_eq!(parse_positive_usize(Some("64")), Some(64));
    }

    fn cpu() -> candle_core::Device {
        candle_core::Device::Cpu
    }

    #[test]
    fn non_metal_keeps_capacity_as_execution_width() {
        let policy = ExecutionPolicy::resolve(&cpu(), 12, false);
        assert_eq!(policy.active_slots, 12);
        assert_eq!(policy.decode_rows, 12);
        assert_eq!(policy.prefill_wave, 12);
        assert!(policy.mixed_step);
    }

    #[test]
    fn metal_auto_narrows_to_validated_slots_and_respects_capacity() {
        // 自动模式：活跃槽位收窄到验证过的上限。
        let auto = ExecutionPolicy::resolve_backend(true, 64, false);
        assert!(auto.active_slots <= METAL_AUTO_MAX_ACTIVE_SLOTS);
        assert!(auto.decode_rows <= auto.active_slots);
        assert!(auto.prefill_wave <= auto.active_slots);

        // 容量小于自动上限时，使用解析出的安全容量而不是硬编码的 8。
        let small = ExecutionPolicy::resolve_backend(true, 3, false);
        assert_eq!(small.active_slots, 3);
        assert_eq!(small.decode_rows, 3);
        assert_eq!(small.prefill_wave, 3);

        // 显式容量：按用户要求执行，便于 16 / 32 / 64 容量扫描。
        let explicit = ExecutionPolicy::resolve_backend(true, 16, true);
        assert_eq!(explicit.active_slots, 16);
    }

    #[test]
    fn metal_prefill_wave_is_independent_from_capacity() {
        let explicit = ExecutionPolicy::resolve_backend(true, 64, true);
        assert_eq!(explicit.active_slots, 64);
        // 不让容量为 64 时自动执行 64 行 prefill。
        assert!(explicit.prefill_wave <= METAL_AUTO_MAX_PREFILL_WAVE);
    }

    #[test]
    fn clamping_never_exceeds_capacity() {
        let policy = ExecutionPolicy {
            active_slots: 16,
            decode_rows: 16,
            mixed_step: true,
            prefill_wave: 16,
        };
        let clamped = policy.clamped_to(4);
        assert_eq!(clamped.active_slots, 4);
        assert_eq!(clamped.decode_rows, 4);
        assert_eq!(clamped.prefill_wave, 4);
    }
}
