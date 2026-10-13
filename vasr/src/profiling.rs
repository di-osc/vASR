//! Optional stage probes. GPU synchronization is diagnostic-only and opt-in.
use candle_core::Device;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StageMeasurement {
    pub calls: u64,
    pub seconds: f64,
    /// Whether this probe explicitly waited for device completion. This remains
    /// host elapsed time, including submission/synchronization, not GPU timestamps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_sync: Option<bool>,
}
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct BatchMeasurement {
    pub batches: u64,
    pub rows: u64,
    pub max_rows: usize,
    pub valid_tokens: u64,
    pub executed_tokens: u64,
}
static BATCHES: OnceLock<Mutex<BTreeMap<&'static str, BatchMeasurement>>> = OnceLock::new();

pub(crate) fn record_batch(
    stage: &'static str,
    rows: usize,
    valid_tokens: usize,
    executed_tokens: usize,
) {
    if !flags().0 {
        return;
    }
    if let Ok(mut counters) = BATCHES.get_or_init(Mutex::default).lock() {
        let counter = counters.entry(stage).or_default();
        counter.batches += 1;
        counter.rows += rows as u64;
        counter.max_rows = counter.max_rows.max(rows);
        counter.valid_tokens += valid_tokens as u64;
        counter.executed_tokens += executed_tokens as u64;
    }
}

static COUNTERS: OnceLock<Mutex<BTreeMap<&'static str, StageMeasurement>>> = OnceLock::new();
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyncMode {
    None,
    All,
    Forward,
}

impl SyncMode {
    fn parse(value: Option<&str>) -> Self {
        match value {
            None => Self::None,
            Some(value) if value.trim().eq_ignore_ascii_case("forward") => Self::Forward,
            Some(_) => Self::All,
        }
    }

    fn should_sync(self, stage: &str) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Forward => matches!(
                stage,
                "encoder_forward" | "prefill_forward" | "decode_forward"
            ),
        }
    }
}

static FLAGS: OnceLock<(bool, SyncMode)> = OnceLock::new();
fn flags() -> (bool, SyncMode) {
    *FLAGS.get_or_init(|| {
        let sync = std::env::var_os("VASR_METAL_STAGE_SYNC")
            .map(|value| value.to_string_lossy().into_owned());
        (
            std::env::var_os("VASR_METAL_STAGE_PROFILE").is_some(),
            SyncMode::parse(sync.as_deref()),
        )
    })
}

/// Synchronized probes include GPU completion; ordinary probes measure host elapsed.
/// Nested stages overlap and must not be added to their enclosing stages.
pub(crate) fn measure<T, E: From<candle_core::Error>>(
    stage: &'static str,
    device: &Device,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    measure_with_autorelease(stage, device, false, operation)
}

#[cfg(feature = "metal-paged-attn")]
pub(crate) fn autorelease_enabled(automatic: bool) -> bool {
    static AUTORELEASE: OnceLock<Option<bool>> = OnceLock::new();
    AUTORELEASE
        .get_or_init(|| {
            std::env::var("VASR_METAL_AUTORELEASEPOOL")
                .ok()
                .map(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
        })
        .unwrap_or(automatic)
}

pub(crate) fn measure_with_autorelease<T, E: From<candle_core::Error>>(
    stage: &'static str,
    device: &Device,
    automatic: bool,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    #[cfg(feature = "metal-paged-attn")]
    {
        let enabled = autorelease_enabled(automatic);
        // Scope only whole forwards; nested stage probes don't add another pool.
        let enabled = enabled
            && matches!(
                stage,
                "encoder_forward" | "prefill_forward" | "decode_forward"
            );
        return vasr_paged_attn::metal::profile::with_autorelease_scope(enabled, device, || {
            measure_inner(stage, device, operation)
        });
    }
    #[cfg(not(feature = "metal-paged-attn"))]
    {
        let _ = automatic;
        measure_inner(stage, device, operation)
    }
}

fn measure_inner<T, E: From<candle_core::Error>>(
    stage: &'static str,
    device: &Device,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let (enabled, sync_mode) = flags();
    if !enabled {
        return operation();
    }
    let sync = sync_mode.should_sync(stage);
    if sync {
        device.synchronize().map_err(E::from)?;
    }
    let start = Instant::now();
    let output = operation()?;
    if sync {
        device.synchronize().map_err(E::from)?;
    }
    if let Ok(mut counters) = COUNTERS.get_or_init(Mutex::default).lock() {
        let counter = counters.entry(stage).or_default();
        counter.calls += 1;
        counter.seconds += start.elapsed().as_secs_f64();
        counter.completion_sync = Some(sync);
    }
    Ok(output)
}

pub fn snapshot() -> BTreeMap<&'static str, StageMeasurement> {
    COUNTERS
        .get()
        .and_then(|c| c.lock().ok().map(|c| c.clone()))
        .unwrap_or_default()
}

pub fn log_snapshot() {
    if flags().0 {
        #[cfg(feature = "metal-paged-attn")]
        {
            let (created, cached) = vasr_paged_attn::metal::pipeline_cache_stats();
            tracing::info!(target: "vasr::stage_profile", created, cached, "paged Metal pipeline cache");
        }
        if let Some(batches) = BATCHES.get().and_then(|c| c.lock().ok().map(|c| c.clone())) {
            if let Ok(json) = serde_json::to_string(&batches) {
                tracing::info!(target: "vasr::stage_profile", batches=json.as_str(), "executed batch shapes cumulative");
            }
        }
        if let Ok(json) = serde_json::to_string(&snapshot()) {
            tracing::info!(target: "vasr::stage_profile", synchronized=flags().1 != SyncMode::None, sync_mode=?flags().1, measurements=json.as_str(), "stage profile cumulative; nested stages overlap");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SyncMode;

    #[test]
    fn forward_sync_waits_for_roots_without_synchronizing_nested_probes() {
        let mode = SyncMode::parse(Some("forward"));
        for stage in ["encoder_forward", "prefill_forward", "decode_forward"] {
            assert!(mode.should_sync(stage), "{stage}");
        }
        for stage in [
            "text_decode",
            "prefill_attention",
            "encoder_ffn",
            "decode_metadata",
            "lm_head",
            "feature_extract",
        ] {
            assert!(!mode.should_sync(stage), "nested probe {stage}");
        }
    }

    #[test]
    fn completion_sync_is_optional_and_distinguishes_nested_stage_measurements() {
        let mut measurement = super::StageMeasurement::default();
        let serialized = serde_json::to_value(&measurement).unwrap();
        assert!(serialized.get("completion_sync").is_none());
        measurement.completion_sync = Some(false);
        assert_eq!(
            serde_json::to_value(&measurement).unwrap()["completion_sync"],
            false
        );
        measurement.completion_sync = Some(true);
        assert_eq!(
            serde_json::to_value(&measurement).unwrap()["completion_sync"],
            true
        );
    }

    #[test]
    fn legacy_and_disabled_probe_sync_keep_their_existing_semantics() {
        for stage in ["encoder_forward", "prefill_attention", "lm_head"] {
            assert!(!SyncMode::parse(None).should_sync(stage));
            assert!(SyncMode::parse(Some("1")).should_sync(stage));
        }
    }
}
