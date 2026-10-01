//! Optional stage probes. GPU synchronization is diagnostic-only and opt-in.
use candle_core::Device;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StageMeasurement {
    pub calls: u64,
    pub seconds: f64,
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
static FLAGS: OnceLock<(bool, bool)> = OnceLock::new();
fn flags() -> (bool, bool) {
    *FLAGS.get_or_init(|| {
        (
            std::env::var_os("VASR_METAL_STAGE_PROFILE").is_some(),
            std::env::var_os("VASR_METAL_STAGE_SYNC").is_some(),
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
    let (enabled, sync) = flags();
    if !enabled {
        return operation();
    }
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
            tracing::info!(target: "vasr::stage_profile", synchronized=flags().1, measurements=json.as_str(), "stage profile cumulative; nested stages overlap");
        }
    }
}
