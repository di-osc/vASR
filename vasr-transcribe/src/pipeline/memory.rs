#[cfg(target_os = "linux")]
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::{
    AsrPipelineMessage, BYTES_PER_MEMORY_PERMIT, DEFAULT_LOADED_QUEUE_MEMORY_FRACTION,
    FALLBACK_STAGE_PARALLELISM, LoadedJob, MIN_LOADED_QUEUE_MEMORY_PERMITS, TranscribeItemOutcome,
};

pub(super) fn default_stage_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(FALLBACK_STAGE_PARALLELISM)
}

pub(super) fn resolve_worker_count(configured: usize) -> usize {
    if configured == 0 {
        default_stage_parallelism()
    } else {
        configured
    }
}

pub(super) fn permits_to_gib(permits: usize) -> f64 {
    permits as f64 * BYTES_PER_MEMORY_PERMIT as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[derive(Clone)]
pub(super) struct LoadedQueueMemoryBudget {
    pub(super) semaphore: Arc<Semaphore>,
    pub(super) capacity_permits: usize,
    pub(super) source: &'static str,
}

pub(super) fn loaded_queue_memory_budget() -> LoadedQueueMemoryBudget {
    let LoadedQueueMemoryBudgetConfig { permits, source } = loaded_queue_memory_budget_permits(
        std::env::var("VASR_LOADED_QUEUE_MEMORY_MB").ok().as_deref(),
        std::env::var("VASR_LOADED_QUEUE_MEMORY_FRACTION")
            .ok()
            .as_deref(),
        total_memory_bytes(),
    );
    LoadedQueueMemoryBudget {
        semaphore: Arc::new(Semaphore::new(permits)),
        capacity_permits: permits,
        source,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LoadedQueueMemoryBudgetConfig {
    pub(super) permits: usize,
    pub(super) source: &'static str,
}

pub(super) fn loaded_queue_memory_budget_permits(
    memory_mb: Option<&str>,
    fraction: Option<&str>,
    total_memory_bytes: Option<u64>,
) -> LoadedQueueMemoryBudgetConfig {
    if let Some(value) = memory_mb {
        match value.parse::<usize>() {
            Ok(mb) if mb > 0 => {
                return LoadedQueueMemoryBudgetConfig {
                    permits: mb,
                    source: "env:VASR_LOADED_QUEUE_MEMORY_MB",
                };
            }
            _ => tracing::warn!(
                target: "vasr_transcribe::pipeline",
                "ignoring invalid VASR_LOADED_QUEUE_MEMORY_MB={value:?}; expected a positive integer"
            ),
        }
    }

    let fraction_from_env = fraction.is_some();
    let memory_fraction = match fraction {
        Some(value) => match value.parse::<f64>() {
            Ok(n) if n.is_finite() && n > 0.0 => n,
            _ => {
                tracing::warn!(
                    target: "vasr_transcribe::pipeline",
                    "ignoring invalid VASR_LOADED_QUEUE_MEMORY_FRACTION={value:?}; expected a positive number"
                );
                DEFAULT_LOADED_QUEUE_MEMORY_FRACTION
            }
        },
        None => DEFAULT_LOADED_QUEUE_MEMORY_FRACTION,
    };

    let permits = total_memory_bytes
        .and_then(|bytes| {
            let permits =
                ((bytes as f64 * memory_fraction) / BYTES_PER_MEMORY_PERMIT as f64).floor();
            if permits.is_finite() && permits >= 1.0 && permits <= usize::MAX as f64 {
                Some(permits as usize)
            } else {
                None
            }
        })
        .filter(|&permits: &usize| permits > 0)
        .unwrap_or_else(default_stage_parallelism)
        .max(MIN_LOADED_QUEUE_MEMORY_PERMITS);
    let source = if total_memory_bytes.is_some() {
        if fraction_from_env {
            "env:VASR_LOADED_QUEUE_MEMORY_FRACTION"
        } else {
            "system-memory-fraction"
        }
    } else {
        "fallback:stage_parallelism"
    };
    LoadedQueueMemoryBudgetConfig { permits, source }
}

pub(super) fn loaded_job_memory_permits(job: &LoadedJob) -> usize {
    let bytes = job
        .waveform
        .samples
        .len()
        .saturating_mul(std::mem::size_of::<f32>());
    bytes
        .saturating_add(BYTES_PER_MEMORY_PERMIT - 1)
        .saturating_div(BYTES_PER_MEMORY_PERMIT)
        .max(1)
}

pub(super) fn vad_to_asr_message_memory_permits(
    message: &Result<AsrPipelineMessage, TranscribeItemOutcome>,
) -> usize {
    let bytes = match message {
        Ok(AsrPipelineMessage::Job(job)) => job
            .waveform
            .samples
            .len()
            .saturating_mul(std::mem::size_of::<f32>()),
        #[cfg(feature = "paged-attn")]
        Ok(AsrPipelineMessage::Segment(segment)) => segment
            .waveform
            .samples
            .len()
            .saturating_mul(std::mem::size_of::<f32>()),
        Err(_) => 0,
    };
    if bytes == 0 {
        return 0;
    }
    bytes
        .saturating_add(BYTES_PER_MEMORY_PERMIT - 1)
        .saturating_div(BYTES_PER_MEMORY_PERMIT)
        .max(1)
}

pub(super) async fn acquire_loaded_job_memory(
    budget: &LoadedQueueMemoryBudget,
    job: &LoadedJob,
) -> Result<OwnedSemaphorePermit> {
    let permits = loaded_job_memory_permits(job).min(budget.capacity_permits);
    let permits = u32::try_from(permits)
        .map_err(|_| anyhow::anyhow!("loaded queue memory permit count overflows u32"))?;
    budget
        .semaphore
        .clone()
        .acquire_many_owned(permits)
        .await
        .map_err(|_| anyhow::anyhow!("loaded queue memory limiter closed"))
}

pub(super) async fn acquire_vad_to_asr_message_memory(
    budget: &LoadedQueueMemoryBudget,
    message: &Result<AsrPipelineMessage, TranscribeItemOutcome>,
) -> Result<Option<OwnedSemaphorePermit>> {
    let permits = vad_to_asr_message_memory_permits(message).min(budget.capacity_permits);
    if permits == 0 {
        return Ok(None);
    }
    let permits = u32::try_from(permits)
        .map_err(|_| anyhow::anyhow!("VAD to ASR queue memory permit count overflows u32"))?;
    let permit = budget
        .semaphore
        .clone()
        .acquire_many_owned(permits)
        .await
        .map_err(|_| anyhow::anyhow!("VAD to ASR queue memory limiter closed"))?;
    Ok(Some(permit))
}

fn total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        total_memory_bytes_from_proc_meminfo("/proc/meminfo").ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn total_memory_bytes_from_proc_meminfo(path: impl AsRef<Path>) -> Result<u64> {
    let text = std::fs::read_to_string(path)?;
    parse_meminfo_kib(&text, "MemTotal:")
        .and_then(|kib| kib.checked_mul(1024))
        .ok_or_else(|| anyhow::anyhow!("MemTotal not found in /proc/meminfo"))
}

#[cfg(target_os = "linux")]
pub(super) fn parse_meminfo_kib(text: &str, key: &str) -> Option<u64> {
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let value = parts.next()?.parse::<u64>().ok()?;
        return Some(value);
    }
    None
}
