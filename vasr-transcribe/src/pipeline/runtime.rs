//! Service lifetime, stage supervision and bounded admission.
use super::{PipelineRequestId, RequestRegistry};
use anyhow::Result;
#[cfg(test)]
use std::future::Future;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU8, Ordering},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tokio::task::{AbortHandle, JoinHandle};

const RUNNING: u8 = 0;
const STOPPING: u8 = 1;
const FAILED: u8 = 2;
const BYTE_UNIT: usize = 1024;

#[derive(Clone, Copy)]
pub(crate) struct PipelineLimits {
    pub(crate) active_slots: usize,
    pub(crate) safe_capacity: usize,
    pub(crate) source_bytes: usize,
}
impl Default for PipelineLimits {
    fn default() -> Self {
        Self {
            active_slots: 64,
            safe_capacity: 64,
            source_bytes: source_memory_budget(None),
        }
    }
}
/// One reservation is transferred from the unread HTTP body to queued sources.
/// Only the loader keeps it once the request has been dispatched.
#[derive(Debug)]
pub(crate) struct SourceReservation {
    _permit: OwnedSemaphorePermit,
}

pub(super) struct PipelineSupervisor {
    pub(super) registry: RequestRegistry,
    pub(super) failed: Arc<AtomicBool>,
    fatal: watch::Sender<Option<String>>,
    state: AtomicU8,
    handles: Mutex<Vec<(&'static str, AbortHandle)>>,
    pub(super) requests: Arc<Semaphore>,
    pub(super) items: Arc<Semaphore>,
    sources: Arc<Semaphore>,
    source_units: usize,
    request_capacity: usize,
    item_capacity: usize,
    debug: bool,
}
impl PipelineSupervisor {
    pub(super) fn new(
        registry: RequestRegistry,
        failed: Arc<AtomicBool>,
        fatal: watch::Sender<Option<String>>,
        limits: PipelineLimits,
    ) -> Arc<Self> {
        let source_units = limits
            .source_bytes
            .div_ceil(BYTE_UNIT)
            .max(1)
            .min(u32::MAX as usize);
        Arc::new(Self {
            registry,
            failed,
            fatal,
            state: AtomicU8::new(RUNNING),
            handles: Mutex::default(),
            requests: Arc::new(Semaphore::new(limits.active_slots.max(1))),
            items: Arc::new(Semaphore::new(
                limits.safe_capacity.saturating_mul(2).max(1),
            )),
            sources: Arc::new(Semaphore::new(source_units)),
            source_units,
            request_capacity: limits.active_slots.max(1),
            item_capacity: limits.safe_capacity.saturating_mul(2).max(1),
            debug: std::env::var_os("VASR_DEBUG_REQUEST_MEM").is_some(),
        })
    }
    pub(super) fn debug_enabled(&self) -> bool {
        self.debug
    }
    pub(super) fn log_snapshot(&self) {
        let requests = self
            .registry
            .pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len();
        let routes = self
            .registry
            .routes
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len();
        tracing::info!(target: "vasr_transcribe::pipeline",
            "metadata_lifetime | stage=registry requests={} routes={} request_budget_available={} request_budget_capacity={} item_budget_available={} item_budget_capacity={} source_budget_available_bytes={} source_budget_capacity_bytes={}",
            requests, routes, self.requests.available_permits(), self.request_capacity,
            self.items.available_permits(), self.item_capacity,
            self.sources.available_permits() * BYTE_UNIT, self.source_units * BYTE_UNIT);
    }
    pub(super) fn supervise<T: StageExit>(
        self: &Arc<Self>,
        stage: &'static str,
        handle: JoinHandle<T>,
    ) {
        {
            let mut handles = self.handles.lock().unwrap_or_else(|p| p.into_inner());
            handles.push((stage, handle.abort_handle()));
            if self.state.load(Ordering::Acquire) != RUNNING {
                handle.abort();
            }
        }
        let supervisor = Arc::downgrade(self);
        tokio::spawn(async move {
            let result = handle.await;
            if let Some(supervisor) = supervisor.upgrade() {
                let error = match result {
                    Ok(exit) => exit
                        .error()
                        .unwrap_or_else(|| "unexpected stage exit".into()),
                    Err(error) => format!("stage task failed: {error}"),
                };
                supervisor.fail(&format!("{stage} failed: {error}"));
            }
        });
    }
    pub(super) fn fail(&self, error: &str) {
        if self
            .state
            .compare_exchange(RUNNING, FAILED, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        self.failed.store(true, Ordering::Release);
        self.close_admission();
        self.registry.fail_all(error);
        self.fatal.send_replace(Some(error.to_owned()));
        self.abort_workers();
    }
    pub(super) fn stop(&self) {
        if self
            .state
            .compare_exchange(RUNNING, STOPPING, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Close the registration gate before draining: a submit worker can
            // keep running until its cooperative abort is next polled.
            self.failed.store(true, Ordering::Release);
            self.close_admission();
            self.registry
                .fail_all("global transcribe pipeline stopping");
            self.abort_workers();
        }
    }
    fn close_admission(&self) {
        self.requests.close();
        self.items.close();
        self.sources.close();
    }
    fn abort_workers(&self) {
        for (_, handle) in self
            .handles
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
        {
            handle.abort();
        }
    }
    pub(crate) async fn reserve_source(&self, bytes: usize) -> Result<Arc<SourceReservation>> {
        if bytes > self.source_units.saturating_mul(BYTE_UNIT) {
            anyhow::bail!(
                "request encoded source budget exceeded: {bytes} bytes exceeds {} bytes",
                self.source_units * BYTE_UNIT
            );
        }
        let units = bytes.div_ceil(BYTE_UNIT).max(1) as u32;
        let permit = self
            .sources
            .clone()
            .acquire_many_owned(units)
            .await
            .map_err(|_| anyhow::anyhow!("global transcribe pipeline stopped"))?;
        Ok(Arc::new(SourceReservation { _permit: permit }))
    }
}
impl Drop for PipelineSupervisor {
    fn drop(&mut self) {
        self.stop();
    }
}
pub(super) trait StageExit: Send + 'static {
    fn error(self) -> Option<String>;
}
impl StageExit for () {
    fn error(self) -> Option<String> {
        None
    }
}
impl StageExit for Result<()> {
    fn error(self) -> Option<String> {
        self.err().map(|error| format!("{error:#}"))
    }
}

#[derive(Default)]
pub(super) struct CancellationSignal {
    canceled: AtomicBool,
    notify: tokio::sync::Notify,
}
impl CancellationSignal {
    pub(super) fn is_canceled(&self) -> bool {
        self.canceled.load(Ordering::Acquire)
    }
    pub(super) async fn canceled(&self) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_canceled() {
                return;
            }
            notified.await;
        }
    }
    fn cancel(&self) {
        self.canceled.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }
}
pub(super) struct RequestCancellation {
    pub(super) registry: RequestRegistry,
    pub(super) id: PipelineRequestId,
    pub(super) signal: Arc<CancellationSignal>,
}
impl Drop for RequestCancellation {
    fn drop(&mut self) {
        self.signal.cancel();
        self.registry.cancel(self.id);
    }
}

pub(super) fn monitor_nested<T: StageExit>(
    supervisor: &Option<Weak<PipelineSupervisor>>,
    stage: &'static str,
    handle: JoinHandle<T>,
) -> Option<JoinHandle<T>> {
    if let Some(supervisor) = supervisor.as_ref().and_then(Weak::upgrade) {
        supervisor.supervise(stage, handle);
        None
    } else {
        Some(handle)
    }
}
/// Propagate a panic from a blocking task so its stage supervisor sees a fatal
/// worker failure. Ordinary per-item errors remain item outcomes.
pub(super) fn blocking_result<T>(
    result: std::result::Result<T, tokio::task::JoinError>,
) -> Result<T> {
    match result {
        Ok(value) => Ok(value),
        Err(error) if error.is_panic() => std::panic::resume_unwind(error.into_panic()),
        Err(error) => Err(anyhow::anyhow!("blocking worker join failed: {error}")),
    }
}

pub(crate) fn source_memory_budget(available_bytes: Option<u64>) -> usize {
    const MIB: u64 = 1024 * 1024;
    (available_bytes
        .map(|bytes| bytes / 16)
        .unwrap_or(2 * 1024 * MIB))
    .clamp(512 * MIB, 2 * 1024 * MIB) as usize
}
#[cfg(test)]
pub(super) async fn injected_worker(
    supervisor: &Arc<PipelineSupervisor>,
    stage: &'static str,
    future: impl Future<Output = Result<()>> + Send + 'static,
) {
    supervisor.supervise(stage, tokio::spawn(future));
}
