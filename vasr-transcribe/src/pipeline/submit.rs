use super::{
    CancellationSignal, DeduplicatedTranscribeInputs, PendingGlobalRequest, PipelineRequestId,
    PipelineSupervisor, RequestItemRef, RequestRegistry, RequestStageProgress, SourceReservation,
    TranscribeInput, TranscribeItemOutcome, deduplicate_transcribe_inputs,
    global_pipeline_failure_outcomes,
};
use anyhow::Result;
use asr_data::AudioSource;
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};

pub(super) struct GlobalTranscribeRequest {
    pub(super) log_id: String,
    pub(super) inputs: Vec<TranscribeInput>,
    pub(super) response_tx: oneshot::Sender<Vec<TranscribeItemOutcome>>,
    pub(super) request_id: Option<PipelineRequestId>,
    pub(super) source_memory: Option<Arc<SourceReservation>>,
    pub(super) cancellation: Option<Arc<CancellationSignal>>,
}
#[derive(Debug)]
pub(super) struct GlobalLoadJob {
    pub(super) internal_index: usize,
    pub(super) request_item: RequestItemRef,
    pub(super) source: AudioSource,
    pub(super) mono: bool,
    pub(super) _item_permit: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
    pub(super) _source_memory: Option<Arc<SourceReservation>>,
}

#[cfg(test)]
pub(super) fn spawn_global_submit_worker(
    submit_rx: mpsc::UnboundedReceiver<GlobalTranscribeRequest>,
    load_job_tx: mpsc::UnboundedSender<GlobalLoadJob>,
    pending_requests: Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: Arc<Mutex<HashMap<usize, RequestItemRef>>>,
    next_request_id: Arc<AtomicU64>,
    next_job_id: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
) -> JoinHandle<Result<()>> {
    spawn_submit_worker(
        submit_rx,
        load_job_tx,
        RequestRegistry::new(pending_requests, job_routes),
        next_request_id,
        next_job_id,
        failed,
        None,
    )
}
pub(super) fn spawn_supervised_submit_worker(
    submit_rx: mpsc::UnboundedReceiver<GlobalTranscribeRequest>,
    load_job_tx: mpsc::UnboundedSender<GlobalLoadJob>,
    pending_requests: Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: Arc<Mutex<HashMap<usize, RequestItemRef>>>,
    next_request_id: Arc<AtomicU64>,
    next_job_id: Arc<AtomicU64>,
    supervisor: Weak<PipelineSupervisor>,
) -> JoinHandle<Result<()>> {
    let owner = supervisor.upgrade().expect("supervisor initialized");
    let _ = (pending_requests, job_routes);
    spawn_submit_worker(
        submit_rx,
        load_job_tx,
        owner.registry.clone(),
        next_request_id,
        next_job_id,
        owner.failed.clone(),
        Some(supervisor),
    )
}
fn spawn_submit_worker(
    mut submit_rx: mpsc::UnboundedReceiver<GlobalTranscribeRequest>,
    load_job_tx: mpsc::UnboundedSender<GlobalLoadJob>,
    registry: RequestRegistry,
    next_request_id: Arc<AtomicU64>,
    next_job_id: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
    supervisor: Option<Weak<PipelineSupervisor>>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let mut dispatches = JoinSet::new();
        loop {
            tokio::select! {
                request = submit_rx.recv() => {
                    let Some(request) = request else { break; };
                    let GlobalTranscribeRequest { log_id, inputs, response_tx, request_id, source_memory, cancellation } = request;
                    if response_tx.is_closed() { continue; }
                    let failure_indices = inputs.iter().map(|input| input.index).collect::<Vec<_>>();
                    if failed.load(Ordering::Acquire) {
                        let _ = response_tx.send(global_pipeline_failure_outcomes(failure_indices, "global transcribe pipeline stopped"));
                        continue;
                    }
                    let request_id = request_id.unwrap_or_else(|| PipelineRequestId(next_request_id.fetch_add(1, Ordering::Relaxed)));
                    let DeduplicatedTranscribeInputs { unique_inputs, duplicate_indices } = deduplicate_transcribe_inputs(inputs);
                    let unique_count = unique_inputs.len();
                    if unique_count == 0 { let _ = response_tx.send(Vec::new()); continue; }
                    let expected = unique_inputs.iter().map(|input| (input.index, "mono")).collect();
                    let jobs = unique_inputs.into_iter().map(|input| GlobalLoadJob {
                        internal_index: 0,
                        request_item: RequestItemRef { request_id, item_index: input.index },
                        source: input.source, mono: input.mono, _item_permit: None, _source_memory: source_memory.clone(),
                    }).collect::<Vec<_>>();
                    {
                        let mut pending = registry.pending.lock().map_err(|_| anyhow::anyhow!("pending request lock poisoned"))?;
                        if response_tx.is_closed() { continue; }
                        if failed.load(Ordering::Acquire) {
                            drop(pending);
                            let _ = response_tx.send(global_pipeline_failure_outcomes(failure_indices, "global transcribe pipeline stopped"));
                            continue;
                        }
                        pending.insert(request_id, PendingGlobalRequest { log_id,
                            remaining: unique_count, expected, duplicate_indices,
                            internal_indices: Vec::new(), item_routes: HashMap::new(), outcomes: Vec::with_capacity(unique_count),
                            stages: RequestStageProgress { load_remaining: unique_count, vad_remaining: unique_count, ..Default::default() }, response_tx });
                    }
                    let registry = registry.clone();
                    let load_job_tx = load_job_tx.clone();
                    let supervisor = supervisor.clone();
                    let next_job_id = next_job_id.clone();
                    dispatches.spawn(async move {
                        for mut job in jobs {
                            if !registry.is_live(request_id) { break; }
                            if let Some(owner) = supervisor.as_ref().and_then(Weak::upgrade) {
                                // Acquire one canonical item at a time in each request.
                                // Concurrent dispatchers let short requests join the same FIFO limiter.
                                let signal = cancellation.as_deref();
                                let permit = tokio::select! {
                                    permit = owner.items.clone().acquire_owned() => permit.map_err(|_| anyhow::anyhow!("item admission closed"))?,
                                    _ = async { match signal { Some(signal) => signal.canceled().await, None => std::future::pending().await } } => break,
                                };
                                if !registry.is_live(request_id) { break; }
                                let permit = Arc::new(permit);
                                job._item_permit = Some(permit);
                            }
                            job.internal_index = next_job_id.fetch_add(1, Ordering::Relaxed) as usize;
                            if !registry.register_work(job.internal_index, job.request_item, job._item_permit.clone())? { break; }
                            if load_job_tx.send(job).is_err() {
                                registry.fail_request(request_id, "global loader channel closed");
                                return Err(anyhow::anyhow!("global loader channel closed"));
                            }
                            tokio::task::yield_now().await;
                        }
                        Ok::<_, anyhow::Error>(())
                    });
                }
                result = dispatches.join_next(), if !dispatches.is_empty() => {
                    result.expect("nonempty dispatch set").map_err(|error| anyhow::anyhow!("submit dispatch join failed: {error}"))??;
                }
            }
        }
        while let Some(result) = dispatches.join_next().await {
            result.map_err(|error| anyhow::anyhow!("submit dispatch join failed: {error}"))??;
        }
        Ok(())
    })
}
