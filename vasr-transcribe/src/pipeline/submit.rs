use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use asr_data::AudioSource;

use super::{
    DeduplicatedTranscribeInputs, PendingGlobalRequest, PipelineRequestId, RequestItemRef,
    RequestStageProgress, TranscribeInput, TranscribeItemOutcome, deduplicate_transcribe_inputs,
    global_pipeline_failure_outcomes, RequestRegistry,
};

pub(super) struct GlobalTranscribeRequest {
    pub(super) log_id: String,
    pub(super) inputs: Vec<TranscribeInput>,
    pub(super) response_tx: oneshot::Sender<Vec<TranscribeItemOutcome>>,
}

#[derive(Debug)]
pub(super) struct GlobalLoadJob {
    pub(super) internal_index: usize,
    pub(super) request_item: RequestItemRef,
    pub(super) source: AudioSource,
    pub(super) mono: bool,
}

pub(super) fn spawn_global_submit_worker(
    mut submit_rx: mpsc::UnboundedReceiver<GlobalTranscribeRequest>,
    load_job_tx: mpsc::UnboundedSender<GlobalLoadJob>,
    pending_requests: Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: Arc<Mutex<HashMap<usize, RequestItemRef>>>,
    next_request_id: Arc<AtomicU64>,
    next_job_id: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(request) = submit_rx.recv().await {
            let GlobalTranscribeRequest {
                log_id,
                inputs,
                response_tx,
            } = request;
            let failure_indices = inputs.iter().map(|input| input.index).collect::<Vec<_>>();
            if failed.load(Ordering::Acquire) {
                let _ = response_tx.send(global_pipeline_failure_outcomes(
                    failure_indices,
                    "global transcribe pipeline stopped after ASR decoder failure",
                ));
                continue;
            }
            let request_id = PipelineRequestId(next_request_id.fetch_add(1, Ordering::Relaxed));
            let DeduplicatedTranscribeInputs {
                unique_inputs,
                duplicate_indices,
            } = deduplicate_transcribe_inputs(inputs);
            let unique_count = unique_inputs.len();
            if unique_count == 0 {
                let _ = response_tx.send(Vec::new());
                continue;
            }
            let expected = unique_inputs.iter().map(|input| (input.index, "mono")).collect();
            let mut load_jobs = Vec::with_capacity(unique_count);
            let mut internal_indices = Vec::with_capacity(unique_count);
            for input in unique_inputs {
                let internal_index = next_job_id.fetch_add(1, Ordering::Relaxed) as usize;
                let request_item = RequestItemRef {
                    request_id,
                    item_index: input.index,
                };
                internal_indices.push(internal_index);
                load_jobs.push(GlobalLoadJob {
                    internal_index,
                    request_item,
                    source: input.source,
                    mono: input.mono,
                });
            }
            if let Ok(mut pending) = pending_requests.lock() {
                if failed.load(Ordering::Acquire) {
                    drop(pending);
                    let _ = response_tx.send(global_pipeline_failure_outcomes(
                        failure_indices,
                        "global transcribe pipeline stopped after ASR decoder failure",
                    ));
                    continue;
                }
                pending.insert(
                    request_id,
                    PendingGlobalRequest {
                        log_id,
                        failure_indices,
                        remaining: unique_count,
                        expected,
                        duplicate_indices,
                        internal_indices,
                        outcomes: Vec::with_capacity(unique_count),
                        stages: RequestStageProgress {
                            load_remaining: unique_count,
                            vad_remaining: unique_count,
                            ..Default::default()
                        },
                        response_tx,
                    },
                );
            } else {
                let _ = response_tx.send(global_pipeline_failure_outcomes(
                    failure_indices,
                    "global transcribe pipeline pending request lock poisoned",
                ));
                continue;
            }
            if let Ok(mut routes) = job_routes.lock() {
                for job in &load_jobs {
                    routes.insert(job.internal_index, job.request_item);
                }
            } else {
                RequestRegistry::new(pending_requests.clone(), job_routes.clone()).fail_all("request route lock poisoned");
                return;
            }
            for job in load_jobs {
                if load_job_tx.send(job).is_err() {
                    RequestRegistry::new(pending_requests.clone(), job_routes.clone()).fail_request(request_id, "global loader channel closed");
                    return;
                }
            }
        }
    })
}
