use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use vasr_data::AudioSource;

use super::{
    DeduplicatedTranscribeInputs, PendingGlobalRequest, PipelineRequestId, RequestItemRef,
    RequestStageProgress, TranscribeInput, TranscribeItemOutcome, deduplicate_transcribe_inputs,
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
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(request) = submit_rx.recv().await {
            let GlobalTranscribeRequest {
                log_id,
                inputs,
                response_tx,
            } = request;
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
            if let Ok(mut routes) = job_routes.lock() {
                for job in &load_jobs {
                    routes.insert(job.internal_index, job.request_item);
                }
            }
            if let Ok(mut pending) = pending_requests.lock() {
                pending.insert(
                    request_id,
                    PendingGlobalRequest {
                        log_id,
                        remaining: unique_count,
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
            }
            for job in load_jobs {
                if load_job_tx.send(job).is_err() {
                    break;
                }
            }
        }
    })
}
