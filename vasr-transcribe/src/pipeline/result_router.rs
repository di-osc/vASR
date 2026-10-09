use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::{
    PendingGlobalRequest, PipelineRequestId, RequestItemRef, RoutedTranscribeItemOutcome,
    TranscribeItemOutcome, expand_duplicate_outcomes,
};

pub(super) fn route_outcome(
    request_item: Option<RequestItemRef>,
    outcome: TranscribeItemOutcome,
) -> RoutedTranscribeItemOutcome {
    match request_item {
        Some(request_item) => RoutedTranscribeItemOutcome::for_request(request_item, outcome),
        None => outcome.into(),
    }
}

pub(super) fn spawn_global_result_router(
    mut asr_to_result_rx: mpsc::Receiver<RoutedTranscribeItemOutcome>,
    pending_requests: Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: Arc<Mutex<HashMap<usize, RequestItemRef>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let debug_metadata = std::env::var_os("VASR_DEBUG_REQUEST_MEM").is_some();
        while let Some(routed) = asr_to_result_rx.recv().await {
            let request_item = match routed.request_item {
                Some(request_item) => request_item,
                None => {
                    let Ok(routes) = job_routes.lock() else {
                        continue;
                    };
                    let Some(request_item) = routes.get(&routed.outcome.index).copied() else {
                        continue;
                    };
                    request_item
                }
            };
            let routed = route_outcome(Some(request_item), routed.outcome);
            let Some(request_item) = routed.request_item else {
                continue;
            };
            let mut completed = None;
            if let Ok(mut pending) = pending_requests.lock()
                && let Some(request) = pending.get_mut(&request_item.request_id)
            {
                request.outcomes.push(routed.outcome);
                request.remaining = request.remaining.saturating_sub(1);
                if request.remaining == 0 {
                    completed = pending.remove(&request_item.request_id);
                }
            }
            if let Some(request) = completed {
                if let Ok(mut routes) = job_routes.lock() {
                    for internal_index in &request.internal_indices {
                        routes.remove(internal_index);
                    }
                }
                let outcomes =
                    expand_duplicate_outcomes(request.outcomes, &request.duplicate_indices);
                let _ = request.response_tx.send(outcomes);
            }
            if debug_metadata {
                // Read separately: never hold one management lock while waiting
                // for another, and don't keep guards across the logging call.
                let requests = pending_requests.lock().ok().map(|pending| pending.len());
                let routes = job_routes.lock().ok().map(|routes| routes.len());
                if let (Some(requests), Some(routes)) = (requests, routes) {
                    tracing::info!(
                        target: "vasr_transcribe::pipeline",
                        "metadata_lifetime | stage=router requests={} routes={}", requests, routes,
                    );
                }
            }
        }
    })
}
