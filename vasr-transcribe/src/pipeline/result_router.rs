#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::sync::{Arc, Mutex};

#[cfg(test)]
use super::{PendingGlobalRequest, PipelineRequestId};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::{RequestItemRef, RoutedTranscribeItemOutcome, TranscribeItemOutcome};

pub(super) fn route_outcome(
    request_item: Option<RequestItemRef>,
    outcome: TranscribeItemOutcome,
) -> RoutedTranscribeItemOutcome {
    match request_item {
        Some(request_item) => RoutedTranscribeItemOutcome::for_request(request_item, outcome),
        None => outcome.into(),
    }
}

#[cfg(test)]
pub(super) fn spawn_global_result_router(
    asr_to_result_rx: mpsc::Receiver<RoutedTranscribeItemOutcome>,
    pending_requests: Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: Arc<Mutex<HashMap<usize, RequestItemRef>>>,
) -> JoinHandle<anyhow::Result<()>> {
    spawn_result_router_with_registry(
        asr_to_result_rx,
        super::RequestRegistry::new(pending_requests, job_routes),
    )
}

pub(super) fn spawn_result_router_with_registry(
    mut asr_to_result_rx: mpsc::Receiver<RoutedTranscribeItemOutcome>,
    registry: super::RequestRegistry,
) -> JoinHandle<anyhow::Result<()>> {
    tokio::spawn(async move {
        let debug_metadata = std::env::var_os("VASR_DEBUG_REQUEST_MEM").is_some();
        while let Some(routed) = asr_to_result_rx.recv().await {
            if let Err(error) = registry.finish(routed) {
                registry.fail_all(&format!("result routing failed: {error}"));
                return Err(error);
            }
            if debug_metadata {
                let requests = registry.pending.lock().ok().map(|pending| pending.len());
                let routes = registry.routes.lock().ok().map(|routes| routes.len());
                if let (Some(requests), Some(routes)) = (requests, routes) {
                    tracing::info!(target: "vasr_transcribe::pipeline",
                        "metadata_lifetime | stage=router requests={} routes={}", requests, routes);
                }
            }
        }
        registry.fail_all("global result channel closed");
        Err(anyhow::anyhow!("global result channel closed"))
    })
}
