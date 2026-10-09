//! Request completion and route ownership. No guard is held across an await.
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use super::{PendingGlobalRequest, PipelineRequestId, RequestItemRef, RoutedTranscribeItemOutcome, expand_duplicate_outcomes, global_pipeline_failure_outcomes};

pub(super) type PendingRequests = Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>;
pub(super) type JobRoutes = Arc<Mutex<HashMap<usize, RequestItemRef>>>;

#[derive(Clone)]
pub(super) struct RequestRegistry {
    pub(super) pending: PendingRequests,
    pub(super) routes: JobRoutes,
}
impl RequestRegistry {
    pub(super) fn new(pending: PendingRequests, routes: JobRoutes) -> Self { Self { pending, routes } }
    pub(super) fn finish(&self, routed: RoutedTranscribeItemOutcome) -> anyhow::Result<()> {
        let item = match routed.request_item {
            Some(item) => item,
            None => *self.routes.lock().map_err(|_| anyhow::anyhow!("request route lock poisoned"))?
                .get(&routed.outcome.index).ok_or_else(|| anyhow::anyhow!("outcome has no request route: {}", routed.outcome.index))?,
        };
        let mut outcome = routed.outcome;
        outcome.index = item.item_index;
        let completed = {
            let mut pending = self.pending.lock().map_err(|_| anyhow::anyhow!("pending request lock poisoned"))?;
            let Some(request) = pending.get_mut(&item.request_id) else { return Ok(()); };
            let key = (outcome.index, outcome.channel);
            if !request.expected.contains(&key) { anyhow::bail!("unexpected request item/channel: {key:?}"); }
            if request.outcomes.iter().any(|old| (old.index, old.channel) == key) { return Ok(()); }
            request.outcomes.push(outcome);
            request.remaining -= 1;
            if request.remaining == 0 { pending.remove(&item.request_id) } else { None }
        };
        if let Some(request) = completed {
            self.remove_routes(&request.internal_indices);
            let _ = request.response_tx.send(expand_duplicate_outcomes(request.outcomes, &request.duplicate_indices));
        }
        Ok(())
    }
    pub(super) fn fail_all(&self, error: &str) {
        // A poisoned bookkeeping lock is itself fatal. Recover only to drain
        // ownership and notify callers; never resume inference with this state.
        let requests = {
            let mut pending = self.pending.lock().unwrap_or_else(|poison| poison.into_inner());
            std::mem::take(&mut *pending).into_values().collect::<Vec<_>>()
        };
        self.routes.lock().unwrap_or_else(|poison| poison.into_inner()).clear();
        for request in requests { request.fail(error); }
    }
    pub(super) fn fail_request(&self, id: PipelineRequestId, error: &str) {
        let request = self.pending.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
        if let Some(request) = request { self.remove_routes(&request.internal_indices); request.fail(error); }
    }
    fn remove_routes(&self, indices: &[usize]) {
        let mut routes = self.routes.lock().unwrap_or_else(|p| p.into_inner());
        for index in indices { routes.remove(index); }
    }
}
impl PendingGlobalRequest {
    pub(super) fn fail(self, error: &str) {
        let completed: HashSet<_> = self.outcomes.iter().map(|item| (item.index, item.channel)).collect();
        let mut outcomes = self.outcomes;
        let mut missing: Vec<_> = self.expected.difference(&completed).copied().collect();
        missing.sort_unstable();
        for (index, channel) in missing {
            let mut failure = global_pipeline_failure_outcomes([index], error).pop().unwrap();
            failure.channel = channel;
            outcomes.push(failure);
        }
        let _ = self.response_tx.send(expand_duplicate_outcomes(outcomes, &self.duplicate_indices));
    }
}
