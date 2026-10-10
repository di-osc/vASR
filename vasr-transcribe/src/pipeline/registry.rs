//! Request completion and route ownership. No guard is held across an await.
use super::{
    PendingGlobalRequest, PipelineRequestId, RequestItemRef, RoutedTranscribeItemOutcome,
    expand_duplicate_outcomes, global_pipeline_failure_outcomes,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

pub(super) type PendingRequests = Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>;
pub(super) type JobRoutes = Arc<Mutex<HashMap<usize, RequestItemRef>>>;

#[derive(Clone)]
pub(super) struct RequestRegistry {
    pub(super) pending: PendingRequests,
    pub(super) routes: JobRoutes,
    retired: Arc<Mutex<BTreeMap<usize, usize>>>,
    retired_requests: Arc<Mutex<BTreeMap<u64, u64>>>,
    item_permits:
        Arc<Mutex<HashMap<(PipelineRequestId, usize), Arc<tokio::sync::OwnedSemaphorePermit>>>>,
}
impl RequestRegistry {
    pub(super) fn new(pending: PendingRequests, routes: JobRoutes) -> Self {
        Self {
            pending,
            routes,
            retired: Arc::default(),
            retired_requests: Arc::default(),
            item_permits: Arc::default(),
        }
    }
    pub(super) fn finish(&self, routed: RoutedTranscribeItemOutcome) -> anyhow::Result<()> {
        let item = match routed.request_item {
            Some(item) => item,
            None => {
                let item = self
                    .routes
                    .lock()
                    .map_err(|_| anyhow::anyhow!("request route lock poisoned"))?
                    .get(&routed.outcome.index)
                    .copied();
                match item {
                    Some(item) => item,
                    None if self.is_retired(routed.outcome.index) => return Ok(()),
                    None => anyhow::bail!("outcome has no request route: {}", routed.outcome.index),
                }
            }
        };
        let mut outcome = routed.outcome;
        outcome.index = item.item_index;
        let completed = {
            let mut pending = self
                .pending
                .lock()
                .map_err(|_| anyhow::anyhow!("pending request lock poisoned"))?;
            let Some(request) = pending.get_mut(&item.request_id) else {
                if self.is_retired_request(item.request_id) {
                    return Ok(());
                }
                anyhow::bail!("outcome has unknown request id: {:?}", item.request_id);
            };
            let key = (outcome.index, outcome.channel);
            if !request.expected.contains(&key) {
                anyhow::bail!("unexpected request item/channel: {key:?}");
            }
            if request
                .outcomes
                .iter()
                .any(|old| (old.index, old.channel) == key)
            {
                return Ok(());
            }
            request.outcomes.push(outcome);
            request.remaining -= 1;
            if let Some(internal_index) = request.item_routes.remove(&key) {
                self.remove_routes(&[internal_index]);
                request
                    .internal_indices
                    .retain(|index| *index != internal_index);
            }
            if request
                .expected
                .iter()
                .filter(|(index, _)| *index == item.item_index)
                .all(|key| {
                    request
                        .outcomes
                        .iter()
                        .any(|old| (old.index, old.channel) == *key)
                })
            {
                self.item_permits
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .remove(&(item.request_id, item.item_index));
            }
            if request.remaining == 0 {
                self.retire_request(item.request_id);
                pending.remove(&item.request_id)
            } else {
                None
            }
        };
        if let Some(request) = completed {
            self.remove_routes(&request.internal_indices);
            let _ = request.response_tx.send(expand_duplicate_outcomes(
                request.outcomes,
                &request.duplicate_indices,
            ));
        }
        Ok(())
    }
    pub(super) fn fail_all(&self, error: &str) {
        // A poisoned bookkeeping lock is itself fatal. Recover only to drain
        // ownership and notify callers; never resume inference with this state.
        let requests = {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for &id in pending.keys() {
                self.retire_request(id);
            }
            std::mem::take(&mut *pending)
                .into_values()
                .collect::<Vec<_>>()
        };
        let indices = self
            .routes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .keys()
            .copied()
            .collect::<Vec<_>>();
        self.remove_routes(&indices);
        self.item_permits
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
        for request in requests {
            request.fail(error);
        }
    }
    pub(super) fn fail_request(&self, id: PipelineRequestId, error: &str) {
        let request = {
            let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            self.retire_request(id);
            pending.remove(&id)
        };
        if let Some(request) = request {
            self.remove_routes(&request.internal_indices);
            request.fail(error);
        }
        self.item_permits
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|(request_id, _), _| *request_id != id);
    }
    pub(super) fn cancel(&self, id: PipelineRequestId) {
        let request = {
            let mut pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
            self.retire_request(id);
            pending.remove(&id)
        };
        if let Some(request) = request {
            self.remove_routes(&request.internal_indices);
        }
        self.item_permits
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|(request_id, _), _| *request_id != id);
    }
    pub(super) fn is_live(&self, id: PipelineRequestId) -> bool {
        self.pending
            .lock()
            .map(|pending| pending.contains_key(&id))
            .unwrap_or(false)
    }
    pub(super) fn register_work(
        &self,
        internal_index: usize,
        item: RequestItemRef,
        permit: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
    ) -> anyhow::Result<bool> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| anyhow::anyhow!("pending request lock poisoned"))?;
        let Some(request) = pending.get_mut(&item.request_id) else {
            // Cancellation can win after ID allocation. Retire this issued ID
            // too, so repeated cancellations do not leave permanent gaps.
            self.remove_routes(&[internal_index]);
            return Ok(false);
        };
        request.internal_indices.push(internal_index);
        request
            .item_routes
            .insert((item.item_index, "mono"), internal_index);
        self.routes
            .lock()
            .map_err(|_| anyhow::anyhow!("request route lock poisoned"))?
            .insert(internal_index, item);
        if let Some(permit) = permit {
            self.item_permits
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert((item.request_id, item.item_index), permit);
        }
        Ok(true)
    }
    #[cfg(test)]
    pub(super) fn retain_item_permit(
        &self,
        item: RequestItemRef,
        permit: Arc<tokio::sync::OwnedSemaphorePermit>,
    ) {
        let pending = self.pending.lock().unwrap_or_else(|p| p.into_inner());
        if pending.contains_key(&item.request_id) {
            self.item_permits
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert((item.request_id, item.item_index), permit);
        }
    }
    fn is_retired_request(&self, id: PipelineRequestId) -> bool {
        self.retired_requests
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .range(..=id.0)
            .next_back()
            .is_some_and(|(_, end)| id.0 <= *end)
    }
    fn retire_request(&self, id: PipelineRequestId) {
        let mut retired = self
            .retired_requests
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut start = id.0;
        let mut end = id.0;
        if let Some((&left, &right)) = retired.range(..=id.0).next_back() {
            if right.saturating_add(1) >= id.0 {
                start = left;
                end = end.max(right);
                retired.remove(&left);
            }
        }
        if let Some((&left, &right)) = retired.range(id.0..).next() {
            if left <= end.saturating_add(1) {
                end = end.max(right);
                retired.remove(&left);
            }
        }
        retired.insert(start, end);
    }
    #[cfg(test)]
    pub(super) fn retired_range_count(&self) -> usize {
        self.retired.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
    fn is_retired(&self, index: usize) -> bool {
        self.retired
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .range(..=index)
            .next_back()
            .is_some_and(|(_, end)| index <= *end)
    }
    fn remove_routes(&self, indices: &[usize]) {
        let mut routes = self.routes.lock().unwrap_or_else(|p| p.into_inner());
        let mut retired = self.retired.lock().unwrap_or_else(|p| p.into_inner());
        for &index in indices {
            routes.remove(&index);
            let mut start = index;
            let mut end = index;
            if let Some((&left, &right)) = retired.range(..=index).next_back() {
                if right.saturating_add(1) >= index {
                    start = left;
                    end = end.max(right);
                    retired.remove(&left);
                }
            }
            if let Some((&left, &right)) = retired.range(index..).next() {
                if left <= end.saturating_add(1) {
                    end = end.max(right);
                    retired.remove(&left);
                }
            }
            retired.insert(start, end);
        }
    }
}
impl PendingGlobalRequest {
    pub(super) fn fail(self, error: &str) {
        let completed: HashSet<_> = self
            .outcomes
            .iter()
            .map(|item| (item.index, item.channel))
            .collect();
        let mut outcomes = self.outcomes;
        let mut missing: Vec<_> = self.expected.difference(&completed).copied().collect();
        missing.sort_unstable();
        for (index, channel) in missing {
            let mut failure = global_pipeline_failure_outcomes([index], error)
                .pop()
                .unwrap();
            failure.channel = channel;
            outcomes.push(failure);
        }
        let _ = self
            .response_tx
            .send(expand_duplicate_outcomes(outcomes, &self.duplicate_indices));
    }
}

pub(super) fn request_is_live(
    pending: &Option<PendingRequests>,
    item: Option<RequestItemRef>,
) -> bool {
    match (pending, item) {
        (Some(pending), Some(item)) => pending
            .lock()
            .map(|requests| requests.contains_key(&item.request_id))
            .unwrap_or(false),
        _ => true,
    }
}
