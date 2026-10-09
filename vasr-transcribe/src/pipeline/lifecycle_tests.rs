use super::*;

fn failed_item(index: usize) -> TranscribeItemOutcome {
    global_pipeline_failure_outcomes([index], "item failure").pop().unwrap()
}
fn pending(indices: Vec<usize>) -> (PendingGlobalRequest, oneshot::Receiver<Vec<TranscribeItemOutcome>>) {
    let (response_tx, rx) = oneshot::channel();
    let request = PendingGlobalRequest {
        log_id: "lifecycle-test".into(),
        remaining: indices.len(),
        expected: indices.iter().map(|index| (*index, "mono")).collect(),
        failure_indices: indices.clone(),
        internal_indices: indices,
        duplicate_indices: Vec::new(),
        outcomes: Vec::new(),
        stages: RequestStageProgress::default(),
        response_tx,
    };
    (request, rx)
}
#[tokio::test]
async fn submit_closed_load_channel_finishes_every_original_index() {
    let (tx, rx) = mpsc::unbounded_channel();
    let (load_tx, load_rx) = mpsc::unbounded_channel();
    drop(load_rx);
    let requests = Arc::new(Mutex::new(HashMap::new()));
    let routes = Arc::new(Mutex::new(HashMap::new()));
    let handle = spawn_global_submit_worker(rx, load_tx, requests.clone(), routes.clone(),
        Arc::new(AtomicU64::new(1)), Arc::new(AtomicU64::new(1)), Arc::new(AtomicBool::new(false)));
    let (response_tx, response_rx) = oneshot::channel();
    tx.send(GlobalTranscribeRequest {
        log_id: "closed-loader".into(),
        inputs: [4, 9].map(|index| TranscribeInput {
            index, source: asr_data::AudioSource::Url(format!("file:///missing-{index}.wav")), mono: true,
        }).into(), response_tx,
    }).unwrap();
    let response = tokio::time::timeout(Duration::from_millis(100), response_rx).await;
    handle.abort();
    let response = response.expect("closed loader must notify pending request").unwrap();
    assert_eq!(response.iter().map(|item| item.index).collect::<Vec<_>>(), [4, 9]);
    assert!(response.iter().all(|item| item.result.is_err()));
    assert!(requests.lock().unwrap().is_empty());
    assert!(routes.lock().unwrap().is_empty());
}
#[tokio::test]
async fn router_duplicate_delivery_does_not_finish_request_early() {
    let id = PipelineRequestId(1);
    let (request, mut response) = pending(vec![4, 9]);
    let requests = Arc::new(Mutex::new(HashMap::from([(id, request)])));
    let routes = Arc::new(Mutex::new(HashMap::new()));
    let (tx, rx) = mpsc::channel(4);
    let handle = spawn_global_result_router(rx, requests, routes);
    for _ in 0..2 {
        tx.send(RoutedTranscribeItemOutcome::for_request(RequestItemRef { request_id: id, item_index: 4 }, failed_item(4))).await.unwrap();
    }
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(matches!(response.try_recv(), Err(oneshot::error::TryRecvError::Empty)), "duplicate outcome completed the request prematurely");
    tx.send(RoutedTranscribeItemOutcome::for_request(RequestItemRef { request_id: id, item_index: 9 }, failed_item(9))).await.unwrap();
    assert_eq!(response.await.unwrap().len(), 2);
    handle.abort();
}
#[tokio::test]
async fn router_closed_channel_fails_unfinished_requests() {
    let (request, response) = pending(vec![4, 9]);
    let requests = Arc::new(Mutex::new(HashMap::from([(PipelineRequestId(1), request)])));
    let routes = Arc::new(Mutex::new(HashMap::new()));
    let (tx, rx) = mpsc::channel(1);
    let handle = spawn_global_result_router(rx, requests.clone(), routes);
    drop(tx);
    handle.await.unwrap();
    let response = tokio::time::timeout(Duration::from_millis(100), response).await
        .expect("closed result router must finish pending requests").unwrap();
    assert_eq!(response.len(), 2);
    assert!(requests.lock().unwrap().is_empty());
}
