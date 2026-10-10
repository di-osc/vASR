use super::*;

fn failed_item(index: usize) -> TranscribeItemOutcome {
    global_pipeline_failure_outcomes([index], "item failure")
        .pop()
        .unwrap()
}
fn pending(
    indices: Vec<usize>,
) -> (
    PendingGlobalRequest,
    oneshot::Receiver<Vec<TranscribeItemOutcome>>,
) {
    let (response_tx, rx) = oneshot::channel();
    let request = PendingGlobalRequest {
        item_routes: indices
            .iter()
            .map(|index| ((*index, "mono"), *index))
            .collect(),
        log_id: "lifecycle-test".into(),
        remaining: indices.len(),
        expected: indices.iter().map(|index| (*index, "mono")).collect(),
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
    let handle = spawn_global_submit_worker(
        rx,
        load_tx,
        requests.clone(),
        routes.clone(),
        Arc::new(AtomicU64::new(1)),
        Arc::new(AtomicU64::new(1)),
        Arc::new(AtomicBool::new(false)),
    );
    let (response_tx, response_rx) = oneshot::channel();
    tx.send(GlobalTranscribeRequest {
        log_id: "closed-loader".into(),
        inputs: [4, 9]
            .map(|index| TranscribeInput {
                index,
                source: asr_data::AudioSource::Url(format!("file:///missing-{index}.wav")),
                mono: true,
            })
            .into(),
        response_tx,
        request_id: None,
        source_memory: None,
        cancellation: None,
    })
    .unwrap();
    let response = tokio::time::timeout(Duration::from_millis(100), response_rx).await;
    handle.abort();
    let response = response
        .expect("closed loader must notify pending request")
        .unwrap();
    assert_eq!(
        response.iter().map(|item| item.index).collect::<Vec<_>>(),
        [4, 9]
    );
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
        tx.send(RoutedTranscribeItemOutcome::for_request(
            RequestItemRef {
                request_id: id,
                item_index: 4,
            },
            failed_item(4),
        ))
        .await
        .unwrap();
    }
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(
        matches!(
            response.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ),
        "duplicate outcome completed the request prematurely"
    );
    tx.send(RoutedTranscribeItemOutcome::for_request(
        RequestItemRef {
            request_id: id,
            item_index: 9,
        },
        failed_item(9),
    ))
    .await
    .unwrap();
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
    assert!(handle.await.unwrap().is_err());
    let response = tokio::time::timeout(Duration::from_millis(100), response)
        .await
        .expect("closed result router must finish pending requests")
        .unwrap();
    assert_eq!(response.len(), 2);
    assert!(requests.lock().unwrap().is_empty());
}

struct PanickingAsr;
impl vasr::OfflineAsrModel for PanickingAsr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn transcribe(&self, _: &[Waveform], _: &AsrOptions) -> anyhow::Result<Vec<Timeline>> {
        panic!("injected ASR worker failure")
    }
}
fn panic_pipeline() -> GlobalAsyncTranscribePipeline {
    GlobalAsyncTranscribePipeline::start(
        AudioLoader,
        None,
        Arc::new(PanickingAsr),
        AsrOptions::default(),
        VadOptions::default(),
        PipelineWorkerOptions::default(),
    )
}
#[tokio::test]
async fn worker_failure_notifies_service_and_every_request() {
    let pipeline = panic_pipeline();
    let mut fatal = pipeline.subscribe_fatal_errors();
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        pipeline.run(vec![TranscribeInput {
            index: 7,
            source: asr_data::AudioSource::from_pcm_s16le(vec![0; 320], 16_000, 1),
            mono: true,
        }]),
    )
    .await
    .expect("worker failure must resolve request");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].index, 7);
    assert!(result[0].result.is_err());
    tokio::time::timeout(Duration::from_millis(100), fatal.changed())
        .await
        .expect("worker failure must notify service")
        .unwrap();
    assert!(
        fatal
            .borrow()
            .as_ref()
            .is_some_and(|error| error.contains("injected ASR worker failure")),
        "the fixture must reach its intended ASR panic: {:?}",
        fatal.borrow()
    );
}
#[tokio::test]
async fn closed_submit_preserves_all_input_indices() {
    let mut pipeline = panic_pipeline();
    let (tx, rx) = mpsc::unbounded_channel();
    drop(rx);
    pipeline.submit_tx = tx;
    let response = pipeline
        .run(
            [4, 9]
                .map(|index| TranscribeInput {
                    index,
                    source: asr_data::AudioSource::Url(format!("file:///missing-{index}")),
                    mono: true,
                })
                .into(),
        )
        .await;
    assert_eq!(
        response.iter().map(|item| item.index).collect::<Vec<_>>(),
        [4, 9]
    );
}
#[test]
fn global_failure_retains_completed_channel_and_fills_only_missing_channel() {
    let (mut request, mut response) = pending(vec![4]);
    request.expected = [(4, "left"), (4, "right")].into();
    let mut completed = failed_item(4);
    completed.channel = "left";
    completed.bad_component = Some("loader");
    request.outcomes.push(completed);
    request.fail("decoder died");
    let response = response.try_recv().unwrap();
    assert_eq!(response.len(), 2);
    assert_eq!(response[0].channel, "left");
    assert_eq!(response[0].bad_component, Some("loader"));
    assert_eq!(response[1].channel, "right");
    assert_eq!(response[1].bad_component, Some("recognizer"));
}

#[tokio::test]
async fn raw_duplicate_after_completed_request_does_not_fail_other_request() {
    let first_id = PipelineRequestId(1);
    let second_id = PipelineRequestId(2);
    let (first, first_response) = pending(vec![4]);
    let (second, mut second_response) = pending(vec![9]);
    let requests = Arc::new(Mutex::new(HashMap::from([
        (first_id, first),
        (second_id, second),
    ])));
    let routes = Arc::new(Mutex::new(HashMap::from([
        (
            4,
            RequestItemRef {
                request_id: first_id,
                item_index: 4,
            },
        ),
        (
            9,
            RequestItemRef {
                request_id: second_id,
                item_index: 9,
            },
        ),
    ])));
    let (tx, rx) = mpsc::channel(4);
    let handle = spawn_global_result_router(rx, requests, routes);
    tx.send(failed_item(4).into()).await.unwrap();
    first_response.await.unwrap();
    tx.send(failed_item(4).into()).await.unwrap();
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(
        matches!(
            second_response.try_recv(),
            Err(oneshot::error::TryRecvError::Empty)
        ),
        "late raw duplicate killed unrelated request"
    );
    tx.send(failed_item(9).into()).await.unwrap();
    assert_eq!(second_response.await.unwrap()[0].index, 9);
    handle.abort();
}

fn supervisor_fixture(
    limits: PipelineLimits,
) -> (Arc<PipelineSupervisor>, watch::Receiver<Option<String>>) {
    let registry = RequestRegistry::new(Arc::default(), Arc::default());
    let (fatal_tx, fatal_rx) = watch::channel(None);
    (
        PipelineSupervisor::new(registry, Arc::new(AtomicBool::new(false)), fatal_tx, limits),
        fatal_rx,
    )
}
#[tokio::test]
async fn every_critical_stage_panic_error_or_unexpected_exit_fails_all_and_closes_admission() {
    for stage in [
        "submit",
        "load",
        "VAD",
        "ASR",
        "encoder",
        "prefill",
        "decode",
        "result router",
    ] {
        for kind in ["panic", "error", "exit"] {
            let (supervisor, mut fatal) = supervisor_fixture(PipelineLimits {
                active_slots: 1,
                safe_capacity: 1,
                source_bytes: 1024,
            });
            let id = PipelineRequestId(1);
            let (request, response) = pending(vec![4, 9]);
            supervisor
                .registry
                .pending
                .lock()
                .unwrap()
                .insert(id, request);
            supervisor.registry.routes.lock().unwrap().insert(
                4,
                RequestItemRef {
                    request_id: id,
                    item_index: 4,
                },
            );
            injected_worker(&supervisor, stage, async move {
                match kind {
                    "panic" => panic!("injected {stage} panic"),
                    "error" => Err(anyhow::anyhow!("injected {stage} error")),
                    _ => Ok(()),
                }
            })
            .await;
            tokio::time::timeout(Duration::from_secs(1), fatal.changed())
                .await
                .unwrap()
                .unwrap();
            let error = fatal.borrow().clone().unwrap();
            assert!(error.contains(stage), "{error}");
            assert!(error.contains(kind) || kind == "exit", "{error}");
            let outcomes = response.await.unwrap();
            assert_eq!(
                outcomes
                    .iter()
                    .map(|outcome| outcome.index)
                    .collect::<Vec<_>>(),
                [4, 9]
            );
            assert!(supervisor.registry.pending.lock().unwrap().is_empty());
            assert!(supervisor.registry.routes.lock().unwrap().is_empty());
            assert!(supervisor.requests.clone().acquire_owned().await.is_err());
            assert!(supervisor.items.clone().acquire_owned().await.is_err());
            assert!(supervisor.reserve_source(1).await.is_err());
        }
    }
}
#[tokio::test]
async fn stopping_is_normal_and_wakes_blocked_admission() {
    let (supervisor, fatal) = supervisor_fixture(PipelineLimits {
        active_slots: 1,
        safe_capacity: 1,
        source_bytes: 1024,
    });
    let _request = supervisor.requests.clone().acquire_owned().await.unwrap();
    let waiter = tokio::spawn({
        let supervisor = supervisor.clone();
        async move { supervisor.requests.clone().acquire_owned().await }
    });
    supervisor.supervise("idle worker", tokio::spawn(std::future::pending::<()>()));
    supervisor.stop();
    assert!(waiter.await.unwrap().is_err());
    tokio::task::yield_now().await;
    assert!(matches!(fatal.has_changed(), Ok(false)));
    assert!(fatal.borrow().is_none());
}
#[tokio::test]
async fn source_budget_waits_and_rejects_oversized_reservations() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits {
        active_slots: 1,
        safe_capacity: 1,
        source_bytes: 1024,
    });
    assert!(
        supervisor.reserve_source(2048).await.is_err(),
        "oversized reservation must not silently underaccount source bytes"
    );
    let first = supervisor.reserve_source(1024).await.unwrap();
    let mut second = Box::pin(supervisor.reserve_source(1));
    assert!(
        tokio::time::timeout(Duration::from_millis(10), &mut second)
            .await
            .is_err()
    );
    drop(first);
    let second = tokio::time::timeout(Duration::from_millis(100), second)
        .await
        .unwrap()
        .unwrap();
    drop(second);
    assert!(
        tokio::time::timeout(Duration::from_millis(100), supervisor.reserve_source(1024))
            .await
            .is_ok()
    );
}
#[tokio::test]
async fn cancellation_removes_registration_but_inflight_item_lease_drains_on_completion() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits {
        active_slots: 1,
        safe_capacity: 1,
        source_bytes: 1024,
    });
    let id = PipelineRequestId(1);
    let (request, _response) = pending(vec![4]);
    supervisor
        .registry
        .pending
        .lock()
        .unwrap()
        .insert(id, request);
    let item = RequestItemRef {
        request_id: id,
        item_index: 4,
    };
    supervisor.registry.routes.lock().unwrap().insert(4, item);
    let permit = Arc::new(supervisor.items.clone().acquire_owned().await.unwrap());
    supervisor.registry.retain_item_permit(item, permit.clone());
    let guard = RequestCancellation {
        registry: supervisor.registry.clone(),
        id,
        signal: Arc::default(),
    };
    drop(guard);
    assert!(supervisor.registry.pending.lock().unwrap().is_empty());
    assert!(supervisor.registry.routes.lock().unwrap().is_empty());
    assert_eq!(
        supervisor.items.available_permits(),
        1,
        "inflight work must still own its item slot"
    );
    supervisor.registry.finish(failed_item(4).into()).unwrap();
    drop(permit);
    assert_eq!(supervisor.items.available_permits(), 2);
}
#[tokio::test]
async fn unknown_raw_outcome_is_corruption_even_after_retirement() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits::default());
    assert!(
        supervisor
            .registry
            .finish(failed_item(1234).into())
            .is_err()
    );
}
#[test]
fn encoded_source_memory_budget_formula_is_bounded() {
    assert_eq!(
        source_memory_budget(Some(48 * 1024 * 1024 * 1024)),
        2 * 1024 * 1024 * 1024
    );
    assert_eq!(
        source_memory_budget(Some(16 * 1024 * 1024 * 1024)),
        1024 * 1024 * 1024
    );
    assert_eq!(source_memory_budget(Some(1024)), 512 * 1024 * 1024);
}

fn request_message(
    id: u64,
    indices: impl IntoIterator<Item = usize>,
    signal: Arc<CancellationSignal>,
    source_memory: Option<Arc<SourceReservation>>,
) -> (
    GlobalTranscribeRequest,
    oneshot::Receiver<Vec<TranscribeItemOutcome>>,
) {
    let (response_tx, response_rx) = oneshot::channel();
    (
        GlobalTranscribeRequest {
            log_id: format!("test-{id}"),
            request_id: Some(PipelineRequestId(id)),
            inputs: indices
                .into_iter()
                .map(|index| TranscribeInput {
                    index,
                    mono: true,
                    source: AudioSource::Url(format!("file:///missing-{id}-{index}")),
                })
                .collect(),
            response_tx,
            cancellation: Some(signal),
            source_memory,
        },
        response_rx,
    )
}
fn submit_fixture(
    supervisor: &Arc<PipelineSupervisor>,
) -> (
    mpsc::UnboundedSender<GlobalTranscribeRequest>,
    mpsc::UnboundedReceiver<GlobalLoadJob>,
    JoinHandle<Result<()>>,
) {
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();
    let (load_tx, load_rx) = mpsc::unbounded_channel();
    let handle = spawn_supervised_submit_worker(
        submit_rx,
        load_tx,
        supervisor.registry.pending.clone(),
        supervisor.registry.routes.clone(),
        Arc::new(AtomicU64::new(100)),
        Arc::new(AtomicU64::new(1)),
        Arc::downgrade(supervisor),
    );
    (submit_tx, load_rx, handle)
}
#[tokio::test]
async fn oversized_request_dispatches_per_item_and_short_request_progresses() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits {
        active_slots: 2,
        safe_capacity: 1,
        source_bytes: 1024,
    });
    let (tx, mut load_rx, handle) = submit_fixture(&supervisor);
    let (long, long_response) = request_message(1, 0..32, Arc::default(), None);
    tx.send(long).unwrap();
    let first = tokio::time::timeout(Duration::from_millis(100), load_rx.recv())
        .await
        .unwrap()
        .unwrap();
    let second = tokio::time::timeout(Duration::from_millis(100), load_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(supervisor.items.available_permits(), 0);
    assert_eq!(
        supervisor.registry.routes.lock().unwrap().len(),
        2,
        "unadmitted items must not allocate active routes"
    );
    assert_eq!(
        supervisor.registry.pending.lock().unwrap()[&PipelineRequestId(1)]
            .internal_indices
            .len(),
        2
    );
    let (short, short_response) = request_message(2, [99], Arc::default(), None);
    tx.send(short).unwrap();
    tokio::task::yield_now().await;
    supervisor
        .registry
        .finish(RoutedTranscribeItemOutcome::for_request(
            first.request_item,
            failed_item(first.request_item.item_index),
        ))
        .unwrap();
    drop(first);
    supervisor
        .registry
        .finish(RoutedTranscribeItemOutcome::for_request(
            second.request_item,
            failed_item(second.request_item.item_index),
        ))
        .unwrap();
    drop(second);
    let mut short_seen = false;
    for _ in 0..3 {
        let job = tokio::time::timeout(Duration::from_millis(100), load_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(
            supervisor.registry.routes.lock().unwrap().len() <= 2,
            "completed item routes accumulated behind the large request"
        );
        short_seen |= job.request_item.request_id == PipelineRequestId(2);
        supervisor
            .registry
            .finish(RoutedTranscribeItemOutcome::for_request(
                job.request_item,
                failed_item(job.request_item.item_index),
            ))
            .unwrap();
        drop(job);
        if short_seen {
            break;
        }
    }
    assert!(
        short_seen,
        "short request was held behind all 32 long-request items"
    );
    assert_eq!(short_response.await.unwrap()[0].index, 99);
    supervisor.registry.cancel(PipelineRequestId(1));
    drop(long_response);
    handle.abort();
}
#[tokio::test]
async fn canceled_item_waiter_drops_source_and_never_resurrects_routes() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits {
        active_slots: 1,
        safe_capacity: 1,
        source_bytes: 1024,
    });
    let (tx, mut load_rx, handle) = submit_fixture(&supervisor);
    let held = supervisor
        .items
        .clone()
        .acquire_many_owned(2)
        .await
        .unwrap();
    let signal = Arc::new(CancellationSignal::default());
    let guard = RequestCancellation {
        registry: supervisor.registry.clone(),
        id: PipelineRequestId(1),
        signal: signal.clone(),
    };
    let (request, response) = request_message(
        1,
        [4, 9],
        signal,
        Some(supervisor.reserve_source(1024).await.unwrap()),
    );
    tx.send(request).unwrap();
    for _ in 0..100 {
        if supervisor.registry.is_live(PipelineRequestId(1)) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(supervisor.registry.is_live(PipelineRequestId(1)));
    drop(response);
    drop(guard);
    let reservation =
        tokio::time::timeout(Duration::from_millis(100), supervisor.reserve_source(1024))
            .await
            .expect("canceled waiter kept encoded source ownership")
            .unwrap();
    assert!(supervisor.registry.pending.lock().unwrap().is_empty());
    assert!(supervisor.registry.routes.lock().unwrap().is_empty());
    drop(held);
    tokio::task::yield_now().await;
    assert!(
        load_rx.try_recv().is_err(),
        "canceled item was published after waking"
    );
    assert_eq!(supervisor.items.available_permits(), 2);
    drop(reservation);
    handle.abort();
}
#[tokio::test]
async fn canceled_before_submit_is_never_registered() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits::default());
    let (tx, mut load_rx, handle) = submit_fixture(&supervisor);
    let (request, response) = request_message(1, [4], Arc::default(), None);
    drop(response);
    tx.send(request).unwrap();
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;
    assert!(load_rx.try_recv().is_err());
    assert!(supervisor.registry.pending.lock().unwrap().is_empty());
    assert!(supervisor.registry.routes.lock().unwrap().is_empty());
    handle.abort();
}

struct BlockingAsr {
    started: AtomicBool,
    gate: (Mutex<bool>, std::sync::Condvar),
}
impl vasr::OfflineAsrModel for BlockingAsr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn transcribe(&self, audio: &[Waveform], _: &AsrOptions) -> Result<Vec<Timeline>> {
        if !self.started.swap(true, Ordering::AcqRel) {
            let (lock, notify) = &self.gate;
            let mut released = lock.lock().unwrap();
            while !*released {
                released = notify.wait(released).unwrap();
            }
        }
        Ok(audio
            .iter()
            .map(|waveform| Timeline::new("test", waveform.duration_ms() as usize))
            .collect())
    }
}
impl BlockingAsr {
    fn release(&self) {
        *self.gate.0.lock().unwrap() = true;
        self.gate.1.notify_all();
    }
}
impl Drop for BlockingAsr {
    fn drop(&mut self) {
        self.release();
    }
}
#[tokio::test]
async fn canceled_inflight_request_keeps_work_budget_until_completion_and_other_request_succeeds() {
    let asr = Arc::new(BlockingAsr {
        started: AtomicBool::new(false),
        gate: (Mutex::new(false), std::sync::Condvar::new()),
    });
    struct ReleaseOnDrop(Arc<BlockingAsr>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.0.release();
        }
    }
    let _release = ReleaseOnDrop(asr.clone());
    let pipeline = Arc::new(GlobalAsyncTranscribePipeline::start_with_limits(
        AudioLoader,
        None,
        asr.clone(),
        AsrOptions::default(),
        VadOptions::default(),
        PipelineWorkerOptions::default(),
        PipelineLimits {
            active_slots: 2,
            safe_capacity: 1,
            source_bytes: 2048,
        },
    ));
    let input = |index| TranscribeInput {
        index,
        source: AudioSource::from_pcm_s16le(vec![0; 320], 16_000, 1),
        mono: true,
    };
    let first = tokio::spawn({
        let pipeline = pipeline.clone();
        let input = input(4);
        async move { pipeline.run(vec![input]).await }
    });
    let wait = tokio::time::timeout(Duration::from_secs(1), async {
        while !asr.started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await;
    if wait.is_err() {
        asr.release();
        panic!("fixture did not reach the in-flight ASR call");
    }
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    assert!(
        pipeline
            .supervisor
            .registry
            .pending
            .lock()
            .unwrap()
            .is_empty()
    );
    assert!(
        pipeline
            .supervisor
            .registry
            .routes
            .lock()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        pipeline.supervisor.items.available_permits(),
        1,
        "canceled forward is still running"
    );
    let source = tokio::time::timeout(Duration::from_millis(100), pipeline.reserve_source(2048))
        .await
        .unwrap()
        .unwrap();
    drop(source);
    let second = tokio::spawn({
        let pipeline = pipeline.clone();
        let input = input(9);
        async move { pipeline.run(vec![input]).await }
    });
    asr.release();
    let outcomes = tokio::time::timeout(Duration::from_secs(1), second)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].index, 9);
    assert!(outcomes[0].result.is_ok());
    assert_eq!(pipeline.supervisor.items.available_permits(), 2);
    assert!(
        pipeline
            .supervisor
            .registry
            .pending
            .lock()
            .unwrap()
            .is_empty()
    );
    assert!(
        pipeline
            .supervisor
            .registry
            .routes
            .lock()
            .unwrap()
            .is_empty()
    );
    assert!(pipeline.fatal_error_tx.borrow().is_none());
}
#[tokio::test]
async fn canceled_raw_result_does_not_finish_unrelated_request() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits::default());
    let (first, _) = pending(vec![4]);
    let (second, mut response) = pending(vec![9]);
    supervisor.registry.pending.lock().unwrap().extend([
        (PipelineRequestId(1), first),
        (PipelineRequestId(2), second),
    ]);
    supervisor.registry.routes.lock().unwrap().extend([
        (
            4,
            RequestItemRef {
                request_id: PipelineRequestId(1),
                item_index: 4,
            },
        ),
        (
            9,
            RequestItemRef {
                request_id: PipelineRequestId(2),
                item_index: 9,
            },
        ),
    ]);
    supervisor.registry.cancel(PipelineRequestId(1));
    supervisor.registry.finish(failed_item(4).into()).unwrap();
    assert!(matches!(
        response.try_recv(),
        Err(oneshot::error::TryRecvError::Empty)
    ));
    supervisor.registry.finish(failed_item(9).into()).unwrap();
    assert_eq!(response.await.unwrap()[0].index, 9);
}
#[test]
fn unknown_request_reference_is_corruption_not_cancellation() {
    let registry = RequestRegistry::new(Arc::default(), Arc::default());
    assert!(
        registry
            .finish(RoutedTranscribeItemOutcome::for_request(
                RequestItemRef {
                    request_id: PipelineRequestId(999),
                    item_index: 4,
                },
                failed_item(4)
            ))
            .is_err()
    );
}
#[tokio::test]
async fn dropping_last_pipeline_owner_marks_stopping_before_channels_close() {
    let pipeline = panic_pipeline();
    let supervisor = pipeline.supervisor.clone();
    let fatal = pipeline.subscribe_fatal_errors();
    drop(pipeline);
    assert!(supervisor.requests.is_closed());
    tokio::task::yield_now().await;
    assert!(
        fatal.borrow().is_none(),
        "normal ownership teardown became a fatal worker exit"
    );
}

#[tokio::test]
async fn oversized_source_admission_finishes_every_original_index_without_enqueue() {
    let pipeline = GlobalAsyncTranscribePipeline::start_with_limits(
        AudioLoader,
        None,
        Arc::new(PanickingAsr),
        AsrOptions::default(),
        VadOptions::default(),
        PipelineWorkerOptions::default(),
        PipelineLimits {
            active_slots: 1,
            safe_capacity: 1,
            source_bytes: 1024,
        },
    );
    let outcomes = pipeline
        .run(
            [4, 9]
                .map(|index| TranscribeInput {
                    index,
                    mono: true,
                    source: AudioSource::EncodedBytes(vec![0; 600]),
                })
                .into(),
        )
        .await;
    assert_eq!(
        outcomes
            .iter()
            .map(|outcome| outcome.index)
            .collect::<Vec<_>>(),
        [4, 9]
    );
    assert!(outcomes.iter().all(|outcome| {
        outcome
            .result
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("encoded source budget")
    }));
    assert!(
        pipeline
            .supervisor
            .registry
            .pending
            .lock()
            .unwrap()
            .is_empty()
    );
    assert!(
        pipeline
            .supervisor
            .registry
            .routes
            .lock()
            .unwrap()
            .is_empty()
    );
    assert!(pipeline.fatal_error_tx.borrow().is_none());
}

#[tokio::test]
async fn finished_channel_routes_retire_before_request_finishes_and_late_raw_duplicate_is_safe() {
    let (request, response) = pending(vec![4, 9]);
    let id = PipelineRequestId(1);
    let registry = RequestRegistry::new(
        Arc::new(Mutex::new(HashMap::from([(id, request)]))),
        Arc::new(Mutex::new(HashMap::from([
            (
                4,
                RequestItemRef {
                    request_id: id,
                    item_index: 4,
                },
            ),
            (
                9,
                RequestItemRef {
                    request_id: id,
                    item_index: 9,
                },
            ),
        ]))),
    );
    registry.finish(failed_item(4).into()).unwrap();
    assert_eq!(registry.routes.lock().unwrap().len(), 1);
    registry.finish(failed_item(4).into()).unwrap();
    assert_eq!(registry.pending.lock().unwrap()[&id].remaining, 1);
    registry.finish(failed_item(9).into()).unwrap();
    assert_eq!(response.await.unwrap().len(), 2);
    assert!(registry.routes.lock().unwrap().is_empty());
}
#[tokio::test]
async fn stereo_channel_routes_retire_individually_but_item_permit_waits_for_both_channels() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits {
        active_slots: 1,
        safe_capacity: 1,
        source_bytes: 1024,
    });
    let id = PipelineRequestId(1);
    let item = RequestItemRef {
        request_id: id,
        item_index: 4,
    };
    let (mut request, response) = pending(vec![4]);
    request.remaining = 2;
    request.expected = [(4, "left"), (4, "right")].into();
    request.internal_indices = vec![10, 11];
    request.item_routes = [((4, "left"), 10), ((4, "right"), 11)].into();
    supervisor
        .registry
        .pending
        .lock()
        .unwrap()
        .insert(id, request);
    supervisor
        .registry
        .routes
        .lock()
        .unwrap()
        .extend([(10, item), (11, item)]);
    supervisor.registry.retain_item_permit(
        item,
        Arc::new(supervisor.items.clone().acquire_owned().await.unwrap()),
    );
    let mut left = failed_item(10);
    left.channel = "left";
    supervisor.registry.finish(left.into()).unwrap();
    assert_eq!(supervisor.registry.routes.lock().unwrap().len(), 1);
    assert_eq!(
        supervisor.items.available_permits(),
        1,
        "left channel freed the whole canonical item prematurely"
    );
    let mut late_left = failed_item(10);
    late_left.channel = "left";
    supervisor.registry.finish(late_left.into()).unwrap();
    let mut right = failed_item(11);
    right.channel = "right";
    supervisor.registry.finish(right.into()).unwrap();
    assert_eq!(supervisor.items.available_permits(), 2);
    assert!(supervisor.registry.routes.lock().unwrap().is_empty());
    let outcomes = response.await.unwrap();
    assert_eq!(
        outcomes
            .iter()
            .map(|outcome| (outcome.index, outcome.channel))
            .collect::<Vec<_>>(),
        [(4, "left"), (4, "right")]
    );
}
#[tokio::test]
async fn canceled_unadmitted_items_do_not_consume_internal_job_ids() {
    let (supervisor, _) = supervisor_fixture(PipelineLimits {
        active_slots: 1,
        safe_capacity: 1,
        source_bytes: 1024,
    });
    let (tx, rx) = mpsc::unbounded_channel();
    let (load_tx, mut load_rx) = mpsc::unbounded_channel();
    let next_job_id = Arc::new(AtomicU64::new(1));
    let handle = spawn_supervised_submit_worker(
        rx,
        load_tx,
        supervisor.registry.pending.clone(),
        supervisor.registry.routes.clone(),
        Arc::new(AtomicU64::new(100)),
        next_job_id.clone(),
        Arc::downgrade(&supervisor),
    );
    let held = supervisor
        .items
        .clone()
        .acquire_many_owned(2)
        .await
        .unwrap();
    let signal = Arc::new(CancellationSignal::default());
    let guard = RequestCancellation {
        registry: supervisor.registry.clone(),
        id: PipelineRequestId(1),
        signal: signal.clone(),
    };
    let (request, response) = request_message(1, 0..32, signal, None);
    tx.send(request).unwrap();
    for _ in 0..100 {
        if supervisor.registry.is_live(PipelineRequestId(1)) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(supervisor.registry.is_live(PipelineRequestId(1)));
    assert_eq!(
        next_job_id.load(Ordering::Acquire),
        1,
        "unadmitted canceled items leave permanent holes in retired ID ranges"
    );
    drop(response);
    drop(guard);
    drop(held);
    tokio::task::yield_now().await;
    assert!(load_rx.try_recv().is_err());
    handle.abort();
}

struct PanickingVad;
impl VadModel for PanickingVad {
    fn detect(&self, _: &Waveform, _: &VadOptions) -> Result<Vec<vasr_vad::VadSegment>> {
        panic!("injected real VAD subtask panic")
    }
    fn new_stream(&self, _: &VadOptions) -> Result<Box<dyn vasr_vad::StreamingVadSession>> {
        anyhow::bail!("streaming VAD is not used in this test")
    }
}

async fn assert_last_vad_subtask_failure_is_supervised(
    vad: Option<Arc<dyn VadModel>>,
    close_downstream: bool,
    expected_error: &str,
) {
    let (supervisor, mut fatal) = supervisor_fixture(PipelineLimits::default());
    let id = PipelineRequestId(1);
    let (request, response) = pending(vec![4, 9]);
    supervisor
        .registry
        .pending
        .lock()
        .unwrap()
        .insert(id, request);
    supervisor
        .registry
        .routes
        .lock()
        .unwrap()
        .extend([4, 9].map(|index| {
            (
                index,
                RequestItemRef {
                    request_id: id,
                    item_index: index,
                },
            )
        }));
    let (loader_tx, loader_rx) = mpsc::unbounded_channel();
    let (asr_tx, asr_rx) = mpsc::unbounded_channel();
    let _asr_rx = if close_downstream {
        drop(asr_rx);
        None
    } else {
        Some(asr_rx)
    };
    let memory = LoadedQueueMemoryBudget {
        semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        capacity_permits: 1,
        source: "real-vad-test",
    };
    supervisor.supervise(
        "VAD",
        spawn_vad_worker(
            vad,
            VadOptions::default(),
            2,
            loader_rx,
            asr_tx,
            memory,
            Some(supervisor.registry.pending.clone()),
            Arc::default(),
        ),
    );
    loader_tx
        .send(Ok(LoadedJob {
            index: 4,
            request_item: Some(RequestItemRef {
                request_id: id,
                item_index: 4,
            }),
            channel: "mono",
            waveform: Waveform::new(vec![0.0; 160], 16_000),
            timing: Default::default(),
            spans: Default::default(),
            _item_permit: None,
            _queue_memory_permit: None,
        }))
        .unwrap();
    // Keep the upstream sender alive: the outer worker cannot rely on a later job
    // or channel closure to observe the only subtask's failure.
    tokio::time::timeout(Duration::from_secs(1), fatal.changed())
        .await
        .expect("last VAD subtask failure must notify service while loader stays open")
        .unwrap();
    assert!(
        fatal
            .borrow()
            .as_ref()
            .is_some_and(|error| error.contains(expected_error)),
        "intended VAD failure was not reached: {:?}",
        fatal.borrow()
    );
    let outcomes = tokio::time::timeout(Duration::from_secs(1), response)
        .await
        .expect("VAD failure must finish every pending item")
        .unwrap();
    assert_eq!(
        outcomes
            .iter()
            .map(|outcome| outcome.index)
            .collect::<Vec<_>>(),
        [4, 9]
    );
    assert!(outcomes.iter().all(|outcome| outcome.result.is_err()));
    assert!(supervisor.registry.pending.lock().unwrap().is_empty());
    assert!(supervisor.registry.routes.lock().unwrap().is_empty());
    drop(loader_tx);
}

#[tokio::test]
async fn last_vad_subtask_panic_fails_service_with_upstream_still_open() {
    assert_last_vad_subtask_failure_is_supervised(
        Some(Arc::new(PanickingVad)),
        false,
        "injected real VAD subtask panic",
    )
    .await;
}

#[tokio::test]
async fn last_vad_subtask_closed_downstream_fails_service_with_upstream_still_open() {
    assert_last_vad_subtask_failure_is_supervised(None, true, "ASR stage channel closed").await;
}

#[tokio::test]
async fn stopping_rejects_submit_registration_before_worker_abort_is_polled() {
    for queued_before_stop in [true, false] {
        let (supervisor, fatal) = supervisor_fixture(PipelineLimits::default());
        // Exercise a submit worker that has not yet polled a cooperative abort.
        // No supervisor registration keeps this interleaving deterministic.
        let (tx, mut load_rx, handle) = submit_fixture(&supervisor);
        let (request, response) = request_message(1, [4, 9], Arc::default(), None);
        if queued_before_stop {
            tx.send(request).unwrap();
            supervisor.stop();
        } else {
            supervisor.stop();
            tx.send(request).unwrap();
        }
        let result = tokio::time::timeout(Duration::from_secs(1), response).await;
        handle.abort();
        let outcomes = result
            .expect("submission after stop must resolve before worker abort")
            .unwrap();
        assert_eq!(
            outcomes
                .iter()
                .map(|outcome| outcome.index)
                .collect::<Vec<_>>(),
            [4, 9]
        );
        assert!(outcomes.iter().all(|outcome| outcome.result.is_err()));
        assert!(supervisor.registry.pending.lock().unwrap().is_empty());
        assert!(supervisor.registry.routes.lock().unwrap().is_empty());
        assert!(load_rx.try_recv().is_err());
        assert!(
            fatal.borrow().is_none(),
            "normal stop must not publish a fatal error"
        );
    }
}

#[tokio::test]
async fn public_compat_zero_capacity_keeps_automatic_window() {
    let pipeline = panic_pipeline();
    assert_eq!(pipeline.supervisor.requests.available_permits(), 8);
    assert_eq!(pipeline.supervisor.items.available_permits(), 16);
}

#[tokio::test]
async fn public_compat_repeated_caller_indices_finish_distinct_failed_sources() {
    let pipeline = panic_pipeline();
    let inputs = ["one", "two"]
        .map(|name| TranscribeInput {
            index: 4,
            source: AudioSource::Path(std::path::PathBuf::from(format!(
                "/missing-public-compat-{name}.wav"
            ))),
            mono: true,
        })
        .into();
    let outcomes = tokio::time::timeout(Duration::from_secs(1), pipeline.run(inputs))
        .await
        .expect("different sources sharing a caller index must both complete");
    assert_eq!(
        outcomes.iter().map(|item| item.index).collect::<Vec<_>>(),
        [4, 4]
    );
    assert!(outcomes.iter().all(|item| item.result.is_err()));
    assert!(
        pipeline
            .supervisor
            .registry
            .pending
            .lock()
            .unwrap()
            .is_empty()
    );
    assert!(
        pipeline
            .supervisor
            .registry
            .routes
            .lock()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn public_compat_repeated_indices_preserve_success_duplicates_and_channels() {
    for (sizes, mono, channels, expected) in [
        ([320, 640], true, 1, 2),
        ([320, 320], true, 1, 2),
        ([640, 1280], false, 2, 4),
    ] {
        let asr = Arc::new(BlockingAsr {
            started: AtomicBool::new(true),
            gate: (Mutex::new(true), std::sync::Condvar::new()),
        });
        let pipeline = GlobalAsyncTranscribePipeline::start(
            AudioLoader,
            None,
            asr,
            AsrOptions::default(),
            VadOptions::default(),
            PipelineWorkerOptions::default(),
        );
        let inputs = sizes
            .map(|size| TranscribeInput {
                index: 4,
                mono,
                source: AudioSource::from_pcm_s16le(vec![0; size], 16_000, channels),
            })
            .into();
        let outcomes = tokio::time::timeout(Duration::from_secs(1), pipeline.run(inputs))
            .await
            .unwrap();
        assert_eq!(outcomes.len(), expected);
        assert!(
            outcomes
                .iter()
                .all(|item| item.index == 4 && item.result.is_ok())
        );
        if !mono {
            assert_eq!(
                outcomes
                    .iter()
                    .filter(|item| item.channel == "left")
                    .count(),
                2
            );
            assert_eq!(
                outcomes
                    .iter()
                    .filter(|item| item.channel == "right")
                    .count(),
                2
            );
        }
        assert!(
            pipeline
                .supervisor
                .registry
                .pending
                .lock()
                .unwrap()
                .is_empty()
        );
        assert!(
            pipeline
                .supervisor
                .registry
                .routes
                .lock()
                .unwrap()
                .is_empty()
        );
    }
}

#[tokio::test]
async fn public_compat_aliases_restore_indices_on_admission_failure() {
    let pipeline = panic_pipeline();
    pipeline.shutdown();
    let outcomes = pipeline
        .run(
            [0, 1]
                .map(|_| TranscribeInput {
                    index: 4,
                    mono: true,
                    source: AudioSource::EncodedBytes(vec![0]),
                })
                .into(),
        )
        .await;
    assert_eq!(
        outcomes.iter().map(|item| item.index).collect::<Vec<_>>(),
        [4, 4]
    );
    assert!(outcomes.iter().all(|item| item.result.is_err()));
}

#[tokio::test]
async fn public_compat_cancelled_aliases_reclaim_all_registration_and_permits() {
    let asr = Arc::new(BlockingAsr {
        started: AtomicBool::new(false),
        gate: (Mutex::new(false), std::sync::Condvar::new()),
    });
    struct Release(Arc<BlockingAsr>);
    impl Drop for Release {
        fn drop(&mut self) {
            self.0.release();
        }
    }
    let _release = Release(asr.clone());
    let pipeline = Arc::new(GlobalAsyncTranscribePipeline::start(
        AudioLoader,
        None,
        asr.clone(),
        AsrOptions::default(),
        VadOptions::default(),
        PipelineWorkerOptions::default(),
    ));
    let task = tokio::spawn({
        let pipeline = pipeline.clone();
        async move {
            pipeline
                .run(
                    [320, 640]
                        .map(|size| TranscribeInput {
                            index: 4,
                            mono: true,
                            source: AudioSource::from_pcm_s16le(vec![0; size], 16_000, 1),
                        })
                        .into(),
                )
                .await
        }
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while !asr.started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    asr.release();
    tokio::time::timeout(Duration::from_secs(1), async {
        while pipeline.supervisor.items.available_permits() != 16 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(pipeline.supervisor.requests.available_permits(), 8);
    assert!(
        pipeline
            .supervisor
            .registry
            .pending
            .lock()
            .unwrap()
            .is_empty()
    );
    assert!(
        pipeline
            .supervisor
            .registry
            .routes
            .lock()
            .unwrap()
            .is_empty()
    );
}
