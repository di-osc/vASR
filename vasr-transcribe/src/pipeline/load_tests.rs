use super::*;
use asr_data::Waveform;
use std::time::Duration;
use tokio::sync::{Semaphore, oneshot};

fn budget() -> LoadedQueueMemoryBudget {
    LoadedQueueMemoryBudget {
        semaphore: Arc::new(Semaphore::new(1)),
        capacity_permits: 1,
        source: "test",
    }
}

fn jobs(request_item: Option<RequestItemRef>) -> Vec<LoadedJob> {
    ["left", "right"]
        .into_iter()
        .enumerate()
        .map(|(index, channel)| LoadedJob {
            _item_permit: None,
            index: 10 + index,
            request_item,
            channel,
            waveform: Waveform::new(vec![index as f32; 16], 16_000),
            timing: Default::default(),
            spans: Default::default(),
            _queue_memory_permit: None,
        })
        .collect()
}

fn task_stats() -> LoaderTaskStats {
    LoaderTaskStats {
        worker_id: 7,
        jobs: 2,
        ..Default::default()
    }
}

#[tokio::test]
async fn local_channels_progress_with_only_one_channel_of_memory() {
    let memory = budget();
    let stats = Arc::new(Mutex::new(LoaderStageStats::default()));
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut tasks = JoinSet::new();
    tasks.spawn(async { (Ok(jobs(None)), task_stats()) });
    tokio::time::timeout(Duration::from_secs(1), async {
        let send = send_next_loaded(&mut tasks, &memory, &tx, &stats);
        let consume = async {
            for (index, channel) in [(10, "left"), (11, "right")] {
                let job = rx.recv().await.unwrap().unwrap();
                assert_eq!((job.index, job.channel), (index, channel));
                assert_eq!(job.waveform.samples, vec![(index - 10) as f32; 16]);
                assert!(job.request_item.is_none());
                assert!(job._queue_memory_permit.is_some());
                assert_eq!(memory.semaphore.available_permits(), 0);
                assert!(rx.try_recv().is_err());
                drop(job);
            }
        };
        let (sent, ()) = tokio::join!(send, consume);
        assert_eq!(sent.unwrap(), Some(7));
    })
    .await
    .expect("channels must be sent before waiting for more memory");
    assert_eq!(memory.semaphore.available_permits(), 1);
    assert_eq!(stats.lock().unwrap().jobs, 2);
    assert_eq!(
        send_next_loaded(&mut tasks, &memory, &tx, &stats)
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn global_channels_register_all_routes_before_first_send_and_progress() {
    let memory = budget();
    let stats = Arc::new(Mutex::new(LoaderStageStats::default()));
    let (tx, mut rx) = mpsc::unbounded_channel();
    let item = RequestItemRef {
        request_id: PipelineRequestId(42),
        item_index: 3,
    };
    let (response_tx, _response_rx) = oneshot::channel();
    let pending = Arc::new(Mutex::new(HashMap::from([(
        item.request_id,
        PendingGlobalRequest {
            item_routes: HashMap::new(),
            log_id: "test".into(),
            remaining: 1,
            expected: [(3, "mono")].into(),
            duplicate_indices: Vec::new(),
            internal_indices: vec![10],
            outcomes: Vec::new(),
            stages: super::super::stages::RequestStageProgress {
                load_remaining: 1,
                vad_remaining: 1,
                ..Default::default()
            },
            response_tx,
        },
    )])));
    let routes = Arc::new(Mutex::new(HashMap::from([(10, item)])));
    let next_job_id = AtomicU64::new(11);
    tokio::time::timeout(Duration::from_secs(1), async {
        let send = send_global_loaded_result(
            Ok((Ok(jobs(Some(item))), task_stats())),
            &memory,
            &tx,
            &pending,
            &routes,
            &next_job_id,
            &stats,
        );
        let consume = async {
            for (index, channel) in [(10, "left"), (11, "right")] {
                let job = rx.recv().await.unwrap().unwrap();
                assert_eq!((job.index, job.channel), (index, channel));
                assert_eq!(job.waveform.samples, vec![(index - 10) as f32; 16]);
                let routed = job.request_item.unwrap();
                assert_eq!((routed.request_id, routed.item_index), (item.request_id, 3));
                {
                    let pending = pending.lock().unwrap();
                    let request = &pending[&item.request_id];
                    assert_eq!(request.remaining, 2);
                    assert_eq!(request.internal_indices, vec![10, 11]);
                    assert_eq!(request.stages.load_remaining, 0);
                    assert_eq!(request.stages.vad_remaining, 2);
                    assert_eq!(request.stages.load.audio_seconds, 0.002);
                    let routes = routes.lock().unwrap();
                    for index in [10, 11] {
                        assert_eq!(routes[&index].request_id, item.request_id);
                        assert_eq!(routes[&index].item_index, 3);
                    }
                }
                assert_eq!(memory.semaphore.available_permits(), 0);
                assert!(rx.try_recv().is_err());
                drop(job);
            }
        };
        let (sent, ()) = tokio::join!(send, consume);
        assert_eq!(sent.unwrap(), 7);
    })
    .await
    .expect("global channels must not reserve all permits before sending");
    assert_eq!(memory.semaphore.available_permits(), 1);
    assert_eq!(stats.lock().unwrap().jobs, 2);
}

// Exercise the same failure cases through both forwarding paths.
async fn forward(
    global: bool,
    message: Result<Vec<LoadedJob>, TranscribeItemOutcome>,
    memory: &LoadedQueueMemoryBudget,
    tx: &mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
) -> Result<usize> {
    let stats = Arc::new(Mutex::new(LoaderStageStats::default()));
    if global {
        send_global_loaded_result(
            Ok((message, task_stats())),
            memory,
            tx,
            &Arc::new(Mutex::new(HashMap::new())),
            &Arc::new(Mutex::new(HashMap::new())),
            &AtomicU64::new(11),
            &stats,
        )
        .await
    } else {
        let mut tasks = JoinSet::new();
        tasks.spawn(async move { (message, task_stats()) });
        Ok(send_next_loaded(&mut tasks, memory, tx, &stats)
            .await?
            .unwrap())
    }
}

#[tokio::test]
async fn forwarding_preserves_closed_channel_and_limiter_errors() {
    for global in [false, true] {
        let memory = budget();
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let error = forward(global, Ok(jobs(None)), &memory, &tx)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "loaded channel closed");
        assert_eq!(memory.semaphore.available_permits(), 1);

        let (tx, mut rx) = mpsc::unbounded_channel();
        memory.semaphore.close();
        let error = forward(global, Ok(jobs(None)), &memory, &tx)
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "loaded queue memory limiter closed");
        assert!(rx.try_recv().is_err());
    }
}

#[tokio::test]
async fn forwarding_preserves_loader_outcomes_without_acquiring_memory() {
    for global in [false, true] {
        let memory = budget();
        memory.semaphore.close();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let outcome = TranscribeItemOutcome {
            index: 9,
            channel: "mono",
            result: Err(anyhow::anyhow!("decode failed")),
            audio_seconds: 0.0,
            timing: Default::default(),
            spans: Default::default(),
            bad_component: Some("loader"),
        };
        assert_eq!(
            forward(global, Err(outcome), &memory, &tx).await.unwrap(),
            7
        );
        let outcome = rx.recv().await.unwrap().unwrap_err();
        assert_eq!(
            (outcome.index, outcome.channel, outcome.bad_component),
            (9, "mono", Some("loader"))
        );
        assert_eq!(outcome.result.unwrap_err().to_string(), "decode failed");
        assert_eq!(
            forward(global, Ok(Vec::new()), &memory, &tx).await.unwrap(),
            7
        );
        assert!(rx.try_recv().is_err());
    }
}

#[tokio::test]
async fn repeated_stereo_cancellation_before_publication_keeps_retired_ranges_bounded() {
    let registry = super::super::registry::RequestRegistry::new(Arc::default(), Arc::default());
    let next_job_id = Arc::new(AtomicU64::new(1));
    let memory = budget();
    let stats = Arc::default();
    let (tx, mut rx) = mpsc::unbounded_channel();
    for id in 1..=16 {
        let internal_index = next_job_id.fetch_add(1, Ordering::Relaxed) as usize;
        let item = RequestItemRef {
            request_id: PipelineRequestId(id),
            item_index: 4,
        };
        let (response_tx, response_rx) = oneshot::channel();
        registry.pending.lock().unwrap().insert(
            item.request_id,
            PendingGlobalRequest {
                item_routes: [((4, "mono"), internal_index)].into(),
                log_id: format!("stereo-cancel-{id}"),
                remaining: 1,
                expected: [(4, "mono")].into(),
                duplicate_indices: Vec::new(),
                internal_indices: vec![internal_index],
                outcomes: Vec::new(),
                stages: super::super::stages::RequestStageProgress {
                    load_remaining: 1,
                    vad_remaining: 1,
                    ..Default::default()
                },
                response_tx,
            },
        );
        registry.routes.lock().unwrap().insert(internal_index, item);
        let mut tasks = JoinSet::new();
        spawn_global_loader_task(
            &mut tasks,
            AudioLoader,
            GlobalLoadJob {
                internal_index,
                request_item: item,
                source: asr_data::AudioSource::from_pcm_s16le(vec![0; 640], 16_000, 2),
                mono: false,
                _item_permit: None,
                _source_memory: None,
            },
            0,
        );
        let result = tasks.join_next().await.unwrap();
        assert_eq!(
            result.as_ref().unwrap().0.as_ref().unwrap().len(),
            2,
            "fixture must finish the real stereo loader task"
        );
        // Cancel after loading but before the outer loader publishes channels.
        // The extra channel must still belong to this request if it has an ID.
        registry.cancel(item.request_id);
        assert!(response_rx.await.is_err());
        send_global_loaded_result(
            result,
            &memory,
            &tx,
            &registry.pending,
            &registry.routes,
            &next_job_id,
            &stats,
        )
        .await
        .unwrap();
        assert!(
            rx.try_recv().is_err(),
            "canceled stereo channels were published"
        );
    }
    assert_eq!(
        registry.retired_range_count(),
        1,
        "canceled stereo extra IDs leave permanent retirement holes"
    );
    assert_eq!(
        next_job_id.load(Ordering::Acquire),
        17,
        "unpublished canceled channels must not consume extra work IDs"
    );
    assert!(registry.pending.lock().unwrap().is_empty());
    assert!(registry.routes.lock().unwrap().is_empty());
    assert_eq!(memory.semaphore.available_permits(), 1);
}
