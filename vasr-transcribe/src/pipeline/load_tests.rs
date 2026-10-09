use super::*;
use std::time::Duration;
use tokio::sync::{Semaphore, oneshot};
use asr_data::Waveform;

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
            log_id: "test".into(),
            failure_indices: vec![3],
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
    tokio::time::timeout(Duration::from_secs(1), async {
        let send = send_global_loaded_result(
            Ok((Ok(jobs(Some(item))), task_stats())),
            &memory,
            &tx,
            &pending,
            &routes,
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
