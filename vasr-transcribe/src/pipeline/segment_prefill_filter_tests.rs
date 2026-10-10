use super::super::{PendingGlobalRequest, PipelineRequestId, RequestStageProgress};
use super::*;
use tokio::sync::{Semaphore, oneshot};

fn segment(id: u64, offset: usize) -> AsrSegmentJob {
    AsrSegmentJob {
        _item_permit: None,
        job_index: id as usize,
        request_item: Some(RequestItemRef {
            request_id: PipelineRequestId(id),
            item_index: 0,
        }),
        channel: "mono",
        timing: Default::default(),
        spans: Default::default(),
        job_audio_seconds: 0.001,
        speech_annotations: Vec::new(),
        segment_index: offset,
        segment_count: 1,
        offset,
        waveform: Waveform::new(vec![id as f32; 16], 16_000),
    }
}
fn message(segments: Vec<AsrSegmentJob>, stage_start: Instant) -> SegmentEncoderOutput {
    SegmentEncoderOutput::Encoded(SegmentEncodedMessage {
        segments,
        stage_start,
        encoder_timing: Default::default(),
        batch: Err(anyhow::anyhow!("original encoder failure")),
    })
}
fn pending() -> (
    PendingGlobalRequest,
    oneshot::Receiver<Vec<super::super::TranscribeItemOutcome>>,
) {
    let (response_tx, response_rx) = oneshot::channel();
    (
        PendingGlobalRequest {
            item_routes: HashMap::from([((0, "mono"), 0)]),
            log_id: "prefill-filter-test".into(),
            remaining: 1,
            expected: HashSet::from([(0, "mono")]),
            internal_indices: vec![0],
            duplicate_indices: Vec::new(),
            outcomes: Vec::new(),
            stages: RequestStageProgress::default(),
            response_tx,
        },
        response_rx,
    )
}

#[tokio::test]
async fn queued_mixed_batch_drops_cancelled_segments_and_releases_their_permits() -> Result<()> {
    let (live, _live_response) = pending();
    let (cancelled, _cancelled_response) = pending();
    let requests = Arc::new(Mutex::new(HashMap::from([
        (PipelineRequestId(1), live),
        (PipelineRequestId(2), cancelled),
    ])));
    let permits = Arc::new(Semaphore::new(1));
    let permit = Arc::new(permits.clone().acquire_owned().await?);
    let released = Arc::downgrade(&permit);
    let mut cancelled_segment = segment(2, 1);
    cancelled_segment._item_permit = Some(permit);
    let stage_start = Instant::now();
    let (tx, mut rx) = mpsc::unbounded_channel();
    tx.send(message(
        vec![segment(1, 2), cancelled_segment, segment(1, 0)],
        stage_start,
    ))?;
    requests.lock().unwrap().remove(&PipelineRequestId(2));
    let retained = retain_live_encoded_message(rx.recv().await.unwrap(), &Some(requests))?.unwrap();
    let SegmentEncoderOutput::Encoded(retained) = retained;
    assert_eq!(
        retained
            .segments
            .iter()
            .map(|s| (s.job_index, s.offset))
            .collect::<Vec<_>>(),
        [(1, 2), (1, 0)]
    );
    assert_eq!(retained.stage_start, stage_start);
    assert_eq!(
        retained.batch.unwrap_err().to_string(),
        "original encoder failure"
    );
    assert!(released.upgrade().is_none());
    assert_eq!(permits.available_permits(), 1);
    Ok(())
}

#[tokio::test]
async fn queued_cancelled_prefix_does_not_consume_the_following_live_batch() -> Result<()> {
    let (live, _response) = pending();
    let requests = Some(Arc::new(Mutex::new(HashMap::from([(
        PipelineRequestId(1),
        live,
    )]))));
    let (tx, mut rx) = mpsc::unbounded_channel();
    tx.send(message(vec![segment(2, 0)], Instant::now()))?;
    tx.send(message(vec![segment(1, 4)], Instant::now()))?;
    assert!(retain_live_encoded_message(rx.recv().await.unwrap(), &requests)?.is_none());
    let SegmentEncoderOutput::Encoded(retained) =
        retain_live_encoded_message(rx.recv().await.unwrap(), &requests)?.unwrap();
    assert_eq!(retained.segments[0].offset, 4);
    Ok(())
}

#[test]
fn all_live_messages_preserve_original_segment_storage_and_untracked_requests() -> Result<()> {
    let mut untracked = segment(1, 3);
    untracked.request_item = None;
    let segments = vec![untracked];
    let original = segments.as_ptr();
    let SegmentEncoderOutput::Encoded(retained) =
        retain_live_encoded_message(message(segments, Instant::now()), &None)?.unwrap();
    assert_eq!(retained.segments.as_ptr(), original);
    assert_eq!(retained.segments[0].offset, 3);
    assert!(retained.batch.is_err());
    Ok(())
}

struct UnusedAsr;
impl AsrModel for UnusedAsr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn transcribe(&self, _: &[Waveform], _: &AsrOptions) -> Result<Vec<asr_data::Timeline>> {
        panic!("cancelled/error encoded messages must not invoke model inference")
    }
}
#[tokio::test]
async fn real_prefill_worker_filters_mixed_queued_failures_before_delivery() -> Result<()> {
    let (live, _response) = pending();
    let requests = Some(Arc::new(Mutex::new(HashMap::from([(
        PipelineRequestId(1),
        live,
    )]))));
    let (tx, rx) = mpsc::unbounded_channel();
    let (out_tx, mut out_rx) = mpsc::unbounded_channel();
    tx.send(message(vec![segment(2, 0)], Instant::now()))?;
    tx.send(message(
        vec![segment(1, 2), segment(2, 1), segment(1, 0)],
        Instant::now(),
    ))?;
    drop(tx);
    let worker = spawn_segment_prefill_worker(
        Arc::new(UnusedAsr),
        AsrOptions::default(),
        rx,
        out_tx,
        None,
        requests,
    );
    worker.await??;
    let SegmentPrefillMessage::Failure(failure) = out_rx.recv().await.unwrap() else {
        panic!("expected original encoder failure")
    };
    assert_eq!(
        failure
            .segments
            .iter()
            .map(|s| (s.job_index, s.offset))
            .collect::<Vec<_>>(),
        [(1, 2), (1, 0)]
    );
    assert_eq!(failure.error.to_string(), "original encoder failure");
    assert!(out_rx.recv().await.is_none());
    Ok(())
}
