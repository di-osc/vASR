use super::*;
use std::future::Future;
use std::task::{Context, Poll, Waker};
use std::time::Duration;

fn request(job_index: usize, offset: usize, stage_start: Instant) -> SegmentEncodeRequest {
    SegmentEncodeRequest {
        segment: AsrSegmentJob {
            _item_permit: None,
            job_index,
            request_item: None,
            channel: "mono",
            timing: Default::default(),
            spans: Default::default(),
            job_audio_seconds: 0.001,
            speech_annotations: Vec::new(),
            segment_index: offset,
            segment_count: 1,
            offset,
            waveform: Waveform::new(vec![job_index as f32; 16], 16_000),
        },
        stage_start,
    }
}

#[tokio::test]
async fn batch_limit_leaves_queued_segments_and_preserves_sort_and_stage_start() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let now = Instant::now();
    let earliest = now - Duration::from_secs(1);
    for (job, offset, start) in [(1, 2, now), (1, 0, earliest), (3, 0, now), (4, 0, now)] {
        tx.send(SegmentEncodeMessage::Segment(request(job, offset, start)))
            .unwrap();
    }
    drop(tx);
    let first =
        collect_encoder_segment_batch(request(2, 0, now), &mut rx, &mut None, 3, 120.0).await;
    assert_eq!(first.reason, "segment_limit");
    assert_eq!(first.stage_start, earliest);
    assert_eq!(
        first
            .segments
            .iter()
            .map(|s| (s.job_index, s.offset))
            .collect::<Vec<_>>(),
        vec![(1, 0), (1, 2), (2, 0)]
    );
    assert_eq!(rx.len(), 2);

    let SegmentEncodeMessage::Segment(next) = rx.recv().await.unwrap();
    let second = collect_encoder_segment_batch(next, &mut rx, &mut None, 3, 120.0).await;
    assert_eq!(second.reason, "source_closed");
    assert_eq!(
        second
            .segments
            .iter()
            .map(|s| s.job_index)
            .collect::<Vec<_>>(),
        vec![3, 4]
    );
    assert!(rx.recv().await.is_none());
}

#[tokio::test]
async fn single_segment_limit_does_not_consume_next_message() {
    for limit in [0, 1] {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tx.send(SegmentEncodeMessage::Segment(request(1, 0, Instant::now())))
            .unwrap();
        let batch = collect_encoder_segment_batch(
            request(0, 0, Instant::now()),
            &mut rx,
            &mut None,
            limit,
            120.0,
        )
        .await;
        assert_eq!(batch.reason, "segment_limit");
        assert_eq!(batch.segments.len(), 1);
        assert_eq!(rx.len(), 1);
    }
}

#[tokio::test]
async fn deadline_is_not_reset_by_later_arrivals_or_ready_queue() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    // Manually poll to control arrivals without needing Tokio's test-util feature.
    let mut pending = None;
    let mut batch = Box::pin(collect_encoder_segment_batch(
        request(0, 0, Instant::now()),
        &mut rx,
        &mut pending,
        100,
        120.0,
    ));
    let mut cx = Context::from_waker(Waker::noop());
    assert!(batch.as_mut().poll(&mut cx).is_pending());
    tx.send(SegmentEncodeMessage::Segment(request(1, 0, Instant::now())))
        .unwrap();
    assert!(batch.as_mut().poll(&mut cx).is_pending());
    // Deliberately block this test's runtime: the explicit clock check must also
    // stop a ready recv, independently of when Tokio polls its timer driver.
    std::thread::sleep(Duration::from_millis(
        ASR_ENCODER_BATCH_COLLECT_WAIT_MS + 10,
    ));
    tx.send(SegmentEncodeMessage::Segment(request(2, 0, Instant::now())))
        .unwrap();
    tx.send(SegmentEncodeMessage::Segment(request(3, 0, Instant::now())))
        .unwrap();
    let Poll::Ready(result) = batch.as_mut().poll(&mut cx) else {
        panic!("an arrival must not restart the collection deadline");
    };
    assert_eq!(result.reason, "wait_timeout");
    // A recv already in flight may accept one arrival at the boundary, but
    // must not drain subsequent ready messages after the hard deadline.
    assert!((2..=3).contains(&result.segments.len()));
    drop(batch);
    assert_eq!(result.segments.len() + rx.len(), 4);
    assert!(!rx.is_empty());
}

#[tokio::test]
async fn continuous_arrivals_still_flush_before_producer_finishes() {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let producer = tokio::spawn(async move {
        let mut index = 1;
        loop {
            tokio::time::sleep(Duration::from_millis(1)).await;
            if tx
                .send(SegmentEncodeMessage::Segment(request(
                    index,
                    0,
                    Instant::now(),
                )))
                .is_err()
            {
                break;
            }
            index += 1;
        }
    });
    let result = tokio::time::timeout(
        Duration::from_millis(250),
        collect_encoder_segment_batch(
            request(0, 0, Instant::now()),
            &mut rx,
            &mut None,
            usize::MAX,
            120.0,
        ),
    )
    .await;
    producer.abort();
    let batch = result.expect("continuous input must not keep extending the deadline");
    assert_eq!(batch.reason, "wait_timeout");
    assert!(!batch.segments.is_empty());
}

#[tokio::test]
async fn idle_open_source_flushes_first_segment() {
    let (_tx, mut rx) = mpsc::unbounded_channel();
    let batch = tokio::time::timeout(
        Duration::from_secs(1),
        collect_encoder_segment_batch(request(0, 0, Instant::now()), &mut rx, &mut None, 32, 120.0),
    )
    .await
    .unwrap();
    assert_eq!(batch.reason, "wait_timeout");
    assert_eq!(batch.segments.len(), 1);
}

#[test]
fn retained_waveforms_and_inference_targets_share_the_mask_and_original_order() -> Result<()> {
    let inference = SegmentAsrInference {
        targets: vec![(9, 2, 8), (4, 1, 3), (9, 0, 1)],
        segment_audio_seconds: 99.,
        active_job_count: 2,
    };
    let waveforms = vec![
        Waveform::new(vec![2.; 16], 16_000),
        Waveform::new(vec![1.; 32], 16_000),
        Waveform::new(vec![3.; 48], 16_000),
    ];
    let first = waveforms[0].samples.as_ptr();
    let last = waveforms[2].samples.as_ptr();
    let (inference, waveforms) = inference.retain_waveforms(waveforms, &[true, false, true])?;
    assert_eq!(inference.targets, [(9, 2, 8), (9, 0, 1)]);
    assert_eq!(inference.active_job_count, 1);
    assert_eq!(inference.segment_audio_seconds, 0.004);
    assert_eq!(waveforms[0].samples, vec![2.; 16]);
    assert_eq!(waveforms[1].samples, vec![3.; 48]);
    assert_eq!(waveforms[0].samples.as_ptr(), first);
    assert_eq!(waveforms[1].samples.as_ptr(), last);
    let targets_ptr = inference.targets.as_ptr();
    let waveform_ptr = waveforms.as_ptr();
    let (inference, waveforms) = inference.retain_waveforms(waveforms, &[true, true])?;
    assert_eq!(inference.targets.as_ptr(), targets_ptr);
    assert_eq!(waveforms.as_ptr(), waveform_ptr);
    let (inference, waveforms) = inference.retain_waveforms(waveforms, &[false, false])?;
    assert!(inference.targets.is_empty() && waveforms.is_empty());
    assert_eq!(inference.active_job_count, 0);
    assert_eq!(inference.segment_audio_seconds, 0.);
    Ok(())
}
