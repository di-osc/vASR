use super::{
    ASR_ENCODER_BATCH_COLLECT_WAIT_MS, AsrOptions, AsrSegmentJob, AsrStageStats,
    PendingGlobalRequest, PipelineRequestId, StageSpan, TranscribeItemTiming,
    audio_encoder_timing_seconds, format_seconds, format_speed, record_asr_encode_stats,
    record_request_encode_stages,
};
use anyhow::Result;
use asr_data::Waveform;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vasr::OfflineAsrModel as AsrModel;
use vasr::models::qwen3asr::{AudioFeatureEncodeTimings, EncodedAsrBatch, Qwen3ASR};

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentEncodeRequest {
    pub(super) segment: AsrSegmentJob,
    pub(super) stage_start: Instant,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
struct SegmentEncodeBatch {
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    reason: &'static str,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentEncodedMessage {
    pub(super) segments: Vec<AsrSegmentJob>,
    pub(super) stage_start: Instant,
    pub(super) encoder_timing: TranscribeItemTiming,
    pub(super) batch: Result<SegmentEncodedAsrBatch>,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
pub(super) enum SegmentEncodeMessage {
    Segment(SegmentEncodeRequest),
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
pub(super) enum SegmentEncoderOutput {
    Encoded(SegmentEncodedMessage),
}

#[cfg(feature = "paged-attn")]
pub(super) fn spawn_segment_encoder_worker(
    asr: Arc<dyn AsrModel>,
    asr_options: AsrOptions,
    mut request_rx: mpsc::UnboundedReceiver<SegmentEncodeMessage>,
    encoded_tx: mpsc::UnboundedSender<SegmentEncoderOutput>,
    stats: Arc<Mutex<AsrStageStats>>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut pending = VecDeque::new();
        while let Some(request) = match pending.pop_front() {
            Some(request) => Some(request),
            None => request_rx.recv().await,
        } {
            let SegmentEncodeMessage::Segment(request) = request;
            let request =
                collect_encoder_segment_batch(request, &mut request_rx, &mut pending).await;
            let encode_start = Instant::now();
            let segments_for_result = request.segments.clone();
            let segment_audio_seconds = segments_for_result
                .iter()
                .map(|segment| segment.waveform.duration_seconds())
                .sum::<f64>();
            let batch = tokio::task::spawn_blocking({
                let asr = Arc::clone(&asr);
                let asr_options = asr_options.clone();
                let segments = request.segments;
                move || run_segment_asr_encode(asr.as_ref(), &segments, &asr_options)
            })
            .await
            .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR audio encoder join error: {error}")));
            let encode_wall_seconds = encode_start.elapsed().as_secs_f64();
            let encode_span = StageSpan::new(encode_start, Instant::now());
            let encode_timings = batch
                .as_ref()
                .map(|batch| batch.timings)
                .unwrap_or_default();
            let encoder_timing = audio_encoder_timing_seconds(encode_timings);
            record_asr_encode_stats(
                &stats,
                segments_for_result.len(),
                segment_audio_seconds,
                encode_wall_seconds,
            );
            let encode_failed = batch.is_err();
            let mut segments_for_result = segments_for_result;
            for segment in &mut segments_for_result {
                segment.spans.audio_encoder = Some(encode_span);
            }
            record_request_encode_stages(&pending_requests, &segments_for_result, encode_span);
            let request_count = {
                let mut request_ids = HashSet::new();
                for segment in &segments_for_result {
                    if let Some(request_item) = segment.request_item {
                        request_ids.insert(request_item.request_id);
                    }
                }
                request_ids.len()
            };
            tracing::debug!(
                target: "vasr_transcribe::pipeline",
                "encode batch DONE | jobs={} reqs={} seg={} audio={} elapsed={} speed={} reason={} bad={}",
                segments_for_result
                    .iter()
                    .map(|segment| segment.job_index)
                    .collect::<HashSet<_>>()
                    .len(),
                request_count,
                segments_for_result.len(),
                format_seconds(segment_audio_seconds),
                format_seconds(encode_wall_seconds),
                format_speed(segment_audio_seconds, encode_wall_seconds),
                request.reason,
                usize::from(encode_failed),
            );
            let message = SegmentEncodedMessage {
                segments: segments_for_result,
                stage_start: request.stage_start,
                encoder_timing,
                batch,
            };
            if encoded_tx
                .send(SegmentEncoderOutput::Encoded(message))
                .is_err()
            {
                break;
            }
        }
    })
}
#[cfg(feature = "paged-attn")]
async fn collect_encoder_segment_batch(
    first: SegmentEncodeRequest,
    request_rx: &mut mpsc::UnboundedReceiver<SegmentEncodeMessage>,
    _pending: &mut VecDeque<SegmentEncodeMessage>,
) -> SegmentEncodeBatch {
    let mut stage_start = first.stage_start;
    let mut segments = vec![first.segment];
    let mut reason = "wait_timeout";
    while let Ok(SegmentEncodeMessage::Segment(request)) = request_rx.try_recv() {
        stage_start = stage_start.min(request.stage_start);
        segments.push(request.segment);
    }
    loop {
        match tokio::time::timeout(
            std::time::Duration::from_millis(ASR_ENCODER_BATCH_COLLECT_WAIT_MS),
            request_rx.recv(),
        )
        .await
        {
            Ok(Some(SegmentEncodeMessage::Segment(request))) => {
                stage_start = stage_start.min(request.stage_start);
                segments.push(request.segment);
            }
            Ok(None) => {
                reason = "source_closed";
                break;
            }
            Err(tokio::time::error::Elapsed { .. }) => break,
        }
    }
    segments.sort_by_key(|segment| (segment.job_index, segment.offset));
    SegmentEncodeBatch {
        segments,
        stage_start,
        reason,
    }
}
#[cfg(feature = "paged-attn")]
pub(super) fn send_segment_encode_request(
    asr_to_encoder_tx: &mpsc::UnboundedSender<SegmentEncodeMessage>,
    request: SegmentEncodeRequest,
) {
    if asr_to_encoder_tx
        .send(SegmentEncodeMessage::Segment(request))
        .is_err()
    {
        tracing::warn!(
            target: "vasr_transcribe::pipeline",
            "ASR audio feature encoder closed"
        );
    }
}
#[derive(Debug, Clone)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentAsrInference {
    pub(super) targets: Vec<(usize, usize, usize)>,
    pub(super) segment_audio_seconds: f64,
    pub(super) active_job_count: usize,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentEncodedAsrBatch {
    inference: SegmentAsrInference,
    waveforms: Vec<Waveform>,
    encoded: EncodedAsrBatch,
    timings: AudioFeatureEncodeTimings,
}

#[cfg(feature = "paged-attn")]
impl SegmentEncodedAsrBatch {
    pub(super) fn into_parts(self) -> (SegmentAsrInference, Vec<Waveform>, EncodedAsrBatch) {
        (self.inference, self.waveforms, self.encoded)
    }
}
#[cfg(feature = "paged-attn")]
pub(super) fn run_segment_asr_encode(
    asr: &dyn AsrModel,
    segments: &[AsrSegmentJob],
    asr_options: &AsrOptions,
) -> Result<SegmentEncodedAsrBatch> {
    let qwen3 = asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("staged ASR requires Qwen3ASR"))?;
    let waveforms: Vec<Waveform> = segments
        .iter()
        .map(|segment| segment.waveform.clone())
        .collect();
    let targets: Vec<(usize, usize, usize)> = segments
        .iter()
        .map(|segment| (segment.job_index, segment.segment_index, segment.offset))
        .collect();
    let active_job_count = segments
        .iter()
        .map(|segment| segment.job_index)
        .collect::<HashSet<_>>()
        .len();
    let segment_audio_seconds: f64 = waveforms.iter().map(Waveform::duration_seconds).sum();
    let encoded =
        qwen3.encode_waveforms_audio_features_to_cpu(waveforms.as_slice(), asr_options)?;
    let timings = encoded.audio_feature_encode_timings();
    Ok(SegmentEncodedAsrBatch {
        inference: SegmentAsrInference {
            targets,
            segment_audio_seconds,
            active_job_count,
        },
        waveforms,
        encoded,
        timings,
    })
}
