use super::{
    ASR_ENCODER_BATCH_COLLECT_WAIT_MS, AsrOptions, AsrSegmentJob, AsrStageStats,
    PendingGlobalRequest, PipelineRequestId, StageSpan, TranscribeItemTiming,
    audio_encoder_timing_seconds, format_seconds, format_speed, record_asr_encode_stats,
    record_request_encode_stages,
};
use anyhow::Result;
use asr_data::Waveform;
use std::collections::{HashMap, HashSet};
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
// 参数是流水线各阶段的句柄与配置，逐项传递比打包成结构体更直观。
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_segment_encoder_worker(
    asr: Arc<dyn AsrModel>,
    asr_options: AsrOptions,
    mut request_rx: mpsc::UnboundedReceiver<SegmentEncodeMessage>,
    encoded_tx: mpsc::UnboundedSender<SegmentEncoderOutput>,
    stats: Arc<Mutex<AsrStageStats>>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    max_segments: usize,
    max_audio_seconds: f64,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let mut pending = None;
        loop {
            let first = match pending.take() {
                Some(request) => request,
                None => match request_rx.recv().await {
                    Some(SegmentEncodeMessage::Segment(request)) => request,
                    None => break,
                },
            };
            let mut request = collect_encoder_segment_batch(
                first,
                &mut request_rx,
                &mut pending,
                max_segments,
                max_audio_seconds,
            )
            .await;
            request
                .segments
                .retain(|segment| super::request_is_live(&pending_requests, segment.request_item));
            if request.segments.is_empty() {
                continue;
            }
            let encode_start = Instant::now();
            let segments_for_result = request.segments.clone();
            let segment_audio_seconds = segments_for_result
                .iter()
                .map(|segment| segment.waveform.duration_seconds())
                .sum::<f64>();
            let oversized_segment_seconds = request
                .segments
                .iter()
                .map(|segment| segment.waveform.duration_seconds())
                .find(|seconds| *seconds > max_audio_seconds);
            let batch = if let Some(seconds) = oversized_segment_seconds {
                Err(anyhow::anyhow!(
                    "ASR encoder segment duration {seconds:.3}s exceeds --encoder-batch-max-audio-seconds={max_audio_seconds:.3}; reduce --vad-max-segment-sec or raise the encoder batch duration"
                ))
            } else {
                super::blocking_result(
                    tokio::task::spawn_blocking({
                        let asr = Arc::clone(&asr);
                        let asr_options = asr_options.clone();
                        let segments = request.segments;
                        move || run_segment_asr_encode(asr.as_ref(), &segments, &asr_options)
                    })
                    .await,
                )?
            };
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
            if let Err(error) = &batch {
                tracing::warn!(
                    target: "vasr_transcribe::pipeline",
                    "ASR audio encoder batch failed | segments={} audio_seconds={:.3} max_audio_seconds={:.3} reason={} error={:#}",
                    segments_for_result.len(),
                    segment_audio_seconds,
                    max_audio_seconds,
                    request.reason,
                    error
                );
            }
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
                anyhow::bail!("prefill channel closed");
            }
        }
        Ok(())
    })
}
#[cfg(feature = "paged-attn")]
async fn collect_encoder_segment_batch(
    first: SegmentEncodeRequest,
    request_rx: &mut mpsc::UnboundedReceiver<SegmentEncodeMessage>,
    pending: &mut Option<SegmentEncodeRequest>,
    max_segments: usize,
    max_audio_seconds: f64,
) -> SegmentEncodeBatch {
    // Start the collection window with the first received segment, not its upstream stage time.
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_millis(ASR_ENCODER_BATCH_COLLECT_WAIT_MS);
    let max_segments = max_segments.max(1);
    let mut stage_start = first.stage_start;
    let mut audio_seconds = first.segment.waveform.duration_seconds();
    let mut reason = if audio_seconds > max_audio_seconds {
        "segment_exceeds_audio_limit"
    } else {
        "wait_timeout"
    };
    let mut segments = vec![first.segment];
    loop {
        if audio_seconds > max_audio_seconds {
            break;
        }
        if segments.len() >= max_segments {
            reason = "segment_limit";
            break;
        }
        if audio_seconds >= max_audio_seconds {
            reason = "audio_duration_limit";
            break;
        }
        // timeout_at alone may accept an immediately ready recv even after the deadline.
        if tokio::time::Instant::now() >= deadline {
            break;
        }
        match tokio::time::timeout_at(deadline, request_rx.recv()).await {
            Ok(Some(SegmentEncodeMessage::Segment(request))) => {
                let next_audio_seconds = request.segment.waveform.duration_seconds();
                if audio_seconds + next_audio_seconds > max_audio_seconds {
                    // Keep the first non-fitting segment for the next batch, in FIFO order.
                    *pending = Some(request);
                    reason = "audio_duration_limit";
                    break;
                }
                audio_seconds += next_audio_seconds;
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
) -> Result<()> {
    if asr_to_encoder_tx
        .send(SegmentEncodeMessage::Segment(request))
        .is_err()
    {
        anyhow::bail!("ASR audio feature encoder closed");
    }
    Ok(())
}
#[derive(Debug, Clone)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentAsrInference {
    pub(super) targets: Vec<(usize, usize, usize)>,
    pub(super) segment_audio_seconds: f64,
    pub(super) active_job_count: usize,
}

#[cfg(feature = "paged-attn")]
impl SegmentAsrInference {
    fn retain_waveforms(
        self,
        waveforms: Vec<Waveform>,
        keep: &[bool],
    ) -> Result<(Self, Vec<Waveform>)> {
        if keep.len() != waveforms.len() || keep.len() != self.targets.len() {
            anyhow::bail!("segment waveform retention lengths do not match");
        }
        if keep.iter().all(|&keep| keep) {
            return Ok((self, waveforms));
        }
        let targets = self
            .targets
            .into_iter()
            .zip(keep)
            .filter_map(|(target, &keep)| keep.then_some(target))
            .collect::<Vec<_>>();
        let waveforms = waveforms
            .into_iter()
            .zip(keep)
            .filter_map(|(waveform, &keep)| keep.then_some(waveform))
            .collect::<Vec<_>>();
        let segment_audio_seconds = waveforms.iter().map(Waveform::duration_seconds).sum();
        let active_job_count = targets
            .iter()
            .map(|target| target.0)
            .collect::<HashSet<_>>()
            .len();
        Ok((
            Self {
                targets,
                segment_audio_seconds,
                active_job_count,
            },
            waveforms,
        ))
    }
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
    pub(super) fn retain_audios(self, keep: &[bool]) -> Result<Self> {
        if keep.len() != self.waveforms.len()
            || keep.len() != self.inference.targets.len()
            || keep.len() != self.encoded.audio_len()
        {
            anyhow::bail!("segment encoded retention lengths do not match");
        }
        if keep.iter().all(|&keep| keep) {
            return Ok(self);
        }
        let encoded = self.encoded.retain_audios(keep)?;
        let (inference, waveforms) = self.inference.retain_waveforms(self.waveforms, keep)?;
        Ok(Self {
            inference,
            waveforms,
            encoded,
            timings: self.timings,
        })
    }

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
    let encoded = match qwen3
        .encode_waveforms_audio_features_to_cpu(waveforms.as_slice(), asr_options)
    {
        Ok(encoded) => encoded,
        Err(error) if is_cuda_oom(&error) && segments.len() > 1 => {
            let split_at = split_encoder_batch_at_half_duration(segments);
            tracing::warn!(
                target: "vasr_transcribe::pipeline",
                "retrying ASR audio encoder OOM with smaller batches | segments={} audio_seconds={:.3} split_at={}",
                segments.len(),
                segment_audio_seconds,
                split_at
            );
            let left = run_segment_asr_encode(asr, &segments[..split_at], asr_options)?;
            let right = run_segment_asr_encode(asr, &segments[split_at..], asr_options)?;
            return merge_segment_asr_batches(left, right, segments);
        }
        Err(error) => return Err(error),
    };
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

#[cfg(feature = "paged-attn")]
fn is_cuda_oom(error: &anyhow::Error) -> bool {
    let message = format!("{error:#}").to_ascii_lowercase();
    message.contains("cuda_error_out_of_memory") || message.contains("out of memory")
}

#[cfg(feature = "paged-attn")]
fn split_encoder_batch_at_half_duration(segments: &[AsrSegmentJob]) -> usize {
    let total_seconds = segments
        .iter()
        .map(|segment| segment.waveform.duration_seconds())
        .sum::<f64>();
    let halfway = total_seconds / 2.0;
    let mut accumulated = 0.0;
    for (index, segment) in segments.iter().take(segments.len() - 1).enumerate() {
        accumulated += segment.waveform.duration_seconds();
        if accumulated >= halfway {
            return index + 1;
        }
    }
    segments.len() / 2
}

#[cfg(feature = "paged-attn")]
fn merge_segment_asr_batches(
    left: SegmentEncodedAsrBatch,
    right: SegmentEncodedAsrBatch,
    segments: &[AsrSegmentJob],
) -> Result<SegmentEncodedAsrBatch> {
    let mut timings = left.timings;
    timings.prepare_us = timings.prepare_us.saturating_add(right.timings.prepare_us);
    timings.stack_us = timings.stack_us.saturating_add(right.timings.stack_us);
    timings.forward_us = timings.forward_us.saturating_add(right.timings.forward_us);
    timings.d2h_us = timings.d2h_us.saturating_add(right.timings.d2h_us);
    timings.audio_tokens = timings
        .audio_tokens
        .saturating_add(right.timings.audio_tokens);

    let mut targets = left.inference.targets;
    targets.extend(right.inference.targets);
    let mut waveforms = left.waveforms;
    waveforms.extend(right.waveforms);
    let encoded = EncodedAsrBatch::merge(vec![left.encoded, right.encoded])?;
    let segment_audio_seconds = segments
        .iter()
        .map(|segment| segment.waveform.duration_seconds())
        .sum();
    let active_job_count = segments
        .iter()
        .map(|segment| segment.job_index)
        .collect::<HashSet<_>>()
        .len();

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

#[cfg(all(test, feature = "paged-attn"))]
#[path = "segment_encode_tests.rs"]
mod tests;
