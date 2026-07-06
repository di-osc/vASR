use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::{AsrSegmentJob, PendingGlobalRequest, PipelineRequestId, StageSpan};

#[derive(Debug, Default)]
pub(super) struct RequestStageProgress {
    pub(super) load_remaining: usize,
    pub(super) vad_remaining: usize,
    pub(super) encode_remaining: usize,
    pub(super) decode_remaining: usize,
    pub(super) load_logged: bool,
    pub(super) vad_logged: bool,
    pub(super) encode_logged: bool,
    pub(super) decode_logged: bool,
    pub(super) load: StageProgress,
    pub(super) vad: StageProgress,
    pub(super) encode: StageProgress,
    pub(super) prefill: StageProgress,
    pub(super) decode: StageProgress,
}

#[derive(Debug, Default)]
pub(super) struct StageProgress {
    pub(super) run_span: Option<StageSpan>,
    pub(super) queue_span: Option<StageSpan>,
    pub(super) active_spans: Vec<StageSpan>,
    pub(super) queue_spans: Vec<StageSpan>,
    pub(super) audio_seconds: f64,
}

pub(super) fn record_request_load_stage(
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    request_id: PipelineRequestId,
    span: StageSpan,
    audio_seconds: f64,
    loaded_jobs: usize,
) {
    if let Ok(mut pending) = pending_requests.lock() {
        let Some(request) = pending.get_mut(&request_id) else {
            return;
        };
        request.stages.load.run_span = Some(merge_optional_span(
            request.stages.load.run_span,
            Some(span),
        ));
        request.stages.load.active_spans.push(span);
        request.stages.load.audio_seconds += audio_seconds;
        if loaded_jobs > 1 {
            request.stages.vad_remaining += loaded_jobs - 1;
        }
        request.stages.load_remaining = request.stages.load_remaining.saturating_sub(1);
        if request.stages.load_remaining == 0 && !request.stages.load_logged {
            request.stages.load_logged = true;
            log_request_stage(&request.log_id, "load", &request.stages.load);
        }
    }
}

pub(super) fn record_request_vad_stage(
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    request_id: PipelineRequestId,
    span: StageSpan,
    queue_span: Option<StageSpan>,
    audio_seconds: f64,
    segment_count: usize,
) {
    if let Ok(mut pending) = pending_requests.lock() {
        let Some(request) = pending.get_mut(&request_id) else {
            return;
        };
        request.stages.vad.run_span =
            Some(merge_optional_span(request.stages.vad.run_span, Some(span)));
        request.stages.vad.queue_span = merge_stage_span(request.stages.vad.queue_span, queue_span);
        request.stages.vad.active_spans.push(span);
        if let Some(queue_span) = queue_span {
            request.stages.vad.queue_spans.push(queue_span);
        }
        request.stages.vad.audio_seconds += audio_seconds;
        request.stages.encode_remaining += segment_count;
        request.stages.decode_remaining += segment_count;
        request.stages.vad_remaining = request.stages.vad_remaining.saturating_sub(1);
        if request.stages.vad_remaining == 0 && !request.stages.vad_logged {
            request.stages.vad_logged = true;
            log_request_stage(&request.log_id, "vad", &request.stages.vad);
            log_ready_downstream_stages(request);
        }
    }
}

#[cfg(feature = "paged-attn")]
fn record_request_encode_stage(
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    request_id: PipelineRequestId,
    span: StageSpan,
    queue_span: Option<StageSpan>,
    audio_seconds: f64,
    segment_count: usize,
) {
    if let Ok(mut pending) = pending_requests.lock() {
        let Some(request) = pending.get_mut(&request_id) else {
            return;
        };
        request.stages.encode.run_span = Some(merge_optional_span(
            request.stages.encode.run_span,
            Some(span),
        ));
        request.stages.encode.queue_span =
            merge_stage_span(request.stages.encode.queue_span, queue_span);
        request.stages.encode.active_spans.push(span);
        if let Some(queue_span) = queue_span {
            request.stages.encode.queue_spans.push(queue_span);
        }
        request.stages.encode.audio_seconds += audio_seconds;
        request.stages.encode_remaining = request
            .stages
            .encode_remaining
            .saturating_sub(segment_count);
        log_ready_downstream_stages(request);
    }
}

#[cfg(feature = "paged-attn")]
pub(super) fn record_request_encode_stages(
    pending_requests: &Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    segments: &[AsrSegmentJob],
    span: StageSpan,
) {
    let Some(pending_requests) = pending_requests else {
        return;
    };
    let mut by_request: HashMap<PipelineRequestId, (f64, usize, Option<StageSpan>)> =
        HashMap::new();
    for segment in segments {
        let Some(request_item) = segment.request_item else {
            continue;
        };
        let entry = by_request.entry(request_item.request_id).or_default();
        entry.0 += segment.waveform.duration_seconds();
        entry.1 += 1;
        let queue_span = segment
            .spans
            .vad
            .and_then(|vad| StageSpan::try_new(vad.end, span.start));
        entry.2 = merge_stage_span(entry.2, queue_span);
    }
    for (request_id, (audio_seconds, segment_count, queue_span)) in by_request {
        record_request_encode_stage(
            pending_requests,
            request_id,
            span,
            queue_span,
            audio_seconds,
            segment_count,
        );
    }
}

#[allow(dead_code)]
fn record_request_decode_stage(
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    request_id: PipelineRequestId,
    decode_span: StageSpan,
    prefill_span: Option<StageSpan>,
    prefill_queue_span: Option<StageSpan>,
    decode_queue_span: Option<StageSpan>,
    audio_seconds: f64,
    segment_count: usize,
) {
    if let Ok(mut pending) = pending_requests.lock() {
        let Some(request) = pending.get_mut(&request_id) else {
            return;
        };
        if let Some(prefill_span) = prefill_span {
            request.stages.prefill.run_span = Some(merge_optional_span(
                request.stages.prefill.run_span,
                Some(prefill_span),
            ));
            request.stages.prefill.queue_span =
                merge_stage_span(request.stages.prefill.queue_span, prefill_queue_span);
            request.stages.prefill.active_spans.push(prefill_span);
            if let Some(prefill_queue_span) = prefill_queue_span {
                request.stages.prefill.queue_spans.push(prefill_queue_span);
            }
            request.stages.prefill.audio_seconds += audio_seconds;
        }
        request.stages.decode.run_span = Some(merge_optional_span(
            request.stages.decode.run_span,
            Some(decode_span),
        ));
        request.stages.decode.queue_span =
            merge_stage_span(request.stages.decode.queue_span, decode_queue_span);
        request.stages.decode.active_spans.push(decode_span);
        if let Some(decode_queue_span) = decode_queue_span {
            request.stages.decode.queue_spans.push(decode_queue_span);
        }
        request.stages.decode.audio_seconds += audio_seconds;
        request.stages.decode_remaining = request
            .stages
            .decode_remaining
            .saturating_sub(segment_count);
        log_ready_downstream_stages(request);
    }
}

pub(super) fn log_ready_downstream_stages(request: &mut PendingGlobalRequest) {
    if !request.stages.vad_logged {
        return;
    }
    if request.stages.encode_remaining == 0
        && request.stages.encode.run_span.is_some()
        && !request.stages.encode_logged
    {
        request.stages.encode_logged = true;
        log_request_stage(&request.log_id, "encode", &request.stages.encode);
    }
    if request.stages.decode_remaining == 0
        && request.stages.decode.run_span.is_some()
        && !request.stages.decode_logged
    {
        request.stages.decode_logged = true;
        if request.stages.prefill.run_span.is_some() {
            log_request_stage(&request.log_id, "prefill", &request.stages.prefill);
        }
        log_request_stage(&request.log_id, "decode", &request.stages.decode);
    }
}

#[cfg(feature = "paged-attn")]
#[allow(dead_code)]
pub(super) fn record_request_decode_stages(
    pending_requests: &Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    segments: &[AsrSegmentJob],
    decode_span: StageSpan,
    prefill_span: Option<StageSpan>,
) {
    let Some(pending_requests) = pending_requests else {
        return;
    };
    let mut by_request: HashMap<
        PipelineRequestId,
        (f64, usize, Option<StageSpan>, Option<StageSpan>),
    > = HashMap::new();
    for segment in segments {
        let Some(request_item) = segment.request_item else {
            continue;
        };
        let entry = by_request.entry(request_item.request_id).or_default();
        entry.0 += segment.waveform.duration_seconds();
        entry.1 += 1;
        let (prefill_queue_span, decode_queue_span) = if let Some(prefill_span) = prefill_span {
            (
                segment
                    .spans
                    .audio_encoder
                    .and_then(|encode| StageSpan::try_new(encode.end, prefill_span.start)),
                StageSpan::try_new(prefill_span.end, decode_span.start),
            )
        } else {
            (
                None,
                segment
                    .spans
                    .audio_encoder
                    .and_then(|encode| StageSpan::try_new(encode.end, decode_span.start)),
            )
        };
        entry.2 = merge_stage_span(entry.2, prefill_queue_span);
        entry.3 = merge_stage_span(entry.3, decode_queue_span);
    }
    for (request_id, (audio_seconds, segment_count, prefill_queue_span, decode_queue_span)) in
        by_request
    {
        record_request_decode_stage(
            pending_requests,
            request_id,
            decode_span,
            prefill_span,
            prefill_queue_span,
            decode_queue_span,
            audio_seconds,
            segment_count,
        );
    }
}

fn log_request_stage(request_id: &str, stage: &'static str, progress: &StageProgress) {
    let active_seconds = merged_spans_seconds(progress.active_spans.clone());
    let wall_seconds = merge_stage_span(progress.queue_span, progress.run_span)
        .map(|span| span.end.duration_since(span.start).as_secs_f64())
        .unwrap_or_default();
    let queue_seconds = (wall_seconds - active_seconds).max(0.0);
    let elapsed_seconds = wall_seconds;
    let speed = progress.audio_seconds / active_seconds.max(f64::EPSILON);
    let e2e_speed = progress.audio_seconds / elapsed_seconds.max(f64::EPSILON);
    let audio_label = match stage {
        "load" | "vad" => "input_audio",
        _ => "speech_audio",
    };
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "{} {:<7} {:<5} | {}={:>6.1}s | elapsed={:>6.3}s | active={:>6.3}s | queue={:>6.3}s wall={:>6.3}s | speed={:>6.1}x e2e={:>6.1}x",
        request_id,
        stage,
        "DONE",
        audio_label,
        progress.audio_seconds,
        elapsed_seconds,
        active_seconds,
        queue_seconds,
        wall_seconds,
        speed,
        e2e_speed
    );
}

pub(super) fn merge_optional_span(left: Option<StageSpan>, right: Option<StageSpan>) -> StageSpan {
    match (left, right) {
        (Some(left), Some(right)) => StageSpan {
            start: left.start.min(right.start),
            end: left.end.max(right.end),
        },
        (Some(span), None) | (None, Some(span)) => span,
        (None, None) => {
            let now = Instant::now();
            StageSpan::new(now, now)
        }
    }
}

pub(super) fn merge_stage_span(
    left: Option<StageSpan>,
    right: Option<StageSpan>,
) -> Option<StageSpan> {
    match (left, right) {
        (Some(left), Some(right)) => Some(StageSpan {
            start: left.start.min(right.start),
            end: left.end.max(right.end),
        }),
        (Some(span), None) | (None, Some(span)) => Some(span),
        (None, None) => None,
    }
}

pub(super) fn merged_spans_seconds(mut spans: Vec<StageSpan>) -> f64 {
    if spans.is_empty() {
        return 0.0;
    }
    spans.sort_by_key(|span| span.start);
    let mut total = 0.0;
    let mut current = spans[0];
    for span in spans.into_iter().skip(1) {
        if span.start <= current.end {
            if span.end > current.end {
                current.end = span.end;
            }
        } else {
            total += current.seconds();
            current = span;
        }
    }
    total + current.seconds()
}
