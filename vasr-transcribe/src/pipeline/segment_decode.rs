use super::segment_encode::SegmentAsrInference;
use super::segment_prefill::{SegmentPrefillMessage, SegmentPreparedEvent};
use super::vad::offset_annotations;
use super::{
    AsrOptions, AsrSegmentJob, AsrStageStats, PendingGlobalRequest, PipelineRequestId,
    RequestItemRef, RoutedTranscribeItemOutcome, StageSpan, TranscribeItemOutcome,
    TranscribeItemSpans, TranscribeItemTiming, allocated_batch_seconds, allocation_scale,
    merge_optional_span, outcome_from_result, prefill_span_from_decode, record_asr_batch_stats,
    route_outcome,
};
use anyhow::{Result, bail};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vasr::OfflineAsrModel as AsrModel;
#[cfg(feature = "timing")]
use vasr::models::qwen3asr::TranscribeTimings;
use vasr::models::qwen3asr::{Qwen3ASR, StreamingMetadataDecoder, qwen_outputs_to_timelines};
use vasr_data::{Annotation, Timeline, Waveform};

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentJobState {
    index: usize,
    request_item: Option<RequestItemRef>,
    channel: &'static str,
    timing: TranscribeItemTiming,
    spans: TranscribeItemSpans,
    audio_seconds: f64,
    speech_annotations: Vec<Annotation>,
    segment_annotations: Vec<Option<Vec<Annotation>>>,
    segment_count: usize,
    segments_done: usize,
}

#[derive(Default)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentAsrState {
    pub(super) jobs: HashMap<usize, SegmentJobState>,
}

type SegmentDecoderReceiver = mpsc::UnboundedReceiver<SegmentPrefillMessage>;

#[cfg(feature = "paged-attn")]
pub(super) fn spawn_segment_decoder_worker(
    asr: Arc<dyn AsrModel>,
    asr_options: AsrOptions,
    decoder_rx: SegmentDecoderReceiver,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut segment_state = SegmentAsrState::default();
        let result = tokio::task::spawn_blocking({
            let asr = Arc::clone(&asr);
            let asr_options = asr_options.clone();
            let stats = Arc::clone(&stats);
            let pending_requests = pending_requests.clone();
            let asr_to_result_tx = asr_to_result_tx.clone();
            move || {
                run_global_segment_decode_worker(
                    asr.as_ref(),
                    asr_options,
                    decoder_rx,
                    &mut segment_state,
                    asr_to_result_tx,
                    stats,
                    pending_requests,
                )
            }
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR global decoder join error: {error}")));
        if let Err(error) = result {
            tracing::error!(
                target: "vasr_transcribe::pipeline",
                "ASR global decoder failed: {error:#}"
            );
        }
    })
}
#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn run_global_segment_decode_worker(
    asr: &dyn AsrModel,
    asr_options: AsrOptions,
    mut decoder_rx: SegmentDecoderReceiver,
    segment_state: &mut SegmentAsrState,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    _pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
) -> Result<()> {
    let qwen3 = asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("dynamic staged ASR decode requires Qwen3ASR"))?;
    let runtime_options = vasr::models::qwen3asr::qwen_transcribe_options(&asr_options);
    let (prefill_tx, prefill_rx) = std_mpsc::channel();
    let (meta_tx, meta_rx) = std_mpsc::channel();
    let mut stream_decoder = StreamingMetadataDecoder::new(Vec::new(), Vec::new());
    let mut stream_segments = Vec::new();
    let mut stream_waveforms = Vec::new();
    let mut stream_batch_lens = Vec::new();
    let mut stream_stage_starts = Vec::new();
    let mut stream_chunk_base = 0usize;
    let pending_chunks = RefCell::new(BTreeMap::new());

    let forward_thread = std::thread::spawn(move || {
        let mut next_global_chunk = 0usize;
        let mut producer_chunk_ranges: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
        let mut producer_chunk_counts: HashMap<usize, usize> = HashMap::new();
        while let Some(message) = decoder_rx.blocking_recv() {
            match message {
                SegmentPrefillMessage::Prefill {
                    producer_id,
                    result,
                } => {
                    let result = result.map(|mut output| {
                        if let Some(ranges) = producer_chunk_ranges.get(&producer_id) {
                            for seq in &mut output.sequences {
                                let local = seq.client_index;
                                if let Some((local_start, global_start, _len)) =
                                    ranges.iter().find(|(local_start, _, len)| {
                                        local >= *local_start
                                            && local < local_start.saturating_add(*len)
                                    })
                                {
                                    seq.client_index =
                                        global_start.saturating_add(local - local_start);
                                }
                            }
                            output.total_rows = next_global_chunk;
                        }
                        output
                    });
                    if prefill_tx.send(result).is_err() {
                        break;
                    }
                }
                SegmentPrefillMessage::Prepared {
                    producer_id,
                    result,
                } => {
                    if let Ok(SegmentPreparedEvent::Prepared(prepared)) = &result {
                        let chunk_count = prepared.meta.chunk_len();
                        let local_start = producer_chunk_counts.entry(producer_id).or_default();
                        producer_chunk_ranges.entry(producer_id).or_default().push((
                            *local_start,
                            next_global_chunk,
                            chunk_count,
                        ));
                        *local_start = local_start.saturating_add(chunk_count);
                        next_global_chunk = next_global_chunk.saturating_add(chunk_count);
                    }
                    if meta_tx
                        .send(SegmentPrefillMessage::Prepared {
                            producer_id,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                SegmentPrefillMessage::Failure(failure) => {
                    if meta_tx
                        .send(SegmentPrefillMessage::Failure(failure))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });

    let mut on_complete = |chunk_index: usize, ids: Vec<u32>| -> Result<()> {
        pending_chunks.borrow_mut().insert(chunk_index, ids);
        Ok(())
    };
    let mut on_tick = || -> Result<()> {
        drain_global_decoder_meta(
            &meta_rx,
            &mut stream_decoder,
            segment_state,
            &mut stream_segments,
            &mut stream_waveforms,
            &mut stream_batch_lens,
            &mut stream_stage_starts,
            &asr_to_result_tx,
            &stats,
        )?;
        {
            let mut pending_chunks = pending_chunks.borrow_mut();
            drain_completed_chunks(
                qwen3,
                &mut pending_chunks,
                &mut stream_decoder,
                segment_state,
                stream_segments.as_mut_slice(),
                stream_waveforms.as_mut_slice(),
                stream_batch_lens.as_slice(),
                stream_stage_starts.as_slice(),
                stream_chunk_base,
                &asr_to_result_tx,
            )?;
        }
        reset_stream_decode_state_if_idle(
            &pending_chunks.borrow(),
            &mut stream_decoder,
            segment_state,
            &mut stream_segments,
            &mut stream_waveforms,
            &mut stream_batch_lens,
            &mut stream_stage_starts,
            &mut stream_chunk_base,
        );
        Ok(())
    };

    #[cfg(feature = "timing")]
    let mut timings = TranscribeTimings::default();
    let decode_result = qwen3.decode_global_prefilled_dynamic_ids_streaming(
        prefill_rx,
        runtime_options,
        &mut on_complete,
        &mut on_tick,
        #[cfg(feature = "timing")]
        Some(&mut timings.generation),
    );
    drop(on_tick);
    let _ = forward_thread.join();
    drain_global_decoder_meta(
        &meta_rx,
        &mut stream_decoder,
        segment_state,
        &mut stream_segments,
        &mut stream_waveforms,
        &mut stream_batch_lens,
        &mut stream_stage_starts,
        &asr_to_result_tx,
        &stats,
    )?;
    drain_completed_chunks(
        qwen3,
        &mut pending_chunks.borrow_mut(),
        &mut stream_decoder,
        segment_state,
        stream_segments.as_mut_slice(),
        stream_waveforms.as_mut_slice(),
        stream_batch_lens.as_slice(),
        stream_stage_starts.as_slice(),
        stream_chunk_base,
        &asr_to_result_tx,
    )?;

    decode_result?;
    if !pending_chunks.borrow().is_empty() {
        bail!(
            "dynamic ASR streaming metadata missing for {} completed chunks",
            pending_chunks.borrow().len()
        );
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
fn handle_global_decoder_message(
    message: SegmentPrefillMessage,
    decoder: &mut StreamingMetadataDecoder,
    state: &mut SegmentAsrState,
    segments: &mut Vec<AsrSegmentJob>,
    waveforms: &mut Vec<Waveform>,
    batch_lens: &mut Vec<usize>,
    stage_starts: &mut Vec<Instant>,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) -> Result<()> {
    match message {
        SegmentPrefillMessage::Prefill { .. } => Ok(()),
        SegmentPrefillMessage::Prepared {
            result: Ok(SegmentPreparedEvent::Prepared(prepared)),
            ..
        } => {
            let segment_count = prepared.segments.len();
            decoder.push_metadata(prepared.meta, prepared.waveforms.len());
            prepare_segment_job_state(state, prepared.segments.as_slice());
            batch_lens.push(prepared.waveforms.len());
            waveforms.extend(prepared.waveforms);
            segments.extend(prepared.segments);
            stage_starts.extend(std::iter::repeat_n(prepared.stage_start, segment_count));
            Ok(())
        }
        SegmentPrefillMessage::Prepared {
            result: Err(failure),
            ..
        }
        | SegmentPrefillMessage::Failure(failure) => {
            let outcomes = finish_segment_asr_batch(
                state,
                failure.segments.as_slice(),
                failure.stage_start,
                Err(failure.error),
                None,
                Some(stats),
            );
            for outcome in outcomes {
                if asr_to_result_tx.blocking_send(outcome.into()).is_err() {
                    bail!("ASR result router closed");
                }
            }
            Ok(())
        }
    }
}

#[cfg(feature = "paged-attn")]
fn drain_global_decoder_meta(
    meta_rx: &std_mpsc::Receiver<SegmentPrefillMessage>,
    decoder: &mut StreamingMetadataDecoder,
    state: &mut SegmentAsrState,
    segments: &mut Vec<AsrSegmentJob>,
    waveforms: &mut Vec<Waveform>,
    batch_lens: &mut Vec<usize>,
    stage_starts: &mut Vec<Instant>,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) -> Result<()> {
    for message in meta_rx.try_iter() {
        handle_global_decoder_message(
            message,
            decoder,
            state,
            segments,
            waveforms,
            batch_lens,
            stage_starts,
            asr_to_result_tx,
            stats,
        )?;
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
fn drain_completed_chunks(
    qwen3: &Qwen3ASR,
    pending_chunks: &mut BTreeMap<usize, Vec<u32>>,
    decoder: &mut StreamingMetadataDecoder,
    state: &mut SegmentAsrState,
    segments: &mut [AsrSegmentJob],
    waveforms: &mut [Waveform],
    batch_lens: &[usize],
    stage_starts: &[Instant],
    chunk_base: usize,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
) -> Result<()> {
    let chunk_end = chunk_base.saturating_add(decoder.chunk_count());
    let ready: Vec<usize> = pending_chunks
        .range(..chunk_end)
        .map(|(chunk_index, _)| *chunk_index)
        .collect();
    for chunk_index in ready {
        let Some(ids) = pending_chunks.remove(&chunk_index) else {
            continue;
        };
        let local_chunk_index = chunk_index.saturating_sub(chunk_base);
        if let Some(outcomes) = accept_completed_chunk(
            qwen3,
            decoder,
            state,
            segments,
            waveforms,
            batch_lens,
            stage_starts,
            local_chunk_index,
            ids,
        )? {
            for outcome in outcomes {
                if asr_to_result_tx.blocking_send(outcome.into()).is_err() {
                    bail!("ASR result router closed");
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn reset_stream_decode_state_if_idle(
    pending_chunks: &BTreeMap<usize, Vec<u32>>,
    decoder: &mut StreamingMetadataDecoder,
    state: &SegmentAsrState,
    segments: &mut Vec<AsrSegmentJob>,
    waveforms: &mut Vec<Waveform>,
    batch_lens: &mut Vec<usize>,
    stage_starts: &mut Vec<Instant>,
    chunk_base: &mut usize,
) {
    if !pending_chunks.is_empty() || !state.jobs.is_empty() || decoder.chunk_count() == 0 {
        return;
    }
    *chunk_base = chunk_base.saturating_add(decoder.chunk_count());
    *decoder = StreamingMetadataDecoder::new(Vec::new(), Vec::new());
    segments.clear();
    waveforms.clear();
    batch_lens.clear();
    stage_starts.clear();
}

#[cfg(feature = "paged-attn")]
fn accept_completed_chunk(
    qwen3: &Qwen3ASR,
    decoder: &mut StreamingMetadataDecoder,
    state: &mut SegmentAsrState,
    segments: &mut [AsrSegmentJob],
    waveforms: &mut [Waveform],
    batch_lens: &[usize],
    stage_starts: &[Instant],
    chunk_index: usize,
    ids: Vec<u32>,
) -> Result<Option<Vec<RoutedTranscribeItemOutcome>>> {
    let Some((batch_index, audio_index, output)) =
        decoder.accept_chunk(qwen3.processor(), chunk_index, ids)?
    else {
        return Ok(None);
    };
    let segment_index = batch_lens
        .iter()
        .take(batch_index)
        .sum::<usize>()
        .saturating_add(audio_index);
    let Some(segment) = segments.get(segment_index) else {
        bail!(
            "dynamic ASR streaming segment index out of range: batch={} audio={} segment_index={} segments={}",
            batch_index,
            audio_index,
            segment_index,
            segments.len()
        );
    };
    let stage_start = stage_starts
        .get(segment_index)
        .copied()
        .unwrap_or_else(Instant::now);
    let mut timelines =
        qwen_outputs_to_timelines(vec![output], &waveforms[segment_index..segment_index + 1])?;
    let timeline = timelines
        .pop()
        .ok_or_else(|| anyhow::anyhow!("missing streaming ASR timeline"))?;
    let batch = SegmentAsrInference {
        targets: vec![(segment.job_index, segment.segment_index, segment.offset)],
        segment_audio_seconds: segment.waveform.duration_seconds(),
        active_job_count: 1,
    };
    let outcomes = finish_segment_asr_batch(
        state,
        std::slice::from_ref(segment),
        stage_start,
        Ok((batch, vec![timeline])),
        None,
        None,
    );
    release_stream_segment_audio(segments, waveforms, segment_index);
    Ok(Some(outcomes))
}

#[cfg(feature = "paged-attn")]
fn release_stream_segment_audio(
    segments: &mut [AsrSegmentJob],
    waveforms: &mut [Waveform],
    segment_index: usize,
) {
    if let Some(segment) = segments.get_mut(segment_index) {
        segment.waveform.samples = Vec::new();
        segment.waveform.encoded_data = None;
    }
    if let Some(waveform) = waveforms.get_mut(segment_index) {
        waveform.samples = Vec::new();
        waveform.encoded_data = None;
    }
}
#[cfg(feature = "paged-attn")]
pub(super) fn prepare_segment_job_state(state: &mut SegmentAsrState, segments: &[AsrSegmentJob]) {
    for segment in segments {
        let entry = state
            .jobs
            .entry(segment.job_index)
            .or_insert_with(|| SegmentJobState {
                index: segment.job_index,
                request_item: segment.request_item,
                channel: segment.channel,
                timing: segment.timing,
                spans: segment.spans,
                audio_seconds: segment.job_audio_seconds,
                speech_annotations: segment.speech_annotations.clone(),
                segment_annotations: vec![None; segment.segment_count],
                segment_count: segment.segment_count,
                segments_done: 0,
            });
        if !segment.speech_annotations.is_empty() {
            entry.speech_annotations = segment.speech_annotations.clone();
        }
        if entry.segment_count != segment.segment_count {
            tracing::warn!(
                target: "vasr_transcribe::pipeline",
                "VAD segment count changed for job={} previous={} new={}",
                segment.job_index,
                entry.segment_count,
                segment.segment_count
            );
            entry.segment_count = segment.segment_count;
            entry
                .segment_annotations
                .resize_with(segment.segment_count, || None);
        }
    }
}

#[cfg(feature = "paged-attn")]
fn finish_segment_job(job: SegmentJobState) -> TranscribeItemOutcome {
    let mut timeline = Timeline::new("offline_audio");
    timeline.annotations.extend(job.speech_annotations);
    for annotations in job.segment_annotations.into_iter().flatten() {
        timeline.annotations.extend(annotations);
    }
    TranscribeItemOutcome {
        index: job.index,
        channel: job.channel,
        result: Ok(timeline),
        audio_seconds: job.audio_seconds,
        timing: job.timing,
        spans: job.spans,
        bad_component: None,
    }
}

#[cfg(feature = "paged-attn")]
fn routed_segment_job_outcome(job: SegmentJobState) -> RoutedTranscribeItemOutcome {
    let request_item = job.request_item;
    let outcome = finish_segment_job(job);
    match request_item {
        Some(request_item) => RoutedTranscribeItemOutcome::for_request(request_item, outcome),
        None => outcome.into(),
    }
}
#[cfg(feature = "paged-attn")]
pub(super) fn finish_segment_asr_batch(
    state: &mut SegmentAsrState,
    segments: &[AsrSegmentJob],
    stage_start: Instant,
    batch: Result<(SegmentAsrInference, Vec<Timeline>)>,
    asr_timing: Option<TranscribeItemTiming>,
    stats: Option<&Arc<Mutex<AsrStageStats>>>,
) -> Vec<RoutedTranscribeItemOutcome> {
    if segments.is_empty() {
        return Vec::new();
    }
    prepare_segment_job_state(state, segments);

    let mut completed = Vec::new();
    match batch {
        Ok((inference, timelines)) => {
            let elapsed = stage_start.elapsed().as_secs_f64();
            let asr_span = StageSpan::new(stage_start, Instant::now());
            let prefill_span = prefill_span_from_decode(asr_span, asr_timing);
            if let Some(stats) = stats {
                record_asr_batch_stats(
                    stats,
                    inference.active_job_count,
                    segments.len(),
                    inference.segment_audio_seconds,
                    elapsed,
                );
            }
            if timelines.len() == inference.targets.len() {
                let target_count = inference.targets.len();
                for ((job_index, segment_index, offset), asr_timeline) in
                    inference.targets.into_iter().zip(timelines)
                {
                    if let Some(job) = state.jobs.get_mut(&job_index) {
                        if segment_index >= job.segment_annotations.len() {
                            tracing::warn!(
                                target: "vasr_transcribe::pipeline",
                                "VAD segment index out of range for job={} segment_index={} segment_count={}",
                                job_index,
                                segment_index,
                                job.segment_annotations.len()
                            );
                            continue;
                        }
                        if job.segment_annotations[segment_index].is_none() {
                            job.segments_done = job.segments_done.saturating_add(1);
                        }
                        let segment_audio_seconds = segments
                            .iter()
                            .find(|segment| {
                                segment.job_index == job_index
                                    && segment.segment_index == segment_index
                            })
                            .map(|segment| segment.waveform.duration_seconds())
                            .unwrap_or(0.0);
                        job.timing.asr_seconds += allocated_batch_seconds(
                            segment_audio_seconds,
                            inference.segment_audio_seconds,
                            target_count,
                            elapsed,
                        );
                        if let Some(asr_timing) = asr_timing {
                            let scale = allocation_scale(
                                segment_audio_seconds,
                                inference.segment_audio_seconds,
                                target_count,
                            );
                            job.timing.add_scaled(asr_timing, scale);
                        }
                        job.spans.asr = Some(merge_optional_span(job.spans.asr, Some(asr_span)));
                        job.spans.prefill =
                            Some(merge_optional_span(job.spans.prefill, prefill_span));
                        job.spans.decode =
                            Some(merge_optional_span(job.spans.decode, Some(asr_span)));
                        job.segment_annotations[segment_index] =
                            Some(offset_annotations(asr_timeline.annotations, offset));
                        if job.segments_done >= job.segment_count {
                            if let Some(job) = state.jobs.remove(&job_index) {
                                completed.push(routed_segment_job_outcome(job));
                            }
                        }
                    }
                }
            } else {
                let err = format!(
                    "ASR segment batch returned {} timelines for {} segments",
                    timelines.len(),
                    segments.len()
                );
                let affected: HashSet<usize> =
                    segments.iter().map(|segment| segment.job_index).collect();
                let affected_count = affected.len();
                for job_index in affected {
                    if let Some(job) = state.jobs.remove(&job_index) {
                        let request_item = job.request_item;
                        let mut timing = job.timing;
                        timing.asr_seconds += allocated_batch_seconds(
                            segments
                                .iter()
                                .filter(|segment| segment.job_index == job_index)
                                .map(|segment| segment.waveform.duration_seconds())
                                .sum(),
                            inference.segment_audio_seconds,
                            affected_count,
                            elapsed,
                        );
                        if let Some(asr_timing) = asr_timing {
                            let item_audio_seconds = segments
                                .iter()
                                .filter(|segment| segment.job_index == job_index)
                                .map(|segment| segment.waveform.duration_seconds())
                                .sum();
                            let scale = allocation_scale(
                                item_audio_seconds,
                                inference.segment_audio_seconds,
                                affected_count,
                            );
                            timing.add_scaled(asr_timing, scale);
                        }
                        let mut spans = job.spans;
                        spans.asr = Some(merge_optional_span(spans.asr, Some(asr_span)));
                        spans.prefill = Some(merge_optional_span(spans.prefill, prefill_span));
                        spans.decode = Some(merge_optional_span(spans.decode, Some(asr_span)));
                        let outcome = outcome_from_result(
                            job.index,
                            job.channel,
                            job.audio_seconds,
                            timing,
                            spans,
                            Err(anyhow::anyhow!(err.clone())),
                        );
                        completed.push(route_outcome(request_item, outcome));
                    }
                }
            }
        }
        Err(error) => {
            let elapsed = stage_start.elapsed().as_secs_f64();
            let asr_span = StageSpan::new(stage_start, Instant::now());
            let prefill_span = prefill_span_from_decode(asr_span, asr_timing);
            let reason = error.to_string();
            let affected: HashSet<usize> =
                segments.iter().map(|segment| segment.job_index).collect();
            let total_audio_seconds: f64 = segments
                .iter()
                .map(|segment| segment.waveform.duration_seconds())
                .sum();
            for job_index in affected {
                if let Some(job) = state.jobs.remove(&job_index) {
                    let request_item = job.request_item;
                    let mut timing = job.timing;
                    let mut spans = job.spans;
                    spans.asr = Some(merge_optional_span(spans.asr, Some(asr_span)));
                    spans.prefill = Some(merge_optional_span(spans.prefill, prefill_span));
                    spans.decode = Some(merge_optional_span(spans.decode, Some(asr_span)));
                    timing.asr_seconds += allocated_batch_seconds(
                        segments
                            .iter()
                            .filter(|segment| segment.job_index == job_index)
                            .map(|segment| segment.waveform.duration_seconds())
                            .sum(),
                        total_audio_seconds,
                        1,
                        elapsed,
                    );
                    let outcome = outcome_from_result(
                        job.index,
                        job.channel,
                        job.audio_seconds,
                        timing,
                        spans,
                        Err(anyhow::anyhow!(reason.clone())),
                    );
                    completed.push(route_outcome(request_item, outcome));
                }
            }
        }
    }

    completed
}
