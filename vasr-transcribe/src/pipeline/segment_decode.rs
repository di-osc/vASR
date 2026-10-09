use super::segment_encode::SegmentAsrInference;
use super::segment_prefill::{SegmentPrefillMessage, SegmentPreparedEvent};
use super::vad::offset_annotations;
use super::{
    AsrOptions, AsrSegmentJob, AsrStageStats, PendingGlobalRequest, PipelineRequestId,
    RequestItemRef, RoutedTranscribeItemOutcome, StageSpan, TranscribeItemOutcome,
    TranscribeItemSpans, TranscribeItemTiming, allocated_batch_seconds, allocation_scale,
    merge_optional_span, outcome_from_result,
    prefill_span_from_decode, record_asr_batch_stats,
};
use anyhow::{Result, bail};
use asr_data::{AudioEvent, Timeline, Waveform};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc as std_mpsc};
use std::time::Instant;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use vasr::OfflineAsrModel as AsrModel;
#[cfg(feature = "timing")]
use vasr::models::qwen3asr::TranscribeTimings;
use vasr::models::qwen3asr::{Qwen3ASR, StreamingMetadataDecoder, qwen_outputs_to_timelines};

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentJobState {
    index: usize,
    request_item: Option<RequestItemRef>,
    channel: &'static str,
    timing: TranscribeItemTiming,
    spans: TranscribeItemSpans,
    audio_seconds: f64,
    speech_annotations: Vec<AudioEvent>,
    segment_annotations: Vec<Option<Vec<AudioEvent>>>,
    segment_count: usize,
    segments_done: usize,
    failure_reason: Option<String>,
}

#[derive(Default)]
#[cfg(feature = "paged-attn")]
pub(super) struct SegmentAsrState {
    pub(super) jobs: HashMap<usize, SegmentJobState>,
}

type SegmentDecoderReceiver = mpsc::UnboundedReceiver<SegmentPrefillMessage>;

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_segment_decoder_worker(
    asr: Arc<dyn AsrModel>,
    asr_options: AsrOptions,
    decoder_rx: SegmentDecoderReceiver,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    failed: Option<Arc<AtomicBool>>,
    fatal_error_tx: Option<watch::Sender<Option<String>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut segment_state = SegmentAsrState::default();
        let result = tokio::task::spawn_blocking({
            let asr = Arc::clone(&asr);
            let asr_options = asr_options.clone();
            let stats = Arc::clone(&stats);
            let asr_to_result_tx = asr_to_result_tx.clone();
            move || {
                run_global_segment_decode_worker(
                    asr.as_ref(),
                    asr_options,
                    decoder_rx,
                    &mut segment_state,
                    asr_to_result_tx,
                    stats,
                )
            }
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR global decoder join error: {error}")));
        if let Err(error) = result {
            let error = format!("ASR global decoder failed: {error:#}");
            tracing::error!(
                target: "vasr_transcribe::pipeline",
                "{error}"
            );
            if let Some(failed) = failed {
                failed.store(true, Ordering::Release);
            }
            fail_pending_requests(&pending_requests, &error);
            if let Some(fatal_error_tx) = fatal_error_tx {
                fatal_error_tx.send_replace(Some(error));
            }
        }
    })
}

#[cfg(feature = "paged-attn")]
fn fail_pending_requests(
    pending_requests: &Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    error: &str,
) {
    let Some(pending_requests) = pending_requests else {
        return;
    };
    let requests = match pending_requests.lock() {
        Ok(mut pending) => std::mem::take(&mut *pending)
            .into_values()
            .collect::<Vec<_>>(),
        Err(_) => {
            tracing::error!(
                target: "vasr_transcribe::pipeline",
                "failed to notify pending requests after decoder failure: pending request lock poisoned"
            );
            return;
        }
    };
    for request in requests { request.fail(error); }
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
) -> Result<()> {
    let qwen3 = asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("dynamic staged ASR decode requires Qwen3ASR"))?;
    let runtime_options = vasr::models::qwen3asr::qwen_transcribe_options(&asr_options);
    let (prefill_tx, prefill_rx) = std_mpsc::channel();
    let (meta_tx, meta_rx) = std_mpsc::channel();
    let mut stream_batches = StreamBatchRegistry::default();
    let pending_chunks = RefCell::new(BTreeMap::new());
    let debug_metadata = std::env::var_os("VASR_DEBUG_REQUEST_MEM").is_some();
    let mut last_metadata_snapshot = None;

    let forward_thread = std::thread::spawn(move || {
        let mut next_global_chunk = 0usize;
        let mut producer_chunk_ranges: HashMap<usize, Vec<(usize, usize, usize, usize)>> =
            HashMap::new();
        let mut producer_chunk_counts: HashMap<usize, usize> = HashMap::new();
        while let Some(message) = decoder_rx.blocking_recv() {
            match message {
                SegmentPrefillMessage::Prefill {
                    producer_id,
                    result,
                } => {
                    let result = result.and_then(|mut output| {
                        let ranges = producer_chunk_ranges.get_mut(&producer_id);
                        if !output.sequences.is_empty() && ranges.is_none() {
                            return Err(anyhow::anyhow!(
                                "dynamic staged ASR prefill has no chunk metadata for producer={producer_id} sequences={}",
                                output.sequences.len()
                            ));
                        }
                        if let Some(ranges) = ranges {
                            for seq in &mut output.sequences {
                                let local = seq.client_index;
                                let Some((local_start, global_start, _len, remaining)) =
                                    ranges.iter_mut().find(|(local_start, _, len, _)| {
                                        local >= *local_start
                                            && local < local_start.saturating_add(*len)
                                    })
                                else {
                                    return Err(anyhow::anyhow!(
                                        "dynamic staged ASR producer chunk index out of range: producer={producer_id} local_chunk={local}"
                                    ));
                                };
                                seq.client_index =
                                    global_start.saturating_add(local - *local_start);
                                *remaining = remaining.checked_sub(1).ok_or_else(|| {
                                    anyhow::anyhow!("prefill metadata range emitted too many sequences")
                                })?;
                            }
                            ranges.retain(|(_, _, _, remaining)| *remaining > 0);
                        }
                        output.total_rows = next_global_chunk;
                        Ok(output)
                    });
                    if prefill_tx.send(result).is_err() {
                        break;
                    }
                }
                SegmentPrefillMessage::Prepared {
                    producer_id,
                    result,
                } => {
                    // Producers are created serially; seeing a newer producer
                    // means all messages from older producers have been sent.
                    producer_chunk_ranges.retain(|&id, _| id >= producer_id);
                    producer_chunk_counts.retain(|&id, _| id >= producer_id);
                    if let Ok(SegmentPreparedEvent::Prepared(prepared)) = &result {
                        let chunk_count = prepared.meta.chunk_len();
                        let local_start = producer_chunk_counts.entry(producer_id).or_default();
                        producer_chunk_ranges.entry(producer_id).or_default().push((
                            *local_start,
                            next_global_chunk,
                            chunk_count,
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
            if debug_metadata {
                tracing::info!(
                    target: "vasr_transcribe::pipeline",
                    "metadata_lifetime | stage=producer producers={} ranges={} waiting_chunks={}",
                    producer_chunk_counts.len(),
                    producer_chunk_ranges.values().map(Vec::len).sum::<usize>(),
                    producer_chunk_ranges.values().flatten().map(|range| range.3).sum::<usize>(),
                );
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
            &mut stream_batches,
            segment_state,
            &asr_to_result_tx,
            &stats,
        )?;
        let completed = {
            let mut pending_chunks = pending_chunks.borrow_mut();
            drain_completed_chunks(
                qwen3,
                &mut pending_chunks,
                &mut stream_batches,
                segment_state,
            )?
        };
        // 每个条目独立完成、返回和回收；不再等到全局 decode 空闲再统一返回，
        // 否则持续来流会让短请求的结果一直排在其他请求后面。
        for outcome in completed {
            if asr_to_result_tx.blocking_send(outcome).is_err() {
                bail!("ASR result router closed");
            }
        }
        stream_batches.retire_completed(|batch| batch.decoder.all_chunks_completed());
        if debug_metadata {
            let snapshot = (
                stream_batches.batches.len(),
                segment_state.jobs.len(),
                pending_chunks.borrow().len(),
            );
            if last_metadata_snapshot.replace(snapshot) != Some(snapshot) {
                tracing::info!(
                    target: "vasr_transcribe::pipeline",
                    "metadata_lifetime | stage=decoder batches={} jobs={} pending_chunks={}",
                    snapshot.0, snapshot.1, snapshot.2,
                );
            }
        }
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
    // 显式结束 `on_tick` 的可变借用以释放对回调上下文的借用；闭包不实现 `Drop`，用 `let _` 移动即可。
    let _ = on_tick;
    if let Err(error) = decode_result {
        let _ = forward_thread.join();
        return Err(error);
    }
    let _ = forward_thread.join();
    drain_global_decoder_meta(
        &meta_rx,
        &mut stream_batches,
        segment_state,
        &asr_to_result_tx,
        &stats,
    )?;
    let completed = drain_completed_chunks(
        qwen3,
        &mut pending_chunks.borrow_mut(),
        &mut stream_batches,
        segment_state,
    )?;
    // 解码结束时同样逐条返回，保持与 tick 路径一致的生命周期。
    for outcome in completed {
        if asr_to_result_tx.blocking_send(outcome).is_err() {
            bail!("ASR result router closed");
        }
    }
    stream_batches.retire_completed(|batch| batch.decoder.all_chunks_completed());

    if !pending_chunks.borrow().is_empty() {
        bail!(
            "dynamic ASR streaming metadata missing for {} completed chunks",
            pending_chunks.borrow().len()
        );
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
struct StreamDecodeBatch {
    decoder: StreamingMetadataDecoder,
    segments: Vec<AsrSegmentJob>,
    waveforms: Vec<Waveform>,
    batch_lens: Vec<usize>,
    stage_starts: Vec<Instant>,
}

/// Global IDs never change, while completed batches can be dropped independently.
#[cfg(feature = "paged-attn")]
struct StreamBatchRegistry<T> {
    batches: BTreeMap<usize, (usize, T)>,
    next_chunk: usize,
}

#[cfg(feature = "paged-attn")]
impl<T> Default for StreamBatchRegistry<T> {
    fn default() -> Self {
        Self {
            batches: BTreeMap::new(),
            next_chunk: 0,
        }
    }
}

#[cfg(feature = "paged-attn")]
impl<T> StreamBatchRegistry<T> {
    fn push(&mut self, chunks: usize, batch: T) -> Result<()> {
        if chunks == 0 {
            bail!("streaming metadata batch contains no chunks");
        }
        let end = self
            .next_chunk
            .checked_add(chunks)
            .ok_or_else(|| anyhow::anyhow!("streaming global chunk index overflow"))?;
        self.batches.insert(self.next_chunk, (end, batch));
        self.next_chunk = end;
        Ok(())
    }

    fn get_mut(&mut self, global_chunk: usize) -> Result<(&mut T, usize)> {
        if let Some((&start, (end, batch))) = self.batches.range_mut(..=global_chunk).next_back()
            && global_chunk < *end
        {
            return Ok((batch, global_chunk - start));
        }
        bail!(
            "streaming completion has no active metadata: chunk={global_chunk} next={}",
            self.next_chunk
        )
    }

    fn retire_completed(&mut self, complete: impl Fn(&T) -> bool) {
        self.batches.retain(|_, (_, batch)| !complete(batch));
    }
}

#[cfg(feature = "paged-attn")]
fn handle_global_decoder_message(
    message: SegmentPrefillMessage,
    batches: &mut StreamBatchRegistry<StreamDecodeBatch>,
    state: &mut SegmentAsrState,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) -> Result<()> {
    match message {
        SegmentPrefillMessage::Prefill { .. } => Ok(()),
        SegmentPrefillMessage::Prepared {
            result: Ok(SegmentPreparedEvent::Prepared(prepared)),
            ..
        } => {
            let chunk_count = prepared.meta.chunk_len();
            let segment_count = prepared.segments.len();
            let audio_count = prepared.waveforms.len();
            prepare_segment_job_state(state, prepared.segments.as_slice());
            batches.push(
                chunk_count,
                StreamDecodeBatch {
                    decoder: StreamingMetadataDecoder::new(vec![prepared.meta], vec![audio_count]),
                    segments: prepared.segments,
                    waveforms: prepared.waveforms,
                    batch_lens: vec![audio_count],
                    stage_starts: vec![prepared.stage_start; segment_count],
                },
            )
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
                if asr_to_result_tx.blocking_send(outcome).is_err() {
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
    batches: &mut StreamBatchRegistry<StreamDecodeBatch>,
    state: &mut SegmentAsrState,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) -> Result<()> {
    for message in meta_rx.try_iter() {
        handle_global_decoder_message(message, batches, state, asr_to_result_tx, stats)?;
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
fn drain_completed_chunks(
    qwen3: &Qwen3ASR,
    pending_chunks: &mut BTreeMap<usize, Vec<u32>>,
    batches: &mut StreamBatchRegistry<StreamDecodeBatch>,
    state: &mut SegmentAsrState,
) -> Result<Vec<RoutedTranscribeItemOutcome>> {
    // A completion can arrive before its Prepared message is drained. Keep
    // future global IDs pending until metadata registers their fixed range.
    let ready = pending_chunks
        .range(..batches.next_chunk)
        .map(|(&chunk, _)| chunk)
        .collect::<Vec<_>>();
    let mut completed = Vec::new();
    for chunk in ready {
        let (batch, local_chunk) = batches.get_mut(chunk)?;
        let ids = pending_chunks
            .remove(&chunk)
            .ok_or_else(|| anyhow::anyhow!("missing pending streaming completion"))?;
        if let Some(outcomes) = accept_completed_chunk(
            qwen3,
            &mut batch.decoder,
            state,
            &mut batch.segments,
            &mut batch.waveforms,
            &batch.batch_lens,
            &batch.stage_starts,
            local_chunk,
            ids,
        )? {
            completed.extend(outcomes);
        }
    }
    Ok(completed)
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
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
    }
    if let Some(waveform) = waveforms.get_mut(segment_index) {
        waveform.samples = Vec::new();
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
                failure_reason: None,
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
    let duration = (job.audio_seconds * 1000.0).round() as usize;
    let mut timeline = Timeline::new("offline_audio", duration);
    if let Err(err) = vasr::extend_predictions(&mut timeline, job.speech_annotations) {
        return TranscribeItemOutcome {
            index: job.index,
            channel: job.channel,
            result: Err(err),
            audio_seconds: job.audio_seconds,
            timing: job.timing,
            spans: job.spans,
            bad_component: Some("recognizer"),
        };
    }
    for annotations in job.segment_annotations.into_iter().flatten() {
        if let Err(err) = vasr::extend_predictions(&mut timeline, annotations) {
            return TranscribeItemOutcome {
                index: job.index,
                channel: job.channel,
                result: Err(err),
                audio_seconds: job.audio_seconds,
                timing: job.timing,
                spans: job.spans,
                bad_component: Some("recognizer"),
            };
        }
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
    let outcome = match job.failure_reason.clone() {
        Some(reason) => outcome_from_result(
            job.index,
            job.channel,
            job.audio_seconds,
            job.timing,
            job.spans,
            Err(anyhow::anyhow!(reason)),
        ),
        None => finish_segment_job(job),
    };
    match request_item {
        Some(request_item) => RoutedTranscribeItemOutcome::for_request(request_item, outcome),
        None => outcome.into(),
    }
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn fail_segment_asr_batch(
    state: &mut SegmentAsrState,
    segments: &[AsrSegmentJob],
    reason: &str,
    asr_span: StageSpan,
    prefill_span: Option<StageSpan>,
    asr_timing: Option<TranscribeItemTiming>,
    elapsed: f64,
    total_audio_seconds: f64,
    active_job_count: usize,
) -> Vec<RoutedTranscribeItemOutcome> {
    let affected: HashSet<usize> = segments.iter().map(|segment| segment.job_index).collect();
    let mut finished_jobs = Vec::new();
    for job_index in affected {
        let job_segments = segments
            .iter()
            .filter(|segment| segment.job_index == job_index)
            .collect::<Vec<_>>();
        if let Some(job) = state.jobs.get_mut(&job_index) {
            job.failure_reason.get_or_insert_with(|| reason.to_string());
            job.spans.asr = Some(merge_optional_span(job.spans.asr, Some(asr_span)));
            job.spans.prefill = Some(merge_optional_span(job.spans.prefill, prefill_span));
            job.spans.decode = Some(merge_optional_span(job.spans.decode, Some(asr_span)));
            let item_audio_seconds = job_segments
                .iter()
                .map(|segment| segment.waveform.duration_seconds())
                .sum::<f64>();
            job.timing.asr_seconds += allocated_batch_seconds(
                item_audio_seconds,
                total_audio_seconds,
                active_job_count,
                elapsed,
            );
            if let Some(asr_timing) = asr_timing {
                job.timing.add_scaled(
                    asr_timing,
                    allocation_scale(item_audio_seconds, total_audio_seconds, active_job_count),
                );
            }
            for segment in job_segments {
                if let Some(done) = job.segment_annotations.get_mut(segment.segment_index)
                    && done.is_none()
                {
                    *done = Some(Vec::new());
                    job.segments_done = job.segments_done.saturating_add(1);
                }
            }
            if job.segments_done >= job.segment_count {
                finished_jobs.push(job_index);
            }
        }
    }

    finished_jobs
        .into_iter()
        .filter_map(|job_index| state.jobs.remove(&job_index))
        .map(routed_segment_job_outcome)
        .collect()
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
                            Some(offset_annotations(asr_timeline.prediction.clone(), offset));
                        if job.segments_done >= job.segment_count
                            && let Some(job) = state.jobs.remove(&job_index)
                        {
                            completed.push(routed_segment_job_outcome(job));
                        }
                    }
                }
            } else {
                let err = format!(
                    "ASR segment batch returned {} timelines for {} segments",
                    timelines.len(),
                    segments.len()
                );
                completed.extend(fail_segment_asr_batch(
                    state,
                    segments,
                    err.as_str(),
                    asr_span,
                    prefill_span,
                    asr_timing,
                    elapsed,
                    inference.segment_audio_seconds,
                    inference.active_job_count,
                ));
            }
        }
        Err(error) => {
            let elapsed = stage_start.elapsed().as_secs_f64();
            let asr_span = StageSpan::new(stage_start, Instant::now());
            let prefill_span = prefill_span_from_decode(asr_span, asr_timing);
            let total_audio_seconds: f64 = segments
                .iter()
                .map(|segment| segment.waveform.duration_seconds())
                .sum();
            completed.extend(fail_segment_asr_batch(
                state,
                segments,
                &format!("{error:#}"),
                asr_span,
                prefill_span,
                asr_timing,
                elapsed,
                total_audio_seconds,
                1,
            ));
        }
    }

    completed
}

#[cfg(all(test, feature = "paged-attn"))]
mod stream_batch_lifecycle_tests {
    use super::StreamBatchRegistry;

    #[test]
    fn long_request_does_not_pin_completed_short_requests_or_reuse_global_ids() -> anyhow::Result<()>
    {
        let mut batches = StreamBatchRegistry::default();
        batches.push(2, vec![false, false])?;
        // Complete out of order while one old request keeps the stream active.
        batches.get_mut(1)?.0[1] = true;
        for _ in 0..1000 {
            let global = batches.next_chunk;
            batches.push(1, vec![false])?;
            let (batch, local) = batches.get_mut(global)?;
            assert_eq!(local, 0);
            batch[local] = true;
            batches.retire_completed(|batch| batch.iter().all(|&done| done));
            assert_eq!(batches.batches.len(), 1);
            assert!(batches.get_mut(global).is_err());
            assert!(!batches.get_mut(0)?.0[0]);
        }
        batches.get_mut(0)?.0[0] = true;
        batches.retire_completed(|batch| batch.iter().all(|&done| done));
        assert!(batches.batches.is_empty());
        let global = batches.next_chunk;
        batches.push(2, vec![false, false])?;
        assert_eq!(batches.get_mut(global + 1)?.1, 1);
        assert!(batches.get_mut(global - 1).is_err());
        assert!(batches.get_mut(global + 2).is_err());
        Ok(())
    }
}
