use super::segment_encode::{SegmentEncodedAsrBatch, SegmentEncodedMessage, SegmentEncoderOutput};
use super::{
    AsrSegmentJob, RequestItemRef, apply_batch_timing_to_segments, asr_prefill_coalesce_wait,
    format_seconds, format_speed,
};
use anyhow::{Result, bail};
use asr_data::Waveform;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vasr::models::qwen3asr::{
    EncodedAsrBatchMetadata, PrefillStageOutput, PreparedAudioBatch, PreparedAudioBatchPoll,
    Qwen3ASR,
};
use vasr::{AsrOptions, OfflineAsrModel as AsrModel};

#[cfg(feature = "paged-attn")]
#[derive(Debug, Clone, Copy)]
pub(super) struct SegmentLogItem {
    request_item: Option<RequestItemRef>,
    audio_seconds: f64,
}

#[cfg(feature = "paged-attn")]
fn segment_log_items(segments: &[AsrSegmentJob]) -> Vec<SegmentLogItem> {
    segments
        .iter()
        .map(|segment| SegmentLogItem {
            request_item: segment.request_item,
            audio_seconds: segment.waveform.duration_seconds(),
        })
        .collect()
}

#[cfg(feature = "paged-attn")]
fn log_prefill_batch_done(
    all_segments: &[SegmentLogItem],
    output: &PrefillStageOutput,
    elapsed_seconds: f64,
) {
    let mut segments = Vec::new();
    for sequence in &output.sequences {
        if let Some(&segment) = all_segments.get(sequence.client_index) {
            segments.push(segment);
        }
    }
    if segments.is_empty() {
        return;
    }
    let audio_seconds = segments
        .iter()
        .map(|segment| segment.audio_seconds)
        .sum::<f64>();
    let request_count = segments
        .iter()
        .filter_map(|segment| segment.request_item.map(|item| item.request_id))
        .collect::<HashSet<_>>()
        .len();
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "prefill batch DONE | reqs={} seg={} audio={} elapsed={} speed={} reason={}",
        request_count,
        segments.len(),
        format_seconds(audio_seconds),
        format_seconds(elapsed_seconds),
        format_speed(audio_seconds, elapsed_seconds),
        output.reason.as_str(),
    );
}
#[cfg(feature = "paged-attn")]
pub(super) enum SegmentPrefillMessage {
    Prefill {
        producer_id: usize,
        result: Result<PrefillStageOutput>,
    },
    Prepared {
        producer_id: usize,
        result: Result<SegmentPreparedEvent, SegmentEncodedFailure>,
    },
    Failure(SegmentEncodedFailure),
}

#[cfg(feature = "paged-attn")]
pub(super) enum SegmentPreparedEvent {
    Prepared(SegmentPreparedMetadata),
}

#[cfg(feature = "paged-attn")]
struct PrefillCoalescedPrepared {
    prepared: PreparedAudioBatch,
}

#[cfg(feature = "paged-attn")]
enum PrefillCoalescedPoll {
    Ready(PrefillCoalescedPrepared),
    Empty,
    Closed,
}

#[cfg(feature = "paged-attn")]
pub(super) fn spawn_segment_prefill_worker(
    asr: Arc<dyn AsrModel>,
    asr_options: AsrOptions,
    encoded_rx: mpsc::UnboundedReceiver<SegmentEncoderOutput>,
    prefilled_tx: mpsc::UnboundedSender<SegmentPrefillMessage>,
    decode_idle_flush: Option<std::time::Duration>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let encoded_rx = Arc::new(tokio::sync::Mutex::new(encoded_rx));
        let mut next_producer_id = 0usize;
        loop {
            let first = {
                let mut encoded_rx = encoded_rx.lock().await;
                encoded_rx.recv().await
            };
            let Some(first) = first else {
                break;
            };
            let SegmentEncoderOutput::Encoded(first) = first;
            let SegmentEncodedMessage {
                mut segments,
                stage_start,
                encoder_timing,
                batch,
            } = first;
            apply_batch_timing_to_segments(&mut segments, encoder_timing);
            let initial = match batch {
                Ok(batch) => batch,
                Err(error) => {
                    if prefilled_tx
                        .send(SegmentPrefillMessage::Failure(SegmentEncodedFailure {
                            segments,
                            stage_start,
                            error,
                        }))
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
            };
            let result = tokio::task::spawn_blocking({
                let producer_id = next_producer_id;
                next_producer_id = next_producer_id.saturating_add(1);
                let asr = Arc::clone(&asr);
                let asr_options = asr_options.clone();
                let encoded_rx = Arc::clone(&encoded_rx);
                let prefilled_tx = prefilled_tx.clone();
                let runtime_handle = tokio::runtime::Handle::current();
                move || {
                    run_global_segment_prefill(
                        asr,
                        segments,
                        stage_start,
                        initial,
                        encoded_rx,
                        runtime_handle,
                        &asr_options,
                        decode_idle_flush,
                        prefilled_tx,
                        producer_id,
                    )
                }
            })
            .await
            .unwrap_or_else(|error| {
                Err(anyhow::anyhow!(
                    "ASR dynamic prefill worker join error: {error}"
                ))
            });
            if let Err(error) = result {
                tracing::warn!(
                    target: "vasr_transcribe::pipeline",
                    "ASR dynamic prefill failed: {error}"
                );
            }
            if decode_idle_flush.is_none() {
                break;
            }
        }
    })
}
#[cfg(feature = "paged-attn")]
#[derive(Debug)]
pub(super) struct SegmentEncodedFailure {
    pub(super) segments: Vec<AsrSegmentJob>,
    pub(super) stage_start: Instant,
    pub(super) error: anyhow::Error,
}

#[cfg(feature = "paged-attn")]
#[derive(Debug)]
pub(super) struct SegmentPreparedMetadata {
    pub(super) segments: Vec<AsrSegmentJob>,
    pub(super) waveforms: Vec<Waveform>,
    pub(super) meta: EncodedAsrBatchMetadata,
    pub(super) stage_start: Instant,
}
#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn run_global_segment_prefill(
    asr: Arc<dyn AsrModel>,
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    initial: SegmentEncodedAsrBatch,
    encoded_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: tokio::runtime::Handle,
    asr_options: &AsrOptions,
    decode_idle_flush: Option<std::time::Duration>,
    decoder_tx: mpsc::UnboundedSender<SegmentPrefillMessage>,
    producer_id: usize,
) -> Result<()> {
    let qwen3 = asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("dynamic staged ASR prefill requires Qwen3ASR"))?;
    let (_inference, waveforms, encoded) = initial.into_parts();
    let (initial_prepared, initial_meta) = qwen3.prepare_encoded_audio_features(encoded)?;
    let prefill_log_segments = segment_log_items(&segments);
    decoder_tx
        .send(SegmentPrefillMessage::Prepared {
            producer_id,
            result: Ok(SegmentPreparedEvent::Prepared(SegmentPreparedMetadata {
                segments,
                waveforms,
                meta: initial_meta,
                stage_start,
            })),
        })
        .map_err(|_| anyhow::anyhow!("ASR global decoder closed"))?;

    let runtime_options = vasr::models::qwen3asr::qwen_transcribe_options(asr_options);
    run_segment_asr_prefill_dynamic_worker(
        qwen3,
        initial_prepared,
        encoded_rx,
        runtime_handle,
        runtime_options,
        decode_idle_flush,
        decoder_tx,
        producer_id,
        prefill_log_segments,
    );
    Ok(())
}
#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn run_segment_asr_prefill_dynamic_worker(
    qwen3: &Qwen3ASR,
    initial_prepared: PreparedAudioBatch,
    encoded_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: tokio::runtime::Handle,
    runtime_options: vasr::models::qwen3asr::TranscribeOptions,
    decode_idle_flush: Option<std::time::Duration>,
    decoder_tx: mpsc::UnboundedSender<SegmentPrefillMessage>,
    producer_id: usize,
    prefill_log_segments: Vec<SegmentLogItem>,
) {
    let run = (|| -> Result<()> {
        let prefill_log_segments = Mutex::new(prefill_log_segments);
        let mut source_closed = false;
        let mut poll_prepared = |may_block: bool| -> Result<PreparedAudioBatchPoll> {
            if decode_idle_flush.is_some() && !may_block {
                return Ok(PreparedAudioBatchPoll::Empty);
            }
            // 单次轮询即可得出结果，因此无需循环重试。
            if source_closed {
                return Ok(PreparedAudioBatchPoll::Closed);
            }
            let message = if may_block {
                recv_segment_encoded_message_blocking(
                    &encoded_rx,
                    &runtime_handle,
                    decode_idle_flush,
                    &mut source_closed,
                )
            } else {
                try_recv_segment_encoded_message(&encoded_rx, &runtime_handle, &mut source_closed)
            };
            let Some(message) = message else {
                return Ok(PreparedAudioBatchPoll::Empty);
            };
            Ok(
                match prepare_coalesced_segment_prefill_batch(
                    qwen3,
                    message,
                    &encoded_rx,
                    &runtime_handle,
                    &decoder_tx,
                    producer_id,
                    &prefill_log_segments,
                )? {
                    PrefillCoalescedPoll::Ready(coalesced) => {
                        PreparedAudioBatchPoll::Ready(coalesced.prepared)
                    }
                    PrefillCoalescedPoll::Empty => PreparedAudioBatchPoll::Empty,
                    PrefillCoalescedPoll::Closed => PreparedAudioBatchPoll::Closed,
                },
            )
        };
        let mut producer = qwen3.dynamic_prefill_producer(
            initial_prepared,
            &mut poll_prepared,
            runtime_options,
        )?;
        loop {
            let stage_start = Instant::now();
            let message = producer.run_stage(true)?;
            #[cfg(feature = "timing")]
            let elapsed_seconds = message.timings.prefill_us as f64 / 1_000_000.0;
            #[cfg(not(feature = "timing"))]
            let elapsed_seconds = stage_start.elapsed().as_secs_f64();
            #[cfg(feature = "timing")]
            let _ = stage_start;
            let wave = message.prefill_wave;
            let done = producer.is_done();
            if wave > 0 {
                log_prefill_batch_done(
                    &prefill_log_segments
                        .lock()
                        .map_err(|_| anyhow::anyhow!("prefill log segments lock poisoned"))?,
                    &message,
                    elapsed_seconds,
                );
            }
            if wave > 0
                && decoder_tx
                    .send(SegmentPrefillMessage::Prefill {
                        producer_id,
                        result: Ok(message),
                    })
                    .is_err()
            {
                return Ok(());
            }
            if done {
                break;
            }
            if wave == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        Ok(())
    })();
    if let Err(error) = run {
        let _ = decoder_tx.send(SegmentPrefillMessage::Prefill {
            producer_id,
            result: Err(error),
        });
    }
}

#[cfg(feature = "paged-attn")]
fn prepare_coalesced_segment_prefill_batch(
    qwen3: &Qwen3ASR,
    first: SegmentEncoderOutput,
    encoded_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: &tokio::runtime::Handle,
    decoder_tx: &mpsc::UnboundedSender<SegmentPrefillMessage>,
    producer_id: usize,
    prefill_log_segments: &Mutex<Vec<SegmentLogItem>>,
) -> Result<PrefillCoalescedPoll> {
    let mut prepared_batches = Vec::new();

    let mut current = Some(first);
    let deadline = std::time::Instant::now() + asr_prefill_coalesce_wait();
    loop {
        let Some(message) = current.take() else {
            break;
        };
        if !prepare_segment_encoded_for_prefill(
            qwen3,
            message,
            decoder_tx,
            producer_id,
            prefill_log_segments,
            &mut prepared_batches,
        )? {
            return Ok(PrefillCoalescedPoll::Closed);
        }

        if std::time::Instant::now() >= deadline {
            break;
        }
        current =
            try_recv_segment_encoded_message_for_coalesce(encoded_rx, runtime_handle, deadline);
        if current.is_none() {
            break;
        }
    }

    if prepared_batches.is_empty() {
        return Ok(PrefillCoalescedPoll::Empty);
    }

    Ok(PrefillCoalescedPoll::Ready(PrefillCoalescedPrepared {
        prepared: merge_prepared_audio_batches(prepared_batches)?,
    }))
}

#[cfg(feature = "paged-attn")]
fn prepare_segment_encoded_for_prefill(
    qwen3: &Qwen3ASR,
    message: SegmentEncoderOutput,
    decoder_tx: &mpsc::UnboundedSender<SegmentPrefillMessage>,
    producer_id: usize,
    prefill_log_segments: &Mutex<Vec<SegmentLogItem>>,
    prepared_batches: &mut Vec<PreparedAudioBatch>,
) -> Result<bool> {
    let SegmentEncoderOutput::Encoded(message) = message;
    match message.batch {
        Ok(batch) => {
            let (_next_inference, next_waveforms, next_encoded) = batch.into_parts();
            let (prepared, meta) = qwen3.prepare_encoded_audio_features(next_encoded)?;
            prefill_log_segments
                .lock()
                .map_err(|_| anyhow::anyhow!("prefill log segments lock poisoned"))?
                .extend(segment_log_items(&message.segments));
            if decoder_tx
                .send(SegmentPrefillMessage::Prepared {
                    producer_id,
                    result: Ok(SegmentPreparedEvent::Prepared(SegmentPreparedMetadata {
                        segments: message.segments,
                        waveforms: next_waveforms,
                        meta,
                        stage_start: message.stage_start,
                    })),
                })
                .is_err()
            {
                return Ok(false);
            }
            prepared_batches.push(prepared);
        }
        Err(error) => {
            if decoder_tx
                .send(SegmentPrefillMessage::Prepared {
                    producer_id,
                    result: Err(SegmentEncodedFailure {
                        segments: message.segments,
                        stage_start: message.stage_start,
                        error,
                    }),
                })
                .is_err()
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(feature = "paged-attn")]
fn try_recv_segment_encoded_message_for_coalesce(
    encoded_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: &tokio::runtime::Handle,
    deadline: std::time::Instant,
) -> Option<SegmentEncoderOutput> {
    let now = std::time::Instant::now();
    if now >= deadline {
        return None;
    }
    let remaining = deadline.saturating_duration_since(now);
    match runtime_handle.block_on(async {
        let mut encoded_rx = encoded_rx.lock().await;
        tokio::time::timeout(remaining, encoded_rx.recv()).await
    }) {
        Ok(Some(message)) => Some(message),
        Ok(None) | Err(tokio::time::error::Elapsed { .. }) => None,
    }
}

#[cfg(feature = "paged-attn")]
fn merge_prepared_audio_batches(batches: Vec<PreparedAudioBatch>) -> Result<PreparedAudioBatch> {
    let mut prepared = Vec::new();
    let mut audio_seconds = Vec::new();
    let mut audio_features = Vec::new();
    for batch in batches {
        prepared.extend(batch.prepared);
        audio_seconds.extend(batch.audio_seconds);
        audio_features.push(batch.audio_features);
    }

    let audio_features = if audio_features.len() == 1 {
        audio_features
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing prepared audio features"))?
    } else {
        let first_dims = audio_features
            .first()
            .map(|tensor| tensor.dims().to_vec())
            .unwrap_or_default();
        if first_dims.len() != 2 {
            bail!("prepared audio features must be rank-2, got dims={first_dims:?}");
        }
        candle_core::Tensor::cat(audio_features.as_slice(), 0)?
    };

    Ok(PreparedAudioBatch {
        prepared,
        audio_features,
        audio_seconds,
    })
}

#[cfg(feature = "paged-attn")]
fn recv_segment_encoded_message_blocking(
    encoded_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: &tokio::runtime::Handle,
    timeout: Option<std::time::Duration>,
    source_closed: &mut bool,
) -> Option<SegmentEncoderOutput> {
    if let Some(timeout) = timeout {
        match runtime_handle.block_on(async {
            let mut encoded_rx = encoded_rx.lock().await;
            tokio::time::timeout(timeout, encoded_rx.recv()).await
        }) {
            Ok(Some(message)) => Some(message),
            Ok(None) | Err(tokio::time::error::Elapsed { .. }) => {
                *source_closed = true;
                None
            }
        }
    } else {
        match runtime_handle.block_on(async {
            let mut encoded_rx = encoded_rx.lock().await;
            encoded_rx.recv().await
        }) {
            Some(message) => Some(message),
            None => {
                *source_closed = true;
                None
            }
        }
    }
}

#[cfg(feature = "paged-attn")]
fn try_recv_segment_encoded_message(
    encoded_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: &tokio::runtime::Handle,
    source_closed: &mut bool,
) -> Option<SegmentEncoderOutput> {
    match runtime_handle.block_on(async {
        let mut encoded_rx = encoded_rx.lock().await;
        encoded_rx.try_recv()
    }) {
        Ok(message) => Some(message),
        Err(mpsc::error::TryRecvError::Empty) => None,
        Err(mpsc::error::TryRecvError::Disconnected) => {
            *source_closed = true;
            None
        }
    }
}
