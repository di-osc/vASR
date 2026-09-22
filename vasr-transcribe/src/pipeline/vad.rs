use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, JoinSet};
use vasr::{AudioInput, SAMPLE_RATE_HZ, normalize_audio_input};
use vasr_data::{TimeRange, Waveform};
use vasr_vad::{VadModel, VadOptions, VadSegment};

use super::{
    AsrJob, AsrPipelineMessage, LoadedJob, LoadedQueueMemoryBudget, PendingGlobalRequest,
    PipelineRequestId, StageSpan, TranscribeItemOutcome, TranscribeItemSpans, TranscribeItemTiming,
    VadStageStats, VadTaskStats, VadToAsrMessage, acquire_vad_to_asr_message_memory,
    log_vad_summary, record_request_vad_stage, record_vad_task_stats, vad_preparation_stats,
};

#[cfg(feature = "paged-attn")]
use super::speech_job_to_segments;

/// Output of the VAD stage, ready for batched ASR.
#[derive(Debug, Clone)]
pub(super) struct VadPrepared {
    /// 语音活动标注，仅在 `paged-attn` 分段解码路径中消费。
    #[cfg_attr(not(feature = "paged-attn"), allow(dead_code))]
    pub(super) speech_annotations: Vec<vasr_data::TimeSpan>,
    pub(super) segments: Vec<VadSegment>,
    /// 分段音频切片，仅在 `paged-attn` 分段解码路径中消费。
    #[cfg_attr(not(feature = "paged-attn"), allow(dead_code))]
    pub(super) slices: Vec<Waveform>,
}

/// Result of the VAD stage. `Disabled` is the only case that should fall back
/// to raw ASR; `NoSpeech` means VAD ran and found nothing to recognize.
#[derive(Debug, Clone)]
pub(super) enum VadPreparation {
    Disabled,
    NoSpeech,
    Speech(VadPrepared),
}

pub(super) fn prepare_vad_stage(
    vad: Option<&dyn VadModel>,
    waveform: &Waveform,
    vad_options: &VadOptions,
) -> Result<VadPreparation> {
    let Some(vad) = vad else {
        return Ok(VadPreparation::Disabled);
    };
    let vad_waveform = prepare_waveform_for_vad(waveform)?;
    let segments = pad_vad_segments(
        vad.detect(&vad_waveform, vad_options)?,
        vad_options.pad_ms as usize,
        waveform.duration_ms().round() as usize,
    );
    if segments.is_empty() {
        return Ok(VadPreparation::NoSpeech);
    }
    let speech_annotations = segments
        .iter()
        .map(|segment| vasr::activity_span(segment.range, "speech", vasr::VAD_SOURCE))
        .collect();
    let slices = segments
        .iter()
        .map(|segment| {
            waveform.slice_ms(segment.range.start_ms as u64, segment.range.end_ms as u64)
        })
        .collect();
    Ok(VadPreparation::Speech(VadPrepared {
        speech_annotations,
        segments,
        slices,
    }))
}

fn pad_vad_segments(
    mut segments: Vec<VadSegment>,
    pad_ms: usize,
    audio_ms: usize,
) -> Vec<VadSegment> {
    if pad_ms == 0 {
        return segments;
    }
    for segment in &mut segments {
        segment.range = TimeRange::new(
            segment.range.start_ms.saturating_sub(pad_ms),
            segment.range.end_ms.saturating_add(pad_ms).min(audio_ms),
        );
    }
    segments
}

#[cfg(feature = "paged-attn")]
pub(super) fn offset_annotations(
    annotations: Vec<vasr_data::TimeSpan>,
    offset: usize,
) -> Vec<vasr_data::TimeSpan> {
    vasr::offset_spans(annotations, offset)
}

fn prepare_waveform_for_vad(waveform: &Waveform) -> Result<Waveform> {
    let mono = waveform.to_mono()?;
    if mono.sample_rate == SAMPLE_RATE_HZ {
        return Ok(mono);
    }
    let samples = normalize_audio_input(&AudioInput::Waveform {
        samples: &mono.samples,
        sample_rate: mono.sample_rate,
    })?;
    Ok(Waveform::new(samples, SAMPLE_RATE_HZ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_vad_worker(
    vad: Option<Arc<dyn VadModel>>,
    vad_options: VadOptions,
    max_in_flight: usize,
    mut loader_to_vad_rx: mpsc::UnboundedReceiver<Result<LoadedJob, TranscribeItemOutcome>>,
    vad_to_asr_tx: mpsc::UnboundedSender<VadToAsrMessage>,
    vad_to_asr_queue_memory: LoadedQueueMemoryBudget,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    stats: Arc<Mutex<VadStageStats>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let wall_start = Instant::now();
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        let mut free_workers = (0..max_in_flight).rev().collect::<Vec<_>>();
        while let Some(message) = loader_to_vad_rx.recv().await {
            match message {
                Ok(job) => {
                    if tasks.len() >= max_in_flight
                        && let Ok(Some(worker_id)) = send_next_vad_task(&mut tasks, &stats).await
                    {
                        free_workers.push(worker_id);
                    }
                    let worker_id = free_workers.pop().unwrap_or(tasks.len());
                    spawn_vad_task(
                        &mut tasks,
                        vad.clone(),
                        vad_options.clone(),
                        vad_to_asr_tx.clone(),
                        vad_to_asr_queue_memory.clone(),
                        pending_requests.clone(),
                        job,
                        worker_id,
                    );
                }
                Err(outcome) => {
                    let mut send_wait_seconds = 0.0;
                    let _ = send_vad_to_asr_timed(
                        &vad_to_asr_tx,
                        &vad_to_asr_queue_memory,
                        Err(outcome),
                        &mut send_wait_seconds,
                    )
                    .await;
                }
            }
        }
        while !tasks.is_empty() {
            if let Ok(Some(worker_id)) = send_next_vad_task(&mut tasks, &stats).await {
                free_workers.push(worker_id);
            }
        }
        if let Ok(mut stats) = stats.lock() {
            stats.wall_seconds = wall_start.elapsed().as_secs_f64();
            log_vad_summary(&stats);
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn spawn_vad_task(
    tasks: &mut JoinSet<Result<VadTaskStats, TranscribeItemOutcome>>,
    vad: Option<Arc<dyn VadModel>>,
    vad_options: VadOptions,
    vad_to_asr_tx: mpsc::UnboundedSender<VadToAsrMessage>,
    vad_to_asr_queue_memory: LoadedQueueMemoryBudget,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    job: LoadedJob,
    worker_id: usize,
) {
    tasks.spawn(async move {
        let stage_start = Instant::now();
        let request_item = job.request_item;
        let mut timing = job.timing;
        let audio_seconds = job.waveform.duration_seconds();
        let job_index = job.index;
        let job_channel = job.channel;
        let job_timing = job.timing;
        let mut spans = job.spans;
        let vad_queue_span = spans
            .loader
            .and_then(|loader| StageSpan::try_new(loader.end, stage_start));
        let prepared = tokio::task::spawn_blocking({
            let vad = vad.clone();
            let waveform = job.waveform.clone();
            let vad_options = vad_options.clone();
            move || prepare_vad_stage(vad.as_deref(), &waveform, &vad_options)
        })
        .await;
        let compute_seconds = stage_start.elapsed().as_secs_f64();
        let vad_span = StageSpan::new(stage_start, Instant::now());
        spans.vad = Some(vad_span);
        timing.vad_seconds += compute_seconds;
        let mut send_wait_seconds = 0.0;

        match prepared {
            Ok(Ok(prepared)) => {
                let (segments, speech_seconds) = vad_preparation_stats(&prepared);
                let no_speech = matches!(prepared, VadPreparation::NoSpeech);
                let asr_job = AsrJob {
                    index: job_index,
                    request_item,
                    channel: job.channel,
                    timing,
                    spans,
                    prepared,
                    waveform: job.waveform,
                };
                match asr_job.prepared {
                    #[cfg(feature = "paged-attn")]
                    VadPreparation::Speech(_) => {
                        let segment_jobs = match speech_job_to_segments(asr_job) {
                            Ok(segments) => segments,
                            Err(error) => {
                                send_vad_to_asr_timed(
                                    &vad_to_asr_tx,
                                    &vad_to_asr_queue_memory,
                                    Err(TranscribeItemOutcome {
                                        index: job_index,
                                        channel: job.channel,
                                        result: Err(error),
                                        audio_seconds,
                                        timing,
                                        spans,
                                        bad_component: Some("vad"),
                                    }),
                                    &mut send_wait_seconds,
                                )
                                .await
                                .map_err(|error| {
                                    TranscribeItemOutcome {
                                        index: job_index,
                                        channel: job.channel,
                                        result: Err(error),
                                        audio_seconds,
                                        timing,
                                        spans,
                                        bad_component: Some("vad"),
                                    }
                                })?;
                                Vec::new()
                            }
                        };
                        for segment in segment_jobs {
                            send_vad_to_asr_timed(
                                &vad_to_asr_tx,
                                &vad_to_asr_queue_memory,
                                Ok(AsrPipelineMessage::Segment(segment)),
                                &mut send_wait_seconds,
                            )
                            .await
                            .map_err(|error| {
                                TranscribeItemOutcome {
                                    index: job_index,
                                    channel: job.channel,
                                    result: Err(error),
                                    audio_seconds,
                                    timing,
                                    spans,
                                    bad_component: Some("asr"),
                                }
                            })?;
                        }
                    }
                    #[cfg(not(feature = "paged-attn"))]
                    VadPreparation::Speech(_) => {
                        send_vad_to_asr_timed(
                            &vad_to_asr_tx,
                            &vad_to_asr_queue_memory,
                            Err(TranscribeItemOutcome {
                                index: job_index,
                                channel: job.channel,
                                result: Err(anyhow::anyhow!("speech ASR requires paged-attn")),
                                audio_seconds,
                                timing,
                                spans,
                                bad_component: Some("asr"),
                            }),
                            &mut send_wait_seconds,
                        )
                        .await
                        .map_err(|error| TranscribeItemOutcome {
                            index: job_index,
                            channel: job.channel,
                            result: Err(error),
                            audio_seconds,
                            timing,
                            spans,
                            bad_component: Some("asr"),
                        })?;
                    }
                    _ => {
                        send_vad_to_asr_timed(
                            &vad_to_asr_tx,
                            &vad_to_asr_queue_memory,
                            Ok(AsrPipelineMessage::Job(asr_job)),
                            &mut send_wait_seconds,
                        )
                        .await
                        .map_err(|error| TranscribeItemOutcome {
                            index: job_index,
                            channel: job.channel,
                            result: Err(error),
                            audio_seconds,
                            timing,
                            spans,
                            bad_component: Some("asr"),
                        })?;
                    }
                }
                if let (Some(pending_requests), Some(request_item)) =
                    (&pending_requests, request_item)
                {
                    record_request_vad_stage(
                        pending_requests,
                        request_item.request_id,
                        vad_span,
                        vad_queue_span,
                        audio_seconds,
                        segments,
                    );
                }
                let wall_seconds = stage_start.elapsed().as_secs_f64();
                Ok(VadTaskStats {
                    jobs: 1,
                    no_speech: usize::from(no_speech),
                    segments,
                    audio_seconds,
                    speech_seconds,
                    compute_seconds,
                    worker_id,
                    send_wait_seconds,
                    wall_seconds,
                    ..Default::default()
                })
            }
            Ok(Err(error)) => {
                send_vad_to_asr_timed(
                    &vad_to_asr_tx,
                    &vad_to_asr_queue_memory,
                    Err(TranscribeItemOutcome {
                        index: job_index,
                        channel: job.channel,
                        result: Err(error),
                        audio_seconds,
                        timing,
                        spans,
                        bad_component: Some("vad"),
                    }),
                    &mut send_wait_seconds,
                )
                .await
                .map_err(|error| TranscribeItemOutcome {
                    index: job_index,
                    channel: job.channel,
                    result: Err(error),
                    audio_seconds,
                    timing,
                    spans,
                    bad_component: Some("vad"),
                })?;
                if let (Some(pending_requests), Some(request_item)) =
                    (&pending_requests, request_item)
                {
                    record_request_vad_stage(
                        pending_requests,
                        request_item.request_id,
                        vad_span,
                        vad_queue_span,
                        audio_seconds,
                        0,
                    );
                }
                let wall_seconds = stage_start.elapsed().as_secs_f64();
                Ok(VadTaskStats {
                    jobs: 1,
                    failed: 1,
                    audio_seconds,
                    compute_seconds,
                    worker_id,
                    send_wait_seconds,
                    wall_seconds,
                    ..Default::default()
                })
            }
            Err(error) => {
                send_vad_to_asr_timed(
                    &vad_to_asr_tx,
                    &vad_to_asr_queue_memory,
                    Err(TranscribeItemOutcome {
                        index: job_index,
                        channel: job_channel,
                        result: Err(anyhow::anyhow!("VAD join error: {error}")),
                        audio_seconds,
                        timing: job_timing,
                        spans,
                        bad_component: Some("vad"),
                    }),
                    &mut send_wait_seconds,
                )
                .await
                .map_err(|error| TranscribeItemOutcome {
                    index: job_index,
                    channel: job_channel,
                    result: Err(error),
                    audio_seconds,
                    timing: job_timing,
                    spans,
                    bad_component: Some("vad"),
                })?;
                if let (Some(pending_requests), Some(request_item)) =
                    (&pending_requests, request_item)
                {
                    record_request_vad_stage(
                        pending_requests,
                        request_item.request_id,
                        vad_span,
                        vad_queue_span,
                        audio_seconds,
                        0,
                    );
                }
                let wall_seconds = stage_start.elapsed().as_secs_f64();
                Ok(VadTaskStats {
                    jobs: 1,
                    failed: 1,
                    audio_seconds,
                    compute_seconds,
                    worker_id,
                    send_wait_seconds,
                    wall_seconds,
                    ..Default::default()
                })
            }
        }
    });
}

async fn send_vad_to_asr_timed(
    tx: &mpsc::UnboundedSender<VadToAsrMessage>,
    budget: &LoadedQueueMemoryBudget,
    message: Result<AsrPipelineMessage, TranscribeItemOutcome>,
    wait_seconds: &mut f64,
) -> Result<()> {
    let send_start = Instant::now();
    let permit = acquire_vad_to_asr_message_memory(budget, &message).await?;
    let result = tx
        .send(VadToAsrMessage {
            message,
            _queue_memory_permit: permit,
        })
        .map_err(|_| anyhow::anyhow!("ASR stage channel closed"));
    *wait_seconds += send_start.elapsed().as_secs_f64();
    result
}

/// 从任务集合取下一个 VAD 任务结果。
///
/// `TranscribeItemOutcome` 体积较大，但它是流水线统一的结果载体，
/// 逐个 Box 会牵动上下游类型，因此这里保留原样并显式放开该 lint。
#[allow(clippy::result_large_err)]
async fn send_next_vad_task(
    tasks: &mut JoinSet<Result<VadTaskStats, TranscribeItemOutcome>>,
    stats: &Arc<Mutex<VadStageStats>>,
) -> Result<Option<usize>, TranscribeItemOutcome> {
    match tasks.join_next().await {
        Some(Ok(result)) => {
            let worker_id = result.as_ref().ok().map(|task_stats| task_stats.worker_id);
            if let Ok(task_stats) = &result {
                record_vad_task_stats(stats, task_stats);
            }
            result.map(|_| worker_id)
        }
        Some(Err(error)) => Err(TranscribeItemOutcome {
            index: 0,
            channel: "mono",
            result: Err(anyhow::anyhow!("VAD task join error: {error}")),
            audio_seconds: 0.0,
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            bad_component: Some("vad"),
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct RecordingVad {
        seen: Mutex<Vec<(u32, u16)>>,
    }

    impl VadModel for RecordingVad {
        fn detect(&self, waveform: &Waveform, _options: &VadOptions) -> Result<Vec<VadSegment>> {
            self.seen
                .lock()
                .unwrap()
                .push((waveform.sample_rate, waveform.channels));
            Ok(vec![VadSegment {
                range: TimeRange::new(0, 500),
                probability: 0.9,
            }])
        }

        fn new_stream(
            &self,
            _options: &VadOptions,
        ) -> Result<Box<dyn vasr_vad::StreamingVadSession>> {
            anyhow::bail!("streaming VAD is not used in this test")
        }
    }

    #[test]
    fn vad_stage_resamples_for_vad_but_slices_original_waveform() -> Result<()> {
        let vad = Arc::new(RecordingVad::default());
        let waveform = Waveform::new_with_channels(vec![0.0; 48_000 * 2], 48_000, 2);

        let prepared = prepare_vad_stage(Some(vad.as_ref()), &waveform, &VadOptions::default())?;

        assert_eq!(vad.seen.lock().unwrap().as_slice(), &[(SAMPLE_RATE_HZ, 1)]);
        let VadPreparation::Speech(prepared) = prepared else {
            panic!("expected speech");
        };
        assert_eq!(prepared.slices.len(), 1);
        assert_eq!(prepared.slices[0].sample_rate, 48_000);
        assert_eq!(prepared.slices[0].channels, 2);
        assert_eq!(prepared.slices[0].duration_ms(), 500.0);
        Ok(())
    }

    #[test]
    fn vad_stage_applies_padding_before_slicing() -> Result<()> {
        let vad = Arc::new(RecordingVad::default());
        let waveform = Waveform::new(vec![0.0; 16_000], 16_000);
        let prepared = prepare_vad_stage(
            Some(vad.as_ref()),
            &waveform,
            &VadOptions {
                pad_ms: 100,
                ..VadOptions::default()
            },
        )?;

        let VadPreparation::Speech(prepared) = prepared else {
            panic!("expected speech");
        };
        assert_eq!(prepared.segments[0].range.start_ms, 0);
        assert_eq!(prepared.segments[0].range.end_ms, 600);
        assert_eq!(prepared.slices[0].duration_ms(), 600.0);
        Ok(())
    }
}
