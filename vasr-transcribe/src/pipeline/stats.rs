//! Stage counters, timing allocation and throughput logging.
use super::*;

#[derive(Debug, Default)]
pub(super) struct LoaderStageStats {
    pub(super) jobs: usize,
    pub(super) failed: usize,
    pub(super) audio_seconds: f64,
    pub(super) compute_seconds: f64,
    pub(super) worker_compute_seconds: HashMap<usize, f64>,
    pub(super) wall_seconds: f64,
}

#[derive(Debug, Default)]
pub(super) struct LoaderTaskStats {
    pub(super) jobs: usize,
    pub(super) failed: usize,
    pub(super) audio_seconds: f64,
    pub(super) compute_seconds: f64,
    pub(super) worker_id: usize,
}

#[derive(Debug, Default)]
pub(super) struct VadStageStats {
    pub(super) jobs: usize,
    pub(super) failed: usize,
    pub(super) no_speech: usize,
    pub(super) segments: usize,
    pub(super) audio_seconds: f64,
    pub(super) speech_seconds: f64,
    pub(super) compute_seconds: f64,
    pub(super) worker_compute_seconds: HashMap<usize, f64>,
    pub(super) send_wait_seconds: f64,
    pub(super) wall_seconds: f64,
}

#[derive(Debug, Default)]
pub(super) struct VadTaskStats {
    pub(super) jobs: usize,
    pub(super) failed: usize,
    pub(super) no_speech: usize,
    pub(super) segments: usize,
    pub(super) audio_seconds: f64,
    pub(super) speech_seconds: f64,
    pub(super) compute_seconds: f64,
    pub(super) worker_id: usize,
    pub(super) send_wait_seconds: f64,
    pub(super) wall_seconds: f64,
}

#[derive(Debug, Default)]
pub(super) struct AsrStageStats {
    pub(super) batches: usize,
    pub(super) jobs: usize,
    pub(super) segments: usize,
    pub(super) audio_seconds: f64,
    pub(super) wall_seconds: f64,
    pub(super) encode_batches: usize,
    pub(super) encode_segments: usize,
    pub(super) encode_audio_seconds: f64,
    pub(super) encode_seconds: f64,
    pub(super) decode_batches: usize,
    pub(super) decode_segments: usize,
    pub(super) decode_audio_seconds: f64,
    pub(super) decode_seconds: f64,
    pub(super) audio_encoder_seconds: f64,
    pub(super) prefill_seconds: f64,
    pub(super) prefill_tokens: usize,
    pub(super) decode_model_seconds: f64,
    pub(super) decode_tokens: usize,
    pub(super) decode_steps: usize,
}

#[derive(Debug, Clone, Default)]
pub(super) struct LoaderStageStatsSnapshot {
    pub(super) jobs: usize,
    pub(super) failed: usize,
    pub(super) audio_seconds: f64,
    pub(super) compute_seconds: f64,
    pub(super) worker_compute_seconds: HashMap<usize, f64>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct VadStageStatsSnapshot {
    pub(super) jobs: usize,
    pub(super) failed: usize,
    pub(super) no_speech: usize,
    pub(super) segments: usize,
    pub(super) audio_seconds: f64,
    pub(super) speech_seconds: f64,
    pub(super) compute_seconds: f64,
    pub(super) worker_compute_seconds: HashMap<usize, f64>,
    pub(super) send_wait_seconds: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct AsrStageStatsSnapshot {
    pub(super) batches: usize,
    pub(super) jobs: usize,
    pub(super) segments: usize,
    pub(super) audio_seconds: f64,
    pub(super) wall_seconds: f64,
    pub(super) encode_batches: usize,
    pub(super) encode_segments: usize,
    pub(super) encode_audio_seconds: f64,
    pub(super) encode_seconds: f64,
    pub(super) decode_batches: usize,
    pub(super) decode_segments: usize,
    pub(super) decode_audio_seconds: f64,
    pub(super) decode_seconds: f64,
    pub(super) audio_encoder_seconds: f64,
    pub(super) prefill_seconds: f64,
    pub(super) prefill_tokens: usize,
    pub(super) decode_model_seconds: f64,
    pub(super) decode_tokens: usize,
    pub(super) decode_steps: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct StageMetrics {
    pub(super) speedup: f64,
    pub(super) rtf: f64,
}

pub(super) fn stage_metrics(audio_seconds: f64, wall_seconds: f64) -> StageMetrics {
    if audio_seconds <= 0.0 || wall_seconds <= 0.0 {
        return StageMetrics {
            speedup: 0.0,
            rtf: 0.0,
        };
    }
    StageMetrics {
        speedup: audio_seconds / wall_seconds,
        rtf: wall_seconds / audio_seconds,
    }
}

pub(super) fn allocated_batch_seconds(
    item_audio_seconds: f64,
    total_audio_seconds: f64,
    item_count: usize,
    elapsed_seconds: f64,
) -> f64 {
    if elapsed_seconds <= 0.0 {
        return 0.0;
    }
    if total_audio_seconds > 0.0 && item_audio_seconds > 0.0 {
        return elapsed_seconds * item_audio_seconds / total_audio_seconds;
    }
    if item_count > 0 {
        return elapsed_seconds / item_count as f64;
    }
    elapsed_seconds
}

pub(super) fn prefill_span_from_decode(
    decode_span: StageSpan,
    timing: Option<TranscribeItemTiming>,
) -> Option<StageSpan> {
    let prefill_seconds = timing?.prefill_seconds;
    if prefill_seconds <= 0.0 {
        return None;
    }
    let prefill_duration = Duration::from_secs_f64(prefill_seconds);
    let decode_duration = decode_span.end.duration_since(decode_span.start);
    let duration = prefill_duration.min(decode_duration);
    Some(StageSpan::new(
        decode_span.start,
        decode_span.start + duration,
    ))
}

#[cfg(feature = "paged-attn")]
pub(super) fn apply_batch_timing_to_segments(
    segments: &mut [AsrSegmentJob],
    timing: TranscribeItemTiming,
) {
    if segments.is_empty() {
        return;
    }
    let total_audio_seconds: f64 = segments
        .iter()
        .map(|segment| segment.waveform.duration_seconds())
        .sum();
    let item_count = segments.len();
    for segment in segments {
        let scale = allocation_scale(
            segment.waveform.duration_seconds(),
            total_audio_seconds,
            item_count,
        );
        segment.timing.add_scaled(timing, scale);
    }
}

pub(super) fn allocation_scale(
    item_audio_seconds: f64,
    total_audio_seconds: f64,
    item_count: usize,
) -> f64 {
    if total_audio_seconds > 0.0 && item_audio_seconds > 0.0 {
        item_audio_seconds / total_audio_seconds
    } else if item_count > 0 {
        1.0 / item_count as f64
    } else {
        1.0
    }
}

#[cfg(feature = "paged-attn")]
pub(super) fn audio_encoder_timing_seconds(
    timings: AudioFeatureEncodeTimings,
) -> TranscribeItemTiming {
    TranscribeItemTiming {
        audio_encoder_seconds: timing_us_to_seconds(
            timings
                .prepare_us
                .saturating_add(timings.stack_us)
                .saturating_add(timings.forward_us)
                .saturating_add(timings.d2h_us),
        ),
        audio_encoder_prepare_seconds: timing_us_to_seconds(timings.prepare_us),
        audio_encoder_stack_seconds: timing_us_to_seconds(timings.stack_us),
        audio_encoder_forward_seconds: timing_us_to_seconds(timings.forward_us),
        audio_encoder_d2h_seconds: timing_us_to_seconds(timings.d2h_us),
        ..Default::default()
    }
}

#[cfg(feature = "timing")]
pub(super) fn transcribe_timing_seconds(timings: &TranscribeTimings) -> TranscribeItemTiming {
    let generation = &timings.generation;
    TranscribeItemTiming {
        audio_encoder_seconds: timing_us_to_seconds(timings.audio_encoder_us),
        prefill_seconds: timing_us_to_seconds(generation.prefill_us),
        prefill_inputs_seconds: timing_us_to_seconds(generation.prefill_inputs_us),
        prefill_rope_seconds: timing_us_to_seconds(generation.prefill_rope_us),
        prefill_metadata_seconds: timing_us_to_seconds(generation.prefill_metadata_us),
        prefill_mask_seconds: timing_us_to_seconds(generation.prefill_mask_us),
        prefill_forward_seconds: timing_us_to_seconds(generation.prefill_forward_us),
        prefill_gather_seconds: timing_us_to_seconds(generation.prefill_gather_us),
        prefill_decode_setup_seconds: timing_us_to_seconds(generation.prefill_decode_setup_us),
        prefill_argmax_seconds: timing_us_to_seconds(generation.prefill_argmax_us),
        decode_seconds: timing_us_to_seconds(generation.decode_us),
        decode_token_tensor_seconds: timing_us_to_seconds(generation.decode_token_tensor_us),
        decode_embed_seconds: timing_us_to_seconds(generation.decode_embed_us),
        decode_position_seconds: timing_us_to_seconds(generation.decode_position_us),
        decode_metadata_seconds: timing_us_to_seconds(generation.decode_metadata_us),
        decode_graph_replay_seconds: timing_us_to_seconds(generation.decode_graph_replay_us),
        decode_forward_seconds: timing_us_to_seconds(generation.decode_forward_us),
        decode_pre_argmax_sync_seconds: timing_us_to_seconds(generation.decode_pre_argmax_sync_us),
        decode_argmax_seconds: timing_us_to_seconds(generation.decode_argmax_us),
        tokenizer_decode_seconds: timing_us_to_seconds(timings.tokenizer_decode_us),
        ..Default::default()
    }
}

pub(super) fn vad_preparation_stats(prepared: &VadPreparation) -> (usize, f64) {
    match prepared {
        VadPreparation::Speech(prepared) => {
            let speech_seconds = prepared
                .segments
                .iter()
                .map(|segment| segment.range.duration() as f64 / 1000.0)
                .sum();
            (prepared.segments.len(), speech_seconds)
        }
        VadPreparation::Disabled | VadPreparation::NoSpeech => (0, 0.0),
    }
}

pub(super) fn record_loader_task_stats(
    stats: &Arc<Mutex<LoaderStageStats>>,
    task: &LoaderTaskStats,
) {
    if let Ok(mut stats) = stats.lock() {
        stats.jobs += task.jobs;
        stats.failed += task.failed;
        stats.audio_seconds += task.audio_seconds;
        stats.compute_seconds += task.compute_seconds;
        *stats
            .worker_compute_seconds
            .entry(task.worker_id)
            .or_default() += task.compute_seconds;
    }
}

pub(super) fn record_vad_task_stats(stats: &Arc<Mutex<VadStageStats>>, task: &VadTaskStats) {
    if let Ok(mut stats) = stats.lock() {
        stats.jobs += task.jobs;
        stats.failed += task.failed;
        stats.no_speech += task.no_speech;
        stats.segments += task.segments;
        stats.audio_seconds += task.audio_seconds;
        stats.speech_seconds += task.speech_seconds;
        stats.compute_seconds += task.compute_seconds;
        *stats
            .worker_compute_seconds
            .entry(task.worker_id)
            .or_default() += task.compute_seconds;
        stats.send_wait_seconds += task.send_wait_seconds;
        stats.wall_seconds += task.wall_seconds;
    }
}

#[cfg(feature = "paged-attn")]
pub(super) fn record_asr_encode_stats(
    stats: &Arc<Mutex<AsrStageStats>>,
    segments: usize,
    audio_seconds: f64,
    wall_seconds: f64,
) {
    if let Ok(mut stats) = stats.lock() {
        stats.encode_batches += 1;
        stats.encode_segments += segments;
        stats.encode_audio_seconds += audio_seconds;
        stats.encode_seconds += wall_seconds;
    }
}

#[cfg(feature = "timing")]
#[allow(dead_code)]
pub(super) fn record_asr_timing_stats(
    stats: &Arc<Mutex<AsrStageStats>>,
    timings: &TranscribeTimings,
) {
    if let Ok(mut stats) = stats.lock() {
        stats.audio_encoder_seconds += timings.audio_encoder_us as f64 / 1_000_000.0;
        stats.prefill_seconds += timings.generation.prefill_us as f64 / 1_000_000.0;
        stats.prefill_tokens = stats
            .prefill_tokens
            .saturating_add(timings.generation.prefill_tokens);
        stats.decode_model_seconds += timings.generation.decode_us as f64 / 1_000_000.0;
        stats.decode_tokens = stats
            .decode_tokens
            .saturating_add(timings.generation.tokens_generated);
        stats.decode_steps = stats.decode_steps.saturating_add(timings.generation.steps);
    }
}

#[allow(dead_code)]
pub(super) fn record_asr_decode_stats(
    stats: &Arc<Mutex<AsrStageStats>>,
    segments: usize,
    audio_seconds: f64,
    wall_seconds: f64,
) {
    if let Ok(mut stats) = stats.lock() {
        stats.decode_batches += 1;
        stats.decode_segments += segments;
        stats.decode_audio_seconds += audio_seconds;
        stats.decode_seconds += wall_seconds;
    }
}

pub(super) fn record_asr_batch_stats(
    stats: &Arc<Mutex<AsrStageStats>>,
    jobs: usize,
    segments: usize,
    audio_seconds: f64,
    wall_seconds: f64,
) {
    if let Ok(mut stats) = stats.lock() {
        stats.batches += 1;
        stats.jobs += jobs;
        stats.segments += segments;
        stats.audio_seconds += audio_seconds;
        stats.wall_seconds += wall_seconds;
    }
}

pub(super) fn spawn_component_throughput_reporter(
    service_id: Arc<Mutex<String>>,
    loader_stats: Arc<Mutex<LoaderStageStats>>,
    vad_stats: Arc<Mutex<VadStageStats>>,
    asr_stats: Arc<Mutex<AsrStageStats>>,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await;
        let mut last_instant = Instant::now();
        let mut last_loader = snapshot_loader_stats(&loader_stats);
        let mut last_vad = snapshot_vad_stats(&vad_stats);
        let mut last_asr = snapshot_asr_stats(&asr_stats);
        loop {
            ticker.tick().await;
            let now = Instant::now();
            let window_seconds = now.duration_since(last_instant).as_secs_f64();
            last_instant = now;

            let loader = snapshot_loader_stats(&loader_stats);
            let vad = snapshot_vad_stats(&vad_stats);
            let asr = snapshot_asr_stats(&asr_stats);

            let loader_delta = loader_delta(last_loader.clone(), loader.clone());
            let vad_delta = vad_delta(last_vad.clone(), vad.clone());
            let asr_delta = asr_delta(last_asr, asr);
            last_loader = loader;
            last_vad = vad;
            last_asr = asr;

            // Optional probes also track decode steps outside the legacy counters.
            vasr::profiling::log_snapshot();
            if loader_delta.jobs == 0
                && loader_delta.failed == 0
                && vad_delta.jobs == 0
                && vad_delta.failed == 0
                && asr_delta.encode_batches == 0
                && asr_delta.decode_batches == 0
                && asr_delta.batches == 0
            {
                continue;
            }

            let service_id = service_id
                .lock()
                .map(|service_id| service_id.clone())
                .unwrap_or_else(|_| String::from("service"));
            log_component_throughput_window(
                &service_id,
                window_seconds,
                loader_delta,
                vad_delta,
                asr_delta,
            );
        }
    })
}

pub(super) fn snapshot_loader_stats(
    stats: &Arc<Mutex<LoaderStageStats>>,
) -> LoaderStageStatsSnapshot {
    stats
        .lock()
        .map(|stats| LoaderStageStatsSnapshot {
            jobs: stats.jobs,
            failed: stats.failed,
            audio_seconds: stats.audio_seconds,
            compute_seconds: stats.compute_seconds,
            worker_compute_seconds: stats.worker_compute_seconds.clone(),
        })
        .unwrap_or_default()
}

pub(super) fn snapshot_vad_stats(stats: &Arc<Mutex<VadStageStats>>) -> VadStageStatsSnapshot {
    stats
        .lock()
        .map(|stats| VadStageStatsSnapshot {
            jobs: stats.jobs,
            failed: stats.failed,
            no_speech: stats.no_speech,
            segments: stats.segments,
            audio_seconds: stats.audio_seconds,
            speech_seconds: stats.speech_seconds,
            compute_seconds: stats.compute_seconds,
            worker_compute_seconds: stats.worker_compute_seconds.clone(),
            send_wait_seconds: stats.send_wait_seconds,
        })
        .unwrap_or_default()
}

pub(super) fn snapshot_asr_stats(stats: &Arc<Mutex<AsrStageStats>>) -> AsrStageStatsSnapshot {
    stats
        .lock()
        .map(|stats| AsrStageStatsSnapshot {
            batches: stats.batches,
            jobs: stats.jobs,
            segments: stats.segments,
            audio_seconds: stats.audio_seconds,
            wall_seconds: stats.wall_seconds,
            encode_batches: stats.encode_batches,
            encode_segments: stats.encode_segments,
            encode_audio_seconds: stats.encode_audio_seconds,
            encode_seconds: stats.encode_seconds,
            decode_batches: stats.decode_batches,
            decode_segments: stats.decode_segments,
            decode_audio_seconds: stats.decode_audio_seconds,
            decode_seconds: stats.decode_seconds,
            audio_encoder_seconds: stats.audio_encoder_seconds,
            prefill_seconds: stats.prefill_seconds,
            prefill_tokens: stats.prefill_tokens,
            decode_model_seconds: stats.decode_model_seconds,
            decode_tokens: stats.decode_tokens,
            decode_steps: stats.decode_steps,
        })
        .unwrap_or_default()
}

pub(super) fn loader_delta(
    before: LoaderStageStatsSnapshot,
    after: LoaderStageStatsSnapshot,
) -> LoaderStageStatsSnapshot {
    LoaderStageStatsSnapshot {
        jobs: after.jobs.saturating_sub(before.jobs),
        failed: after.failed.saturating_sub(before.failed),
        audio_seconds: (after.audio_seconds - before.audio_seconds).max(0.0),
        compute_seconds: (after.compute_seconds - before.compute_seconds).max(0.0),
        worker_compute_seconds: worker_compute_delta(
            before.worker_compute_seconds,
            after.worker_compute_seconds,
        ),
    }
}

pub(super) fn vad_delta(
    before: VadStageStatsSnapshot,
    after: VadStageStatsSnapshot,
) -> VadStageStatsSnapshot {
    VadStageStatsSnapshot {
        jobs: after.jobs.saturating_sub(before.jobs),
        failed: after.failed.saturating_sub(before.failed),
        no_speech: after.no_speech.saturating_sub(before.no_speech),
        segments: after.segments.saturating_sub(before.segments),
        audio_seconds: (after.audio_seconds - before.audio_seconds).max(0.0),
        speech_seconds: (after.speech_seconds - before.speech_seconds).max(0.0),
        compute_seconds: (after.compute_seconds - before.compute_seconds).max(0.0),
        worker_compute_seconds: worker_compute_delta(
            before.worker_compute_seconds,
            after.worker_compute_seconds,
        ),
        send_wait_seconds: (after.send_wait_seconds - before.send_wait_seconds).max(0.0),
    }
}

pub(super) fn worker_compute_delta(
    before: HashMap<usize, f64>,
    after: HashMap<usize, f64>,
) -> HashMap<usize, f64> {
    after
        .into_iter()
        .map(|(worker_id, seconds)| {
            let before_seconds = before.get(&worker_id).copied().unwrap_or_default();
            (worker_id, (seconds - before_seconds).max(0.0))
        })
        .collect()
}

pub(super) fn max_worker_compute_seconds(worker_seconds: &HashMap<usize, f64>) -> f64 {
    worker_seconds.values().copied().fold(0.0, f64::max)
}

pub(super) fn asr_delta(
    before: AsrStageStatsSnapshot,
    after: AsrStageStatsSnapshot,
) -> AsrStageStatsSnapshot {
    AsrStageStatsSnapshot {
        batches: after.batches.saturating_sub(before.batches),
        jobs: after.jobs.saturating_sub(before.jobs),
        segments: after.segments.saturating_sub(before.segments),
        audio_seconds: (after.audio_seconds - before.audio_seconds).max(0.0),
        wall_seconds: (after.wall_seconds - before.wall_seconds).max(0.0),
        encode_batches: after.encode_batches.saturating_sub(before.encode_batches),
        encode_segments: after.encode_segments.saturating_sub(before.encode_segments),
        encode_audio_seconds: (after.encode_audio_seconds - before.encode_audio_seconds).max(0.0),
        encode_seconds: (after.encode_seconds - before.encode_seconds).max(0.0),
        decode_batches: after.decode_batches.saturating_sub(before.decode_batches),
        decode_segments: after.decode_segments.saturating_sub(before.decode_segments),
        decode_audio_seconds: (after.decode_audio_seconds - before.decode_audio_seconds).max(0.0),
        decode_seconds: (after.decode_seconds - before.decode_seconds).max(0.0),
        audio_encoder_seconds: (after.audio_encoder_seconds - before.audio_encoder_seconds)
            .max(0.0),
        prefill_seconds: (after.prefill_seconds - before.prefill_seconds).max(0.0),
        prefill_tokens: after.prefill_tokens.saturating_sub(before.prefill_tokens),
        decode_model_seconds: (after.decode_model_seconds - before.decode_model_seconds).max(0.0),
        decode_tokens: after.decode_tokens.saturating_sub(before.decode_tokens),
        decode_steps: after.decode_steps.saturating_sub(before.decode_steps),
    }
}

pub(super) fn log_component_throughput_window(
    service_id: &str,
    window_seconds: f64,
    loader: LoaderStageStatsSnapshot,
    vad: VadStageStatsSnapshot,
    asr: AsrStageStatsSnapshot,
) {
    let decode_audio = asr.decode_audio_seconds;
    let loader_elapsed = max_worker_compute_seconds(&loader.worker_compute_seconds);
    let vad_elapsed = max_worker_compute_seconds(&vad.worker_compute_seconds);
    let table = render_titled_table(
        format!("Service {service_id} Perf | window={window_seconds:.1}s"),
        ["stage", "count", "audio", "elapsed", "speed", "detail"],
        [
            vec![
                "load".to_string(),
                format!("job={}", loader.jobs),
                format_seconds(loader.audio_seconds),
                format_seconds(loader_elapsed),
                format_speed(loader.audio_seconds, loader_elapsed),
                format!("bad={}", loader.failed),
            ],
            vec![
                "vad".to_string(),
                format!("job={}", vad.jobs),
                format_seconds(vad.audio_seconds),
                format_seconds(vad_elapsed),
                format_speed(vad.audio_seconds, vad_elapsed),
                format!(
                    "speech={} ratio={} seg={} bad={} no_speech={}",
                    format_seconds(vad.speech_seconds),
                    format_percent(vad.speech_seconds, vad.audio_seconds),
                    vad.segments,
                    vad.failed,
                    vad.no_speech
                ),
            ],
            vec![
                "encode".to_string(),
                format!("bat={}", asr.encode_batches),
                format_seconds(asr.encode_audio_seconds),
                format_seconds(asr.encode_seconds),
                format_speed(asr.encode_audio_seconds, asr.encode_seconds),
                format!("seg={}", asr.encode_segments),
            ],
            vec![
                "prefill".to_string(),
                format!("bat={}", asr.decode_batches),
                format_seconds(decode_audio),
                format_seconds(asr.prefill_seconds),
                format_speed(decode_audio, asr.prefill_seconds),
                format!(
                    "tok={} tok/s={} seg={}",
                    asr.prefill_tokens,
                    format_rate(asr.prefill_tokens, asr.prefill_seconds),
                    asr.decode_segments
                ),
            ],
            vec![
                "decode".to_string(),
                format!("bat={}", asr.decode_batches),
                format_seconds(decode_audio),
                format_seconds(asr.decode_model_seconds),
                format_speed(decode_audio, asr.decode_model_seconds),
                format!(
                    "tok={} tok/s={} step={} seg={}",
                    asr.decode_tokens,
                    format_rate(asr.decode_tokens, asr.decode_model_seconds),
                    asr.decode_steps,
                    asr.decode_segments
                ),
            ],
        ],
    );

    tracing::info!(
        target: "vasr_transcribe::pipeline",
        "\n{}",
        table
    );
}

pub(super) fn format_seconds(seconds: f64) -> String {
    format!("{seconds:.2}s")
}

pub(super) fn format_speed(audio_seconds: f64, compute_seconds: f64) -> String {
    format!("{:.1}x", audio_seconds / compute_seconds.max(f64::EPSILON))
}

pub(super) fn format_percent(numerator: f64, denominator: f64) -> String {
    format!("{:.1}%", numerator * 100.0 / denominator.max(f64::EPSILON))
}

pub(super) fn format_rate(count: usize, seconds: f64) -> String {
    format!("{:.0}", count as f64 / seconds.max(f64::EPSILON))
}

pub(super) fn log_loader_summary(stats: &LoaderStageStats) {
    let metrics = stage_metrics(stats.audio_seconds, stats.wall_seconds);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "load | jobs={} failed={} audio={:.1}s wall={:.1}s compute={:.1}s speed={:.1}x",
        stats.jobs,
        stats.failed,
        stats.audio_seconds,
        stats.wall_seconds,
        stats.compute_seconds,
        metrics.speedup
    );
}

pub(super) fn log_vad_summary(stats: &VadStageStats) {
    let metrics = stage_metrics(stats.audio_seconds, stats.wall_seconds);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "vad | jobs={} failed={} no_speech={} segments={} speech={:.1}s audio={:.1}s wall={:.1}s compute={:.1}s send_wait={:.1}s speed={:.1}x",
        stats.jobs,
        stats.failed,
        stats.no_speech,
        stats.segments,
        stats.speech_seconds,
        stats.audio_seconds,
        stats.wall_seconds,
        stats.compute_seconds,
        stats.send_wait_seconds,
        metrics.speedup
    );
}

pub(super) fn timing_us_to_seconds(us: u64) -> f64 {
    us as f64 / 1_000_000.0
}

pub(super) fn log_pipeline(
    job_count: usize,
    outcomes: &[TranscribeItemOutcome],
    wall_seconds: f64,
) {
    let audio_seconds: f64 = outcomes.iter().map(|outcome| outcome.audio_seconds).sum();
    let bad = outcomes
        .iter()
        .filter(|outcome| outcome.result.is_err())
        .count();
    let metrics = stage_metrics(audio_seconds, wall_seconds);
    for outcome in outcomes {
        if let Err(error) = &outcome.result {
            tracing::error!(
                target: "vasr_transcribe::pipeline",
                index = outcome.index,
                component = outcome.bad_component.unwrap_or("unknown"),
                error = %error,
                audio_s = outcome.audio_seconds,
                "Pipeline job failed"
            );
        }
    }
    tracing::info!(
        target: "vasr_transcribe::pipeline",
        "pipeline | batch={} | returned={} | audio={:.2}s | spent={:.2}s | speed={:.2}x | rtf={:.3} | bad={}",
        job_count,
        outcomes.len(),
        audio_seconds,
        wall_seconds,
        metrics.speedup,
        metrics.rtf,
        bad
    );
}
