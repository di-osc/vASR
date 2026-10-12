use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

use asr_data::{
    Audio, AudioChannel, AudioEncoding, AudioEvent, AudioFormat, AudioInfo, AudioSource, Sentence,
    Speech, Timeline, Transcription,
};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, routing::post};
use vasr::{AsrOptions, AudioLoader, OfflineAsrModel as AsrModel};
use vasr_vad::{VadModel, VadOptions};

use crate::pipeline::{
    GlobalAsyncTranscribePipeline, PipelineMemorySnapshot, PipelineWorkerOptions,
    SourceReservation, StageSpan, TranscribeInput, TranscribeItemSpans, TranscribeItemTiming,
};
use crate::protocol::{
    InferenceData, InferencePerformance, SentenceResult, TextResult, TimingBreakdown,
    TranscribeRequest,
};

const TRANSCRIBE_BODY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

pub struct TranscribeService {
    pub service_id: String,
    pub pipeline: Arc<GlobalAsyncTranscribePipeline>,
    pub active_requests: AtomicUsize,
    #[cfg(feature = "cuda-graph")]
    pub baseline_cuda_used_bytes: Option<usize>,
}

pub fn transcribe_router(service: Arc<TranscribeService>) -> Router {
    Router::new()
        .route("/transcribe", post(handle_transcribe))
        .layer(DefaultBodyLimit::max(TRANSCRIBE_BODY_LIMIT_BYTES))
        .layer(middleware::from_fn_with_state(
            service.clone(),
            admit_transcribe,
        ))
        .with_state(service)
}

#[derive(Clone)]
struct AdmittedRequest {
    request_id: String,
    source_memory: Arc<SourceReservation>,
}
struct ActiveRequestGuard {
    service: Arc<TranscribeService>,
    request_id: String,
}
impl ActiveRequestGuard {
    fn new(service: Arc<TranscribeService>, request_id: String) -> Self {
        service.active_requests.fetch_add(1, Ordering::AcqRel);
        Self {
            service,
            request_id,
        }
    }
}
impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        let active_after = self.service.active_requests.fetch_sub(1, Ordering::AcqRel) - 1;
        #[cfg(feature = "cuda-graph")]
        {
            let service = self.service.clone();
            let request_id = self.request_id.clone();
            // CUDA workspace trimming may wait for a forward. Keep it off the
            // async worker on both response completion and cancellation.
            tokio::task::spawn_blocking(move || {
                maybe_trim_cuda_after_request(&request_id, &service, active_after)
            });
        }
        #[cfg(not(feature = "cuda-graph"))]
        let _ = (active_after, &self.request_id);
    }
}
async fn admit_transcribe(
    State(service): State<Arc<TranscribeService>>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    let _request_permit = match service.pipeline.admit_request().await {
        Ok(permit) => permit,
        Err(error) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
        }
    };
    // Acquire before Json/body extraction. Requests without a length reserve
    // the body limit up front; neither waiting case materializes base64.
    let bytes = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(TRANSCRIBE_BODY_LIMIT_BYTES)
        .min(TRANSCRIBE_BODY_LIMIT_BYTES);
    let source_memory = match service.pipeline.reserve_source(bytes).await {
        Ok(permit) => permit,
        Err(error) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
        }
    };
    let request_id = short_request_id();
    let _active = ActiveRequestGuard::new(service, request_id.clone());
    request.extensions_mut().insert(AdmittedRequest {
        request_id,
        source_memory,
    });
    next.run(request).await
}

async fn handle_transcribe(
    axum::extract::State(service): axum::extract::State<Arc<TranscribeService>>,
    axum::extract::Extension(admission): axum::extract::Extension<AdmittedRequest>,
    Json(request): Json<TranscribeRequest>,
) -> Response {
    let start = Instant::now();
    let request_id = admission.request_id;
    let q8_fast_path_start = vasr::models::qwen3asr::model::q8_fast_path_stats();
    let request_items = request.inputs.len();
    let mut inputs = Vec::new();
    let mut data_slots = vec![None; request_items];
    let mut request_metas = (0..request_items)
        .map(|_| RequestMeta::default())
        .collect::<Vec<_>>();
    let mut request_stats = RequestStats::default();
    let mut start_stats = RequestStartStats::default();

    for (index, input) in request.inputs.into_iter().enumerate() {
        let service_id = service.service_id.clone();
        let audio_id = input
            .id
            .unwrap_or_else(|| format!("{}-{index}", request_id));
        let mono = input.mono.unwrap_or(false);
        if mono {
            start_stats.mono_true += 1;
        } else {
            start_stats.mono_false += 1;
        }
        let (source, source_error) = if let Some(url) = input.url {
            start_stats.url += 1;
            (AudioSource::Url(url), None)
        } else if let Some(b64) = input.b64_str {
            start_stats.base64 += 1;
            (AudioSource::Base64(b64), None)
        } else if let Some(modelscope) = input.modelscope {
            start_stats.modelscope += 1;
            match AudioSource::from_modelscope(
                modelscope.repo_id,
                modelscope.file_path,
                modelscope.revision.as_deref(),
            ) {
                Ok(source) => (source, None),
                Err(error) => (
                    AudioSource::from_encoded_bytes(Vec::new()),
                    Some(format!("非法的 ModelScope 音频来源: {error}")),
                ),
            }
        } else {
            start_stats.missing += 1;
            (
                AudioSource::from_encoded_bytes(Vec::new()),
                Some("必须提供 url、b64_str 或 modelscope 音频来源".to_string()),
            )
        };

        let info = match input.info.filter(valid_audio_info) {
            Some(info) => info,
            None if source_error.is_some() => fallback_audio_info(),
            None => match source.aprobe().await {
                Ok(info) => info,
                Err(error) => {
                    tracing::warn!(
                        target: "vasr_transcribe::serve",
                        audio_id,
                        error = %error,
                        "could not probe audio metadata; using fallback AudioInfo"
                    );
                    fallback_audio_info()
                }
            },
        };
        let mut audio = Audio::with_id_from_info(audio_id, source.clone(), &info);
        crate::protocol::clear_legacy_transcribe_metadata(&mut audio);
        if let Some(error) = source_error.as_ref() {
            data_slots[index] = Some(error_data(
                service_id.clone(),
                "loader",
                error.clone(),
                TranscribeItemTiming::default(),
                0.0,
            ));
        }
        request_metas[index] = RequestMeta {
            service_id,
            audio: Some(audio),
        };
        if source_error.is_some() {
            continue;
        }
        inputs.push(TranscribeInput {
            index,
            source,
            mono,
        });
    }
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{} {:<7} {:<5} | items={} src=url:{} base64:{} modelscope:{} missing:{} mono=true:{} false:{}",
        request_id,
        "request",
        "START",
        request_items,
        start_stats.url,
        start_stats.base64,
        start_stats.modelscope,
        start_stats.missing,
        start_stats.mono_true,
        start_stats.mono_false
    );

    // 每个请求独立提交并等待各自的条目完成；模型侧靠共享 Metal forward 锁保证
    // 设备执行安全，不再用全局请求锁把请求串行化，也不再拆成单条请求。
    let outcomes = service
        .pipeline
        .run_admitted(request_id.clone(), inputs, admission.source_memory)
        .await;
    let mut total_audio_duration_seconds = 0.0;
    let mut component_timing = TranscribeItemTiming::default();
    let mut active_spans = Vec::new();

    for outcome in outcomes {
        if outcome.channel == "mono" {
            request_stats.mono += 1;
        } else if outcome.channel == "left" {
            request_stats.stereo += 1;
        }
        request_stats.channels += 1;
        let Some(meta) = request_metas.get(outcome.index) else {
            continue;
        };
        collect_active_spans(outcome.spans, &mut active_spans);
        match outcome.result {
            Ok(timeline) => {
                total_audio_duration_seconds += outcome.audio_seconds;
                component_timing.add_scaled(outcome.timing, 1.0);
                let mut channel_data = InferenceData::from_timeline_channel(
                    meta.service_id.clone(),
                    &timeline,
                    outcome.channel,
                );
                fill_timing_fields(&mut channel_data, outcome.timing, outcome.audio_seconds);
                merge_channel_data(&mut data_slots[outcome.index], channel_data);
            }
            Err(error) => {
                tracing::error!(
                    target: "vasr_transcribe::serve",
                    "{} request BAD   | index={} channel={} component={} audio={:.1}s reason={}",
                    request_id,
                    outcome.index,
                    outcome.channel,
                    outcome.bad_component.unwrap_or("recognizer"),
                    outcome.audio_seconds,
                    error
                );
                component_timing.add_scaled(outcome.timing, 1.0);
                merge_channel_data(
                    &mut data_slots[outcome.index],
                    error_data(
                        meta.service_id.clone(),
                        outcome.bad_component.unwrap_or("recognizer"),
                        format!("{}: {}", outcome.channel, error),
                        outcome.timing,
                        outcome.audio_seconds,
                    ),
                )
            }
        }
    }
    let data = data_slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            let mut data = slot.unwrap_or_else(|| {
                error_data(
                    request_metas[index].service_id.clone(),
                    "loader",
                    "no transcribe outcome".to_string(),
                    TranscribeItemTiming::default(),
                    0.0,
                )
            });
            clean_call_boundary_noise(&mut data);
            data
        })
        .collect::<Vec<_>>();

    let wall = start.elapsed().as_secs_f64();
    let active = active_spans_seconds(active_spans).min(wall).max(0.0);
    let wait = (wall - active).max(0.0);
    let num_items = data.len();
    let bad = data.iter().filter(|item| item.is_bad).count();
    let performance = InferencePerformance {
        batch_wall_seconds: wall,
        num_items,
        throughput_items_per_second: if wall > 0.0 {
            num_items as f64 / wall
        } else {
            0.0
        },
        total_audio_duration_seconds,
        speedup: if wall > 0.0 {
            total_audio_duration_seconds / wall
        } else {
            0.0
        },
        rtf: if total_audio_duration_seconds > 0.0 {
            wall / total_audio_duration_seconds
        } else {
            0.0
        },
    };
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{} {:<7} {:<5} | items={} bad={} | audio={:>6.1}s ch=mono:{} stereo:{} total:{} | active={:>6.3}s queue={:>6.3}s wall={:>6.3}s | speed={:>6.1}x e2e={:>6.1}x",
        request_id,
        "request",
        "DONE",
        performance.num_items,
        bad,
        performance.total_audio_duration_seconds,
        request_stats.mono,
        request_stats.stereo,
        request_stats.channels,
        active,
        wait,
        performance.batch_wall_seconds,
        performance.total_audio_duration_seconds / active.max(f64::EPSILON),
        performance.total_audio_duration_seconds / performance.batch_wall_seconds.max(f64::EPSILON)
    );
    if let Some(snapshot) = collect_request_memory_debug(request_memory_debug_enabled(), {
        let pipeline = Arc::clone(&service.pipeline);
        move || pipeline.memory_snapshot()
    })
    .await
    {
        log_request_memory_debug(
            &request_id,
            &snapshot,
            q8_fast_path_start,
            #[cfg(feature = "cuda-graph")]
            service.baseline_cuda_used_bytes,
        );
    }
    let records = data.iter().zip(request_metas).map(|(inference, meta)| {
        let mut audio = meta.audio.unwrap_or_else(|| {
            Audio::with_id_from_info(
                "audio-unknown",
                AudioSource::from_encoded_bytes(Vec::new()),
                &fallback_audio_info(),
            )
        });
        repair_fallback_audio_info(&mut audio, inference);
        attach_inference_timelines(&mut audio, inference);
        if inference.is_bad {
            crate::protocol::mark_transcription_failed(
                &mut audio,
                inference.bad_component.as_deref().unwrap_or("transcriber"),
                inference
                    .bad_reason
                    .as_deref()
                    .unwrap_or("transcription failed"),
            );
        }
        crate::protocol::clear_legacy_transcribe_metadata(&mut audio);
        audio
    });
    audio_jsonl_response(records)
}

fn fallback_audio_info() -> AudioInfo {
    AudioInfo {
        sample_rate: 1_000,
        channels: 1,
        frame_count: 0,
        source_format: AudioFormat {
            encoding: AudioEncoding::Unknown,
            sample_rate: 1_000,
            channels: 1,
        },
    }
}

fn valid_audio_info(info: &AudioInfo) -> bool {
    info.sample_rate > 0
        && info.channels > 0
        && info.source_format.sample_rate > 0
        && info.source_format.channels > 0
}

fn repair_fallback_audio_info(audio: &mut Audio, inference: &InferenceData) {
    if audio.info.frame_count != 0 || inference.timings.audio <= 0.0 {
        return;
    }
    let stereo = !inference.text.left.is_empty() || !inference.text.right.is_empty();
    let channels = if stereo { 2 } else { 1 };
    let inferred_end_ms = inference
        .sentences
        .iter()
        .map(|sentence| sentence.end_time)
        .max()
        .unwrap_or_default();
    let duration_seconds =
        (inference.timings.audio / f64::from(channels)).max(inferred_end_ms as f64 / 1_000.0);
    audio.info.channels = channels;
    audio.info.source_format.channels = channels;
    audio.info.frame_count = (duration_seconds * f64::from(audio.info.sample_rate)).ceil() as u64;
    let duration_ms = audio.info.timeline_duration_ms();
    let existing_channels = audio.timelines().keys().copied().collect::<Vec<_>>();
    for channel in existing_channels {
        if let Ok(Some(timeline)) = audio.timeline_mut(channel) {
            timeline.duration = duration_ms;
        }
    }
    if stereo && inference.text.mono.is_empty() {
        let _ = audio.remove_timeline(AudioChannel::Mono);
    }
}

fn attach_inference_timelines(audio: &mut Audio, inference: &InferenceData) {
    let duration = audio.info.timeline_duration_ms();
    for (channel_name, channel) in [
        ("mono", AudioChannel::Mono),
        ("left", AudioChannel::Left),
        ("right", AudioChannel::Right),
    ] {
        let has_text = match channel_name {
            "left" => !inference.text.left.is_empty(),
            "right" => !inference.text.right.is_empty(),
            _ => !inference.text.mono.is_empty(),
        };
        let transcript = match channel_name {
            "left" => &inference.text.left,
            "right" => &inference.text.right,
            _ => &inference.text.mono,
        };
        let sentences = inference
            .sentences
            .iter()
            .filter(|sentence| sentence.channel == channel_name)
            .collect::<Vec<_>>();
        if !has_text && sentences.is_empty() {
            continue;
        }

        let mut timeline = Timeline::new(audio.id.clone(), duration);
        let mut annotation_failed = false;
        for item in &sentences {
            let start = item.start_time as usize;
            let end = item.end_time as usize;
            if end <= start {
                continue;
            }
            let sentence = Sentence::new(item.text.clone(), start, end);
            let transcription =
                Transcription::new(item.text.clone()).with_sentences(vec![sentence]);
            let speech = Speech::new().with_transcription(transcription);
            if let Err(error) = timeline.annotate_with(
                start,
                end,
                AudioEvent::from(speech),
                false,
                Some("vasr-transcribe"),
            ) {
                tracing::warn!(
                    target: "vasr_transcribe::serve",
                    audio_id = %audio.id,
                    channel = channel_name,
                    error = %error,
                    "could not attach sentence spans to Audio timeline; falling back to full transcript"
                );
                timeline.prediction.clear();
                annotation_failed = true;
                break;
            }
        }
        if has_text && (sentences.is_empty() || annotation_failed || timeline.prediction.is_empty())
        {
            let speech = Speech::new().with_transcription(Transcription::new(transcript.clone()));
            if let Err(error) = timeline.annotate_with(
                0,
                duration,
                AudioEvent::from(speech),
                false,
                Some("vasr-transcribe"),
            ) {
                tracing::warn!(
                    target: "vasr_transcribe::serve",
                    audio_id = %audio.id,
                    channel = channel_name,
                    error = %error,
                    "could not attach full transcript to Audio timeline"
                );
            }
        }
        match audio.ensure_timeline(channel, None) {
            Ok(slot) => *slot = timeline,
            Err(error) => tracing::warn!(
                target: "vasr_transcribe::serve",
                audio_id = %audio.id,
                channel = channel_name,
                error = %error,
                "could not create Audio timeline for transcription"
            ),
        }
    }
}

fn audio_jsonl_response(records: impl IntoIterator<Item = Audio>) -> Response {
    let mut body = Vec::new();
    for record in records {
        if let Err(error) = serde_json::to_writer(&mut body, &record) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to encode Audio JSONL response: {error}"),
            )
                .into_response();
        }
        body.push(b'\n');
    }
    Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "application/x-ndjson")
        .body(Body::from(body))
        .unwrap_or_else(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build Audio JSONL response: {error}"),
            )
                .into_response()
        })
}

#[cfg(feature = "cuda-graph")]
fn maybe_trim_cuda_after_request(
    request_id: &str,
    service: &TranscribeService,
    active_after: usize,
) {
    if let Some(trim) = service.pipeline.trim_cuda_mempool() {
        log_request_trim_debug(request_id, active_after, trim);
    }
}

fn request_memory_debug_enabled() -> bool {
    std::env::var_os("VASR_DEBUG_CUDA_MEM").is_some()
        || std::env::var_os("VASR_DEBUG_REQUEST_MEM").is_some()
}

async fn collect_request_memory_debug(
    enabled: bool,
    collect: impl FnOnce() -> PipelineMemorySnapshot + Send + 'static,
) -> Option<PipelineMemorySnapshot> {
    if !enabled {
        return None;
    }
    // A cache snapshot can wait for an in-flight GPU forward. Optional
    // diagnostics must never block a Tokio worker that feeds the pipeline.
    match tokio::task::spawn_blocking(collect).await {
        Ok(snapshot) => Some(snapshot),
        Err(error) => {
            tracing::warn!(target: "vasr_transcribe::serve", "request memory diagnostics unavailable: {error}");
            None
        }
    }
}

fn log_request_memory_debug(
    request_id: &str,
    #[cfg_attr(
        not(any(feature = "paged-attn", feature = "cuda-graph")),
        allow(unused_variables)
    )]
    snapshot: &PipelineMemorySnapshot,
    q8_fast_path_start: vasr::models::qwen3asr::model::Q8FastPathStats,
    #[cfg(feature = "cuda-graph")] baseline_cuda_used_bytes: Option<usize>,
) {
    if !request_memory_debug_enabled() {
        return;
    }
    #[cfg(feature = "paged-attn")]
    let kv = snapshot
        .paged_cache
        .as_ref()
        .map(|stats| {
            let usable_blocks = stats.num_blocks.saturating_sub(1);
            format!(
                "kv_free={}/{} usable kv_blocks={} kv_capacity={}tok",
                stats.free_blocks, usable_blocks, stats.num_blocks, stats.max_context_tokens
            )
        })
        .unwrap_or_else(|| "kv=n/a".to_string());
    #[cfg(not(feature = "paged-attn"))]
    let kv = "kv=n/a".to_string();

    #[cfg(all(feature = "paged-attn", feature = "cuda-graph"))]
    let graphs = snapshot
        .cuda_decode_graphs
        .map(|count| format!("cuda_graphs={count}"))
        .unwrap_or_else(|| "cuda_graphs=n/a".to_string());
    #[cfg(not(all(feature = "paged-attn", feature = "cuda-graph")))]
    let graphs = "cuda_graphs=n/a".to_string();

    #[cfg(feature = "cuda-graph")]
    let cuda = snapshot
        .cuda_memory
        .map(|memory| {
            let baseline = baseline_cuda_used_bytes
                .map(|baseline| {
                    let delta_mib =
                        (memory.used_bytes as i128 - baseline as i128) as f64 / (1024.0 * 1024.0);
                    format!(" cuda_delta={delta_mib:+.1}MiB")
                })
                .unwrap_or_default();
            let pool = cuda_pool_fields(memory);
            format!(
                "cuda_used={:.2}GiB cuda_free={:.2}GiB cuda_total={:.2}GiB{}{}",
                bytes_to_gib(memory.used_bytes),
                bytes_to_gib(memory.free_bytes),
                bytes_to_gib(memory.total_bytes),
                baseline,
                pool
            )
        })
        .unwrap_or_else(|| "cuda=n/a".to_string());
    #[cfg(not(feature = "cuda-graph"))]
    let cuda = "cuda=n/a".to_string();
    let q8 = vasr::models::qwen3asr::model::q8_fast_path_stats().saturating_sub(q8_fast_path_start);
    let q8_fast_path = format!(
        "q8_fast=plain_audio:{}calls/{}rows plain_prefill:{}calls/{}rows mmq_prefill:{}calls/{}rows plain_lm_head:{}calls/{}rows mmq_lm_head:{}calls/{}rows mmq_lm_head_skip:{}calls/{}rows prefill_next_id:{}calls/{}rows decode_next_id:{}calls/{}rows plain_decode:{}calls/{}rows mmq_decode:{}calls/{}rows fused_glu:{}calls/{}rows fused_act:{}calls/{}rows fused_qkv:{}calls/{}rows mmq_fused_qkv:{}calls/{}rows",
        q8.plain_audio_calls,
        q8.plain_audio_rows,
        q8.plain_prefill_calls,
        q8.plain_prefill_rows,
        q8.mmq_prefill_calls,
        q8.mmq_prefill_rows,
        q8.plain_lm_head_calls,
        q8.plain_lm_head_rows,
        q8.mmq_lm_head_calls,
        q8.mmq_lm_head_rows,
        q8.prefill_lm_head_mmq_skip_calls,
        q8.prefill_lm_head_mmq_skip_rows,
        q8.prefill_next_id_calls,
        q8.prefill_next_id_rows,
        q8.decode_next_id_calls,
        q8.decode_next_id_rows,
        q8.plain_decode_calls,
        q8.plain_decode_rows,
        q8.mmq_decode_calls,
        q8.mmq_decode_rows,
        q8.fused_glu_decode_calls,
        q8.fused_glu_decode_rows,
        q8.fused_act_calls,
        q8.fused_act_rows,
        q8.fused_qkv_calls,
        q8.fused_qkv_rows,
        q8.mmq_fused_qkv_calls,
        q8.mmq_fused_qkv_rows,
    );

    tracing::info!(
        target: "vasr_transcribe::serve",
        "{} request MEM   | {} | {} | {} | {}",
        request_id,
        cuda,
        kv,
        graphs,
        q8_fast_path,
    );
}

#[cfg(feature = "cuda-graph")]
fn log_request_trim_debug(
    request_id: &str,
    active_requests: usize,
    trim: crate::pipeline::PipelineCudaTrimSnapshot,
) {
    if !request_memory_debug_enabled() {
        return;
    }
    let before = trim
        .before
        .map(cuda_memory_fields)
        .unwrap_or_else(|| "before=n/a".to_string());
    let after = trim
        .after
        .map(cuda_memory_fields)
        .unwrap_or_else(|| "after=n/a".to_string());
    tracing::info!(
        target: "vasr_transcribe::serve",
        "{} request TRIM  | active_requests={} | {} | {}",
        request_id,
        active_requests,
        before,
        after,
    );
}

#[cfg(feature = "cuda-graph")]
fn cuda_memory_fields(memory: crate::pipeline::PipelineCudaMemorySnapshot) -> String {
    format!(
        "used={:.2}GiB free={:.2}GiB total={:.2}GiB{}",
        bytes_to_gib(memory.used_bytes),
        bytes_to_gib(memory.free_bytes),
        bytes_to_gib(memory.total_bytes),
        cuda_pool_fields(memory)
    )
}

#[cfg(feature = "cuda-graph")]
fn cuda_pool_fields(memory: crate::pipeline::PipelineCudaMemorySnapshot) -> String {
    let current = match (memory.pool_used_bytes, memory.pool_reserved_bytes) {
        (Some(used), Some(reserved)) => format!(
            " pool_current={:.2}/{:.2}GiB",
            bytes_to_gib(used),
            bytes_to_gib(reserved)
        ),
        (Some(used), None) => format!(" pool_current={:.2}/n/aGiB", bytes_to_gib(used)),
        (None, Some(reserved)) => {
            format!(" pool_current=n/a/{:.2}GiB", bytes_to_gib(reserved))
        }
        (None, None) => " pool_current=n/a".to_string(),
    };
    let default = match (
        memory.default_pool_used_bytes,
        memory.default_pool_reserved_bytes,
    ) {
        (Some(used), Some(reserved)) => format!(
            " pool_default={:.2}/{:.2}GiB",
            bytes_to_gib(used),
            bytes_to_gib(reserved)
        ),
        (Some(used), None) => format!(" pool_default={:.2}/n/aGiB", bytes_to_gib(used)),
        (None, Some(reserved)) => {
            format!(" pool_default=n/a/{:.2}GiB", bytes_to_gib(reserved))
        }
        (None, None) => " pool_default=n/a".to_string(),
    };
    format!("{current}{default}")
}

#[cfg(feature = "cuda-graph")]
fn bytes_to_gib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn collect_active_spans(spans: TranscribeItemSpans, out: &mut Vec<StageSpan>) {
    out.extend(
        [
            spans.loader,
            spans.vad,
            spans.audio_encoder,
            spans.prefill,
            spans.decode,
        ]
        .into_iter()
        .flatten(),
    );
}

fn active_spans_seconds(mut spans: Vec<StageSpan>) -> f64 {
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

#[derive(Debug, Clone, Copy, Default)]
struct RequestStats {
    mono: usize,
    stereo: usize,
    channels: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct RequestStartStats {
    url: usize,
    base64: usize,
    modelscope: usize,
    missing: usize,
    mono_true: usize,
    mono_false: usize,
}

#[derive(Debug, Default)]
struct RequestMeta {
    service_id: String,
    audio: Option<Audio>,
}

fn short_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..8].to_string()
}

fn merge_channel_data(slot: &mut Option<InferenceData>, channel_data: InferenceData) {
    let Some(existing) = slot else {
        *slot = Some(channel_data);
        return;
    };

    if !channel_data.text.mono.is_empty() {
        existing.text.mono = channel_data.text.mono;
    }
    if !channel_data.text.left.is_empty() {
        existing.text.left = channel_data.text.left;
    }
    if !channel_data.text.right.is_empty() {
        existing.text.right = channel_data.text.right;
    }
    let mut sentences = channel_data.sentences;
    existing.sentences.append(&mut sentences);
    existing
        .sentences
        .sort_by_key(|sentence| (sentence.start_time, sentence.channel.clone()));
    for (idx, sentence) in existing.sentences.iter_mut().enumerate() {
        sentence.sentence_index = idx;
    }
    if channel_data.is_bad {
        existing.is_bad = true;
        existing.bad_component = channel_data.bad_component;
        existing.bad_reason = match (&existing.bad_reason, channel_data.bad_reason) {
            (Some(old), Some(new)) => Some(format!("{old}; {new}")),
            (None, Some(new)) => Some(new),
            (old, None) => old.clone(),
        };
    }
    existing.spent_seconds = existing.spent_seconds.max(channel_data.spent_seconds);
    existing.spent_details.loader = existing
        .spent_details
        .loader
        .max(channel_data.spent_details.loader);
    existing.spent_details.detector = existing
        .spent_details
        .detector
        .max(channel_data.spent_details.detector);
    existing.spent_details.recognizer = existing
        .spent_details
        .recognizer
        .max(channel_data.spent_details.recognizer);
    existing.spent_details.sentencizer = existing
        .spent_details
        .sentencizer
        .max(channel_data.spent_details.sentencizer);
    existing.timings.total = existing.timings.total.max(channel_data.timings.total);
    existing.timings.loader = existing.timings.loader.max(channel_data.timings.loader);
    existing.timings.pipeline = existing.timings.pipeline.max(channel_data.timings.pipeline);
    existing.timings.audio += channel_data.timings.audio;
}

fn clean_call_boundary_noise(data: &mut InferenceData) {
    if data.is_bad || data.sentences.is_empty() {
        return;
    }

    let original_len = data.sentences.len();
    let mut cleaned = Vec::with_capacity(original_len);
    let mut changed = false;
    let mut start_index = 0usize;
    while start_index < data.sentences.len()
        && should_drop_leading_sentence(&data.sentences[start_index])
    {
        start_index += 1;
    }
    changed |= start_index > 0;

    for (relative_index, mut sentence) in data.sentences[start_index..].iter().cloned().enumerate()
    {
        if relative_index < 4 {
            let cleaned_text = remove_boundary_artifact_phrases(sentence.text.as_str());
            changed |= cleaned_text != sentence.text;
            sentence.text = cleaned_text;
            if should_drop_early_artifact_sentence(&sentence) {
                changed = true;
                continue;
            }
        }
        cleaned.push(sentence);
    }

    while cleaned
        .last()
        .is_some_and(should_drop_trailing_artifact_sentence)
    {
        cleaned.pop();
        changed = true;
    }

    if !changed {
        return;
    }

    for (idx, sentence) in cleaned.iter_mut().enumerate() {
        sentence.sentence_index = idx;
    }
    data.sentences = cleaned;
    data.text = rebuild_text_from_sentences(&data.sentences);
}

fn should_drop_leading_sentence(sentence: &SentenceResult) -> bool {
    if sentence.start_time > 30_000 || sentence.end_time > 22_000 {
        return false;
    }
    let text = compact_text(sentence.text.as_str());
    if text.is_empty() {
        return true;
    }
    let has_strong_anchor = contains_any(text.as_str(), STRONG_CALL_ANCHORS);
    let has_noise = contains_any(text.as_str(), BOUNDARY_NOISE_PHRASES);
    let mostly_filler = mostly_filler_text(text.as_str());
    let short_non_call = text.chars().count() <= 8 && !has_strong_anchor;

    (has_noise && !has_strong_anchor) || mostly_filler || short_non_call
}

fn should_drop_early_artifact_sentence(sentence: &SentenceResult) -> bool {
    let text = compact_text(sentence.text.as_str());
    text.is_empty()
        || (sentence.end_time <= 22_000 && contains_any(text.as_str(), BOUNDARY_NOISE_PHRASES))
}

fn should_drop_trailing_artifact_sentence(sentence: &SentenceResult) -> bool {
    if sentence.start_time <= 15_000 {
        return false;
    }
    let text = compact_text(sentence.text.as_str());
    matches!(text.as_str(), "老板" | "好的" | "嗯" | "啊")
        || contains_any(text.as_str(), TRAILING_ARTIFACT_PHRASES)
}

fn rebuild_text_from_sentences(sentences: &[SentenceResult]) -> TextResult {
    let mut text = TextResult::default();
    for sentence in sentences {
        match sentence.channel.as_str() {
            "left" => text.left.push_str(sentence.text.as_str()),
            "right" => text.right.push_str(sentence.text.as_str()),
            _ => text.mono.push_str(sentence.text.as_str()),
        }
    }
    text
}

fn remove_boundary_artifact_phrases(text: &str) -> String {
    let mut cleaned = text.to_string();
    for phrase in BOUNDARY_ARTIFACT_PHRASES {
        cleaned = cleaned.replace(phrase, "");
    }
    cleaned
}

fn compact_text(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_whitespace() && !is_common_punctuation(*ch))
        .collect()
}

fn is_common_punctuation(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。'
            | '？'
            | '！'
            | '、'
            | '；'
            | '：'
            | ','
            | '.'
            | '?'
            | '!'
            | ';'
            | ':'
            | '"'
            | '\''
            | '“'
            | '”'
            | '‘'
            | '’'
            | '('
            | ')'
            | '（'
            | '）'
    )
}

fn mostly_filler_text(text: &str) -> bool {
    let total = text.chars().count();
    if total == 0 {
        return true;
    }
    let filler = text
        .chars()
        .filter(|ch| CALL_FILLER_CHARS.contains(*ch))
        .count();
    filler >= total.saturating_mul(3).div_ceil(5)
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

const CALL_FILLER_CHARS: &str = "嗯啊呃哦哎喂唉呀哈好的是对";
const STRONG_CALL_ANCHORS: &[&str] = &[
    "你好",
    "您好",
    "五八",
    "家政",
    "保姆",
    "找工作",
    "接单",
    "阿姨",
];
const BOUNDARY_NOISE_PHRASES: &[&str] = &[
    "公众号",
    "摩托",
    "英语",
    "老公",
    "APP",
    "下载APP",
    "接视频",
    "资料下载",
    "高密度",
    "连接",
    "不要怀疑",
    "怀疑别人",
    "别说废话",
    "愿意",
    "一点到五",
    "比较啊正常",
    "今天晚上",
    "我明天下班",
    "我打这个电话",
    "非常",
    "调节一下",
    "把钥匙",
    "新进来的嘛",
    "做提成",
];
const BOUNDARY_ARTIFACT_PHRASES: &[&str] = &[
    "不要怀疑",
    "没有再见",
    "不要怀疑别人",
    "然后我让他发资料下载APP过来接视频",
    "资料下载APP过来接视频",
    "你有摩托就怕他",
    "我跟老公",
    "你说英语怎么过？听不懂了。你说，",
    "你说英语怎么过听不懂了你说",
    "高密度连接",
    "啊，那么我把钥匙了。",
    "啊那么我把钥匙了",
    "别说废话",
    "我把他登了公众号",
    "一点到五",
    "我今天晚上",
    "新进来的嘛，你做提成，你想先工作，你都是直接给。",
    "新进来的嘛你做提成你想先工作你都是直接给",
];
const TRAILING_ARTIFACT_PHRASES: &[&str] = &["新进来的嘛", "做提成"];

fn fill_timing_fields(data: &mut InferenceData, timing: TranscribeItemTiming, audio_seconds: f64) {
    let pipeline_seconds = timing.vad_seconds + timing.asr_seconds;
    data.timings = TimingBreakdown {
        total: timing.loader_seconds + pipeline_seconds,
        loader: timing.loader_seconds,
        pipeline: pipeline_seconds,
        audio: audio_seconds,
    };
    data.spent_seconds = data.timings.total;
    data.spent_details.loader = timing.loader_seconds;
    data.spent_details.detector = timing.vad_seconds;
    data.spent_details.recognizer = timing.asr_seconds;
}

fn error_data(
    service_id: String,
    component: &str,
    reason: String,
    timing: TranscribeItemTiming,
    audio_seconds: f64,
) -> InferenceData {
    let mut data = InferenceData {
        service_id,
        spent_seconds: 0.0,
        spent_details: Default::default(),
        timings: TimingBreakdown::default(),
        text: Default::default(),
        sentences: Vec::new(),
        is_bad: true,
        bad_reason: Some(reason),
        bad_component: Some(component.to_string()),
    };
    fill_timing_fields(&mut data, timing, audio_seconds);
    data
}

pub fn build_transcribe_service(
    pipeline: Arc<GlobalAsyncTranscribePipeline>,
) -> Arc<TranscribeService> {
    #[cfg(feature = "cuda-graph")]
    let baseline_cuda_used_bytes = pipeline
        .memory_snapshot()
        .cuda_memory
        .map(|memory| memory.used_bytes);
    let service_id = format!("transcribe-{}", uuid::Uuid::new_v4().simple());
    pipeline.set_service_id(service_id.clone());
    Arc::new(TranscribeService {
        service_id,
        pipeline,
        active_requests: AtomicUsize::new(0),
        #[cfg(feature = "cuda-graph")]
        baseline_cuda_used_bytes,
    })
}

pub fn build_transcribe_service_from_parts(
    vad: Option<Arc<dyn VadModel>>,
    asr: Arc<dyn AsrModel>,
    asr_options: AsrOptions,
    vad_options: VadOptions,
    worker_options: PipelineWorkerOptions,
) -> Arc<TranscribeService> {
    build_transcribe_service(Arc::new(GlobalAsyncTranscribePipeline::start(
        AudioLoader,
        vad,
        asr,
        asr_options,
        vad_options,
        worker_options,
    )))
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn disabled_memory_debug_does_not_collect_or_wait_for_cache() {
        let result = super::collect_request_memory_debug(false, || {
            panic!("disabled diagnostics must not acquire cache locks")
        })
        .await;
        assert!(result.is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn enabled_memory_debug_collects_off_the_async_worker() {
        let caller = std::thread::current().id();
        let result = super::collect_request_memory_debug(true, move || {
            assert_ne!(std::thread::current().id(), caller);
            super::PipelineMemorySnapshot::default()
        })
        .await;
        assert!(result.is_some());
    }
    use std::time::Instant;

    use super::{StageSpan, active_spans_seconds, clean_call_boundary_noise, merge_channel_data};
    use crate::protocol::{InferenceData, SentenceResult, TextResult};

    #[test]
    fn merge_channel_data_combines_left_and_right_text() {
        let mut slot = None;
        merge_channel_data(&mut slot, channel_data("left", "hello", 20));
        merge_channel_data(&mut slot, channel_data("right", "world", 10));

        let data = slot.expect("merged data");
        assert_eq!(data.text.mono, "");
        assert_eq!(data.text.left, "hello");
        assert_eq!(data.text.right, "world");
        assert_eq!(
            data.sentences
                .iter()
                .map(|sentence| (sentence.channel.as_str(), sentence.sentence_index))
                .collect::<Vec<_>>(),
            vec![("right", 0), ("left", 1)]
        );
    }

    #[test]
    fn clean_call_boundary_noise_drops_leading_crosstalk() {
        let mut data = channel_data_with_sentences(vec![
            sentence("嗯。", 920, 1840),
            sentence("我把他登了公众号。", 12_720, 14_870),
            sentence("喂，啊，你好。", 20_870, 23_080),
            sentence("哎，你好，我是小云老师，这个号码加不上。", 23_360, 39_780),
        ]);

        clean_call_boundary_noise(&mut data);

        assert_eq!(data.sentences.len(), 2);
        assert_eq!(data.sentences[0].sentence_index, 0);
        assert_eq!(data.sentences[0].text, "喂，啊，你好。");
        assert!(data.text.mono.starts_with("喂，啊，你好。"));
        assert!(!data.text.mono.contains("公众号"));
    }

    #[test]
    fn clean_call_boundary_noise_removes_artifact_phrase_inside_first_sentence() {
        let mut data = channel_data_with_sentences(vec![
            sentence(
                "喂，您好，我是五八到家的，请问您现在还在找工作吗？然后我让他发资料下载APP过来接视频。喂，喂，谁找工作啊？",
                250,
                13_930,
            ),
            sentence("我没有找工作。", 14_460, 21_450),
        ]);

        clean_call_boundary_noise(&mut data);

        assert_eq!(data.sentences.len(), 2);
        assert!(!data.text.mono.contains("资料下载APP"));
        assert!(data.text.mono.contains("请问您现在还在找工作吗？"));
    }

    #[test]
    fn active_spans_seconds_merges_overlaps() {
        let base = Instant::now();
        let spans = vec![
            StageSpan::new(base, base + std::time::Duration::from_secs(2)),
            StageSpan::new(
                base + std::time::Duration::from_secs(1),
                base + std::time::Duration::from_secs(3),
            ),
            StageSpan::new(
                base + std::time::Duration::from_secs(5),
                base + std::time::Duration::from_secs(6),
            ),
        ];

        assert_eq!(active_spans_seconds(spans), 4.0);
    }

    fn channel_data(channel: &str, text: &str, start_time: u64) -> InferenceData {
        let mut data = channel_data_with_sentences_for_channel(
            channel,
            vec![SentenceResult {
                text: text.to_string(),
                start_time,
                end_time: start_time + 10,
                duration: 10,
                channel: channel.to_string(),
                sentence_index: 0,
            }],
        );
        match channel {
            "left" => data.text.left = text.to_string(),
            "right" => data.text.right = text.to_string(),
            _ => data.text.mono = text.to_string(),
        }
        data
    }

    fn channel_data_with_sentences(sentences: Vec<SentenceResult>) -> InferenceData {
        channel_data_with_sentences_for_channel("mono", sentences)
    }

    fn channel_data_with_sentences_for_channel(
        channel: &str,
        sentences: Vec<SentenceResult>,
    ) -> InferenceData {
        let mono = sentences
            .iter()
            .map(|sentence| sentence.text.as_str())
            .collect::<String>();
        let mut text = TextResult::default();
        match channel {
            "left" => text.left = mono,
            "right" => text.right = mono,
            _ => text.mono = mono,
        }
        InferenceData {
            service_id: String::new(),
            spent_seconds: 0.0,
            spent_details: Default::default(),
            timings: Default::default(),
            text,
            sentences,
            is_bad: false,
            bad_reason: None,
            bad_component: None,
        }
    }

    fn sentence(text: &str, start_time: u64, end_time: u64) -> SentenceResult {
        SentenceResult {
            text: text.to_string(),
            start_time,
            end_time,
            duration: end_time - start_time,
            channel: "mono".to_string(),
            sentence_index: 0,
        }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use std::future::IntoFuture;
    use std::time::Duration;
    use tokio::io::AsyncWriteExt;
    struct EmptyAsr;
    impl AsrModel for EmptyAsr {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn transcribe(
            &self,
            audio: &[asr_data::Waveform],
            _: &AsrOptions,
        ) -> anyhow::Result<Vec<asr_data::Timeline>> {
            Ok(audio
                .iter()
                .map(|_| asr_data::Timeline::new("test", 0))
                .collect())
        }
    }
    fn service() -> Arc<TranscribeService> {
        build_transcribe_service(Arc::new(GlobalAsyncTranscribePipeline::start_with_limits(
            AudioLoader,
            None,
            Arc::new(EmptyAsr),
            AsrOptions::default(),
            VadOptions::default(),
            PipelineWorkerOptions::default(),
            crate::pipeline::PipelineLimits {
                active_slots: 1,
                safe_capacity: 1,
                source_bytes: 1024,
            },
        )))
    }
    #[tokio::test]
    async fn active_counter_guard_cleans_up_on_cancel_and_error() {
        let service = service();
        let guard = ActiveRequestGuard::new(service.clone(), "test".into());
        assert_eq!(service.active_requests.load(Ordering::Acquire), 1);
        drop(guard);
        assert_eq!(service.active_requests.load(Ordering::Acquire), 0);
    }
    #[tokio::test]
    async fn request_admission_precedes_json_body_extraction_and_disconnect_wakes_waiter() {
        let service = service();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server =
            tokio::spawn(axum::serve(listener, transcribe_router(service.clone())).into_future());
        let mut first = tokio::net::TcpStream::connect(addr).await.unwrap();
        first.write_all(b"POST /transcribe HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 512\r\n\r\n").await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_requests.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let client = reqwest::Client::new();
        let mut second = Box::pin(
            client
                .post(format!("http://{addr}/transcribe"))
                .json(&serde_json::json!({"inputs": []}))
                .send(),
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(30), &mut second)
                .await
                .is_err(),
            "a second handler parsed/answered while the first admitted body was still unread"
        );
        assert_eq!(service.active_requests.load(Ordering::Acquire), 1);
        drop(first);
        let response = tokio::time::timeout(Duration::from_secs(1), second)
            .await
            .unwrap()
            .unwrap();
        assert!(response.status().is_success());
        assert_eq!(service.active_requests.load(Ordering::Acquire), 0);
        // Invalid JSON also releases both admission and the RAII counter.
        let invalid = client
            .post(format!("http://{addr}/transcribe"))
            .header("Content-Type", "application/json")
            .body("invalid")
            .send()
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        assert_eq!(service.active_requests.load(Ordering::Acquire), 0);
        server.abort();
    }
}
