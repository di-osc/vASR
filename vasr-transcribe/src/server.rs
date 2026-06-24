use std::sync::Arc;
use std::time::Instant;

use axum::{Json, Router, routing::post};
use axum::extract::DefaultBodyLimit;
use vasr_data::{AudioChannel, AudioLoadOptions, AudioLoader, AudioSource};

use crate::pipeline::{GlobalAsyncTranscribePipeline, TranscribeInput};
use crate::protocol::{
    InferenceData, InferencePerformance, TimingBreakdown, TranscribeRequest, TranscribeResponse,
};
use crate::runtime::{AsrOptions, VadOptions};

const TRANSCRIBE_BODY_LIMIT_BYTES: usize = 512 * 1024 * 1024;

pub struct TranscribeService {
    pub service_id: String,
    pub pipeline: Arc<GlobalAsyncTranscribePipeline>,
}

pub fn transcribe_router(service: Arc<TranscribeService>) -> Router {
    Router::new()
        .route("/transcribe", post(handle_transcribe))
        .route("/inference", post(handle_transcribe))
        .layer(DefaultBodyLimit::max(TRANSCRIBE_BODY_LIMIT_BYTES))
        .with_state(service)
}

async fn handle_transcribe(
    axum::extract::State(service): axum::extract::State<Arc<TranscribeService>>,
    Json(request): Json<TranscribeRequest>,
) -> Json<TranscribeResponse> {
    let start = Instant::now();
    let mut inputs = Vec::new();
    let mut routes = Vec::new();
    let mut data_slots = vec![None; request.inputs.len()];
    let mut request_metas = vec![RequestMeta::default(); request.inputs.len()];
    let loader = AudioLoader;
    let load_options = AudioLoadOptions::default();

    for (index, input) in request.inputs.into_iter().enumerate() {
        request_metas[index].service_id = service.service_id.clone();
        let source = if let Some(url) = input.url {
            AudioSource::Url(url)
        } else if let Some(b64) = input.b64_str {
            AudioSource::Base64(b64)
        } else {
            data_slots[index] = Some(error_data(
                request_metas[index].service_id.clone(),
                "loader",
                "必须提供 url 或 b64_str 之一".to_string(),
                request_metas[index].loader_seconds,
                0.0,
                0.0,
            ));
            continue;
        };
        let load_start = Instant::now();
        let waveform = match load_waveform(loader.clone(), source, load_options.clone()).await {
            Ok(waveform) => {
                request_metas[index].loader_seconds += load_start.elapsed().as_secs_f64();
                waveform
            }
            Err(error) => {
                request_metas[index].loader_seconds += load_start.elapsed().as_secs_f64();
                data_slots[index] = Some(error_data(
                    request_metas[index].service_id.clone(),
                    "loader",
                    error.to_string(),
                    request_metas[index].loader_seconds,
                    0.0,
                    0.0,
                ));
                continue;
            }
        };
        let channels = waveform.channels;
        if input.mono == Some(true) || channels <= 1 {
            let task_index = inputs.len();
            inputs.push(TranscribeInput {
                index: task_index,
                source: AudioSource::Waveform(waveform),
                channel: AudioChannel::Mono,
            });
            routes.push(TaskRoute {
                request_index: index,
                channel: "mono",
            });
        } else {
            let task_index = inputs.len();
            inputs.push(TranscribeInput {
                index: task_index,
                source: AudioSource::Waveform(waveform.clone()),
                channel: AudioChannel::Left,
            });
            routes.push(TaskRoute {
                request_index: index,
                channel: "left",
            });
            let task_index = inputs.len();
            inputs.push(TranscribeInput {
                index: task_index,
                source: AudioSource::Waveform(waveform),
                channel: AudioChannel::Right,
            });
            routes.push(TaskRoute {
                request_index: index,
                channel: "right",
            });
        }
    }

    let pipeline_start = Instant::now();
    let outcomes = service.pipeline.run(inputs).await;
    let pipeline_wall = pipeline_start.elapsed().as_secs_f64();
    let mut total_audio_duration_seconds = 0.0;

    for outcome in outcomes {
        let Some(route) = routes.get(outcome.index).copied() else {
            continue;
        };
        let meta = &request_metas[route.request_index];
        match outcome.result {
            Ok(timeline) => {
                total_audio_duration_seconds += outcome.audio_seconds;
                let mut channel_data = InferenceData::from_timeline_channel(
                    meta.service_id.clone(),
                    &timeline,
                    route.channel,
                );
                fill_timing_fields(
                    &mut channel_data,
                    meta.loader_seconds,
                    pipeline_wall,
                    outcome.audio_seconds,
                );
                merge_channel_data(&mut data_slots[route.request_index], channel_data);
            }
            Err(error) => merge_channel_data(
                &mut data_slots[route.request_index],
                error_data(
                    meta.service_id.clone(),
                    outcome.bad_component.unwrap_or("recognizer"),
                    format!("{}: {}", route.channel, error),
                    meta.loader_seconds,
                    pipeline_wall,
                    outcome.audio_seconds,
                ),
            ),
        }
    }
    let data = data_slots
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.unwrap_or_else(|| {
                error_data(
                    request_metas[index].service_id.clone(),
                    "loader",
                    "no transcribe outcome".to_string(),
                    request_metas[index].loader_seconds,
                    pipeline_wall,
                    0.0,
                )
            })
        })
        .collect::<Vec<_>>();

    let wall = start.elapsed().as_secs_f64();
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
        "request | items={} bad={} audio={:.1}s wall={:.3}s speed={:.1}x rtf={:.4}",
        performance.num_items,
        bad,
        performance.total_audio_duration_seconds,
        performance.batch_wall_seconds,
        performance.speedup,
        performance.rtf
    );
    Json(TranscribeResponse {
        data,
        inference_performance: performance,
    })
}

#[derive(Debug, Clone, Copy)]
struct TaskRoute {
    request_index: usize,
    channel: &'static str,
}

#[derive(Debug, Clone, Default)]
struct RequestMeta {
    service_id: String,
    loader_seconds: f64,
}

async fn load_waveform(
    loader: AudioLoader,
    source: AudioSource,
    load_options: AudioLoadOptions,
) -> anyhow::Result<vasr_data::Waveform> {
    tokio::task::spawn_blocking(move || loader.load(&source, &load_options))
        .await
        .map_err(|error| anyhow::anyhow!("loader join error: {error}"))?
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

fn fill_timing_fields(
    data: &mut InferenceData,
    loader_seconds: f64,
    pipeline_seconds: f64,
    audio_seconds: f64,
) {
    data.timings = TimingBreakdown {
        total: loader_seconds + pipeline_seconds,
        loader: loader_seconds,
        pipeline: pipeline_seconds,
        audio: audio_seconds,
    };
    data.spent_seconds = data.timings.total;
    data.spent_details.loader = loader_seconds;
    data.spent_details.recognizer = pipeline_seconds;
}

fn error_data(
    service_id: String,
    component: &str,
    reason: String,
    loader_seconds: f64,
    pipeline_seconds: f64,
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
    fill_timing_fields(&mut data, loader_seconds, pipeline_seconds, audio_seconds);
    data
}

#[cfg(test)]
mod tests {
    use super::merge_channel_data;
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

    fn channel_data(channel: &str, text: &str, start_time: u64) -> InferenceData {
        let mut data = InferenceData {
            service_id: String::new(),
            spent_seconds: 0.0,
            spent_details: Default::default(),
            text: TextResult::default(),
            sentences: vec![SentenceResult {
                text: text.to_string(),
                start_time,
                end_time: start_time + 10,
                duration: 10,
                channel: channel.to_string(),
                sentence_index: 0,
            }],
            is_bad: false,
            bad_reason: None,
            bad_component: None,
        };
        match channel {
            "left" => data.text.left = text.to_string(),
            "right" => data.text.right = text.to_string(),
            _ => data.text.mono = text.to_string(),
        }
        data
    }
}

pub fn build_transcribe_service(
    pipeline: Arc<GlobalAsyncTranscribePipeline>,
) -> Arc<TranscribeService> {
    Arc::new(TranscribeService {
        service_id: format!("transcribe-{}", uuid::Uuid::new_v4().simple()),
        pipeline,
    })
}

pub fn build_transcribe_service_from_parts(
    offline: Arc<crate::runtime::OfflinePipeline>,
    asr_options: AsrOptions,
    vad_options: VadOptions,
) -> Arc<TranscribeService> {
    build_transcribe_service(Arc::new(GlobalAsyncTranscribePipeline::start(
        vasr_data::AudioLoader,
        offline,
        asr_options,
        vad_options,
    )))
}
