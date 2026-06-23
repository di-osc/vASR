use std::sync::Arc;
use std::time::Instant;

use axum::{Json, Router, routing::post};
use vasr_data::AudioSource;

use crate::pipeline::{GlobalAsyncTranscribePipeline, TranscribeInput};
use crate::protocol::{InferenceData, InferencePerformance, TranscribeRequest, TranscribeResponse};
use crate::runtime::{AsrOptions, VadOptions};

pub struct TranscribeService {
    pub pipeline: Arc<GlobalAsyncTranscribePipeline>,
}

pub fn transcribe_router(service: Arc<TranscribeService>) -> Router {
    Router::new()
        .route("/transcribe", post(handle_transcribe))
        .route("/inference", post(handle_transcribe))
        .with_state(service)
}

async fn handle_transcribe(
    axum::extract::State(service): axum::extract::State<Arc<TranscribeService>>,
    Json(request): Json<TranscribeRequest>,
) -> Json<TranscribeResponse> {
    let start = Instant::now();
    let mut inputs = Vec::new();
    let mut validation_errors = Vec::new();

    for (index, input) in request.inputs.into_iter().enumerate() {
        let source = if let Some(url) = input.url {
            AudioSource::Url(url)
        } else if let Some(b64) = input.b64_str {
            AudioSource::Base64(b64)
        } else {
            validation_errors.push(InferenceData {
                service_id: String::new(),
                spent_seconds: 0.0,
                spent_details: Default::default(),
                text: Default::default(),
                sentences: Vec::new(),
                is_bad: true,
                bad_reason: Some("必须提供 url 或 b64_str 之一".to_string()),
                bad_component: Some("loader".to_string()),
            });
            continue;
        };
        inputs.push(TranscribeInput { index, source });
    }

    let outcomes = service.pipeline.run(inputs).await;
    let mut data = validation_errors;
    let mut total_audio_duration_seconds = 0.0;

    for outcome in outcomes {
        match outcome.result {
            Ok(timeline) => {
                total_audio_duration_seconds += outcome.audio_seconds;
                data.push(InferenceData::from_timeline("", &timeline));
            }
            Err(error) => data.push(error_data(
                outcome.bad_component.unwrap_or("recognizer"),
                error.to_string(),
            )),
        }
    }

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

fn error_data(component: &str, reason: String) -> InferenceData {
    InferenceData {
        service_id: String::new(),
        spent_seconds: 0.0,
        spent_details: Default::default(),
        text: Default::default(),
        sentences: Vec::new(),
        is_bad: true,
        bad_reason: Some(reason),
        bad_component: Some(component.to_string()),
    }
}

pub fn build_transcribe_service(
    pipeline: Arc<GlobalAsyncTranscribePipeline>,
) -> Arc<TranscribeService> {
    Arc::new(TranscribeService { pipeline })
}

pub fn build_transcribe_service_from_parts(
    offline: Arc<crate::runtime::OfflinePipeline>,
    asr_options: AsrOptions,
    vad_options: VadOptions,
) -> Arc<TranscribeService> {
    build_transcribe_service(Arc::new(
        GlobalAsyncTranscribePipeline::start(
            vasr_data::AudioLoader,
            offline,
            asr_options,
            vad_options,
        ),
    ))
}
