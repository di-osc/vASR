use crate::protocol::{
    ClientRealtimeEvent, RealtimeError, ServerRealtimeEvent, SessionInfo, SessionUpdatePayload,
};
use crate::runtime::RealtimePipeline;
use asr_data::{AnnotationPayload, AnnotationStatus, AudioBytesStream};
use base64::Engine;
use futures_util::StreamExt;

use std::sync::Arc;

use axum::{
    Router,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    response::IntoResponse,
    routing::get,
};

pub struct RealtimeService {
    pub make_session: Arc<dyn Fn() -> anyhow::Result<RealtimeSession> + Send + Sync>,
}

pub fn realtime_router(service: Arc<RealtimeService>) -> Router {
    Router::new()
        .route("/v1/realtime", get(handle_realtime))
        .route("/api-ws/v1/realtime", get(handle_realtime))
        .with_state(service)
}

async fn handle_realtime(
    State(service): State<Arc<RealtimeService>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| async move {
        match (service.make_session)() {
            Ok(session) => run_socket(socket, session).await,
            Err(err) => {
                let mut socket = socket;
                let event = error_event("session_init", err.to_string());
                let _ = socket
                    .send(Message::Text(
                        serde_json::to_string(&event).unwrap_or_default().into(),
                    ))
                    .await;
            }
        }
    })
}

async fn run_socket(mut socket: WebSocket, mut session: RealtimeSession) {
    let session_id = format!("sess_{}", uuid::Uuid::new_v4().simple());
    let created = ServerRealtimeEvent::SessionCreated {
        event_id: event_id(),
        session: session.session_info(session_id),
    };
    let _ = socket
        .send(Message::Text(
            serde_json::to_string(&created).unwrap_or_default().into(),
        ))
        .await;

    while let Some(message) = socket.next().await {
        let Ok(message) = message else {
            break;
        };
        match message {
            Message::Text(text) => {
                let Ok(event) = serde_json::from_str::<ClientRealtimeEvent>(&text) else {
                    let err = error_event("invalid_request_error", "invalid realtime event");
                    let _ = socket
                        .send(Message::Text(
                            serde_json::to_string(&err).unwrap_or_default().into(),
                        ))
                        .await;
                    continue;
                };
                let events = match event {
                    ClientRealtimeEvent::SessionUpdate {
                        session: requested_session,
                        sample_rate,
                        input_audio_format,
                    } => {
                        let mut update = requested_session.unwrap_or_default();
                        update.sample_rate = update.sample_rate.or(sample_rate);
                        update.input_audio_format =
                            update.input_audio_format.or(input_audio_format);
                        session.apply_update(&update);
                        vec![ServerRealtimeEvent::SessionUpdated {
                            event_id: event_id(),
                            session: session.session_info(session.session_id.clone()),
                        }]
                    }
                    ClientRealtimeEvent::AudioAppend { audio } => {
                        session.append_base64_audio(&audio)
                    }
                    ClientRealtimeEvent::AudioCommit => session.commit(),
                    ClientRealtimeEvent::SessionFinish => session.finish(),
                };
                for event in events {
                    if socket
                        .send(Message::Text(
                            serde_json::to_string(&event).unwrap_or_default().into(),
                        ))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

pub struct RealtimeSession {
    stream: AudioBytesStream,
    pipeline: RealtimePipeline,
    session_id: String,
    input_audio_format: String,
    sample_rate: u32,
    input_audio_transcription: Option<serde_json::Value>,
    turn_detection: Option<serde_json::Value>,
    current_item_id: Option<String>,
    last_transcript: String,
}

impl RealtimeSession {
    pub fn new(sample_rate: u32, pipeline: RealtimePipeline) -> Self {
        Self {
            stream: AudioBytesStream::new(sample_rate, 1, 100),
            pipeline,
            session_id: String::new(),
            input_audio_format: "pcm".to_string(),
            sample_rate,
            input_audio_transcription: None,
            turn_detection: None,
            current_item_id: None,
            last_transcript: String::new(),
        }
    }

    fn session_info(&mut self, session_id: String) -> SessionInfo {
        if self.session_id.is_empty() {
            self.session_id = session_id;
        }
        SessionInfo {
            id: self.session_id.clone(),
            session_type: "realtime".to_string(),
            audio: serde_json::json!({
                "input_audio_format": self.input_audio_format,
                "sample_rate": self.sample_rate,
            }),
            vad: self
                .turn_detection
                .clone()
                .unwrap_or_else(|| serde_json::json!({})),
            asr: self.input_audio_transcription.clone().unwrap_or_else(
                || serde_json::json!({"language": null, "hotwords": [], "hotwords_id": null}),
            ),
            correct: serde_json::json!({"rule_id": null}),
            enhance: serde_json::json!({}),
            turn_analyzer: serde_json::json!({}),
        }
    }

    fn apply_update(&mut self, update: &SessionUpdatePayload) {
        if let Some(sample_rate) = update.sample_rate {
            self.sample_rate = sample_rate;
        }
        if let Some(format) = &update.input_audio_format {
            self.input_audio_format = format.clone();
        }
        if update.input_audio_transcription.is_some() {
            self.input_audio_transcription = update.input_audio_transcription.clone();
        }
        if update.turn_detection.is_some() {
            self.turn_detection = update.turn_detection.clone();
        }
    }

    pub fn append_base64_audio(&mut self, audio: &str) -> Vec<ServerRealtimeEvent> {
        let bytes = match base64::engine::general_purpose::STANDARD.decode(audio) {
            Ok(bytes) => bytes,
            Err(err) => {
                return vec![error_event("audio_decode_error", err.to_string())];
            }
        };
        let chunks = match self.stream.push(&bytes) {
            Ok(chunks) => chunks,
            Err(err) => {
                return vec![error_event("audio_buffer_error", err.to_string())];
            }
        };
        let mut events = Vec::new();
        for chunk in chunks.iter() {
            match self.pipeline.push_chunk(chunk) {
                Ok(annotations) => events.extend(events_from_annotations(
                    annotations,
                    &mut self.current_item_id,
                    &mut self.last_transcript,
                )),
                Err(err) => events.push(error_event("pipeline_error", err.to_string())),
            }
        }
        events
    }

    /// Flush bytes that do not yet fill a regular 100 ms chunk. The ASR stream
    /// remains open, so clients may append more audio after a commit.
    pub fn commit(&mut self) -> Vec<ServerRealtimeEvent> {
        let chunks = match self.stream.flush() {
            Ok(chunks) => chunks,
            Err(err) => return vec![error_event("audio_buffer_error", err.to_string())],
        };
        let mut events = Vec::new();
        for chunk in chunks.iter() {
            match self.pipeline.push_chunk(chunk) {
                Ok(annotations) => events.extend(events_from_annotations(
                    annotations,
                    &mut self.current_item_id,
                    &mut self.last_transcript,
                )),
                Err(err) => events.push(error_event("pipeline_error", err.to_string())),
            }
        }
        events
    }

    pub fn finish(&mut self) -> Vec<ServerRealtimeEvent> {
        let mut events = self.commit();
        if let Ok(annotations) = self.pipeline.finish() {
            events.extend(events_from_annotations(
                annotations,
                &mut self.current_item_id,
                &mut self.last_transcript,
            ));
        }
        events.push(ServerRealtimeEvent::SessionFinished {
            event_id: event_id(),
        });
        events
    }
}

fn events_from_annotations(
    annotations: Vec<asr_data::Annotation>,
    current_item_id: &mut Option<String>,
    last_transcript: &mut String,
) -> Vec<ServerRealtimeEvent> {
    annotations
        .into_iter()
        .filter_map(|annotation| match annotation.payload {
            AnnotationPayload::Speech if annotation.status == AnnotationStatus::Partial => {
                let item_id = current_item_id
                    .get_or_insert_with(|| format!("item_{}", uuid::Uuid::new_v4().simple()))
                    .clone();
                Some(ServerRealtimeEvent::SpeechStarted {
                    event_id: event_id(),
                    item_id,
                    audio_start_ms: annotation.range.start.0,
                })
            }
            AnnotationPayload::Speech if annotation.status == AnnotationStatus::Final => {
                let item_id = current_item_id
                    .get_or_insert_with(|| format!("item_{}", uuid::Uuid::new_v4().simple()))
                    .clone();
                Some(ServerRealtimeEvent::SpeechStopped {
                    event_id: event_id(),
                    item_id,
                    audio_end_ms: annotation.range.end.0,
                })
            }
            AnnotationPayload::Transcription(segment)
                if annotation.status == AnnotationStatus::Partial =>
            {
                let item_id = current_item_id
                    .get_or_insert_with(|| format!("item_{}", uuid::Uuid::new_v4().simple()))
                    .clone();
                let delta = segment
                    .text
                    .strip_prefix(last_transcript.as_str())
                    .unwrap_or_default()
                    .to_string();
                *last_transcript = segment.text.clone();
                Some(ServerRealtimeEvent::TranscriptionText {
                    event_id: event_id(),
                    item_id,
                    delta,
                    text: segment.text,
                    start_ms: Some(annotation.range.start.0),
                    end_ms: Some(annotation.range.end.0),
                    language: segment.language,
                    emotion: "UNKNOWN".to_string(),
                    speech_type: None,
                    itn: None,
                })
            }
            AnnotationPayload::Transcription(segment)
                if annotation.status == AnnotationStatus::Final =>
            {
                let item_id = current_item_id
                    .get_or_insert_with(|| format!("item_{}", uuid::Uuid::new_v4().simple()))
                    .clone();
                *current_item_id = None;
                last_transcript.clear();
                Some(ServerRealtimeEvent::TranscriptionCompleted {
                    event_id: event_id(),
                    item_id,
                    text: segment.text,
                    start_ms: Some(annotation.range.start.0),
                    end_ms: Some(annotation.range.end.0),
                    language: segment.language,
                    emotion: "UNKNOWN".to_string(),
                    speech_type: None,
                    itn: None,
                })
            }
            _ => None,
        })
        .collect()
}

fn event_id() -> String {
    format!("event_{}", uuid::Uuid::new_v4().simple())
}

fn error_event(code: &str, message: impl Into<String>) -> ServerRealtimeEvent {
    ServerRealtimeEvent::Error {
        event_id: event_id(),
        error: RealtimeError {
            error_type: "invalid_request_error".to_string(),
            code: code.to_string(),
            message: message.into(),
            param: None,
        },
    }
}
