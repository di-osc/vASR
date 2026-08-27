use crate::protocol::{
    ClientRealtimeEvent, RealtimeError, ServerRealtimeEvent, SessionInfo, SessionUpdatePayload,
    SessionUpdateWarning,
};
use crate::runtime::{AsrModel, AsrOptions, RealtimePipeline, VadModel, VadOptions};
use asr_data::{
    Annotation, AnnotationPayload, AnnotationStatus, AudioBytesStream, TimeRange, Waveform,
};
use base64::Engine;
use futures_util::StreamExt;
use serde_json::{Map, Value};
use vasr::OfflineAsrModel;

use std::collections::BTreeSet;
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
                let event = error_event("session_init", err.to_string(), None, None);
                let _ = send_event(&mut socket, &event).await;
            }
        }
    })
}

async fn send_event(socket: &mut WebSocket, event: &ServerRealtimeEvent) -> Result<(), ()> {
    socket
        .send(Message::Text(
            serde_json::to_string(event).unwrap_or_default().into(),
        ))
        .await
        .map_err(|_| ())
}

async fn run_socket(mut socket: WebSocket, mut session: RealtimeSession) {
    let session_id = format!("sess_{}", uuid::Uuid::new_v4().simple());
    let created = ServerRealtimeEvent::SessionCreated {
        event_id: event_id(),
        session: session.session_info(session_id),
    };
    if send_event(&mut socket, &created).await.is_err() {
        return;
    }

    while let Some(message) = socket.next().await {
        let Ok(message) = message else {
            break;
        };
        match message {
            Message::Text(text) => {
                let value = match serde_json::from_str::<Value>(&text) {
                    Ok(value) => value,
                    Err(_) => {
                        let event = error_event(
                            "invalid_request_error",
                            "invalid realtime event",
                            None,
                            None,
                        );
                        let _ = send_event(&mut socket, &event).await;
                        continue;
                    }
                };
                let request_event_id = value
                    .get("event_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let event_type = value.get("type").and_then(Value::as_str);
                if !matches!(
                    event_type,
                    Some(
                        "session.update"
                            | "input_audio_buffer.append"
                            | "input_audio_buffer.commit"
                            | "session.finish"
                    )
                ) {
                    let event = error_event(
                        "unsupported_event",
                        format!("Unsupported event type: {}", event_type.unwrap_or("None")),
                        request_event_id.as_deref(),
                        None,
                    );
                    let _ = send_event(&mut socket, &event).await;
                    continue;
                }
                let event = match serde_json::from_value::<ClientRealtimeEvent>(value) {
                    Ok(event) => event,
                    Err(err) => {
                        let event = error_event(
                            "invalid_request_error",
                            err.to_string(),
                            request_event_id.as_deref(),
                            None,
                        );
                        let _ = send_event(&mut socket, &event).await;
                        continue;
                    }
                };

                let events = match event {
                    ClientRealtimeEvent::SessionUpdate {
                        event_id: request_event_id,
                        session: requested_session,
                    } => handle_session_update(
                        &mut session,
                        requested_session,
                        request_event_id.as_deref(),
                    ),
                    ClientRealtimeEvent::AudioAppend {
                        event_id: request_event_id,
                        audio,
                    } => match audio.filter(|audio| !audio.is_empty()) {
                        Some(audio) => session
                            .append_base64_audio_with_event(&audio, request_event_id.as_deref()),
                        None => vec![error_event(
                            "invalid_audio",
                            "Missing audio payload",
                            request_event_id.as_deref(),
                            Some("audio"),
                        )],
                    },
                    ClientRealtimeEvent::AudioCommit {
                        event_id: request_event_id,
                    } => session.commit_with_event(request_event_id.as_deref()),
                    ClientRealtimeEvent::SessionFinish {
                        event_id: request_event_id,
                    } => session.finish_with_event(request_event_id.as_deref()),
                };
                for event in events {
                    if send_event(&mut socket, &event).await.is_err() {
                        return;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

fn handle_session_update(
    session: &mut RealtimeSession,
    requested_session: Option<Value>,
    request_event_id: Option<&str>,
) -> Vec<ServerRealtimeEvent> {
    let value = requested_session.unwrap_or_else(|| Value::Object(Map::new()));
    if !value.is_object() {
        return vec![error_event(
            "invalid_session_update",
            "session.update requires `session` to be an object.",
            request_event_id,
            Some("session"),
        )];
    }
    let update = match serde_json::from_value::<SessionUpdatePayload>(value) {
        Ok(update) => update,
        Err(err) => {
            return vec![error_event(
                "invalid_session_update",
                err.to_string(),
                request_event_id,
                Some("session"),
            )];
        }
    };
    match session.apply_update(update) {
        Ok(warnings) => vec![ServerRealtimeEvent::SessionUpdated {
            event_id: event_id(),
            session: session.session_info(session.session_id.clone()),
            warnings,
        }],
        Err(err) => vec![error_event(
            "invalid_session_update",
            err.message,
            request_event_id,
            err.param.as_deref(),
        )],
    }
}

pub struct RealtimeSession {
    stream: AudioBytesStream,
    pipeline: Option<RealtimePipeline>,
    vad_model: Option<Arc<dyn VadModel>>,
    asr_model: Option<Arc<dyn AsrModel>>,
    offline_asr_model: Option<Arc<dyn OfflineAsrModel>>,
    vad_options: VadOptions,
    asr_options: AsrOptions,
    offline_asr_options: AsrOptions,
    session_id: String,
    input_audio_format: String,
    sample_rate: u32,
    session_language: Option<String>,
    hotwords: Vec<String>,
    hotwords_id: Option<String>,
    correct_rule_id: Option<String>,
    current_item_id: Option<String>,
    last_transcript: String,
    turn_has_audio: bool,
    asr_turn_has_audio: bool,
    asr_turn_samples: Vec<f32>,
}

impl RealtimeSession {
    pub fn new(sample_rate: u32, pipeline: RealtimePipeline) -> Self {
        Self {
            stream: AudioBytesStream::new(sample_rate, 1, 100),
            pipeline: Some(pipeline),
            vad_model: None,
            asr_model: None,
            offline_asr_model: None,
            vad_options: VadOptions::default(),
            asr_options: AsrOptions::default(),
            offline_asr_options: AsrOptions::default(),
            session_id: String::new(),
            input_audio_format: "pcm".to_string(),
            sample_rate,
            session_language: None,
            hotwords: Vec::new(),
            hotwords_id: None,
            correct_rule_id: None,
            current_item_id: None,
            last_transcript: String::new(),
            turn_has_audio: false,
            asr_turn_has_audio: false,
            asr_turn_samples: Vec::new(),
        }
    }

    pub(crate) fn from_models(
        sample_rate: u32,
        vad_model: Arc<dyn VadModel>,
        asr_model: Arc<dyn AsrModel>,
        vad_options: VadOptions,
        asr_options: AsrOptions,
    ) -> anyhow::Result<Self> {
        Self::from_two_pass_models(
            sample_rate,
            vad_model,
            asr_model,
            None,
            vad_options,
            asr_options.clone(),
            asr_options,
        )
    }

    pub(crate) fn from_two_pass_models(
        sample_rate: u32,
        vad_model: Arc<dyn VadModel>,
        asr_model: Arc<dyn AsrModel>,
        offline_asr_model: Option<Arc<dyn OfflineAsrModel>>,
        vad_options: VadOptions,
        asr_options: AsrOptions,
        offline_asr_options: AsrOptions,
    ) -> anyhow::Result<Self> {
        let vad = vad_model.new_stream(&vad_options)?;
        let asr = asr_model.new_stream(&asr_options)?;
        Ok(Self {
            stream: AudioBytesStream::new(sample_rate, 1, 100),
            pipeline: Some(RealtimePipeline { vad, asr }),
            vad_model: Some(vad_model),
            asr_model: Some(asr_model),
            offline_asr_model,
            vad_options,
            asr_options,
            offline_asr_options,
            session_id: String::new(),
            input_audio_format: "pcm".to_string(),
            sample_rate,
            session_language: None,
            hotwords: Vec::new(),
            hotwords_id: None,
            correct_rule_id: None,
            current_item_id: None,
            last_transcript: String::new(),
            turn_has_audio: false,
            asr_turn_has_audio: false,
            asr_turn_samples: Vec::new(),
        })
    }

    fn session_info(&mut self, session_id: String) -> SessionInfo {
        if self.session_id.is_empty() {
            self.session_id = session_id;
        }
        SessionInfo {
            id: self.session_id.clone(),
            session_type: "realtime".to_string(),
            audio: serde_json::json!({
                "format": self.input_audio_format,
                "sample_rate": self.sample_rate,
            }),
            vad: serde_json::json!({
                "threshold": self.vad_options.threshold,
                "silence_duration_ms": self.vad_options.min_silence_ms,
                "prefix_padding_ms": self.vad_options.pad_ms,
            }),
            asr: serde_json::json!({
                "language": self.session_language,
                "hotwords": self.hotwords,
                "hotwords_id": self.hotwords_id,
            }),
            correct: serde_json::json!({"rule_id": self.correct_rule_id}),
            enhance: serde_json::json!({}),
            turn_analyzer: serde_json::json!({}),
        }
    }

    fn apply_update(
        &mut self,
        update: SessionUpdatePayload,
    ) -> Result<Vec<SessionUpdateWarning>, SessionUpdateError> {
        for legacy in [
            "input_audio_format",
            "sample_rate",
            "input_audio_transcription",
            "turn_detection",
            "speech_gate",
        ] {
            if update.extra.contains_key(legacy) {
                return Err(SessionUpdateError::new(
                    format!(
                        "session.update `{legacy}` is not supported. Use session.audio/asr/correct/vad/enhance/turn_analyzer."
                    ),
                    format!("session.{legacy}"),
                ));
            }
        }

        let audio = section_object(update.audio, "audio")?;
        let asr = section_object(update.asr, "asr")?;
        let correct = section_object(update.correct, "correct")?;
        let vad = section_object(update.vad, "vad")?;
        let enhance = section_object(update.enhance, "enhance")?;
        let turn_analyzer = section_object(update.turn_analyzer, "turn_analyzer")?;
        let mut warnings = Vec::new();
        let mut audio_changed = false;
        let mut pipeline_changed = false;

        if let Some(audio) = audio {
            if let Some(format) = audio.get("format") {
                self.input_audio_format = format
                    .as_str()
                    .ok_or_else(|| {
                        SessionUpdateError::new(
                            "session.update `audio.format` must be a string.",
                            "session.audio.format",
                        )
                    })?
                    .to_string();
                audio_changed = true;
            }
            if let Some(sample_rate) = audio.get("sample_rate") {
                let sample_rate = json_u64(sample_rate, "session.audio.sample_rate")
                    .ok()
                    .and_then(|value| u32::try_from(value).ok())
                    .filter(|value| *value > 0)
                    .ok_or_else(|| {
                        SessionUpdateError::new(
                            "session.update `audio.sample_rate` must be a positive integer.",
                            "session.audio.sample_rate",
                        )
                    })?;
                self.sample_rate = sample_rate;
                audio_changed = true;
            }
        }

        if let Some(asr) = asr {
            self.session_language = None;
            self.hotwords.clear();
            self.hotwords_id = None;
            self.asr_options.language = None;
            self.asr_options.context.clear();
            self.offline_asr_options.language = None;
            self.offline_asr_options.context.clear();
            pipeline_changed = true;
            for key in asr.keys() {
                if !matches!(key.as_str(), "language" | "hotwords" | "hotwords_id") {
                    warnings.push(ignored_parameter_warning("asr", key));
                }
            }
            if let Some(language) = asr.get("language") {
                self.session_language = if language.is_null() {
                    None
                } else {
                    Some(
                        language
                            .as_str()
                            .ok_or_else(|| {
                                SessionUpdateError::new(
                                    "session.update `asr.language` must be a string or null.",
                                    "session.asr.language",
                                )
                            })?
                            .to_string(),
                    )
                };
                self.asr_options.language = self.session_language.clone();
                self.offline_asr_options.language = self.session_language.clone();
                pipeline_changed = true;
            }
            if let Some(hotwords) = asr.get("hotwords") {
                self.hotwords = normalize_hotwords(hotwords);
                self.asr_options.context = self.hotwords.join(", ");
                self.offline_asr_options.context = self.asr_options.context.clone();
                pipeline_changed = true;
            }
            if let Some(hotwords_id) = asr.get("hotwords_id") {
                self.hotwords_id = normalized_optional_string(hotwords_id);
            }
        }

        if let Some(correct) = correct {
            self.correct_rule_id = None;
            for key in correct.keys() {
                if key != "rule_id" {
                    warnings.push(ignored_parameter_warning("correct", key));
                }
            }
            if let Some(rule_id) = correct.get("rule_id") {
                self.correct_rule_id = normalized_optional_string(rule_id);
            }
        }

        if let Some(vad) = vad {
            for key in vad.keys() {
                if !matches!(
                    key.as_str(),
                    "threshold"
                        | "speech_noise_thres"
                        | "silence_duration_ms"
                        | "max_end_silence_time"
                        | "prefix_padding_ms"
                ) {
                    warnings.push(ignored_parameter_warning("vad", key));
                }
            }
            if let Some(value) = vad
                .get("threshold")
                .or_else(|| vad.get("speech_noise_thres"))
            {
                let threshold = json_f32(value, "session.vad.threshold")?;
                if !(0.0..=1.0).contains(&threshold) {
                    return Err(SessionUpdateError::new(
                        "session.update `vad.threshold` must be in [0.0, 1.0].",
                        "session.vad.threshold",
                    ));
                }
                self.vad_options.threshold = threshold;
                pipeline_changed = true;
            }
            if let Some(value) = vad
                .get("silence_duration_ms")
                .or_else(|| vad.get("max_end_silence_time"))
            {
                self.vad_options.min_silence_ms =
                    json_u64(value, "session.vad.silence_duration_ms")?;
                pipeline_changed = true;
            }
            if let Some(value) = vad.get("prefix_padding_ms") {
                self.vad_options.pad_ms = json_u64(value, "session.vad.prefix_padding_ms")?;
                pipeline_changed = true;
            }
        }

        if let Some(enhance) = enhance {
            for (key, value) in &enhance {
                if key == "enhancement_strength" {
                    let strength = json_f32(value, "session.enhance.enhancement_strength")?;
                    if !(0.0..=1.0).contains(&strength) {
                        return Err(SessionUpdateError::new(
                            "session.update `enhance.enhancement_strength` must be in [0.0, 1.0].",
                            "session.enhance.enhancement_strength",
                        ));
                    }
                } else {
                    warnings.push(ignored_parameter_warning("enhance", key));
                }
            }
        }

        if let Some(turn_analyzer) = turn_analyzer {
            for key in turn_analyzer.keys() {
                if !matches!(key.as_str(), "type" | "@realtime_turn_analyzers") {
                    warnings.push(ignored_parameter_warning("turn_analyzer", key));
                }
            }
        }

        if !self.turn_has_audio {
            if audio_changed {
                self.stream = AudioBytesStream::new(self.sample_rate, 1, 100);
            }
            if pipeline_changed {
                self.rebuild_pipeline()
                    .map_err(|err| SessionUpdateError::new(err.to_string(), "session"))?;
            }
        }
        Ok(warnings)
    }

    pub fn append_base64_audio(&mut self, audio: &str) -> Vec<ServerRealtimeEvent> {
        self.append_base64_audio_with_event(audio, None)
    }

    fn append_base64_audio_with_event(
        &mut self,
        audio: &str,
        request_event_id: Option<&str>,
    ) -> Vec<ServerRealtimeEvent> {
        let bytes = match base64::engine::general_purpose::STANDARD.decode(audio) {
            Ok(bytes) => bytes,
            Err(_) => {
                return vec![error_event(
                    "invalid_audio",
                    "Failed to decode base64 audio",
                    request_event_id,
                    Some("audio"),
                )];
            }
        };
        if bytes.is_empty() {
            return vec![error_event(
                "invalid_audio",
                "Missing audio payload",
                request_event_id,
                Some("audio"),
            )];
        }
        self.turn_has_audio = true;
        let chunks = match self.stream.push(&bytes) {
            Ok(chunks) => chunks,
            Err(err) => {
                return vec![error_event(
                    "invalid_audio",
                    err.to_string(),
                    request_event_id,
                    Some("audio"),
                )];
            }
        };
        self.process_chunks(&chunks, true, request_event_id)
    }

    pub fn commit(&mut self) -> Vec<ServerRealtimeEvent> {
        self.commit_with_event(None)
    }

    fn commit_with_event(&mut self, request_event_id: Option<&str>) -> Vec<ServerRealtimeEvent> {
        if !self.turn_has_audio {
            return Vec::new();
        }
        let chunks = match self.stream.flush() {
            Ok(chunks) => chunks,
            Err(err) => {
                return vec![error_event(
                    "invalid_audio",
                    err.to_string(),
                    request_event_id,
                    Some("audio"),
                )];
            }
        };
        let mut events = self.process_chunks(&chunks, false, request_event_id);
        if let Some(pipeline) = self.pipeline.as_mut() {
            match pipeline.finish_vad() {
                Ok(annotations) => events.extend(events_from_annotations(
                    annotations,
                    &mut self.current_item_id,
                    &mut self.last_transcript,
                )),
                Err(err) => events.push(error_event(
                    "pipeline_error",
                    err.to_string(),
                    request_event_id,
                    None,
                )),
            }
        }
        if self.asr_turn_has_audio {
            self.finish_asr_turn(&mut events, request_event_id);
        }
        self.current_item_id = None;
        self.last_transcript.clear();
        self.turn_has_audio = false;
        self.asr_turn_has_audio = false;
        self.asr_turn_samples.clear();
        self.stream = AudioBytesStream::new(self.sample_rate, 1, 100);
        if let Err(err) = self.rebuild_pipeline() {
            events.push(error_event(
                "pipeline_error",
                err.to_string(),
                request_event_id,
                None,
            ));
        }
        events
    }

    pub fn finish(&mut self) -> Vec<ServerRealtimeEvent> {
        self.finish_with_event(None)
    }

    fn finish_with_event(&mut self, request_event_id: Option<&str>) -> Vec<ServerRealtimeEvent> {
        let mut events = self.commit_with_event(request_event_id);
        events.push(ServerRealtimeEvent::SessionFinished {
            event_id: event_id(),
        });
        events
    }

    fn process_chunks(
        &mut self,
        chunks: &[asr_data::AudioChunk],
        finalize_on_vad: bool,
        request_event_id: Option<&str>,
    ) -> Vec<ServerRealtimeEvent> {
        let mut events = Vec::new();
        for chunk in chunks {
            self.asr_turn_has_audio = true;
            self.asr_turn_samples
                .extend_from_slice(&chunk.waveform.samples);
            let annotations = match self.pipeline.as_mut() {
                Some(pipeline) => match pipeline.push_chunk(chunk) {
                    Ok(annotations) => annotations,
                    Err(err) => {
                        events.push(error_event(
                            "pipeline_error",
                            err.to_string(),
                            request_event_id,
                            None,
                        ));
                        continue;
                    }
                },
                None => {
                    events.push(error_event(
                        "pipeline_error",
                        "realtime pipeline is unavailable",
                        request_event_id,
                        None,
                    ));
                    continue;
                }
            };
            let speech_stopped = annotations.iter().any(|annotation| {
                annotation.status == AnnotationStatus::Final
                    && matches!(&annotation.payload, AnnotationPayload::Speech)
            });
            events.extend(events_from_annotations(
                annotations,
                &mut self.current_item_id,
                &mut self.last_transcript,
            ));
            if finalize_on_vad && speech_stopped {
                self.finish_asr_turn(&mut events, request_event_id);
            }
        }
        events
    }

    fn finish_asr_turn(
        &mut self,
        events: &mut Vec<ServerRealtimeEvent>,
        request_event_id: Option<&str>,
    ) {
        if !self.asr_turn_has_audio {
            return;
        }
        let online_final = self
            .pipeline
            .as_mut()
            .map_or_else(|| Ok(Vec::new()), |pipeline| pipeline.finish_asr());
        let final_annotations = if self.offline_asr_model.is_some() {
            match self.transcribe_offline_turn() {
                Ok(annotations) if !annotations.is_empty() => Ok(annotations),
                Ok(_) => online_final,
                Err(err) => {
                    events.push(error_event(
                        "offline_asr_error",
                        err.to_string(),
                        request_event_id,
                        None,
                    ));
                    online_final
                }
            }
        } else {
            online_final
        };
        match final_annotations {
            Ok(annotations) => events.extend(events_from_annotations(
                annotations,
                &mut self.current_item_id,
                &mut self.last_transcript,
            )),
            Err(err) => events.push(error_event(
                "pipeline_error",
                err.to_string(),
                request_event_id,
                None,
            )),
        }
        self.asr_turn_has_audio = false;
        self.asr_turn_samples.clear();
        if let (Some(model), Some(pipeline)) = (&self.asr_model, self.pipeline.as_mut()) {
            match model.new_stream(&self.asr_options) {
                Ok(asr) => pipeline.asr = asr,
                Err(err) => events.push(error_event(
                    "pipeline_error",
                    err.to_string(),
                    request_event_id,
                    None,
                )),
            }
        }
    }

    fn transcribe_offline_turn(&self) -> anyhow::Result<Vec<Annotation>> {
        let Some(model) = &self.offline_asr_model else {
            return Ok(Vec::new());
        };
        if self.asr_turn_samples.is_empty() {
            return Ok(Vec::new());
        }
        let waveform = Waveform::new(self.asr_turn_samples.clone(), self.sample_rate);
        let timelines = model.transcribe(&[waveform], &self.offline_asr_options)?;
        Ok(timelines
            .into_iter()
            .flat_map(|timeline| timeline.annotations)
            .filter(|annotation| {
                annotation.status == AnnotationStatus::Final
                    && matches!(&annotation.payload, AnnotationPayload::Transcription(_))
            })
            .collect())
    }

    fn rebuild_pipeline(&mut self) -> anyhow::Result<()> {
        let (Some(vad_model), Some(asr_model)) = (&self.vad_model, &self.asr_model) else {
            return Ok(());
        };
        self.pipeline = Some(RealtimePipeline {
            vad: vad_model.new_stream(&self.vad_options)?,
            asr: asr_model.new_stream(&self.asr_options)?,
        });
        Ok(())
    }
}

fn section_object(
    value: Option<Value>,
    section: &str,
) -> Result<Option<Map<String, Value>>, SessionUpdateError> {
    match value {
        None => Ok(None),
        Some(Value::Object(value)) => Ok(Some(value)),
        Some(_) => Err(SessionUpdateError::new(
            format!("session.update `{section}` must be an object."),
            format!("session.{section}"),
        )),
    }
}

fn ignored_parameter_warning(section: &str, key: &str) -> SessionUpdateWarning {
    let param = format!("session.{section}.{key}");
    let init_only = matches!(
        key,
        "cache_dir"
            | "checkpoint"
            | "endpoint"
            | "revision"
            | "force_download"
            | "sample_rate"
            | "backend_sample_rate"
            | "disable_update"
            | "disable_log"
            | "disable_pbar"
            | "providers"
            | "intra_op_num_threads"
            | "inter_op_num_threads"
            | "num_threads"
            | "n_fft"
            | "hop_length"
            | "win_length"
            | "binary_path"
            | "model_path"
            | "stream_backend"
            | "stream_log_level"
            | "timeout_seconds"
            | "trust_remote_code"
    );
    let (code, message) = if init_only {
        (
            "init_only_parameter",
            format!("Ignoring init-only session parameter: {param}"),
        )
    } else {
        (
            "unsupported_parameter",
            format!("Ignoring unsupported session parameter: {param}"),
        )
    };
    SessionUpdateWarning {
        warning_type: "session_update_warning".to_string(),
        code: code.to_string(),
        message,
        param,
    }
}

fn normalized_optional_string(value: &Value) -> Option<String> {
    let value = match value {
        Value::Null => return None,
        Value::String(value) => value.clone(),
        value => value.to_string(),
    };
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn normalize_hotwords(value: &Value) -> Vec<String> {
    let items: Vec<String> = match value {
        Value::Null => Vec::new(),
        Value::String(value) => value
            .split(|character: char| {
                character.is_whitespace() || matches!(character, ',' | '，' | '、' | ';' | '；')
            })
            .map(str::to_string)
            .collect(),
        Value::Array(values) => values
            .iter()
            .filter_map(normalized_optional_string)
            .collect(),
        value => normalized_optional_string(value).into_iter().collect(),
    };
    items
        .into_iter()
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn json_f32(value: &Value, param: &str) -> Result<f32, SessionUpdateError> {
    value
        .as_f64()
        .map(|value| value as f32)
        .or_else(|| value.as_str().and_then(|value| value.parse::<f32>().ok()))
        .ok_or_else(|| SessionUpdateError::new(format!("`{param}` must be a number."), param))
}

fn json_u64(value: &Value, param: &str) -> Result<u64, SessionUpdateError> {
    value
        .as_u64()
        .or_else(|| value.as_str().and_then(|value| value.parse::<u64>().ok()))
        .ok_or_else(|| SessionUpdateError::new(format!("`{param}` must be an integer."), param))
}

struct SessionUpdateError {
    message: String,
    param: Option<String>,
}

impl SessionUpdateError {
    fn new(message: impl Into<String>, param: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            param: Some(param.into()),
        }
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
                let (start_ms, end_ms) = transcription_range(annotation.range);
                Some(ServerRealtimeEvent::TranscriptionText {
                    event_id: event_id(),
                    item_id,
                    delta,
                    text: segment.text,
                    start_ms,
                    end_ms,
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
                let (start_ms, end_ms) = transcription_range(annotation.range);
                Some(ServerRealtimeEvent::TranscriptionCompleted {
                    event_id: event_id(),
                    item_id,
                    text: segment.text,
                    start_ms,
                    end_ms,
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

fn transcription_range(range: TimeRange) -> (Option<u64>, Option<u64>) {
    if range == TimeRange::default() {
        (None, None)
    } else {
        (Some(range.start.0), Some(range.end.0))
    }
}

fn event_id() -> String {
    format!("event_{}", uuid::Uuid::new_v4().simple())
}

fn error_event(
    code: &str,
    message: impl Into<String>,
    request_event_id: Option<&str>,
    param: Option<&str>,
) -> ServerRealtimeEvent {
    ServerRealtimeEvent::Error {
        event_id: event_id(),
        error: RealtimeError {
            error_type: "invalid_request_error".to_string(),
            code: code.to_string(),
            message: message.into(),
            param: param.map(str::to_string),
            event_id: request_event_id.map(str::to_string),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{StreamingAsrSession, StreamingVadSession};
    use asr_data::{
        Annotation, AnnotationSource, AudioChunk, DurationMs, TextSpan, Timeline, Token, Waveform,
    };
    use std::sync::Mutex;

    struct TestVadSession;

    impl StreamingVadSession for TestVadSession {
        fn push_chunk(&mut self, _chunk: &AudioChunk) -> anyhow::Result<Vec<Annotation>> {
            Ok(Vec::new())
        }

        fn finish(&mut self) -> anyhow::Result<Vec<Annotation>> {
            Ok(vec![Annotation::new(
                TimeRange::new(DurationMs(0), DurationMs(100)),
                AnnotationPayload::Speech,
                AnnotationSource::Model("test_vad".to_string()),
                AnnotationStatus::Final,
            )])
        }
    }

    struct TestAsrSession;

    impl StreamingAsrSession for TestAsrSession {
        fn push_chunk(&mut self, _chunk: &AudioChunk) -> anyhow::Result<Vec<Annotation>> {
            Ok(Vec::new())
        }

        fn finish(&mut self) -> anyhow::Result<Vec<Annotation>> {
            Ok(vec![Annotation::new(
                TimeRange::default(),
                AnnotationPayload::Transcription(TextSpan {
                    text: "hello".to_string(),
                    tokens: Vec::<Token>::new(),
                    language: Some("en".to_string()),
                }),
                AnnotationSource::Model("test_asr".to_string()),
                AnnotationStatus::Final,
            )])
        }
    }

    struct TestOfflineAsr {
        received_samples: Arc<Mutex<Vec<usize>>>,
    }

    impl OfflineAsrModel for TestOfflineAsr {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn transcribe(
            &self,
            waveforms: &[Waveform],
            _options: &AsrOptions,
        ) -> anyhow::Result<Vec<Timeline>> {
            self.received_samples
                .lock()
                .map_err(|_| anyhow::anyhow!("sample recorder lock poisoned"))?
                .extend(waveforms.iter().map(|waveform| waveform.samples.len()));
            let mut timeline = Timeline::new("test-offline");
            timeline.push(Annotation::new(
                TimeRange::default(),
                AnnotationPayload::Transcription(TextSpan {
                    text: "offline final".to_string(),
                    tokens: Vec::new(),
                    language: Some("en".to_string()),
                }),
                AnnotationSource::Model("test_offline_asr".to_string()),
                AnnotationStatus::Final,
            ));
            Ok(vec![timeline])
        }
    }

    fn test_session() -> RealtimeSession {
        RealtimeSession::new(
            16_000,
            RealtimePipeline {
                vad: Box::new(TestVadSession),
                asr: Box::new(TestAsrSession),
            },
        )
    }

    #[test]
    fn structured_session_update_matches_fasr_shape() {
        let mut session = test_session();
        let events = handle_session_update(
            &mut session,
            Some(serde_json::json!({
                "audio": {"format": "pcm", "sample_rate": 16000},
                "asr": {
                    "language": "Chinese",
                    "hotwords": ["转人工", "开锁", "转人工"],
                    "max_new_tokens": 128
                },
                "vad": {
                    "threshold": 0.5,
                    "silence_duration_ms": 300,
                    "prefix_padding_ms": 120
                },
                "correct": {"rule_id": "house"}
            })),
            None,
        );

        let ServerRealtimeEvent::SessionUpdated {
            session, warnings, ..
        } = &events[0]
        else {
            panic!("expected session.updated");
        };
        assert_eq!(
            session.audio,
            serde_json::json!({"format": "pcm", "sample_rate": 16000})
        );
        assert_eq!(session.asr["language"], "Chinese");
        assert_eq!(
            session.asr["hotwords"],
            serde_json::json!(["开锁", "转人工"])
        );
        assert_eq!(session.vad["threshold"], 0.5);
        assert_eq!(session.vad["silence_duration_ms"], 300);
        assert_eq!(session.vad["prefix_padding_ms"], 120);
        assert_eq!(session.correct["rule_id"], "house");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].param, "session.asr.max_new_tokens");
    }

    #[test]
    fn legacy_session_update_is_rejected_like_fasr() {
        let mut session = test_session();
        let events = handle_session_update(
            &mut session,
            Some(serde_json::json!({"sample_rate": 16000})),
            Some("request_1"),
        );

        let ServerRealtimeEvent::Error { error, .. } = &events[0] else {
            panic!("expected error");
        };
        assert_eq!(error.code, "invalid_session_update");
        assert_eq!(error.param.as_deref(), Some("session.sample_rate"));
        assert_eq!(error.event_id.as_deref(), Some("request_1"));
    }

    #[test]
    fn commit_finishes_the_current_turn() {
        let mut session = test_session();
        session.turn_has_audio = true;
        session.asr_turn_has_audio = true;

        let events = session.commit();
        let speech_stopped = events.iter().find_map(|event| match event {
            ServerRealtimeEvent::SpeechStopped { item_id, .. } => Some(item_id),
            _ => None,
        });
        let completed = events.iter().find_map(|event| match event {
            ServerRealtimeEvent::TranscriptionCompleted { item_id, text, .. } => {
                Some((item_id, text))
            }
            _ => None,
        });
        let Some(speech_item_id) = speech_stopped else {
            panic!("expected speech_stopped");
        };
        let Some((completed_item_id, text)) = completed else {
            panic!("expected transcription completed");
        };
        assert_eq!(speech_item_id, completed_item_id);
        assert_eq!(text, "hello");
        assert!(session.commit().is_empty());
    }

    #[test]
    fn empty_finish_only_emits_session_finished() {
        let mut session = test_session();
        let events = session.finish();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            ServerRealtimeEvent::SessionFinished { .. }
        ));
    }

    #[test]
    fn two_pass_uses_full_turn_audio_for_offline_final() {
        let received_samples = Arc::new(Mutex::new(Vec::new()));
        let mut session = test_session();
        session.offline_asr_model = Some(Arc::new(TestOfflineAsr {
            received_samples: Arc::clone(&received_samples),
        }));
        session.turn_has_audio = true;
        session.asr_turn_has_audio = true;
        session.asr_turn_samples = vec![0.25; 3200];

        let events = session.commit();
        let completed_text = events.iter().find_map(|event| match event {
            ServerRealtimeEvent::TranscriptionCompleted { text, .. } => Some(text.as_str()),
            _ => None,
        });
        assert_eq!(completed_text, Some("offline final"));
        assert_eq!(
            *received_samples.lock().unwrap(),
            vec![3200],
            "offline ASR must receive the entire buffered turn"
        );
        assert!(session.asr_turn_samples.is_empty());
    }
}
