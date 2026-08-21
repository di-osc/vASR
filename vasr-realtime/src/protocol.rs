use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionUpdatePayload {
    #[serde(default)]
    pub input_audio_format: Option<String>,
    #[serde(default)]
    pub sample_rate: Option<u32>,
    #[serde(default)]
    pub input_audio_transcription: Option<Value>,
    #[serde(default)]
    pub turn_detection: Option<Value>,
    #[serde(default)]
    pub enhance: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientRealtimeEvent {
    #[serde(rename = "session.update")]
    SessionUpdate {
        #[serde(default)]
        session: Option<SessionUpdatePayload>,
        // Kept for wire compatibility with the original vASR protocol.
        #[serde(default)]
        sample_rate: Option<u32>,
        #[serde(default)]
        input_audio_format: Option<String>,
    },
    #[serde(rename = "input_audio_buffer.append")]
    AudioAppend { audio: String },
    #[serde(rename = "input_audio_buffer.commit")]
    AudioCommit,
    #[serde(rename = "session.finish")]
    SessionFinish,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    #[serde(rename = "type", default = "default_session_type")]
    pub session_type: String,
    pub audio: Value,
    pub vad: Value,
    pub asr: Value,
    pub correct: Value,
    pub enhance: Value,
    pub turn_analyzer: Value,
}

fn default_session_type() -> String {
    "realtime".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerRealtimeEvent {
    #[serde(rename = "session.created")]
    SessionCreated {
        event_id: String,
        session: SessionInfo,
    },
    #[serde(rename = "session.updated")]
    SessionUpdated {
        event_id: String,
        session: SessionInfo,
    },
    #[serde(rename = "input_audio_buffer.speech_started")]
    SpeechStarted {
        event_id: String,
        item_id: String,
        audio_start_ms: u64,
    },
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    SpeechStopped {
        event_id: String,
        item_id: String,
        audio_end_ms: u64,
    },
    #[serde(rename = "conversation.item.input_audio_transcription.text")]
    TranscriptionText {
        event_id: String,
        item_id: String,
        text: String,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        start_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        emotion: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        speech_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        itn: Option<bool>,
    },
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    TranscriptionCompleted {
        event_id: String,
        item_id: String,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        start_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        emotion: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        speech_type: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        itn: Option<bool>,
    },
    #[serde(rename = "session.finished")]
    SessionFinished { event_id: String },
    #[serde(rename = "error")]
    Error {
        event_id: String,
        error: RealtimeError,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fasr_nested_session_update() {
        let event: ClientRealtimeEvent = serde_json::from_value(serde_json::json!({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm",
                "sample_rate": 16000,
                "input_audio_transcription": {
                    "language": "Chinese",
                    "hotwords": ["SenseVoice"]
                },
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": 500
                }
            }
        }))
        .unwrap();

        match event {
            ClientRealtimeEvent::SessionUpdate { session, .. } => {
                let session = session.unwrap();
                assert_eq!(session.sample_rate, Some(16_000));
                assert_eq!(session.input_audio_format.as_deref(), Some("pcm"));
                assert_eq!(
                    session.input_audio_transcription.unwrap()["language"],
                    "Chinese"
                );
            }
            _ => panic!("expected session.update"),
        }
    }

    #[test]
    fn serializes_fasr_transcription_shape() {
        let value = serde_json::to_value(ServerRealtimeEvent::TranscriptionText {
            event_id: "event_1".to_string(),
            item_id: "item_1".to_string(),
            text: "你好".to_string(),
            delta: "好".to_string(),
            start_ms: Some(100),
            end_ms: Some(600),
            language: Some("zh".to_string()),
            emotion: "UNKNOWN".to_string(),
            speech_type: None,
            itn: None,
        })
        .unwrap();

        assert_eq!(
            value["type"],
            "conversation.item.input_audio_transcription.text"
        );
        assert_eq!(value["event_id"], "event_1");
        assert_eq!(value["item_id"], "item_1");
        assert_eq!(value["text"], "你好");
        assert_eq!(value["delta"], "好");
        assert_eq!(value["emotion"], "UNKNOWN");
        assert!(value.get("speech_type").is_none());
    }

    #[test]
    fn speech_stopped_matches_fasr_fields() {
        let value = serde_json::to_value(ServerRealtimeEvent::SpeechStopped {
            event_id: "event_1".to_string(),
            item_id: "item_1".to_string(),
            audio_end_ms: 900,
        })
        .unwrap();
        assert_eq!(value["type"], "input_audio_buffer.speech_stopped");
        assert_eq!(value["audio_end_ms"], 900);
        assert!(value.get("audio_start_ms").is_none());
    }
}
