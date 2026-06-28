use serde::{Deserialize, Serialize};
use vasr_data::{AnnotationPayload, AnnotationStatus, Timeline};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInputDto {
    pub url: Option<String>,
    pub b64_str: Option<String>,
    pub mono: Option<bool>,
    pub hotword: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeRequest {
    pub inputs: Vec<AudioInputDto>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextResult {
    pub mono: String,
    pub left: String,
    pub right: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SentenceResult {
    pub text: String,
    pub start_time: u64,
    pub end_time: u64,
    pub duration: u64,
    pub channel: String,
    pub sentence_index: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpentDetails {
    #[serde(default)]
    pub loader: f64,
    #[serde(default)]
    pub detector: f64,
    #[serde(default)]
    pub recognizer: f64,
    #[serde(default)]
    pub sentencizer: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimingBreakdown {
    #[serde(default)]
    pub total: f64,
    #[serde(default)]
    pub loader: f64,
    #[serde(default)]
    pub pipeline: f64,
    #[serde(default)]
    pub audio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceData {
    #[serde(rename = "serviceId")]
    pub service_id: String,
    #[serde(rename = "spentSeconds")]
    pub spent_seconds: f64,
    #[serde(rename = "spentDetails")]
    #[serde(default)]
    pub spent_details: SpentDetails,
    #[serde(default)]
    pub timings: TimingBreakdown,
    pub text: TextResult,
    pub sentences: Vec<SentenceResult>,
    pub is_bad: bool,
    pub bad_reason: Option<String>,
    pub bad_component: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePerformance {
    pub batch_wall_seconds: f64,
    pub num_items: usize,
    pub throughput_items_per_second: f64,
    pub total_audio_duration_seconds: f64,
    pub speedup: f64,
    pub rtf: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscribeResponse {
    pub data: Vec<InferenceData>,
    pub inference_performance: InferencePerformance,
}

impl InferenceData {
    pub fn from_timeline(service_id: impl Into<String>, timeline: &Timeline) -> Self {
        Self::from_timeline_channel(service_id, timeline, "mono")
    }

    pub fn from_timeline_channel(
        service_id: impl Into<String>,
        timeline: &Timeline,
        channel: &str,
    ) -> Self {
        let transcript = timeline.transcript();
        let mut sentences = Vec::new();
        for annotation in timeline
            .annotations
            .iter()
            .filter(|annotation| annotation.status == AnnotationStatus::Final)
        {
            match &annotation.payload {
                AnnotationPayload::Transcription(segment)
                | AnnotationPayload::Sentence(segment) => {
                    let duration = annotation.range.duration().0;
                    sentences.push(SentenceResult {
                        text: segment.text.clone(),
                        start_time: annotation.range.start.0,
                        end_time: annotation.range.end.0,
                        duration,
                        channel: channel.to_string(),
                        sentence_index: sentences.len(),
                    });
                }
                _ => {}
            }
        }
        let mut text = TextResult::default();
        match channel {
            "left" => text.left = transcript.text,
            "right" => text.right = transcript.text,
            _ => text.mono = transcript.text,
        }
        Self {
            service_id: service_id.into(),
            spent_seconds: 0.0,
            spent_details: SpentDetails::default(),
            timings: TimingBreakdown::default(),
            text,
            sentences,
            is_bad: false,
            bad_reason: None,
            bad_component: None,
        }
    }
}
