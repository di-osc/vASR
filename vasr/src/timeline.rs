use anyhow::Result;
use asr_data::{Annotation, AudioActivity, TimeRange, TimeSpan, Timeline, Transcription, Waveform};

pub const ASR_SOURCE: &str = "qwen3_asr";
pub const VAD_SOURCE: &str = "vad";

#[derive(Debug, Clone, PartialEq)]
pub struct StreamEvent {
    pub range: TimeRange,
    pub annotation: Annotation,
    pub source: String,
    pub is_final: bool,
}

impl StreamEvent {
    pub fn activity(range: TimeRange, event: &str, source: &str, is_final: bool) -> Self {
        Self {
            range,
            annotation: Annotation::Activity(AudioActivity::new().with_event(event)),
            source: source.to_string(),
            is_final,
        }
    }

    pub fn transcription(
        range: TimeRange,
        text: impl Into<String>,
        language: Option<String>,
        source: &str,
        is_final: bool,
    ) -> Self {
        Self {
            range,
            annotation: Annotation::Transcription(Transcription {
                text: text.into(),
                tokens: Vec::new(),
                language,
                confidence: None,
            }),
            source: source.to_string(),
            is_final,
        }
    }

    pub fn into_span(self) -> TimeSpan {
        TimeSpan::new(self.range, self.annotation, Some(self.source))
    }
}

pub fn prediction_span(range: TimeRange, annotation: Annotation, source: &str) -> TimeSpan {
    TimeSpan::new(range, annotation, Some(source.to_string()))
}

pub fn activity_span(range: TimeRange, event: &str, source: &str) -> TimeSpan {
    prediction_span(
        range,
        Annotation::Activity(AudioActivity::new().with_event(event)),
        source,
    )
}

pub fn transcription_span(
    range: TimeRange,
    text: impl Into<String>,
    language: Option<String>,
    source: &str,
) -> TimeSpan {
    prediction_span(
        range,
        Annotation::Transcription(Transcription {
            text: text.into(),
            tokens: Vec::new(),
            language,
            confidence: None,
        }),
        source,
    )
}

pub fn language_span(range: TimeRange, language: impl Into<String>, source: &str) -> TimeSpan {
    prediction_span(range, Annotation::Language(language.into()), source)
}

pub fn waveform_range(waveform: &Waveform) -> TimeRange {
    TimeRange::new(0, waveform.duration_ms().round() as usize)
}

pub fn timeline_from_spans(
    audio_id: impl Into<String>,
    duration: usize,
    spans: impl IntoIterator<Item = TimeSpan>,
) -> Result<Timeline> {
    let mut timeline = Timeline::new(audio_id, duration);
    extend_predictions(&mut timeline, spans)?;
    Ok(timeline)
}

pub fn extend_predictions(
    timeline: &mut Timeline,
    spans: impl IntoIterator<Item = TimeSpan>,
) -> Result<()> {
    for span in spans {
        // 预测轨道要求非空 source，直接沿用 span 自带的来源。
        let TimeSpan {
            range,
            annotation,
            source,
            ..
        } = span;
        timeline.annotate_span_with(
            range.start_ms,
            range.end_ms,
            annotation,
            false,
            source.as_deref(),
        )?;
    }
    Ok(())
}

pub fn offset_spans(spans: Vec<TimeSpan>, offset: usize) -> Vec<TimeSpan> {
    spans
        .into_iter()
        .map(|mut span| {
            span.range.start_ms = span.range.start_ms.saturating_add(offset);
            span.range.end_ms = span.range.end_ms.saturating_add(offset);
            span
        })
        .collect()
}

pub fn prediction_text(timeline: &Timeline) -> String {
    let mut parts = timeline
        .prediction
        .iter()
        .filter_map(|span| match &span.annotation {
            Annotation::Transcription(transcription) => {
                Some((span.range.start_ms, transcription.text.clone()))
            }
            Annotation::Sentence(sentence) => Some((span.range.start_ms, sentence.text.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    parts.sort_by_key(|(start, _)| *start);
    parts
        .into_iter()
        .map(|(_, text)| text)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn reference_text(timeline: &Timeline) -> String {
    timeline.reference_transcript().text
}
