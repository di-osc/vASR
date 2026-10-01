use anyhow::Result;
use asr_data::{AudioEvent, Speech, TimeRange, Timeline, Transcription, Waveform};

pub const ASR_SOURCE: &str = "qwen3_asr";
pub const VAD_SOURCE: &str = "vad";

/// 流式模型吐出的一条事件。
///
/// `event` 只带内容：说话人、语种、转写等。写进时间轴时 `id` / `range` / `source`
/// 由 [`extend_predictions`] 统一填上。
#[derive(Debug, Clone, PartialEq)]
pub struct StreamEvent {
    pub range: TimeRange,
    pub event: AudioEvent,
    pub source: String,
    pub is_final: bool,
}

impl StreamEvent {
    /// 构造活动事件。`speech` 会变成语音事件，其它名字是基类事件。
    pub fn activity(range: TimeRange, event: &str, source: &str, is_final: bool) -> Self {
        Self {
            range,
            event: audio_event(event),
            source: source.to_string(),
            is_final,
        }
    }

    /// 构造一条语音转写事件。
    pub fn transcription(
        range: TimeRange,
        text: impl Into<String>,
        language: Option<String>,
        source: &str,
        is_final: bool,
    ) -> Self {
        let mut speech = Speech::new().with_transcription(Transcription::new(text));
        if let Some(language) = language.filter(|language| !language.trim().is_empty()) {
            speech = speech.with_language(language);
        }
        Self {
            range,
            event: AudioEvent::Speech(speech),
            source: source.to_string(),
            is_final,
        }
    }

    /// 转成带身份、可直接写进 prediction 轨道的事件。
    pub fn into_span(self) -> AudioEvent {
        event_span(self.event, self.range, &self.source)
    }
}

/// `speech` 映射成语音事件，其它事件名映射成基类事件。
fn audio_event(event: &str) -> AudioEvent {
    if event == asr_data::SPEECH_EVENT_NAME {
        AudioEvent::Speech(Speech::new())
    } else {
        AudioEvent::new(event)
    }
}

/// 给事件补上时间范围和来源。
///
/// 只用于 vASR 内部的预测事件载体；真正写入时间轴时 `annotate_with` 会再覆盖一次。
fn event_span(event: AudioEvent, range: TimeRange, source: &str) -> AudioEvent {
    let source = Some(source.to_string());
    match event {
        AudioEvent::Base {
            name, confidence, ..
        } => AudioEvent::Base {
            id: String::new(),
            range,
            source,
            name,
            confidence,
        },
        AudioEvent::Speech(speech) => AudioEvent::Speech(Speech {
            id: String::new(),
            range,
            source,
            ..speech
        }),
    }
}

/// 构造一个预测活动事件。
pub fn activity_span(range: TimeRange, event: &str, source: &str) -> AudioEvent {
    event_span(audio_event(event), range, source)
}

/// 构造一个预测转写事件。
pub fn transcription_span(
    range: TimeRange,
    text: impl Into<String>,
    language: Option<String>,
    source: &str,
) -> AudioEvent {
    let mut speech = Speech::new().with_transcription(Transcription::new(text));
    if let Some(language) = language.filter(|language| !language.trim().is_empty()) {
        speech = speech.with_language(language);
    }
    event_span(AudioEvent::Speech(speech), range, source)
}

pub fn waveform_range(waveform: &Waveform) -> TimeRange {
    TimeRange::new(0, waveform.duration_ms().round() as usize)
}

/// 用一组预测事件构造时间轴。
pub fn timeline_from_spans(
    audio_id: impl Into<String>,
    duration: usize,
    events: impl IntoIterator<Item = AudioEvent>,
) -> Result<Timeline> {
    let mut timeline = Timeline::new(audio_id, duration);
    extend_predictions(&mut timeline, events)?;
    Ok(timeline)
}

/// 把预测事件写进时间轴的 prediction 轨道。
pub fn extend_predictions(
    timeline: &mut Timeline,
    events: impl IntoIterator<Item = AudioEvent>,
) -> Result<()> {
    for event in events {
        let range = event.range();
        let source = event.source().map(str::to_owned);
        timeline.annotate_with(
            range.start_ms,
            range.end_ms,
            event,
            false,
            source.as_deref(),
        )?;
    }
    Ok(())
}

/// 把一组预测事件整体平移到新的时间偏移。
pub fn offset_spans(events: Vec<AudioEvent>, offset: usize) -> Vec<AudioEvent> {
    events
        .into_iter()
        .map(|event| offset_event(event, offset))
        .collect()
}

fn offset_event(event: AudioEvent, offset: usize) -> AudioEvent {
    match event {
        AudioEvent::Base {
            id,
            range,
            source,
            name,
            confidence,
        } => AudioEvent::Base {
            id,
            range: shift_range(range, offset),
            source,
            name,
            confidence,
        },
        AudioEvent::Speech(mut speech) => {
            speech.range = shift_range(speech.range, offset);
            AudioEvent::Speech(speech)
        }
    }
}

fn shift_range(range: TimeRange, offset: usize) -> TimeRange {
    TimeRange::new(
        range.start_ms.saturating_add(offset),
        range.end_ms.saturating_add(offset),
    )
}

/// 拼接 prediction 轨道上的全部转写文本。
pub fn prediction_text(timeline: &Timeline) -> String {
    let mut parts = timeline
        .prediction
        .iter()
        .filter_map(|event| {
            event
                .transcription()
                .map(|transcription| (event.range().start_ms, transcription.text.clone()))
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
