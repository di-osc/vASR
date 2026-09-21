use std::path::Path;

use anyhow::Result;
use asr_data::Waveform;
use candle_core::Device;
use vasr::StreamEvent;
use vasr::timeline::activity_span;
use vasr_vad as model_vad;

use crate::runtime::model::{StreamingVadSession, VadModel, VadOptions, VadSegment};

#[derive(Debug, Clone)]
pub struct FireRedVadModel {
    inner: model_vad::FireRedVadModel,
}

impl FireRedVadModel {
    pub fn from_pretrained(path: Option<impl AsRef<Path>>) -> Result<Self> {
        Ok(Self {
            inner: model_vad::FireRedVadModel::from_pretrained_streaming(path)?,
        })
    }

    pub fn from_pretrained_on_device(
        path: Option<impl AsRef<Path>>,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            inner: model_vad::FireRedVadModel::from_pretrained_streaming_on_device(path, device)?,
        })
    }
}

impl VadModel for FireRedVadModel {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        self.inner
            .detect(waveform, &model_options(options))
            .map(model_segments)
    }

    fn new_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadSession>> {
        Ok(Box::new(FireRedVadStream {
            inner: model_vad::VadModel::new_stream(&self.inner, &model_options(options))?,
        }))
    }
}

struct FireRedVadStream {
    inner: Box<dyn model_vad::StreamingVadSession>,
}

impl StreamingVadSession for FireRedVadStream {
    fn push_chunk(&mut self, chunk: &asr_data::AudioChunk) -> Result<Vec<StreamEvent>> {
        Ok(self
            .inner
            .push_chunk(chunk)?
            .into_iter()
            .map(vad_event)
            .collect())
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        Ok(self.inner.finish()?.into_iter().map(vad_event).collect())
    }
}

fn vad_event(event: model_vad::VadEvent) -> StreamEvent {
    let span = activity_span(event.range, "speech", "firered_vad");
    StreamEvent {
        range: span.range,
        annotation: span.annotation,
        source: span.source.unwrap_or_else(|| "firered_vad".to_string()),
        is_final: matches!(event.status, model_vad::VadEventStatus::Final),
    }
}

fn model_options(options: &VadOptions) -> model_vad::VadOptions {
    model_vad::VadOptions {
        threshold: options.threshold,
        min_speech_ms: options.min_speech_ms,
        min_silence_ms: options.min_silence_ms,
        max_segment_ms: options.max_segment_ms,
        pad_ms: options.pad_ms,
        num_threads: options.num_threads,
    }
}

fn model_segments(segments: Vec<model_vad::VadSegment>) -> Vec<VadSegment> {
    segments
        .into_iter()
        .map(|segment| VadSegment {
            range: segment.range,
            probability: segment.probability,
        })
        .collect()
}
