use std::path::Path;

use anyhow::Result;
use asr_data::Waveform;
use vasr_vad as model_vad;

use crate::runtime::model::{StreamingVadSession, VadModel, VadOptions, VadSegment};

#[derive(Debug, Clone)]
pub struct FsmnVadModel {
    inner: model_vad::FsmnVadModel,
}

impl FsmnVadModel {
    pub fn from_pretrained(path: Option<impl AsRef<Path>>) -> Result<Self> {
        Ok(Self {
            inner: model_vad::FsmnVadModel::from_pretrained(path)?,
        })
    }
}

impl VadModel for FsmnVadModel {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        Ok(self
            .inner
            .detect(waveform, &model_options(options))
            .map(model_segments)?)
    }

    fn new_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadSession>> {
        Ok(Box::new(FsmnVadStream {
            inner: model_vad::VadModel::new_stream(&self.inner, &model_options(options))?,
        }))
    }
}

struct FsmnVadStream {
    inner: Box<dyn model_vad::StreamingVadSession>,
}

impl StreamingVadSession for FsmnVadStream {
    fn push_chunk(&mut self, chunk: &asr_data::AudioChunk) -> Result<Vec<asr_data::Annotation>> {
        self.inner.push_chunk(chunk)
    }

    fn finish(&mut self) -> Result<Vec<asr_data::Annotation>> {
        self.inner.finish()
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
