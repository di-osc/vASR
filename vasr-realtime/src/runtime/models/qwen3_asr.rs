use anyhow::{Context, Result};
use candle_core::Device;
use vasr::models::qwen3asr::{
    AudioInput, LoadOptions, Qwen3Asr, StreamOptions as RuntimeStreamOptions,
};
use vasr::{StreamEvent, timeline::ASR_SOURCE};
use vasr_data::{TimeRange, Timeline, Waveform};

use crate::runtime::model::{AsrOptions, StreamingAsrModel, StreamingAsrSession};

impl vasr::OfflineAsrModel for Qwen3AsrModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn transcribe(&self, waveforms: &[Waveform], options: &AsrOptions) -> Result<Vec<Timeline>> {
        vasr::OfflineAsrModel::transcribe(&self.inner, waveforms, options)
    }
}

pub struct Qwen3AsrModel {
    inner: Qwen3Asr,
}

impl Qwen3AsrModel {
    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &Device,
        load_options: &LoadOptions,
    ) -> Result<Self> {
        let inner = Qwen3Asr::from_pretrained(model_id_or_path, device, load_options)
            .with_context(|| format!("failed to load Qwen3-ASR model from {model_id_or_path:?}"))?;
        Ok(Self { inner })
    }

    pub fn inner(&self) -> &Qwen3Asr {
        &self.inner
    }
}

impl StreamingAsrModel for Qwen3AsrModel {
    fn new_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrSession>> {
        let stream = self.inner.new_stream(RuntimeStreamOptions {
            context: options.context.clone(),
            language: options.language.clone(),
            ..RuntimeStreamOptions::default()
        })?;
        Ok(Box::new(Qwen3AsrStreamModel {
            inner: Some(stream),
        }))
    }
}

struct Qwen3AsrStreamModel {
    inner: Option<vasr::models::qwen3asr::AsrStream>,
}

impl StreamingAsrSession for Qwen3AsrStreamModel {
    fn push_chunk(&mut self, chunk: &vasr_data::AudioChunk) -> Result<Vec<StreamEvent>> {
        let stream = self
            .inner
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR stream is already finished"))?;
        let output = stream.push_audio_chunk(&AudioInput::Waveform {
            samples: &chunk.samples,
            sample_rate: chunk.sample_rate,
        })?;
        Ok(output
            .map(|output| {
                vec![StreamEvent::transcription(
                    TimeRange::default(),
                    output.text,
                    Some(output.language),
                    ASR_SOURCE,
                    false,
                )]
            })
            .unwrap_or_default())
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        let stream = self
            .inner
            .take()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR stream is already finished"))?;
        let output = stream.finish()?;
        Ok(vec![StreamEvent::transcription(
            TimeRange::default(),
            output.text,
            Some(output.language),
            ASR_SOURCE,
            true,
        )])
    }
}
