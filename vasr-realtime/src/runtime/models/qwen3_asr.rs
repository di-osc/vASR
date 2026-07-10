use anyhow::{Context, Result};
use candle_core::Device;
use vasr::models::qwen3asr::{
    AudioInput, LoadOptions, Qwen3Asr, StreamOptions as RuntimeStreamOptions,
};
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, TextSpan, TimeRange, Token,
};

use crate::runtime::model::{AsrOptions, StreamingAsrModel, StreamingAsrSession};

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
    fn push_chunk(&mut self, chunk: &vasr_data::AudioChunk) -> Result<Vec<Annotation>> {
        let stream = self
            .inner
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR stream is already finished"))?;
        let output = stream.push_audio_chunk(&AudioInput::Waveform {
            samples: &chunk.waveform.samples,
            sample_rate: chunk.waveform.sample_rate,
        })?;
        Ok(output
            .map(|output| {
                vec![Annotation::new(
                    chunk.range,
                    AnnotationPayload::Transcription(TextSpan {
                        text: output.text,
                        tokens: Vec::<Token>::new(),
                        language: Some(output.language),
                    }),
                    AnnotationSource::Model("qwen3_asr".to_string()),
                    AnnotationStatus::Partial,
                )]
            })
            .unwrap_or_default())
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        let stream = self
            .inner
            .take()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR stream is already finished"))?;
        let output = stream.finish()?;
        Ok(vec![Annotation::new(
            TimeRange::default(),
            AnnotationPayload::Transcription(TextSpan {
                text: output.text,
                tokens: Vec::new(),
                language: Some(output.language),
            }),
            AnnotationSource::Model("qwen3_asr".to_string()),
            AnnotationStatus::Final,
        )])
    }
}
