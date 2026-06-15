use anyhow::Result;
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioChunk, TextSpan,
    TimeRange, Timeline, Waveform,
};

use crate::asr::traits::{AsrOptions, OfflineAsrModel, StreamingAsrModel, StreamingAsrSession};

use super::{
    AsrStream, AsrTranscription, AudioInput, Batch, Qwen3Asr, StreamOptions, TranscribeOptions,
};

impl OfflineAsrModel for Qwen3Asr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline> {
        let mut timelines = self.transcribe_batch(std::slice::from_ref(waveform), options)?;
        timelines
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Qwen3-ASR returned no transcription"))
    }

    fn transcribe_batch(
        &self,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        let runtime_options = qwen_transcribe_options(options);
        let audio = waveforms
            .iter()
            .map(|waveform| AudioInput::Waveform {
                samples: &waveform.samples,
                sample_rate: waveform.sample_rate,
            })
            .collect::<Vec<_>>();
        let outputs = Qwen3Asr::transcribe(self, audio, runtime_options)?;
        qwen_outputs_to_timelines(outputs, waveforms)
    }
}

impl StreamingAsrModel for Qwen3Asr {
    fn new_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrSession>> {
        let stream = Qwen3Asr::new_stream(
            self,
            StreamOptions {
                context: options.context.clone(),
                language: options.language.clone(),
                max_new_tokens: options.max_new_tokens,
                ..StreamOptions::default()
            },
        )?;
        Ok(Box::new(Qwen3AsrStreamingSession {
            inner: Some(stream),
        }))
    }
}

struct Qwen3AsrStreamingSession {
    inner: Option<AsrStream>,
}

impl StreamingAsrSession for Qwen3AsrStreamingSession {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>> {
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
                        tokens: Vec::new(),
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

pub fn qwen_transcribe_options(options: &AsrOptions) -> TranscribeOptions {
    TranscribeOptions {
        context: Batch::one(options.context.clone()),
        language: Batch::one(options.language.clone()),
        return_timestamps: false,
        max_new_tokens: options.max_new_tokens,
        decode_max_seqs: options.decode_max_seqs,
        max_batch_audio_sec: options.max_batch_audio_sec,
        chunk_max_sec: options.chunk_max_sec,
        bucket_by_length: !super::inference::utils::continuous_paged_batch_enabled(),
    }
}

pub fn qwen_outputs_to_timelines(
    outputs: Vec<AsrTranscription>,
    waveforms: &[Waveform],
) -> Result<Vec<Timeline>> {
    if outputs.len() != waveforms.len() {
        anyhow::bail!(
            "Qwen3-ASR returned {} transcriptions for {} inputs",
            outputs.len(),
            waveforms.len()
        );
    }
    Ok(outputs
        .into_iter()
        .zip(waveforms)
        .map(|(output, waveform)| qwen_output_to_timeline(output, waveform))
        .collect())
}

fn qwen_output_to_timeline(output: AsrTranscription, waveform: &Waveform) -> Timeline {
    let mut timeline = Timeline::new("qwen3_asr_audio");
    let range = TimeRange::new(
        vasr_data::DurationMs(0),
        vasr_data::DurationMs(waveform.duration_ms().round() as u64),
    );
    if !output.language.is_empty() {
        timeline.push(Annotation::new(
            range,
            AnnotationPayload::Language(output.language.clone()),
            AnnotationSource::Model("qwen3_asr".to_string()),
            AnnotationStatus::Final,
        ));
    }
    timeline.push(Annotation::new(
        range,
        AnnotationPayload::Transcription(TextSpan {
            text: output.text,
            tokens: Vec::new(),
            language: Some(output.language),
        }),
        AnnotationSource::Model("qwen3_asr".to_string()),
        AnnotationStatus::Final,
    ));
    timeline
}
