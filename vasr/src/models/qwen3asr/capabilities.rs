use anyhow::Result;
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioChunk, TextSpan,
    TimeRange, Timeline, Waveform,
};

use crate::traits::{AsrOptions, OfflineAsrModel, StreamingAsrModel, StreamingAsrSession};

#[cfg(feature = "paged-attn")]
use super::EncodedAsrBatchPoll;
use super::{AsrStream, AsrTranscription, AudioInput, Batch, EncodedAsrBatch, Qwen3Asr};
use super::{StreamOptions, TranscribeOptions};

impl OfflineAsrModel for Qwen3Asr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn transcribe(&self, waveforms: &[Waveform], options: &AsrOptions) -> Result<Vec<Timeline>> {
        let runtime_options = qwen_transcribe_options(options);
        let audio = waveforms
            .iter()
            .map(|waveform| AudioInput::Waveform {
                samples: &waveform.samples,
                sample_rate: waveform.sample_rate,
            })
            .collect::<Vec<_>>();
        #[cfg(feature = "timing")]
        let outputs = {
            let (outputs, timings) = Qwen3Asr::transcribe_timed(self, audio, runtime_options)?;
            log_transcribe_timings(waveforms.len(), &timings);
            outputs
        };
        #[cfg(not(feature = "timing"))]
        let outputs = Qwen3Asr::transcribe(self, audio, runtime_options)?;
        qwen_outputs_to_timelines(outputs, waveforms)
    }
}

impl Qwen3Asr {
    pub fn encode_waveforms_audio_features_to_cpu(
        &self,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<EncodedAsrBatch> {
        let runtime_options = qwen_transcribe_options(options);
        let audio = waveforms
            .iter()
            .map(|waveform| AudioInput::Waveform {
                samples: &waveform.samples,
                sample_rate: waveform.sample_rate,
            })
            .collect::<Vec<_>>();
        self.encode_audio_features_to_cpu(audio, runtime_options)
    }

    pub fn decode_cpu_audio_features_to_timelines(
        &self,
        encoded: EncodedAsrBatch,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        let runtime_options = qwen_transcribe_options(options);
        let outputs = self.decode_cpu_audio_features(encoded, runtime_options)?;
        qwen_outputs_to_timelines(outputs, waveforms)
    }

    #[cfg(feature = "paged-attn")]
    pub fn decode_cpu_audio_features_dynamic_outputs(
        &self,
        encoded: EncodedAsrBatch,
        mut poll_encoded: impl FnMut(bool) -> Result<EncodedAsrBatchPoll>,
        options: &AsrOptions,
    ) -> Result<Vec<Vec<AsrTranscription>>> {
        let runtime_options = qwen_transcribe_options(options);
        self.decode_cpu_audio_features_dynamic(encoded, &mut poll_encoded, runtime_options)
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

#[cfg(feature = "timing")]
fn timing_sec(us: u64) -> f64 {
    us as f64 / 1_000_000.0
}

#[cfg(feature = "timing")]
fn tokens_per_sec(tokens: usize, us: u64) -> f64 {
    if us == 0 {
        0.0
    } else {
        tokens as f64 / timing_sec(us)
    }
}

#[cfg(feature = "timing")]
fn log_transcribe_timings(items: usize, t: &super::TranscribeTimings) {
    if std::env::var_os("VASR_DEBUG_QWEN3_ASR_TIMING").is_none() {
        return;
    }
    tracing::debug!(
        target: "vasr::models::qwen3asr",
        "qwen3_asr_timing | items={} | chunks={} | batches={} | total={:.3}s | normalize={:.3}s | chunking={:.3}s | prepare={:.3}s | prepare_norm={:.3}s | prepare_tok_lookup={:.3}s | prepare_feat={:.3}s | prepare_tok_expand={:.3}s | prepare_pad={:.3}s | stack={:.3}s | audio_encoder={:.3}s | prefill={:.3}s | prefill_tokens={} | prefill_tok_s={:.1} | prefill_inputs={:.3}s | prefill_rope={:.3}s | prefill_metadata={:.3}s | prefill_mask={:.3}s | prefill_forward={:.3}s | prefill_gather={:.3}s | prefill_decode_setup={:.3}s | prefill_argmax={:.3}s | decode={:.3}s | decode_tok_s={:.1} | decode_token_tensor={:.3}s | decode_embed={:.3}s | decode_position={:.3}s | decode_metadata={:.3}s | decode_graph_replay={:.3}s | decode_forward={:.3}s | decode_pre_argmax_sync={:.3}s | decode_argmax={:.3}s | decode_steps={} | generated_tokens={} | token_decode={:.3}s",
        items,
        t.chunks,
        t.batches,
        timing_sec(t.total_us),
        timing_sec(t.audio_normalize_us),
        timing_sec(t.audio_chunking_us),
        timing_sec(t.processor_prepare_batch_us),
        timing_sec(t.processor_prepare_normalize_us),
        timing_sec(t.processor_prepare_token_lookup_us),
        timing_sec(t.processor_prepare_feature_extract_us),
        timing_sec(t.processor_prepare_tokenize_expand_us),
        timing_sec(t.processor_prepare_pad_us),
        timing_sec(t.stack_features_us),
        timing_sec(t.audio_encoder_us),
        timing_sec(t.generation.prefill_us),
        t.generation.prefill_tokens,
        tokens_per_sec(t.generation.prefill_tokens, t.generation.prefill_us),
        timing_sec(t.generation.prefill_inputs_us),
        timing_sec(t.generation.prefill_rope_us),
        timing_sec(t.generation.prefill_metadata_us),
        timing_sec(t.generation.prefill_mask_us),
        timing_sec(t.generation.prefill_forward_us),
        timing_sec(t.generation.prefill_gather_us),
        timing_sec(t.generation.prefill_decode_setup_us),
        timing_sec(t.generation.prefill_argmax_us),
        timing_sec(t.generation.decode_us),
        tokens_per_sec(t.generation.tokens_generated, t.generation.decode_us),
        timing_sec(t.generation.decode_token_tensor_us),
        timing_sec(t.generation.decode_embed_us),
        timing_sec(t.generation.decode_position_us),
        timing_sec(t.generation.decode_metadata_us),
        timing_sec(t.generation.decode_graph_replay_us),
        timing_sec(t.generation.decode_forward_us),
        timing_sec(t.generation.decode_pre_argmax_sync_us),
        timing_sec(t.generation.decode_argmax_us),
        t.generation.steps,
        t.generation.tokens_generated,
        timing_sec(t.tokenizer_decode_us),
    );
}
