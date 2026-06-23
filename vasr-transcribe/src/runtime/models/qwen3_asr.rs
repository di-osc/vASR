use anyhow::{Context, Result};
use candle_core::Device;
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::EncodedAsrBatchPoll;
#[cfg(feature = "timing")]
use vasr::models::qwen3asr::TranscribeTimings;
use vasr::models::qwen3asr::{
    AsrTranscription, AudioInput, EncodedAsrBatch, LoadOptions, Qwen3Asr,
};
use vasr::{qwen_outputs_to_timelines, qwen_transcribe_options};
use vasr_data::{Timeline, Waveform};

use crate::runtime::model::{AsrModel, AsrOptions};

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

    pub fn encode_audio_features_to_cpu(
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
        self.inner
            .encode_audio_features_to_cpu(audio, runtime_options)
    }

    pub fn decode_cpu_audio_features(
        &self,
        encoded: EncodedAsrBatch,
        waveforms: &[Waveform],
        options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        let runtime_options = qwen_transcribe_options(options);
        let outputs = self
            .inner
            .decode_cpu_audio_features(encoded, runtime_options)?;
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
        self.inner
            .decode_cpu_audio_features_dynamic(encoded, &mut poll_encoded, runtime_options)
    }
}

impl AsrModel for Qwen3AsrModel {
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
        #[cfg(feature = "timing")]
        let outputs = {
            let (outputs, timings) = self.inner.transcribe_timed(audio, runtime_options)?;
            log_transcribe_timings(waveforms.len(), &timings);
            outputs
        };
        #[cfg(not(feature = "timing"))]
        let outputs = self.inner.transcribe(audio, runtime_options)?;
        qwen_outputs_to_timelines(outputs, waveforms)
    }
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
fn log_transcribe_timings(items: usize, t: &TranscribeTimings) {
    if std::env::var_os("VASR_DEBUG_QWEN3_ASR_TIMING").is_none() {
        return;
    }
    tracing::debug!(
        target: "vasr_transcribe::runtime::models::qwen3_asr",
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
