pub mod capabilities;
pub mod config;
pub mod inference;
pub mod model;
pub mod processor;

pub use crate::audio;
pub use crate::error;
pub use vasr_hub::modelscope::download;

use anyhow::Result;
use candle_core::Device;
use std::path::PathBuf;
use std::sync::Arc;

pub use crate::traits::{AsrOptions, OfflineAsrModel, StreamingAsrModel, StreamingAsrSession};
pub use capabilities::{qwen_outputs_to_timelines, qwen_transcribe_options};
#[cfg(feature = "paged-attn")]
pub use inference::batch_scheduler::{AsrBatchScheduler, AsrBatchSchedulerConfig};
pub use inference::streaming::AsrStream;
pub use inference::transcribe::EncodedAsrBatch;
#[cfg(feature = "paged-attn")]
pub use inference::transcribe::EncodedAsrBatchPoll;
#[cfg(feature = "timing")]
pub use inference::transcribe::TranscribeTimings;
pub use inference::types::{AsrTranscription, Batch, StreamOptions, TranscribeOptions};
#[cfg(feature = "timing")]
pub use model::generation::GenerationTimings;
#[cfg(feature = "paged-attn")]
pub use model::paged_batch_engine::{PagedBatchConfig, PagedBatchState, PagedDecodeSlot};
pub use model::weights::LoadOptions;
pub use model::weights::WeightsMemoryStats;
pub use processor::AsrProcessor;
pub use vasr_data::audio::input::AudioInput;
#[cfg(feature = "paged-attn")]
pub use vasr_paged_attn::{PagedCacheConfig, PagedCacheMemory, PagedCacheStats};

#[derive(Debug)]
pub struct Qwen3ASR {
    device: Arc<Device>,
    config: config::AsrConfig,
    processor: Arc<AsrProcessor>,
    model_dir: PathBuf,
    weights_paths: Vec<PathBuf>,
    thinker: Arc<model::thinker::ThinkerForConditionalGeneration>,
    #[cfg(feature = "paged-attn")]
    paged_cache: Option<vasr_paged_attn::SharedPagedCacheRuntime>,
}

pub type Qwen3Asr = Qwen3ASR;

impl Qwen3ASR {
    /// Canonical language names supported by the official Qwen3-ASR stack.
    pub fn supported_languages(&self) -> &'static [&'static str] {
        inference::utils::SUPPORTED_LANGUAGES
    }

    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let (config, model) =
            model::weights::load_model_from_pretrained(model_id_or_path, device, opts)?;
        let thinker_type = config
            .thinker_config
            .model_type
            .as_deref()
            .unwrap_or_default();
        if thinker_type.contains("forced_aligner") {
            anyhow::bail!(
                "loaded a forced aligner checkpoint (thinker_config.model_type={thinker_type:?}); use the forced aligner API instead"
            );
        }
        let tokenizer = processor::tokenizer::Tokenizer::from_pretrained(model_id_or_path)?;
        let processor = AsrProcessor::new(tokenizer);
        #[cfg(feature = "paged-attn")]
        let paged_cache = if device.is_metal() || device.is_cuda() {
            let (num_layers, num_kv_heads, head_dim) = model.thinker.paged_cache_config();
            let config = opts
                .paged_cache
                .unwrap_or_else(|| vasr_paged_attn::PagedCacheConfig {
                    block_size: 32,
                    memory: if device.is_cuda() {
                        vasr_paged_attn::PagedCacheMemory::GpuMemoryFraction {
                            fraction: 0.8,
                            reserved_bytes: 0,
                        }
                    } else {
                        vasr_paged_attn::PagedCacheMemory::ContextSize(100_000)
                    },
                });
            Some(Arc::new(std::sync::Mutex::new(
                vasr_paged_attn::PagedCacheRuntime::new(
                    num_layers,
                    num_kv_heads,
                    head_dim,
                    opts.dtype,
                    device,
                    config,
                )?,
            )))
        } else {
            None
        };
        let model_dir = model.model_dir;
        let weights_paths = model.weights_paths;
        let thinker = Arc::new(model.thinker);
        Ok(Self {
            device: Arc::new(device.clone()),
            config,
            processor: Arc::new(processor),
            model_dir,
            weights_paths,
            thinker,
            #[cfg(feature = "paged-attn")]
            paged_cache,
        })
    }

    pub fn device(&self) -> &Device {
        self.device.as_ref()
    }

    pub fn config(&self) -> &config::AsrConfig {
        &self.config
    }

    pub fn processor(&self) -> &AsrProcessor {
        self.processor.as_ref()
    }

    pub fn model_dir(&self) -> &PathBuf {
        &self.model_dir
    }

    pub fn weights_paths(&self) -> &[PathBuf] {
        self.weights_paths.as_slice()
    }

    pub fn thinker(&self) -> &model::thinker::ThinkerForConditionalGeneration {
        self.thinker.as_ref()
    }

    pub fn weights_memory_stats(
        &self,
        load_dtype: candle_core::DType,
    ) -> Result<WeightsMemoryStats> {
        model::weights::weights_memory_stats(&self.weights_paths, load_dtype)
    }

    #[cfg(feature = "paged-attn")]
    pub fn paged_cache_stats(&self) -> Option<PagedCacheStats> {
        self.paged_cache
            .as_ref()
            .and_then(|runtime| runtime.lock().ok().map(|runtime| runtime.stats()))
    }

    /// Pre-capture a vLLM-style padded CUDA decode graph at max_batch.
    #[cfg(all(feature = "paged-attn", feature = "cuda-graph"))]
    pub fn prewarm_cuda_decode_graphs(
        &self,
        max_batch: usize,
    ) -> Result<vasr_paged_attn::cuda_graph::CudaGraphPrewarmStats> {
        let Some(runtime) = self.paged_cache.as_ref() else {
            return Ok(vasr_paged_attn::cuda_graph::CudaGraphPrewarmStats::empty());
        };
        let mut guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        Ok(guard.prewarm_cuda_decode_graphs(
            self.thinker.as_ref(),
            self.device.as_ref(),
            max_batch,
        )?)
    }

    pub fn transcribe(
        &self,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<Vec<AsrTranscription>> {
        inference::transcribe::transcribe(
            self.thinker.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            #[cfg(feature = "paged-attn")]
            self.paged_cache.as_ref(),
            &audio,
            &opts,
        )
    }

    pub fn encode_audio_features_to_cpu(
        &self,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<EncodedAsrBatch> {
        inference::transcribe::encode_audio_features_to_cpu(
            self.thinker.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            #[cfg(feature = "paged-attn")]
            self.paged_cache.as_ref(),
            &audio,
            &opts,
        )
    }

    pub fn decode_cpu_audio_features(
        &self,
        encoded: EncodedAsrBatch,
        opts: TranscribeOptions,
    ) -> Result<Vec<AsrTranscription>> {
        inference::transcribe::decode_cpu_audio_features(
            self.thinker.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            #[cfg(feature = "paged-attn")]
            self.paged_cache.as_ref(),
            encoded,
            &opts,
        )
    }

    #[cfg(feature = "timing")]
    pub fn decode_cpu_audio_features_timed(
        &self,
        encoded: EncodedAsrBatch,
        opts: TranscribeOptions,
    ) -> Result<(Vec<AsrTranscription>, TranscribeTimings)> {
        inference::transcribe::decode_cpu_audio_features_timed(
            self.thinker.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            #[cfg(feature = "paged-attn")]
            self.paged_cache.as_ref(),
            encoded,
            &opts,
        )
    }

    #[cfg(feature = "paged-attn")]
    pub fn decode_cpu_audio_features_dynamic(
        &self,
        encoded: EncodedAsrBatch,
        mut poll_encoded: impl FnMut(bool) -> Result<EncodedAsrBatchPoll>,
        opts: TranscribeOptions,
    ) -> Result<Vec<Vec<AsrTranscription>>> {
        inference::transcribe::decode_cpu_audio_features_dynamic(
            self.thinker.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            self.paged_cache.as_ref(),
            encoded,
            &mut poll_encoded,
            &opts,
        )
    }

    #[cfg(all(feature = "paged-attn", feature = "timing"))]
    pub fn decode_cpu_audio_features_dynamic_timed(
        &self,
        encoded: EncodedAsrBatch,
        mut poll_encoded: impl FnMut(bool) -> Result<EncodedAsrBatchPoll>,
        opts: TranscribeOptions,
    ) -> Result<(Vec<Vec<AsrTranscription>>, TranscribeTimings)> {
        inference::transcribe::decode_cpu_audio_features_dynamic_timed(
            self.thinker.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            self.paged_cache.as_ref(),
            encoded,
            &mut poll_encoded,
            &opts,
        )
    }

    #[cfg(feature = "timing")]
    pub fn transcribe_timed(
        &self,
        audio: Vec<AudioInput<'_>>,
        opts: TranscribeOptions,
    ) -> Result<(Vec<AsrTranscription>, TranscribeTimings)> {
        inference::transcribe::transcribe_timed(
            self.thinker.as_ref(),
            self.processor.as_ref(),
            self.device.as_ref(),
            #[cfg(feature = "paged-attn")]
            self.paged_cache.as_ref(),
            &audio,
            &opts,
        )
    }

    #[cfg(feature = "timing")]
    pub fn decode_text_timed(
        &self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<(String, u64, usize, usize)> {
        let (text, timings) = self.decode_text_timed_with_metrics(prompt, max_new_tokens, true)?;
        Ok((
            text,
            timings.decode_us,
            timings.steps,
            timings.tokens_generated,
        ))
    }

    #[cfg(feature = "timing")]
    pub fn decode_text_timed_with_metrics(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        stop_at_eos: bool,
    ) -> Result<(String, GenerationTimings)> {
        if prompt.is_empty() {
            return Ok((String::new(), GenerationTimings::default()));
        }

        let prompt = if max_new_tokens == 0 {
            format!("{prompt} ")
        } else {
            prompt.to_string()
        };
        let input_ids = self.processor.tokenizer.encode(&prompt)?;
        if input_ids.is_empty() {
            return Ok((String::new(), GenerationTimings::default()));
        }

        let attention_mask = vec![1u32; input_ids.len()];

        let eos_token_ids = if stop_at_eos {
            vec![
                self.processor
                    .tokenizer
                    .token_to_id(processor::chat_template::IM_END)?,
                self.processor.tokenizer.token_to_id("<|endoftext|>")?,
            ]
        } else {
            vec![u32::MAX]
        };
        let eos_token_ids = {
            let mut ids = eos_token_ids;
            ids.sort_unstable();
            ids.dedup();
            ids
        };

        let input_rows: [&[u32]; 1] = [input_ids.as_slice()];
        let attention_rows: [&[u32]; 1] = [attention_mask.as_slice()];

        let (generated, timings) = {
            #[cfg(feature = "paged-attn")]
            {
                model::generation::greedy_generate_cached_batch_timed_with_paged_runtime(
                    self.thinker.as_ref(),
                    self.device.as_ref(),
                    input_rows.as_slice(),
                    attention_rows.as_slice(),
                    None,
                    max_new_tokens,
                    eos_token_ids.as_slice(),
                    self.paged_cache.as_ref(),
                )?
            }
            #[cfg(not(feature = "paged-attn"))]
            {
                model::generation::greedy_generate_cached_batch_timed(
                    self.thinker.as_ref(),
                    self.device.as_ref(),
                    input_rows.as_slice(),
                    attention_rows.as_slice(),
                    None,
                    max_new_tokens,
                    eos_token_ids.as_slice(),
                )?
            }
        };

        let gen_ids = generated.into_iter().next().unwrap_or_default();
        let text = self.processor.tokenizer.decode(gen_ids.as_slice())?;
        Ok((text, timings))
    }

    pub fn new_stream(&self, opts: StreamOptions) -> Result<AsrStream> {
        inference::streaming::new_stream(
            Arc::clone(&self.thinker),
            Arc::clone(&self.processor),
            Arc::clone(&self.device),
            &opts,
        )
    }

    pub fn require_ready(&self) -> Result<()> {
        self.processor.require_ready()
    }
}
