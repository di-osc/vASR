//! Candle implementation of the SenseVoiceSmall online CTC recognizer.
//!
//! The network and streaming behavior follow fasr's `sensevoice_online` model:
//! retained OnlineFbank/LFR features are re-encoded when enough new frames are
//! available, followed by greedy CTC decoding of the cumulative transcript.

mod frontend;
mod model;
mod tokenizer;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use vasr_data::TimeRange;

use crate::{AsrOptions, StreamEvent, StreamingAsrModel, StreamingAsrSession};

use frontend::SenseVoiceFrontend;
use model::SenseVoiceModel;
use tokenizer::SenseVoiceTokenizer;

const SENSEVOICE_REPO: &str = "iic/SenseVoiceSmall";
const SENSEVOICE_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_CHUNK_SIZE: usize = 4;

#[derive(Debug, Clone)]
pub struct SenseVoiceLoadOptions {
    pub dtype: DType,
    /// Number of newly available 60 ms LFR frames required before decoding.
    pub chunk_size: usize,
    /// Maximum retained LFR frames. Zero keeps the complete stream history.
    pub max_history: usize,
    pub language: String,
    pub textnorm: bool,
}

impl Default for SenseVoiceLoadOptions {
    fn default() -> Self {
        Self {
            dtype: DType::F32,
            chunk_size: DEFAULT_CHUNK_SIZE,
            max_history: 0,
            language: "zh".to_string(),
            textnorm: false,
        }
    }
}

pub struct SenseVoiceOnline {
    inner: Arc<SenseVoiceInner>,
    options: SenseVoiceLoadOptions,
}

struct SenseVoiceInner {
    model: SenseVoiceModel,
    frontend: SenseVoiceFrontend,
    tokenizer: SenseVoiceTokenizer,
    device: Device,
}

impl SenseVoiceOnline {
    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &Device,
        options: &SenseVoiceLoadOptions,
    ) -> Result<Self> {
        if options.chunk_size == 0 {
            bail!("SenseVoice chunk_size must be greater than zero");
        }
        let model_dir = resolve_model_dir(model_id_or_path)?;
        validate_model_dir(&model_dir)?;
        let frontend = SenseVoiceFrontend::new(&model_dir.join("am.mvn"))?;
        let tokenizer = SenseVoiceTokenizer::from_file(&model_dir.join("tokens.json"))?;
        let model = SenseVoiceModel::load(
            &model_dir.join("model.pt"),
            tokenizer.vocab_size(),
            options.dtype,
            device,
        )?;
        Ok(Self {
            inner: Arc::new(SenseVoiceInner {
                model,
                frontend,
                tokenizer,
                device: device.clone(),
            }),
            options: options.clone(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SENSEVOICE_SAMPLE_RATE
    }
}

impl StreamingAsrModel for SenseVoiceOnline {
    fn new_stream(&self, options: &AsrOptions) -> Result<Box<dyn StreamingAsrSession>> {
        let language = options
            .language
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(self.options.language.as_str())
            .to_string();
        validate_language(&language)?;
        Ok(Box::new(SenseVoiceSession {
            inner: Arc::clone(&self.inner),
            chunk_size: self.options.chunk_size,
            max_history: self.options.max_history,
            language,
            textnorm: self.options.textnorm,
            samples: Vec::new(),
            last_decoded_frames: 0,
            last_model_text: String::new(),
            finished: false,
        }))
    }
}

struct SenseVoiceSession {
    inner: Arc<SenseVoiceInner>,
    chunk_size: usize,
    max_history: usize,
    language: String,
    textnorm: bool,
    samples: Vec<f32>,
    last_decoded_frames: usize,
    last_model_text: String,
    finished: bool,
}

impl SenseVoiceSession {
    fn decode(&mut self, is_final: bool) -> Result<Option<String>> {
        let mut features = self.inner.frontend.extract(&self.samples, is_final)?;
        if is_final && features.is_empty() {
            features.push(vec![0.0; self.inner.frontend.feature_dim()]);
        }
        let total_frames = features.len();
        let new_frames = total_frames.saturating_sub(self.last_decoded_frames);
        if !is_final && new_frames < self.chunk_size {
            return Ok(None);
        }
        self.last_decoded_frames = total_frames;
        if features.is_empty() {
            return Ok(is_final.then_some(self.last_model_text.clone()));
        }
        if self.max_history > 0 && features.len() > self.max_history {
            features.drain(..features.len() - self.max_history);
        }

        let rows = features.len();
        let dim = self.inner.frontend.feature_dim();
        let flat = features.into_iter().flatten().collect::<Vec<_>>();
        let input = Tensor::from_vec(flat, (1, rows, dim), &self.inner.device)?
            .to_dtype(self.inner.model.dtype())?;
        let decoded = self.inner.model.transcribe(
            &input,
            &self.language,
            self.textnorm,
            &self.inner.tokenizer,
        )?;

        let changed = decoded.text != self.last_model_text;
        self.last_model_text = decoded.text;
        if is_final || changed {
            Ok(Some(self.last_model_text.clone()))
        } else {
            Ok(None)
        }
    }

    fn event(&self, text: String, is_final: bool) -> StreamEvent {
        StreamEvent::transcription(
            TimeRange::default(),
            text,
            Some(self.language.clone()),
            "sensevoice_online",
            is_final,
        )
    }
}

impl StreamingAsrSession for SenseVoiceSession {
    fn push_chunk(&mut self, chunk: &vasr_data::AudioChunk) -> Result<Vec<StreamEvent>> {
        if self.finished {
            bail!("SenseVoice stream is already finished");
        }
        if chunk.sample_rate != SENSEVOICE_SAMPLE_RATE {
            bail!(
                "SenseVoice streaming expects {SENSEVOICE_SAMPLE_RATE} Hz audio, got {} Hz",
                chunk.sample_rate
            );
        }
        self.samples.extend_from_slice(&chunk.samples);
        Ok(self
            .decode(false)?
            .map(|text| vec![self.event(text, false)])
            .unwrap_or_default())
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        if self.finished {
            bail!("SenseVoice stream is already finished");
        }
        self.finished = true;
        let text = self.decode(true)?.unwrap_or_default();
        Ok(vec![self.event(text, true)])
    }
}

fn validate_language(language: &str) -> Result<()> {
    match language.trim().to_ascii_lowercase().as_str() {
        "auto" | "zh" | "en" | "yue" | "ja" | "ko" | "nospeech" => Ok(()),
        _ => bail!("unsupported SenseVoice language {language:?}"),
    }
}

fn resolve_model_dir(model_id_or_path: &str) -> Result<PathBuf> {
    let path = Path::new(model_id_or_path);
    if path.exists() {
        return Ok(path.to_path_buf());
    }
    let repo = if model_id_or_path.trim().is_empty() {
        SENSEVOICE_REPO
    } else {
        model_id_or_path
    };
    let cache = crate::modelscope_cache_dir();
    let legacy = cache.join(repo);
    if legacy.exists() {
        return Ok(legacy);
    }
    let target = cache
        .join("models")
        .join(repo.replace('/', "--"))
        .join("snapshots")
        .join("master");
    if target.exists() {
        return Ok(target);
    }
    block_on_async(crate::modelscope::download_model(repo, &cache))
        .with_context(|| format!("failed to download SenseVoice model {repo:?}"))?;
    Ok(target)
}

fn validate_model_dir(path: &Path) -> Result<()> {
    for name in ["model.pt", "am.mvn", "tokens.json"] {
        let file = path.join(name);
        if !file.is_file() {
            bail!("SenseVoice model is missing {}", file.display());
        }
    }
    Ok(())
}

fn block_on_async<F: std::future::Future + Send>(future: F) -> F::Output
where
    F::Output: Send,
{
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
                rt.block_on(future)
            })
            .join()
            .expect("tokio download thread panicked")
    })
}
