use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use asr_data::{AudioChunk, TimeRange, Waveform};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias, ops::sigmoid};
use gemm::Parallelism;
use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};
use rayon::prelude::*;

use crate::traits::run_with_vad_threads;
use crate::{StreamingVadSession, VadEvent, VadEventStatus, VadModel, VadOptions, VadSegment};

const FIRERED_SAMPLE_RATE: u32 = 16_000;
const FIRERED_FRAME_SHIFT_MS: u64 = 10;
const FIRERED_FRAME_LENGTH_MS: u64 = 25;
const FIRERED_INPUT_DIM: usize = 80;
const FIRERED_HIDDEN_DIM: usize = 256;
const FIRERED_PROJ_DIM: usize = 128;
const FIRERED_BLOCKS: usize = 8;
const FIRERED_FSMN_ORDER: usize = 20;
const FIRERED_DEFAULT_THRESHOLD: f32 = 0.4;
const FIRERED_DEFAULT_MAX_SPEECH_FRAME: usize = 2000;
const FIRERED_VAD_REPO: &str = "xukaituo/FireRedVAD";
const FIRERED_OFFLINE_MODEL_DIR: &str = "VAD";
const FIRERED_STREAM_MODEL_DIR: &str = "Stream-VAD";

#[derive(Debug, Clone, Copy, Default)]
pub struct FireRedVadTiming {
    pub frontend_seconds: f64,
    pub forward_seconds: f64,
    pub postprocess_seconds: f64,
    pub frames: usize,
}

#[derive(Debug, Clone)]
pub struct FireRedVadDetection {
    pub segments: Vec<VadSegment>,
    pub timing: FireRedVadTiming,
}

#[derive(Debug, Clone)]
pub struct FireRedVadModel {
    weights: Arc<FireRedVadWeights>,
    model_dir: PathBuf,
    streaming: bool,
}

impl FireRedVadModel {
    pub fn from_pretrained(path: Option<impl AsRef<Path>>) -> Result<Self> {
        Self::from_pretrained_on_device(path, &firered_vad_device()?)
    }

    pub fn from_pretrained_streaming(path: Option<impl AsRef<Path>>) -> Result<Self> {
        Self::from_pretrained_streaming_on_device(path, &firered_vad_device()?)
    }

    pub fn from_pretrained_on_device(
        path: Option<impl AsRef<Path>>,
        device: &Device,
    ) -> Result<Self> {
        Self::load(path, device, false)
    }

    pub fn from_pretrained_streaming_on_device(
        path: Option<impl AsRef<Path>>,
        device: &Device,
    ) -> Result<Self> {
        Self::load(path, device, true)
    }

    fn load(path: Option<impl AsRef<Path>>, device: &Device, streaming: bool) -> Result<Self> {
        let model_subdir = if streaming {
            FIRERED_STREAM_MODEL_DIR
        } else {
            FIRERED_OFFLINE_MODEL_DIR
        };
        let model_dir = match path {
            Some(path) => resolve_firered_vad_model_dir(path.as_ref(), model_subdir)?,
            None => default_firered_vad_model_dir(model_subdir)?,
        };
        let weights = Arc::new(FireRedVadWeights::load_on_device(
            &model_dir, device, streaming,
        )?);
        Ok(Self {
            weights,
            model_dir,
            streaming,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn detect_with_timing(
        &self,
        waveform: &Waveform,
        options: &VadOptions,
    ) -> Result<FireRedVadDetection> {
        if waveform.sample_rate != FIRERED_SAMPLE_RATE {
            bail!(
                "FireRedVAD expects 16kHz mono audio, got sample_rate={}",
                waveform.sample_rate
            );
        }
        if waveform.channels != 1 {
            bail!(
                "FireRedVAD expects mono audio, got channels={}",
                waveform.channels
            );
        }

        let mut timing = FireRedVadTiming::default();
        let frontend_start = Instant::now();
        let feats = self.weights.frontend.extract(&waveform.samples)?;
        timing.frontend_seconds = frontend_start.elapsed().as_secs_f64();
        timing.frames = feats.dim(0)?;
        if timing.frames == 0 {
            return Ok(FireRedVadDetection {
                segments: Vec::new(),
                timing,
            });
        }

        let forward_start = Instant::now();
        let probs = self.weights.forward_probs(&feats, options.num_threads)?;
        timing.forward_seconds = forward_start.elapsed().as_secs_f64();

        let post_start = Instant::now();
        let segments = FireRedVadPostprocessor::from_options(options)
            .process_to_segments(&probs, waveform.duration_seconds());
        timing.postprocess_seconds = post_start.elapsed().as_secs_f64();
        Ok(FireRedVadDetection { segments, timing })
    }

    pub fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        Ok(self.detect_with_timing(waveform, options)?.segments)
    }
}

impl VadModel for FireRedVadModel {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        FireRedVadModel::detect(self, waveform, options)
    }

    fn new_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadSession>> {
        if !self.streaming {
            bail!(
                "FireRedVAD offline weights do not support streaming; load the Stream-VAD checkpoint"
            );
        }
        Ok(Box::new(FireRedStreamingVadSession::new(
            Arc::clone(&self.weights),
            options,
        )))
    }
}

struct FireRedStreamingVadSession {
    weights: Arc<FireRedVadWeights>,
    options: VadOptions,
    samples: Vec<f32>,
    processed_frames: usize,
    forward_state: FireRedStreamForwardState,
    postprocessor: FireRedStreamPostprocessor,
}

impl FireRedStreamingVadSession {
    fn new(weights: Arc<FireRedVadWeights>, options: &VadOptions) -> Self {
        Self {
            weights,
            options: options.clone(),
            samples: Vec::new(),
            processed_frames: 0,
            forward_state: FireRedStreamForwardState::default(),
            postprocessor: FireRedStreamPostprocessor::from_options(options),
        }
    }

    fn process_available_frames(&mut self) -> Result<Vec<VadEvent>> {
        let feats = self.weights.frontend.extract(&self.samples)?;
        let frame_count = feats.dim(0)?;
        if frame_count <= self.processed_frames {
            return Ok(Vec::new());
        }
        let probs = if self.weights.cpu.is_some() {
            // 流式 CPU 路径只消费尚未处理过的帧，切出连续块即可。
            let new_feats = feats
                .narrow(
                    0,
                    self.processed_frames,
                    frame_count - self.processed_frames,
                )?
                .contiguous()?;
            self.weights.forward_probs_streaming(
                &new_feats,
                self.options.num_threads,
                &mut self.forward_state,
            )?
        } else {
            self.weights
                .forward_probs(&feats, self.options.num_threads)?
                .into_iter()
                .skip(self.processed_frames)
                .collect()
        };
        let mut events = Vec::new();
        for probability in probs {
            for boundary in self.postprocessor.process(probability) {
                events.push(boundary.into_event());
            }
        }
        self.processed_frames = frame_count;
        Ok(events)
    }
}

#[derive(Default)]
struct FireRedStreamForwardState {
    fsmn_caches: Vec<Vec<f32>>,
}

impl StreamingVadSession for FireRedStreamingVadSession {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<VadEvent>> {
        if chunk.sample_rate != FIRERED_SAMPLE_RATE {
            bail!(
                "FireRed Stream-VAD expects 16kHz mono audio, got sample_rate={}",
                chunk.sample_rate
            );
        }
        if chunk.channels != 1 {
            bail!(
                "FireRed Stream-VAD expects mono audio, got channels={}",
                chunk.channels
            );
        }
        self.samples.extend_from_slice(&chunk.samples);
        self.process_available_frames()
    }

    fn finish(&mut self) -> Result<Vec<VadEvent>> {
        let mut events = self.process_available_frames()?;
        if let Some(start_frame) = self.postprocessor.finish() {
            let end_ms = samples_to_ms(self.samples.len());
            let start_ms = frame_to_ms(start_frame).min(end_ms);
            events.push(VadEvent {
                range: TimeRange::new(start_ms, end_ms),
                status: VadEventStatus::Final,
            });
        }
        Ok(events)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FireRedStreamState {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FireRedStreamBoundary {
    Start {
        start_frame: usize,
        frame: usize,
    },
    End {
        start_frame: usize,
        end_frame: usize,
    },
}

impl FireRedStreamBoundary {
    fn into_event(self) -> VadEvent {
        match self {
            Self::Start { start_frame, frame } => VadEvent {
                range: TimeRange::new(frame_to_ms(start_frame), frame_to_ms(frame)),
                status: VadEventStatus::Partial,
            },
            Self::End {
                start_frame,
                end_frame,
            } => VadEvent {
                range: TimeRange::new(frame_to_ms(start_frame), frame_to_ms(end_frame)),
                status: VadEventStatus::Final,
            },
        }
    }
}

struct FireRedStreamPostprocessor {
    smooth_window_size: usize,
    speech_threshold: f32,
    pad_start_frame: usize,
    min_speech_frame: usize,
    max_speech_frame: usize,
    min_silence_frame: usize,
    smooth_window: VecDeque<f32>,
    smooth_window_sum: f32,
    state: FireRedStreamState,
    frame_count: usize,
    speech_count: usize,
    silence_count: usize,
    hit_max_speech: bool,
    last_speech_start_frame: Option<usize>,
    last_speech_end_frame: Option<usize>,
}

impl FireRedStreamPostprocessor {
    fn from_options(options: &VadOptions) -> Self {
        let smooth_window_size = 5;
        Self {
            smooth_window_size,
            speech_threshold: options.threshold,
            pad_start_frame: smooth_window_size.max(ms_to_frame_count_allow_zero(options.pad_ms)),
            min_speech_frame: ms_to_frame_count(options.min_speech_ms),
            max_speech_frame: ms_to_frame_count(options.max_segment_ms),
            min_silence_frame: ms_to_frame_count(options.min_silence_ms),
            smooth_window: VecDeque::with_capacity(smooth_window_size),
            smooth_window_sum: 0.0,
            state: FireRedStreamState::Silence,
            frame_count: 0,
            speech_count: 0,
            silence_count: 0,
            hit_max_speech: false,
            last_speech_start_frame: None,
            last_speech_end_frame: None,
        }
    }

    fn process(&mut self, probability: f32) -> Vec<FireRedStreamBoundary> {
        self.frame_count += 1;
        self.smooth_window.push_back(probability);
        self.smooth_window_sum += probability;
        if self.smooth_window.len() > self.smooth_window_size
            && let Some(first) = self.smooth_window.pop_front()
        {
            self.smooth_window_sum -= first;
        }
        let smoothed = self.smooth_window_sum / self.smooth_window.len() as f32;
        let is_speech = smoothed >= self.speech_threshold;
        let mut boundaries = Vec::new();

        if self.hit_max_speech {
            self.last_speech_start_frame = Some(self.frame_count);
            boundaries.push(FireRedStreamBoundary::Start {
                start_frame: self.frame_count,
                frame: self.frame_count,
            });
            self.hit_max_speech = false;
        }

        match self.state {
            FireRedStreamState::Silence => {
                if is_speech {
                    self.state = FireRedStreamState::PossibleSpeech;
                    self.speech_count += 1;
                } else {
                    self.silence_count += 1;
                    self.speech_count = 0;
                }
            }
            FireRedStreamState::PossibleSpeech => {
                if is_speech {
                    self.speech_count += 1;
                    if self.speech_count >= self.min_speech_frame {
                        self.state = FireRedStreamState::Speech;
                        let start_frame = self
                            .frame_count
                            .saturating_sub(self.speech_count - 1)
                            .saturating_sub(self.pad_start_frame)
                            .max(1)
                            .max(self.last_speech_end_frame.map_or(1, |frame| frame + 1));
                        self.last_speech_start_frame = Some(start_frame);
                        self.silence_count = 0;
                        boundaries.push(FireRedStreamBoundary::Start {
                            start_frame,
                            frame: self.frame_count,
                        });
                    }
                } else {
                    self.state = FireRedStreamState::Silence;
                    self.silence_count = 1;
                    self.speech_count = 0;
                }
            }
            FireRedStreamState::Speech => {
                self.speech_count += 1;
                if is_speech {
                    self.silence_count = 0;
                    self.maybe_end_max_speech(&mut boundaries);
                } else {
                    self.state = FireRedStreamState::PossibleSilence;
                    self.silence_count += 1;
                }
            }
            FireRedStreamState::PossibleSilence => {
                self.speech_count += 1;
                if is_speech {
                    self.state = FireRedStreamState::Speech;
                    self.silence_count = 0;
                    self.maybe_end_max_speech(&mut boundaries);
                } else {
                    self.silence_count += 1;
                    if self.silence_count >= self.min_silence_frame {
                        self.state = FireRedStreamState::Silence;
                        self.end_speech(self.frame_count, &mut boundaries);
                        self.speech_count = 0;
                    }
                }
            }
        }
        boundaries
    }

    fn maybe_end_max_speech(&mut self, boundaries: &mut Vec<FireRedStreamBoundary>) {
        if self.speech_count >= self.max_speech_frame {
            self.hit_max_speech = true;
            self.speech_count = 0;
            self.end_speech(self.frame_count, boundaries);
        }
    }

    fn end_speech(&mut self, end_frame: usize, boundaries: &mut Vec<FireRedStreamBoundary>) {
        if let Some(start_frame) = self.last_speech_start_frame.take() {
            boundaries.push(FireRedStreamBoundary::End {
                start_frame,
                end_frame,
            });
            self.last_speech_end_frame = Some(end_frame);
        }
    }

    fn finish(&mut self) -> Option<usize> {
        self.hit_max_speech = false;
        self.last_speech_start_frame.take()
    }
}

#[derive(Debug)]
struct FireRedVadWeights {
    frontend: FireRedFrontend,
    fc1: Linear,
    fc2: Linear,
    fsmn1: FireRedFsmn,
    blocks: Vec<FireRedDfsmnBlock>,
    dnn: Linear,
    out: Linear,
    device: Device,
    cpu: Option<FireRedCpuWeights>,
}

#[derive(Debug)]
struct FireRedCpuWeights {
    fc1: CpuLinear,
    fc2: CpuLinear,
    fsmn1: FireRedCpuFsmn,
    blocks: Vec<FireRedCpuBlock>,
    dnn: CpuLinear,
    out: CpuLinear,
}

#[derive(Debug)]
struct FireRedCpuBlock {
    fc1: CpuLinear,
    fc2: CpuLinear,
    fsmn: FireRedCpuFsmn,
}

#[derive(Debug, Clone)]
struct FireRedCpuFsmn {
    lookback: Vec<f32>,
    lookahead: Option<Vec<f32>>,
}

#[derive(Debug)]
struct CpuLinear {
    weight_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

impl FireRedVadWeights {
    fn load_on_device(model_dir: &Path, device: &Device, causal: bool) -> Result<Self> {
        let cmvn_path = model_dir.join("cmvn.ark");
        let frontend = FireRedFrontend::new(&cmvn_path)?;
        let pth_path = model_dir.join("model.pth.tar");
        let safetensors_path = model_dir.join("model.safetensors");
        let vb = if nonempty_file(&safetensors_path)? {
            unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, device) }
                .with_context(|| format!("failed to load {}", model_dir.display()))?
        } else if nonempty_file(&pth_path)? {
            VarBuilder::from_pth_with_state(&pth_path, DType::F32, "model_state_dict", device)
                .with_context(|| format!("failed to load {}", pth_path.display()))?
        } else {
            bail!(
                "FireRedVAD model directory is incomplete: {} (expected model.pth.tar or model.safetensors)",
                model_dir.display()
            );
        };

        let dfsmn = vb.pp("dfsmn");
        let fc1 = linear(
            FIRERED_INPUT_DIM,
            FIRERED_HIDDEN_DIM,
            dfsmn.pp("fc1").pp("0"),
        )?;
        let fc2 = linear(
            FIRERED_HIDDEN_DIM,
            FIRERED_PROJ_DIM,
            dfsmn.pp("fc2").pp("0"),
        )?;
        let fsmn1 = FireRedFsmn::new(dfsmn.pp("fsmn1"), causal)?;
        let mut blocks = Vec::with_capacity(FIRERED_BLOCKS - 1);
        for i in 0..FIRERED_BLOCKS - 1 {
            blocks.push(FireRedDfsmnBlock::new(dfsmn.pp("fsmns").pp(i), causal)?);
        }
        let dnn = linear(
            FIRERED_PROJ_DIM,
            FIRERED_HIDDEN_DIM,
            dfsmn.pp("dnns").pp("0"),
        )?;
        let out = linear(FIRERED_HIDDEN_DIM, 1, vb.pp("out"))?;
        let cpu = if device.is_cpu() {
            Some(FireRedCpuWeights {
                fc1: CpuLinear::from_linear(&fc1)?,
                fc2: CpuLinear::from_linear(&fc2)?,
                fsmn1: FireRedCpuFsmn::from_fsmn(&fsmn1)?,
                blocks: blocks
                    .iter()
                    .map(FireRedCpuBlock::from_block)
                    .collect::<Result<Vec<_>>>()?,
                dnn: CpuLinear::from_linear(&dnn)?,
                out: CpuLinear::from_linear(&out)?,
            })
        } else {
            None
        };
        Ok(Self {
            frontend,
            fc1,
            fc2,
            fsmn1,
            blocks,
            dnn,
            out,
            device: device.clone(),
            cpu,
        })
    }

    fn forward_probs(&self, feats: &Tensor, num_threads: usize) -> Result<Vec<f32>> {
        let dim = feats.dim(1)?;
        if dim != FIRERED_INPUT_DIM {
            bail!("FireRedVAD expects feature dim {FIRERED_INPUT_DIM}, got {dim}");
        }
        if let Some(cpu) = &self.cpu {
            return run_with_vad_threads(num_threads, || cpu.forward_probs(feats))?;
        }
        // 前端特征固定在 CPU，这里先补上 batch 维再搬到模型所在设备。
        let x = feats.unsqueeze(0)?.contiguous()?.to_device(&self.device)?;
        let mut x = self.fc1.forward(&x)?.relu()?;
        x = self.fc2.forward(&x)?.relu()?;
        x = self.fsmn1.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x = self.dnn.forward(&x)?.relu()?;
        x = sigmoid(&self.out.forward(&x)?)?;
        Ok(x.squeeze(0)?.squeeze(1)?.to_vec1::<f32>()?)
    }

    fn forward_probs_streaming(
        &self,
        feats: &Tensor,
        num_threads: usize,
        state: &mut FireRedStreamForwardState,
    ) -> Result<Vec<f32>> {
        let cpu = self
            .cpu
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("streaming CPU fast path is unavailable"))?;
        run_with_vad_threads(num_threads, || {
            cpu.forward_probs_streaming(feats, &mut state.fsmn_caches)
        })?
    }
}

#[derive(Debug)]
struct FireRedDfsmnBlock {
    fc1: Linear,
    fc2: Linear,
    fsmn: FireRedFsmn,
}

impl FireRedDfsmnBlock {
    fn new(vb: VarBuilder, causal: bool) -> Result<Self> {
        Ok(Self {
            fc1: linear(FIRERED_PROJ_DIM, FIRERED_HIDDEN_DIM, vb.pp("fc1").pp("0"))?,
            fc2: linear_no_bias(FIRERED_HIDDEN_DIM, FIRERED_PROJ_DIM, vb.pp("fc2"))?,
            fsmn: FireRedFsmn::new(vb.pp("fsmn"), causal)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let residual = input;
        let x = self.fc1.forward(input)?.relu()?;
        let x = self.fc2.forward(&x)?;
        Ok((self.fsmn.forward(&x)? + residual)?)
    }
}

impl FireRedCpuWeights {
    fn forward_probs(&self, feats: &Tensor) -> Result<Vec<f32>> {
        let frames = feats.dim(0)?;
        let mut x = feats.flatten_all()?.to_vec1::<f32>()?;
        x = self.fc1.forward(&x, frames, true);
        x = self.fc2.forward(&x, frames, true);
        self.fsmn1.apply(&mut x, frames);
        for block in &self.blocks {
            let residual = x;
            let mut y = block.fc1.forward(&residual, frames, true);
            y = block.fc2.forward(&y, frames, false);
            block.fsmn.apply(&mut y, frames);
            y.par_iter_mut()
                .zip(residual.par_iter())
                .for_each(|(value, residual)| *value += *residual);
            x = y;
        }
        x = self.dnn.forward(&x, frames, true);
        let logits = self.out.forward(&x, frames, false);
        Ok(logits
            .into_iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect())
    }

    /// 流式 CPU 前向：只对新增帧计算，FSMN 卷积状态通过 `caches` 跨调用累积。
    ///
    /// # Errors
    ///
    /// 输入不是二维张量或无法物化成连续 `f32` 数据时返回错误。
    fn forward_probs_streaming(
        &self,
        feats: &Tensor,
        caches: &mut Vec<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let frames = feats.dim(0)?;
        if frames == 0 {
            return Ok(Vec::new());
        }
        if caches.len() != FIRERED_BLOCKS {
            *caches = vec![Vec::new(); FIRERED_BLOCKS];
        }
        let mut x = feats.flatten_all()?.to_vec1::<f32>()?;
        x = self.fc1.forward(&x, frames, true);
        x = self.fc2.forward(&x, frames, true);
        self.fsmn1.apply_streaming(&mut x, frames, &mut caches[0]);
        for (index, block) in self.blocks.iter().enumerate() {
            let residual = x;
            let mut y = block.fc1.forward(&residual, frames, true);
            y = block.fc2.forward(&y, frames, false);
            block
                .fsmn
                .apply_streaming(&mut y, frames, &mut caches[index + 1]);
            y.par_iter_mut()
                .zip(residual.par_iter())
                .for_each(|(value, residual)| *value += *residual);
            x = y;
        }
        x = self.dnn.forward(&x, frames, true);
        Ok(self
            .out
            .forward(&x, frames, false)
            .into_iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect())
    }
}

impl FireRedCpuBlock {
    fn from_block(block: &FireRedDfsmnBlock) -> Result<Self> {
        Ok(Self {
            fc1: CpuLinear::from_linear(&block.fc1)?,
            fc2: CpuLinear::from_linear(&block.fc2)?,
            fsmn: FireRedCpuFsmn::from_fsmn(&block.fsmn)?,
        })
    }
}

impl FireRedCpuFsmn {
    fn from_fsmn(fsmn: &FireRedFsmn) -> Result<Self> {
        Ok(Self {
            lookback: fsmn
                .lookback_cpu
                .clone()
                .ok_or_else(|| anyhow::anyhow!("FireRedVAD CPU FSMN missing lookback weights"))?,
            lookahead: fsmn.lookahead_cpu.clone(),
        })
    }

    /// 对 `input` 就地做 FSMN lookback / lookahead 卷积。
    ///
    /// 参数取切片而非 `&mut Vec`，避免调用方为借用检查额外构造容器。
    fn apply(&self, input: &mut [f32], frames: usize) {
        let src = input.to_vec();
        input
            .par_chunks_mut(FIRERED_PROJ_DIM)
            .enumerate()
            .for_each(|(t, dst)| {
                for (c, value) in dst.iter_mut().enumerate() {
                    let kernel = c * FIRERED_FSMN_ORDER;
                    let mut sum = src[t * FIRERED_PROJ_DIM + c];
                    for k in 0..FIRERED_FSMN_ORDER {
                        if let Some(source_t) = (t + k).checked_sub(FIRERED_FSMN_ORDER - 1)
                            && source_t < frames
                        {
                            sum += src[source_t * FIRERED_PROJ_DIM + c] * self.lookback[kernel + k];
                        }
                        let ahead_t = t + k + 1;
                        if let Some(lookahead) = &self.lookahead
                            && frames > 1
                            && ahead_t < frames
                        {
                            sum += src[ahead_t * FIRERED_PROJ_DIM + c] * lookahead[kernel + k];
                        }
                    }
                    *value = sum;
                }
            });
    }

    fn apply_streaming(&self, input: &mut Vec<f32>, frames: usize, cache: &mut Vec<f32>) {
        debug_assert!(self.lookahead.is_none());
        let cache_frames = cache.len() / FIRERED_PROJ_DIM;
        let mut combined = Vec::with_capacity(cache.len() + input.len());
        combined.extend_from_slice(cache);
        combined.extend_from_slice(input);
        let mut out = input.clone();
        out.par_chunks_mut(FIRERED_PROJ_DIM)
            .enumerate()
            .for_each(|(t, dst)| {
                let combined_t = cache_frames + t;
                for (channel, value) in dst.iter_mut().enumerate() {
                    let kernel = channel * FIRERED_FSMN_ORDER;
                    let mut sum = combined[combined_t * FIRERED_PROJ_DIM + channel];
                    for k in 0..FIRERED_FSMN_ORDER {
                        if let Some(source_t) = (combined_t + k).checked_sub(FIRERED_FSMN_ORDER - 1)
                        {
                            sum += combined[source_t * FIRERED_PROJ_DIM + channel]
                                * self.lookback[kernel + k];
                        }
                    }
                    *value = sum;
                }
            });
        let retained_frames = (cache_frames + frames).min(FIRERED_FSMN_ORDER - 1);
        let retained_values = retained_frames * FIRERED_PROJ_DIM;
        *cache = combined[combined.len() - retained_values..].to_vec();
        *input = out;
    }
}

impl CpuLinear {
    fn from_linear(linear: &Linear) -> Result<Self> {
        let dims = linear.weight().dims();
        if dims.len() != 2 {
            bail!("FireRedVAD linear weight must be 2D, got {dims:?}");
        }
        let out_dim = dims[0];
        let in_dim = dims[1];
        let weight = linear.weight().flatten_all()?.to_vec1::<f32>()?;
        let mut weight_t = vec![0.0; weight.len()];
        for out_idx in 0..out_dim {
            for in_idx in 0..in_dim {
                weight_t[in_idx * out_dim + out_idx] = weight[out_idx * in_dim + in_idx];
            }
        }
        Ok(Self {
            weight_t,
            bias: linear
                .bias()
                .map(|bias| bias.flatten_all()?.to_vec1::<f32>())
                .transpose()?,
            in_dim,
            out_dim,
        })
    }

    fn forward(&self, input: &[f32], rows: usize, relu: bool) -> Vec<f32> {
        let mut out = vec![0.0; rows * self.out_dim];
        let threads = rayon::current_num_threads();
        unsafe {
            gemm::gemm(
                rows,
                self.out_dim,
                self.in_dim,
                out.as_mut_ptr(),
                1,
                self.out_dim as isize,
                false,
                input.as_ptr(),
                1,
                self.in_dim as isize,
                self.weight_t.as_ptr(),
                1,
                self.out_dim as isize,
                0.0,
                1.0,
                false,
                false,
                false,
                if threads > 1 {
                    Parallelism::Rayon(threads)
                } else {
                    Parallelism::None
                },
            );
        }
        if let Some(bias) = &self.bias {
            out.par_chunks_mut(self.out_dim).for_each(|row| {
                for idx in 0..self.out_dim {
                    row[idx] += bias[idx];
                }
            });
        }
        if relu {
            out.par_iter_mut().for_each(|value| {
                if *value < 0.0 {
                    *value = 0.0;
                }
            });
        }
        out
    }
}

#[derive(Debug)]
struct FireRedFsmn {
    lookback_weight: Tensor,
    lookahead_weight: Option<Tensor>,
    lookback_cpu: Option<Vec<f32>>,
    lookahead_cpu: Option<Vec<f32>>,
}

impl FireRedFsmn {
    fn new(vb: VarBuilder, causal: bool) -> Result<Self> {
        let lookback_weight = vb
            .pp("lookback_filter")
            .get((FIRERED_PROJ_DIM, 1, FIRERED_FSMN_ORDER), "weight")?;
        let lookahead_weight = if causal {
            None
        } else {
            Some(
                vb.pp("lookahead_filter")
                    .get((FIRERED_PROJ_DIM, 1, FIRERED_FSMN_ORDER), "weight")?,
            )
        };
        let lookback_cpu = tensor_cpu_vec(&lookback_weight)?;
        let lookahead_cpu = lookahead_weight
            .as_ref()
            .map(tensor_cpu_vec)
            .transpose()?
            .flatten();
        Ok(Self {
            lookback_weight,
            lookahead_weight,
            lookback_cpu,
            lookahead_cpu,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.device().is_cpu()
            && let Some(lookback) = &self.lookback_cpu
        {
            return firered_fsmn_cpu(input, lookback, self.lookahead_cpu.as_deref());
        }

        let x = input.permute((0, 2, 1))?;
        let lookback = depthwise_conv1d_left(&x, &self.lookback_weight)?;
        let mut memory = x.broadcast_add(&lookback)?;
        if x.dim(2)? > 1
            && let Some(lookahead_weight) = &self.lookahead_weight
        {
            let lookahead = depthwise_conv1d_ahead(&x, lookahead_weight)?;
            memory = memory.broadcast_add(&lookahead)?;
        }
        Ok(memory.permute((0, 2, 1))?)
    }
}

fn tensor_cpu_vec(tensor: &Tensor) -> Result<Option<Vec<f32>>> {
    if tensor.device().is_cpu() {
        Ok(Some(tensor.flatten_all()?.to_vec1::<f32>()?))
    } else {
        Ok(None)
    }
}

fn depthwise_conv1d_left(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let padded = input.pad_with_zeros(2, FIRERED_FSMN_ORDER - 1, 0)?;
    let windows = padded.unfold(2, FIRERED_FSMN_ORDER, 1)?;
    let kernel = weight.reshape((1usize, FIRERED_PROJ_DIM, 1usize, FIRERED_FSMN_ORDER))?;
    Ok(windows.broadcast_mul(&kernel)?.sum_keepdim(3)?.squeeze(3)?)
}

fn depthwise_conv1d_ahead(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let padded = input.pad_with_zeros(2, 0, FIRERED_FSMN_ORDER)?;
    let narrowed = padded.narrow(2, 1, input.dim(2)? + FIRERED_FSMN_ORDER - 1)?;
    let windows = narrowed.unfold(2, FIRERED_FSMN_ORDER, 1)?;
    let kernel = weight.reshape((1usize, FIRERED_PROJ_DIM, 1usize, FIRERED_FSMN_ORDER))?;
    Ok(windows.broadcast_mul(&kernel)?.sum_keepdim(3)?.squeeze(3)?)
}

fn firered_fsmn_cpu(input: &Tensor, lookback: &[f32], lookahead: Option<&[f32]>) -> Result<Tensor> {
    let (batch, frames, channels) = input.dims3()?;
    if batch != 1 || channels != FIRERED_PROJ_DIM {
        bail!("unexpected FireRed FSMN input shape: {:?}", input.dims());
    }
    let src = input.flatten_all()?.to_vec1::<f32>()?;
    let mut out = src.clone();
    for t in 0..frames {
        for c in 0..channels {
            let dst = t * channels + c;
            let kernel = c * FIRERED_FSMN_ORDER;
            for k in 0..FIRERED_FSMN_ORDER {
                if let Some(source_t) = (t + k).checked_sub(FIRERED_FSMN_ORDER - 1)
                    && source_t < frames
                {
                    out[dst] += src[source_t * channels + c] * lookback[kernel + k];
                }
                let ahead_t = t + k + 1;
                if let Some(lookahead) = lookahead
                    && frames > 1
                    && ahead_t < frames
                {
                    out[dst] += src[ahead_t * channels + c] * lookahead[kernel + k];
                }
            }
        }
    }
    Ok(Tensor::from_vec(
        out,
        (1usize, frames, channels),
        input.device(),
    )?)
}

#[derive(Debug, Clone)]
struct FireRedFrontend {
    means: Vec<f32>,
    inverse_std: Vec<f32>,
}

impl FireRedFrontend {
    fn new(cmvn_path: &Path) -> Result<Self> {
        let (means, inverse_std) = read_kaldi_binary_cmvn(cmvn_path)?;
        if means.len() != FIRERED_INPUT_DIM || inverse_std.len() != FIRERED_INPUT_DIM {
            bail!(
                "FireRedVAD CMVN expects {FIRERED_INPUT_DIM} dims, got means={} vars={}",
                means.len(),
                inverse_std.len()
            );
        }
        Ok(Self { means, inverse_std })
    }

    /// 从归一化到 `[-1, 1]` 的采样提取 fbank + CMVN 特征。
    ///
    /// 返回 CPU 上的 `(frames, FIRERED_INPUT_DIM)` F32 张量；是否搬运到计算设备
    /// 由 `forward_probs*` 根据模型所在设备决定。
    fn extract(&self, samples: &[f32]) -> Result<Tensor> {
        let waveform = samples
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
            .collect::<Vec<_>>();
        let opt = FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: FIRERED_SAMPLE_RATE as f32,
                dither: 0.0,
                frame_shift_ms: FIRERED_FRAME_SHIFT_MS as f32,
                frame_length_ms: FIRERED_FRAME_LENGTH_MS as f32,
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: FIRERED_INPUT_DIM as i32,
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };
        let mut fbank = OnlineFbank::new(opt);
        fbank.accept_waveform(FIRERED_SAMPLE_RATE as f32, &waveform);
        let frames = fbank.num_ready_frames() as usize;
        let mut out = Vec::with_capacity(frames * FIRERED_INPUT_DIM);
        for i in 0..frames as i32 {
            let frame = fbank
                .get_frame(i)
                .ok_or_else(|| anyhow::anyhow!("missing FireRed fbank frame {i}"))?;
            for (d, value) in frame.iter().enumerate() {
                out.push((*value - self.means[d]) * self.inverse_std[d]);
            }
        }
        Ok(Tensor::from_vec(
            out,
            (frames, FIRERED_INPUT_DIM),
            &Device::Cpu,
        )?)
    }
}

#[derive(Debug, Clone)]
struct FireRedVadPostprocessor {
    smooth_window_size: usize,
    prob_threshold: f32,
    min_speech_ms: u64,
    min_speech_frame: usize,
    max_speech_frame: usize,
    min_silence_frame: usize,
    merge_silence_frame: usize,
    extend_speech_frame: usize,
}

impl FireRedVadPostprocessor {
    fn from_options(options: &VadOptions) -> Self {
        Self {
            smooth_window_size: 5,
            prob_threshold: if options.threshold > 0.0 {
                options.threshold
            } else {
                FIRERED_DEFAULT_THRESHOLD
            },
            min_speech_ms: options.min_speech_ms,
            min_speech_frame: ms_to_frame_count(options.min_speech_ms),
            max_speech_frame: if options.max_segment_ms > 0 {
                ms_to_frame_count(options.max_segment_ms)
            } else {
                FIRERED_DEFAULT_MAX_SPEECH_FRAME
            },
            min_silence_frame: ms_to_frame_count(options.min_silence_ms),
            merge_silence_frame: 0,
            extend_speech_frame: 0,
        }
    }

    fn process_to_segments(&self, probs: &[f32], wav_dur_seconds: f64) -> Vec<VadSegment> {
        if probs.is_empty() {
            return Vec::new();
        }
        let smoothed = self.smooth_prob(probs);
        let binary = smoothed
            .iter()
            .map(|prob| usize::from(*prob >= self.prob_threshold))
            .collect::<Vec<_>>();
        let decisions = self.smooth_preds_with_state_machine(&binary);
        let decisions = self.fix_smooth_window_start(&decisions);
        let decisions = self.merge_short_silence_segments(&decisions);
        let decisions = self.extend_speech_segments(&decisions);
        let decisions = self.split_long_speech_segments(&decisions, probs);
        self.decision_to_segments(&decisions, Some(wav_dur_seconds))
    }

    fn smooth_prob(&self, probs: &[f32]) -> Vec<f32> {
        if self.smooth_window_size <= 1 {
            return probs.to_vec();
        }
        let mut out = vec![0.0; probs.len()];
        let mut sum = 0.0;
        for i in 0..probs.len() {
            sum += probs[i];
            if i >= self.smooth_window_size {
                sum -= probs[i - self.smooth_window_size];
            }
            let denom = (i + 1).min(self.smooth_window_size) as f32;
            out[i] = sum / denom;
        }
        out
    }

    fn smooth_preds_with_state_machine(&self, binary: &[usize]) -> Vec<usize> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum State {
            Silence,
            PossibleSpeech,
            Speech,
            PossibleSilence,
        }
        let mut decisions = vec![0; binary.len()];
        let mut state = State::Silence;
        let mut speech_start = None;
        let mut silence_start = None;
        for (t, is_speech) in binary.iter().copied().enumerate() {
            match state {
                State::Silence if is_speech == 1 => {
                    state = State::PossibleSpeech;
                    speech_start = Some(t);
                }
                State::PossibleSpeech if is_speech == 1 => {
                    if t - speech_start.unwrap_or(t) >= self.min_speech_frame {
                        let start = speech_start.unwrap_or(t);
                        state = State::Speech;
                        decisions[start..t].fill(1);
                    }
                }
                State::PossibleSpeech => {
                    state = State::Silence;
                    speech_start = None;
                }
                State::Speech if is_speech == 0 => {
                    state = State::PossibleSilence;
                    silence_start = Some(t);
                }
                State::PossibleSilence if is_speech == 0 => {
                    if t - silence_start.unwrap_or(t) >= self.min_silence_frame {
                        state = State::Silence;
                        speech_start = None;
                    }
                }
                State::PossibleSilence => {
                    state = State::Speech;
                    silence_start = None;
                }
                _ => {}
            }
            decisions[t] = usize::from(matches!(state, State::Speech | State::PossibleSilence));
        }
        decisions
    }

    fn fix_smooth_window_start(&self, decisions: &[usize]) -> Vec<usize> {
        let mut out = decisions.to_vec();
        for t in 1..decisions.len() {
            if decisions[t - 1] == 0 && decisions[t] == 1 {
                let start = t.saturating_sub(self.smooth_window_size);
                out[start..t].fill(1);
            }
        }
        out
    }

    fn merge_short_silence_segments(&self, decisions: &[usize]) -> Vec<usize> {
        if self.merge_silence_frame == 0 {
            return decisions.to_vec();
        }
        let mut out = decisions.to_vec();
        let mut silence_start = None;
        for t in 1..decisions.len() {
            if decisions[t - 1] == 1 && decisions[t] == 0 && silence_start.is_none() {
                silence_start = Some(t);
            } else if decisions[t - 1] == 0
                && decisions[t] == 1
                && let Some(start) = silence_start.take()
                && t - start < self.merge_silence_frame
            {
                out[start..t].fill(1);
            }
        }
        out
    }

    fn extend_speech_segments(&self, decisions: &[usize]) -> Vec<usize> {
        if self.extend_speech_frame == 0 {
            return decisions.to_vec();
        }
        let mut out = decisions.to_vec();
        for (t, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 {
                let start = t.saturating_sub(self.extend_speech_frame);
                let end = (t + self.extend_speech_frame + 1).min(decisions.len());
                out[start..end].fill(1);
            }
        }
        out
    }

    fn split_long_speech_segments(&self, decisions: &[usize], probs: &[f32]) -> Vec<usize> {
        let mut out = decisions.to_vec();
        let mut speech_start = None;
        for (t, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 && speech_start.is_none() {
                speech_start = Some(t);
            } else if decision == 0
                && let Some(start) = speech_start.take()
            {
                self.split_long_speech_segment(&mut out, probs, start, t);
            }
        }
        if let Some(start) = speech_start {
            self.split_long_speech_segment(&mut out, probs, start, decisions.len());
        }
        out
    }

    fn split_long_speech_segment(
        &self,
        decisions: &mut [usize],
        probs: &[f32],
        start: usize,
        end: usize,
    ) {
        let end = end.min(probs.len()).min(decisions.len());
        if start < end && end.saturating_sub(start) > self.max_speech_frame {
            for split in self.find_split_points(&probs[start..end]) {
                decisions[start + split] = 0;
            }
        }
    }

    fn find_split_points(&self, probs: &[f32]) -> Vec<usize> {
        let mut splits = Vec::new();
        let mut start = 0usize;
        while probs.len().saturating_sub(start) > self.max_speech_frame {
            let window_start = start + self.max_speech_frame / 2;
            let window_end = start + self.max_speech_frame;
            let min_idx = probs[window_start..window_end]
                .iter()
                .enumerate()
                .min_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                .map(|(idx, _)| window_start + idx)
                .unwrap_or(window_start);
            splits.push(min_idx);
            start = min_idx + 1;
        }
        splits
    }

    fn decision_to_segments(&self, decisions: &[usize], wav_dur: Option<f64>) -> Vec<VadSegment> {
        self.decision_to_seconds(decisions, wav_dur)
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(
                    (start * 1000.0).round() as usize,
                    (end * 1000.0).round() as usize,
                ),
                probability: self.prob_threshold,
            })
            .filter(|segment| segment.range.duration() >= self.min_speech_ms as usize)
            .collect()
    }

    fn decision_to_seconds(&self, decisions: &[usize], wav_dur: Option<f64>) -> Vec<(f64, f64)> {
        let mut segments = Vec::new();
        let mut speech_start = None;
        for (t, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 && speech_start.is_none() {
                speech_start = Some(t);
            } else if decision == 0
                && let Some(start) = speech_start.take()
            {
                segments.push((start as f64 * 0.010, t as f64 * 0.010));
            }
        }
        if let Some(start) = speech_start {
            let mut end = decisions.len() as f64 * 0.010 + 0.025;
            if let Some(dur) = wav_dur {
                end = end.min(dur);
            }
            segments.push((start as f64 * 0.010, end));
        }
        segments
    }
}

fn ms_to_frame_count(ms: u64) -> usize {
    ms.div_ceil(FIRERED_FRAME_SHIFT_MS).max(1) as usize
}

fn ms_to_frame_count_allow_zero(ms: u64) -> usize {
    ms.div_ceil(FIRERED_FRAME_SHIFT_MS) as usize
}

fn frame_to_ms(frame: usize) -> usize {
    frame
        .saturating_sub(1)
        .saturating_mul(FIRERED_FRAME_SHIFT_MS as usize)
}

fn samples_to_ms(samples: usize) -> usize {
    samples.saturating_mul(1000) / FIRERED_SAMPLE_RATE as usize
}

fn read_kaldi_binary_cmvn(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read FireRed CMVN {}", path.display()))?;
    if bytes.len() < 16 || bytes[0] != 0 || &bytes[1..4] != b"BDM" {
        bail!("unsupported FireRed CMVN format: {}", path.display());
    }
    let mut offset = 4usize;
    if bytes.get(offset) == Some(&b' ') {
        offset += 1;
    }
    let rows = read_kaldi_i32(&bytes, &mut offset)? as usize;
    let cols = read_kaldi_i32(&bytes, &mut offset)? as usize;
    if rows != 2 || cols != FIRERED_INPUT_DIM + 1 {
        bail!("unexpected FireRed CMVN shape {rows}x{cols}");
    }
    let need = rows * cols * 8;
    if bytes.len() < offset + need {
        bail!("truncated FireRed CMVN matrix: {}", path.display());
    }
    let (chunks, _remainder) = bytes[offset..offset + need].as_chunks::<8>();
    let mut values = Vec::with_capacity(rows * cols);
    for chunk in chunks {
        values.push(f64::from_le_bytes(*chunk));
    }
    let count = values[FIRERED_INPUT_DIM].max(1.0);
    let mut means = Vec::with_capacity(FIRERED_INPUT_DIM);
    let mut inverse_std = Vec::with_capacity(FIRERED_INPUT_DIM);
    for d in 0..FIRERED_INPUT_DIM {
        let mean = values[d] / count;
        let variance = (values[cols + d] / count - mean * mean).max(1e-20);
        means.push(mean as f32);
        inverse_std.push((1.0 / variance.sqrt()) as f32);
    }
    Ok((means, inverse_std))
}

fn read_kaldi_i32(bytes: &[u8], offset: &mut usize) -> Result<i32> {
    if bytes.get(*offset) != Some(&4) || *offset + 5 > bytes.len() {
        bail!("invalid Kaldi binary int marker");
    }
    *offset += 1;
    let value = i32::from_le_bytes(
        bytes[*offset..*offset + 4]
            .try_into()
            .map_err(|_| anyhow::anyhow!("truncated Kaldi binary int value"))?,
    );
    *offset += 4;
    Ok(value)
}

fn validate_firered_vad_model_dir(model_dir: &Path) -> Result<()> {
    if !model_dir.is_dir() {
        bail!(
            "FireRedVAD model path is not a directory: {}",
            model_dir.display()
        );
    }
    if !nonempty_file(&model_dir.join("cmvn.ark"))? {
        bail!("FireRedVAD cmvn.ark not found in {}", model_dir.display());
    }
    if nonempty_file(&model_dir.join("model.pth.tar"))?
        || nonempty_file(&model_dir.join("model.safetensors"))?
    {
        return Ok(());
    }
    bail!(
        "FireRedVAD model directory is incomplete: {} (expected cmvn.ark and model.pth.tar or model.safetensors)",
        model_dir.display()
    )
}

fn resolve_firered_vad_model_dir(path: &Path, preferred_subdir: &str) -> Result<PathBuf> {
    let nested = path.join(preferred_subdir);
    if validate_firered_vad_model_dir(&nested).is_ok() {
        return Ok(nested);
    }
    validate_firered_vad_model_dir(path)?;
    Ok(path.to_path_buf())
}

fn default_firered_vad_model_dir(preferred_subdir: &str) -> Result<PathBuf> {
    let legacy = modelhub::modelscope_cache_dir()
        .join(FIRERED_VAD_REPO)
        .join(preferred_subdir);
    if validate_firered_vad_model_dir(&legacy).is_ok() {
        return Ok(legacy);
    }

    let target = modelscope_model_snapshot_dir(&modelhub::modelscope_cache_dir(), FIRERED_VAD_REPO)
        .join(preferred_subdir);
    if validate_firered_vad_model_dir(&target).is_ok() {
        return Ok(target);
    }

    let cache = modelhub::modelscope_cache_dir();
    tracing::warn!(
        target: "vasr_vad::firered",
        "FireRedVAD model not found locally, downloading {} from ModelScope",
        FIRERED_VAD_REPO
    );
    let downloaded = block_on_async(modelhub::modelscope::download_model(
        FIRERED_VAD_REPO,
        &cache,
    ))?;
    downloaded
        .with_context(|| format!("failed to download FireRedVAD model {FIRERED_VAD_REPO:?}"))?;
    let target = modelscope_model_snapshot_dir(&cache, FIRERED_VAD_REPO).join(preferred_subdir);
    validate_firered_vad_model_dir(&target)?;
    Ok(target)
}

fn modelscope_model_snapshot_dir(cache: &Path, repo_id: &str) -> PathBuf {
    cache
        .join("models")
        .join(repo_id.replace('/', "--"))
        .join("snapshots")
        .join("master")
}

fn nonempty_file(path: &Path) -> Result<bool> {
    let meta = match std::fs::metadata(path) {
        Ok(meta) => meta,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err).with_context(|| format!("failed to stat {}", path.display())),
    };
    Ok(meta.is_file() && meta.len() > 0)
}

fn firered_vad_device() -> Result<Device> {
    match std::env::var("VASR_VAD_DEVICE").as_deref() {
        Ok("cuda") => Ok(Device::new_cuda(0)?),
        Ok("cpu") | Err(_) => Ok(Device::Cpu),
        Ok(other) => bail!("unsupported VASR_VAD_DEVICE={other:?}; expected cpu or cuda"),
    }
}

/// 在独立线程上用一次性 tokio runtime 阻塞执行 `future`。
///
/// 返回值包一层 [`Result`]，让运行时创建失败与线程 panic 都能向上传播，
/// 避免在库代码里使用 `unwrap` / `expect`。
fn block_on_async<F: std::future::Future + Send>(future: F) -> Result<F::Output>
where
    F::Output: Send,
{
    std::thread::scope(|scope| {
        let handle = scope.spawn(|| -> Result<F::Output> {
            let runtime = tokio::runtime::Runtime::new()
                .map_err(|error| anyhow::anyhow!("failed to create tokio runtime: {error}"))?;
            Ok(runtime.block_on(future))
        });
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("tokio download thread panicked"))?
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use asr_data::AudioChunk;

    fn realtime_options() -> VadOptions {
        VadOptions {
            threshold: 0.5,
            min_speech_ms: 80,
            min_silence_ms: 500,
            max_segment_ms: 20_000,
            pad_ms: 50,
            num_threads: 0,
        }
    }

    #[test]
    fn stream_postprocessor_emits_start_and_end_boundaries() {
        let mut postprocessor = FireRedStreamPostprocessor::from_options(&realtime_options());
        let starts = (0..8)
            .flat_map(|_| postprocessor.process(1.0))
            .collect::<Vec<_>>();
        assert_eq!(
            starts,
            vec![FireRedStreamBoundary::Start {
                start_frame: 1,
                frame: 8,
            }]
        );

        let ends = (0..52)
            .flat_map(|_| postprocessor.process(0.0))
            .collect::<Vec<_>>();
        assert_eq!(
            ends,
            vec![FireRedStreamBoundary::End {
                start_frame: 1,
                end_frame: 60,
            }]
        );
    }

    #[test]
    fn stream_postprocessor_finishes_an_open_segment() {
        let mut postprocessor = FireRedStreamPostprocessor::from_options(&realtime_options());
        for _ in 0..8 {
            postprocessor.process(1.0);
        }
        assert_eq!(postprocessor.finish(), Some(1));
        assert_eq!(postprocessor.finish(), None);
    }

    #[test]
    #[ignore = "requires a local FireRedVAD checkpoint; set VASR_FIRERED_VAD_ROOT"]
    fn stream_checkpoint_loads_and_accepts_audio() -> Result<()> {
        let root = std::env::var("VASR_FIRERED_VAD_ROOT")?;
        let model = FireRedVadModel::from_pretrained_streaming(Some(root))?;
        assert!(model.model_dir().ends_with(FIRERED_STREAM_MODEL_DIR));

        let samples = (0..4000)
            .map(|index| ((index as f32 * 0.017).sin() * 0.1).clamp(-1.0, 1.0))
            .collect::<Vec<_>>();
        let feats = model.weights.frontend.extract(&samples)?;
        let expected = model.weights.forward_probs(&feats, 0)?;
        let frame_count = feats.dim(0)?;
        let split = 7.min(frame_count);
        let mut state = FireRedStreamForwardState::default();
        let mut actual = model.weights.forward_probs_streaming(
            &feats.narrow(0, 0, split)?.contiguous()?,
            0,
            &mut state,
        )?;
        actual.extend(model.weights.forward_probs_streaming(
            &feats.narrow(0, split, frame_count - split)?.contiguous()?,
            0,
            &mut state,
        )?);
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 1e-5, "streaming max error: {max_error}");

        let mut stream = VadModel::new_stream(&model, &realtime_options())?;
        let chunk = AudioChunk {
            samples: vec![0.0; 1600],
            sample_rate: FIRERED_SAMPLE_RATE,
            channels: 1,
            source_format: None,
            index: 0,
            offset_ms: 0,
            is_final: true,
        };
        stream.push_chunk(&chunk)?;
        stream.finish()?;
        Ok(())
    }
}
