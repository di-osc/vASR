use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias, ops::sigmoid};
use gemm::Parallelism;
use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};
use ndarray::Array2;
use rayon::prelude::*;
use vasr_data::{DurationMs, TimeRange, Waveform};

use crate::traits::run_with_vad_threads;
use crate::{StreamingVadSession, VadModel, VadOptions, VadSegment};

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
}

impl FireRedVadModel {
    pub fn from_pretrained(path: Option<impl AsRef<Path>>) -> Result<Self> {
        Self::from_pretrained_on_device(path, &firered_vad_device()?)
    }

    pub fn from_pretrained_on_device(
        path: Option<impl AsRef<Path>>,
        device: &Device,
    ) -> Result<Self> {
        let model_dir = match path {
            Some(path) => {
                let model_dir = path.as_ref().to_path_buf();
                validate_firered_vad_model_dir(&model_dir)?;
                model_dir
            }
            None => default_firered_vad_model_dir()?,
        };
        let weights = Arc::new(FireRedVadWeights::load_on_device(&model_dir, device)?);
        Ok(Self { weights, model_dir })
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
        timing.frames = feats.shape()[0];
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

    fn new_stream(&self, _options: &VadOptions) -> Result<Box<dyn StreamingVadSession>> {
        bail!("FireRedVAD native implementation currently supports offline VAD only")
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
    lookahead: Vec<f32>,
}

#[derive(Debug)]
struct CpuLinear {
    weight_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

impl FireRedVadWeights {
    fn load_on_device(model_dir: &Path, device: &Device) -> Result<Self> {
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
        let fsmn1 = FireRedFsmn::new(dfsmn.pp("fsmn1"))?;
        let mut blocks = Vec::with_capacity(FIRERED_BLOCKS - 1);
        for i in 0..FIRERED_BLOCKS - 1 {
            blocks.push(FireRedDfsmnBlock::new(dfsmn.pp("fsmns").pp(i))?);
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

    fn forward_probs(&self, feats: &Array2<f32>, num_threads: usize) -> Result<Vec<f32>> {
        let frames = feats.shape()[0];
        let dim = feats.shape()[1];
        if dim != FIRERED_INPUT_DIM {
            bail!("FireRedVAD expects feature dim {FIRERED_INPUT_DIM}, got {dim}");
        }
        if let Some(cpu) = &self.cpu {
            return run_with_vad_threads(num_threads, || cpu.forward_probs(feats))?;
        }
        let x = Tensor::from_vec(
            feats.iter().copied().collect::<Vec<_>>(),
            (1usize, frames, FIRERED_INPUT_DIM),
            &self.device,
        )?;
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
}

#[derive(Debug)]
struct FireRedDfsmnBlock {
    fc1: Linear,
    fc2: Linear,
    fsmn: FireRedFsmn,
}

impl FireRedDfsmnBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(FIRERED_PROJ_DIM, FIRERED_HIDDEN_DIM, vb.pp("fc1").pp("0"))?,
            fc2: linear_no_bias(FIRERED_HIDDEN_DIM, FIRERED_PROJ_DIM, vb.pp("fc2"))?,
            fsmn: FireRedFsmn::new(vb.pp("fsmn"))?,
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
    fn forward_probs(&self, feats: &Array2<f32>) -> Result<Vec<f32>> {
        let frames = feats.shape()[0];
        let mut x = feats.iter().copied().collect::<Vec<_>>();
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
            lookahead: fsmn
                .lookahead_cpu
                .clone()
                .ok_or_else(|| anyhow::anyhow!("FireRedVAD CPU FSMN missing lookahead weights"))?,
        })
    }

    fn apply(&self, input: &mut Vec<f32>, frames: usize) {
        let src = input.clone();
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
                        if frames > 1 && ahead_t < frames {
                            sum += src[ahead_t * FIRERED_PROJ_DIM + c] * self.lookahead[kernel + k];
                        }
                    }
                    *value = sum;
                }
            });
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
    lookahead_weight: Tensor,
    lookback_cpu: Option<Vec<f32>>,
    lookahead_cpu: Option<Vec<f32>>,
}

impl FireRedFsmn {
    fn new(vb: VarBuilder) -> Result<Self> {
        let lookback_weight = vb
            .pp("lookback_filter")
            .get((FIRERED_PROJ_DIM, 1, FIRERED_FSMN_ORDER), "weight")?;
        let lookahead_weight = vb
            .pp("lookahead_filter")
            .get((FIRERED_PROJ_DIM, 1, FIRERED_FSMN_ORDER), "weight")?;
        let lookback_cpu = tensor_cpu_vec(&lookback_weight)?;
        let lookahead_cpu = tensor_cpu_vec(&lookahead_weight)?;
        Ok(Self {
            lookback_weight,
            lookahead_weight,
            lookback_cpu,
            lookahead_cpu,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.device().is_cpu()
            && let (Some(lookback), Some(lookahead)) = (&self.lookback_cpu, &self.lookahead_cpu)
        {
            return firered_fsmn_cpu(input, lookback, lookahead);
        }

        let x = input.permute((0, 2, 1))?;
        let lookback = depthwise_conv1d_left(&x, &self.lookback_weight)?;
        let mut memory = x.broadcast_add(&lookback)?;
        if x.dim(2)? > 1 {
            let lookahead = depthwise_conv1d_ahead(&x, &self.lookahead_weight)?;
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

fn firered_fsmn_cpu(input: &Tensor, lookback: &[f32], lookahead: &[f32]) -> Result<Tensor> {
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
                if frames > 1 && ahead_t < frames {
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

    fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
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
        Ok(Array2::from_shape_vec((frames, FIRERED_INPUT_DIM), out)?)
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
            } else if decisions[t - 1] == 0 && decisions[t] == 1 {
                if let Some(start) = silence_start.take() {
                    let silence_frames = t - start;
                    if silence_frames < self.merge_silence_frame {
                        out[start..t].fill(1);
                    }
                }
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
        if start < end {
            if end.saturating_sub(start) > self.max_speech_frame {
                for split in self.find_split_points(&probs[start..end]) {
                    decisions[start + split] = 0;
                }
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
                    DurationMs((start * 1000.0).round() as u64),
                    DurationMs((end * 1000.0).round() as u64),
                ),
                probability: self.prob_threshold,
            })
            .filter(|segment| segment.range.duration().0 >= self.min_speech_ms)
            .collect()
    }

    fn decision_to_seconds(&self, decisions: &[usize], wav_dur: Option<f64>) -> Vec<(f64, f64)> {
        let mut segments = Vec::new();
        let mut speech_start = None;
        for (t, decision) in decisions.iter().copied().enumerate() {
            if decision == 1 && speech_start.is_none() {
                speech_start = Some(t);
            } else if decision == 0 && speech_start.is_some() {
                let start = speech_start.take().unwrap();
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
    let mut values = Vec::with_capacity(rows * cols);
    for chunk in bytes[offset..offset + need].chunks_exact(8) {
        values.push(f64::from_le_bytes(chunk.try_into().expect("8 bytes")));
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
    let value = i32::from_le_bytes(bytes[*offset..*offset + 4].try_into().expect("4 bytes"));
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

fn default_firered_vad_model_dir() -> Result<PathBuf> {
    let target = vasr_hub::cache_dir().join(FIRERED_VAD_REPO).join("VAD");
    if validate_firered_vad_model_dir(&target).is_ok() {
        return Ok(target);
    }

    let cache = vasr_hub::cache_dir();
    tracing::warn!(
        target: "vasr_vad::firered",
        "FireRedVAD model not found locally, downloading {} from ModelScope",
        FIRERED_VAD_REPO
    );
    block_on_async(vasr_hub::modelscope::download::download_model(
        FIRERED_VAD_REPO,
        &cache,
    ))
    .with_context(|| format!("failed to download FireRedVAD model {FIRERED_VAD_REPO:?}"))?;
    let target = cache.join(FIRERED_VAD_REPO).join("VAD");
    validate_firered_vad_model_dir(&target)?;
    Ok(target)
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
