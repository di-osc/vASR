use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias, ops::softmax_last_dim};
use gemm::Parallelism;
use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};
use rayon::prelude::*;
use vasr_data::{AudioActivity, AudioChunk, TimeRange, Timeline, Waveform};

use crate::traits::run_with_vad_threads;
use crate::{StreamingVadSession, VadEvent, VadEventStatus, VadModel, VadOptions, VadSegment};

mod e2e;
use e2e::{E2EVadConfig, E2EVadModel};

pub const FSMN_VAD_CHUNK_SIZE: usize = 512;
const FSMN_VAD_STREAM_SCORE_SAMPLES: usize = 10 * FSMN_VAD_SAMPLE_RATE as usize;
const FSMN_VAD_FRAME_SHIFT_SAMPLES: usize = 160;
const FSMN_VAD_FRAME_LENGTH_SAMPLES: usize = 400;
const FSMN_VAD_FEAT_CHUNK_SIZE: usize = 6000;
const FSMN_VAD_REPO: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";
const FSMN_VAD_LAYERS: usize = 4;
const FSMN_VAD_PROJ_DIM: usize = 128;
const FSMN_VAD_CACHE_FRAMES: usize = 19;
const FSMN_VAD_SAMPLE_RATE: u32 = 16_000;
const FSMN_VAD_SOURCE: &str = "fsmn_vad";
const VAD_REFINE_SILENCE_MS: [u64; 2] = [500, 300];

#[derive(Debug, Clone, Copy, Default)]
pub struct FsmnVadTiming {
    pub pcm_seconds: f64,
    pub frontend_seconds: f64,
    pub forward_seconds: f64,
    pub segmenter_seconds: f64,
    pub chunks: usize,
    pub forward_ops: FsmnVadForwardTiming,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FsmnVadForwardTiming {
    pub in_linear1_seconds: f64,
    pub in_linear2_seconds: f64,
    pub block_linear_seconds: [f64; FSMN_VAD_LAYERS],
    pub block_memory_seconds: [f64; FSMN_VAD_LAYERS],
    pub block_affine_seconds: [f64; FSMN_VAD_LAYERS],
    pub out_linear1_seconds: f64,
    pub out_linear2_seconds: f64,
    pub softmax_seconds: f64,
}

impl FsmnVadTiming {
    fn add_pcm(&mut self, duration: Duration) {
        self.pcm_seconds += duration.as_secs_f64();
    }

    fn add_segmenter(&mut self, duration: Duration) {
        self.segmenter_seconds += duration.as_secs_f64();
    }
}

#[derive(Debug, Clone)]
pub struct FsmnVadDetection {
    pub segments: Vec<VadSegment>,
    pub timing: FsmnVadTiming,
}

#[derive(Debug, Clone)]
pub struct FsmnVadFrontend {
    inner: WavFrontend,
}

impl FsmnVadFrontend {
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        validate_fsmn_vad_model_dir(model_dir)?;
        Ok(Self {
            inner: WavFrontend::new(WavFrontendConfig {
                sample_rate: FSMN_VAD_SAMPLE_RATE as i32,
                lfr_m: 5,
                lfr_n: 1,
                cmvn_file: Some(model_dir.join("am.mvn")),
                ..Default::default()
            })?,
        })
    }

    /// 从归一化到 `[-1, 1]` 的 16 kHz 单声道采样提取 LFR + CMVN 特征。
    ///
    /// 返回 CPU 上的 `(frames, feature_dim)` F32 张量；是否搬运到计算设备
    /// 由 `forward_frame_scores*` 根据模型所在设备决定。
    pub fn extract_features_from_normalized_f32(&self, samples: &[f32]) -> Result<Tensor> {
        self.inner.extract_features_from_normalized_f32(samples)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FsmnVadPostProcessor;

impl FsmnVadPostProcessor {
    pub fn segments_from_frame_scores<F>(
        &self,
        waveform: &Waveform,
        frame_scores: &[Vec<f32>],
        options: &VadOptions,
        mut refine_long_segment: F,
    ) -> Result<Vec<VadSegment>>
    where
        F: FnMut(&Waveform, &VadSegment, &VadOptions, u64) -> Result<Vec<VadSegment>>,
    {
        let mut e2e = build_e2e_model(options);
        let max_end_sil = options.min_silence_ms as i32;
        let mut ms_segments = Vec::new();
        let total_frames = frame_scores.len();
        let mut frame_offset = 0usize;

        while frame_offset < total_frames {
            let step = FSMN_VAD_FEAT_CHUNK_SIZE.min(total_frames - frame_offset);
            let is_final = frame_offset + step >= total_frames;
            let wave_start = frame_offset * FSMN_VAD_FRAME_SHIFT_SAMPLES;
            let wave_end = if is_final {
                waveform.samples.len()
            } else {
                ((frame_offset + step - 1) * FSMN_VAD_FRAME_SHIFT_SAMPLES
                    + FSMN_VAD_FRAME_LENGTH_SAMPLES)
                    .min(waveform.samples.len())
            };
            ms_segments.extend(e2e.detect_chunk(
                &frame_scores[frame_offset..frame_offset + step],
                &waveform.samples[wave_start..wave_end],
                is_final,
                max_end_sil,
            ));
            frame_offset += step;
        }

        let raw_segments = ms_segments
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(start as usize, end as usize),
                probability: options.threshold,
            })
            .filter(|segment| segment_duration_ms(segment) >= options.min_speech_ms)
            .collect::<Vec<_>>();
        self.split_segments_for_asr(waveform, &raw_segments, options, &mut refine_long_segment)
    }

    fn split_segments_for_asr<F>(
        &self,
        waveform: &Waveform,
        segments: &[VadSegment],
        options: &VadOptions,
        refine_long_segment: &mut F,
    ) -> Result<Vec<VadSegment>>
    where
        F: FnMut(&Waveform, &VadSegment, &VadOptions, u64) -> Result<Vec<VadSegment>>,
    {
        let max_segment_ms = options.max_segment_ms;
        if segments.is_empty() || max_segment_ms == 0 {
            return Ok(segments.to_vec());
        }

        let mut split = Vec::with_capacity(segments.len());
        for segment in segments {
            if segment_duration_ms(segment) <= max_segment_ms {
                split.push(segment.clone());
                continue;
            }

            let refined = self.refine_long_segment(
                waveform,
                segment,
                options,
                max_segment_ms,
                refine_long_segment,
            )?;
            split.extend(refined);
        }

        Ok(split)
    }

    fn refine_long_segment<F>(
        &self,
        waveform: &Waveform,
        segment: &VadSegment,
        options: &VadOptions,
        max_segment_ms: u64,
        refine_long_segment: &mut F,
    ) -> Result<Vec<VadSegment>>
    where
        F: FnMut(&Waveform, &VadSegment, &VadOptions, u64) -> Result<Vec<VadSegment>>,
    {
        let mut current = vec![segment.clone()];
        for silence_ms in VAD_REFINE_SILENCE_MS {
            if options.min_silence_ms <= silence_ms {
                continue;
            }

            let mut refined = Vec::new();
            let mut changed = false;
            for candidate in current {
                if segment_duration_ms(&candidate) <= max_segment_ms {
                    refined.push(candidate);
                    continue;
                }

                let local = refine_long_segment(waveform, &candidate, options, silence_ms)?;
                if local.is_empty() {
                    refined.push(candidate);
                    continue;
                }
                changed = true;
                refined.extend(local);
            }

            current = refined;
            if current
                .iter()
                .all(|candidate| segment_duration_ms(candidate) <= max_segment_ms)
            {
                return Ok(current);
            }
            if !changed {
                break;
            }
        }

        Ok(current
            .into_iter()
            .flat_map(|candidate| hard_split_segment(&candidate, max_segment_ms))
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct FsmnVadModel {
    weights: Arc<FsmnVadWeights>,
    model_dir: PathBuf,
}

impl FsmnVadModel {
    pub fn from_pretrained(path: Option<impl AsRef<Path>>) -> Result<Self> {
        Self::from_pretrained_on_device(path, &fsmn_vad_device()?)
    }

    pub fn from_pretrained_on_device(
        path: Option<impl AsRef<Path>>,
        device: &Device,
    ) -> Result<Self> {
        let model_dir = match path {
            Some(path) => {
                let model_dir = path.as_ref().to_path_buf();
                validate_fsmn_vad_model_dir(&model_dir)?;
                model_dir
            }
            None => download_fsmn_vad()?,
        };
        let weights = Arc::new(FsmnVadWeights::load_on_device(&model_dir, device)?);
        Ok(Self { weights, model_dir })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn annotate(&self, waveform: &Waveform, options: &VadOptions) -> Timeline {
        let duration = waveform.duration_ms().round() as usize;
        let mut timeline = Timeline::new("vad_audio", duration);
        if let Ok(segments) = self.detect(waveform, options) {
            for segment in segments {
                // 写入预测轨道：参考标注不允许携带 source。
                let _ = timeline.annotate_span_with(
                    segment.range.start_ms,
                    segment.range.end_ms,
                    AudioActivity::new().with_event("speech"),
                    false,
                    Some(FSMN_VAD_SOURCE),
                );
            }
        }
        timeline
    }

    fn new_session(&self, options: &VadOptions) -> Result<FsmnVadSession> {
        FsmnVadSession::new(Arc::clone(&self.weights), options.clone())
    }

    pub fn detect_with_timing(
        &self,
        waveform: &Waveform,
        options: &VadOptions,
    ) -> Result<FsmnVadDetection> {
        if waveform.sample_rate != FSMN_VAD_SAMPLE_RATE {
            bail!(
                "FSMN VAD expects 16kHz mono audio, got sample_rate={}",
                waveform.sample_rate
            );
        }
        let detection = self.detect_offline(waveform, options)?;
        let segments = detection.segments;
        let timing = detection.timing;
        tracing::trace!(
            target: "vasr_vad::fsmn",
            "fsmn_vad | audio={:.2}s | chunks={} | segments={} | pcm={:.3}s | frontend={:.3}s | forward={:.3}s | segmenter={:.3}s",
            waveform.duration_seconds(),
            timing.chunks,
            segments.len(),
            timing.pcm_seconds,
            timing.frontend_seconds,
            timing.forward_seconds,
            timing.segmenter_seconds,
        );
        Ok(FsmnVadDetection { segments, timing })
    }

    /// 直接在已提取的 `(frames, feature_dim)` 特征上跑前向并返回逐帧分数。
    pub fn forward_frame_scores_from_features(
        &self,
        feats: &Tensor,
        num_threads: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let mut caches = self.weights.zero_caches()?;
        self.weights
            .forward_frame_scores(feats, &mut caches, num_threads)
    }

    fn detect_offline(
        &self,
        waveform: &Waveform,
        options: &VadOptions,
    ) -> Result<FsmnVadDetection> {
        let mut timing = FsmnVadTiming::default();

        let frontend_start = Instant::now();
        let feats = self
            .weights
            .frontend
            .extract_features_from_normalized_f32(&waveform.samples)?;
        timing.frontend_seconds += frontend_start.elapsed().as_secs_f64();

        let total_frames = feats.dim(0)?;
        if total_frames == 0 {
            return Ok(FsmnVadDetection {
                segments: Vec::new(),
                timing,
            });
        }

        let mut caches = self.weights.zero_caches()?;
        let mut e2e = build_e2e_model(options);
        let max_end_sil = options.min_silence_ms as i32;
        let mut ms_segments = Vec::new();
        let forward_start = Instant::now();
        let mut frame_offset = 0usize;

        while frame_offset < total_frames {
            let step = FSMN_VAD_FEAT_CHUNK_SIZE.min(total_frames - frame_offset);
            let is_final = frame_offset + step >= total_frames;
            // 前向只读取 `feats`，这里切出连续视图即可，无需拷贝整块特征。
            let feat_chunk = feats.narrow(0, frame_offset, step)?.contiguous()?;
            let frame_scores = self.weights.forward_frame_scores_with_timing(
                &feat_chunk,
                &mut caches,
                options.num_threads,
                &mut timing.forward_ops,
            )?;
            let wave_start = frame_offset * FSMN_VAD_FRAME_SHIFT_SAMPLES;
            let wave_end = if is_final {
                waveform.samples.len()
            } else {
                ((frame_offset + step - 1) * FSMN_VAD_FRAME_SHIFT_SAMPLES
                    + FSMN_VAD_FRAME_LENGTH_SAMPLES)
                    .min(waveform.samples.len())
            };
            let segmenter_start = Instant::now();
            ms_segments.extend(e2e.detect_chunk(
                &frame_scores,
                &waveform.samples[wave_start..wave_end],
                is_final,
                max_end_sil,
            ));
            timing.add_segmenter(segmenter_start.elapsed());
            frame_offset += step;
        }

        timing.forward_seconds += forward_start.elapsed().as_secs_f64();
        timing.chunks = 1;

        let raw_segments = ms_segments
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(start as usize, end as usize),
                probability: options.threshold,
            })
            .filter(|segment| segment_duration_ms(segment) >= options.min_speech_ms)
            .collect::<Vec<_>>();
        let segments = self.split_segments_for_asr(waveform, &raw_segments, options)?;

        Ok(FsmnVadDetection { segments, timing })
    }

    fn detect_raw_segments(
        &self,
        waveform: &Waveform,
        options: &VadOptions,
    ) -> Result<Vec<VadSegment>> {
        let mut local_options = options.clone();
        local_options.max_segment_ms = 0;
        Ok(self.detect_offline(waveform, &local_options)?.segments)
    }

    pub fn segments_from_frame_scores(
        &self,
        waveform: &Waveform,
        frame_scores: &[Vec<f32>],
        options: &VadOptions,
    ) -> Result<Vec<VadSegment>> {
        let mut e2e = build_e2e_model(options);
        let max_end_sil = options.min_silence_ms as i32;
        let mut ms_segments = Vec::new();
        let total_frames = frame_scores.len();
        let mut frame_offset = 0usize;

        while frame_offset < total_frames {
            let step = FSMN_VAD_FEAT_CHUNK_SIZE.min(total_frames - frame_offset);
            let is_final = frame_offset + step >= total_frames;
            let wave_start = frame_offset * FSMN_VAD_FRAME_SHIFT_SAMPLES;
            let wave_end = if is_final {
                waveform.samples.len()
            } else {
                ((frame_offset + step - 1) * FSMN_VAD_FRAME_SHIFT_SAMPLES
                    + FSMN_VAD_FRAME_LENGTH_SAMPLES)
                    .min(waveform.samples.len())
            };
            ms_segments.extend(e2e.detect_chunk(
                &frame_scores[frame_offset..frame_offset + step],
                &waveform.samples[wave_start..wave_end],
                is_final,
                max_end_sil,
            ));
            frame_offset += step;
        }

        let raw_segments = ms_segments
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(start as usize, end as usize),
                probability: options.threshold,
            })
            .filter(|segment| segment_duration_ms(segment) >= options.min_speech_ms)
            .collect::<Vec<_>>();
        self.split_segments_for_asr(waveform, &raw_segments, options)
    }

    fn split_segments_for_asr(
        &self,
        waveform: &Waveform,
        segments: &[VadSegment],
        options: &VadOptions,
    ) -> Result<Vec<VadSegment>> {
        let max_segment_ms = options.max_segment_ms;
        if segments.is_empty() || max_segment_ms == 0 {
            return Ok(segments.to_vec());
        }

        let mut split = Vec::with_capacity(segments.len());
        for segment in segments {
            if segment_duration_ms(segment) <= max_segment_ms {
                split.push(segment.clone());
                continue;
            }

            let refined = self.refine_long_segment(waveform, segment, options, max_segment_ms)?;
            split.extend(refined);
        }

        Ok(split)
    }

    fn refine_long_segment(
        &self,
        waveform: &Waveform,
        segment: &VadSegment,
        options: &VadOptions,
        max_segment_ms: u64,
    ) -> Result<Vec<VadSegment>> {
        let mut current = vec![segment.clone()];
        for silence_ms in VAD_REFINE_SILENCE_MS {
            if options.min_silence_ms <= silence_ms {
                continue;
            }

            let mut refined = Vec::new();
            let mut changed = false;
            for candidate in current {
                if segment_duration_ms(&candidate) <= max_segment_ms {
                    refined.push(candidate);
                    continue;
                }

                let local = self
                    .detect_segment_with_min_silence(waveform, &candidate, options, silence_ms)?;
                if local.is_empty() {
                    refined.push(candidate);
                    continue;
                }
                changed = true;
                refined.extend(local);
            }

            current = refined;
            if current
                .iter()
                .all(|candidate| segment_duration_ms(candidate) <= max_segment_ms)
            {
                return Ok(current);
            }
            if !changed {
                break;
            }
        }

        Ok(current
            .into_iter()
            .flat_map(|candidate| hard_split_segment(&candidate, max_segment_ms))
            .collect())
    }

    fn detect_segment_with_min_silence(
        &self,
        waveform: &Waveform,
        segment: &VadSegment,
        options: &VadOptions,
        min_silence_ms: u64,
    ) -> Result<Vec<VadSegment>> {
        let local_waveform =
            waveform.slice_ms(segment.range.start_ms as u64, segment.range.end_ms as u64);
        let mut refined_options = options.clone();
        refined_options.min_silence_ms = min_silence_ms;
        refined_options.max_segment_ms = 0;
        let local_segments = self.detect_raw_segments(&local_waveform, &refined_options)?;
        Ok(local_segments
            .into_iter()
            .map(|mut local| {
                // 局部检测结果平移回原音频时间轴。
                local.range.start_ms = local.range.start_ms.saturating_add(segment.range.start_ms);
                local.range.end_ms = local
                    .range
                    .end_ms
                    .saturating_add(segment.range.start_ms)
                    .min(segment.range.end_ms);
                local
            })
            .filter(|local| local.range.end_ms > local.range.start_ms)
            .collect())
    }
}

fn hard_split_segment(segment: &VadSegment, max_segment_ms: u64) -> Vec<VadSegment> {
    if max_segment_ms == 0 || segment_duration_ms(segment) <= max_segment_ms {
        return vec![segment.clone()];
    }

    // 时间轴坐标是 `usize` 毫秒，这里把选项里的 `u64` 阈值转换一次复用。
    let max_segment_ms = max_segment_ms as usize;
    let mut split = Vec::new();
    let mut start = segment.range.start_ms;
    let end = segment.range.end_ms;
    while start < end {
        let next = start.saturating_add(max_segment_ms).min(end);
        split.push(VadSegment {
            range: TimeRange::new(start, next),
            probability: segment.probability,
        });
        start = next;
    }
    split
}

fn segment_duration_ms(segment: &VadSegment) -> u64 {
    segment.range.end_ms.saturating_sub(segment.range.start_ms) as u64
}

fn fsmn_vad_device() -> Result<Device> {
    let requested = std::env::var("VASR_VAD_DEVICE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase());
    match requested.as_deref() {
        Some("cuda") | Some("gpu") => {
            #[cfg(any(feature = "cuda", feature = "cuda-vad"))]
            {
                return fsmn_vad_cuda_device();
            }
            #[cfg(not(any(feature = "cuda", feature = "cuda-vad")))]
            {
                bail!(
                    "VASR_VAD_DEVICE=cuda requires building with `--features cuda-vad` or `cuda`"
                );
            }
        }
        Some("metal") => {
            #[cfg(feature = "metal")]
            {
                return fsmn_vad_metal_device();
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("VASR_VAD_DEVICE=metal requires building with `--features metal`");
            }
        }
        Some("cpu") => {}
        None => {
            #[cfg(any(feature = "cuda", feature = "cuda-vad"))]
            {
                return fsmn_vad_cuda_device();
            }
        }
        Some(other) => {
            bail!("unknown VASR_VAD_DEVICE {other:?}; expected cpu, metal, or cuda")
        }
    }
    Ok(Device::Cpu)
}

#[cfg(any(feature = "cuda", feature = "cuda-vad"))]
fn fsmn_vad_cuda_device() -> Result<Device> {
    let device = Device::new_cuda(0)
        .map_err(|err| anyhow::anyhow!("failed to create CUDA device 0: {err}"))?;
    tracing::info!(
        target: "vasr_vad::fsmn",
        "FSMN VAD using CUDA device 0"
    );
    Ok(device)
}

#[cfg(feature = "metal")]
fn fsmn_vad_metal_device() -> Result<Device> {
    let device = Device::new_metal(0)
        .map_err(|err| anyhow::anyhow!("failed to create Metal device 0: {err}"))?;
    tracing::info!(
        target: "vasr_vad::fsmn",
        "FSMN VAD using Metal device 0"
    );
    Ok(device)
}

fn build_e2e_model(options: &VadOptions) -> E2EVadModel {
    // 用结构体更新语法一次性构造，避免 Default 之后再逐字段赋值。
    let config = E2EVadConfig {
        speech_noise_thres: options.threshold,
        ..E2EVadConfig::default()
    };
    E2EVadModel::new(config)
}

impl FsmnVadModel {
    pub fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        Ok(self.detect_with_timing(waveform, options)?.segments)
    }
}

impl VadModel for FsmnVadModel {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        FsmnVadModel::detect(self, waveform, options)
    }

    fn new_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadSession>> {
        Ok(Box::new(self.new_session(options)?))
    }
}

struct FsmnVadWeights {
    frontend: WavFrontend,
    in_linear1: Linear,
    in_linear2: Linear,
    fsmn_blocks: Vec<FsmnVadBlock>,
    out_linear1: Linear,
    out_linear2: Linear,
    device: Device,
    cpu: Option<FsmnCpuWeights>,
}

impl std::fmt::Debug for FsmnVadWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FsmnVadWeights").finish_non_exhaustive()
    }
}

impl FsmnVadWeights {
    fn load_on_device(model_dir: &Path, device: &Device) -> Result<Self> {
        let model_path = model_dir.join("model.pt");
        let cmvn_path = model_dir.join("am.mvn");
        if !model_path.exists() {
            bail!("FSMN VAD model.pt not found in {}", model_dir.display());
        }
        if !cmvn_path.exists() {
            bail!("FSMN VAD am.mvn not found in {}", model_dir.display());
        }
        let frontend = WavFrontend::new(WavFrontendConfig {
            sample_rate: FSMN_VAD_SAMPLE_RATE as i32,
            lfr_m: 5,
            lfr_n: 1,
            cmvn_file: Some(cmvn_path),
            ..Default::default()
        })?;

        let vb = VarBuilder::from_pth(&model_path, DType::F32, device)
            .with_context(|| format!("failed to load FSMN VAD weights {}", model_path.display()))?;
        let encoder_vb = vb.pp("encoder");
        let in_linear1 = linear(400, 140, encoder_vb.pp("in_linear1").pp("linear"))?;
        let in_linear2 = linear(140, 250, encoder_vb.pp("in_linear2").pp("linear"))?;
        let out_linear1 = linear(250, 140, encoder_vb.pp("out_linear1").pp("linear"))?;
        let out_linear2 = linear(140, 248, encoder_vb.pp("out_linear2").pp("linear"))?;

        let mut fsmn_blocks = Vec::with_capacity(FSMN_VAD_LAYERS);
        for i in 0..FSMN_VAD_LAYERS {
            fsmn_blocks.push(FsmnVadBlock::new(encoder_vb.pp("fsmn").pp(i))?);
        }
        let cpu = if device.is_cpu() {
            Some(FsmnCpuWeights::from_layers(
                &in_linear1,
                &in_linear2,
                &fsmn_blocks,
                &out_linear1,
                &out_linear2,
            )?)
        } else {
            None
        };

        Ok(Self {
            frontend,
            in_linear1,
            in_linear2,
            fsmn_blocks,
            out_linear1,
            out_linear2,
            device: device.clone(),
            cpu,
        })
    }

    fn zero_caches(&self) -> Result<Vec<Tensor>> {
        let cache = Tensor::zeros(
            (1usize, FSMN_VAD_PROJ_DIM, FSMN_VAD_CACHE_FRAMES, 1usize),
            DType::F32,
            &self.device,
        )?;
        Ok((0..FSMN_VAD_LAYERS)
            .map(|_| cache.clone())
            .collect::<Vec<_>>())
    }

    fn forward_frame_scores(
        &self,
        feats: &Tensor,
        caches: &mut [Tensor],
        num_threads: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let mut timing = FsmnVadForwardTiming::default();
        self.forward_frame_scores_with_timing(feats, caches, num_threads, &mut timing)
    }

    fn forward_frame_scores_with_timing(
        &self,
        feats: &Tensor,
        caches: &mut [Tensor],
        num_threads: usize,
        timing: &mut FsmnVadForwardTiming,
    ) -> Result<Vec<Vec<f32>>> {
        let (t, d) = feats.dims2()?;
        if d != 400 {
            bail!("FSMN VAD expects feature dim 400, got {d}");
        }
        if t == 0 {
            return Ok(Vec::new());
        }
        if let Some(cpu) = &self.cpu
            && self.device.is_cpu()
        {
            return cpu.forward_frame_scores(feats, caches, num_threads, timing);
        }

        // 前端特征固定在 CPU，这里先补上 batch 维再搬到模型所在设备。
        let speech = feats.unsqueeze(0)?.contiguous()?.to_device(&self.device)?;

        let mut x = self.in_linear1.forward(&speech)?;
        x = self.in_linear2.forward(&x)?.relu()?;
        for (idx, block) in self.fsmn_blocks.iter().enumerate() {
            let (new_x, new_cache) = block.forward(&x, &caches[idx])?;
            x = new_x;
            caches[idx] = new_cache;
        }
        x = self.out_linear1.forward(&x)?;
        x = self.out_linear2.forward(&x)?;
        let logits = softmax_last_dim(&x)?;

        let dims = logits.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] == 0 {
            bail!("unexpected FSMN VAD logits shape: {:?}", dims);
        }
        let frames = dims[1];
        let classes = dims[2];
        let data = logits.flatten_all()?.to_vec1::<f32>()?;
        Ok((0..frames)
            .map(|frame_idx| {
                (0..classes)
                    .map(|class_idx| data[frame_idx * classes + class_idx])
                    .collect()
            })
            .collect())
    }
}

struct FsmnCpuWeights {
    in_linear1: FsmnCpuLinear,
    in_linear2: FsmnCpuLinear,
    blocks: Vec<FsmnCpuBlock>,
    out_linear1: FsmnCpuLinear,
    out_linear2: FsmnCpuLinear,
}

struct FsmnCpuBlock {
    linear: FsmnCpuLinear,
    affine: FsmnCpuLinear,
    conv_left_weight: Vec<f32>,
}

struct FsmnCpuLinear {
    weight_t: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

impl FsmnCpuWeights {
    fn from_layers(
        in_linear1: &Linear,
        in_linear2: &Linear,
        blocks: &[FsmnVadBlock],
        out_linear1: &Linear,
        out_linear2: &Linear,
    ) -> Result<Self> {
        Ok(Self {
            in_linear1: FsmnCpuLinear::from_linear(in_linear1)?,
            in_linear2: FsmnCpuLinear::from_linear(in_linear2)?,
            blocks: blocks
                .iter()
                .map(FsmnCpuBlock::from_block)
                .collect::<Result<Vec<_>>>()?,
            out_linear1: FsmnCpuLinear::from_linear(out_linear1)?,
            out_linear2: FsmnCpuLinear::from_linear(out_linear2)?,
        })
    }

    fn forward_frame_scores(
        &self,
        feats: &Tensor,
        caches: &mut [Tensor],
        num_threads: usize,
        timing: &mut FsmnVadForwardTiming,
    ) -> Result<Vec<Vec<f32>>> {
        let frames = feats.dim(0)?;
        let mut cpu_caches = tensor_caches_to_cpu(caches)?;
        let scores = run_with_vad_threads(num_threads, || {
            self.forward_frame_scores_cpu(feats, &mut cpu_caches, timing)
        })??;
        for (cache, cpu_cache) in caches.iter_mut().zip(cpu_caches) {
            *cache = Tensor::from_vec(
                cpu_cache,
                (1usize, FSMN_VAD_PROJ_DIM, FSMN_VAD_CACHE_FRAMES, 1usize),
                &Device::Cpu,
            )?;
        }
        if scores.len() != frames {
            bail!(
                "FSMN CPU fast path returned {} frames for {frames} input frames",
                scores.len()
            );
        }
        Ok(scores)
    }

    fn forward_frame_scores_cpu(
        &self,
        feats: &Tensor,
        caches: &mut [Vec<f32>],
        timing: &mut FsmnVadForwardTiming,
    ) -> Result<Vec<Vec<f32>>> {
        let frames = feats.dim(0)?;
        // CPU 快速路径（gemm）要求行主序连续数据，`flatten_all` 会按需物化拷贝。
        let feats = feats.flatten_all()?.to_vec1::<f32>()?;
        let start = Instant::now();
        let mut x = self.in_linear1.forward(&feats, frames, false);
        timing.in_linear1_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        x = self.in_linear2.forward(&x, frames, true);
        timing.in_linear2_seconds += start.elapsed().as_secs_f64();

        for (idx, (block, cache)) in self.blocks.iter().zip(caches.iter_mut()).enumerate() {
            x = block.forward(x, frames, cache, idx, timing)?;
        }

        let start = Instant::now();
        x = self.out_linear1.forward(&x, frames, false);
        timing.out_linear1_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        x = self.out_linear2.forward(&x, frames, false);
        timing.out_linear2_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let rows = softmax_rows(&x, frames, self.out_linear2.out_dim);
        timing.softmax_seconds += start.elapsed().as_secs_f64();
        Ok(rows)
    }
}

impl FsmnCpuBlock {
    fn from_block(block: &FsmnVadBlock) -> Result<Self> {
        let conv_left_weight = block
            .conv_left_weight_cpu
            .clone()
            .ok_or_else(|| anyhow::anyhow!("FSMN CPU block missing conv weights"))?;
        Ok(Self {
            linear: FsmnCpuLinear::from_linear(&block.linear)?,
            affine: FsmnCpuLinear::from_linear(&block.affine)?,
            conv_left_weight,
        })
    }

    /// 单层 FSMN 前向：线性投影 + memory 卷积 + affine，并滚动更新 `cache`。
    fn forward(
        &self,
        input: Vec<f32>,
        frames: usize,
        cache: &mut [f32],
        idx: usize,
        timing: &mut FsmnVadForwardTiming,
    ) -> Result<Vec<f32>> {
        let start = Instant::now();
        let projected = self.linear.forward(&input, frames, false);
        timing.block_linear_seconds[idx] += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let memory = fsmn_block_memory_vec(&projected, cache, &self.conv_left_weight, frames)?;
        timing.block_memory_seconds[idx] += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let out = self.affine.forward(&memory, frames, true);
        timing.block_affine_seconds[idx] += start.elapsed().as_secs_f64();
        Ok(out)
    }
}

impl FsmnCpuLinear {
    fn from_linear(linear: &Linear) -> Result<Self> {
        let dims = linear.weight().dims();
        if dims.len() != 2 {
            bail!("FSMN VAD linear weight must be 2D, got {dims:?}");
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

fn tensor_caches_to_cpu(caches: &[Tensor]) -> Result<Vec<Vec<f32>>> {
    caches
        .iter()
        .map(|cache| {
            if cache.dims() != [1, FSMN_VAD_PROJ_DIM, FSMN_VAD_CACHE_FRAMES, 1] {
                bail!("unexpected FSMN cache shape: {:?}", cache.dims());
            }
            Ok(cache.flatten_all()?.to_vec1::<f32>()?)
        })
        .collect()
}

/// 对 `input` 做 FSMN 卷积并滚动更新 `cache`。
///
/// `cache` 取切片而非 `&mut Vec`，避免调用方为借用检查额外构造容器。
fn fsmn_block_memory_vec(
    input: &[f32],
    cache: &mut [f32],
    kernel: &[f32],
    frames: usize,
) -> Result<Vec<f32>> {
    if input.len() != frames * FSMN_VAD_PROJ_DIM {
        bail!("unexpected FSMN input length: {}", input.len());
    }
    if cache.len() != FSMN_VAD_PROJ_DIM * FSMN_VAD_CACHE_FRAMES {
        bail!("unexpected FSMN cache length: {}", cache.len());
    }
    if kernel.len() != FSMN_VAD_PROJ_DIM * FSMN_VAD_CONV_KERNEL {
        bail!("unexpected FSMN kernel length: {}", kernel.len());
    }

    let channels = FSMN_VAD_PROJ_DIM;
    let old_cache = cache.to_vec();
    let mut out = input.to_vec();
    out.par_chunks_mut(channels)
        .enumerate()
        .for_each(|(frame, dst)| {
            for k in 0..FSMN_VAD_CONV_KERNEL {
                let source_time = frame + k;
                if source_time < FSMN_VAD_CACHE_FRAMES {
                    for channel in 0..channels {
                        dst[channel] += old_cache[channel * FSMN_VAD_CACHE_FRAMES + source_time]
                            * kernel[channel * FSMN_VAD_CONV_KERNEL + k];
                    }
                } else {
                    let input_base = (source_time - FSMN_VAD_CACHE_FRAMES) * channels;
                    for channel in 0..channels {
                        dst[channel] += input[input_base + channel]
                            * kernel[channel * FSMN_VAD_CONV_KERNEL + k];
                    }
                }
            }
        });

    let combined_len = FSMN_VAD_CACHE_FRAMES + frames;
    cache
        .par_chunks_mut(FSMN_VAD_CACHE_FRAMES)
        .enumerate()
        .for_each(|(channel, dst)| {
            for (j, value) in dst.iter_mut().enumerate() {
                let source_time = combined_len - FSMN_VAD_CACHE_FRAMES + j;
                *value = if source_time < FSMN_VAD_CACHE_FRAMES {
                    old_cache[channel * FSMN_VAD_CACHE_FRAMES + source_time]
                } else {
                    input[(source_time - FSMN_VAD_CACHE_FRAMES) * channels + channel]
                };
            }
        });
    Ok(out)
}

fn softmax_rows(logits: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &logits[row_idx * cols..(row_idx + 1) * cols];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut out = row
                .iter()
                .map(|value| (*value - max).exp())
                .collect::<Vec<_>>();
            let sum = out.iter().sum::<f32>();
            if sum != 0.0 {
                for value in &mut out {
                    *value /= sum;
                }
            }
            out
        })
        .collect()
}

struct FsmnVadBlock {
    linear: Linear,
    affine: Linear,
    conv_left_weight: Tensor,
    conv_left_weight_cpu: Option<Vec<f32>>,
}

impl FsmnVadBlock {
    fn new(vb: VarBuilder) -> Result<Self> {
        let linear_layer = linear_no_bias(250, FSMN_VAD_PROJ_DIM, vb.pp("linear").pp("linear"))?;
        let affine = linear(FSMN_VAD_PROJ_DIM, 250, vb.pp("affine").pp("linear"))?;
        let conv_left_weight = vb
            .pp("fsmn_block")
            .pp("conv_left")
            .get((FSMN_VAD_PROJ_DIM, 1, 20, 1), "weight")?;
        let conv_left_weight_cpu = if conv_left_weight.device().is_cpu() {
            Some(conv_left_weight.flatten_all()?.to_vec1::<f32>()?)
        } else {
            None
        };
        Ok(Self {
            linear: linear_layer,
            affine,
            conv_left_weight,
            conv_left_weight_cpu,
        })
    }

    fn forward(&self, input: &Tensor, cache: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.linear.forward(input)?;
        let (out, new_cache) = if let Some(kernel) = &self.conv_left_weight_cpu
            && x.device().is_cpu()
            && cache.device().is_cpu()
        {
            fsmn_block_memory_cpu(&x, cache, kernel)?
        } else {
            let x_per = x.unsqueeze(1)?.permute((0, 3, 2, 1))?;
            let y_left = Tensor::cat(&[cache, &x_per], 2)?;
            let y_left_t = y_left.dim(2)?;
            let new_cache = y_left
                .narrow(2, y_left_t - FSMN_VAD_CACHE_FRAMES, FSMN_VAD_CACHE_FRAMES)?
                .contiguous()?
                .detach();
            let y_left = fsmn_depthwise_conv_left(
                &y_left,
                &self.conv_left_weight,
                &self.conv_left_weight_cpu,
            )?;
            let out = x_per.add(&y_left)?;
            let out = out.permute((0, 3, 2, 1))?.squeeze(1)?;
            (out, new_cache)
        };
        let out = self.affine.forward(&out)?.relu()?;
        Ok((out, new_cache))
    }
}

fn fsmn_block_memory_cpu(
    projected: &Tensor,
    cache: &Tensor,
    kernel: &[f32],
) -> Result<(Tensor, Tensor)> {
    let (batch, frames, channels) = projected.dims3()?;
    if batch != 1 {
        bail!("FSMN CPU memory expects batch=1, got {batch}");
    }
    if channels != FSMN_VAD_PROJ_DIM {
        bail!("FSMN CPU memory expects {FSMN_VAD_PROJ_DIM} channels, got {channels}");
    }
    if cache.dims() != [1, channels, FSMN_VAD_CACHE_FRAMES, 1] {
        bail!("unexpected FSMN cache shape: {:?}", cache.dims());
    }
    if kernel.len() != channels * FSMN_VAD_CONV_KERNEL {
        bail!("unexpected FSMN kernel length: {}", kernel.len());
    }

    let input = projected.flatten_all()?.to_vec1::<f32>()?;
    let cache = cache.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; frames * channels];
    let mut new_cache = vec![0f32; channels * FSMN_VAD_CACHE_FRAMES];

    out.copy_from_slice(&input);
    for frame in 0..frames {
        let out_base = frame * channels;
        for k in 0..FSMN_VAD_CONV_KERNEL {
            let source_time = frame + k;
            if source_time < FSMN_VAD_CACHE_FRAMES {
                for channel in 0..channels {
                    out[out_base + channel] += cache[channel * FSMN_VAD_CACHE_FRAMES + source_time]
                        * kernel[channel * FSMN_VAD_CONV_KERNEL + k];
                }
            } else {
                let input_base = (source_time - FSMN_VAD_CACHE_FRAMES) * channels;
                for channel in 0..channels {
                    out[out_base + channel] +=
                        input[input_base + channel] * kernel[channel * FSMN_VAD_CONV_KERNEL + k];
                }
            }
        }
    }

    let combined_len = FSMN_VAD_CACHE_FRAMES + frames;
    for channel in 0..channels {
        let cache_base = channel * FSMN_VAD_CACHE_FRAMES;
        for j in 0..FSMN_VAD_CACHE_FRAMES {
            let source_time = combined_len - FSMN_VAD_CACHE_FRAMES + j;
            new_cache[cache_base + j] = if source_time < FSMN_VAD_CACHE_FRAMES {
                cache[cache_base + source_time]
            } else {
                input[(source_time - FSMN_VAD_CACHE_FRAMES) * channels + channel]
            };
        }
    }

    Ok((
        Tensor::from_vec(out, (1usize, frames, channels), projected.device())?,
        Tensor::from_vec(
            new_cache,
            (1usize, channels, FSMN_VAD_CACHE_FRAMES, 1usize),
            projected.device(),
        )?,
    ))
}

const FSMN_VAD_CONV_KERNEL: usize = 20;

/// Depthwise time convolution for FSMN left context.
/// Implemented with `unfold` + multiply-accumulate so CPU and CUDA stay numerically aligned.
/// Candle's grouped `conv2d` is incorrect on CUDA for this kernel shape.
fn fsmn_depthwise_conv_left(
    y_left: &Tensor,
    weight: &Tensor,
    weight_cpu: &Option<Vec<f32>>,
) -> Result<Tensor> {
    let (batch, channels, time, width) = y_left.dims4()?;
    if batch != 1 {
        bail!("FSMN depthwise conv expects batch=1, got {batch}");
    }
    if width != 1 {
        bail!("FSMN depthwise conv expects width=1, got {width}");
    }
    if channels != FSMN_VAD_PROJ_DIM {
        bail!("FSMN depthwise conv expects {FSMN_VAD_PROJ_DIM} channels, got {channels}");
    }
    if time < FSMN_VAD_CONV_KERNEL {
        bail!(
            "FSMN depthwise conv input too short: {time} < {}",
            FSMN_VAD_CONV_KERNEL
        );
    }
    let weight_dims = weight.dims();
    if weight_dims != [channels, 1, FSMN_VAD_CONV_KERNEL, 1] {
        bail!("unexpected FSMN conv weight shape: {weight_dims:?}");
    }

    if y_left.device().is_cpu()
        && let Some(weight_cpu) = weight_cpu
    {
        return fsmn_depthwise_conv_left_cpu(y_left, weight_cpu, channels, time);
    }

    let windows = y_left.unfold(2, FSMN_VAD_CONV_KERNEL, 1)?;
    let kernel = weight.reshape((1, channels, 1, 1, FSMN_VAD_CONV_KERNEL))?;
    let out = windows.broadcast_mul(&kernel)?.sum_keepdim(4)?.squeeze(4)?;
    Ok(out)
}

fn fsmn_depthwise_conv_left_cpu(
    y_left: &Tensor,
    kernel: &[f32],
    channels: usize,
    time: usize,
) -> Result<Tensor> {
    let input = y_left.flatten_all()?.to_vec1::<f32>()?;
    let out_time = time - FSMN_VAD_CONV_KERNEL + 1;
    let mut out = vec![0f32; channels * out_time];

    for channel in 0..channels {
        let input_base = channel * time;
        let kernel_base = channel * FSMN_VAD_CONV_KERNEL;
        let out_base = channel * out_time;
        for t in 0..out_time {
            let mut sum = 0f32;
            for k in 0..FSMN_VAD_CONV_KERNEL {
                sum += input[input_base + t + k] * kernel[kernel_base + k];
            }
            out[out_base + t] = sum;
        }
    }

    Ok(Tensor::from_vec(
        out,
        (1usize, channels, out_time, 1usize),
        y_left.device(),
    )?)
}

pub struct FsmnVadSession {
    weights: Arc<FsmnVadWeights>,
    caches: Vec<Tensor>,
    e2e: E2EVadModel,
    vad_options: VadOptions,
    pending: Vec<i16>,
    timing: FsmnVadTiming,
}

impl FsmnVadSession {
    fn new(weights: Arc<FsmnVadWeights>, options: VadOptions) -> Result<Self> {
        let caches = weights.zero_caches()?;
        Ok(Self {
            weights,
            caches,
            e2e: build_e2e_model(&options),
            vad_options: options,
            pending: Vec::with_capacity(FSMN_VAD_STREAM_SCORE_SAMPLES),
            timing: FsmnVadTiming::default(),
        })
    }

    fn push_samples_f32(&mut self, samples: &[f32]) -> Result<Vec<VadSegment>> {
        let pcm_start = Instant::now();
        let pcm = samples_f32_to_i16(samples);
        self.timing.add_pcm(pcm_start.elapsed());
        self.push_samples_i16(&pcm)
    }

    fn push_samples_i16(&mut self, samples: &[i16]) -> Result<Vec<VadSegment>> {
        let mut segments = Vec::new();
        let mut input = samples;

        if !self.pending.is_empty() {
            let needed = FSMN_VAD_STREAM_SCORE_SAMPLES - self.pending.len();
            let take = needed.min(input.len());
            self.pending.extend_from_slice(&input[..take]);
            input = &input[take..];
            if self.pending.len() == FSMN_VAD_STREAM_SCORE_SAMPLES {
                let chunk = std::mem::take(&mut self.pending);
                segments.extend(self.score_and_segment(&chunk, false)?);
            }
        }

        let (full_chunks, tail) = split_full_chunks(input, FSMN_VAD_STREAM_SCORE_SAMPLES);
        let (chunks, _remainder) = full_chunks.as_chunks::<FSMN_VAD_STREAM_SCORE_SAMPLES>();
        for chunk_slice in chunks {
            segments.extend(self.score_and_segment(chunk_slice, false)?);
        }
        self.pending.extend_from_slice(tail);
        Ok(segments
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(start as usize, end as usize),
                probability: self.vad_options.threshold,
            })
            .collect())
    }

    fn finish_segments(&mut self) -> Result<Vec<VadSegment>> {
        let mut segments = Vec::new();
        if !self.pending.is_empty() {
            let chunk = std::mem::take(&mut self.pending);
            segments.extend(self.score_and_segment(&chunk, true)?);
        } else {
            let segmenter_start = Instant::now();
            segments.extend(
                self.e2e
                    .finalize_streaming(self.vad_options.min_silence_ms as i32),
            );
            self.timing.add_segmenter(segmenter_start.elapsed());
        }
        Ok(segments
            .into_iter()
            .map(|(start, end)| VadSegment {
                range: TimeRange::new(start as usize, end as usize),
                probability: self.vad_options.threshold,
            })
            .collect())
    }

    fn score_and_segment(&mut self, chunk: &[i16], is_final: bool) -> Result<Vec<(u64, u64)>> {
        if chunk.is_empty() {
            return Ok(Vec::new());
        }
        let padded;
        let chunk = if chunk.len() < FSMN_VAD_CHUNK_SIZE {
            padded = {
                let mut padded = vec![0i16; FSMN_VAD_CHUNK_SIZE];
                padded[..chunk.len()].copy_from_slice(chunk);
                padded
            };
            padded.as_slice()
        } else {
            chunk
        };
        let frontend_start = Instant::now();
        let feats = self.weights.frontend.extract_features(chunk)?;
        self.timing.frontend_seconds += frontend_start.elapsed().as_secs_f64();

        let forward_start = Instant::now();
        let frame_scores = self.weights.forward_frame_scores(
            &feats,
            &mut self.caches,
            self.vad_options.num_threads,
        )?;
        self.timing.forward_seconds += forward_start.elapsed().as_secs_f64();
        self.timing.chunks += 1;

        let wave_chunk: Vec<f32> = chunk
            .iter()
            .map(|&sample| f32::from(sample) / f32::from(i16::MAX))
            .collect();
        let segmenter_start = Instant::now();
        let segments = self.e2e.detect_chunk(
            &frame_scores,
            &wave_chunk,
            is_final,
            self.vad_options.min_silence_ms as i32,
        );
        self.timing.add_segmenter(segmenter_start.elapsed());
        Ok(segments)
    }
}

fn split_full_chunks(samples: &[i16], chunk_size: usize) -> (&[i16], &[i16]) {
    let full_len = samples.len() / chunk_size * chunk_size;
    samples.split_at(full_len)
}

fn samples_f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|sample| {
            (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX))
                .round()
                .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
        })
        .collect()
}

fn extract_cmvn_vector(text: &str, section: &str) -> Result<Vec<f32>> {
    let section_start = text
        .find(section)
        .ok_or_else(|| anyhow::anyhow!("missing {section} section"))?;
    let after_section = &text[section_start + section.len()..];
    let learn_rate = "<LearnRateCoef>";
    let learn_start = after_section
        .find(learn_rate)
        .ok_or_else(|| anyhow::anyhow!("missing {learn_rate} after {section}"))?;
    let after_learn = &after_section[learn_start + learn_rate.len()..];
    let bracket_start = after_learn
        .find('[')
        .ok_or_else(|| anyhow::anyhow!("missing vector start after {section}"))?;
    let after_bracket = &after_learn[bracket_start + 1..];
    let bracket_end = after_bracket
        .find(']')
        .ok_or_else(|| anyhow::anyhow!("missing vector end after {section}"))?;
    let values = after_bracket[..bracket_end]
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f32>()
                .with_context(|| format!("invalid CMVN value {token:?} in {section}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() {
        bail!("empty CMVN vector in {section}");
    }
    Ok(values)
}

impl StreamingVadSession for FsmnVadSession {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<VadEvent>> {
        if chunk.sample_rate != FSMN_VAD_SAMPLE_RATE {
            bail!(
                "FSMN VAD expects 16kHz mono audio, got sample_rate={}",
                chunk.sample_rate
            );
        }
        let segments = self.push_samples_f32(&chunk.samples)?;
        Ok(segments
            .into_iter()
            .map(|segment| VadEvent {
                range: segment.range,
                status: VadEventStatus::Final,
            })
            .collect())
    }

    fn finish(&mut self) -> Result<Vec<VadEvent>> {
        Ok(self
            .finish_segments()?
            .into_iter()
            .map(|segment| VadEvent {
                range: segment.range,
                status: VadEventStatus::Final,
            })
            .collect())
    }
}

#[cfg(test)]
mod legacy_vad_segmenter {
    use std::collections::VecDeque;

    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum VadState {
        Waiting,
        Recording,
    }

    pub(super) struct VadSegmenter {
        options: VadOptions,
        state: VadState,
        history: VecDeque<i16>,
        speech_start: usize,
        silence_chunks: usize,
        current_chunks: usize,
        chunk_cursor: usize,
    }

    impl VadSegmenter {
        pub(super) fn new(options: VadOptions) -> Self {
            Self {
                options,
                state: VadState::Waiting,
                history: VecDeque::new(),
                speech_start: 0,
                silence_chunks: 0,
                current_chunks: 0,
                chunk_cursor: 0,
            }
        }

        pub(super) fn push_probability(&mut self, probability: f32) -> Vec<VadSegment> {
            let chunk_start = self.chunk_cursor * FSMN_VAD_CHUNK_SIZE;
            self.chunk_cursor += 1;
            match self.state {
                VadState::Waiting => {
                    self.push_history_chunk();
                    if probability >= self.options.threshold {
                        self.state = VadState::Recording;
                        self.speech_start = chunk_start.saturating_sub(self.history.len());
                        self.silence_chunks = 0;
                        self.current_chunks = 1;
                        self.history.clear();
                    }
                    Vec::new()
                }
                VadState::Recording => {
                    self.current_chunks += 1;
                    if probability >= self.options.threshold {
                        self.silence_chunks = 0;
                        if self.current_samples() >= ms_to_samples(9_000) {
                            return self.finalize(chunk_start + FSMN_VAD_CHUNK_SIZE, false);
                        }
                        return Vec::new();
                    }
                    self.silence_chunks += 1;
                    if self.silence_chunks * FSMN_VAD_CHUNK_SIZE
                        >= ms_to_samples(self.options.min_silence_ms)
                    {
                        return self.finalize(chunk_start + FSMN_VAD_CHUNK_SIZE, true);
                    }
                    Vec::new()
                }
            }
        }

        pub(super) fn finish(&mut self) -> Vec<VadSegment> {
            let end = self.chunk_cursor * FSMN_VAD_CHUNK_SIZE;
            self.finalize(end, false)
        }

        fn push_history_chunk(&mut self) {
            let rollback = ms_to_samples(200);
            self.history
                .extend(std::iter::repeat_n(0i16, FSMN_VAD_CHUNK_SIZE));
            while self.history.len() > rollback {
                self.history.pop_front();
            }
        }

        fn current_samples(&self) -> usize {
            self.current_chunks * FSMN_VAD_CHUNK_SIZE
        }

        fn finalize(&mut self, end: usize, trim_tail: bool) -> Vec<VadSegment> {
            if self.state != VadState::Recording {
                return Vec::new();
            }
            let end = if trim_tail {
                end.saturating_sub(self.silence_chunks * FSMN_VAD_CHUNK_SIZE)
            } else {
                end
            };
            let start = self.speech_start.min(end);
            let duration = end.saturating_sub(start);
            self.state = VadState::Waiting;
            self.silence_chunks = 0;
            self.current_chunks = 0;
            self.history.clear();
            if duration < ms_to_samples(self.options.min_speech_ms) {
                return Vec::new();
            }
            vec![VadSegment {
                range: sample_range(start, end),
                probability: self.options.threshold,
            }]
        }
    }
}

#[derive(Debug, Clone)]
struct WavFrontendConfig {
    sample_rate: i32,
    frame_length_ms: f32,
    frame_shift_ms: f32,
    n_mels: usize,
    lfr_m: usize,
    lfr_n: usize,
    cmvn_file: Option<PathBuf>,
}

impl Default for WavFrontendConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            n_mels: 80,
            lfr_m: 7,
            lfr_n: 6,
            cmvn_file: None,
        }
    }
}

#[derive(Debug, Clone)]
struct WavFrontend {
    config: WavFrontendConfig,
    cmvn_means: Option<Vec<f32>>,
    cmvn_vars: Option<Vec<f32>>,
}

impl WavFrontend {
    fn new(config: WavFrontendConfig) -> Result<Self> {
        let (cmvn_means, cmvn_vars) = if let Some(cmvn_path) = &config.cmvn_file {
            let (means, vars) = Self::load_cmvn(cmvn_path)?;
            (Some(means), Some(vars))
        } else {
            (None, None)
        };
        Ok(Self {
            config,
            cmvn_means,
            cmvn_vars,
        })
    }

    /// 解析 Kaldi `am.mvn`，返回 `(AddShift, Rescale)` 两个等长向量。
    fn load_cmvn(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read CMVN file {}", path.display()))?;
        let means = extract_cmvn_vector(&text, "<AddShift>")
            .with_context(|| format!("failed to parse AddShift CMVN in {}", path.display()))?;
        let vars = extract_cmvn_vector(&text, "<Rescale>")
            .with_context(|| format!("failed to parse Rescale CMVN in {}", path.display()))?;
        if means.len() != vars.len() {
            bail!(
                "CMVN file {} has mismatched AddShift/Rescale dims: {} vs {}",
                path.display(),
                means.len(),
                vars.len()
            );
        }
        Ok((means, vars))
    }

    fn extract_features(&self, waveform: &[i16]) -> Result<Tensor> {
        self.extract_features_from_pcm(waveform)
    }

    fn extract_features_from_pcm(&self, samples: &[i16]) -> Result<Tensor> {
        let waveform = samples
            .iter()
            .map(|&sample| f32::from(sample))
            .collect::<Vec<_>>();
        let fbank = self.compute_fbank_features(&waveform)?;
        let lfr = self.apply_lfr(&fbank)?;
        self.apply_cmvn(&lfr)
    }

    fn extract_features_from_normalized_f32(&self, samples: &[f32]) -> Result<Tensor> {
        let waveform = samples
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
            .collect::<Vec<_>>();
        let fbank = self.compute_fbank_features(&waveform)?;
        let lfr = self.apply_lfr(&fbank)?;
        self.apply_cmvn(&lfr)
    }

    /// 用 Kaldi fbank（C 实现）计算对数梅尔谱。
    ///
    /// 结果固定在 CPU 上，形状为 `(frames, n_mels)`。
    fn compute_fbank_features(&self, waveform: &[f32]) -> Result<Tensor> {
        let opt = FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: self.config.sample_rate as f32,
                // `c"hamming"` 字面量在编译期完成 NUL 终止校验。
                window_type: c"hamming".as_ptr(),
                dither: 0.0,
                frame_shift_ms: self.config.frame_shift_ms,
                frame_length_ms: self.config.frame_length_ms,
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: self.config.n_mels as i32,
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };
        let mut fbank = OnlineFbank::new(opt);
        fbank.accept_waveform(self.config.sample_rate as f32, waveform);
        let frames = fbank.num_ready_frames();
        let mut out = Vec::with_capacity(frames as usize * self.config.n_mels);
        for i in 0..frames {
            let frame = fbank
                .get_frame(i)
                .ok_or_else(|| anyhow::anyhow!("missing fbank frame {i}"))?;
            out.extend_from_slice(frame);
        }
        Ok(Tensor::from_vec(
            out,
            (frames as usize, self.config.n_mels),
            &Device::Cpu,
        )?)
    }

    /// 按 LFR-m/LFR-n 把 fbank 特征拼接成 LFR 特征。
    ///
    /// 与 Kaldi `apply-lfr` 行为一致：先在左侧复制 `(lfr_m - 1) / 2` 行第 0 帧填充，
    /// 再按步长 `lfr_n` 连续取 `lfr_m` 行展平拼接；末尾不足 `lfr_m` 行时重复
    /// 最后一帧补齐，因此输出宽度恒为 `n_mels * lfr_m`。
    ///
    /// 输入为 `(frames, n_mels)`，输出为 `(frames_lfr, n_mels * lfr_m)`，均在 CPU 上。
    /// 输入先用 `flatten_all` 取成连续切片，填充与窗口展开一次写入 `Vec` 后构造张量，
    /// 避免逐窗张量算子带来的开销。
    ///
    /// # Errors
    ///
    /// 输入不是二维张量或张量运算失败时返回错误。
    fn apply_lfr(&self, fbank: &Tensor) -> Result<Tensor> {
        let (t, n_mels) = fbank.dims2()?;
        let lfr_m = self.config.lfr_m;
        let feat_dim = n_mels * lfr_m;
        if t == 0 {
            return Ok(Tensor::zeros((0, feat_dim), DType::F32, &Device::Cpu)?);
        }
        let t_lfr = t.div_ceil(self.config.lfr_n);
        let left_padding_rows = (lfr_m - 1) / 2;

        // 左侧填充第 0 帧；尾部额外重复最后一帧 `lfr_m - 1` 次，
        // 这样每个窗口都能取满 `lfr_m` 行，末尾补帧分支即可省去。
        let features = fbank.flatten_all()?.to_vec1::<f32>()?;
        let mut padded = Vec::with_capacity((t + left_padding_rows + lfr_m - 1) * n_mels);
        for _ in 0..left_padding_rows {
            padded.extend_from_slice(&features[..n_mels]);
        }
        padded.extend_from_slice(&features);
        let last_row = &features[(t - 1) * n_mels..];
        for _ in 1..lfr_m {
            padded.extend_from_slice(last_row);
        }

        let mut out = Vec::with_capacity(t_lfr * feat_dim);
        for i in 0..t_lfr {
            let start = i * self.config.lfr_n * n_mels;
            out.extend_from_slice(&padded[start..start + feat_dim]);
        }
        Ok(Tensor::from_vec(out, (t_lfr, feat_dim), &Device::Cpu)?)
    }

    /// 应用 CMVN 归一化：`(feats + means) * vars`，沿帧维广播。
    ///
    /// `cmvn_means` / `cmvn_vars` 缺失或维度不匹配时原样返回输入，与旧实现保持一致。
    ///
    /// # Errors
    ///
    /// 输入不是二维张量或广播失败时返回错误。
    fn apply_cmvn(&self, feats: &Tensor) -> Result<Tensor> {
        let (Some(means), Some(vars)) = (&self.cmvn_means, &self.cmvn_vars) else {
            return Ok(feats.clone());
        };
        let dim = feats.dim(1)?;
        if means.len() != dim || vars.len() != dim {
            return Ok(feats.clone());
        }
        let device = feats.device();
        let means = Tensor::from_vec(means.clone(), (1, dim), device)?;
        let vars = Tensor::from_vec(vars.clone(), (1, dim), device)?;
        Ok(feats.broadcast_add(&means)?.broadcast_mul(&vars)?)
    }
}

fn download_fsmn_vad() -> Result<PathBuf> {
    let cache = modelhub::modelscope_cache_dir();
    let legacy = cache.join(FSMN_VAD_REPO);
    if fsmn_vad_cache_complete(&legacy)? {
        return Ok(legacy);
    }
    let target = modelscope_model_snapshot_dir(&cache, FSMN_VAD_REPO);
    if fsmn_vad_cache_complete(&target)? {
        return Ok(target);
    }
    if target.exists() {
        tracing::warn!(
            target: "vasr_vad::fsmn",
            "FSMN VAD cache is incomplete at {}, downloading/repairing from ModelScope",
            target.display()
        );
    }
    let downloaded = block_on_async(modelhub::modelscope::download_model(FSMN_VAD_REPO, &cache))?;
    downloaded.with_context(|| format!("failed to download FSMN VAD model {FSMN_VAD_REPO:?}"))?;
    validate_fsmn_vad_model_dir(&target)?;
    Ok(target)
}

fn modelscope_model_snapshot_dir(cache: &Path, repo_id: &str) -> PathBuf {
    cache
        .join("models")
        .join(repo_id.replace('/', "--"))
        .join("snapshots")
        .join("master")
}

fn fsmn_vad_cache_complete(model_dir: &Path) -> Result<bool> {
    if !model_dir.is_dir() {
        return Ok(false);
    }
    Ok(nonempty_file(&model_dir.join("model.pt"))? && nonempty_file(&model_dir.join("am.mvn"))?)
}

fn validate_fsmn_vad_model_dir(model_dir: &Path) -> Result<()> {
    if !model_dir.is_dir() {
        bail!(
            "FSMN VAD model path is not a directory: {}",
            model_dir.display()
        );
    }
    if fsmn_vad_cache_complete(model_dir)? {
        return Ok(());
    }
    bail!(
        "FSMN VAD model directory is incomplete: {} (expected non-empty model.pt and am.mvn)",
        model_dir.display()
    );
}

fn nonempty_file(path: &Path) -> Result<bool> {
    let meta = match std::fs::metadata(path) {
        Ok(meta) => meta,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err).with_context(|| format!("failed to stat {}", path.display())),
    };
    Ok(meta.is_file() && meta.len() > 0)
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

// 以下采样点 / 毫秒换算只在测试里的流式分段器与断言中使用。
#[cfg(test)]
fn sample_range(start: usize, end: usize) -> TimeRange {
    TimeRange::new(sample_to_ms(start), sample_to_ms(end))
}

#[cfg(test)]
fn sample_to_ms(sample: usize) -> usize {
    sample.saturating_mul(1000) / FSMN_VAD_SAMPLE_RATE as usize
}

#[cfg(test)]
fn ms_to_samples(ms: u64) -> usize {
    (ms as usize).saturating_mul(FSMN_VAD_SAMPLE_RATE as usize) / 1000
}

#[cfg(test)]
mod tests {
    use super::legacy_vad_segmenter::VadSegmenter;
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(feature = "metal")]
    #[test]
    fn fsmn_vad_device_accepts_metal() -> Result<()> {
        // SAFETY: the test sets the process environment before device resolution.
        unsafe { std::env::set_var("VASR_VAD_DEVICE", "metal") };
        let device = fsmn_vad_device()?;
        assert!(device.is_metal());
        Ok(())
    }

    fn unique_tmp_dir(prefix: &str) -> Result<PathBuf> {
        let mut dir = std::env::temp_dir();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| anyhow::anyhow!("system time before unix epoch: {e:?}"))?;
        dir.push(format!("{prefix}_{}", now.as_nanos()));
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    fn write_minimal_fsmn_vad(dir: &Path) -> Result<()> {
        fs::write(dir.join("model.pt"), [1u8])?;
        fs::write(dir.join("am.mvn"), [1u8])?;
        Ok(())
    }

    #[test]
    fn fsmn_vad_cache_complete_accepts_required_files() -> Result<()> {
        let dir = unique_tmp_dir("fsmn_vad_complete")?;
        write_minimal_fsmn_vad(&dir)?;

        assert!(fsmn_vad_cache_complete(&dir)?);
        Ok(())
    }

    #[test]
    fn fsmn_vad_cache_complete_rejects_missing_cmvn() -> Result<()> {
        let dir = unique_tmp_dir("fsmn_vad_missing_cmvn")?;
        fs::write(dir.join("model.pt"), [1u8])?;

        assert!(!fsmn_vad_cache_complete(&dir)?);
        Ok(())
    }

    #[test]
    fn fsmn_vad_cache_complete_rejects_empty_model() -> Result<()> {
        let dir = unique_tmp_dir("fsmn_vad_empty_model")?;
        write_minimal_fsmn_vad(&dir)?;
        fs::write(dir.join("model.pt"), [])?;

        assert!(!fsmn_vad_cache_complete(&dir)?);
        Ok(())
    }

    #[test]
    fn segmenter_keeps_sessions_independent() {
        let options = VadOptions {
            threshold: 0.5,
            min_speech_ms: 1,
            min_silence_ms: 1,
            ..VadOptions::default()
        };
        let mut first = VadSegmenter::new(options.clone());
        let mut second = VadSegmenter::new(options);

        assert!(first.push_probability(0.9).is_empty());
        assert!(second.push_probability(0.1).is_empty());
        let first_segments = first.push_probability(0.1);
        let second_segments = second.finish();

        assert_eq!(first_segments.len(), 1);
        assert!(second_segments.is_empty());
    }

    #[test]
    fn segmenter_rolls_back_speech_start() {
        let options = VadOptions {
            threshold: 0.5,
            min_speech_ms: 1,
            min_silence_ms: 1,
            ..VadOptions::default()
        };
        let mut segmenter = VadSegmenter::new(options);

        segmenter.push_probability(0.1);
        segmenter.push_probability(0.1);
        segmenter.push_probability(0.9);
        let segments = segmenter.push_probability(0.1);

        assert_eq!(segments.len(), 1);
        assert!(segments[0].range.start_ms < sample_to_ms(2 * FSMN_VAD_CHUNK_SIZE));
    }

    #[test]
    fn hard_split_segment_respects_max_segment_ms() {
        let segment = VadSegment {
            range: TimeRange::new(100, 760),
            probability: 0.9,
        };

        let split = hard_split_segment(&segment, 250);

        let ranges = split
            .iter()
            .map(|segment| (segment.range.start_ms, segment.range.end_ms))
            .collect::<Vec<_>>();
        assert_eq!(ranges, vec![(100, 350), (350, 600), (600, 760)]);
    }

    fn split_segments_without_refine(
        segments: &[VadSegment],
        max_segment_ms: u64,
    ) -> Vec<VadSegment> {
        segments
            .iter()
            .flat_map(|segment| hard_split_segment(segment, max_segment_ms))
            .collect()
    }

    #[test]
    fn splitting_raw_segments_individually_matches_batch_split_for_hard_split() -> Result<()> {
        let options = VadOptions {
            max_segment_ms: 250,
            ..VadOptions::default()
        };
        let raw_segments = vec![
            VadSegment {
                range: TimeRange::new(100, 760),
                probability: 0.9,
            },
            VadSegment {
                range: TimeRange::new(900, 1300),
                probability: 0.9,
            },
        ];

        let batch = split_segments_without_refine(&raw_segments, options.max_segment_ms);
        let individual = raw_segments
            .iter()
            .flat_map(|segment| hard_split_segment(segment, options.max_segment_ms))
            .collect::<Vec<_>>();

        assert_eq!(individual, batch);
        Ok(())
    }

    #[test]
    #[ignore = "downloads funasr/fsmn-vad and uses local raw_audios fixture"]
    fn fsmn_vad_detects_local_raw_audio() -> Result<()> {
        let wav = std::path::Path::new("raw_audios/audio (13).wav");
        if !wav.exists() {
            return Ok(());
        }
        let waveform = load_pcm16_wav(wav)?;
        let model = FsmnVadModel::from_pretrained(None::<&str>)?;
        let segments = model.detect(&waveform, &VadOptions::default())?;
        assert!(
            !segments.is_empty(),
            "FSMN VAD should detect speech in {}",
            wav.display()
        );
        Ok(())
    }

    #[test]
    #[ignore = "uses local FSMN VAD and raw_audios fixture"]
    fn split_raw_segments_individually_matches_batch_split_on_local_audio() -> Result<()> {
        let wav = std::path::Path::new("raw_audios/audio (13).wav");
        let model_dir = std::path::Path::new(".cache/fsmn-vad");
        if !wav.exists() || !model_dir.exists() {
            return Ok(());
        }
        let waveform = load_pcm16_wav(wav)?;
        let model = FsmnVadModel::from_pretrained(Some(model_dir))?;
        let options = VadOptions::default();
        let raw_segments = model.detect_raw_segments(&waveform, &options)?;

        let batch = model.split_segments_for_asr(&waveform, &raw_segments, &options)?;
        let mut individual = Vec::new();
        for segment in &raw_segments {
            individual.extend(model.split_segments_for_asr(
                &waveform,
                std::slice::from_ref(segment),
                &options,
            )?);
        }

        assert_eq!(individual, batch);
        assert_eq!(batch, model.detect(&waveform, &options)?);
        Ok(())
    }

    #[cfg(feature = "cuda-vad")]
    #[test]
    #[ignore = "requires CUDA toolkit and .cache/fsmn-vad"]
    fn forward_frame_scores_match_cpu_and_cuda() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let model_dir = workspace.join(".cache/fsmn-vad");
        let wav = workspace.join("raw_audios/audio (13).wav");
        if !model_dir.exists() || !wav.exists() {
            return Ok(());
        }

        let cpu = FsmnVadWeights::load_on_device(&model_dir, &Device::Cpu)?;
        let cuda = FsmnVadWeights::load_on_device(&model_dir, &Device::new_cuda(0)?)?;
        let waveform = load_pcm16_wav(&wav)?;
        let pcm = samples_f32_to_i16(&waveform.samples);
        let feats = cpu.frontend.extract_features_from_pcm(&pcm)?;

        let mut cpu_caches = cpu.zero_caches()?;
        let mut cuda_caches = cuda.zero_caches()?;
        let cpu_scores = cpu.forward_frame_scores(&feats, &mut cpu_caches, 0)?;
        let cuda_scores = cuda.forward_frame_scores(&feats, &mut cuda_caches, 0)?;
        assert_eq!(cpu_scores.len(), cuda_scores.len());

        let mut max_diff = 0.0f32;
        let mut worst = (0usize, 0usize, 0.0f32, 0.0f32);
        for (frame_idx, (cpu_frame, cuda_frame)) in
            cpu_scores.iter().zip(cuda_scores.iter()).enumerate()
        {
            assert_eq!(cpu_frame.len(), cuda_frame.len());
            for (class_idx, (cpu_score, cuda_score)) in
                cpu_frame.iter().zip(cuda_frame.iter()).enumerate()
            {
                let diff = (cpu_score - cuda_score).abs();
                if diff > max_diff {
                    max_diff = diff;
                    worst = (frame_idx, class_idx, *cpu_score, *cuda_score);
                }
            }
        }
        println!(
            "forward_score_max_diff={max_diff} worst_frame={} worst_class={} cpu={} cuda={}",
            worst.0, worst.1, worst.2, worst.3
        );
        assert!(
            max_diff < 1e-4,
            "CPU/CUDA forward scores diverged: max_diff={max_diff}"
        );
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a local FSMN VAD checkpoint and Metal device"]
    fn forward_frame_scores_match_cpu_and_metal() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let model_dir = workspace.join(".cache/fsmn-vad");
        let wav = workspace.join("raw_audios/audio (13).wav");
        if !model_dir.exists() || !wav.exists() {
            return Ok(());
        }

        let cpu = FsmnVadWeights::load_on_device(&model_dir, &Device::Cpu)?;
        let metal = FsmnVadWeights::load_on_device(&model_dir, &Device::new_metal(0)?)?;
        let waveform = load_pcm16_wav(&wav)?;
        let pcm = samples_f32_to_i16(&waveform.samples);
        let feats = cpu.frontend.extract_features_from_pcm(&pcm)?;

        let mut cpu_caches = cpu.zero_caches()?;
        let mut metal_caches = metal.zero_caches()?;
        let cpu_scores = cpu.forward_frame_scores(&feats, &mut cpu_caches, 0)?;
        let metal_scores = metal.forward_frame_scores(&feats, &mut metal_caches, 0)?;
        assert_eq!(cpu_scores.len(), metal_scores.len());

        let max_diff = cpu_scores
            .iter()
            .zip(&metal_scores)
            .flat_map(|(cpu_frame, metal_frame)| cpu_frame.iter().zip(metal_frame))
            .map(|(cpu_score, metal_score)| (cpu_score - metal_score).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "CPU/Metal forward scores diverged: max_diff={max_diff}"
        );
        Ok(())
    }

    #[test]
    fn forward_frame_scores_cpu_fast_path_matches_candle_path() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let model_dir = [
            PathBuf::from("/workspace/data/models/asr").join(FSMN_VAD_REPO),
            workspace.join(".cache/fsmn-vad"),
            modelhub::modelscope_cache_dir().join(FSMN_VAD_REPO),
            modelscope_model_snapshot_dir(&modelhub::modelscope_cache_dir(), FSMN_VAD_REPO),
        ]
        .into_iter()
        .find(|path| path.join("model.pt").exists() && path.join("am.mvn").exists());
        let wav = workspace.join("assets/vad_example.wav");
        let Some(model_dir) = model_dir else {
            return Ok(());
        };
        if !wav.exists() {
            return Ok(());
        }

        let mut weights = FsmnVadWeights::load_on_device(&model_dir, &Device::Cpu)?;
        let waveform = load_pcm16_wav(&wav)?;
        let feats = weights
            .frontend
            .extract_features_from_normalized_f32(&waveform.samples)?;

        let mut fast_caches = weights.zero_caches()?;
        let fast_scores = weights.forward_frame_scores(&feats, &mut fast_caches, 1)?;
        let cpu = weights
            .cpu
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing FSMN CPU fast path weights"))?;
        let mut candle_caches = weights.zero_caches()?;
        let candle_scores = weights.forward_frame_scores(&feats, &mut candle_caches, 1)?;
        weights.cpu = Some(cpu);

        assert_eq!(fast_scores.len(), candle_scores.len());
        let mut max_diff = 0.0f32;
        for (fast_frame, candle_frame) in fast_scores.iter().zip(candle_scores.iter()) {
            assert_eq!(fast_frame.len(), candle_frame.len());
            for (fast, candle) in fast_frame.iter().zip(candle_frame.iter()) {
                max_diff = max_diff.max((fast - candle).abs());
            }
        }
        assert!(
            max_diff < 1e-4,
            "FSMN CPU fast path diverged from Candle path: max_diff={max_diff}"
        );
        Ok(())
    }

    #[test]
    fn detect_segments_cpu_fast_path_matches_candle_path() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let model_dir = [
            PathBuf::from("/workspace/data/models/asr").join(FSMN_VAD_REPO),
            workspace.join(".cache/fsmn-vad"),
            modelhub::modelscope_cache_dir().join(FSMN_VAD_REPO),
            modelscope_model_snapshot_dir(&modelhub::modelscope_cache_dir(), FSMN_VAD_REPO),
        ]
        .into_iter()
        .find(|path| path.join("model.pt").exists() && path.join("am.mvn").exists());
        let wav = workspace.join("assets/vad_example.wav");
        let Some(model_dir) = model_dir else {
            return Ok(());
        };
        if !wav.exists() {
            return Ok(());
        }

        let waveform = load_pcm16_wav(&wav)?;
        let options = VadOptions {
            num_threads: 1,
            ..VadOptions::default()
        };
        let fast = FsmnVadModel {
            weights: Arc::new(FsmnVadWeights::load_on_device(&model_dir, &Device::Cpu)?),
            model_dir: model_dir.clone(),
        };
        let mut candle_weights = FsmnVadWeights::load_on_device(&model_dir, &Device::Cpu)?;
        candle_weights.cpu = None;
        let candle = FsmnVadModel {
            weights: Arc::new(candle_weights),
            model_dir,
        };

        assert_eq!(
            fast.detect_with_timing(&waveform, &options)?.segments,
            candle.detect_with_timing(&waveform, &options)?.segments
        );
        Ok(())
    }

    #[cfg(feature = "cuda-vad")]
    #[test]
    #[ignore = "requires CUDA toolkit and raw_audios fixture"]
    fn detect_segments_match_cpu_and_cuda_on_raw_audios() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let model_dir = workspace.join(".cache/fsmn-vad");
        let audio_dir = workspace.join("raw_audios");
        if !model_dir.exists() || !audio_dir.is_dir() {
            return Ok(());
        }

        let mut cpu_segments = Vec::new();
        let mut cuda_segments = Vec::new();
        for device in ["cpu", "cuda"] {
            // SAFETY: this ignored test is single-threaded.
            unsafe { std::env::set_var("VASR_VAD_DEVICE", device) };
            let model = FsmnVadModel::from_pretrained(Some(model_dir.as_path()))?;
            let mut files = std::fs::read_dir(&audio_dir)?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            files.retain(|path| path.extension().and_then(|ext| ext.to_str()) == Some("wav"));
            files.sort();
            let mut ranges = Vec::new();
            for path in files {
                let waveform = load_pcm16_wav(&path)?;
                for segment in model.detect(&waveform, &VadOptions::default())? {
                    ranges.push((segment.range.start_ms, segment.range.end_ms));
                }
            }
            if device == "cpu" {
                cpu_segments = ranges;
            } else {
                cuda_segments = ranges;
            }
        }
        assert_eq!(cpu_segments, cuda_segments);
        Ok(())
    }

    #[test]
    #[ignore = "benchmark FSMN VAD on raw_audios; set VASR_VAD_DEVICE=cpu|metal|cuda"]
    fn fsmn_vad_benchmark_raw_audios() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let audio_dir = workspace.join("raw_audios");
        let model_dir = workspace.join(".cache/fsmn-vad");
        if !audio_dir.is_dir() || !model_dir.exists() {
            return Ok(());
        }
        let mut files = std::fs::read_dir(audio_dir)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        files.retain(|path| path.extension().and_then(|ext| ext.to_str()) == Some("wav"));
        files.sort();

        let model = FsmnVadModel::from_pretrained(Some(model_dir.as_path()))?;
        let options = VadOptions::default();
        let bench_start = Instant::now();
        let mut audio_seconds = 0.0f64;
        let mut segments = 0usize;
        for path in &files {
            let waveform = load_pcm16_wav(path)?;
            audio_seconds += waveform.duration_seconds();
            segments += model.detect(&waveform, &options)?.len();
        }
        let vad_seconds = bench_start.elapsed().as_secs_f64();
        println!(
            "fsmn_vad_bench files={} audio_seconds={audio_seconds:.3} segments={segments} vad_seconds={vad_seconds:.3} speedup={:.1}x rtf={:.4}",
            files.len(),
            audio_seconds / vad_seconds,
            vad_seconds / audio_seconds
        );
        Ok(())
    }

    #[derive(serde::Deserialize)]
    struct FasrVadFixtureFile {
        name: String,
        segments: Vec<[i32; 2]>,
    }

    #[derive(serde::Deserialize)]
    struct FasrVadFixture {
        files: Vec<FasrVadFixtureFile>,
    }

    fn assert_vad_matches_fasr_fixture() -> Result<()> {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let audio_dir = workspace.join("raw_audios");
        let model_dir = [
            workspace.join(".cache/fsmn-vad"),
            std::path::PathBuf::from(
                "/workspace/data/models/asr/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            ),
            modelhub::modelscope_cache_dir().join(FSMN_VAD_REPO),
            modelscope_model_snapshot_dir(&modelhub::modelscope_cache_dir(), FSMN_VAD_REPO),
        ]
        .into_iter()
        .find(|path| path.join("model.pt").exists() && path.join("am.mvn").exists());
        let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/fsmn_vad_fasr_segments.json");
        if !audio_dir.is_dir() || !fixture_path.exists() {
            return Ok(());
        }
        let Some(model_dir) = model_dir else {
            return Ok(());
        };

        let fixture: FasrVadFixture =
            serde_json::from_str(&std::fs::read_to_string(&fixture_path)?)?;
        let model = FsmnVadModel::from_pretrained(Some(model_dir.as_path()))?;
        let options = VadOptions::default();

        let fixture_file = std::env::var("VASR_FSMN_VAD_FIXTURE_FILE").ok();

        for expected_file in fixture.files {
            if fixture_file
                .as_deref()
                .is_some_and(|filter| filter != expected_file.name)
            {
                continue;
            }
            let wav = audio_dir.join(&expected_file.name);
            if !wav.exists() {
                continue;
            }
            let waveform = load_pcm16_wav(&wav)?;
            let segments = model.detect(&waveform, &options)?;
            let actual = segments
                .iter()
                .map(|segment| [segment.range.start_ms as i32, segment.range.end_ms as i32])
                .collect::<Vec<_>>();
            assert_eq!(
                actual.len(),
                expected_file.segments.len(),
                "segment count mismatch for {}: expected={:?} actual={:?}",
                expected_file.name,
                expected_file.segments,
                actual,
            );
            for (idx, ([actual_start, actual_end], [start, end])) in
                actual.iter().zip(expected_file.segments).enumerate()
            {
                assert_eq!(
                    [*actual_start, *actual_end],
                    [start, end],
                    "segment mismatch for {} at index {}",
                    expected_file.name,
                    idx,
                );
            }
            assert!(
                !actual.is_empty(),
                "fixture file produced no segments for {}",
                expected_file.name
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "uses local FSMN VAD, raw_audios fixture, and funasr_onnx reference JSON"]
    fn fsmn_vad_matches_fasr_funasr_onnx_segments_all_raw_audios() -> Result<()> {
        assert_vad_matches_fasr_fixture()
    }

    #[cfg(feature = "cuda-vad")]
    #[test]
    #[ignore = "requires CUDA toolkit, raw_audios fixture, and funasr_onnx reference JSON"]
    fn fsmn_vad_cuda_matches_fasr_funasr_onnx_segments_all_raw_audios() -> Result<()> {
        // SAFETY: this ignored test is single-threaded.
        unsafe { std::env::set_var("VASR_VAD_DEVICE", "cuda") };
        assert_vad_matches_fasr_fixture()
    }

    #[test]
    #[ignore = "uses local FSMN VAD and raw_audios fixture"]
    fn offline_fast_path_matches_streaming_session_segments() -> Result<()> {
        let wav = std::path::Path::new("raw_audios/audio (13).wav");
        let model_dir = std::path::Path::new(".cache/fsmn-vad");
        if !wav.exists() || !model_dir.exists() {
            return Ok(());
        }
        let waveform = load_pcm16_wav(wav)?;
        let model = FsmnVadModel::from_pretrained(Some(model_dir))?;
        let options = VadOptions::default();

        let fast = model.detect_with_timing(&waveform, &options)?.segments;
        let mut session = model.new_session(&options)?;
        session.push_samples_f32(&waveform.samples)?;
        let streaming = session.finish_segments()?;

        assert_eq!(fast, streaming);
        Ok(())
    }

    #[test]
    fn split_full_chunks_keeps_unprocessed_tail() {
        let samples = (0..(FSMN_VAD_CHUNK_SIZE * 2 + 17))
            .map(|sample| sample as i16)
            .collect::<Vec<_>>();

        let (full_chunks, tail) = split_full_chunks(&samples, FSMN_VAD_CHUNK_SIZE);
        let chunks = full_chunks.as_chunks::<FSMN_VAD_CHUNK_SIZE>().0;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0][0], 0);
        assert_eq!(chunks[1][0], FSMN_VAD_CHUNK_SIZE as i16);
        assert_eq!(tail.len(), 17);
        assert_eq!(tail[0], (FSMN_VAD_CHUNK_SIZE * 2) as i16);
    }

    #[test]
    fn cmvn_parser_reads_addshift_and_rescale_vectors() -> Result<()> {
        let path = std::env::temp_dir().join(format!("vasr-test-am-{}.mvn", std::process::id()));
        std::fs::write(
            &path,
            "<Nnet>\n\
             <Splice> 2 2\n\
             [ 0 ]\n\
             <AddShift> 2 2\n\
             <LearnRateCoef> 0 [ -1.5 -2.5 ]\n\
             <Rescale> 2 2\n\
             <LearnRateCoef> 0 [ 0.25 0.5 ]\n\
             </Nnet>\n",
        )?;

        let (means, vars) = WavFrontend::load_cmvn(&path)?;

        assert_eq!(means.to_vec(), vec![-1.5, -2.5]);
        assert_eq!(vars.to_vec(), vec![0.25, 0.5]);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    /// LFR 的朴素参考实现（纯索引，不依赖 Tensor），用于校验 `WavFrontend::apply_lfr`。
    ///
    /// 语义与 Kaldi `apply-lfr` 一致：左侧复制第 0 帧填充 `(lfr_m - 1) / 2` 行，
    /// 按步长 `lfr_n` 取 `lfr_m` 行展平拼接，末尾不足时重复最后一帧补齐。
    fn reference_lfr(
        fbank: &[f32],
        frames: usize,
        n_mels: usize,
        lfr_m: usize,
        lfr_n: usize,
    ) -> Vec<f32> {
        if frames == 0 {
            return Vec::new();
        }
        let left = (lfr_m - 1) / 2;
        let padded_frames = frames + left;
        let mut padded = Vec::with_capacity(padded_frames * n_mels);
        for _ in 0..left {
            padded.extend_from_slice(&fbank[..n_mels]);
        }
        padded.extend_from_slice(fbank);

        let mut out = Vec::with_capacity(frames.div_ceil(lfr_n) * n_mels * lfr_m);
        for i in 0..frames.div_ceil(lfr_n) {
            let start = i * lfr_n;
            let end = (start + lfr_m).min(padded_frames);
            out.extend_from_slice(&padded[start * n_mels..end * n_mels]);
            for _ in end..start + lfr_m {
                out.extend_from_slice(&padded[(padded_frames - 1) * n_mels..]);
            }
        }
        out
    }

    /// 覆盖 LFR 的三种分支：`lfr_n > 1` 的正常窗口、尾部补帧、以及空输入。
    #[test]
    fn lfr_matches_naive_reference() -> Result<()> {
        let n_mels = 4usize;
        for (frames, lfr_m, lfr_n) in [(11usize, 5usize, 1usize), (4, 5, 2), (1, 5, 2), (0, 5, 2)] {
            let fbank: Vec<f32> = (0..frames * n_mels).map(|i| i as f32 * 0.25).collect();
            let frontend = WavFrontend::new(WavFrontendConfig {
                n_mels,
                lfr_m,
                lfr_n,
                cmvn_file: None,
                ..Default::default()
            })?;

            let input = Tensor::from_vec(fbank.clone(), (frames, n_mels), &Device::Cpu)?;
            let lfr = frontend.apply_lfr(&input)?;

            let expected = reference_lfr(&fbank, frames, n_mels, lfr_m, lfr_n);
            assert_eq!(lfr.dims(), [frames.div_ceil(lfr_n), n_mels * lfr_m]);
            assert_eq!(
                lfr.flatten_all()?.to_vec1::<f32>()?,
                expected,
                "LFR mismatch for frames={frames} lfr_m={lfr_m} lfr_n={lfr_n}"
            );
        }
        Ok(())
    }

    /// 校验 CMVN 的广播加乘，并确认维度不匹配时原样返回输入。
    #[test]
    fn cmvn_matches_naive_reference() -> Result<()> {
        let n_mels = 3usize;
        let lfr_m = 3usize;
        let frames = 5usize;
        let dim = n_mels * lfr_m;
        let means: Vec<f32> = (0..dim).map(|i| i as f32 - 2.0).collect();
        let vars: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.125).collect();

        let mut frontend = WavFrontend::new(WavFrontendConfig {
            n_mels,
            lfr_m,
            lfr_n: 1,
            cmvn_file: None,
            ..Default::default()
        })?;
        let feats: Vec<f32> = (0..frames * dim).map(|i| i as f32 * 0.5).collect();
        let input = Tensor::from_vec(feats.clone(), (frames, dim), &Device::Cpu)?;

        // 未配置 CMVN 时应原样通过。
        assert_eq!(
            frontend
                .apply_cmvn(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            feats
        );

        frontend.cmvn_means = Some(means.clone());
        frontend.cmvn_vars = Some(vars.clone());
        let expected: Vec<f32> = feats
            .iter()
            .enumerate()
            .map(|(i, value)| (value + means[i % dim]) * vars[i % dim])
            .collect();
        assert_eq!(
            frontend
                .apply_cmvn(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            expected
        );

        // 维度不匹配时同样原样返回，与旧实现保持一致。
        frontend.cmvn_means = Some(vec![0.0; dim + 1]);
        assert_eq!(
            frontend
                .apply_cmvn(&input)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            feats
        );
        Ok(())
    }

    fn load_pcm16_wav(path: &std::path::Path) -> Result<Waveform> {
        let data = std::fs::read(path)?;
        if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
            bail!("not a RIFF/WAVE file: {}", path.display());
        }
        let mut offset = 12usize;
        let mut sample_rate = None;
        let mut channels = None;
        let mut data_range = None;
        while offset + 8 <= data.len() {
            let id = &data[offset..offset + 4];
            let size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?) as usize;
            offset += 8;
            if offset + size > data.len() {
                break;
            }
            match id {
                b"fmt " if size >= 16 => {
                    let format = u16::from_le_bytes(data[offset..offset + 2].try_into()?);
                    channels = Some(u16::from_le_bytes(data[offset + 2..offset + 4].try_into()?));
                    sample_rate =
                        Some(u32::from_le_bytes(data[offset + 4..offset + 8].try_into()?));
                    let bits = u16::from_le_bytes(data[offset + 14..offset + 16].try_into()?);
                    if format != 1 || bits != 16 {
                        bail!("expected PCM16 WAV, got format={format} bits={bits}");
                    }
                }
                b"data" => data_range = Some(offset..offset + size),
                _ => {}
            }
            offset += size + (size % 2);
        }
        let sample_rate = sample_rate.ok_or_else(|| anyhow::anyhow!("WAV missing fmt chunk"))?;
        let channels = channels.ok_or_else(|| anyhow::anyhow!("WAV missing channel count"))?;
        let data_range = data_range.ok_or_else(|| anyhow::anyhow!("WAV missing data chunk"))?;
        let mut samples = Vec::new();
        for frame in data[data_range].chunks_exact(usize::from(channels) * 2) {
            let mut sum = 0f32;
            for ch in 0..usize::from(channels) {
                let start = ch * 2;
                sum += f32::from(i16::from_le_bytes(frame[start..start + 2].try_into()?));
            }
            samples.push(sum / f32::from(channels) / f32::from(i16::MAX));
        }
        Ok(Waveform::new(samples, sample_rate))
    }
}
