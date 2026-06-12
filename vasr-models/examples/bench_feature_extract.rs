use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use vasr_models::processor::feature_extractor;

#[derive(Debug, Clone, Copy)]
struct Config {
    feature_size: usize,
    sampling_rate_hz: u32,
    hop_length: usize,
    n_fft: usize,
    max_frequency_hz: f32,
    mel_floor: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            feature_size: 128,
            sampling_rate_hz: 16_000,
            hop_length: 160,
            n_fft: 400,
            max_frequency_hz: 8_000.0,
            mel_floor: 1e-10,
        }
    }
}

#[derive(Debug, Clone)]
struct Features {
    input_features: Vec<Vec<f32>>,
    feature_attention_mask: Vec<u8>,
}

struct CpuF64DenseExtractor {
    cfg: Config,
    window: Vec<f64>,
    mel_filters_t: Vec<Vec<f64>>,
    fft: Arc<dyn rustfft::Fft<f64>>,
}

struct CpuF32Extractor {
    cfg: Config,
    window: Vec<f32>,
    mel_filters_t: Vec<Vec<f32>>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

struct CpuF32SparseExtractor {
    cfg: Config,
    window: Vec<f32>,
    mel_filters_sparse: Vec<Vec<(usize, f32)>>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl CpuF64DenseExtractor {
    fn new() -> Self {
        let cfg = Config::default();
        let window = hann_window_periodic_f64(cfg.n_fft);
        let mel_filters_t = mel_filter_bank_slaney_t_f64(
            1 + cfg.n_fft / 2,
            cfg.feature_size,
            0.0,
            f64::from(cfg.max_frequency_hz),
            cfg.sampling_rate_hz,
        );
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        Self {
            cfg,
            window,
            mel_filters_t,
            fft,
        }
    }

    fn extract_with_padded_waveform_len(&self, wav: &[f32], padded_len: usize) -> Result<Features> {
        if padded_len < wav.len() {
            bail!(
                "padded_len {padded_len} is shorter than wav len {}",
                wav.len()
            );
        }
        let frames_total = padded_len / self.cfg.hop_length;
        let frames = if wav.is_empty() {
            0
        } else {
            wav.len().div_ceil(self.cfg.hop_length).min(frames_total)
        };
        if frames == 0 {
            return Ok(Features {
                input_features: vec![vec![]; self.cfg.feature_size],
                feature_attention_mask: vec![],
            });
        }
        let input_features = self.extract_log_mel(padded_len, frames, |src| {
            wav.get(src).copied().map(f64::from).unwrap_or(0.0)
        })?;
        Ok(Features {
            input_features,
            feature_attention_mask: vec![1; frames],
        })
    }

    fn extract_log_mel<F>(
        &self,
        source_len: usize,
        frames: usize,
        sample_at: F,
    ) -> Result<Vec<Vec<f32>>>
    where
        F: Fn(usize) -> f64,
    {
        if source_len == 0 {
            return Ok(vec![vec![]; self.cfg.feature_size]);
        }
        let n_freq = 1 + self.cfg.n_fft / 2;
        let mut buf = vec![Complex::new(0.0f64, 0.0f64); self.cfg.n_fft];
        let mut mags = vec![0.0f64; n_freq];
        let mut log_spec: Vec<Vec<f32>> = (0..self.cfg.feature_size)
            .map(|_| Vec::with_capacity(frames))
            .collect();
        let mut max_log = f32::NEG_INFINITY;
        for frame_idx in 0..frames {
            let start = frame_idx * self.cfg.hop_length;
            for i in 0..self.cfg.n_fft {
                let idx = start as isize + i as isize - (self.cfg.n_fft / 2) as isize;
                let src = reflect_index(idx, source_len);
                buf[i] = Complex::new(sample_at(src) * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for k in 0..n_freq {
                let c = buf[k];
                mags[k] = c.re.mul_add(c.re, c.im * c.im);
            }
            for (mel_row, filter_row) in log_spec.iter_mut().zip(self.mel_filters_t.iter()) {
                let mut sum = 0.0f64;
                for (w, mag) in filter_row.iter().zip(mags.iter()) {
                    sum = (*w).mul_add(*mag, sum);
                }
                if sum < f64::from(self.cfg.mel_floor) {
                    sum = f64::from(self.cfg.mel_floor);
                }
                let log10 = sum.log10() as f32;
                max_log = max_log.max(log10);
                mel_row.push(log10);
            }
        }
        compress_log_spec(&mut log_spec, max_log);
        Ok(log_spec)
    }
}

impl CpuF32Extractor {
    fn new() -> Self {
        let cfg = Config::default();
        let window = hann_window_periodic_f32(cfg.n_fft);
        let mel_filters_t = mel_filter_bank_slaney_t_f32(
            1 + cfg.n_fft / 2,
            cfg.feature_size,
            0.0,
            cfg.max_frequency_hz,
            cfg.sampling_rate_hz,
        );
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        Self {
            cfg,
            window,
            mel_filters_t,
            fft,
        }
    }

    fn extract_with_padded_waveform_len(&self, wav: &[f32], padded_len: usize) -> Result<Features> {
        if padded_len < wav.len() {
            bail!(
                "padded_len {padded_len} is shorter than wav len {}",
                wav.len()
            );
        }
        let frames_total = padded_len / self.cfg.hop_length;
        let frames = if wav.is_empty() {
            0
        } else {
            wav.len().div_ceil(self.cfg.hop_length).min(frames_total)
        };
        if frames == 0 {
            return Ok(Features {
                input_features: vec![vec![]; self.cfg.feature_size],
                feature_attention_mask: vec![],
            });
        }
        let input_features = self.extract_log_mel(padded_len, frames, |src| {
            wav.get(src).copied().unwrap_or(0.0)
        })?;
        Ok(Features {
            input_features,
            feature_attention_mask: vec![1; frames],
        })
    }

    fn extract_log_mel<F>(
        &self,
        source_len: usize,
        frames: usize,
        sample_at: F,
    ) -> Result<Vec<Vec<f32>>>
    where
        F: Fn(usize) -> f32,
    {
        if source_len == 0 {
            return Ok(vec![vec![]; self.cfg.feature_size]);
        }
        let n_freq = 1 + self.cfg.n_fft / 2;
        let mut buf = vec![Complex::new(0.0f32, 0.0f32); self.cfg.n_fft];
        let mut mags = vec![0.0f32; n_freq];
        let mut log_spec: Vec<Vec<f32>> = (0..self.cfg.feature_size)
            .map(|_| Vec::with_capacity(frames))
            .collect();
        let mut max_log = f32::NEG_INFINITY;
        for frame_idx in 0..frames {
            let start = frame_idx * self.cfg.hop_length;
            for i in 0..self.cfg.n_fft {
                let idx = start as isize + i as isize - (self.cfg.n_fft / 2) as isize;
                let src = reflect_index(idx, source_len);
                buf[i] = Complex::new(sample_at(src) * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for k in 0..n_freq {
                let c = buf[k];
                mags[k] = c.re.mul_add(c.re, c.im * c.im);
            }
            for (mel_row, filter_row) in log_spec.iter_mut().zip(self.mel_filters_t.iter()) {
                let mut sum = 0.0f32;
                for (w, mag) in filter_row.iter().zip(mags.iter()) {
                    sum = (*w).mul_add(*mag, sum);
                }
                if sum < self.cfg.mel_floor {
                    sum = self.cfg.mel_floor;
                }
                let log10 = sum.log10();
                max_log = max_log.max(log10);
                mel_row.push(log10);
            }
        }
        compress_log_spec(&mut log_spec, max_log);
        Ok(log_spec)
    }
}

impl CpuF32SparseExtractor {
    fn new() -> Self {
        let dense = CpuF32Extractor::new();
        let mel_filters_sparse = dense
            .mel_filters_t
            .iter()
            .map(|row| {
                row.iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, w)| *w != 0.0)
                    .collect::<Vec<_>>()
            })
            .collect();
        Self {
            cfg: dense.cfg,
            window: dense.window,
            mel_filters_sparse,
            fft: dense.fft,
        }
    }

    fn extract_with_padded_waveform_len(&self, wav: &[f32], padded_len: usize) -> Result<Features> {
        if padded_len < wav.len() {
            bail!(
                "padded_len {padded_len} is shorter than wav len {}",
                wav.len()
            );
        }
        let frames_total = padded_len / self.cfg.hop_length;
        let frames = if wav.is_empty() {
            0
        } else {
            wav.len().div_ceil(self.cfg.hop_length).min(frames_total)
        };
        if frames == 0 {
            return Ok(Features {
                input_features: vec![vec![]; self.cfg.feature_size],
                feature_attention_mask: vec![],
            });
        }
        let input_features = self.extract_log_mel(padded_len, frames, |src| {
            wav.get(src).copied().unwrap_or(0.0)
        })?;
        Ok(Features {
            input_features,
            feature_attention_mask: vec![1; frames],
        })
    }

    fn extract_log_mel<F>(
        &self,
        source_len: usize,
        frames: usize,
        sample_at: F,
    ) -> Result<Vec<Vec<f32>>>
    where
        F: Fn(usize) -> f32,
    {
        if source_len == 0 {
            return Ok(vec![vec![]; self.cfg.feature_size]);
        }
        let n_freq = 1 + self.cfg.n_fft / 2;
        let mut buf = vec![Complex::new(0.0f32, 0.0f32); self.cfg.n_fft];
        let mut mags = vec![0.0f32; n_freq];
        let mut log_spec: Vec<Vec<f32>> = (0..self.cfg.feature_size)
            .map(|_| Vec::with_capacity(frames))
            .collect();
        let mut max_log = f32::NEG_INFINITY;
        for frame_idx in 0..frames {
            let start = frame_idx * self.cfg.hop_length;
            for i in 0..self.cfg.n_fft {
                let idx = start as isize + i as isize - (self.cfg.n_fft / 2) as isize;
                let src = reflect_index(idx, source_len);
                buf[i] = Complex::new(sample_at(src) * self.window[i], 0.0);
            }
            self.fft.process(&mut buf);
            for k in 0..n_freq {
                let c = buf[k];
                mags[k] = c.re.mul_add(c.re, c.im * c.im);
            }
            for (mel_row, filter_row) in log_spec.iter_mut().zip(self.mel_filters_sparse.iter()) {
                let mut sum = 0.0f32;
                for &(idx, w) in filter_row {
                    sum = w.mul_add(mags[idx], sum);
                }
                if sum < self.cfg.mel_floor {
                    sum = self.cfg.mel_floor;
                }
                let log10 = sum.log10();
                max_log = max_log.max(log10);
                mel_row.push(log10);
            }
        }
        compress_log_spec(&mut log_spec, max_log);
        Ok(log_spec)
    }
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let wavs = load_wavs(&args)?;
    if wavs.is_empty() {
        bail!("no waveforms loaded");
    }
    let max_samples = wavs.iter().map(Vec::len).max().unwrap_or(0);
    let audio_sec: f64 = wavs.iter().map(|w| w.len() as f64 / 16_000.0).sum();
    let total_frames: usize = wavs.iter().map(|w| w.len().div_ceil(160)).sum();
    println!(
        "input: files={} audio={:.2}s frames={} max_samples={} rayon_threads={}",
        wavs.len(),
        audio_sec,
        total_frames,
        max_samples,
        rayon::current_num_threads()
    );

    let baseline = bench("baseline_f64", args.iters, audio_sec, || {
        wavs.par_iter()
            .map(|wav| {
                feature_extractor::extract_features_with_padded_waveform_len(wav, max_samples)
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let legacy_dense = CpuF64DenseExtractor::new();
    let legacy = bench("legacy_f64_dense", args.iters, audio_sec, || {
        wavs.par_iter()
            .map(|wav| legacy_dense.extract_with_padded_waveform_len(wav, max_samples))
            .collect::<Result<Vec<_>>>()
    })?;
    compare_feature_sets(
        "legacy_f64_dense_vs_current_sparse",
        &baseline.value,
        &legacy.value,
    )?;

    let dense = CpuF32Extractor::new();
    let cpu_dense = bench("cpu_f32_reuse_dense", args.iters, audio_sec, || {
        wavs.par_iter()
            .map(|wav| dense.extract_with_padded_waveform_len(wav, max_samples))
            .collect::<Result<Vec<_>>>()
    })?;
    compare_feature_sets("cpu_f32_reuse_dense", &baseline.value, &cpu_dense.value)?;

    let sparse = CpuF32SparseExtractor::new();
    let cpu_sparse = bench("cpu_f32_reuse_sparse", args.iters, audio_sec, || {
        wavs.par_iter()
            .map(|wav| sparse.extract_with_padded_waveform_len(wav, max_samples))
            .collect::<Result<Vec<_>>>()
    })?;
    compare_feature_sets("cpu_f32_reuse_sparse", &baseline.value, &cpu_sparse.value)?;

    if args.gpu {
        match Device::new_cuda_with_stream(0) {
            Ok(device) => bench_gpu_mel_only(&cpu_sparse.value, &device, args.iters, audio_sec)?,
            Err(err) => eprintln!("gpu_mel_only: skipped, failed to create CUDA device: {err:#}"),
        }
    }

    Ok(())
}

struct BenchResult<T> {
    value: T,
}

fn bench<T, F>(name: &str, iters: usize, audio_sec: f64, mut f: F) -> Result<BenchResult<T>>
where
    F: FnMut() -> Result<T>,
{
    let iters = iters.max(1);
    let mut value = None;
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let start = Instant::now();
        let out = f()?;
        total += start.elapsed();
        value = Some(out);
    }
    let avg = total / iters as u32;
    println!(
        "{name}: {:.1}ms avg over {} iter(s), speed={:.1}x",
        avg.as_secs_f64() * 1000.0,
        iters,
        audio_sec / avg.as_secs_f64()
    );
    Ok(BenchResult {
        value: value.expect("at least one iteration"),
    })
}

fn bench_gpu_mel_only(
    features: &[Features],
    device: &Device,
    iters: usize,
    audio_sec: f64,
) -> Result<()> {
    let max_frames = features
        .iter()
        .filter_map(|f| f.input_features.first().map(Vec::len))
        .max()
        .unwrap_or(0);
    let mut flat = Vec::with_capacity(features.len() * 128 * max_frames);
    for f in features {
        for mel in 0..128 {
            let row = f
                .input_features
                .get(mel)
                .ok_or_else(|| anyhow::anyhow!("missing mel row {mel}"))?;
            flat.extend_from_slice(row);
            if row.len() < max_frames {
                flat.extend(std::iter::repeat_n(0.0f32, max_frames - row.len()));
            }
        }
    }
    let iters = iters.max(1);
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let start = Instant::now();
        let t = Tensor::from_vec(flat.clone(), (features.len(), 128, max_frames), device)?;
        let y = ((t.log()? / std::f64::consts::LN_10)? + 4.0)? / 4.0;
        device.synchronize()?;
        drop(y);
        total += start.elapsed();
    }
    let avg = total / iters as u32;
    println!(
        "gpu_elementwise_transfer_probe: {:.1}ms avg over {} iter(s), speed={:.1}x (includes H2D transfer; not full fbank)",
        avg.as_secs_f64() * 1000.0,
        iters,
        audio_sec / avg.as_secs_f64()
    );
    Ok(())
}

#[derive(Debug)]
struct Args {
    input: Option<PathBuf>,
    seconds: f32,
    count: usize,
    iters: usize,
    gpu: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut input = None;
        let mut seconds = 10.0f32;
        let mut count = 64usize;
        let mut iters = 3usize;
        let mut gpu = false;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => {
                    input = Some(PathBuf::from(args.next().context("--input needs a path")?))
                }
                "--seconds" => seconds = args.next().context("--seconds needs a value")?.parse()?,
                "--count" => count = args.next().context("--count needs a value")?.parse()?,
                "--iters" => iters = args.next().context("--iters needs a value")?.parse()?,
                "--gpu" => gpu = true,
                "--help" | "-h" => {
                    println!(
                        "usage: bench_feature_extract [--input DIR] [--seconds N] [--count N] [--iters N] [--gpu]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown arg {other:?}"),
            }
        }
        Ok(Self {
            input,
            seconds,
            count,
            iters,
            gpu,
        })
    }
}

fn load_wavs(args: &Args) -> Result<Vec<Vec<f32>>> {
    if let Some(input) = args.input.as_deref() {
        let mut out = Vec::new();
        for path in collect_files(input)? {
            if let Ok(wav) = load_wav_16k_mono(&path) {
                out.push(wav);
            }
        }
        if !out.is_empty() {
            return Ok(out);
        }
        eprintln!(
            "no supported wav files loaded from {}, using synthetic input",
            input.display()
        );
    }
    let samples = (args.seconds * 16_000.0).round().max(1.0) as usize;
    Ok((0..args.count)
        .map(|i| synthetic_wav(samples + (i % 13) * 1600, i))
        .collect())
}

fn collect_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            out.extend(collect_files(&path)?);
        } else if path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("wav"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn load_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file: {}", path.display());
    }
    let mut offset = 12usize;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits = 0u16;
    let mut format = 0u16;
    let mut data = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let len = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        if offset + len > bytes.len() {
            break;
        }
        match id {
            b"fmt " => {
                if len >= 16 {
                    format = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                    channels =
                        u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap());
                    sample_rate =
                        u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
                    bits = u16::from_le_bytes(bytes[offset + 14..offset + 16].try_into().unwrap());
                }
            }
            b"data" => data = Some((offset, len)),
            _ => {}
        }
        offset += len + (len % 2);
    }
    if format != 1 || bits != 16 || sample_rate != 16_000 || channels == 0 {
        bail!(
            "unsupported wav {}: format={format} bits={bits} rate={sample_rate} channels={channels}",
            path.display()
        );
    }
    let (start, len) = data.context("missing data chunk")?;
    let samples = len / 2 / channels as usize;
    let mut out = Vec::with_capacity(samples);
    for i in 0..samples {
        let mut acc = 0.0f32;
        for ch in 0..channels as usize {
            let pos = start + (i * channels as usize + ch) * 2;
            let s = i16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
            acc += f32::from(s) / 32768.0;
        }
        out.push(acc / channels as f32);
    }
    Ok(out)
}

fn synthetic_wav(samples: usize, seed: usize) -> Vec<f32> {
    let f1 = 220.0 + (seed % 17) as f32 * 13.0;
    let f2 = 780.0 + (seed % 11) as f32 * 29.0;
    (0..samples)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            (2.0 * std::f32::consts::PI * f1 * t).sin() * 0.18
                + (2.0 * std::f32::consts::PI * f2 * t).cos() * 0.05
        })
        .collect()
}

fn compare_feature_sets(
    name: &str,
    baseline: &[feature_extractor::Features],
    got: &[Features],
) -> Result<()> {
    if baseline.len() != got.len() {
        bail!("{name}: len mismatch {} != {}", baseline.len(), got.len());
    }
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut n = 0usize;
    for (b, g) in baseline.iter().zip(got.iter()) {
        if b.feature_attention_mask != g.feature_attention_mask {
            bail!("{name}: feature_attention_mask mismatch");
        }
        for (br, gr) in b.input_features.iter().zip(g.input_features.iter()) {
            if br.len() != gr.len() {
                bail!("{name}: row len mismatch {} != {}", br.len(), gr.len());
            }
            for (&a, &bb) in br.iter().zip(gr.iter()) {
                let d = (a - bb).abs();
                max_diff = max_diff.max(d);
                sum_diff += f64::from(d);
                n += 1;
            }
        }
    }
    let mean = if n == 0 { 0.0 } else { sum_diff / n as f64 };
    println!("{name}: max_diff={max_diff:.6e} mean_diff={mean:.6e}");
    Ok(())
}

fn compress_log_spec(log_spec: &mut [Vec<f32>], max_log: f32) {
    let floor = max_log - 8.0;
    for mel in log_spec {
        for v in mel {
            if *v < floor {
                *v = floor;
            }
            *v = (*v + 4.0) / 4.0;
        }
    }
}

fn hann_window_periodic_f32(n: usize) -> Vec<f32> {
    let n_f32 = n as f32;
    let two_pi = std::f32::consts::PI * 2.0;
    (0..n)
        .map(|i| {
            let x = two_pi * (i as f32) / n_f32;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn reflect_index(i: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let mut j = i.rem_euclid(period);
    let len_i = len as isize;
    if j >= len_i {
        j = period - j;
    }
    j as usize
}

fn hertz_to_mel_slaney(freq_hz: f32) -> f32 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / 6.4f32.ln();
    let mut mels = 3.0 * freq_hz / 200.0;
    if freq_hz >= min_log_hertz {
        mels = min_log_mel + (freq_hz / min_log_hertz).ln() * logstep;
    }
    mels
}

fn mel_to_hertz_slaney(mels: f32) -> f32 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 6.4f32.ln() / 27.0;
    let mut freq = 200.0 * mels / 3.0;
    if mels >= min_log_mel {
        freq = min_log_hertz * (logstep * (mels - min_log_mel)).exp();
    }
    freq
}

fn linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f32 - 1.0);
    (0..n).map(|i| start + step * (i as f32)).collect()
}

fn mel_filter_bank_slaney_t_f32(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    min_frequency_hz: f32,
    max_frequency_hz: f32,
    sampling_rate_hz: u32,
) -> Vec<Vec<f32>> {
    let mel_min = hertz_to_mel_slaney(min_frequency_hz);
    let mel_max = hertz_to_mel_slaney(max_frequency_hz);
    let mel_freqs = linspace(mel_min, mel_max, num_mel_filters + 2);
    let filter_freqs: Vec<f32> = mel_freqs.into_iter().map(mel_to_hertz_slaney).collect();
    let nyquist_hz = (sampling_rate_hz / 2) as f32;
    let fft_freqs = linspace(0.0, nyquist_hz, num_frequency_bins);
    let mut filters_t = vec![vec![0.0; num_frequency_bins]; num_mel_filters];
    for mel_idx in 0..num_mel_filters {
        let left = filter_freqs[mel_idx];
        let center = filter_freqs[mel_idx + 1];
        let right = filter_freqs[mel_idx + 2];
        let denom_left = (center - left).max(f32::MIN_POSITIVE);
        let denom_right = (right - center).max(f32::MIN_POSITIVE);
        let enorm = 2.0 / (right - left).max(f32::MIN_POSITIVE);
        for (k, f) in fft_freqs.iter().copied().enumerate() {
            let down = (f - left) / denom_left;
            let up = (right - f) / denom_right;
            let w = down.min(up).max(0.0);
            filters_t[mel_idx][k] = w * enorm;
        }
    }
    filters_t
}

fn hann_window_periodic_f64(n: usize) -> Vec<f64> {
    let n_f64 = n as f64;
    let two_pi = std::f64::consts::PI * 2.0;
    (0..n)
        .map(|i| {
            let x = two_pi * (i as f64) / n_f64;
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn hertz_to_mel_slaney_f64(freq_hz: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / 6.4f64.ln();
    let mut mels = 3.0 * freq_hz / 200.0;
    if freq_hz >= min_log_hertz {
        mels = min_log_mel + (freq_hz / min_log_hertz).ln() * logstep;
    }
    mels
}

fn mel_to_hertz_slaney_f64(mels: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 6.4f64.ln() / 27.0;
    let mut freq = 200.0 * mels / 3.0;
    if mels >= min_log_mel {
        freq = min_log_hertz * (logstep * (mels - min_log_mel)).exp();
    }
    freq
}

fn linspace_f64(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| start + step * (i as f64)).collect()
}

fn mel_filter_bank_slaney_t_f64(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    min_frequency_hz: f64,
    max_frequency_hz: f64,
    sampling_rate_hz: u32,
) -> Vec<Vec<f64>> {
    let mel_min = hertz_to_mel_slaney_f64(min_frequency_hz);
    let mel_max = hertz_to_mel_slaney_f64(max_frequency_hz);
    let mel_freqs = linspace_f64(mel_min, mel_max, num_mel_filters + 2);
    let filter_freqs: Vec<f64> = mel_freqs.into_iter().map(mel_to_hertz_slaney_f64).collect();
    let nyquist_hz = (sampling_rate_hz / 2) as f64;
    let fft_freqs = linspace_f64(0.0, nyquist_hz, num_frequency_bins);
    let mut filters_t = vec![vec![0.0; num_frequency_bins]; num_mel_filters];
    for mel_idx in 0..num_mel_filters {
        let left = filter_freqs[mel_idx];
        let center = filter_freqs[mel_idx + 1];
        let right = filter_freqs[mel_idx + 2];
        let denom_left = (center - left).max(f64::MIN_POSITIVE);
        let denom_right = (right - center).max(f64::MIN_POSITIVE);
        let enorm = 2.0 / (right - left).max(f64::MIN_POSITIVE);
        for (k, f) in fft_freqs.iter().copied().enumerate() {
            let down = (f - left) / denom_left;
            let up = (right - f) / denom_right;
            let w = down.min(up).max(0.0);
            filters_t[mel_idx][k] = w * enorm;
        }
    }
    filters_t
}
