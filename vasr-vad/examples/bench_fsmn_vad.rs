use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use asr_data::Waveform;
use vasr_vad::{FsmnVadForwardTiming, FsmnVadModel, VadOptions};

const FSMN_VAD_REPO: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";

#[derive(Debug)]
struct Args {
    audio_path: PathBuf,
    model_path: Option<PathBuf>,
    device: Option<String>,
    warmup: usize,
    repeat: usize,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if let Some(device) = &args.device {
        // SAFETY: this example sets the process environment before model load
        // and before starting any worker threads.
        unsafe { std::env::set_var("VASR_VAD_DEVICE", device) };
    }

    let waveform = load_pcm16_wav(&args.audio_path)
        .with_context(|| format!("failed to load WAV {}", args.audio_path.display()))?;
    validate_waveform(&waveform, &args.audio_path)?;

    let model_path = args.model_path.or_else(default_model_path);
    let load_start = Instant::now();
    let model = FsmnVadModel::from_pretrained(model_path.as_deref())?;
    let load_elapsed = load_start.elapsed();
    let options = VadOptions::default();

    for _ in 0..args.warmup {
        let _ = model.detect(&waveform, &options)?;
    }

    let mut runs = Vec::with_capacity(args.repeat);
    let mut segments = 0usize;
    for _ in 0..args.repeat {
        let start = Instant::now();
        let detection = model.detect_with_timing(&waveform, &options)?;
        let elapsed = start.elapsed();
        segments = detection.segments.len();
        runs.push(RunTiming {
            elapsed,
            pcm_seconds: detection.timing.pcm_seconds,
            frontend_seconds: detection.timing.frontend_seconds,
            forward_seconds: detection.timing.forward_seconds,
            segmenter_seconds: detection.timing.segmenter_seconds,
            forward_ops: detection.timing.forward_ops,
        });
    }

    let audio_seconds = waveform.duration_seconds();
    let avg = average_timing(&runs);
    let device = std::env::var("VASR_VAD_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    println!(
        "fsmn_vad_bench device={} audio={} model={} sample_rate={} duration_ms={} segments={} warmup={} repeat={} load_ms={:.3}",
        device,
        args.audio_path.display(),
        model.model_dir().display(),
        waveform.sample_rate,
        waveform.duration_ms() as u64,
        segments,
        args.warmup,
        args.repeat,
        load_elapsed.as_secs_f64() * 1000.0,
    );
    println!(
        "avg_ms={:.3} min_ms={:.3} max_ms={:.3} speedup={:.2}x rtf={:.6}",
        avg.elapsed.as_secs_f64() * 1000.0,
        runs.iter()
            .map(|run| run.elapsed)
            .min()
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0,
        runs.iter()
            .map(|run| run.elapsed)
            .max()
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0,
        audio_seconds / avg.elapsed.as_secs_f64(),
        avg.elapsed.as_secs_f64() / audio_seconds,
    );
    println!(
        "avg_components_ms pcm={:.3} frontend={:.3} forward={:.3} segmenter={:.3}",
        avg.pcm_seconds * 1000.0,
        avg.frontend_seconds * 1000.0,
        avg.forward_seconds * 1000.0,
        avg.segmenter_seconds * 1000.0,
    );
    println!(
        "avg_forward_ops_ms in_linear1={:.3} in_linear2={:.3} out_linear1={:.3} out_linear2={:.3} softmax={:.3}",
        avg.forward_ops.in_linear1_seconds * 1000.0,
        avg.forward_ops.in_linear2_seconds * 1000.0,
        avg.forward_ops.out_linear1_seconds * 1000.0,
        avg.forward_ops.out_linear2_seconds * 1000.0,
        avg.forward_ops.softmax_seconds * 1000.0,
    );
    for idx in 0..4 {
        println!(
            "avg_forward_block_{idx}_ms linear={:.3} memory={:.3} affine={:.3}",
            avg.forward_ops.block_linear_seconds[idx] * 1000.0,
            avg.forward_ops.block_memory_seconds[idx] * 1000.0,
            avg.forward_ops.block_affine_seconds[idx] * 1000.0,
        );
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct RunTiming {
    elapsed: Duration,
    pcm_seconds: f64,
    frontend_seconds: f64,
    forward_seconds: f64,
    segmenter_seconds: f64,
    forward_ops: FsmnVadForwardTiming,
}

fn average_timing(runs: &[RunTiming]) -> RunTiming {
    let count = runs.len() as f64;
    let elapsed_seconds = runs
        .iter()
        .map(|run| run.elapsed.as_secs_f64())
        .sum::<f64>()
        / count;
    RunTiming {
        elapsed: Duration::from_secs_f64(elapsed_seconds),
        pcm_seconds: runs.iter().map(|run| run.pcm_seconds).sum::<f64>() / count,
        frontend_seconds: runs.iter().map(|run| run.frontend_seconds).sum::<f64>() / count,
        forward_seconds: runs.iter().map(|run| run.forward_seconds).sum::<f64>() / count,
        segmenter_seconds: runs.iter().map(|run| run.segmenter_seconds).sum::<f64>() / count,
        forward_ops: average_forward_ops(runs, count),
    }
}

fn average_forward_ops(runs: &[RunTiming], count: f64) -> FsmnVadForwardTiming {
    let mut avg = FsmnVadForwardTiming::default();
    for run in runs {
        let ops = run.forward_ops;
        avg.in_linear1_seconds += ops.in_linear1_seconds / count;
        avg.in_linear2_seconds += ops.in_linear2_seconds / count;
        avg.out_linear1_seconds += ops.out_linear1_seconds / count;
        avg.out_linear2_seconds += ops.out_linear2_seconds / count;
        avg.softmax_seconds += ops.softmax_seconds / count;
        for idx in 0..4 {
            avg.block_linear_seconds[idx] += ops.block_linear_seconds[idx] / count;
            avg.block_memory_seconds[idx] += ops.block_memory_seconds[idx] / count;
            avg.block_affine_seconds[idx] += ops.block_affine_seconds[idx] / count;
        }
    }
    avg
}

fn parse_args() -> Result<Args> {
    let mut audio_path = PathBuf::from("assets/vad_example.wav");
    let mut model_path = None;
    let mut device = None;
    let mut warmup = 1usize;
    let mut repeat = 5usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--audio" => audio_path = next_path(&mut args, "--audio")?,
            "--model" => model_path = Some(next_path(&mut args, "--model")?),
            "--device" => device = Some(next_value(&mut args, "--device")?),
            "--warmup" => warmup = parse_usize(&next_value(&mut args, "--warmup")?, "--warmup")?,
            "--repeat" => repeat = parse_usize(&next_value(&mut args, "--repeat")?, "--repeat")?,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other if !other.starts_with('-') => audio_path = PathBuf::from(other),
            other => bail!("unknown argument {other:?}; pass --help for usage"),
        }
    }

    if repeat == 0 {
        bail!("--repeat must be greater than zero");
    }
    if let Some(device) = &device
        && device != "cpu"
        && device != "metal"
        && device != "cuda"
    {
        bail!("--device must be cpu, metal, or cuda");
    }

    Ok(Args {
        audio_path,
        model_path,
        device,
        warmup,
        repeat,
    })
}

fn print_help() {
    println!(
        "Usage: cargo run -p vasr-vad --example bench_fsmn_vad -- [OPTIONS] [AUDIO]\n\
\n\
Options:\n\
  --audio PATH      WAV file to benchmark (default: assets/vad_example.wav)\n\
  --model PATH      FSMN VAD model directory; defaults to known local/cache paths\n\
  --device DEVICE   cpu, metal, or cuda; equivalent to VASR_VAD_DEVICE\n\
  --warmup N        warmup iterations (default: 1)\n\
  --repeat N        measured iterations (default: 5)\n\
  -h, --help        show this help"
    );
}

fn next_path(args: &mut impl Iterator<Item = String>, name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(next_value(args, name)?))
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))
}

fn parse_usize(value: &str, name: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("invalid {name} value {value:?}"))
}

fn default_model_path() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("/workspace/data/models/asr/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        PathBuf::from(".cache/fsmn-vad"),
        modelhub::modelscope_cache_dir().join(FSMN_VAD_REPO),
        modelscope_model_snapshot_dir(&modelhub::modelscope_cache_dir(), FSMN_VAD_REPO),
    ];
    candidates
        .into_iter()
        .find(|path| path.join("model.pt").exists() && path.join("am.mvn").exists())
}

fn modelscope_model_snapshot_dir(cache: &Path, repo_id: &str) -> PathBuf {
    cache
        .join("models")
        .join(repo_id.replace('/', "--"))
        .join("snapshots")
        .join("master")
}

fn validate_waveform(waveform: &Waveform, path: &Path) -> Result<()> {
    if waveform.sample_rate != 16_000 {
        bail!(
            "FSMN VAD expects 16kHz audio, got sample_rate={} for {}",
            waveform.sample_rate,
            path.display()
        );
    }
    if waveform.channels != 1 {
        bail!(
            "FSMN VAD expects mono audio, got channels={} for {}",
            waveform.channels,
            path.display()
        );
    }
    Ok(())
}

fn load_pcm16_wav(path: &Path) -> Result<Waveform> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("expected RIFF/WAVE file");
    }

    let mut offset = 12usize;
    let mut sample_rate = None;
    let mut channels = None;
    let mut bits_per_sample = None;
    let mut data = None;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into()?) as usize;
        offset += 8;
        if offset + chunk_size > bytes.len() {
            bail!("invalid WAV chunk size");
        }

        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    bail!("invalid WAV fmt chunk");
                }
                let audio_format = u16::from_le_bytes(bytes[offset..offset + 2].try_into()?);
                if audio_format != 1 {
                    bail!("only PCM WAV is supported, got format={audio_format}");
                }
                channels = Some(u16::from_le_bytes(
                    bytes[offset + 2..offset + 4].try_into()?,
                ));
                sample_rate = Some(u32::from_le_bytes(
                    bytes[offset + 4..offset + 8].try_into()?,
                ));
                bits_per_sample = Some(u16::from_le_bytes(
                    bytes[offset + 14..offset + 16].try_into()?,
                ));
            }
            b"data" => data = Some(&bytes[offset..offset + chunk_size]),
            _ => {}
        }

        offset += chunk_size + (chunk_size % 2);
    }

    let sample_rate = sample_rate.ok_or_else(|| anyhow::anyhow!("WAV missing fmt chunk"))?;
    let channels = channels.ok_or_else(|| anyhow::anyhow!("WAV missing channel count"))?;
    let bits_per_sample =
        bits_per_sample.ok_or_else(|| anyhow::anyhow!("WAV missing bits_per_sample"))?;
    if bits_per_sample != 16 {
        bail!("only PCM16 WAV is supported, got bits_per_sample={bits_per_sample}");
    }
    if channels != 1 {
        bail!("only mono WAV is supported, got channels={channels}");
    }
    let data = data.ok_or_else(|| anyhow::anyhow!("WAV missing data chunk"))?;
    if data.len() % 2 != 0 {
        bail!("PCM16 data length must be even");
    }

    let samples = data
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| {
            let pcm = i16::from_le_bytes(*chunk);
            f32::from(pcm) / f32::from(i16::MAX)
        })
        .collect();
    Ok(Waveform::new(samples, sample_rate))
}
