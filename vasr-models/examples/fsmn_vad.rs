use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use vasr_data::Waveform;
use vasr_models::vad::{FsmnVadModel, VadOptions};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let audio_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("vad_example.wav"));
    let model_path = args.next().map(PathBuf::from).or_else(default_model_path);

    let waveform = load_pcm16_wav(&audio_path)
        .with_context(|| format!("failed to load WAV {}", audio_path.display()))?;
    if waveform.sample_rate != 16_000 {
        bail!(
            "FSMN VAD expects 16kHz audio, got sample_rate={} for {}",
            waveform.sample_rate,
            audio_path.display()
        );
    }
    if waveform.channels != 1 {
        bail!(
            "FSMN VAD expects mono audio, got channels={} for {}",
            waveform.channels,
            audio_path.display()
        );
    }

    let load_start = Instant::now();
    let model = FsmnVadModel::from_pretrained(model_path.as_deref())?;
    eprintln!(
        "loaded FSMN VAD from {} in {:.3}s",
        model.model_dir().display(),
        load_start.elapsed().as_secs_f64()
    );

    let options = VadOptions::default();
    let detection = model.detect_with_timing(&waveform, &options)?;

    println!(
        "audio={} sample_rate={} duration_ms={} threshold={} min_silence_ms={} segments={}",
        audio_path.display(),
        waveform.sample_rate,
        waveform.duration_ms() as u64,
        options.threshold,
        options.min_silence_ms,
        detection.segments.len(),
    );
    for (idx, segment) in detection.segments.iter().enumerate() {
        println!(
            "{idx:03} start_ms={} end_ms={} duration_ms={}",
            segment.range.start.0,
            segment.range.end.0,
            segment.range.duration().0,
        );
    }

    Ok(())
}

fn default_model_path() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("/workspace/data/models/asr/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        PathBuf::from(".cache/fsmn-vad"),
        vasr_models::modelscope_cache_dir().join("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
    ];
    candidates
        .into_iter()
        .find(|path| path.join("model.pt").exists() && path.join("am.mvn").exists())
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
            b"data" => {
                data = Some(&bytes[offset..offset + chunk_size]);
            }
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
        .chunks_exact(2)
        .map(|chunk| {
            let pcm = i16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from(pcm) / f32::from(i16::MAX)
        })
        .collect();
    Ok(Waveform::new(samples, sample_rate))
}
