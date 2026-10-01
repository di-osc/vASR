//! Sustained FSMN VAD throughput over a preloaded dataset, with boundary evidence.
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use candle_core::Device;
use serde::Serialize;
use asr_data::{AudioDb, AudioDbMode, AudioQuery, MAX_QUERY_LIMIT, Waveform};
use vasr_vad::{FsmnVadModel, VadOptions};

const MEASURED_ROUNDS: usize = 3;

struct Args {
    dataset: PathBuf,
    model: Option<PathBuf>,
    device: String,
    output: PathBuf,
}

struct Record {
    id: String,
    waveform: Waveform,
    reference: serde_json::Value,
}

#[derive(Serialize)]
struct Boundary {
    start_ms: usize,
    end_ms: usize,
    probability: f32,
}

#[derive(Serialize)]
struct RecordRun {
    id: String,
    wall_seconds: f64,
    pcm_seconds: f64,
    frontend_seconds: f64,
    forward_seconds: f64,
    segmenter_seconds: f64,
    segments: Vec<Boundary>,
}

#[derive(Serialize)]
struct Round {
    round: usize,
    wall_seconds: f64,
    audio_seconds: f64,
    speedup: f64,
    records: Vec<RecordRun>,
}

fn serving_options() -> VadOptions {
    VadOptions {
        threshold: 0.6,
        min_speech_ms: 250,
        min_silence_ms: 800,
        max_segment_ms: 30_000,
        pad_ms: 0,
        num_threads: 0,
    }
}

fn load_records(dataset: &std::path::Path) -> Result<(Vec<Record>, f64, f64)> {
    let query_start = Instant::now();
    let db = AudioDb::open(dataset, AudioDbMode::ReadOnly)?;
    let mut audios = Vec::new();
    let mut after = None;
    loop {
        let batch = db.query(&AudioQuery {
            limit: MAX_QUERY_LIMIT,
            after: after.clone(),
            ..AudioQuery::default()
        })?;
        if batch.is_empty() {
            break;
        }
        after = batch.last().map(|audio| audio.id.clone());
        audios.extend(batch);
    }
    ensure!(!audios.is_empty(), "dataset contains no audio records");
    let query_seconds = query_start.elapsed().as_secs_f64();
    let decode_start = Instant::now();
    let mut records = Vec::with_capacity(audios.len());
    for mut audio in audios {
        let reference = serde_json::to_value(
            audio
                .timelines()
                .iter()
                .map(|(channel, timeline)| {
                    (
                        channel,
                        serde_json::json!({
                            "id": timeline.id,
                            "audio_id": timeline.audio_id,
                            "duration_ms": timeline.duration,
                            "reference": timeline.reference,
                        }),
                    )
                })
                .collect::<std::collections::BTreeMap<_, _>>(),
        )?;
        let waveform = audio
            .as_waveform()
            .with_context(|| format!("decode audio {}", audio.id))?
            .to_mono()?
            .resample(16_000)?;
        ensure!(
            waveform.sample_rate == 16_000 && waveform.channels == 1,
            "{} was not converted to 16kHz mono",
            audio.id
        );
        records.push(Record {
            id: audio.id,
            waveform,
            reference,
        });
    }
    Ok((records, query_seconds, decode_start.elapsed().as_secs_f64()))
}

fn main() -> Result<()> {
    let args = parse_args(std::env::args().skip(1))?;
    let (records, query_seconds, waveform_decode_seconds) = load_records(&args.dataset)?;
    let audio_seconds: f64 = records
        .iter()
        .map(|record| record.waveform.duration_seconds())
        .sum();
    ensure!(audio_seconds > 0.0, "dataset has no nonempty audio");
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::new_metal(0)?,
        _ => unreachable!("validated device"),
    };
    let load_start = Instant::now();
    let model = FsmnVadModel::from_pretrained_on_device(args.model.as_deref(), &device)?;
    device.synchronize()?;
    let model_load_seconds = load_start.elapsed().as_secs_f64();
    let options = serving_options();
    let warmup_start = Instant::now();
    for record in &records {
        model.detect_with_timing(&record.waveform, &options)?;
    }
    device.synchronize()?;
    let warmup_seconds = warmup_start.elapsed().as_secs_f64();
    println!(
        "preloaded records={} audio_seconds={audio_seconds:.3} query_seconds={query_seconds:.3} waveform_decode_seconds={waveform_decode_seconds:.3} model_load_seconds={model_load_seconds:.3} full_dataset_warmup_seconds={warmup_seconds:.3}",
        records.len()
    );

    let mut rounds = Vec::with_capacity(MEASURED_ROUNDS);
    for round in 1..=MEASURED_ROUNDS {
        let mut results = Vec::with_capacity(records.len());
        let round_start = Instant::now();
        for record in &records {
            let start = Instant::now();
            // Frame scores are copied to CPU inside detection, so each call includes
            // GPU completion and CPU segmentation/refinement, as in the serving path.
            let detection = model.detect_with_timing(&record.waveform, &options)?;
            let wall_seconds = start.elapsed().as_secs_f64();
            results.push(RecordRun {
                id: record.id.clone(),
                wall_seconds,
                pcm_seconds: detection.timing.pcm_seconds,
                frontend_seconds: detection.timing.frontend_seconds,
                forward_seconds: detection.timing.forward_seconds,
                segmenter_seconds: detection.timing.segmenter_seconds,
                segments: detection
                    .segments
                    .into_iter()
                    .map(|segment| Boundary {
                        start_ms: segment.range.start_ms,
                        end_ms: segment.range.end_ms,
                        probability: segment.probability,
                    })
                    .collect(),
            });
        }
        device.synchronize()?;
        let wall_seconds = round_start.elapsed().as_secs_f64();
        let speedup = audio_seconds / wall_seconds;
        println!(
            "round={round} device={} wall_seconds={wall_seconds:.3} audio_seconds={audio_seconds:.3} speedup={speedup:.2}x",
            args.device
        );
        rounds.push(Round {
            round,
            wall_seconds,
            audio_seconds,
            speedup,
            records: results,
        });
    }
    let measured_wall_seconds: f64 = rounds.iter().map(|round| round.wall_seconds).sum();
    let measured_audio_seconds = audio_seconds * MEASURED_ROUNDS as f64;
    let sustained_speedup = measured_audio_seconds / measured_wall_seconds;
    let report = serde_json::json!({
        "dataset": args.dataset,
        "device": args.device,
        "model": model.model_dir(),
        "sample_rate": 16000,
        "channels": 1,
        "options": {"threshold": options.threshold, "min_speech_ms": options.min_speech_ms,
            "min_silence_ms": options.min_silence_ms, "max_segment_ms": options.max_segment_ms,
            "pad_ms": options.pad_ms, "num_threads": options.num_threads},
        "preload": {"query_seconds": query_seconds, "waveform_decode_seconds": waveform_decode_seconds,
            "model_load_seconds": model_load_seconds, "warmup_seconds": warmup_seconds, "warmup_dataset_rounds": 1},
        "record_count": records.len(),
        "dataset_audio_seconds": audio_seconds,
        "measured_audio_seconds": measured_audio_seconds,
        "measured_wall_seconds": measured_wall_seconds,
        "sustained_speedup": sustained_speedup,
        "reference_records": records.iter().map(|record| serde_json::json!({
            "id": record.id, "duration_seconds": record.waveform.duration_seconds(), "timelines": record.reference,
        })).collect::<Vec<_>>(),
        "rounds": rounds,
    });
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "sustained_speedup={sustained_speedup:.2}x measured_audio_seconds={measured_audio_seconds:.3} measured_wall_seconds={measured_wall_seconds:.3} report={}",
        args.output.display()
    );
    Ok(())
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args> {
    let mut dataset = PathBuf::from("datasets/call-100.db");
    let mut model = None;
    let mut device = String::from("cpu");
    let mut output = None;
    while let Some(arg) = args.next() {
        if matches!(arg.as_str(), "--help" | "-h") {
            println!(
                "Usage: cargo run --release -p vasr-vad --example bench_fsmn_dataset [--features metal] -- --dataset datasets/call-100.db --device cpu|metal [--model PATH] [--output PATH]\nOne full dataset warmup, followed by three measured rounds; waveform decoding excluded."
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .with_context(|| format!("{arg} requires a value"))?;
        match arg.as_str() {
            "--dataset" | "-i" => dataset = PathBuf::from(value),
            "--device" => device = value,
            "--model" => model = Some(PathBuf::from(value)),
            "--output" | "-o" => output = Some(PathBuf::from(value)),
            _ => bail!("unknown argument {arg:?}"),
        }
    }
    ensure!(
        matches!(device.as_str(), "cpu" | "metal"),
        "--device must be cpu or metal"
    );
    let output =
        output.unwrap_or_else(|| PathBuf::from(format!("results/fsmn_dataset_{device}.json")));
    Ok(Args {
        dataset,
        model,
        device,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dataset_bench_options_match_serving() {
        let options = serving_options();
        assert_eq!(options.threshold, 0.6);
        assert_eq!(
            (
                options.min_speech_ms,
                options.min_silence_ms,
                options.max_segment_ms,
                options.pad_ms
            ),
            (250, 800, 30_000, 0)
        );
        let args = parse_args(
            ["--device", "metal", "--dataset", "datasets/example1"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(
            args.output,
            PathBuf::from("results/fsmn_dataset_metal.json")
        );
        assert!(parse_args(["--device", "invalid"].into_iter().map(String::from)).is_err());
    }
}
