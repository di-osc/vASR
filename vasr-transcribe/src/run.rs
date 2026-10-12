use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dataset::{is_vasr_input, read_vasr_audios};
use crate::device::resolve_device;
use crate::pipeline::{TranscribeInput, TranscribeItemOutcome};
use crate::protocol::{InferenceData, InferencePerformance, TranscribeResponse};
use crate::serve::{
    TranscribePipelineArgs, build_async_transcribe_pipeline_with_reservation, validate_pipeline,
};
use crate::serve_tuning::{
    Backend, ResolvedServeTuning, ServeTuningOverrides, estimate_model_memory_profile,
    probe_hardware, resolve_tuning,
};
use anyhow::{Context, Result, bail};
use asr_data::{Audio, AudioChannel, AudioInfo, AudioSource};
use clap::Args;

#[derive(Debug, Clone, Args)]
pub struct RunTranscribeArgs {
    #[command(flatten)]
    pub pipeline: TranscribePipelineArgs,

    /// Audio file, directory, or vASR AudioDb (`.db` / `.vasr` / `.vasr.msgpack`).
    #[arg(short = 'i', long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output JSONL file. Defaults to `{input}-transcriptions.jsonl` beside the input.
    #[arg(short = 'o', long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Recursively scan subdirectories when `--input` is a directory.
    #[arg(long, default_value_t = false)]
    pub recursive: bool,

    /// ASR language hint.
    #[arg(long, env = "VASR_LANGUAGE", default_value = "Chinese")]
    pub language: Option<String>,

    /// Process at most this many audio files.
    #[arg(long)]
    pub limit: Option<usize>,
}

pub async fn run_local(args: RunTranscribeArgs) -> Result<()> {
    let mut args = args;
    validate_pipeline(&args.pipeline)?;
    if !args.input.exists() {
        bail!("input path does not exist: {}", args.input.display());
    }

    let items = collect_run_inputs(&args.input, args.recursive)?;
    if items.is_empty() {
        bail!("no inputs found under {}", args.input.display());
    }
    let mut items = items;
    if let Some(limit) = args.limit {
        items.truncate(limit);
    }
    if items.is_empty() {
        bail!("no inputs remain after applying --limit");
    }
    let output_path = resolve_output_jsonl_path(args.output.as_deref(), &args.input)?;

    probe_run_input_durations(&mut items).await?;
    let durations = items
        .iter()
        .map(|item| item.duration_seconds.unwrap_or(0.0));
    let total_input_audio_seconds = durations.clone().sum::<f64>();
    if !total_input_audio_seconds.is_finite() {
        bail!("total input audio duration is out of range");
    }
    let (tuning, estimated_kv_tokens, estimated_sequences, safe_capacity, kv_bytes_per_token) =
        resolve_run_capacity(&args.pipeline, args.language.as_deref(), durations)?;
    let resolved_max_num_seqs = estimated_sequences
        .min(args.pipeline.model.max_num_seqs)
        .min(safe_capacity)
        .max(1);
    args.pipeline.model.max_num_seqs = resolved_max_num_seqs;
    args.pipeline.encoder_batch_max_audio_seconds = tuning.encoder_batch_max_audio_seconds;
    args.pipeline.encoder_batch_max_segments = tuning.encoder_batch_max_segments;

    let estimated_kv_bytes = kv_bytes_per_token
        .map(|bytes_per_token| estimated_kv_tokens.saturating_mul(bytes_per_token));
    tracing::info!(
        target: "vasr_transcribe::run",
        "start | inputs={} audio={:.1}s kv={}tok/{} sequences={}/{} max_num_seqs={} encoder={}seg/{:.1}s fallback={}",
        items.len(),
        total_input_audio_seconds,
        estimated_kv_tokens,
        estimated_kv_bytes
            .map(bytes_to_gib)
            .map(|gib| format!("{gib:.2}GiB"))
            .unwrap_or_else(|| "unknown".to_string()),
        estimated_sequences,
        safe_capacity,
        resolved_max_num_seqs,
        tuning.encoder_batch_max_segments,
        tuning.encoder_batch_max_audio_seconds,
        tuning.used_fallback,
    );
    if estimated_sequences > resolved_max_num_seqs {
        tracing::info!(
            target: "vasr_transcribe::run",
            estimated_sequences,
            safe_capacity,
            max_num_seqs = resolved_max_num_seqs,
            "estimated dataset KV work exceeds the resident cache; run will process it through the bounded cache"
        );
    }

    let pipeline = build_async_transcribe_pipeline_with_reservation(
        &args.pipeline,
        args.language.clone(),
        tuning.reserved_gpu_bytes,
    )?;
    let inputs = items
        .iter()
        .enumerate()
        .map(|(index, item)| TranscribeInput {
            index,
            source: item.source.clone(),
            mono: true,
        })
        .collect::<Vec<_>>();

    let batch_start = Instant::now();
    let outcomes = pipeline.run(inputs).await;
    let batch_wall = batch_start.elapsed().as_secs_f64();
    if outcomes.len() != items.len() {
        bail!(
            "transcribe pipeline returned {} result(s) for {} input(s)",
            outcomes.len(),
            items.len()
        );
    }

    let mut total_audio_seconds = 0.0;
    let mut bad_count = 0usize;
    let postprocess_start = Instant::now();
    let mut response_seconds = 0.0;
    let mut write_seconds = 0.0;

    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    let output_file = File::create(&output_path)
        .with_context(|| format!("failed to create JSONL output {}", output_path.display()))?;
    let mut output = BufWriter::new(output_file);

    for (item, outcome) in items.iter().zip(outcomes.iter()) {
        total_audio_seconds += outcome.audio_seconds;

        let response_start = Instant::now();
        let response =
            transcribe_response_for_outcome(&item.service_path, outcome, batch_wall, items.len());
        response_seconds += response_start.elapsed().as_secs_f64();
        if response.data.first().is_some_and(|item| item.is_bad) {
            bad_count += 1;
            let reason = outcome
                .result
                .as_ref()
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown error".to_string());
            let component = outcome.bad_component.unwrap_or("unknown");
            tracing::error!(
                target: "vasr_transcribe::run",
                path = %item.service_path.display(),
                component,
                error = %reason,
                "Transcription failed"
            );
        }

        let write_start = Instant::now();
        response.data.first().context("missing inference result")?;
        let audio = audio_record_for_outcome(item, outcome)?;
        serde_json::to_writer(&mut output, &audio).with_context(|| {
            format!(
                "failed to encode Audio JSONL record for {}",
                item.service_path.display()
            )
        })?;
        output.write_all(b"\n").with_context(|| {
            format!("failed to write JSONL record to {}", output_path.display())
        })?;
        write_seconds += write_start.elapsed().as_secs_f64();

        tracing::debug!(
            target: "vasr_transcribe::run",
            "Wrote Audio JSONL record for `{}`.",
            item.service_path.display()
        );
    }
    output
        .flush()
        .with_context(|| format!("failed to flush JSONL output {}", output_path.display()))?;

    let speedup = total_audio_seconds / batch_wall.max(f64::EPSILON);
    let rtf = batch_wall / total_audio_seconds.max(f64::EPSILON);
    tracing::debug!(
        target: "vasr_transcribe::run",
        "postprocess | mode=run | records={} | response={:.3}s | write={:.3}s | total={:.3}s",
        items.len(),
        response_seconds,
        write_seconds,
        postprocess_start.elapsed().as_secs_f64(),
    );
    tracing::info!(
        target: "vasr_transcribe::run",
        "run done | inputs={} failed={} audio={:.1}s wall={:.1}s speed={:.1}x rtf={:.4} output={}",
        items.len(),
        bad_count,
        total_audio_seconds,
        batch_wall,
        speedup,
        rtf,
        output_path.display()
    );

    if bad_count > 0 {
        bail!("{bad_count} of {} transcription(s) failed", items.len());
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct RunInputItem {
    source: AudioSource,
    audio: Option<Audio>,
    audio_info: Option<AudioInfo>,
    duration_seconds: Option<f64>,
    service_path: PathBuf,
}

async fn probe_run_input_durations(items: &mut [RunInputItem]) -> Result<()> {
    for item in items {
        if item.duration_seconds.is_none() {
            let info = item.source.aprobe().await.with_context(|| {
                format!(
                    "failed to probe audio duration for {}",
                    item.service_path.display()
                )
            })?;
            item.duration_seconds = Some(info.duration_ms() / 1000.0);
            item.audio_info = Some(info);
        } else if item.audio_info.is_none() {
            item.audio_info = item.audio.as_ref().map(|audio| audio.info.clone());
        }
        let duration_seconds = item.duration_seconds.unwrap_or(0.0);
        if !duration_seconds.is_finite() || duration_seconds < 0.0 {
            bail!(
                "invalid audio duration for {}: {duration_seconds}",
                item.service_path.display()
            );
        }
    }
    Ok(())
}

fn resolve_run_capacity(
    pipeline: &TranscribePipelineArgs,
    language: Option<&str>,
    durations_seconds: impl IntoIterator<Item = f64> + Clone,
) -> Result<(ResolvedServeTuning, u64, usize, usize, Option<u64>)> {
    let device = resolve_device(&pipeline.model.device)?;
    let backend = if device.is_cuda() {
        Backend::Cuda
    } else if device.is_metal() {
        Backend::Metal
    } else {
        Backend::Cpu
    };
    let hardware = match probe_hardware(&device) {
        Ok(hardware) => Some(hardware),
        Err(error) => {
            tracing::warn!(
                target: "vasr_transcribe::run",
                "automatic memory probe unavailable ({error}); run will use one decode sequence"
            );
            None
        }
    };
    let model_profile = estimate_model_memory_profile(
        &pipeline.model.model,
        pipeline.model.isq.as_deref(),
        backend,
    );
    if model_profile.is_none() {
        tracing::warn!(
            target: "vasr_transcribe::run",
            model = %pipeline.model.model,
            "model memory profile unavailable; run will use one decode sequence"
        );
    }

    let max_segment_seconds = f64::from(pipeline.vad.vad_max_segment_sec);
    let padded_segment_seconds =
        max_segment_seconds + 2.0 * pipeline.vad.vad_pad_ms as f64 / 1000.0;
    let tuning = resolve_tuning(
        hardware,
        model_profile,
        max_segment_seconds,
        pipeline.vad.vad_pad_ms,
        pipeline.model.max_hotword_tokens,
        pipeline.model.kv_peak_ratio,
        ServeTuningOverrides {
            max_num_seqs: None,
            gpu_memory_utilization: Some(pipeline.model.gpu_memory_utilization),
            encoder_batch_max_audio_seconds: Some(padded_segment_seconds),
            encoder_batch_max_segments: Some(1),
        },
    )?;

    vasr::set_cache_dir(pipeline.model.cache_dir.clone());
    let model_id = vasr::models::qwen3asr::resolve_qwen3_asr_model_id(&pipeline.model.model);
    let kv_work = vasr::models::qwen3asr::builder::estimate_kv_cache_work(
        model_id,
        language,
        durations_seconds.clone(),
        pipeline.vad.vad_max_segment_sec,
        pipeline.model.max_hotword_tokens,
        pipeline.model.kv_peak_ratio,
    )?;
    let estimated_kv_tokens =
        u64::try_from(kv_work.total_blocks.saturating_mul(kv_work.block_size)).unwrap_or(u64::MAX);
    let per_sequence_blocks = kv_work.blocks_per_sequence.max(1);
    let estimated_sequences = kv_work.total_blocks.div_ceil(per_sequence_blocks).max(1);
    let safe_capacity = if tuning.used_fallback {
        1
    } else {
        tuning.max_num_seqs.max(1)
    };
    Ok((
        tuning,
        estimated_kv_tokens,
        estimated_sequences,
        safe_capacity,
        model_profile.map(|profile| profile.kv_bytes_per_token),
    ))
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn transcribe_response_for_outcome(
    path: &Path,
    outcome: &TranscribeItemOutcome,
    batch_wall: f64,
    batch_size: usize,
) -> TranscribeResponse {
    let service_id = path.display().to_string();
    let data = match &outcome.result {
        Ok(timeline) => InferenceData::from_timeline(&service_id, timeline),
        Err(error) => error_inference_data(
            &service_id,
            outcome.bad_component.unwrap_or("recognizer"),
            error.to_string(),
        ),
    };

    TranscribeResponse {
        data: vec![data],
        inference_performance: InferencePerformance {
            batch_wall_seconds: batch_wall,
            num_items: batch_size,
            throughput_items_per_second: batch_size as f64 / batch_wall.max(f64::EPSILON),
            total_audio_duration_seconds: outcome.audio_seconds,
            speedup: outcome.audio_seconds / batch_wall.max(f64::EPSILON),
            rtf: batch_wall / outcome.audio_seconds.max(f64::EPSILON),
        },
    }
}

fn audio_record_for_outcome(item: &RunInputItem, outcome: &TranscribeItemOutcome) -> Result<Audio> {
    let info = item
        .audio_info
        .as_ref()
        .context("audio probe information is missing")?;
    let mut audio = item.audio.clone().unwrap_or_else(|| {
        Audio::with_id_from_info(
            item.service_path.display().to_string(),
            item.source.clone(),
            info,
        )
    });
    let channel = audio_channel(outcome.channel)?;
    for channel in [AudioChannel::Mono, AudioChannel::Left, AudioChannel::Right] {
        if let Some(timeline) = audio.timeline_mut(channel)? {
            timeline.prediction.clear();
        }
    }
    crate::protocol::clear_legacy_transcribe_metadata(&mut audio);
    crate::protocol::clear_transcription_error(&mut audio);
    if let Err(error) = &outcome.result {
        crate::protocol::mark_transcription_failed(
            &mut audio,
            outcome.bad_component.unwrap_or("recognizer"),
            &error.to_string(),
        );
    }
    let timeline = audio.ensure_timeline(channel, None)?;
    if let Ok(transcribed) = &outcome.result {
        timeline.prediction.clone_from(&transcribed.prediction);
    }
    Ok(audio)
}

fn audio_channel(name: &str) -> Result<AudioChannel> {
    match name {
        "mono" => Ok(AudioChannel::Mono),
        "left" => Ok(AudioChannel::Left),
        "right" => Ok(AudioChannel::Right),
        _ => bail!("unsupported transcription channel `{name}`"),
    }
}

fn error_inference_data(service_id: &str, component: &str, reason: String) -> InferenceData {
    InferenceData {
        service_id: service_id.to_string(),
        spent_seconds: 0.0,
        spent_details: Default::default(),
        timings: Default::default(),
        text: Default::default(),
        sentences: Vec::new(),
        is_bad: true,
        bad_reason: Some(reason),
        bad_component: Some(component.to_string()),
    }
}

pub fn collect_audio_inputs(input: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        if is_audio_file(input) {
            return Ok(vec![input.to_path_buf()]);
        }
        bail!("unsupported audio file extension: {}", input.display());
    }
    if !input.is_dir() {
        bail!("input path is not a file or directory: {}", input.display());
    }

    let mut files = Vec::new();
    collect_audio_in_dir(input, recursive, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_run_inputs(input: &Path, recursive: bool) -> Result<Vec<RunInputItem>> {
    if is_vasr_input(input) {
        return collect_vasr_inputs(input);
    }
    let files = collect_audio_inputs(input, recursive)?;
    Ok(files
        .into_iter()
        .map(|path| RunInputItem {
            source: AudioSource::Path(path.clone()),
            audio: None,
            audio_info: None,
            duration_seconds: None,
            service_path: path,
        })
        .collect())
}

fn collect_vasr_inputs(input: &Path) -> Result<Vec<RunInputItem>> {
    let audios = read_vasr_audios(input)?;
    if audios.is_empty() {
        bail!("vASR record list is empty: {}", input.display());
    }
    // 数据集目录本身就是根目录；单个 `.vasr` 文件则与输出并排放在其所在目录。
    let base_dir = if input.is_dir() {
        input
    } else {
        input.parent().unwrap_or_else(|| Path::new("."))
    };
    audios
        .into_iter()
        .map(|audio| {
            let audio_id = audio.audio_id();
            let source = audio.source.clone();
            let info = audio.info.clone();
            let duration_seconds = Some(info.duration_ms() / 1000.0);
            let record_path = base_dir.join(audio_id);
            Ok(RunInputItem {
                source,
                audio: Some(audio),
                audio_info: Some(info),
                duration_seconds,
                service_path: record_path,
            })
        })
        .collect()
}

fn collect_audio_in_dir(dir: &Path, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if recursive {
                collect_audio_in_dir(&path, true, out)?;
            }
            continue;
        }
        if is_audio_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "wav" | "mp3" | "flac" | "ogg" | "m4a" | "aac" | "opus" | "webm"
            )
        })
        .unwrap_or(false)
}

pub fn resolve_output_jsonl_path(output: Option<&Path>, input: &Path) -> Result<PathBuf> {
    if let Some(output) = output {
        if output.is_dir() {
            bail!(
                "--output must be a JSONL file, not directory `{}`",
                output.display()
            );
        }
        if output.extension().is_some_and(|ext| ext != "jsonl") {
            bail!(
                "--output must use the .jsonl extension: `{}`",
                output.display()
            );
        }
        return Ok(output.to_path_buf());
    }

    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid input name: {}", input.display()))?;
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    Ok(parent.join(format!("{stem}-transcriptions.jsonl")))
}

#[cfg(test)]
mod tests {
    use super::{collect_audio_inputs, collect_run_inputs, resolve_output_jsonl_path};
    use asr_data::{Audio, AudioDb, AudioEncoding, AudioFormat, AudioInfo, AudioSource};
    use std::path::Path;

    fn test_audio_info() -> AudioInfo {
        AudioInfo {
            sample_rate: 16_000,
            channels: 1,
            frame_count: 16_000,
            source_format: AudioFormat {
                encoding: AudioEncoding::Unknown,
                sample_rate: 16_000,
                channels: 1,
            },
        }
    }

    #[test]
    fn resolve_output_jsonl_path_honors_explicit_file() {
        let output = resolve_output_jsonl_path(
            Some(Path::new("/tmp/out/transcriptions.jsonl")),
            Path::new("/data/audio (1).wav"),
        )
        .expect("resolve output");
        assert_eq!(output, Path::new("/tmp/out/transcriptions.jsonl"));
    }

    #[test]
    fn resolve_output_jsonl_path_defaults_next_to_input() {
        let output = resolve_output_jsonl_path(None, Path::new("/data/audio (1).wav"))
            .expect("resolve output");
        assert_eq!(output, Path::new("/data/audio (1)-transcriptions.jsonl"));
    }

    #[test]
    fn collect_audio_inputs_accepts_single_file() {
        let dir = std::env::temp_dir().join(format!("vasr-run-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let wav = dir.join("sample.wav");
        std::fs::write(&wav, b"RIFF").expect("write wav");

        let files = collect_audio_inputs(&wav, false).expect("collect file");
        assert_eq!(files, vec![wav]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn collect_run_inputs_accepts_vasr_record_list() {
        let dir =
            std::env::temp_dir().join(format!("vasr-run-record-list-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let audio = dir.join("audio one.wav");
        let input = dir.join("records.vasr.msgpack");

        let records = [Audio::with_id_from_info(
            "call-1",
            AudioSource::from_url(format!("file://{}", audio.display()).replace(' ', "%20")),
            &test_audio_info(),
        )];
        let db = AudioDb::create(&input).expect("create db");
        db.insert(&records[0]).expect("insert audio");

        let items = collect_run_inputs(&input, false).expect("collect vasr records");

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].service_path, dir.join("call-1"));
        assert_eq!(items[0].audio.as_ref().unwrap().id, "call-1");
        assert_eq!(
            items[0].source,
            AudioSource::Url(format!("file://{}", audio.display()).replace(' ', "%20"))
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn collect_run_inputs_maps_embedded_audio_to_encoded_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "vasr-run-embedded-record-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let input = dir.join("records.vasr.msgpack");

        let bytes = b"encoded-audio".to_vec();
        let records = [Audio::with_id_from_info(
            "embedded-call",
            AudioSource::from_encoded_bytes(bytes.clone()),
            &test_audio_info(),
        )];
        let db = AudioDb::create(&input).expect("create db");
        db.insert(&records[0]).expect("insert audio");

        let items = collect_run_inputs(&input, false).expect("collect vasr records");

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].source, AudioSource::EncodedBytes(bytes));

        std::fs::remove_dir_all(&dir).ok();
    }
}
