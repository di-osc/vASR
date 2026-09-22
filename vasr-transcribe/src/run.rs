use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dataset::{is_vasr_data_file, read_vasr_audios};
use crate::pipeline::{TranscribeInput, TranscribeItemOutcome};
use crate::protocol::{InferenceData, InferencePerformance, TranscribeResponse};
use crate::serve::{TranscribePipelineArgs, build_async_transcribe_pipeline, validate_pipeline};
use anyhow::{Context, Result, bail};
use clap::Args;
use vasr_data::AudioSource;

#[derive(Debug, Clone, Args)]
pub struct RunTranscribeArgs {
    #[command(flatten)]
    pub pipeline: TranscribePipelineArgs,

    /// Audio file, directory containing audio files, or vASR MessagePack data file.
    #[arg(short = 'i', long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output file or directory. Defaults to `{stem}-transcription.json` beside each input.
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

    let multiple_inputs = items.len() > 1 || args.input.is_dir();
    validate_output_target(args.output.as_deref(), multiple_inputs)?;

    let pipeline = build_async_transcribe_pipeline(&args.pipeline, args.language.clone())?;
    let inputs = items
        .iter()
        .enumerate()
        .map(|(index, item)| TranscribeInput {
            index,
            source: item.source.clone(),
            mono: true,
        })
        .collect::<Vec<_>>();

    tracing::info!(
        target: "vasr_transcribe::run",
        "run | inputs={} input={}",
        items.len(),
        args.input.display(),
    );

    let batch_start = Instant::now();
    let outcomes = pipeline.run(inputs).await;
    let batch_wall = batch_start.elapsed().as_secs_f64();

    let mut total_audio_seconds = 0.0;
    let mut bad_count = 0usize;
    let postprocess_start = Instant::now();
    let mut response_seconds = 0.0;
    let mut write_seconds = 0.0;

    for (item, outcome) in items.iter().zip(outcomes.iter()) {
        total_audio_seconds += outcome.audio_seconds;
        let output_path =
            resolve_output_json_path(args.output.as_deref(), &item.output_path, multiple_inputs)?;
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }

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
        let json = serde_json::to_vec_pretty(&response)
            .with_context(|| format!("failed to encode JSON for {}", output_path.display()))?;
        fs::write(&output_path, json)
            .with_context(|| format!("failed to write JSON to {}", output_path.display()))?;
        write_seconds += write_start.elapsed().as_secs_f64();

        tracing::debug!(
            target: "vasr_transcribe::run",
            "Wrote `{}`.",
            output_path.display()
        );
    }

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
        "run done | inputs={} failed={} audio={:.1}s wall={:.1}s speed={:.1}x rtf={:.4}",
        items.len(),
        bad_count,
        total_audio_seconds,
        batch_wall,
        speedup,
        rtf
    );

    if bad_count > 0 {
        bail!("{bad_count} of {} transcription(s) failed", items.len());
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct RunInputItem {
    source: AudioSource,
    output_path: PathBuf,
    service_path: PathBuf,
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
    if input.is_file() && is_vasr_data_file(input) {
        return collect_vasr_inputs(input);
    }
    let files = collect_audio_inputs(input, recursive)?;
    Ok(files
        .into_iter()
        .map(|path| RunInputItem {
            source: AudioSource::Path(path.clone()),
            output_path: path.clone(),
            service_path: path,
        })
        .collect())
}

fn collect_vasr_inputs(input: &Path) -> Result<Vec<RunInputItem>> {
    let audios = read_vasr_audios(input)?;
    if audios.is_empty() {
        bail!("vASR record list is empty: {}", input.display());
    }
    let base_dir = input.parent().unwrap_or_else(|| Path::new("."));
    audios
        .into_iter()
        .map(|audio| {
            let audio_id = audio.audio_id();
            let source = audio.source.clone();
            let record_path = base_dir.join(audio_id);
            Ok(RunInputItem {
                source,
                output_path: record_path.clone(),
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

fn validate_output_target(output: Option<&Path>, multiple_inputs: bool) -> Result<()> {
    if multiple_inputs {
        if let Some(output) = output
            && output.extension().is_some_and(|ext| ext == "json")
        {
            bail!(
                "when transcribing multiple files, --output must be a directory, not `{}`",
                output.display()
            );
        }
        return Ok(());
    }
    Ok(())
}

pub fn resolve_output_json_path(
    output: Option<&Path>,
    audio: &Path,
    multiple_inputs: bool,
) -> Result<PathBuf> {
    if let Some(output) = output
        && !multiple_inputs
        && output.extension().is_some_and(|ext| ext == "json")
    {
        return Ok(output.to_path_buf());
    }

    let stem = audio
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("invalid audio file name: {}", audio.display()))?;
    let file_name = format!("{stem}-transcription.json");

    if let Some(output) = output {
        return Ok(output.join(file_name));
    }

    let parent = audio.parent().unwrap_or_else(|| Path::new("."));
    Ok(parent.join(file_name))
}

#[cfg(test)]
mod tests {
    use super::{collect_audio_inputs, collect_run_inputs, resolve_output_json_path};
    use std::path::Path;
    use vasr_data::{Audio, AudioDb, AudioEncoding, AudioFormat, AudioInfo, AudioSource};

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
    fn resolve_output_json_path_uses_stem_for_directory_output() {
        let output = resolve_output_json_path(
            Some(Path::new("/tmp/out")),
            Path::new("/data/audio (1).wav"),
            true,
        )
        .expect("resolve output");
        assert_eq!(output, Path::new("/tmp/out/audio (1)-transcription.json"));
    }

    #[test]
    fn resolve_output_json_path_defaults_next_to_input() {
        let output = resolve_output_json_path(None, Path::new("/data/audio (1).wav"), true)
            .expect("resolve output");
        assert_eq!(output, Path::new("/data/audio (1)-transcription.json"));
    }

    #[test]
    fn resolve_output_json_path_honors_explicit_json_file() {
        let output = resolve_output_json_path(
            Some(Path::new("/tmp/custom.transcribe.json")),
            Path::new("/data/audio.wav"),
            false,
        )
        .expect("resolve output");
        assert_eq!(output, Path::new("/tmp/custom.transcribe.json"));
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
        assert_eq!(items[0].output_path, dir.join("call-1"));
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
