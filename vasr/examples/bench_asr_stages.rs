use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use vasr::qwen3_asr::{AudioInput, Batch, LoadOptions, Qwen3Asr, TranscribeOptions};
#[cfg(feature = "paged-attn")]
use vasr_paged_attn::{PagedCacheConfig, PagedCacheMemory};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let model_id = args
        .next()
        .unwrap_or_else(|| "Qwen/Qwen3-ASR-0.6B".to_string());
    let audio_paths_arg = args
        .next()
        .unwrap_or_else(|| "fixtures/audio/asr_en_16k.wav".to_string());
    let repeats = args
        .next()
        .as_deref()
        .unwrap_or("3")
        .parse::<usize>()
        .context("failed to parse repeats")?;
    let max_new_tokens = args
        .next()
        .as_deref()
        .unwrap_or("128")
        .parse::<usize>()
        .context("failed to parse max_new_tokens")?;
    let max_batch_audio_sec = args
        .next()
        .as_deref()
        .unwrap_or("0")
        .parse::<f32>()
        .context("failed to parse max_batch_audio_sec")?;
    let decode_max_seqs = args
        .next()
        .as_deref()
        .unwrap_or("0")
        .parse::<usize>()
        .context("failed to parse decode_max_seqs")?;
    let dtype = args
        .next()
        .as_deref()
        .map(parse_dtype)
        .transpose()?
        .unwrap_or(DType::BF16);
    let isq = args.next();

    while let Some(arg) = args.next() {
        return Err(anyhow::anyhow!(
            "unknown argument {arg:?}; usage: bench_asr_stages [model] [audio1[,audio2...]] [repeats] [max_new_tokens] [max_batch_audio_sec] [decode_max_seqs] [dtype] [isq]"
        ));
    }

    let audio_paths: Vec<PathBuf> = audio_paths_arg.split(',').map(PathBuf::from).collect();
    if audio_paths.is_empty() {
        anyhow::bail!("at least one audio path is required");
    }

    let device = default_gpu_device()?;
    let use_flash_attn = cfg!(feature = "flash-attn") && device.is_cuda();
    let start_load = Instant::now();
    let model = Qwen3Asr::from_pretrained(
        &model_id,
        &device,
        &LoadOptions {
            dtype,
            use_flash_attn,
            isq,
            #[cfg(feature = "paged-attn")]
            paged_cache: Some(default_paged_cache_config(&device)),
        },
    )
    .context("failed to load model")?;
    println!(
        "load_ms={:.3} model={:?} audio_files={} dtype={:?}",
        start_load.elapsed().as_secs_f64() * 1000.0,
        model_id,
        audio_paths.len(),
        dtype
    );

    let opts = TranscribeOptions {
        context: Batch::one(String::new()),
        language: Batch::one(None),
        return_timestamps: false,
        max_new_tokens,
        max_batch_audio_sec,
        chunk_max_sec: None,
        bucket_by_length: false,
        decode_max_seqs,
    };

    for run in 0..repeats {
        let audio = audio_inputs(audio_paths.as_slice());

        let encode_start = Instant::now();
        let encoded = model.encode_audio_features_to_cpu(audio, opts.clone())?;
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = Instant::now();
        let outputs = model.decode_cpu_audio_features(encoded, opts.clone())?;
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let estimated_text_tokens = estimate_text_tokens(&model, outputs.as_slice())?;

        let timed_audio = audio_inputs(audio_paths.as_slice());
        let timed_start = Instant::now();
        let (_timed_outputs, timings) = model.transcribe_timed(timed_audio, opts.clone())?;
        let timed_wall_ms = timed_start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "run={} staged_encode_ms={:.3} staged_decode_ms={:.3} staged_total_ms={:.3} timed_wall_ms={:.3} audio_encoder_ms={:.3} prefill_ms={:.3} decode_ms={:.3} prefill_tokens={} generated_tokens={} decode_steps={} prefill_tok_s={:.1} decode_tok_s={:.1} estimated_text_tokens={}",
            run + 1,
            encode_ms,
            decode_ms,
            encode_ms + decode_ms,
            timed_wall_ms,
            us_to_ms(timings.audio_encoder_us),
            us_to_ms(timings.generation.prefill_us),
            us_to_ms(timings.generation.decode_us),
            timings.generation.prefill_tokens,
            timings.generation.tokens_generated,
            timings.generation.steps,
            tokens_per_sec(
                timings.generation.prefill_tokens,
                timings.generation.prefill_us
            ),
            tokens_per_sec(
                timings.generation.tokens_generated,
                timings.generation.decode_us
            ),
            estimated_text_tokens,
        );
    }

    Ok(())
}

fn audio_inputs(paths: &[PathBuf]) -> Vec<AudioInput<'_>> {
    paths
        .iter()
        .map(|path| AudioInput::Path(Path::new(path)))
        .collect()
}

fn estimate_text_tokens(
    model: &Qwen3Asr,
    outputs: &[vasr::qwen3_asr::AsrTranscription],
) -> Result<usize> {
    outputs.iter().try_fold(0usize, |acc, output| {
        let len = model
            .processor()
            .tokenizer
            .encode(output.text.as_str())?
            .len();
        Ok(acc.saturating_add(len))
    })
}

fn us_to_ms(us: u64) -> f64 {
    us as f64 / 1000.0
}

fn tokens_per_sec(tokens: usize, us: u64) -> f64 {
    if us == 0 {
        0.0
    } else {
        tokens as f64 * 1_000_000.0 / us as f64
    }
}

fn parse_dtype(value: &str) -> Result<DType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        other => anyhow::bail!("unknown dtype {other:?}; expected f32, f16, or bf16"),
    }
}

fn default_gpu_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::new_cuda_with_stream(0).context("failed to create CUDA device 0");
    }

    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        return Device::new_metal(0).context("failed to create Metal device 0");
    }

    #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
    {
        anyhow::bail!("bench_asr_stages requires a GPU build; enable `cuda` or `metal`");
    }
}

#[cfg(feature = "paged-attn")]
fn default_paged_cache_config(device: &Device) -> PagedCacheConfig {
    let memory = if device.is_cuda() {
        PagedCacheMemory::GpuMemoryFraction {
            fraction: 0.4,
            reserved_bytes: 0,
        }
    } else {
        PagedCacheMemory::ContextSize(100_000)
    };
    PagedCacheConfig {
        block_size: 32,
        memory,
    }
}
