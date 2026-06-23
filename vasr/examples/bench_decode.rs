use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use vasr::qwen3_asr::model::kv_cache::KVCache;
use vasr::qwen3_asr::model::thinker::get_rope_index;
use vasr::qwen3_asr::{LoadOptions, Qwen3Asr};
#[cfg(feature = "paged-attn")]
use vasr_paged_attn::{PagedCacheConfig, PagedCacheMemory};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_id = args
        .next()
        .unwrap_or_else(|| "Qwen/Qwen3-ASR-0.6B".to_string());
    let batch = parse_arg(args.next(), "batch", 8usize)?;
    let prompt_len = parse_arg(args.next(), "prompt_len", 512usize)?;
    let decode_steps = parse_arg(args.next(), "decode_steps", 128usize)?;
    let repeats = parse_arg(args.next(), "repeats", 5usize)?;
    let dtype = args
        .next()
        .as_deref()
        .map(parse_dtype)
        .transpose()?
        .unwrap_or(DType::BF16);
    let isq = args.next();

    let device = default_gpu_device()?;
    let model = load_model(&model_id, &device, dtype, isq)?;
    let input_ids = synthetic_input_ids(batch, prompt_len, &device)?;
    let attention_mask = Tensor::ones((batch, prompt_len), DType::U32, &device)?;
    let position_ids = get_rope_index(&attention_mask)?.0;
    let inputs_embeds = model.thinker().embed_tokens(&input_ids)?;
    let prompt_lens = vec![prompt_len as i64; batch];
    let decode_position_ids =
        position_ids_for_decode_steps_batch(prompt_lens.as_slice(), decode_steps, &device)?;

    let mut total_ms = 0.0;
    for run in 0..repeats {
        let mut kv_cache =
            KVCache::with_max_seq_len(model.thinker().num_text_layers(), prompt_len + decode_steps);
        let _prefill_logits = model.thinker().forward_embeds_with_kv_cache(
            &attention_mask,
            &position_ids,
            &inputs_embeds,
            &mut kv_cache,
        )?;
        let mut next_ids = vec![1000u32; batch];

        let start = Instant::now();
        for step in 0..decode_steps {
            let input_ids_new = Tensor::from_vec(next_ids.clone(), (batch, 1usize), &device)?;
            let inputs_embeds_new = model.thinker().embed_tokens(&input_ids_new)?;
            let logits = model.thinker().forward_decode_one_without_padding(
                &decode_position_ids[step],
                &inputs_embeds_new,
                &mut kv_cache,
            )?;
            next_ids = next_ids_from_logits(&logits, batch)?;
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed_ms;
        let generated_tokens = batch * decode_steps;
        println!(
            "run={} batch={} prompt_len={} decode_steps={} generated_tokens={} decode_ms={:.3} decode_tok_s={:.1}",
            run + 1,
            batch,
            prompt_len,
            decode_steps,
            generated_tokens,
            elapsed_ms,
            generated_tokens as f64 / (elapsed_ms / 1000.0),
        );
    }
    println!("avg_decode_ms={:.3}", total_ms / repeats.max(1) as f64);
    Ok(())
}

fn next_ids_from_logits(logits: &Tensor, batch: usize) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(batch);
    for row in 0..batch {
        let step_logits = logits.i((row, 0usize))?;
        out.push(step_logits.argmax(0usize)?.to_scalar::<u32>()?);
    }
    Ok(out)
}

fn synthetic_input_ids(batch: usize, seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut ids = Vec::with_capacity(batch * seq_len);
    for i in 0..batch * seq_len {
        ids.push(1000u32 + (i % 1024) as u32);
    }
    Tensor::from_vec(ids, (batch, seq_len), device).map_err(Into::into)
}

fn position_ids_for_decode_steps_batch(
    prompt_lens: &[i64],
    max_steps: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let batch = prompt_lens.len();
    let mut out = Vec::with_capacity(max_steps);
    for step in 0..max_steps {
        let step_i64 =
            i64::try_from(step).map_err(|_| anyhow::anyhow!("step overflows i64: {step}"))?;
        let mut pos = Vec::with_capacity(batch);
        for &prompt_len in prompt_lens {
            if prompt_len <= 0 {
                anyhow::bail!("prompt_len must be > 0, got {prompt_len}");
            }
            pos.push(prompt_len + step_i64);
        }
        let pos_t = Tensor::from_vec(pos, (batch, 1usize), device)?;
        out.push(pos_t.unsqueeze(0)?.broadcast_as((3usize, batch, 1usize))?);
    }
    Ok(out)
}

fn load_model(
    model_id: &str,
    device: &Device,
    dtype: DType,
    isq: Option<String>,
) -> Result<Qwen3Asr> {
    Qwen3Asr::from_pretrained(
        model_id,
        device,
        &LoadOptions {
            dtype,
            use_flash_attn: cfg!(feature = "flash-attn") && device.is_cuda(),
            isq,
            #[cfg(feature = "paged-attn")]
            paged_cache: Some(default_paged_cache_config(device)),
        },
    )
    .context("failed to load Qwen3-ASR model")
}

fn parse_arg<T>(value: Option<String>, name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    value
        .as_deref()
        .map(str::parse::<T>)
        .transpose()
        .with_context(|| format!("failed to parse {name}"))
        .map(|v| v.unwrap_or(default))
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
        anyhow::bail!("bench_decode requires a GPU build; enable `cuda` or `metal`");
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
