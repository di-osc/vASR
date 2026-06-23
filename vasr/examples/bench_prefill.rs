use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
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
    let seq_len = parse_arg(args.next(), "seq_len", 512usize)?;
    let repeats = parse_arg(args.next(), "repeats", 10usize)?;
    let dtype = args
        .next()
        .as_deref()
        .map(parse_dtype)
        .transpose()?
        .unwrap_or(DType::BF16);
    let isq = args.next();

    let device = default_gpu_device()?;
    let model = load_model(&model_id, &device, dtype, isq)?;
    let input_ids = synthetic_input_ids(batch, seq_len, &device)?;
    let attention_mask = Tensor::ones((batch, seq_len), DType::U32, &device)?;
    let position_ids = get_rope_index(&attention_mask)?.0;
    let inputs_embeds = model.thinker().embed_tokens(&input_ids)?;

    let mut total_ms = 0.0;
    for run in 0..repeats {
        let mut kv_cache =
            KVCache::with_max_seq_len(model.thinker().num_text_layers(), seq_len + 1);
        let start = Instant::now();
        let logits = model.thinker().forward_embeds_with_kv_cache(
            &attention_mask,
            &position_ids,
            &inputs_embeds,
            &mut kv_cache,
        )?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed_ms;
        let (_, _, vocab) = logits.dims3()?;
        println!(
            "run={} batch={} seq_len={} prefill_tokens={} vocab={} prefill_ms={:.3} prefill_tok_s={:.1}",
            run + 1,
            batch,
            seq_len,
            batch * seq_len,
            vocab,
            elapsed_ms,
            (batch * seq_len) as f64 / (elapsed_ms / 1000.0),
        );
    }
    println!("avg_prefill_ms={:.3}", total_ms / repeats.max(1) as f64,);
    Ok(())
}

fn synthetic_input_ids(batch: usize, seq_len: usize, device: &Device) -> Result<Tensor> {
    let mut ids = Vec::with_capacity(batch * seq_len);
    for i in 0..batch * seq_len {
        ids.push(1000u32 + (i % 1024) as u32);
    }
    Tensor::from_vec(ids, (batch, seq_len), device).map_err(Into::into)
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
        anyhow::bail!("bench_prefill requires a GPU build; enable `cuda` or `metal`");
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
