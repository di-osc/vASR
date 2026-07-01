//! Model weight loading (safetensors + name mapping).

use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use candle_nn::var_builder::SimpleBackend;
use serde::Deserialize;

use crate::config::AsrConfig;

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub dtype: candle_core::DType,
    pub use_flash_attn: bool,
    pub isq: Option<String>,
    #[cfg(feature = "paged-attn")]
    pub paged_cache: Option<crate::model::paged_cache_runtime::PagedCacheConfig>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            dtype: candle_core::DType::F32,
            use_flash_attn: false,
            isq: None,
            #[cfg(feature = "paged-attn")]
            paged_cache: None,
        }
    }
}

pub(crate) fn load_model_from_pretrained(
    model_id_or_path: &str,
    device: &Device,
    opts: &LoadOptions,
) -> Result<(AsrConfig, super::LoadedQwen3Asr)> {
    validate_load_options(device, opts)?;

    let model_dir = resolve_model_dir(model_id_or_path)?;
    let config = load_config_json(&model_dir)?;
    let weights_paths = ensure_weights_files(model_id_or_path, &model_dir)?;

    let weight_refs: Vec<&Path> = weights_paths.iter().map(PathBuf::as_path).collect();
    let vb = unsafe {
        var_builder_from_safetensors(
            weight_refs.as_slice(),
            opts.dtype,
            device,
            opts.isq.is_some(),
        )?
    };
    let thinker = super::thinker::ThinkerForConditionalGeneration::load(
        &config.thinker_config,
        vb.pp("thinker"),
        device,
        opts.use_flash_attn,
        opts.isq.as_deref(),
    )?;

    Ok((
        config,
        super::LoadedQwen3Asr {
            model_dir,
            weights_paths,
            thinker,
        },
    ))
}

unsafe fn var_builder_from_safetensors(
    paths: &[&Path],
    dtype: DType,
    device: &Device,
    stage_on_cpu: bool,
) -> candle_core::Result<VarBuilder<'static>> {
    if device.is_metal() || (device.is_cuda() && stage_on_cpu) {
        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };
        let backend = CpuStagedSafeTensors {
            tensors,
            stage_isq_linears_on_cpu: device.is_cuda() && stage_on_cpu,
        };
        Ok(VarBuilder::from_backend(
            Box::new(backend),
            dtype,
            device.clone(),
        ))
    } else {
        unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
    }
}

struct CpuStagedSafeTensors {
    tensors: candle_core::safetensors::MmapedSafetensors,
    stage_isq_linears_on_cpu: bool,
}

impl CpuStagedSafeTensors {
    fn load_tensor(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.tensors.load(name, &Device::Cpu)?.to_dtype(dtype)?;
        if self.stage_isq_linears_on_cpu && is_isq_linear_tensor(name) {
            Ok(tensor)
        } else {
            tensor.to_device(device)
        }
    }
}

fn is_isq_linear_tensor(name: &str) -> bool {
    let name = name.trim_end_matches(".weight").trim_end_matches(".bias");
    name.ends_with(".q_proj")
        || name.ends_with(".k_proj")
        || name.ends_with(".v_proj")
        || name.ends_with(".o_proj")
        || name.ends_with(".out_proj")
        || name.ends_with(".gate_proj")
        || name.ends_with(".up_proj")
        || name.ends_with(".down_proj")
        || name.ends_with(".lm_head")
        || name.ends_with(".fc1")
        || name.ends_with(".fc2")
        || name.ends_with(".conv_out")
        || name.ends_with(".proj1")
        || name.ends_with(".proj2")
}

impl SimpleBackend for CpuStagedSafeTensors {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _hint: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load_tensor(name, dtype, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!(
                "shape mismatch for {name}: expected={shape:?}, got={:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.get(name).is_ok()
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        self.load_tensor(name, dtype, device)
    }
}

#[cfg(not(feature = "flash-attn"))]
fn validate_load_options(_device: &Device, opts: &LoadOptions) -> Result<()> {
    if opts.use_flash_attn {
        bail!(
            "flash-attn was requested but is not enabled in this build. Rebuild with `--features flash-attn`."
        );
    }
    Ok(())
}

#[cfg(feature = "flash-attn")]
fn validate_load_options(device: &Device, opts: &LoadOptions) -> Result<()> {
    if !opts.use_flash_attn {
        return Ok(());
    }

    if !device.is_cuda() {
        bail!(
            "flash-attn requires a CUDA device, got {:?}",
            device.location()
        );
    }

    match opts.dtype {
        candle_core::DType::F16 | candle_core::DType::BF16 => {}
        other => bail!("flash-attn requires dtype f16/bf16, got {other:?}"),
    }

    Ok(())
}

fn block_on_async<F: std::future::Future + Send>(future: F) -> F::Output
where
    F::Output: Send,
{
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
                rt.block_on(future)
            })
            .join()
            .expect("tokio download thread panicked")
    })
}

fn modelscope_download(repo_id: &str) -> Result<PathBuf> {
    let cache = crate::modelscope_cache_dir();
    let target = cache.join(repo_id);
    if qwen3_asr_local_cache_complete(&target)? {
        tracing::info!(
            "Qwen3-ASR model `{repo_id}` cache is complete at {}, using local files",
            target.display()
        );
        return Ok(target);
    }

    if target.exists() {
        tracing::warn!(
            "Qwen3-ASR model `{repo_id}` cache is incomplete at {}, downloading/repairing from ModelScope",
            target.display()
        );
    } else {
        tracing::info!("Downloading model `{repo_id}` to {}", cache.display());
    }
    block_on_async(crate::download::download_model(repo_id, &cache))
        .with_context(|| format!("failed to download model {repo_id:?} from ModelScope"))?;
    validate_qwen3_asr_local_model_dir(&target)?;
    Ok(target)
}

fn resolve_model_dir(model_id_or_path: &str) -> Result<PathBuf> {
    let path = Path::new(model_id_or_path);
    if path.exists() {
        validate_qwen3_asr_local_model_dir(path)
            .with_context(|| format!("invalid local Qwen3-ASR model directory {path:?}"))?;
        return Ok(path.to_path_buf());
    }

    modelscope_download(model_id_or_path)
        .with_context(|| format!("failed to resolve model directory for {model_id_or_path:?}"))
}

fn load_config_json(model_dir: &Path) -> Result<AsrConfig> {
    let config_path = model_dir.join("config.json");
    let data = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read config.json from {config_path:?}"))?;
    let cfg: AsrConfig = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse config.json from {config_path:?}"))?;
    Ok(cfg)
}

fn qwen3_asr_local_cache_complete(model_dir: &Path) -> Result<bool> {
    if !model_dir.is_dir() {
        return Ok(false);
    }

    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        return Ok(false);
    }
    load_config_json(model_dir)?;

    if !qwen3_asr_tokenizer_files_present(model_dir) {
        return Ok(false);
    }

    qwen3_asr_weight_files_complete(model_dir)
}

fn validate_qwen3_asr_local_model_dir(model_dir: &Path) -> Result<()> {
    if !model_dir.is_dir() {
        bail!("model path is not a directory: {model_dir:?}");
    }

    load_config_json(model_dir)?;
    validate_qwen3_asr_tokenizer_files(model_dir)?;
    validate_qwen3_asr_weight_files(model_dir)?;
    Ok(())
}

fn validate_qwen3_asr_tokenizer_files(model_dir: &Path) -> Result<()> {
    if qwen3_asr_tokenizer_files_present(model_dir) {
        return Ok(());
    }
    bail!("missing tokenizer files in {model_dir:?}: expected tokenizer.json or vocab.json");
}

fn qwen3_asr_tokenizer_files_present(model_dir: &Path) -> bool {
    model_dir.join("tokenizer.json").is_file() || model_dir.join("vocab.json").is_file()
}

fn validate_qwen3_asr_weight_files(model_dir: &Path) -> Result<()> {
    if qwen3_asr_weight_files_complete(model_dir)? {
        return Ok(());
    }
    bail!("missing or empty safetensors weights in {model_dir:?}");
}

fn qwen3_asr_weight_files_complete(model_dir: &Path) -> Result<bool> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let shards = shard_filenames_from_index(&index_path)?;
        for shard in shards {
            if !nonempty_file(&model_dir.join(shard))? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    nonempty_file(&model_dir.join("model.safetensors"))
}

fn nonempty_file(path: &Path) -> Result<bool> {
    let meta = match fs::metadata(path) {
        Ok(meta) => meta,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(err).with_context(|| format!("failed to stat file {path:?}")),
    };
    if !meta.is_file() {
        return Ok(false);
    }
    Ok(meta.len() > 0)
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    #[serde(default)]
    metadata: SafetensorsIndexMetadata,
    weight_map: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
struct SafetensorsIndexMetadata {
    total_size: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct WeightsMemoryStats {
    pub checkpoint_bytes: usize,
    pub parameter_bytes: usize,
}

pub fn weights_memory_stats(paths: &[PathBuf], load_dtype: DType) -> Result<WeightsMemoryStats> {
    let checkpoint_bytes = checkpoint_weight_bytes(paths)?;
    let source_dtype_bytes = infer_checkpoint_dtype_bytes(paths)?;
    let parameter_bytes = checkpoint_bytes
        .checked_div(source_dtype_bytes)
        .and_then(|params| params.checked_mul(load_dtype.size_in_bytes()))
        .ok_or_else(|| anyhow::anyhow!("failed to calculate loaded parameter bytes"))?;
    Ok(WeightsMemoryStats {
        checkpoint_bytes,
        parameter_bytes,
    })
}

fn checkpoint_weight_bytes(paths: &[PathBuf]) -> Result<usize> {
    if let Some(index_path) = paths
        .first()
        .and_then(|path| path.parent())
        .map(|dir| dir.join("model.safetensors.index.json"))
        .filter(|path| path.exists())
    {
        let data = fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read safetensors index from {index_path:?}"))?;
        let idx: SafetensorsIndex = serde_json::from_str(&data)
            .with_context(|| format!("failed to parse safetensors index from {index_path:?}"))?;
        if let Some(total_size) = idx.metadata.total_size {
            return Ok(total_size);
        }
    }

    let mut total = 0usize;
    for path in paths {
        let len = fs::metadata(path)
            .with_context(|| format!("failed to stat safetensors file {path:?}"))?
            .len();
        total = total
            .checked_add(len as usize)
            .ok_or_else(|| anyhow::anyhow!("safetensors checkpoint size overflows usize"))?;
    }
    Ok(total)
}

fn infer_checkpoint_dtype_bytes(paths: &[PathBuf]) -> Result<usize> {
    let Some(path) = paths.first() else {
        bail!("no safetensors files to inspect");
    };
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to read safetensors file header from {path:?}"))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .with_context(|| format!("failed to read safetensors header length from {path:?}"))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)
        .with_context(|| format!("failed to read safetensors header from {path:?}"))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header)
        .with_context(|| format!("failed to parse safetensors header from {path:?}"))?;
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = value
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor {name:?} has no dtype"))?;
        return safetensors_dtype_size(dtype)
            .ok_or_else(|| anyhow::anyhow!("unsupported safetensors dtype {dtype:?}"));
    }
    bail!("safetensors file {path:?} has no tensors")
}

fn safetensors_dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" => Some(1),
        "I16" | "U16" | "F16" | "BF16" => Some(2),
        "I32" | "U32" | "F32" => Some(4),
        "C64" | "F64" | "I64" | "U64" => Some(8),
        "F4" | "F6_E2M3" | "F6_E3M2" => Some(1),
        _ => None,
    }
}

fn shard_filenames_from_index(index_path: &Path) -> Result<Vec<String>> {
    let data = std::fs::read_to_string(index_path)
        .with_context(|| format!("failed to read safetensors index from {index_path:?}"))?;
    let idx: SafetensorsIndex = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse safetensors index from {index_path:?}"))?;

    let mut shards: BTreeSet<String> = BTreeSet::new();
    for (param, filename) in idx.weight_map {
        let fname = filename.as_str().ok_or_else(|| {
            anyhow::anyhow!(
                "safetensors index weight_map value must be a string for key {param:?}: {index_path:?}"
            )
        })?;
        if fname.trim().is_empty() {
            bail!("safetensors index contains empty shard filename for key {param:?}");
        }
        shards.insert(fname.to_string());
    }

    if shards.is_empty() {
        bail!("safetensors index contains no shards: {index_path:?}");
    }

    Ok(shards.into_iter().collect())
}

fn shard_paths_from_local_index(model_dir: &Path, index_path: &Path) -> Result<Vec<PathBuf>> {
    let filenames = shard_filenames_from_index(index_path)?;
    let mut out: Vec<PathBuf> = Vec::with_capacity(filenames.len());
    for fname in filenames {
        let p = model_dir.join(fname);
        if !p.exists() {
            bail!("missing sharded safetensors file {p:?} referenced by {index_path:?}");
        }
        out.push(p);
    }
    Ok(out)
}

fn ensure_weights_files(_model_id_or_path: &str, model_dir: &Path) -> Result<Vec<PathBuf>> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        return shard_paths_from_local_index(model_dir, &index_path);
    }

    bail!("no model.safetensors or model.safetensors.index.json found in {model_dir:?}");
}

#[cfg(test)]
mod tests {
    use super::{is_isq_linear_tensor, qwen3_asr_local_cache_complete, shard_filenames_from_index};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_tmp_dir(prefix: &str) -> anyhow::Result<PathBuf> {
        let mut dir = std::env::temp_dir();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| anyhow::anyhow!("system time before unix epoch: {e:?}"))?;
        dir.push(format!("{prefix}_{}", now.as_nanos()));
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    fn write_minimal_local_model(
        dir: &std::path::Path,
        tokenizer: bool,
        sharded: bool,
    ) -> anyhow::Result<()> {
        fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "model_type": "qwen3_asr",
                "thinker_config": {
                    "model_type": "qwen3_asr_thinker",
                    "audio_config": {},
                    "text_config": {}
                }
            })
            .to_string(),
        )?;
        if tokenizer {
            fs::write(dir.join("tokenizer.json"), "{}")?;
        }
        if sharded {
            fs::write(
                dir.join("model.safetensors.index.json"),
                serde_json::json!({
                    "weight_map": {
                        "thinker.lm_head.weight": "model-00001-of-00001.safetensors"
                    }
                })
                .to_string(),
            )?;
            fs::write(dir.join("model-00001-of-00001.safetensors"), [1u8])?;
        } else {
            fs::write(dir.join("model.safetensors"), [1u8])?;
        }
        Ok(())
    }

    #[test]
    fn test_shard_filenames_from_index_dedupes_and_sorts() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_shard_index")?;
        let index_path = dir.join("model.safetensors.index.json");

        let index = serde_json::json!({
            "weight_map": {
                "a": "model-00002-of-00002.safetensors",
                "b": "model-00001-of-00002.safetensors",
                "c": "model-00001-of-00002.safetensors"
            }
        });
        fs::write(&index_path, index.to_string())?;

        let got = shard_filenames_from_index(&index_path)?;
        let want = vec![
            "model-00001-of-00002.safetensors".to_string(),
            "model-00002-of-00002.safetensors".to_string(),
        ];
        if got != want {
            anyhow::bail!("unexpected shard list: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_isq_linear_tensor_matches_all_qwen3_asr_linear_targets() {
        for name in [
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight",
            "thinker.audio_tower.layers.0.self_attn.k_proj.weight",
            "thinker.audio_tower.layers.0.self_attn.v_proj.weight",
            "thinker.audio_tower.layers.0.self_attn.out_proj.weight",
            "thinker.audio_tower.layers.0.fc1.weight",
            "thinker.audio_tower.layers.0.fc2.weight",
            "thinker.audio_tower.conv_out.weight",
            "thinker.audio_tower.proj1.weight",
            "thinker.audio_tower.proj2.weight",
            "thinker.model.layers.0.self_attn.q_proj.weight",
            "thinker.model.layers.0.self_attn.k_proj.weight",
            "thinker.model.layers.0.self_attn.v_proj.weight",
            "thinker.model.layers.0.self_attn.o_proj.weight",
            "thinker.model.layers.0.mlp.gate_proj.weight",
            "thinker.model.layers.0.mlp.up_proj.weight",
            "thinker.model.layers.0.mlp.down_proj.weight",
            "thinker.lm_head.weight",
        ] {
            assert!(
                is_isq_linear_tensor(name),
                "{name} should be staged for ISQ"
            );
        }

        for name in [
            "thinker.model.embed_tokens.weight",
            "thinker.model.layers.0.input_layernorm.weight",
            "thinker.audio_tower.layers.0.self_attn_layer_norm.weight",
        ] {
            assert!(
                !is_isq_linear_tensor(name),
                "{name} should not be staged as an ISQ linear"
            );
        }
    }

    #[test]
    fn test_ensure_weights_files_local_single_file() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_weights_single")?;
        let weights_path = dir.join("model.safetensors");
        fs::write(&weights_path, [])?;

        let dir_s = dir.to_string_lossy();
        let got = super::ensure_weights_files(dir_s.as_ref(), &dir)?;
        let want = vec![weights_path];
        if got != want {
            anyhow::bail!("unexpected weights paths: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn test_ensure_weights_files_local_sharded_index() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_weights_sharded")?;
        let index_path = dir.join("model.safetensors.index.json");

        let index = serde_json::json!({
            "weight_map": {
                "a": "model-00002-of-00002.safetensors",
                "b": "model-00001-of-00002.safetensors"
            }
        });
        fs::write(&index_path, index.to_string())?;

        let shard1 = dir.join("model-00001-of-00002.safetensors");
        let shard2 = dir.join("model-00002-of-00002.safetensors");
        fs::write(&shard1, [])?;
        fs::write(&shard2, [])?;

        let dir_s = dir.to_string_lossy();
        let got = super::ensure_weights_files(dir_s.as_ref(), &dir)?;
        let want = vec![shard1, shard2];
        if got != want {
            anyhow::bail!("unexpected weights paths: got={got:?} want={want:?}");
        }
        Ok(())
    }

    #[test]
    fn qwen3_asr_local_cache_complete_accepts_single_file_model() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_complete_single")?;
        write_minimal_local_model(&dir, true, false)?;

        assert!(qwen3_asr_local_cache_complete(&dir)?);
        Ok(())
    }

    #[test]
    fn qwen3_asr_local_cache_complete_accepts_sharded_model() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_complete_sharded")?;
        write_minimal_local_model(&dir, true, true)?;

        assert!(qwen3_asr_local_cache_complete(&dir)?);
        Ok(())
    }

    #[test]
    fn qwen3_asr_local_cache_complete_rejects_missing_tokenizer() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_missing_tokenizer")?;
        write_minimal_local_model(&dir, false, false)?;

        assert!(!qwen3_asr_local_cache_complete(&dir)?);
        Ok(())
    }

    #[test]
    fn qwen3_asr_local_cache_complete_rejects_empty_weights() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_empty_weights")?;
        write_minimal_local_model(&dir, true, false)?;
        fs::write(dir.join("model.safetensors"), [])?;

        assert!(!qwen3_asr_local_cache_complete(&dir)?);
        Ok(())
    }

    #[test]
    fn qwen3_asr_local_cache_complete_errors_on_invalid_config() -> anyhow::Result<()> {
        let dir = unique_tmp_dir("qwen3_asr_invalid_config")?;
        write_minimal_local_model(&dir, true, false)?;
        fs::write(dir.join("config.json"), "{not-json")?;

        let err = qwen3_asr_local_cache_complete(&dir)
            .expect_err("invalid config should not be treated as an incomplete cache");
        assert!(err.to_string().contains("failed to parse config.json"));
        Ok(())
    }
}
