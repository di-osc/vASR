use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CONFIG_TOML: &str = r#"# vASR realtime 2-pass configuration
version = 1

[service]
host = "0.0.0.0"
port = 8000

# Realtime speech boundary detector. FireRed Stream-VAD is currently the
# supported online VAD backend.
[models.vad]
kind = "firered"
# checkpoint = "/path/to/FireRedVAD"
device = "cpu"
threshold = 0.5
min_speech_ms = 80
min_silence_ms = 500
max_segment_ms = 20000
prefix_padding_ms = 50

# First pass: low-latency partial transcripts.
[models.online]
kind = "sensevoice-online"
checkpoint = "iic/SenseVoiceSmall"
device = "cuda"
language = "zh"
chunk_size = 4
max_history = 0
textnorm = false

# Second pass: whole-utterance final transcript after VAD speech end.
[models.offline]
kind = "qwen3-asr"
checkpoint = "Qwen/Qwen3-ASR-0.6B"
device = "cuda"
# isq = "q8_0"
# language = "Chinese"
gpu_memory_utilization = 0.65
cuda_graph_max_seqs = 512
"#;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VasrConfig {
    pub version: u32,
    pub service: ServiceConfig,
    pub models: ModelsConfig,
}

impl Default for VasrConfig {
    fn default() -> Self {
        Self {
            version: 1,
            service: ServiceConfig::default(),
            models: ModelsConfig::default(),
        }
    }
}

impl VasrConfig {
    pub fn from_toml_str(value: &str) -> Result<Self> {
        toml::from_str(value).context("failed to parse vASR TOML configuration")
    }

    pub fn load(path: &Path) -> Result<Self> {
        let value = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        Self::from_toml_str(&value).with_context(|| format!("invalid config {}", path.display()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ServiceConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelsConfig {
    pub vad: VadModelConfig,
    pub online: AsrModelConfig,
    pub offline: AsrModelConfig,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            vad: VadModelConfig::default(),
            online: AsrModelConfig::sensevoice_online_default(),
            offline: AsrModelConfig::qwen3_offline_default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VadModelConfig {
    pub kind: String,
    pub checkpoint: Option<String>,
    pub device: String,
    pub threshold: f32,
    pub min_speech_ms: u64,
    pub min_silence_ms: u64,
    pub max_segment_ms: u64,
    pub prefix_padding_ms: u64,
    pub num_threads: usize,
}

impl Default for VadModelConfig {
    fn default() -> Self {
        Self {
            kind: "firered".to_string(),
            checkpoint: None,
            device: "cpu".to_string(),
            threshold: 0.5,
            min_speech_ms: 80,
            min_silence_ms: 500,
            max_segment_ms: 20_000,
            prefix_padding_ms: 50,
            num_threads: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AsrModelConfig {
    pub kind: String,
    pub checkpoint: String,
    pub device: String,
    pub isq: Option<String>,
    pub language: Option<String>,
    pub gpu_memory_utilization: f32,
    pub cuda_graph_max_seqs: usize,
    pub chunk_size: usize,
    pub max_history: usize,
    pub textnorm: bool,
}

impl AsrModelConfig {
    pub fn sensevoice_online_default() -> Self {
        Self {
            kind: "sensevoice-online".to_string(),
            checkpoint: "iic/SenseVoiceSmall".to_string(),
            device: "cuda".to_string(),
            isq: None,
            language: Some("zh".to_string()),
            gpu_memory_utilization: 0.65,
            cuda_graph_max_seqs: 0,
            chunk_size: 4,
            max_history: 0,
            textnorm: false,
        }
    }

    pub fn qwen3_offline_default() -> Self {
        Self {
            kind: "qwen3-asr".to_string(),
            checkpoint: "Qwen/Qwen3-ASR-0.6B".to_string(),
            device: "cuda".to_string(),
            isq: None,
            language: None,
            gpu_memory_utilization: 0.65,
            cuda_graph_max_seqs: 512,
            chunk_size: 4,
            max_history: 0,
            textnorm: false,
        }
    }
}

impl Default for AsrModelConfig {
    fn default() -> Self {
        Self::qwen3_offline_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_template_parses() {
        let config = VasrConfig::from_toml_str(DEFAULT_CONFIG_TOML).unwrap();
        assert_eq!(config.models.vad.kind, "firered");
        assert_eq!(config.models.online.kind, "sensevoice-online");
        assert_eq!(config.models.offline.kind, "qwen3-asr");
        assert_eq!(config.models.online.chunk_size, 4);
    }
}
