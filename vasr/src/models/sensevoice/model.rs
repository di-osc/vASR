use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder, conv1d_no_bias,
    embedding, layer_norm, linear, ops::softmax_last_dim,
};

use super::tokenizer::SenseVoiceTokenizer;

const INPUT_SIZE: usize = 560;
const OUTPUT_SIZE: usize = 512;
const ATTENTION_HEADS: usize = 4;
const HEAD_DIM: usize = OUTPUT_SIZE / ATTENTION_HEADS;
const LINEAR_UNITS: usize = 2048;
const ENCODER_BLOCKS: usize = 50;
const TP_BLOCKS: usize = 20;
const SANM_KERNEL_SIZE: usize = 11;
const QUERY_FRAMES: usize = 4;
const EMBEDDING_SIZE: usize = 16;

pub struct SenseVoiceModel {
    first_layer: EncoderLayer,
    layers: Vec<EncoderLayer>,
    after_norm: LayerNorm,
    tp_layers: Vec<EncoderLayer>,
    tp_norm: LayerNorm,
    ctc: Linear,
    embedding: Embedding,
    dtype: DType,
}

pub struct SenseVoiceDecode {
    pub text: String,
    #[allow(dead_code)]
    pub language: Option<String>,
    #[allow(dead_code)]
    pub emotion: Option<String>,
}

impl SenseVoiceModel {
    pub fn load(
        model_path: &Path,
        vocab_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_pth(model_path, dtype, device).with_context(|| {
            format!("failed to load SenseVoice weights {}", model_path.display())
        })?;
        let encoder = vb.pp("encoder");
        let first_layer = EncoderLayer::load(INPUT_SIZE, encoder.pp("encoders0").pp(0))?;
        let mut layers = Vec::with_capacity(ENCODER_BLOCKS - 1);
        for index in 0..ENCODER_BLOCKS - 1 {
            layers.push(EncoderLayer::load(
                OUTPUT_SIZE,
                encoder.pp("encoders").pp(index),
            )?);
        }
        let after_norm = layer_norm(OUTPUT_SIZE, 1e-5, encoder.pp("after_norm"))?;
        let mut tp_layers = Vec::with_capacity(TP_BLOCKS);
        for index in 0..TP_BLOCKS {
            tp_layers.push(EncoderLayer::load(
                OUTPUT_SIZE,
                encoder.pp("tp_encoders").pp(index),
            )?);
        }
        let tp_norm = layer_norm(OUTPUT_SIZE, 1e-5, encoder.pp("tp_norm"))?;
        let ctc = linear(OUTPUT_SIZE, vocab_size, vb.pp("ctc").pp("ctc_lo"))?;
        let embedding = embedding(EMBEDDING_SIZE, INPUT_SIZE, vb.pp("embed"))?;
        Ok(Self {
            first_layer,
            layers,
            after_norm,
            tp_layers,
            tp_norm,
            ctc,
            embedding,
            dtype,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn transcribe(
        &self,
        features: &Tensor,
        language: &str,
        textnorm: bool,
        tokenizer: &SenseVoiceTokenizer,
    ) -> Result<SenseVoiceDecode> {
        let query_ids = [
            language_id(language),
            1u32,
            2u32,
            if textnorm { 14 } else { 15 },
        ];
        let query = Tensor::new(&query_ids, features.device())?.unsqueeze(0)?;
        let query = self.embedding.forward(&query)?.to_dtype(self.dtype)?;
        let x = Tensor::cat(&[&query, features], 1)?;
        let mut x = add_sinusoidal_position(&(x * (OUTPUT_SIZE as f64).sqrt())?)?;
        x = self.first_layer.forward(&x)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        x = self.after_norm.forward(&x)?;
        for layer in &self.tp_layers {
            x = layer.forward(&x)?;
        }
        x = self.tp_norm.forward(&x)?;
        let logits = self.ctc.forward(&x)?.squeeze(0)?.to_dtype(DType::F32)?;
        let time = logits.dim(0)?;
        if time < QUERY_FRAMES {
            bail!("SenseVoice encoder returned only {time} frames");
        }
        let best = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let rich = &best[..QUERY_FRAMES];
        let token_ids = greedy_ctc(&best[QUERY_FRAMES..]);
        Ok(SenseVoiceDecode {
            text: tokenizer.decode(&token_ids),
            language: rich.first().and_then(|id| language_from_token(*id)),
            emotion: rich.get(1).and_then(|id| emotion_from_token(*id)),
        })
    }
}

struct EncoderLayer {
    input_size: usize,
    qkv: Linear,
    output: Linear,
    fsmn: Conv1d,
    feed_forward_1: Linear,
    feed_forward_2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    fn load(input_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let attention = vb.pp("self_attn");
        let qkv = linear(input_size, OUTPUT_SIZE * 3, attention.pp("linear_q_k_v"))?;
        let output = linear(OUTPUT_SIZE, OUTPUT_SIZE, attention.pp("linear_out"))?;
        let fsmn = conv1d_no_bias(
            OUTPUT_SIZE,
            OUTPUT_SIZE,
            SANM_KERNEL_SIZE,
            Conv1dConfig {
                padding: (SANM_KERNEL_SIZE - 1) / 2,
                groups: OUTPUT_SIZE,
                ..Default::default()
            },
            attention.pp("fsmn_block"),
        )?;
        let feed_forward = vb.pp("feed_forward");
        let feed_forward_1 = linear(OUTPUT_SIZE, LINEAR_UNITS, feed_forward.pp("w_1"))?;
        let feed_forward_2 = linear(LINEAR_UNITS, OUTPUT_SIZE, feed_forward.pp("w_2"))?;
        let norm1 = layer_norm(input_size, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(OUTPUT_SIZE, 1e-5, vb.pp("norm2"))?;
        Ok(Self {
            input_size,
            qkv,
            output,
            fsmn,
            feed_forward_1,
            feed_forward_2,
            norm1,
            norm2,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let normalized = self.norm1.forward(input)?;
        let qkv = self.qkv.forward(&normalized)?;
        let query = qkv.narrow(2, 0, OUTPUT_SIZE)?;
        let key = qkv.narrow(2, OUTPUT_SIZE, OUTPUT_SIZE)?;
        let value = qkv.narrow(2, OUTPUT_SIZE * 2, OUTPUT_SIZE)?;
        let (batch, time, _) = qkv.dims3()?;

        let query = query
            .reshape((batch, time, ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let key = key
            .reshape((batch, time, ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let value_heads = value
            .reshape((batch, time, ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?;
        let query = contiguous_on_accelerator(query)?;
        let key_t = contiguous_on_accelerator(key.transpose(2, 3)?)?;
        let value_heads = contiguous_on_accelerator(value_heads)?;
        let attention = query
            .matmul(&key_t)?
            .affine(1.0 / (HEAD_DIM as f64).sqrt(), 0.0)?;
        let attention = softmax_last_dim(&attention)?.matmul(&value_heads)?;
        let attention =
            attention
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, time, OUTPUT_SIZE))?;
        let attention = self.output.forward(&attention)?;

        let memory = self
            .fsmn
            .forward(&value.transpose(1, 2)?)?
            .transpose(1, 2)?;
        let attended = (attention + (memory + value)?)?;
        let x = if self.input_size == OUTPUT_SIZE {
            (input + attended)?
        } else {
            attended
        };
        let residual = x.clone();
        let feed_forward = self
            .feed_forward_1
            .forward(&self.norm2.forward(&x)?)?
            .relu()?;
        let feed_forward = self.feed_forward_2.forward(&feed_forward)?;
        Ok((residual + feed_forward)?)
    }
}

fn contiguous_on_accelerator(tensor: Tensor) -> Result<Tensor> {
    if tensor.device().is_cuda() || tensor.device().is_metal() {
        Ok(tensor.contiguous()?)
    } else {
        Ok(tensor)
    }
}

fn add_sinusoidal_position(input: &Tensor) -> Result<Tensor> {
    let (batch, time, depth) = input.dims3()?;
    if batch != 1 || depth % 2 != 0 {
        bail!(
            "invalid SenseVoice positional input shape {:?}",
            input.dims()
        );
    }
    let half = depth / 2;
    let increment = 10_000.0f64.ln() / (half.saturating_sub(1).max(1) as f64);
    let mut values = Vec::with_capacity(time * depth);
    for position in 1..=time {
        for index in 0..half {
            let scaled = position as f64 * (-increment * index as f64).exp();
            values.push(scaled.sin() as f32);
        }
        for index in 0..half {
            let scaled = position as f64 * (-increment * index as f64).exp();
            values.push(scaled.cos() as f32);
        }
    }
    let position =
        Tensor::from_vec(values, (1, time, depth), input.device())?.to_dtype(input.dtype())?;
    Ok((input + position)?)
}

fn greedy_ctc(best_ids: &[u32]) -> Vec<u32> {
    let mut output = Vec::new();
    let mut previous = 0;
    for &token in best_ids {
        if token == 0 {
            previous = 0;
        } else if token != previous {
            output.push(token);
            previous = token;
        }
    }
    output
}

fn language_id(language: &str) -> u32 {
    match language.trim().to_ascii_lowercase().as_str() {
        "zh" => 3,
        "en" => 4,
        "yue" => 7,
        "ja" => 11,
        "ko" => 12,
        "nospeech" => 13,
        _ => 0,
    }
}

fn language_from_token(token: u32) -> Option<String> {
    match token {
        24884 => Some("zh"),
        24885 => Some("en"),
        24888 => Some("yue"),
        24892 => Some("ja"),
        24896 => Some("ko"),
        24992 => Some("nospeech"),
        _ => None,
    }
    .map(str::to_string)
}

fn emotion_from_token(token: u32) -> Option<String> {
    match token {
        25001 => Some("HAPPY"),
        25002 => Some("SAD"),
        25003 => Some("ANGRY"),
        25004 => Some("NEUTRAL"),
        25009 => Some("UNK"),
        _ => None,
    }
    .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::greedy_ctc;

    #[test]
    fn greedy_ctc_collapses_runs_and_resets_on_blank() {
        assert_eq!(greedy_ctc(&[0, 2, 2, 0, 2, 3, 3, 0]), vec![2, 2, 3]);
    }
}
