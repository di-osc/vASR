//! Forced aligner model wrapper.
//!
//! This implements the core inference loop used by the official Python stack:
//! `../../../Qwen3-ASR/qwen_asr/inference/qwen3_forced_aligner.py`.

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use std::collections::BTreeSet;

use crate::aligner::AlignerModel;
use crate::asr::qwen3asr;
use crate::audio::input::AudioInput;
use qwen3asr::config::AsrConfig;
use qwen3asr::model::{AsrModel, weights::LoadOptions};
use qwen3asr::processor::AsrProcessor;
use qwen3asr::processor::asr_processor::PreparedInputs;
use vasr_data::Waveform;

use super::processor::ForcedAlignProcessor;

#[derive(Debug, Clone, PartialEq)]
pub struct ForcedAlignItem {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}

#[derive(Debug, Clone)]
pub struct ForcedAlignResult {
    pub items: Vec<ForcedAlignItem>,
}

#[derive(Debug)]
pub struct Qwen3ForcedAligner {
    device: Device,
    config: AsrConfig,
    processor: AsrProcessor,
    model: AsrModel,

    aligner_processor: ForcedAlignProcessor,
    timestamp_token_id: u32,
    timestamp_segment_time_ms: f32,
}

impl Qwen3ForcedAligner {
    pub fn from_pretrained(
        model_id_or_path: &str,
        device: &Device,
        opts: &LoadOptions,
    ) -> Result<Self> {
        Self::from_pretrained_with_processor(
            model_id_or_path,
            device,
            opts,
            ForcedAlignProcessor::new(),
        )
    }

    pub fn from_pretrained_with_processor(
        model_id_or_path: &str,
        device: &Device,
        opts: &LoadOptions,
        aligner_processor: ForcedAlignProcessor,
    ) -> Result<Self> {
        let (config, model) =
            qwen3asr::model::weights::load_model_from_pretrained(model_id_or_path, device, opts)?;

        let thinker_type = config
            .thinker_config
            .model_type
            .as_deref()
            .unwrap_or_default();
        if !thinker_type.contains("forced_aligner") {
            bail!("model is not a forced aligner: thinker_config.model_type={thinker_type:?}");
        }

        let timestamp_token_id = config
            .timestamp_token_id
            .context("config.json missing timestamp_token_id")?;
        let timestamp_segment_time = config
            .timestamp_segment_time
            .context("config.json missing timestamp_segment_time")?;
        if timestamp_segment_time <= 0.0 {
            bail!("timestamp_segment_time must be > 0, got {timestamp_segment_time}");
        }

        let tokenizer =
            qwen3asr::processor::tokenizer::Tokenizer::from_pretrained(model_id_or_path)?;
        let processor = AsrProcessor::new(tokenizer);

        Ok(Self {
            device: device.clone(),
            config,
            processor,
            model,
            aligner_processor,
            timestamp_token_id,
            timestamp_segment_time_ms: timestamp_segment_time as f32,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &AsrConfig {
        &self.config
    }

    pub fn processor(&self) -> &AsrProcessor {
        &self.processor
    }

    pub fn require_ready(&self) -> Result<()> {
        self.processor.require_ready()
    }

    /// Return supported language names, if exposed by the checkpoint/config.
    ///
    /// This mirrors the official Python wrapper which returns a sorted, lowercased list,
    /// or `None` if the model does not provide language constraints.
    pub fn get_supported_languages(&self) -> Option<Vec<String>> {
        supported_languages_lowercase_sorted(self.config.support_languages.as_slice())
    }

    pub fn align(
        &self,
        waveforms: &[Waveform],
        text: &[String],
        language: &[String],
    ) -> Result<Vec<ForcedAlignResult>> {
        let audio = waveforms
            .iter()
            .map(|waveform| AudioInput::Waveform {
                samples: &waveform.samples,
                sample_rate: waveform.sample_rate,
            })
            .collect::<Vec<_>>();
        self.align_inputs(audio.as_slice(), text, language)
    }

    fn align_inputs(
        &self,
        audio: &[AudioInput<'_>],
        text: &[String],
        language: &[String],
    ) -> Result<Vec<ForcedAlignResult>> {
        if audio.is_empty() {
            return Ok(vec![]);
        }
        if text.len() != audio.len() {
            bail!(
                "batch size mismatch: audio={} text={}",
                audio.len(),
                text.len()
            );
        }

        let languages: Vec<String> = if language.len() == 1 && audio.len() > 1 {
            vec![language[0].clone(); audio.len()]
        } else {
            language.to_vec()
        };
        if languages.len() != audio.len() {
            bail!(
                "batch size mismatch: audio={} language={}",
                audio.len(),
                languages.len()
            );
        }

        // Build per-item prompt and keep the word_list for later parsing.
        let mut prompts: Vec<String> = Vec::with_capacity(audio.len());
        let mut word_lists: Vec<Vec<String>> = Vec::with_capacity(audio.len());
        for (t, lang) in text.iter().zip(languages.iter()) {
            let (words, prompt) = self.aligner_processor.encode_timestamp(t, lang)?;
            word_lists.push(words);
            prompts.push(prompt);
        }

        // Prepare batch inputs (tokenization + audio features + placeholder expansion).
        let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(audio.len());
        for (p, a) in prompts.iter().zip(audio.iter()) {
            items.push((p.as_str(), a));
        }
        let prepared = self.processor.prepare_batch(items.as_slice())?;

        let audio_token_id = self.model.thinker.audio_token_id();
        let mut audio_placeholder_count: usize = 0;
        for (i, p) in prepared.iter().enumerate() {
            let n = p
                .input_ids
                .iter()
                .filter(|&&id| id == audio_token_id)
                .count();
            audio_placeholder_count = audio_placeholder_count.checked_add(n).ok_or_else(|| {
                anyhow::anyhow!("audio placeholder count overflow at batch index {i}: adding {n}")
            })?;
        }

        let input_ids = stack_u32_2d(
            prepared
                .iter()
                .map(|p| p.input_ids.as_slice())
                .collect::<Vec<_>>()
                .as_slice(),
            "input_ids",
            &self.device,
        )?;
        let attention_mask = stack_u32_2d(
            prepared
                .iter()
                .map(|p| p.attention_mask.as_slice())
                .collect::<Vec<_>>()
                .as_slice(),
            "attention_mask",
            &self.device,
        )?;
        let (input_features, feature_lens) = stack_features(&prepared, &self.device)?;
        let audio_features = self
            .model
            .thinker
            .get_audio_features_with_lens(&input_features, feature_lens.as_slice())?;

        let logits = self.model.thinker.forward_with_audio_features(
            &input_ids,
            &attention_mask,
            Some(&audio_features),
            audio_placeholder_count,
        )?;
        let pred_ids = logits.argmax(2usize)?;
        let pred_ids = pred_ids.to_vec2::<u32>()?;

        let input_ids_vec = prepared
            .iter()
            .map(|p| p.input_ids.as_slice())
            .collect::<Vec<_>>();

        let mut out: Vec<ForcedAlignResult> = Vec::with_capacity(audio.len());
        for ((ids, sample_pred), words) in input_ids_vec
            .into_iter()
            .zip(pred_ids.into_iter())
            .zip(word_lists.into_iter())
        {
            let ts_positions = ids
                .iter()
                .enumerate()
                .filter_map(|(i, &tok)| (tok == self.timestamp_token_id).then_some(i))
                .collect::<Vec<_>>();

            let expected = words
                .len()
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("timestamp count overflow"))?;
            if ts_positions.len() != expected {
                bail!(
                    "timestamp token count mismatch: expected={} got={} timestamp_token_id={}",
                    expected,
                    ts_positions.len(),
                    self.timestamp_token_id
                );
            }

            let mut timestamp_ms: Vec<f32> = Vec::with_capacity(ts_positions.len());
            for &pos in &ts_positions {
                let class_id = sample_pred
                    .get(pos)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("pred_ids missing position {pos}"))?;
                timestamp_ms.push(class_id as f32 * self.timestamp_segment_time_ms);
            }

            let items = self
                .aligner_processor
                .parse_timestamp(&words, &timestamp_ms)?;
            out.push(ForcedAlignResult {
                items: items
                    .into_iter()
                    .map(|it| ForcedAlignItem {
                        text: it.text,
                        start_time: round3(it.start_time_ms as f64 / 1000.0),
                        end_time: round3(it.end_time_ms as f64 / 1000.0),
                    })
                    .collect(),
            });
        }

        Ok(out)
    }
}

impl AlignerModel for Qwen3ForcedAligner {
    fn align(
        &self,
        waveforms: &[Waveform],
        text: &[String],
        language: &[String],
    ) -> Result<Vec<ForcedAlignResult>> {
        Qwen3ForcedAligner::align(self, waveforms, text, language)
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn supported_languages_lowercase_sorted(langs: &[String]) -> Option<Vec<String>> {
    if langs.is_empty() {
        return None;
    }

    let mut set: BTreeSet<String> = BTreeSet::new();
    for x in langs {
        let s = x.trim();
        if s.is_empty() {
            continue;
        }
        set.insert(s.to_ascii_lowercase());
    }

    if set.is_empty() {
        return None;
    }

    Some(set.into_iter().collect())
}

fn stack_u32_2d(rows: &[&[u32]], what: &'static str, device: &Device) -> Result<Tensor> {
    let batch = rows.len();
    if batch == 0 {
        bail!("{what} is empty");
    }
    let seq = rows
        .first()
        .map(|r| r.len())
        .ok_or_else(|| anyhow::anyhow!("{what} missing first row"))?;
    if seq == 0 {
        bail!("{what} rows are empty");
    }

    let mut flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq));
    for (i, row) in rows.iter().enumerate() {
        if row.len() != seq {
            bail!(
                "{what}[{i}] length mismatch: expected={seq}, got={}",
                row.len()
            );
        }
        flat.extend_from_slice(row);
    }

    Ok(Tensor::from_vec(flat, (batch, seq), device)?)
}

fn stack_features(prepared: &[PreparedInputs], device: &Device) -> Result<(Tensor, Vec<usize>)> {
    let batch = prepared.len();
    if batch == 0 {
        bail!("prepared is empty");
    }

    let mel = prepared
        .first()
        .map(|p| p.input_features.len())
        .ok_or_else(|| anyhow::anyhow!("prepared missing first item"))?;
    if mel == 0 {
        bail!("prepared input_features has zero mel bins");
    }
    let frames = prepared
        .first()
        .and_then(|p| p.input_features.first())
        .map(|r| r.len())
        .ok_or_else(|| anyhow::anyhow!("prepared input_features missing first row"))?;
    if frames == 0 {
        bail!("prepared input_features has zero frames");
    }

    let mut feats: Vec<f32> = Vec::with_capacity(batch.saturating_mul(mel).saturating_mul(frames));
    let mut feature_lens: Vec<usize> = Vec::with_capacity(batch);

    for (i, p) in prepared.iter().enumerate() {
        if p.input_features.len() != mel {
            bail!(
                "prepared[{i}].input_features mel mismatch: expected={mel}, got={}",
                p.input_features.len()
            );
        }
        if p.feature_attention_mask.len() != frames {
            bail!(
                "prepared[{i}].feature_attention_mask len mismatch: expected={frames}, got={}",
                p.feature_attention_mask.len()
            );
        }
        for row in &p.input_features {
            if row.len() != frames {
                bail!(
                    "prepared[{i}].input_features row len mismatch: expected={frames}, got={}",
                    row.len()
                );
            }
            feats.extend_from_slice(row);
        }
        feature_lens.push(p.feature_attention_mask.iter().filter(|&&x| x != 0).count());
    }

    let input_features = Tensor::from_vec(feats, (batch, mel, frames), device)?;
    Ok((input_features, feature_lens))
}

#[cfg(test)]
mod tests {
    use super::{ForcedAlignResult, supported_languages_lowercase_sorted};

    #[test]
    fn test_forced_align_result_is_cloneable() {
        let r = ForcedAlignResult { items: vec![] };
        let _ = r.clone();
    }

    #[test]
    fn test_supported_languages_lowercase_sorted_dedupes_and_sorts() -> anyhow::Result<()> {
        let langs = vec![
            "English".to_string(),
            "chinese".to_string(),
            "English".to_string(),
            " ".to_string(),
        ];
        let got = supported_languages_lowercase_sorted(&langs)
            .ok_or_else(|| anyhow::anyhow!("expected Some languages"))?;
        if got != vec!["chinese".to_string(), "english".to_string()] {
            anyhow::bail!("unexpected languages: {got:?}");
        }
        Ok(())
    }
}
