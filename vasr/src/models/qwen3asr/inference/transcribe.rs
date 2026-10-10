//! Offline transcription pipeline.

#[cfg(feature = "paged-attn")]
#[path = "metadata_decoder.rs"]
mod metadata_decoder;
#[cfg(feature = "paged-attn")]
pub use metadata_decoder::StreamingMetadataDecoder;

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

use crate::audio::SAMPLE_RATE_HZ;
use crate::audio::input::AudioInput;
#[cfg(feature = "paged-attn")]
use crate::inference::batch_scheduler::{PreparedAudioBatch, PreparedAudioBatchPoll};
use crate::inference::types::{AsrTranscription, TranscribeOptions};
use crate::inference::utils::{
    MAX_ASR_INPUT_SECONDS, merge_languages, normalize_language_name, parse_asr_output,
    validate_language,
};
use crate::model::cuda_mem_debug::log_cuda_mem;
#[cfg(feature = "timing")]
use crate::model::generation::GenerationTimings;
#[cfg(not(feature = "paged-attn"))]
use crate::model::generation::greedy_generate_cached_batch;
#[cfg(all(feature = "timing", not(feature = "paged-attn")))]
use crate::model::generation::greedy_generate_cached_batch_timed;
#[cfg(feature = "paged-attn")]
use crate::model::paged_cache_runtime::SharedPagedCacheRuntime;
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::processor::chat_template;
use crate::processor::{AsrProcessor, asr_processor::PreparedInputs};

fn duration_to_us(d: std::time::Duration) -> u64 {
    let us = d.as_micros();
    if us > u128::from(u64::MAX) {
        u64::MAX
    } else {
        us as u64
    }
}

#[cfg(feature = "timing")]
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TranscribeTimings {
    pub audio_normalize_us: u64,
    pub audio_chunking_us: u64,
    pub processor_prepare_batch_us: u64,
    pub processor_prepare_normalize_us: u64,
    pub processor_prepare_token_lookup_us: u64,
    pub processor_prepare_feature_extract_us: u64,
    pub processor_prepare_tokenize_expand_us: u64,
    pub processor_prepare_pad_us: u64,
    pub stack_features_us: u64,
    pub audio_encoder_us: u64,
    pub tokenizer_decode_us: u64,
    pub generation: GenerationTimings,
    pub total_us: u64,
    pub chunks: usize,
    pub batches: usize,
}

#[cfg(feature = "timing")]
fn add_prepare_batch_timings(
    out: &mut TranscribeTimings,
    t: &crate::processor::asr_processor::PrepareBatchTimings,
) {
    out.processor_prepare_normalize_us = out
        .processor_prepare_normalize_us
        .saturating_add(t.normalize_us);
    out.processor_prepare_token_lookup_us = out
        .processor_prepare_token_lookup_us
        .saturating_add(t.token_lookup_us);
    out.processor_prepare_feature_extract_us = out
        .processor_prepare_feature_extract_us
        .saturating_add(t.feature_extract_us);
    out.processor_prepare_tokenize_expand_us = out
        .processor_prepare_tokenize_expand_us
        .saturating_add(t.tokenize_expand_us);
    out.processor_prepare_pad_us = out.processor_prepare_pad_us.saturating_add(t.pad_us);
}

fn stack_features(prepared: &[PreparedInputs], device: &Device) -> Result<(Tensor, Vec<usize>)> {
    stack_features_inner(prepared, device, false)
}

fn stack_features_varlen(
    prepared: &[PreparedInputs],
    device: &Device,
) -> Result<(Tensor, Vec<usize>)> {
    stack_features_inner(prepared, device, true)
}

fn stack_features_inner(
    prepared: &[PreparedInputs],
    device: &Device,
    allow_variable_frames: bool,
) -> Result<(Tensor, Vec<usize>)> {
    let batch = prepared.len();
    if batch == 0 {
        bail!("prepared is empty");
    }

    let mel = if allow_variable_frames {
        prepared
            .iter()
            .map(|p| p.input_features.len())
            .max()
            .unwrap_or(0)
    } else {
        prepared
            .first()
            .map(|p| p.input_features.len())
            .ok_or_else(|| anyhow::anyhow!("prepared missing first item"))?
    };
    if mel == 0 {
        bail!("prepared input_features has zero mel bins");
    }
    let frames = if allow_variable_frames {
        prepared
            .iter()
            .flat_map(|p| p.input_features.iter().map(|row| row.len()))
            .max()
            .unwrap_or(0)
    } else {
        prepared
            .first()
            .and_then(|p| p.input_features.first())
            .map(|r| r.len())
            .ok_or_else(|| anyhow::anyhow!("prepared input_features missing first row"))?
    };

    let mut feats: Vec<f32> = Vec::with_capacity(batch.saturating_mul(mel).saturating_mul(frames));
    let mut feature_lens: Vec<usize> = Vec::with_capacity(batch);

    for (i, p) in prepared.iter().enumerate() {
        if p.input_features.len() != mel {
            if allow_variable_frames {
                bail!(
                    "prepared[{i}].input_features mel mismatch after feature padding: expected={mel}, got={}",
                    p.input_features.len()
                );
            }
            bail!(
                "prepared[{i}].input_features mel mismatch: expected={mel}, got={}",
                p.input_features.len()
            );
        }
        if !allow_variable_frames && p.feature_attention_mask.len() != frames {
            bail!(
                "prepared[{i}].feature_attention_mask len mismatch: expected={frames}, got={}",
                p.feature_attention_mask.len()
            );
        }
        for row in &p.input_features {
            if allow_variable_frames {
                if row.len() > frames {
                    bail!(
                        "prepared[{i}].input_features row len exceeds max frames: max={frames}, got={}",
                        row.len()
                    );
                }
                feats.extend_from_slice(row);
                if row.len() < frames {
                    feats.extend(std::iter::repeat_n(0.0f32, frames - row.len()));
                }
            } else if row.len() != frames {
                bail!(
                    "prepared[{i}].input_features row len mismatch: expected={frames}, got={}",
                    row.len()
                );
            } else {
                feats.extend_from_slice(row);
            }
        }
        if allow_variable_frames {
            for _ in p.input_features.len()..mel {
                feats.extend(std::iter::repeat_n(0.0f32, frames));
            }
        }
        feature_lens.push(p.feature_attention_mask.iter().filter(|&&x| x != 0).count());
    }

    let input_features = Tensor::from_vec(feats, (batch, mel, frames), device)?;
    Ok((input_features, feature_lens))
}

pub(crate) fn build_eos_token_ids(processor: &AsrProcessor) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(2);
    out.push(processor.tokenizer.token_to_id(chat_template::IM_END)?);
    out.push(processor.tokenizer.token_to_id("<|endoftext|>")?);
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

pub(crate) fn paged_token_pad_id(processor: &AsrProcessor) -> Result<u32> {
    processor.tokenizer.token_to_id("<|endoftext|>")
}

#[cfg(feature = "paged-attn")]
fn use_varlen_paged_prepare(paged_runtime: Option<&SharedPagedCacheRuntime>) -> bool {
    paged_runtime.is_some() && crate::inference::batch_scheduler::continuous_paged_batch_enabled()
}

fn prepared_prompt_len(prepared: &PreparedInputs) -> usize {
    prepared.attention_mask.iter().filter(|&&x| x != 0).count()
}

fn rebatch_prepared_inputs(
    prepared: &[PreparedInputs],
    pad_id: u32,
    decode_max_seqs: usize,
    continuous_paged: bool,
) -> Vec<Vec<(usize, PreparedInputs)>> {
    if prepared.is_empty() {
        return Vec::new();
    }

    let count_cap = if decode_max_seqs == 0 {
        prepared.len().max(1)
    } else {
        decode_max_seqs
    };

    let mut order: Vec<(usize, usize)> = prepared
        .iter()
        .enumerate()
        .map(|(idx, item)| (idx, prepared_prompt_len(item)))
        .collect();
    order.sort_unstable_by_key(|(_, len)| *len);

    let mut grouped: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut current: Vec<(usize, usize)> = Vec::new();
    let mut current_min = 0usize;
    let mut current_max = 0usize;

    for (idx, len) in order {
        let should_split = if current.is_empty() {
            false
        } else {
            let next_max = current_max.max(len);
            current.len() >= count_cap
                || next_max > current_min.saturating_mul(4).div_ceil(3)
                || next_max.saturating_sub(current_min) > 256
        };
        if should_split {
            grouped.push(current);
            current = Vec::new();
            current_min = 0;
            current_max = 0;
        }
        if current.is_empty() {
            current_min = len;
            current_max = len;
        } else {
            current_max = current_max.max(len);
        }
        current.push((idx, len));
    }
    if !current.is_empty() {
        grouped.push(current);
    }

    grouped
        .into_iter()
        .map(|group| {
            let group_max = group.iter().map(|(_, len)| *len).max().unwrap_or(0);
            group
                .into_iter()
                .map(|(idx, len)| {
                    let src = &prepared[idx];
                    let mut item = PreparedInputs {
                        input_ids: src.input_ids[..len].to_vec(),
                        attention_mask: vec![1u32; len],
                        input_features: src.input_features.clone(),
                        feature_attention_mask: src.feature_attention_mask.clone(),
                    };
                    if !continuous_paged && len < group_max {
                        item.input_ids
                            .extend(std::iter::repeat_n(pad_id, group_max - len));
                        item.attention_mask
                            .extend(std::iter::repeat_n(0u32, group_max - len));
                    }
                    (idx, item)
                })
                .collect()
        })
        .collect()
}

#[derive(Debug, Clone)]
struct EncodedPreparedGroup {
    chunk_indices: Vec<usize>,
    prepared: Vec<PreparedInputs>,
    audio_features_cpu: Tensor,
}

impl EncodedPreparedGroup {
    fn audio_token_lengths(&self) -> Result<Vec<usize>> {
        if self.chunk_indices.len() != self.prepared.len() {
            bail!("encoded group prepared/chunk length mismatch");
        }
        let (rows, _) = self.audio_features_cpu.dims2()?;
        let lengths = self
            .prepared
            .iter()
            .map(|row| {
                let frames = row
                    .feature_attention_mask
                    .iter()
                    .filter(|&&value| value != 0)
                    .count();
                crate::processor::feat_lengths::feat_extract_output_length(frames)
            })
            .collect::<Vec<_>>();
        let expected = lengths.iter().try_fold(0usize, |total, &length| {
            total
                .checked_add(length)
                .ok_or_else(|| anyhow::anyhow!("encoded feature token count overflow"))
        })?;
        if expected != rows {
            bail!("encoded group feature token rows mismatch: expected={expected}, got={rows}");
        }
        Ok(lengths)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioFeatureEncodeTimings {
    pub prepare_us: u64,
    pub stack_us: u64,
    pub forward_us: u64,
    pub d2h_us: u64,
    pub audio_tokens: usize,
}

impl AudioFeatureEncodeTimings {
    fn add_assign(&mut self, other: Self) {
        self.prepare_us = self.prepare_us.saturating_add(other.prepare_us);
        self.stack_us = self.stack_us.saturating_add(other.stack_us);
        self.forward_us = self.forward_us.saturating_add(other.forward_us);
        self.d2h_us = self.d2h_us.saturating_add(other.d2h_us);
        self.audio_tokens = self.audio_tokens.saturating_add(other.audio_tokens);
    }
}

#[derive(Debug)]
pub struct EncodedAsrBatch {
    audio_len: usize,
    chunks: Vec<ChunkItem>,
    chunk_audio_seconds: Vec<f64>,
    forced_langs_norm: Vec<Option<String>>,
    groups: Vec<EncodedPreparedGroup>,
    timings: AudioFeatureEncodeTimings,
}

#[cfg(feature = "paged-attn")]
#[derive(Debug)]
pub struct EncodedAsrBatchMetadata {
    audio_len: usize,
    chunks: Vec<ChunkItem>,
    chunk_audio_seconds: Vec<f64>,
    forced_langs_norm: Vec<Option<String>>,
}

#[cfg(feature = "paged-attn")]
impl EncodedAsrBatchMetadata {
    pub fn audio_len(&self) -> usize {
        self.audio_len
    }

    pub fn chunk_len(&self) -> usize {
        self.chunks.len()
    }

    /// Input audio index and duration for each chunk, in producer client order.
    pub fn chunk_audio_metadata(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.chunks
            .iter()
            .zip(&self.chunk_audio_seconds)
            .map(|(chunk, &seconds)| (chunk.orig_index, seconds))
    }

    fn merge(metas: Vec<Self>) -> Self {
        let audio_len = metas.iter().map(|meta| meta.audio_len).sum();
        let chunk_count = metas.iter().map(|meta| meta.chunks.len()).sum();
        let mut chunks = Vec::with_capacity(chunk_count);
        let mut chunk_audio_seconds = Vec::with_capacity(chunk_count);
        let mut forced_langs_norm = Vec::with_capacity(audio_len);
        let mut audio_offset = 0usize;

        for mut meta in metas {
            for chunk in &mut meta.chunks {
                chunk.orig_index = chunk.orig_index.saturating_add(audio_offset);
            }
            audio_offset = audio_offset.saturating_add(meta.audio_len);
            chunks.extend(meta.chunks);
            chunk_audio_seconds.extend(meta.chunk_audio_seconds);
            forced_langs_norm.extend(meta.forced_langs_norm);
        }

        Self {
            audio_len,
            chunks,
            chunk_audio_seconds,
            forced_langs_norm,
        }
    }
}

#[cfg(feature = "paged-attn")]
#[derive(Debug)]
pub enum EncodedAsrBatchPoll {
    Ready(EncodedAsrBatch),
    Empty,
    Closed,
}

#[cfg(feature = "paged-attn")]
struct DynamicPreparedSource<'a> {
    poll_encoded: &'a mut (dyn FnMut(bool) -> Result<EncodedAsrBatchPoll> + Send),
    metas: Vec<EncodedAsrBatchMetadata>,
    batch_lens: Vec<usize>,
    chunks_seen: usize,
    batches_seen: usize,
}

#[cfg(feature = "paged-attn")]
impl<'a> DynamicPreparedSource<'a> {
    fn new(
        initial_meta: EncodedAsrBatchMetadata,
        initial_chunks: usize,
        poll_encoded: &'a mut (dyn FnMut(bool) -> Result<EncodedAsrBatchPoll> + Send),
    ) -> Self {
        Self {
            poll_encoded,
            batch_lens: vec![initial_meta.audio_len],
            metas: vec![initial_meta],
            chunks_seen: initial_chunks,
            batches_seen: 1,
        }
    }

    fn poll_prepared(&mut self, may_block: bool) -> Result<PreparedAudioBatchPoll> {
        match (self.poll_encoded)(may_block)? {
            EncodedAsrBatchPoll::Ready(encoded) => {
                let (prepared, meta) = encoded.into_prepared_audio_batch_and_metadata()?;
                self.chunks_seen = self.chunks_seen.saturating_add(prepared.prepared.len());
                self.batches_seen = self.batches_seen.saturating_add(1);
                self.batch_lens.push(meta.audio_len);
                self.metas.push(meta);
                Ok(PreparedAudioBatchPoll::Ready(prepared))
            }
            EncodedAsrBatchPoll::Empty => Ok(PreparedAudioBatchPoll::Empty),
            EncodedAsrBatchPoll::Closed => Ok(PreparedAudioBatchPoll::Closed),
        }
    }

    fn into_parts(self) -> (Vec<EncodedAsrBatchMetadata>, Vec<usize>, usize, usize) {
        (
            self.metas,
            self.batch_lens,
            self.chunks_seen,
            self.batches_seen,
        )
    }
}

#[cfg(feature = "paged-attn")]
pub fn decode_ids_with_metadata(
    processor: &AsrProcessor,
    ids: Vec<Vec<u32>>,
    metas: Vec<EncodedAsrBatchMetadata>,
    batch_lens: Vec<usize>,
) -> Result<Vec<Vec<AsrTranscription>>> {
    let merged = EncodedAsrBatchMetadata::merge(metas);
    if ids.len() != merged.chunks.len() {
        bail!(
            "dynamic staged ASR output size mismatch: expected={}, got={}",
            merged.chunks.len(),
            ids.len()
        );
    }
    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); merged.audio_len];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); merged.audio_len];
    for (chunk, ids) in merged.chunks.iter().zip(ids) {
        let raw = processor.tokenizer.decode(ids.as_slice())?;
        // 只在显式打开时记下原文，用来核对特别长的生成。
        if std::env::var_os("VASR_DEBUG_RAW_ASR").is_some() {
            let audio_seconds = chunk.wav.len() as f64 / f64::from(SAMPLE_RATE_HZ);
            tracing::debug!(
                target: "vasr::inference::transcribe",
                orig_idx = chunk.orig_index,
                tokens = ids.len(),
                audio_seconds,
                raw = raw.as_str(),
                "raw ASR decode"
            );
        }
        let forced = merged
            .forced_langs_norm
            .get(chunk.orig_index)
            .and_then(|x| x.as_deref());
        let (lang, text) = parse_asr_output(Some(raw.as_str()), forced);
        out_langs[chunk.orig_index].push(lang);
        out_texts[chunk.orig_index].push(text);
    }

    let outputs: Vec<AsrTranscription> = out_langs
        .into_iter()
        .zip(out_texts)
        .map(|(langs, texts)| AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps: None,
        })
        .collect();
    let mut grouped = Vec::with_capacity(batch_lens.len());
    let mut cursor = 0usize;
    for len in batch_lens {
        let end = cursor.saturating_add(len);
        if end > outputs.len() {
            bail!(
                "dynamic staged ASR batch split mismatch: end={} outputs={}",
                end,
                outputs.len()
            );
        }
        grouped.push(outputs[cursor..end].to_vec());
        cursor = end;
    }
    if cursor != outputs.len() {
        bail!(
            "dynamic staged ASR unused outputs: used={} outputs={}",
            cursor,
            outputs.len()
        );
    }
    Ok(grouped)
}

impl EncodedAsrBatch {
    /// Keep the requested input audios and remap chunk metadata to their new indices.
    /// Encoder work is not repeated: retained CPU feature rows are sliced from the
    /// existing tensors. Identity selection returns the original batch unchanged.
    pub fn select_audios(self, indices: &[usize]) -> Result<Self> {
        let mut audio_map = vec![None; self.audio_len];
        for (new_index, &old_index) in indices.iter().enumerate() {
            let destination = audio_map
                .get_mut(old_index)
                .ok_or_else(|| anyhow::anyhow!("encoded audio index out of range: {old_index}"))?;
            if destination.replace(new_index).is_some() {
                bail!("encoded audio selected more than once: {old_index}");
            }
        }
        if indices.iter().copied().eq(0..self.audio_len) {
            return Ok(self);
        }
        if self.forced_langs_norm.len() != self.audio_len
            || self.chunk_audio_seconds.len() != self.chunks.len()
        {
            bail!("encoded audio metadata lengths do not match the batch");
        }
        let forced_langs_norm = indices
            .iter()
            .map(|&index| self.forced_langs_norm[index].clone())
            .collect();
        let mut chunk_map = vec![None; self.chunks.len()];
        let mut chunks = Vec::new();
        let mut chunk_audio_seconds = Vec::new();
        for (old_index, (mut chunk, seconds)) in self
            .chunks
            .into_iter()
            .zip(self.chunk_audio_seconds)
            .enumerate()
        {
            let selected = audio_map.get(chunk.orig_index).ok_or_else(|| {
                anyhow::anyhow!(
                    "encoded chunk audio index out of range: {}",
                    chunk.orig_index
                )
            })?;
            if let Some(new_audio_index) = *selected {
                chunk_map[old_index] = Some(chunks.len());
                chunk.orig_index = new_audio_index;
                chunks.push(chunk);
                chunk_audio_seconds.push(seconds);
            }
        }
        let mut groups = Vec::new();
        if !indices.is_empty() {
            for group in self.groups {
                let lengths = group.audio_token_lengths()?;
                let original_len = group.prepared.len();
                let mut offset = 0;
                let mut prepared = Vec::new();
                let mut chunk_indices = Vec::new();
                let mut features = Vec::new();
                for ((old_chunk_index, row), tokens) in group
                    .chunk_indices
                    .into_iter()
                    .zip(group.prepared)
                    .zip(lengths)
                {
                    let selected = chunk_map.get(old_chunk_index).ok_or_else(|| {
                        anyhow::anyhow!("encoded group chunk index out of range: {old_chunk_index}")
                    })?;
                    if let Some(new_chunk_index) = *selected {
                        chunk_indices.push(new_chunk_index);
                        prepared.push(row);
                        features.push(group.audio_features_cpu.narrow(0, offset, tokens)?);
                    }
                    offset += tokens;
                }
                if prepared.is_empty() {
                    continue;
                }
                let audio_features_cpu = if prepared.len() == original_len {
                    group.audio_features_cpu
                } else if features.len() == 1 {
                    features.pop().expect("one retained row")
                } else {
                    Tensor::cat(features.as_slice(), 0)?
                };
                groups.push(EncodedPreparedGroup {
                    chunk_indices,
                    prepared,
                    audio_features_cpu,
                });
            }
        }
        Ok(Self {
            audio_len: indices.len(),
            chunks,
            chunk_audio_seconds,
            forced_langs_norm,
            groups,
            timings: self.timings,
        })
    }

    /// Keep input audios in their original order using one mask for the whole batch.
    pub fn retain_audios(self, keep: &[bool]) -> Result<Self> {
        if keep.len() != self.audio_len {
            bail!(
                "encoded audio retention mask length mismatch: expected={}, got={}",
                self.audio_len,
                keep.len()
            );
        }
        if keep.iter().all(|&keep| keep) {
            return Ok(self);
        }
        self.select_audios(
            &keep
                .iter()
                .enumerate()
                .filter_map(|(index, &keep)| keep.then_some(index))
                .collect::<Vec<_>>(),
        )
    }

    pub fn len(&self) -> usize {
        self.audio_len
    }

    pub fn is_empty(&self) -> bool {
        self.audio_len == 0
    }

    pub fn audio_len(&self) -> usize {
        self.audio_len
    }

    pub fn audio_feature_encode_timings(&self) -> AudioFeatureEncodeTimings {
        self.timings
    }

    #[cfg(feature = "paged-attn")]
    fn into_meta_and_groups(self) -> (EncodedAsrBatchMetadata, Vec<EncodedPreparedGroup>) {
        (
            EncodedAsrBatchMetadata {
                audio_len: self.audio_len,
                chunks: self.chunks,
                chunk_audio_seconds: self.chunk_audio_seconds,
                forced_langs_norm: self.forced_langs_norm,
            },
            self.groups,
        )
    }

    pub fn merge(batches: Vec<Self>) -> Result<Self> {
        let audio_len = batches.iter().map(|batch| batch.audio_len).sum();
        let chunk_count = batches.iter().map(|batch| batch.chunks.len()).sum();
        let group_count = batches.iter().map(|batch| batch.groups.len()).sum();
        let mut chunks = Vec::with_capacity(chunk_count);
        let mut chunk_audio_seconds = Vec::with_capacity(chunk_count);
        let mut forced_langs_norm = Vec::with_capacity(audio_len);
        let mut groups = Vec::with_capacity(group_count);
        let mut timings = AudioFeatureEncodeTimings::default();
        let mut audio_offset = 0usize;
        let mut chunk_offset = 0usize;

        for mut batch in batches {
            timings.add_assign(batch.timings);
            for chunk in &mut batch.chunks {
                chunk.orig_index = chunk.orig_index.saturating_add(audio_offset);
            }
            for group in &mut batch.groups {
                for chunk_idx in &mut group.chunk_indices {
                    *chunk_idx = chunk_idx.saturating_add(chunk_offset);
                }
            }
            audio_offset = audio_offset.saturating_add(batch.audio_len);
            chunk_offset = chunk_offset.saturating_add(batch.chunks.len());
            chunks.extend(batch.chunks);
            chunk_audio_seconds.extend(batch.chunk_audio_seconds);
            forced_langs_norm.extend(batch.forced_langs_norm);
            groups.extend(batch.groups);
        }

        Ok(Self {
            audio_len,
            chunks,
            chunk_audio_seconds,
            forced_langs_norm,
            groups,
            timings,
        })
    }

    #[cfg(feature = "paged-attn")]
    fn groups_into_prepared_audio_batch(
        groups: Vec<EncodedPreparedGroup>,
        chunk_audio_seconds: &[f64],
        chunks: &[ChunkItem],
    ) -> Result<PreparedAudioBatch> {
        // The scheduler assigns client indices in prepared-row order. Normalize
        // length-bucketed groups to chunk order so metadata and per-audio chunk
        // concatenation stay aligned. The normal sequential path is untouched.
        let groups = if groups
            .iter()
            .flat_map(|group| group.chunk_indices.iter().copied())
            .eq(0..chunks.len())
        {
            groups
        } else {
            let mut ordered = (0..chunks.len())
                .map(|_| None)
                .collect::<Vec<Option<(PreparedInputs, Tensor)>>>();
            for group in groups {
                let lengths = group.audio_token_lengths()?;
                let mut offset = 0;
                for ((index, prepared), tokens) in group
                    .chunk_indices
                    .into_iter()
                    .zip(group.prepared)
                    .zip(lengths)
                {
                    let destination = ordered.get_mut(index).ok_or_else(|| {
                        anyhow::anyhow!("encoded group chunk index out of range: {index}")
                    })?;
                    if destination
                        .replace((
                            prepared,
                            group.audio_features_cpu.narrow(0, offset, tokens)?,
                        ))
                        .is_some()
                    {
                        bail!("encoded chunk prepared more than once: {index}");
                    }
                    offset += tokens;
                }
            }
            let mut prepared = Vec::with_capacity(ordered.len());
            let mut features = Vec::with_capacity(ordered.len());
            for (index, row) in ordered.into_iter().enumerate() {
                let (input, feature) = row.ok_or_else(|| {
                    anyhow::anyhow!("encoded chunk missing prepared row: {index}")
                })?;
                prepared.push(input);
                features.push(feature);
            }
            let audio_features_cpu = Tensor::cat(features.as_slice(), 0)?;
            vec![EncodedPreparedGroup {
                chunk_indices: (0..chunks.len()).collect(),
                prepared,
                audio_features_cpu,
            }]
        };
        let mut prepared = Vec::new();
        let mut audio_seconds = Vec::new();
        let mut speech_rate_ratios = Vec::new();
        let mut audio_feature_parts = Vec::new();
        for group in groups {
            for &chunk_idx in &group.chunk_indices {
                audio_seconds.push(*chunk_audio_seconds.get(chunk_idx).ok_or_else(|| {
                    anyhow::anyhow!("missing chunk audio seconds for chunk {chunk_idx}")
                })?);
                let chunk = chunks
                    .get(chunk_idx)
                    .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
                speech_rate_ratios.push(super::speech_rate::speech_rate_ratio(
                    chunk.wav.as_slice(),
                    SAMPLE_RATE_HZ,
                ));
            }
            prepared.extend(group.prepared);
            audio_feature_parts.push(group.audio_features_cpu);
        }
        let audio_features = if audio_feature_parts.is_empty() {
            Tensor::zeros((0usize, 0usize), candle_core::DType::F32, &Device::Cpu)?
        } else if audio_feature_parts.len() == 1 {
            audio_feature_parts
                .pop()
                .ok_or_else(|| anyhow::anyhow!("missing staged audio features"))?
        } else {
            Tensor::cat(audio_feature_parts.as_slice(), 0)?
        };
        Ok(PreparedAudioBatch {
            prepared,
            audio_features,
            audio_seconds,
            speech_rate_ratios,
        })
    }

    #[cfg(feature = "paged-attn")]
    pub fn into_prepared_audio_batch(self) -> Result<PreparedAudioBatch> {
        Self::groups_into_prepared_audio_batch(
            self.groups,
            self.chunk_audio_seconds.as_slice(),
            self.chunks.as_slice(),
        )
    }

    #[cfg(feature = "paged-attn")]
    pub fn into_prepared_audio_batch_and_metadata(
        self,
    ) -> Result<(PreparedAudioBatch, EncodedAsrBatchMetadata)> {
        let (meta, groups) = self.into_meta_and_groups();
        let prepared = Self::groups_into_prepared_audio_batch(
            groups,
            meta.chunk_audio_seconds.as_slice(),
            meta.chunks.as_slice(),
        )?;
        Ok((prepared, meta))
    }

    #[cfg(feature = "paged-attn")]
    pub fn to_prepared_audio_batch(&self) -> Result<PreparedAudioBatch> {
        Self::groups_into_prepared_audio_batch(
            self.groups.clone(),
            &self.chunk_audio_seconds,
            &self.chunks,
        )
    }
}

#[cfg(feature = "timing")]
fn merge_generation_timings(dst: &mut TranscribeTimings, src: &GenerationTimings) {
    dst.generation.prompt_tensors_us = dst
        .generation
        .prompt_tensors_us
        .saturating_add(src.prompt_tensors_us);
    dst.generation.prefill_us = dst.generation.prefill_us.saturating_add(src.prefill_us);
    dst.generation.prefill_inputs_us = dst
        .generation
        .prefill_inputs_us
        .saturating_add(src.prefill_inputs_us);
    dst.generation.prefill_rope_us = dst
        .generation
        .prefill_rope_us
        .saturating_add(src.prefill_rope_us);
    dst.generation.prefill_metadata_us = dst
        .generation
        .prefill_metadata_us
        .saturating_add(src.prefill_metadata_us);
    dst.generation.prefill_mask_us = dst
        .generation
        .prefill_mask_us
        .saturating_add(src.prefill_mask_us);
    dst.generation.prefill_forward_us = dst
        .generation
        .prefill_forward_us
        .saturating_add(src.prefill_forward_us);
    dst.generation.prefill_gather_us = dst
        .generation
        .prefill_gather_us
        .saturating_add(src.prefill_gather_us);
    dst.generation.prefill_decode_setup_us = dst
        .generation
        .prefill_decode_setup_us
        .saturating_add(src.prefill_decode_setup_us);
    dst.generation.prefill_argmax_us = dst
        .generation
        .prefill_argmax_us
        .saturating_add(src.prefill_argmax_us);
    dst.generation.prefill_tokens = dst
        .generation
        .prefill_tokens
        .saturating_add(src.prefill_tokens);
    dst.generation.decode_us = dst.generation.decode_us.saturating_add(src.decode_us);
    dst.generation.decode_queue_us = dst
        .generation
        .decode_queue_us
        .saturating_add(src.decode_queue_us);
    dst.generation.decode_token_tensor_us = dst
        .generation
        .decode_token_tensor_us
        .saturating_add(src.decode_token_tensor_us);
    dst.generation.decode_embed_us = dst
        .generation
        .decode_embed_us
        .saturating_add(src.decode_embed_us);
    dst.generation.decode_position_us = dst
        .generation
        .decode_position_us
        .saturating_add(src.decode_position_us);
    dst.generation.decode_metadata_us = dst
        .generation
        .decode_metadata_us
        .saturating_add(src.decode_metadata_us);
    dst.generation.decode_graph_replay_us = dst
        .generation
        .decode_graph_replay_us
        .saturating_add(src.decode_graph_replay_us);
    dst.generation.decode_forward_us = dst
        .generation
        .decode_forward_us
        .saturating_add(src.decode_forward_us);
    dst.generation.decode_pre_argmax_sync_us = dst
        .generation
        .decode_pre_argmax_sync_us
        .saturating_add(src.decode_pre_argmax_sync_us);
    dst.generation.decode_argmax_us = dst
        .generation
        .decode_argmax_us
        .saturating_add(src.decode_argmax_us);
    dst.generation.prompt_len = dst.generation.prompt_len.max(src.prompt_len);
    dst.generation.steps = dst.generation.steps.saturating_add(src.steps);
    dst.generation.tokens_generated = dst
        .generation
        .tokens_generated
        .saturating_add(src.tokens_generated);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_raw_prepared_batch(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    prepared: &[PreparedInputs],
    max_new_tokens: usize,
    decode_max_seqs: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<GeneratedRaw>> {
    if prepared.is_empty() {
        return Ok(vec![]);
    }

    let max_new_tokens = if max_new_tokens == 0 {
        256
    } else {
        max_new_tokens
    };

    #[cfg(feature = "paged-attn")]
    let varlen_stack = use_varlen_paged_prepare(paged_runtime);
    #[cfg(not(feature = "paged-attn"))]
    let varlen_stack = false;
    let effective_decode_max_seqs = if decode_max_seqs == 0 {
        prepared.len().max(1)
    } else {
        decode_max_seqs
    };
    let pad_id = paged_token_pad_id(processor)?;
    let micro_batches =
        rebatch_prepared_inputs(prepared, pad_id, effective_decode_max_seqs, varlen_stack);
    let mut out: Vec<Option<GeneratedRaw>> = vec![None; prepared.len()];

    for batch in micro_batches {
        let batch_prepared: Vec<PreparedInputs> = batch.iter().map(|(_, p)| p.clone()).collect();
        log_cuda_mem(device, "asr_microbatch_start");
        let ids_rows: Vec<&[u32]> = batch_prepared
            .iter()
            .map(|p| p.input_ids.as_slice())
            .collect();
        let attn_rows: Vec<&[u32]> = batch_prepared
            .iter()
            .map(|p| p.attention_mask.as_slice())
            .collect();
        let (input_features, feature_lens) = if varlen_stack {
            stack_features_varlen(batch_prepared.as_slice(), device)?
        } else {
            stack_features(batch_prepared.as_slice(), device)?
        };
        log_cuda_mem(device, "after_stack_features");
        #[cfg(feature = "paged-attn")]
        let audio_features = with_metal_model_execution_and_sync(device, paged_runtime, || {
            thinker.get_audio_features_with_lens(&input_features, feature_lens.as_slice())
        })?;
        #[cfg(not(feature = "paged-attn"))]
        let audio_features =
            thinker.get_audio_features_with_lens(&input_features, feature_lens.as_slice())?;
        log_cuda_mem(device, "after_audio_encoder");
        drop(input_features);
        #[cfg(feature = "paged-attn")]
        let gen_ids = if let Some(runtime) = paged_runtime {
            use crate::inference::batch_scheduler::run_paged_prepared_batch;
            run_paged_prepared_batch(
                thinker,
                device,
                runtime,
                batch_prepared.as_slice(),
                &audio_features,
                max_new_tokens,
                eos_token_ids,
                batch_prepared.len(),
                pad_id,
                None,
            )?
        } else {
            let _ = (ids_rows.as_slice(), attn_rows.as_slice());
            bail!("paged-attn build requires a shared paged runtime for offline ASR generation")
        };
        #[cfg(not(feature = "paged-attn"))]
        let gen_ids = greedy_generate_cached_batch(
            thinker,
            device,
            ids_rows.as_slice(),
            attn_rows.as_slice(),
            Some(&audio_features),
            max_new_tokens,
            eos_token_ids,
        )?;
        drop(audio_features);

        if gen_ids.len() != batch_prepared.len() {
            bail!(
                "internal error: generated micro-batch size mismatch: expected={}, got={}",
                batch_prepared.len(),
                gen_ids.len()
            );
        }

        for ((orig_idx, _), ids) in batch.into_iter().zip(gen_ids) {
            let raw = processor.tokenizer.decode(ids.as_slice())?;
            if std::env::var_os("VASR_DEBUG_RAW_ASR").is_some() {
                tracing::debug!(
                    target: "vasr::inference::transcribe",
                    orig_idx,
                    tokens = ids.len(),
                    raw = raw.as_str(),
                    "raw ASR decode"
                );
            }
            out[orig_idx] = Some(GeneratedRaw {
                text: raw,
                token_count: ids.len(),
            });
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, text)| text.ok_or_else(|| anyhow::anyhow!("missing generated text {i}")))
        .collect()
}

fn normalize_forced_language(lang: Option<String>) -> Result<Option<String>> {
    let Some(lang) = lang else {
        return Ok(None);
    };
    let s = lang.trim();
    if s.is_empty() {
        return Ok(None);
    }
    let normalized = normalize_language_name(s)?;
    validate_language(normalized.as_str())?;
    Ok(Some(normalized))
}

#[derive(Debug, Clone)]
struct ChunkItem {
    orig_index: usize,
    wav: Vec<f32>,
    offset_sec: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct GeneratedRaw {
    text: String,
    token_count: usize,
}

#[cfg(feature = "timing")]
#[derive(Debug, Clone, Copy, Default)]
struct BatchDist {
    min: f64,
    p50: f64,
    p90: f64,
    max: f64,
    avg: f64,
}

#[cfg(feature = "timing")]
fn dist_from_usize(values: &[usize]) -> BatchDist {
    if values.is_empty() {
        return BatchDist::default();
    }
    let mut sorted = values.iter().map(|&v| v as f64).collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.total_cmp(b));
    dist_from_sorted(sorted.as_slice())
}

#[cfg(feature = "timing")]
fn dist_from_f32(values: &[f32]) -> BatchDist {
    if values.is_empty() {
        return BatchDist::default();
    }
    let mut sorted = values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.total_cmp(b));
    dist_from_sorted(sorted.as_slice())
}

#[cfg(feature = "timing")]
fn dist_from_sorted(sorted: &[f64]) -> BatchDist {
    let len = sorted.len();
    let percentile = |pct: usize| -> f64 {
        let idx = len.saturating_sub(1).saturating_mul(pct).div_ceil(100);
        sorted[idx.min(len - 1)]
    };
    let sum: f64 = sorted.iter().sum();
    BatchDist {
        min: sorted[0],
        p50: percentile(50),
        p90: percentile(90),
        max: sorted[len - 1],
        avg: sum / len as f64,
    }
}

fn raw_preview(text: &str) -> String {
    const LIMIT: usize = 160;
    let mut out = text.chars().take(LIMIT).collect::<String>();
    if text.chars().count() > LIMIT {
        out.push_str("...");
    }
    out
}

fn chunk_offset_sec(chunk: &ChunkItem) -> f32 {
    chunk.offset_sec
}

fn log_max_new_tokens_hit(chunk_idx: usize, chunk: &ChunkItem, token_count: usize, raw: &str) {
    tracing::warn!(
        target: "vasr_transcribe::pipeline",
        "ASR chunk reached max_new_tokens | chunk_idx={} orig_index={} audio_sec={:.2} offset_sec={:.2} tokens={} raw_preview={:?}",
        chunk_idx,
        chunk.orig_index,
        chunk_audio_seconds(chunk.wav.len()),
        chunk_offset_sec(chunk),
        token_count,
        raw_preview(raw).as_str(),
    );
}

fn prepare_items_for_runtime(
    processor: &AsrProcessor,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    items: &[(&str, &AudioInput<'_>)],
) -> Result<Vec<PreparedInputs>> {
    #[cfg(feature = "paged-attn")]
    {
        if use_varlen_paged_prepare(paged_runtime) {
            return processor.prepare_batch_varlen(items);
        }
    }
    processor.prepare_batch(items)
}

fn encode_prepared_audio_features_cpu(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    prepared: &[PreparedInputs],
) -> Result<(Tensor, AudioFeatureEncodeTimings)> {
    let mut timings = AudioFeatureEncodeTimings::default();
    #[cfg(feature = "paged-attn")]
    let varlen_stack = use_varlen_paged_prepare(paged_runtime);
    #[cfg(not(feature = "paged-attn"))]
    let varlen_stack = false;
    let start_stack = std::time::Instant::now();
    let (input_features, feature_lens) = if varlen_stack {
        stack_features_varlen(prepared, device)?
    } else {
        stack_features(prepared, device)?
    };
    timings.stack_us = duration_to_us(start_stack.elapsed());
    crate::profiling::record_batch(
        "encoder_frames",
        feature_lens.len(),
        feature_lens.iter().sum(),
        feature_lens.len() * input_features.dim(2)?,
    );
    #[cfg(feature = "paged-attn")]
    let (audio_features_cpu, forward_us, d2h_us, audio_tokens) =
        with_metal_model_execution(device, paged_runtime, || {
            let start_forward = std::time::Instant::now();
            let audio_features = crate::profiling::measure_with_autorelease(
                "encoder_forward",
                device,
                thinker.metal_auto_packed_prefill(),
                || thinker.get_audio_features_with_lens(&input_features, feature_lens.as_slice()),
            )?;
            let forward_us = duration_to_us(start_forward.elapsed());
            let audio_tokens = audio_features.dims2().map(|(n, _)| n).unwrap_or(0);
            let start_d2h = std::time::Instant::now();
            let audio_features_cpu = audio_features.to_device(&Device::Cpu)?;
            let d2h_us = duration_to_us(start_d2h.elapsed());
            Ok((audio_features_cpu, forward_us, d2h_us, audio_tokens))
        })?;
    #[cfg(not(feature = "paged-attn"))]
    let (audio_features_cpu, forward_us, d2h_us, audio_tokens) = {
        let start_forward = std::time::Instant::now();
        let audio_features =
            thinker.get_audio_features_with_lens(&input_features, feature_lens.as_slice())?;
        let forward_us = duration_to_us(start_forward.elapsed());
        let audio_tokens = audio_features.dims2().map(|(n, _)| n).unwrap_or(0);
        let start_d2h = std::time::Instant::now();
        let audio_features_cpu = audio_features.to_device(&Device::Cpu)?;
        let d2h_us = duration_to_us(start_d2h.elapsed());
        (audio_features_cpu, forward_us, d2h_us, audio_tokens)
    };
    timings.forward_us = forward_us;
    timings.d2h_us = d2h_us;
    timings.audio_tokens = audio_tokens;
    Ok((audio_features_cpu, timings))
}

#[cfg(feature = "paged-attn")]
fn with_metal_model_execution<T>(
    device: &Device,
    paged_runtime: Option<&SharedPagedCacheRuntime>,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if device.is_metal()
        && let Some(runtime) = paged_runtime
    {
        let execution_lock = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?
            .model_execution_lock();
        let _execution_guard = execution_lock
            .lock()
            .map_err(|_| anyhow::anyhow!("Metal model execution lock poisoned"))?;
        return operation();
    }
    operation()
}

#[cfg(feature = "paged-attn")]
fn with_metal_model_execution_and_sync<T>(
    device: &Device,
    paged_runtime: Option<&SharedPagedCacheRuntime>,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_metal_model_execution(device, paged_runtime, || {
        let result = operation();
        if device.is_metal() {
            let sync_result = device.synchronize();
            return match result {
                Ok(output) => {
                    sync_result?;
                    Ok(output)
                }
                Err(error) => {
                    if let Err(sync_error) = sync_result {
                        return Err(error.context(format!(
                            "Metal synchronization also failed after encoder error: {sync_error}"
                        )));
                    }
                    Err(error)
                }
            };
        }
        result
    })
}

#[allow(clippy::too_many_arguments)]
fn generate_raw_prepared_batch_with_audio_features(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    prepared: &[PreparedInputs],
    audio_features: &Tensor,
    max_new_tokens: usize,
    _decode_max_seqs: usize,
    eos_token_ids: &[u32],
) -> Result<Vec<GeneratedRaw>> {
    if prepared.is_empty() {
        return Ok(vec![]);
    }
    let max_new_tokens = if max_new_tokens == 0 {
        256
    } else {
        max_new_tokens
    };
    let ids_rows: Vec<&[u32]> = prepared.iter().map(|p| p.input_ids.as_slice()).collect();
    let attn_rows: Vec<&[u32]> = prepared
        .iter()
        .map(|p| p.attention_mask.as_slice())
        .collect();
    #[cfg(feature = "paged-attn")]
    let pad_id = paged_token_pad_id(processor)?;
    #[cfg(feature = "paged-attn")]
    let gen_ids = if let Some(runtime) = paged_runtime {
        use crate::inference::batch_scheduler::run_paged_prepared_batch;
        run_paged_prepared_batch(
            thinker,
            device,
            runtime,
            prepared,
            audio_features,
            max_new_tokens,
            eos_token_ids,
            _decode_max_seqs,
            pad_id,
            None,
        )?
    } else {
        let _ = (ids_rows.as_slice(), attn_rows.as_slice());
        bail!("paged-attn build requires a shared paged runtime for offline ASR generation")
    };
    #[cfg(not(feature = "paged-attn"))]
    let gen_ids = greedy_generate_cached_batch(
        thinker,
        device,
        ids_rows.as_slice(),
        attn_rows.as_slice(),
        Some(audio_features),
        max_new_tokens,
        eos_token_ids,
    )?;

    if gen_ids.len() != prepared.len() {
        bail!(
            "internal error: generated batch size mismatch: expected={}, got={}",
            prepared.len(),
            gen_ids.len()
        );
    }
    gen_ids
        .into_iter()
        .map(|ids| {
            let token_count = ids.len();
            let text = processor.tokenizer.decode(ids.as_slice())?;
            Ok(GeneratedRaw { text, token_count })
        })
        .collect()
}

struct ChunkAsrBatch<'a> {
    thinker: &'a ThinkerForConditionalGeneration,
    processor: &'a AsrProcessor,
    device: &'a Device,
    #[cfg(feature = "paged-attn")]
    paged_runtime: Option<&'a SharedPagedCacheRuntime>,
    chunks: &'a [ChunkItem],
    prompts: &'a [String],
    forced_langs_norm: &'a [Option<String>],
    max_new_tokens: usize,
    decode_max_seqs: usize,
    max_batch_audio_sec: f32,
    bucket_by_length: bool,
    eos_token_ids: &'a [u32],
}

impl ChunkAsrBatch<'_> {
    fn prepare_items(&self, items: &[(&str, &AudioInput<'_>)]) -> Result<Vec<PreparedInputs>> {
        #[cfg(feature = "paged-attn")]
        {
            if use_varlen_paged_prepare(self.paged_runtime) {
                return self.processor.prepare_batch_varlen(items);
            }
        }
        self.processor.prepare_batch(items)
    }

    #[cfg(feature = "timing")]
    fn prepare_items_timed(
        &self,
        items: &[(&str, &AudioInput<'_>)],
    ) -> Result<(
        Vec<PreparedInputs>,
        crate::processor::asr_processor::PrepareBatchTimings,
    )> {
        #[cfg(feature = "paged-attn")]
        {
            if use_varlen_paged_prepare(self.paged_runtime) {
                return self.processor.prepare_batch_varlen_timed(items);
            }
        }
        self.processor.prepare_batch_timed(items)
    }
}

fn chunk_audio_seconds(samples: usize) -> f32 {
    samples as f32 / SAMPLE_RATE_HZ as f32
}

/// Group chunk indices by total input audio seconds. A non-positive budget keeps
/// all chunks in one pool and leaves runtime capacity to sequence/KV limits.
fn group_chunk_indices(
    ordered_indices: &[usize],
    chunks: &[ChunkItem],
    max_batch_audio_sec: f32,
) -> Result<Vec<Vec<usize>>> {
    if ordered_indices.is_empty() {
        return Ok(Vec::new());
    }
    if max_batch_audio_sec <= 0.0 {
        return Ok(vec![ordered_indices.to_vec()]);
    }

    let mut groups = Vec::new();
    let mut current = Vec::new();
    let mut current_audio_sec = 0.0f32;

    for &index in ordered_indices {
        let chunk = chunks
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("chunk index {index} out of range"))?;
        let audio_sec = chunk_audio_seconds(chunk.wav.len());
        if !current.is_empty() && current_audio_sec + audio_sec > max_batch_audio_sec {
            groups.push(std::mem::take(&mut current));
            current_audio_sec = 0.0;
        }
        current.push(index);
        current_audio_sec += audio_sec;
    }

    if !current.is_empty() {
        groups.push(current);
    }
    Ok(groups)
}

fn run_asr_on_chunks_batched(batch: &ChunkAsrBatch<'_>) -> Result<(Vec<String>, Vec<String>)> {
    if batch.chunks.is_empty() {
        return Ok((vec![], vec![]));
    }
    if batch.prompts.len() != batch.forced_langs_norm.len() {
        bail!(
            "internal error: prompts/forced_langs_norm length mismatch: prompts={} forced_langs_norm={}",
            batch.prompts.len(),
            batch.forced_langs_norm.len()
        );
    }

    if !batch.bucket_by_length {
        let mut per_chunk_lang: Vec<String> = Vec::with_capacity(batch.chunks.len());
        let mut per_chunk_text: Vec<String> = Vec::with_capacity(batch.chunks.len());
        let ordered: Vec<usize> = (0..batch.chunks.len()).collect();
        let index_groups =
            group_chunk_indices(ordered.as_slice(), batch.chunks, batch.max_batch_audio_sec)?;

        for batch_chunk_idx in index_groups {
            let chunk_batch: Vec<&ChunkItem> = batch_chunk_idx
                .iter()
                .map(|&idx| {
                    batch
                        .chunks
                        .get(idx)
                        .ok_or_else(|| anyhow::anyhow!("missing chunk {idx}"))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(chunk_batch.len());
            for c in &chunk_batch {
                audio_inputs.push(AudioInput::Waveform {
                    samples: c.wav.as_slice(),
                    sample_rate: SAMPLE_RATE_HZ,
                });
            }

            let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(chunk_batch.len());
            for (c, a) in chunk_batch.iter().zip(audio_inputs.iter()) {
                let prompt = batch.prompts.get(c.orig_index).ok_or_else(|| {
                    anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index)
                })?;
                items.push((prompt.as_str(), a));
            }

            let prepared = batch.prepare_items(items.as_slice())?;
            let raws = generate_raw_prepared_batch(
                batch.thinker,
                batch.processor,
                batch.device,
                #[cfg(feature = "paged-attn")]
                batch.paged_runtime,
                prepared.as_slice(),
                batch.max_new_tokens,
                batch.decode_max_seqs,
                batch.eos_token_ids,
            )?;

            if raws.len() != chunk_batch.len() {
                bail!(
                    "internal error: raw output batch size mismatch: expected={}, got={}",
                    chunk_batch.len(),
                    raws.len()
                );
            }

            for (c, raw) in chunk_batch.iter().zip(raws.iter()) {
                if raw.token_count >= batch.max_new_tokens {
                    let chunk_idx = batch
                        .chunks
                        .iter()
                        .position(|chunk| std::ptr::eq(chunk, *c))
                        .unwrap_or(c.orig_index);
                    log_max_new_tokens_hit(chunk_idx, c, raw.token_count, raw.text.as_str());
                }
                let forced = batch
                    .forced_langs_norm
                    .get(c.orig_index)
                    .and_then(|x| x.as_deref());
                let (lang, text) = parse_asr_output(Some(raw.text.as_str()), forced);
                per_chunk_lang.push(lang);
                per_chunk_text.push(text);
            }
        }

        return Ok((per_chunk_lang, per_chunk_text));
    }

    let mut per_chunk_lang: Vec<Option<String>> = vec![None; batch.chunks.len()];
    let mut per_chunk_text: Vec<Option<String>> = vec![None; batch.chunks.len()];

    let mut indices: Vec<usize> = (0..batch.chunks.len()).collect();
    indices.sort_by(|&a, &b| {
        let la = batch.chunks[a].wav.len();
        let lb = batch.chunks[b].wav.len();
        lb.cmp(&la).then_with(|| a.cmp(&b))
    });

    let index_groups =
        group_chunk_indices(indices.as_slice(), batch.chunks, batch.max_batch_audio_sec)?;

    for batch_chunk_idx in index_groups {
        let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(batch_chunk_idx.len());
        for &chunk_idx in &batch_chunk_idx {
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            audio_inputs.push(AudioInput::Waveform {
                samples: c.wav.as_slice(),
                sample_rate: SAMPLE_RATE_HZ,
            });
        }

        let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(batch_chunk_idx.len());
        for (&chunk_idx, a) in batch_chunk_idx.iter().zip(audio_inputs.iter()) {
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let prompt = batch
                .prompts
                .get(c.orig_index)
                .ok_or_else(|| anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index))?;
            items.push((prompt.as_str(), a));
        }

        let prepared = batch.prepare_items(items.as_slice())?;
        if prepared.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: prepare_batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                prepared.len()
            );
        }

        let raws = generate_raw_prepared_batch(
            batch.thinker,
            batch.processor,
            batch.device,
            #[cfg(feature = "paged-attn")]
            batch.paged_runtime,
            prepared.as_slice(),
            batch.max_new_tokens,
            batch.decode_max_seqs,
            batch.eos_token_ids,
        )?;
        if raws.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: raw output batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                raws.len()
            );
        }

        for (k, raw) in raws.iter().enumerate() {
            let chunk_idx = *batch_chunk_idx
                .get(k)
                .ok_or_else(|| anyhow::anyhow!("missing chunk index {k}"))?;
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            if raw.token_count >= batch.max_new_tokens {
                log_max_new_tokens_hit(chunk_idx, c, raw.token_count, raw.text.as_str());
            }
            let forced = batch
                .forced_langs_norm
                .get(c.orig_index)
                .and_then(|x| x.as_deref());
            let (lang, text) = parse_asr_output(Some(raw.text.as_str()), forced);

            let slot_lang = per_chunk_lang
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output lang slot {chunk_idx}"))?;
            if slot_lang.is_some() {
                bail!("internal error: duplicate language for chunk {chunk_idx}");
            }
            *slot_lang = Some(lang);

            let slot_text = per_chunk_text
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output text slot {chunk_idx}"))?;
            if slot_text.is_some() {
                bail!("internal error: duplicate text for chunk {chunk_idx}");
            }
            *slot_text = Some(text);
        }
    }

    let mut langs: Vec<String> = Vec::with_capacity(per_chunk_lang.len());
    for (i, lang) in per_chunk_lang.into_iter().enumerate() {
        langs.push(lang.ok_or_else(|| anyhow::anyhow!("missing chunk language {i}"))?);
    }
    let mut texts: Vec<String> = Vec::with_capacity(per_chunk_text.len());
    for (i, text) in per_chunk_text.into_iter().enumerate() {
        texts.push(text.ok_or_else(|| anyhow::anyhow!("missing chunk text {i}"))?);
    }

    Ok((langs, texts))
}

#[cfg(feature = "timing")]
#[allow(clippy::too_many_arguments)]
fn generate_raw_prepared_batch_timed(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    prepared: &[PreparedInputs],
    max_new_tokens: usize,
    decode_max_seqs: usize,
    eos_token_ids: &[u32],
    timings: &mut TranscribeTimings,
) -> Result<Vec<GeneratedRaw>> {
    if prepared.is_empty() {
        return Ok(vec![]);
    }

    let max_new_tokens = if max_new_tokens == 0 {
        256
    } else {
        max_new_tokens
    };
    #[cfg(feature = "paged-attn")]
    let varlen_stack = use_varlen_paged_prepare(paged_runtime);
    #[cfg(not(feature = "paged-attn"))]
    let varlen_stack = false;
    let pad_id = paged_token_pad_id(processor)?;
    let effective_decode_max_seqs = if decode_max_seqs == 0 {
        prepared.len().max(1)
    } else {
        decode_max_seqs
    };
    let micro_batches =
        rebatch_prepared_inputs(prepared, pad_id, effective_decode_max_seqs, varlen_stack);
    let mut out: Vec<Option<GeneratedRaw>> = vec![None; prepared.len()];

    for batch in micro_batches {
        let batch_prepared: Vec<PreparedInputs> = batch.iter().map(|(_, p)| p.clone()).collect();
        log_cuda_mem(device, "asr_microbatch_start");
        let ids_rows: Vec<&[u32]> = batch_prepared
            .iter()
            .map(|p| p.input_ids.as_slice())
            .collect();
        let attn_rows: Vec<&[u32]> = batch_prepared
            .iter()
            .map(|p| p.attention_mask.as_slice())
            .collect();

        let start_stack = std::time::Instant::now();
        let (input_features, feature_lens) = if varlen_stack {
            stack_features_varlen(batch_prepared.as_slice(), device)?
        } else {
            stack_features(batch_prepared.as_slice(), device)?
        };
        log_cuda_mem(device, "after_stack_features");
        timings.stack_features_us = timings
            .stack_features_us
            .saturating_add(duration_to_us(start_stack.elapsed()));

        let start_audio = std::time::Instant::now();
        #[cfg(feature = "paged-attn")]
        let audio_features = with_metal_model_execution_and_sync(device, paged_runtime, || {
            thinker.get_audio_features_with_lens(&input_features, feature_lens.as_slice())
        })?;
        #[cfg(not(feature = "paged-attn"))]
        let audio_features =
            thinker.get_audio_features_with_lens(&input_features, feature_lens.as_slice())?;
        log_cuda_mem(device, "after_audio_encoder");
        let input_features_shape = if std::env::var_os("VASR_DEBUG_RAW_ASR").is_some() {
            Some(input_features.dims().to_vec())
        } else {
            None
        };
        drop(input_features);
        if std::env::var_os("VASR_DEBUG_RAW_ASR").is_some() {
            tracing::debug!(
                target: "vasr::inference::transcribe",
                batch_items = batch_prepared.len(),
                input_features_shape = ?input_features_shape,
                feature_lens = ?feature_lens,
                audio_features_shape = ?audio_features.dims(),
                prompt_lens = ?batch_prepared
                    .iter()
                    .map(|p| p.attention_mask.iter().filter(|&&x| x != 0).count())
                    .collect::<Vec<_>>(),
                audio_placeholders = ?batch_prepared
                    .iter()
                    .map(|p| p.input_ids.iter().filter(|&&id| id == thinker.audio_token_id()).count())
                    .collect::<Vec<_>>(),
                "prepared ASR batch"
            );
        }
        timings.audio_encoder_us = timings
            .audio_encoder_us
            .saturating_add(duration_to_us(start_audio.elapsed()));

        #[cfg(feature = "paged-attn")]
        let (gen_ids, gen_timings) = if let Some(runtime) = paged_runtime {
            use crate::inference::batch_scheduler::run_paged_prepared_batch_timed;
            run_paged_prepared_batch_timed(
                thinker,
                device,
                runtime,
                batch_prepared.as_slice(),
                &audio_features,
                max_new_tokens,
                eos_token_ids,
                decode_max_seqs,
                pad_id,
                None,
            )?
        } else {
            let _ = (ids_rows.as_slice(), attn_rows.as_slice());
            bail!(
                "paged-attn build requires a shared paged runtime for timed offline ASR generation"
            )
        };
        #[cfg(not(feature = "paged-attn"))]
        let (gen_ids, gen_timings) = greedy_generate_cached_batch_timed(
            thinker,
            device,
            ids_rows.as_slice(),
            attn_rows.as_slice(),
            Some(&audio_features),
            max_new_tokens,
            eos_token_ids,
        )?;
        drop(audio_features);

        merge_generation_timings(timings, &gen_timings);

        if gen_ids.len() != batch_prepared.len() {
            bail!(
                "internal error: generated micro-batch size mismatch: expected={}, got={}",
                batch_prepared.len(),
                gen_ids.len()
            );
        }

        let start_decode = std::time::Instant::now();
        for ((orig_idx, _), ids) in batch.into_iter().zip(gen_ids) {
            let token_count = ids.len();
            let raw = processor.tokenizer.decode(ids.as_slice())?;
            if std::env::var_os("VASR_DEBUG_RAW_ASR").is_some() {
                tracing::debug!(
                    target: "vasr::inference::transcribe",
                    orig_idx,
                    tokens = token_count,
                    raw = raw.as_str(),
                    "raw ASR decode"
                );
            }
            out[orig_idx] = Some(GeneratedRaw {
                text: raw,
                token_count,
            });
        }
        timings.tokenizer_decode_us = timings
            .tokenizer_decode_us
            .saturating_add(duration_to_us(start_decode.elapsed()));
    }

    out.into_iter()
        .enumerate()
        .map(|(i, text)| text.ok_or_else(|| anyhow::anyhow!("missing generated text {i}")))
        .collect()
}

#[cfg(feature = "timing")]
fn run_asr_on_chunks_batched_timed(
    batch: &ChunkAsrBatch<'_>,
    timings: &mut TranscribeTimings,
) -> Result<(Vec<String>, Vec<String>)> {
    if batch.chunks.is_empty() {
        return Ok((vec![], vec![]));
    }
    if batch.prompts.len() != batch.forced_langs_norm.len() {
        bail!(
            "internal error: prompts/forced_langs_norm length mismatch: prompts={} forced_langs_norm={}",
            batch.prompts.len(),
            batch.forced_langs_norm.len()
        );
    }

    if !batch.bucket_by_length {
        let mut per_chunk_lang: Vec<String> = Vec::with_capacity(batch.chunks.len());
        let mut per_chunk_text: Vec<String> = Vec::with_capacity(batch.chunks.len());
        let ordered: Vec<usize> = (0..batch.chunks.len()).collect();
        let index_groups =
            group_chunk_indices(ordered.as_slice(), batch.chunks, batch.max_batch_audio_sec)?;

        for batch_chunk_idx in index_groups {
            timings.batches = timings.batches.saturating_add(1);

            let chunk_batch: Vec<&ChunkItem> = batch_chunk_idx
                .iter()
                .map(|&idx| {
                    batch
                        .chunks
                        .get(idx)
                        .ok_or_else(|| anyhow::anyhow!("missing chunk {idx}"))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(chunk_batch.len());
            for c in &chunk_batch {
                audio_inputs.push(AudioInput::Waveform {
                    samples: c.wav.as_slice(),
                    sample_rate: SAMPLE_RATE_HZ,
                });
            }

            let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(chunk_batch.len());
            for (c, a) in chunk_batch.iter().zip(audio_inputs.iter()) {
                let prompt = batch.prompts.get(c.orig_index).ok_or_else(|| {
                    anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index)
                })?;
                items.push((prompt.as_str(), a));
            }

            let start_prepare = std::time::Instant::now();
            let (prepared, prepare_timings) = batch.prepare_items_timed(items.as_slice())?;
            let prepare_us = start_prepare
                .elapsed()
                .as_micros()
                .min(u128::from(u64::MAX)) as u64;
            timings.processor_prepare_batch_us = timings
                .processor_prepare_batch_us
                .saturating_add(prepare_us);
            add_prepare_batch_timings(timings, &prepare_timings);

            // Snapshot aggregate timings before this batch
            let pre_ae = timings.audio_encoder_us;
            let pre_prefill = timings.generation.prefill_us;
            let pre_prefill_inputs = timings.generation.prefill_inputs_us;
            let pre_prefill_rope = timings.generation.prefill_rope_us;
            let pre_prefill_forward = timings.generation.prefill_forward_us;
            let pre_decode = timings.generation.decode_us;
            let pre_decode_forward = timings.generation.decode_forward_us;
            let pre_decode_graph = timings.generation.decode_graph_replay_us;
            let pre_decode_argmax = timings.generation.decode_argmax_us;
            let pre_decode_metadata = timings.generation.decode_metadata_us;
            let pre_decode_token_tensor = timings.generation.decode_token_tensor_us;
            let pre_decode_embed = timings.generation.decode_embed_us;
            let pre_steps = timings.generation.steps;
            let pre_tokens = timings.generation.tokens_generated;

            let raws = generate_raw_prepared_batch_timed(
                batch.thinker,
                batch.processor,
                batch.device,
                #[cfg(feature = "paged-attn")]
                batch.paged_runtime,
                prepared.as_slice(),
                batch.max_new_tokens,
                batch.decode_max_seqs,
                batch.eos_token_ids,
                timings,
            )?;

            // Per-batch verbose timing
            let batch_ae_ms = (timings.audio_encoder_us - pre_ae) as f64 / 1000.0;
            let batch_prefill_ms = (timings.generation.prefill_us - pre_prefill) as f64 / 1000.0;
            let batch_prefill_inputs_ms =
                (timings.generation.prefill_inputs_us - pre_prefill_inputs) as f64 / 1000.0;
            let batch_prefill_rope_ms =
                (timings.generation.prefill_rope_us - pre_prefill_rope) as f64 / 1000.0;
            let batch_prefill_fwd_ms =
                (timings.generation.prefill_forward_us - pre_prefill_forward) as f64 / 1000.0;
            let batch_prefill_other_ms = batch_prefill_ms
                - batch_prefill_inputs_ms
                - batch_prefill_rope_ms
                - batch_prefill_fwd_ms;
            let batch_decode_ms = (timings.generation.decode_us - pre_decode) as f64 / 1000.0;
            let batch_decode_fwd_ms =
                (timings.generation.decode_forward_us - pre_decode_forward) as f64 / 1000.0;
            let batch_decode_graph_ms =
                (timings.generation.decode_graph_replay_us - pre_decode_graph) as f64 / 1000.0;
            let batch_decode_argmax_ms =
                (timings.generation.decode_argmax_us - pre_decode_argmax) as f64 / 1000.0;
            let batch_decode_metadata_ms =
                (timings.generation.decode_metadata_us - pre_decode_metadata) as f64 / 1000.0;
            let batch_decode_token_tensor_ms = (timings.generation.decode_token_tensor_us
                - pre_decode_token_tensor) as f64
                / 1000.0;
            let batch_decode_embed_ms =
                (timings.generation.decode_embed_us - pre_decode_embed) as f64 / 1000.0;
            let batch_steps = timings.generation.steps - pre_steps;
            let batch_tokens = timings.generation.tokens_generated - pre_tokens;
            let token_counts = raws.iter().map(|r| r.token_count).collect::<Vec<_>>();
            let token_dist = dist_from_usize(token_counts.as_slice());
            let audio_seconds = chunk_batch
                .iter()
                .map(|chunk| chunk_audio_seconds(chunk.wav.len()))
                .collect::<Vec<_>>();
            let audio_dist = dist_from_f32(audio_seconds.as_slice());
            let prepare_norm_ms = prepare_timings.normalize_us as f64 / 1000.0;
            let prepare_token_lookup_ms = prepare_timings.token_lookup_us as f64 / 1000.0;
            let prepare_feature_ms = prepare_timings.feature_extract_us as f64 / 1000.0;
            let prepare_tokenize_ms = prepare_timings.tokenize_expand_us as f64 / 1000.0;
            let prepare_pad_ms = prepare_timings.pad_us as f64 / 1000.0;
            let prepare_other_ms = prepare_us as f64 / 1000.0
                - prepare_norm_ms
                - prepare_token_lookup_ms
                - prepare_feature_ms
                - prepare_tokenize_ms
                - prepare_pad_ms;
            if std::env::var_os("VASR_DEBUG_ASR_BATCH_TIMING").is_some() {
                tracing::debug!(
                    target: "vasr::models::qwen3asr",
                    "[vasr batch {}/{}] items={} | audio_sec min/p50/p90/max/avg={:.1}/{:.1}/{:.1}/{:.1}/{:.1} | prepare={:.1}ms (norm={:.1} tok_lookup={:.1} feat={:.1} tok_expand={:.1} pad={:.1} other={:.1}) | audio_enc={:.1}ms | prefill={:.1}ms (inputs={:.1} rope={:.1} fwd={:.1} other={:.1}) | decode={:.1}ms (graph={:.1} fwd={:.1} argmax={:.1} meta={:.1} tok_tensor={:.1} embed={:.1}) | steps={} tokens={} tok/seq min/p50/p90/max/avg={:.0}/{:.0}/{:.0}/{:.0}/{:.1}",
                    timings.batches,
                    batch.chunks.len(),
                    chunk_batch.len(),
                    audio_dist.min,
                    audio_dist.p50,
                    audio_dist.p90,
                    audio_dist.max,
                    audio_dist.avg,
                    prepare_us as f64 / 1000.0,
                    prepare_norm_ms,
                    prepare_token_lookup_ms,
                    prepare_feature_ms,
                    prepare_tokenize_ms,
                    prepare_pad_ms,
                    prepare_other_ms,
                    batch_ae_ms,
                    batch_prefill_ms,
                    batch_prefill_inputs_ms,
                    batch_prefill_rope_ms,
                    batch_prefill_fwd_ms,
                    batch_prefill_other_ms,
                    batch_decode_ms,
                    batch_decode_graph_ms,
                    batch_decode_fwd_ms,
                    batch_decode_argmax_ms,
                    batch_decode_metadata_ms,
                    batch_decode_token_tensor_ms,
                    batch_decode_embed_ms,
                    batch_steps,
                    batch_tokens,
                    token_dist.min,
                    token_dist.p50,
                    token_dist.p90,
                    token_dist.max,
                    token_dist.avg,
                );
            }
            if std::env::var_os("VASR_DEBUG_ASR_BATCH_TIMING").is_some()
                && token_dist.max >= batch.max_new_tokens as f64
            {
                for (c, raw) in chunk_batch.iter().zip(raws.iter()) {
                    if raw.token_count < batch.max_new_tokens {
                        continue;
                    }
                    tracing::debug!(
                        target: "vasr::models::qwen3asr",
                        "[vasr max-new hit] orig_index={} audio_sec={:.2} offset_sec={:.2} tokens={} raw_preview={:?}",
                        c.orig_index,
                        chunk_audio_seconds(c.wav.len()),
                        chunk_offset_sec(c),
                        raw.token_count,
                        raw_preview(raw.text.as_str()),
                    );
                }
            }

            if raws.len() != chunk_batch.len() {
                bail!(
                    "internal error: raw output batch size mismatch: expected={}, got={}",
                    chunk_batch.len(),
                    raws.len()
                );
            }

            for (c, raw) in chunk_batch.iter().zip(raws.iter()) {
                let forced = batch
                    .forced_langs_norm
                    .get(c.orig_index)
                    .and_then(|x| x.as_deref());
                let (lang, text) = parse_asr_output(Some(raw.text.as_str()), forced);
                per_chunk_lang.push(lang);
                per_chunk_text.push(text);
            }
        }

        return Ok((per_chunk_lang, per_chunk_text));
    }

    let mut indices: Vec<usize> = (0..batch.chunks.len()).collect();
    indices.sort_by(|&a, &b| {
        let la = batch.chunks[a].wav.len();
        let lb = batch.chunks[b].wav.len();
        lb.cmp(&la).then_with(|| a.cmp(&b))
    });

    let mut per_chunk_lang: Vec<Option<String>> = vec![None; batch.chunks.len()];
    let mut per_chunk_text: Vec<Option<String>> = vec![None; batch.chunks.len()];

    let index_groups =
        group_chunk_indices(indices.as_slice(), batch.chunks, batch.max_batch_audio_sec)?;

    for batch_chunk_idx in index_groups {
        timings.batches = timings.batches.saturating_add(1);

        let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(batch_chunk_idx.len());
        for &chunk_idx in &batch_chunk_idx {
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            audio_inputs.push(AudioInput::Waveform {
                samples: c.wav.as_slice(),
                sample_rate: SAMPLE_RATE_HZ,
            });
        }

        let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(batch_chunk_idx.len());
        for (&chunk_idx, a) in batch_chunk_idx.iter().zip(audio_inputs.iter()) {
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let prompt = batch
                .prompts
                .get(c.orig_index)
                .ok_or_else(|| anyhow::anyhow!("missing prompt for orig_index {}", c.orig_index))?;
            items.push((prompt.as_str(), a));
        }

        let start_prepare = std::time::Instant::now();
        let (prepared, prepare_timings) = batch.prepare_items_timed(items.as_slice())?;
        let prepare_us = start_prepare
            .elapsed()
            .as_micros()
            .min(u128::from(u64::MAX)) as u64;
        timings.processor_prepare_batch_us = timings
            .processor_prepare_batch_us
            .saturating_add(prepare_us);
        add_prepare_batch_timings(timings, &prepare_timings);
        if prepared.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: prepare_batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                prepared.len()
            );
        }

        // Snapshot aggregate timings before this batch
        let pre_ae = timings.audio_encoder_us;
        let pre_prefill = timings.generation.prefill_us;
        let pre_prefill_inputs = timings.generation.prefill_inputs_us;
        let pre_prefill_rope = timings.generation.prefill_rope_us;
        let pre_prefill_forward = timings.generation.prefill_forward_us;
        let pre_decode = timings.generation.decode_us;
        let pre_decode_forward = timings.generation.decode_forward_us;
        let pre_decode_graph = timings.generation.decode_graph_replay_us;
        let pre_decode_argmax = timings.generation.decode_argmax_us;
        let pre_decode_metadata = timings.generation.decode_metadata_us;
        let pre_decode_token_tensor = timings.generation.decode_token_tensor_us;
        let pre_decode_embed = timings.generation.decode_embed_us;
        let pre_steps = timings.generation.steps;
        let pre_tokens = timings.generation.tokens_generated;

        let raws = generate_raw_prepared_batch_timed(
            batch.thinker,
            batch.processor,
            batch.device,
            #[cfg(feature = "paged-attn")]
            batch.paged_runtime,
            prepared.as_slice(),
            batch.max_new_tokens,
            batch.decode_max_seqs,
            batch.eos_token_ids,
            timings,
        )?;

        // Per-batch verbose timing
        let batch_ae_ms = (timings.audio_encoder_us - pre_ae) as f64 / 1000.0;
        let batch_prefill_ms = (timings.generation.prefill_us - pre_prefill) as f64 / 1000.0;
        let batch_prefill_inputs_ms =
            (timings.generation.prefill_inputs_us - pre_prefill_inputs) as f64 / 1000.0;
        let batch_prefill_rope_ms =
            (timings.generation.prefill_rope_us - pre_prefill_rope) as f64 / 1000.0;
        let batch_prefill_fwd_ms =
            (timings.generation.prefill_forward_us - pre_prefill_forward) as f64 / 1000.0;
        let batch_prefill_other_ms = batch_prefill_ms
            - batch_prefill_inputs_ms
            - batch_prefill_rope_ms
            - batch_prefill_fwd_ms;
        let batch_decode_ms = (timings.generation.decode_us - pre_decode) as f64 / 1000.0;
        let batch_decode_fwd_ms =
            (timings.generation.decode_forward_us - pre_decode_forward) as f64 / 1000.0;
        let batch_decode_graph_ms =
            (timings.generation.decode_graph_replay_us - pre_decode_graph) as f64 / 1000.0;
        let batch_decode_argmax_ms =
            (timings.generation.decode_argmax_us - pre_decode_argmax) as f64 / 1000.0;
        let batch_decode_metadata_ms =
            (timings.generation.decode_metadata_us - pre_decode_metadata) as f64 / 1000.0;
        let batch_decode_token_tensor_ms =
            (timings.generation.decode_token_tensor_us - pre_decode_token_tensor) as f64 / 1000.0;
        let batch_decode_embed_ms =
            (timings.generation.decode_embed_us - pre_decode_embed) as f64 / 1000.0;
        let batch_steps = timings.generation.steps - pre_steps;
        let batch_tokens = timings.generation.tokens_generated - pre_tokens;
        let token_counts = raws.iter().map(|r| r.token_count).collect::<Vec<_>>();
        let token_dist = dist_from_usize(token_counts.as_slice());
        let audio_seconds = batch_chunk_idx
            .iter()
            .map(|&chunk_idx| chunk_audio_seconds(batch.chunks[chunk_idx].wav.len()))
            .collect::<Vec<_>>();
        let audio_dist = dist_from_f32(audio_seconds.as_slice());
        let prepare_norm_ms = prepare_timings.normalize_us as f64 / 1000.0;
        let prepare_token_lookup_ms = prepare_timings.token_lookup_us as f64 / 1000.0;
        let prepare_feature_ms = prepare_timings.feature_extract_us as f64 / 1000.0;
        let prepare_tokenize_ms = prepare_timings.tokenize_expand_us as f64 / 1000.0;
        let prepare_pad_ms = prepare_timings.pad_us as f64 / 1000.0;
        let prepare_other_ms = prepare_us as f64 / 1000.0
            - prepare_norm_ms
            - prepare_token_lookup_ms
            - prepare_feature_ms
            - prepare_tokenize_ms
            - prepare_pad_ms;
        if std::env::var_os("VASR_DEBUG_ASR_BATCH_TIMING").is_some() {
            tracing::debug!(
                target: "vasr::models::qwen3asr",
                "[vasr batch {}/{}] items={} | audio_sec min/p50/p90/max/avg={:.1}/{:.1}/{:.1}/{:.1}/{:.1} | prepare={:.1}ms (norm={:.1} tok_lookup={:.1} feat={:.1} tok_expand={:.1} pad={:.1} other={:.1}) | audio_enc={:.1}ms | prefill={:.1}ms (inputs={:.1} rope={:.1} fwd={:.1} other={:.1}) | decode={:.1}ms (graph={:.1} fwd={:.1} argmax={:.1} meta={:.1} tok_tensor={:.1} embed={:.1}) | steps={} tokens={} tok/seq min/p50/p90/max/avg={:.0}/{:.0}/{:.0}/{:.0}/{:.1}",
                timings.batches,
                batch.chunks.len(),
                batch_chunk_idx.len(),
                audio_dist.min,
                audio_dist.p50,
                audio_dist.p90,
                audio_dist.max,
                audio_dist.avg,
                prepare_us as f64 / 1000.0,
                prepare_norm_ms,
                prepare_token_lookup_ms,
                prepare_feature_ms,
                prepare_tokenize_ms,
                prepare_pad_ms,
                prepare_other_ms,
                batch_ae_ms,
                batch_prefill_ms,
                batch_prefill_inputs_ms,
                batch_prefill_rope_ms,
                batch_prefill_fwd_ms,
                batch_prefill_other_ms,
                batch_decode_ms,
                batch_decode_graph_ms,
                batch_decode_fwd_ms,
                batch_decode_argmax_ms,
                batch_decode_metadata_ms,
                batch_decode_token_tensor_ms,
                batch_decode_embed_ms,
                batch_steps,
                batch_tokens,
                token_dist.min,
                token_dist.p50,
                token_dist.p90,
                token_dist.max,
                token_dist.avg,
            );
        }
        if std::env::var_os("VASR_DEBUG_ASR_BATCH_TIMING").is_some()
            && token_dist.max >= batch.max_new_tokens as f64
        {
            for (k, raw) in raws.iter().enumerate() {
                if raw.token_count < batch.max_new_tokens {
                    continue;
                }
                let Some(&chunk_idx) = batch_chunk_idx.get(k) else {
                    continue;
                };
                let Some(c) = batch.chunks.get(chunk_idx) else {
                    continue;
                };
                tracing::debug!(
                    target: "vasr::models::qwen3asr",
                    "[vasr max-new hit] chunk_idx={} orig_index={} audio_sec={:.2} offset_sec={:.2} tokens={} raw_preview={:?}",
                    chunk_idx,
                    c.orig_index,
                    chunk_audio_seconds(c.wav.len()),
                    chunk_offset_sec(c),
                    raw.token_count,
                    raw_preview(raw.text.as_str()),
                );
            }
        }
        if raws.len() != batch_chunk_idx.len() {
            bail!(
                "internal error: raw output batch size mismatch: expected={}, got={}",
                batch_chunk_idx.len(),
                raws.len()
            );
        }

        for (k, raw) in raws.iter().enumerate() {
            let chunk_idx = *batch_chunk_idx
                .get(k)
                .ok_or_else(|| anyhow::anyhow!("missing chunk index {k}"))?;
            let c = batch
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            if raw.token_count >= batch.max_new_tokens {
                log_max_new_tokens_hit(chunk_idx, c, raw.token_count, raw.text.as_str());
            }
            let forced = batch
                .forced_langs_norm
                .get(c.orig_index)
                .and_then(|x| x.as_deref());
            let (lang, text) = parse_asr_output(Some(raw.text.as_str()), forced);

            let slot_lang = per_chunk_lang
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output lang slot {chunk_idx}"))?;
            if slot_lang.is_some() {
                bail!("internal error: duplicate language for chunk {chunk_idx}");
            }
            *slot_lang = Some(lang);

            let slot_text = per_chunk_text
                .get_mut(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing output text slot {chunk_idx}"))?;
            if slot_text.is_some() {
                bail!("internal error: duplicate text for chunk {chunk_idx}");
            }
            *slot_text = Some(text);
        }
    }

    let mut langs: Vec<String> = Vec::with_capacity(per_chunk_lang.len());
    for (i, lang) in per_chunk_lang.into_iter().enumerate() {
        langs.push(lang.ok_or_else(|| anyhow::anyhow!("missing chunk language {i}"))?);
    }
    let mut texts: Vec<String> = Vec::with_capacity(per_chunk_text.len());
    for (i, text) in per_chunk_text.into_iter().enumerate() {
        texts.push(text.ok_or_else(|| anyhow::anyhow!("missing chunk text {i}"))?);
    }

    Ok((langs, texts))
}

pub fn transcribe(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    audio: &[AudioInput<'_>],
    opts: &TranscribeOptions,
) -> Result<Vec<AsrTranscription>> {
    if opts.return_timestamps {
        bail!(
            "return_timestamps is not supported by qwen3asr transcribe; run an aligner model separately"
        );
    }

    let contexts = opts.context.broadcast(audio.len(), "context")?;
    let forced_langs = opts.language.broadcast(audio.len(), "language")?;
    let mut forced_langs_norm: Vec<Option<String>> = Vec::with_capacity(forced_langs.len());
    for l in forced_langs {
        forced_langs_norm.push(normalize_forced_language(l)?);
    }

    // The official Python stack always chunks long audio to a maximum duration to avoid OOM.
    // Allow an optional smaller user override via `chunk_max_sec`.
    let chunk_max_sec = opts
        .chunk_max_sec
        .unwrap_or(MAX_ASR_INPUT_SECONDS)
        .min(MAX_ASR_INPUT_SECONDS);

    let eos_token_ids = build_eos_token_ids(processor)?;
    if audio.is_empty() {
        return Ok(vec![]);
    }

    let mut prompts: Vec<String> = Vec::with_capacity(audio.len());
    for (ctx, lang) in contexts.iter().zip(forced_langs_norm.iter()) {
        prompts.push(processor.build_text_prompt(ctx.as_str(), lang.as_deref()));
    }

    let mut chunks: Vec<ChunkItem> = Vec::new();
    for (orig_index, item) in audio.iter().enumerate() {
        let wav = crate::audio::normalize::normalize_audio_input(item)?;
        let parts =
            crate::audio::chunking::split_audio_into_chunks(&wav, SAMPLE_RATE_HZ, chunk_max_sec)?;
        for part in parts {
            chunks.push(ChunkItem {
                orig_index,
                wav: part.wav,
                offset_sec: part.offset_sec,
            });
        }
    }

    let asr_batch = ChunkAsrBatch {
        thinker,
        processor,
        device,
        #[cfg(feature = "paged-attn")]
        paged_runtime,
        chunks: chunks.as_slice(),
        prompts: prompts.as_slice(),
        forced_langs_norm: forced_langs_norm.as_slice(),
        max_new_tokens: opts.max_new_tokens,
        decode_max_seqs: opts.decode_max_seqs,
        max_batch_audio_sec: opts.max_batch_audio_sec,
        bucket_by_length: opts.bucket_by_length,
        eos_token_ids: eos_token_ids.as_slice(),
    };
    let (per_chunk_lang, per_chunk_text) = run_asr_on_chunks_batched(&asr_batch)?;

    if per_chunk_lang.len() != chunks.len() || per_chunk_text.len() != chunks.len() {
        bail!(
            "internal error: per_chunk outputs length mismatch: chunks={} langs={} texts={}",
            chunks.len(),
            per_chunk_lang.len(),
            per_chunk_text.len()
        );
    }

    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); audio.len()];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); audio.len()];

    for idx in 0..chunks.len() {
        let c = chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk {idx}"))?;
        let lang = per_chunk_lang
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk language {idx}"))?;
        let text = per_chunk_text
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk text {idx}"))?;

        let langs = out_langs
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output language bucket"))?;
        langs.push(lang.clone());

        let texts = out_texts
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output text bucket"))?;
        texts.push(text.clone());
    }

    let mut out: Vec<AsrTranscription> = Vec::with_capacity(audio.len());
    for (langs, texts) in out_langs.into_iter().zip(out_texts) {
        out.push(AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps: None,
        });
    }

    Ok(out)
}

pub fn encode_audio_features_to_cpu(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    audio: &[AudioInput<'_>],
    opts: &TranscribeOptions,
) -> Result<EncodedAsrBatch> {
    if opts.return_timestamps {
        bail!(
            "return_timestamps is not supported by qwen3asr staged transcribe; run an aligner model separately"
        );
    }

    let contexts = opts.context.broadcast(audio.len(), "context")?;
    let forced_langs = opts.language.broadcast(audio.len(), "language")?;
    let mut forced_langs_norm: Vec<Option<String>> = Vec::with_capacity(forced_langs.len());
    for l in forced_langs {
        forced_langs_norm.push(normalize_forced_language(l)?);
    }
    let chunk_max_sec = opts
        .chunk_max_sec
        .unwrap_or(MAX_ASR_INPUT_SECONDS)
        .min(MAX_ASR_INPUT_SECONDS);

    let mut prompts: Vec<String> = Vec::with_capacity(audio.len());
    for (ctx, lang) in contexts.iter().zip(forced_langs_norm.iter()) {
        prompts.push(processor.build_text_prompt(ctx.as_str(), lang.as_deref()));
    }

    let mut chunks: Vec<ChunkItem> = Vec::new();
    for (orig_index, item) in audio.iter().enumerate() {
        let wav = crate::audio::normalize::normalize_audio_input(item)?;
        let parts =
            crate::audio::chunking::split_audio_into_chunks(&wav, SAMPLE_RATE_HZ, chunk_max_sec)?;
        for part in parts {
            chunks.push(ChunkItem {
                orig_index,
                wav: part.wav,
                offset_sec: part.offset_sec,
            });
        }
    }

    if chunks.is_empty() {
        return Ok(EncodedAsrBatch {
            audio_len: audio.len(),
            chunks,
            chunk_audio_seconds: Vec::new(),
            forced_langs_norm,
            groups: Vec::new(),
            timings: AudioFeatureEncodeTimings::default(),
        });
    }
    let chunk_audio_seconds = chunks
        .iter()
        .map(|chunk| chunk.wav.len() as f64 / SAMPLE_RATE_HZ as f64)
        .collect::<Vec<_>>();

    let mut ordered_indices: Vec<usize> = (0..chunks.len()).collect();
    if opts.bucket_by_length {
        ordered_indices.sort_by(|&a, &b| {
            let la = chunks[a].wav.len();
            let lb = chunks[b].wav.len();
            lb.cmp(&la).then_with(|| a.cmp(&b))
        });
    }
    let index_groups = group_chunk_indices(
        ordered_indices.as_slice(),
        &chunks,
        opts.max_batch_audio_sec,
    )?;
    let mut groups = Vec::with_capacity(index_groups.len());
    let mut timings = AudioFeatureEncodeTimings::default();
    for chunk_indices in index_groups {
        let mut audio_inputs: Vec<AudioInput<'_>> = Vec::with_capacity(chunk_indices.len());
        for &chunk_idx in &chunk_indices {
            let chunk = chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            audio_inputs.push(AudioInput::Waveform {
                samples: chunk.wav.as_slice(),
                sample_rate: SAMPLE_RATE_HZ,
            });
        }

        let mut items: Vec<(&str, &AudioInput<'_>)> = Vec::with_capacity(chunk_indices.len());
        for (&chunk_idx, audio_input) in chunk_indices.iter().zip(audio_inputs.iter()) {
            let chunk = chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            let prompt = prompts.get(chunk.orig_index).ok_or_else(|| {
                anyhow::anyhow!("missing prompt for orig_index {}", chunk.orig_index)
            })?;
            items.push((prompt.as_str(), audio_input));
        }

        let start_prepare = std::time::Instant::now();
        let prepared = prepare_items_for_runtime(
            processor,
            #[cfg(feature = "paged-attn")]
            paged_runtime,
            items.as_slice(),
        )?;
        let prepare_us = duration_to_us(start_prepare.elapsed());
        if prepared.len() != chunk_indices.len() {
            bail!(
                "internal error: staged prepare size mismatch: expected={}, got={}",
                chunk_indices.len(),
                prepared.len()
            );
        }
        let (audio_features_cpu, mut group_timings) = encode_prepared_audio_features_cpu(
            thinker,
            device,
            #[cfg(feature = "paged-attn")]
            paged_runtime,
            prepared.as_slice(),
        )?;
        group_timings.prepare_us = group_timings.prepare_us.saturating_add(prepare_us);
        timings.add_assign(group_timings);
        groups.push(EncodedPreparedGroup {
            chunk_indices,
            prepared,
            audio_features_cpu,
        });
    }

    Ok(EncodedAsrBatch {
        audio_len: audio.len(),
        chunks,
        chunk_audio_seconds,
        forced_langs_norm,
        groups,
        timings,
    })
}

pub fn decode_cpu_audio_features(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    encoded: EncodedAsrBatch,
    opts: &TranscribeOptions,
) -> Result<Vec<AsrTranscription>> {
    let eos_token_ids = build_eos_token_ids(processor)?;
    let mut per_chunk_lang: Vec<Option<String>> = vec![None; encoded.chunks.len()];
    let mut per_chunk_text: Vec<Option<String>> = vec![None; encoded.chunks.len()];

    let mut prepared = Vec::new();
    let mut chunk_indices = Vec::new();
    let mut audio_feature_parts = Vec::new();
    for group in encoded.groups {
        prepared.extend(group.prepared);
        chunk_indices.extend(group.chunk_indices);
        audio_feature_parts.push(group.audio_features_cpu);
    }

    if !prepared.is_empty() {
        let audio_features = if audio_feature_parts.len() == 1 {
            audio_feature_parts
                .pop()
                .ok_or_else(|| anyhow::anyhow!("missing staged audio features"))?
        } else {
            Tensor::cat(audio_feature_parts.as_slice(), 0)?
        };
        let raws = generate_raw_prepared_batch_with_audio_features(
            thinker,
            processor,
            device,
            #[cfg(feature = "paged-attn")]
            paged_runtime,
            prepared.as_slice(),
            &audio_features,
            opts.max_new_tokens,
            opts.decode_max_seqs,
            eos_token_ids.as_slice(),
        )?;
        if raws.len() != chunk_indices.len() {
            bail!(
                "internal error: staged raw output size mismatch: expected={}, got={}",
                chunk_indices.len(),
                raws.len()
            );
        }
        for (&chunk_idx, raw) in chunk_indices.iter().zip(raws.iter()) {
            let chunk = encoded
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            if raw.token_count >= opts.max_new_tokens {
                log_max_new_tokens_hit(chunk_idx, chunk, raw.token_count, raw.text.as_str());
            }
            let forced = encoded
                .forced_langs_norm
                .get(chunk.orig_index)
                .and_then(|x| x.as_deref());
            let (lang, text) = parse_asr_output(Some(raw.text.as_str()), forced);
            per_chunk_lang[chunk_idx] = Some(lang);
            per_chunk_text[chunk_idx] = Some(text);
        }
    }

    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); encoded.audio_len];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); encoded.audio_len];
    for (idx, chunk) in encoded.chunks.iter().enumerate() {
        let lang = per_chunk_lang[idx]
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing staged chunk language {idx}"))?;
        let text = per_chunk_text[idx]
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing staged chunk text {idx}"))?;
        out_langs[chunk.orig_index].push(lang);
        out_texts[chunk.orig_index].push(text);
    }

    Ok(out_langs
        .into_iter()
        .zip(out_texts)
        .map(|(langs, texts)| AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps: None,
        })
        .collect())
}

#[cfg(feature = "timing")]
pub fn decode_cpu_audio_features_timed(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    encoded: EncodedAsrBatch,
    opts: &TranscribeOptions,
) -> Result<(Vec<AsrTranscription>, TranscribeTimings)> {
    let mut timings = TranscribeTimings::default();
    let start_total = std::time::Instant::now();
    let eos_token_ids = build_eos_token_ids(processor)?;
    let mut per_chunk_lang: Vec<Option<String>> = vec![None; encoded.chunks.len()];
    let mut per_chunk_text: Vec<Option<String>> = vec![None; encoded.chunks.len()];

    let mut prepared = Vec::new();
    let mut chunk_indices = Vec::new();
    let mut audio_feature_parts = Vec::new();
    for group in encoded.groups {
        prepared.extend(group.prepared);
        chunk_indices.extend(group.chunk_indices);
        audio_feature_parts.push(group.audio_features_cpu);
    }
    timings.chunks = prepared.len();

    if !prepared.is_empty() {
        let audio_features = if audio_feature_parts.len() == 1 {
            audio_feature_parts
                .pop()
                .ok_or_else(|| anyhow::anyhow!("missing staged audio features"))?
        } else {
            Tensor::cat(audio_feature_parts.as_slice(), 0)?
        };
        let raws = generate_raw_prepared_batch_with_audio_features_timed(
            thinker,
            processor,
            device,
            #[cfg(feature = "paged-attn")]
            paged_runtime,
            prepared.as_slice(),
            &audio_features,
            opts.max_new_tokens,
            opts.decode_max_seqs,
            eos_token_ids.as_slice(),
            &mut timings,
        )?;
        if raws.len() != chunk_indices.len() {
            bail!(
                "internal error: staged raw output size mismatch: expected={}, got={}",
                chunk_indices.len(),
                raws.len()
            );
        }
        for (&chunk_idx, raw) in chunk_indices.iter().zip(raws.iter()) {
            let chunk = encoded
                .chunks
                .get(chunk_idx)
                .ok_or_else(|| anyhow::anyhow!("missing chunk {chunk_idx}"))?;
            if raw.token_count >= opts.max_new_tokens {
                log_max_new_tokens_hit(chunk_idx, chunk, raw.token_count, raw.text.as_str());
            }
            let forced = encoded
                .forced_langs_norm
                .get(chunk.orig_index)
                .and_then(|x| x.as_deref());
            let (lang, text) = parse_asr_output(Some(raw.text.as_str()), forced);
            per_chunk_lang[chunk_idx] = Some(lang);
            per_chunk_text[chunk_idx] = Some(text);
        }
    }

    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); encoded.audio_len];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); encoded.audio_len];
    for (idx, chunk) in encoded.chunks.iter().enumerate() {
        let lang = per_chunk_lang[idx]
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing staged chunk language {idx}"))?;
        let text = per_chunk_text[idx]
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing staged chunk text {idx}"))?;
        out_langs[chunk.orig_index].push(lang);
        out_texts[chunk.orig_index].push(text);
    }

    timings.total_us = duration_to_us(start_total.elapsed());
    let out = out_langs
        .into_iter()
        .zip(out_texts)
        .map(|(langs, texts)| AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps: None,
        })
        .collect();
    Ok((out, timings))
}

#[cfg(feature = "timing")]
#[allow(clippy::too_many_arguments)]
fn generate_raw_prepared_batch_with_audio_features_timed(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    prepared: &[PreparedInputs],
    audio_features: &Tensor,
    max_new_tokens: usize,
    _decode_max_seqs: usize,
    eos_token_ids: &[u32],
    timings: &mut TranscribeTimings,
) -> Result<Vec<GeneratedRaw>> {
    if prepared.is_empty() {
        return Ok(vec![]);
    }
    let max_new_tokens = if max_new_tokens == 0 {
        256
    } else {
        max_new_tokens
    };
    let ids_rows: Vec<&[u32]> = prepared.iter().map(|p| p.input_ids.as_slice()).collect();
    let attn_rows: Vec<&[u32]> = prepared
        .iter()
        .map(|p| p.attention_mask.as_slice())
        .collect();
    #[cfg(feature = "paged-attn")]
    let pad_id = paged_token_pad_id(processor)?;
    #[cfg(feature = "paged-attn")]
    let (gen_ids, gen_timings) = if let Some(runtime) = paged_runtime {
        use crate::inference::batch_scheduler::run_paged_prepared_batch_timed;
        run_paged_prepared_batch_timed(
            thinker,
            device,
            runtime,
            prepared,
            audio_features,
            max_new_tokens,
            eos_token_ids,
            _decode_max_seqs,
            pad_id,
            None,
        )?
    } else {
        let _ = (ids_rows.as_slice(), attn_rows.as_slice());
        bail!("paged-attn build requires a shared paged runtime for timed staged ASR generation")
    };
    #[cfg(not(feature = "paged-attn"))]
    let (gen_ids, gen_timings) = greedy_generate_cached_batch_timed(
        thinker,
        device,
        ids_rows.as_slice(),
        attn_rows.as_slice(),
        Some(audio_features),
        max_new_tokens,
        eos_token_ids,
    )?;
    merge_generation_timings(timings, &gen_timings);
    timings.batches = timings.batches.saturating_add(1);

    if gen_ids.len() != prepared.len() {
        bail!(
            "internal error: generated batch size mismatch: expected={}, got={}",
            prepared.len(),
            gen_ids.len()
        );
    }
    let start_decode = std::time::Instant::now();
    let out = gen_ids
        .into_iter()
        .map(|ids| {
            let token_count = ids.len();
            let text = processor.tokenizer.decode(ids.as_slice())?;
            Ok(GeneratedRaw { text, token_count })
        })
        .collect::<Result<Vec<_>>>()?;
    timings.tokenizer_decode_us = timings
        .tokenizer_decode_us
        .saturating_add(duration_to_us(start_decode.elapsed()));
    Ok(out)
}

#[cfg(feature = "paged-attn")]
pub fn decode_cpu_audio_features_dynamic(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    paged_runtime: Option<&SharedPagedCacheRuntime>,
    encoded: EncodedAsrBatch,
    poll_encoded: &mut (dyn FnMut(bool) -> Result<EncodedAsrBatchPoll> + Send),
    opts: &TranscribeOptions,
) -> Result<Vec<Vec<AsrTranscription>>> {
    let Some(runtime) = paged_runtime else {
        bail!("paged-attn build requires a shared paged runtime for dynamic staged ASR generation");
    };
    let eos_token_ids = build_eos_token_ids(processor)?;
    let (initial, initial_meta) = encoded.into_prepared_audio_batch_and_metadata()?;
    let mut source = DynamicPreparedSource::new(initial_meta, initial.prepared.len(), poll_encoded);
    let mut poll_prepared =
        |may_block| -> Result<PreparedAudioBatchPoll> { source.poll_prepared(may_block) };
    let pad_id = paged_token_pad_id(processor)?;
    let config = crate::inference::batch_scheduler::AsrBatchSchedulerConfig::from_decode_max_seqs(
        opts.decode_max_seqs,
    )
    .with_max_new_tokens(opts.max_new_tokens);
    let mut scheduler = crate::inference::batch_scheduler::AsrBatchScheduler::new(config);
    let ids = scheduler.run_continuous_prepared_dynamic_async(
        thinker,
        device,
        runtime,
        initial,
        &mut poll_prepared,
        eos_token_ids.as_slice(),
        pad_id,
        #[cfg(feature = "timing")]
        None,
    )?;
    let _ = poll_prepared;
    let (metas, batch_lens, _chunks_seen, _batches_seen) = source.into_parts();
    decode_ids_with_metadata(processor, ids, metas, batch_lens)
}

#[cfg(all(feature = "paged-attn", feature = "timing"))]
pub fn decode_cpu_audio_features_dynamic_timed(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    paged_runtime: Option<&SharedPagedCacheRuntime>,
    encoded: EncodedAsrBatch,
    poll_encoded: &mut (dyn FnMut(bool) -> Result<EncodedAsrBatchPoll> + Send),
    opts: &TranscribeOptions,
) -> Result<(Vec<Vec<AsrTranscription>>, TranscribeTimings)> {
    let Some(runtime) = paged_runtime else {
        bail!("paged-attn build requires a shared paged runtime for dynamic staged ASR generation");
    };
    let start_total = std::time::Instant::now();
    let mut timings = TranscribeTimings::default();
    let eos_token_ids = build_eos_token_ids(processor)?;
    let (initial, initial_meta) = encoded.into_prepared_audio_batch_and_metadata()?;
    let mut source = DynamicPreparedSource::new(initial_meta, initial.prepared.len(), poll_encoded);
    let mut poll_prepared =
        |may_block| -> Result<PreparedAudioBatchPoll> { source.poll_prepared(may_block) };
    let pad_id = paged_token_pad_id(processor)?;
    let config = crate::inference::batch_scheduler::AsrBatchSchedulerConfig::from_decode_max_seqs(
        opts.decode_max_seqs,
    )
    .with_max_new_tokens(opts.max_new_tokens);
    let mut scheduler = crate::inference::batch_scheduler::AsrBatchScheduler::new(config);
    let ids = scheduler.run_continuous_prepared_dynamic_async(
        thinker,
        device,
        runtime,
        initial,
        &mut poll_prepared,
        eos_token_ids.as_slice(),
        pad_id,
        Some(&mut timings.generation),
    )?;
    let _ = poll_prepared;
    let (metas, batch_lens, chunks_seen, batches_seen) = source.into_parts();
    timings.chunks = timings.chunks.saturating_add(chunks_seen);
    timings.batches = timings.batches.saturating_add(batches_seen);

    timings.generation.tokens_generated = ids.iter().map(Vec::len).sum();
    let start_decode = std::time::Instant::now();
    let grouped = decode_ids_with_metadata(processor, ids, metas, batch_lens)?;
    timings.tokenizer_decode_us = timings
        .tokenizer_decode_us
        .saturating_add(duration_to_us(start_decode.elapsed()));
    timings.total_us = duration_to_us(start_total.elapsed());
    Ok((grouped, timings))
}

#[cfg(feature = "timing")]
pub fn transcribe_timed(
    thinker: &ThinkerForConditionalGeneration,
    processor: &AsrProcessor,
    device: &Device,
    #[cfg(feature = "paged-attn")] paged_runtime: Option<&SharedPagedCacheRuntime>,
    audio: &[AudioInput<'_>],
    opts: &TranscribeOptions,
) -> Result<(Vec<AsrTranscription>, TranscribeTimings)> {
    if opts.return_timestamps {
        bail!(
            "return_timestamps is not supported by qwen3asr transcribe_timed; run an aligner model separately"
        );
    }

    let start_total = std::time::Instant::now();
    let mut timings = TranscribeTimings::default();

    let contexts = opts.context.broadcast(audio.len(), "context")?;
    let forced_langs = opts.language.broadcast(audio.len(), "language")?;
    let mut forced_langs_norm: Vec<Option<String>> = Vec::with_capacity(forced_langs.len());
    for l in forced_langs {
        forced_langs_norm.push(normalize_forced_language(l)?);
    }

    let chunk_max_sec = opts
        .chunk_max_sec
        .unwrap_or(MAX_ASR_INPUT_SECONDS)
        .min(MAX_ASR_INPUT_SECONDS);

    let eos_token_ids = build_eos_token_ids(processor)?;
    if audio.is_empty() {
        return Ok((vec![], timings));
    }

    let mut prompts: Vec<String> = Vec::with_capacity(audio.len());
    for (ctx, lang) in contexts.iter().zip(forced_langs_norm.iter()) {
        prompts.push(processor.build_text_prompt(ctx.as_str(), lang.as_deref()));
    }

    let mut chunks: Vec<ChunkItem> = Vec::new();
    for (orig_index, item) in audio.iter().enumerate() {
        let start_norm = std::time::Instant::now();
        let wav = crate::audio::normalize::normalize_audio_input(item)?;
        timings.audio_normalize_us = timings
            .audio_normalize_us
            .saturating_add(duration_to_us(start_norm.elapsed()));

        let start_chunk = std::time::Instant::now();
        let parts =
            crate::audio::chunking::split_audio_into_chunks(&wav, SAMPLE_RATE_HZ, chunk_max_sec)?;
        timings.audio_chunking_us = timings
            .audio_chunking_us
            .saturating_add(duration_to_us(start_chunk.elapsed()));

        for part in parts {
            chunks.push(ChunkItem {
                orig_index,
                wav: part.wav,
                offset_sec: part.offset_sec,
            });
        }
    }
    timings.chunks = chunks.len();

    let asr_batch = ChunkAsrBatch {
        thinker,
        processor,
        device,
        #[cfg(feature = "paged-attn")]
        paged_runtime,
        chunks: chunks.as_slice(),
        prompts: prompts.as_slice(),
        forced_langs_norm: forced_langs_norm.as_slice(),
        max_new_tokens: opts.max_new_tokens,
        decode_max_seqs: opts.decode_max_seqs,
        max_batch_audio_sec: opts.max_batch_audio_sec,
        bucket_by_length: opts.bucket_by_length,
        eos_token_ids: eos_token_ids.as_slice(),
    };
    let (per_chunk_lang, per_chunk_text) =
        run_asr_on_chunks_batched_timed(&asr_batch, &mut timings)?;

    if per_chunk_lang.len() != chunks.len() || per_chunk_text.len() != chunks.len() {
        bail!(
            "internal error: per_chunk outputs length mismatch: chunks={} langs={} texts={}",
            chunks.len(),
            per_chunk_lang.len(),
            per_chunk_text.len()
        );
    }

    let mut out_langs: Vec<Vec<String>> = vec![Vec::new(); audio.len()];
    let mut out_texts: Vec<Vec<String>> = vec![Vec::new(); audio.len()];

    for idx in 0..chunks.len() {
        let c = chunks
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk {idx}"))?;
        let lang = per_chunk_lang
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk language {idx}"))?;
        let text = per_chunk_text
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing chunk text {idx}"))?;

        let langs = out_langs
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output language bucket"))?;
        langs.push(lang.clone());

        let texts = out_texts
            .get_mut(c.orig_index)
            .ok_or_else(|| anyhow::anyhow!("missing output text bucket"))?;
        texts.push(text.clone());
    }

    let mut out: Vec<AsrTranscription> = Vec::with_capacity(audio.len());
    for (langs, texts) in out_langs.into_iter().zip(out_texts) {
        out.push(AsrTranscription {
            language: merge_languages(langs),
            text: texts.join(""),
            timestamps: None,
        });
    }

    timings.total_us = duration_to_us(start_total.elapsed());
    Ok((out, timings))
}

#[cfg(test)]
mod batch_grouping_tests {
    use super::{ChunkItem, chunk_audio_seconds, group_chunk_indices};

    fn chunk_with_seconds(seconds: f32) -> ChunkItem {
        let samples = (seconds * 16_000.0).round() as usize;
        ChunkItem {
            orig_index: 0,
            wav: vec![0.0; samples.max(1)],
            offset_sec: 0.0,
        }
    }

    #[test]
    fn group_chunk_indices_respects_audio_seconds_budget() -> anyhow::Result<()> {
        let chunks = vec![
            chunk_with_seconds(20.0),
            chunk_with_seconds(25.0),
            chunk_with_seconds(10.0),
            chunk_with_seconds(30.0),
        ];
        let indices = vec![0, 1, 2, 3];
        let groups = group_chunk_indices(indices.as_slice(), &chunks, 60.0)?;
        assert_eq!(groups, vec![vec![0, 1, 2], vec![3]]);
        Ok(())
    }

    #[test]
    fn group_chunk_indices_keeps_oversized_chunk_alone() -> anyhow::Result<()> {
        let chunks = vec![chunk_with_seconds(65.0), chunk_with_seconds(5.0)];
        let groups = group_chunk_indices(&[0, 1], &chunks, 60.0)?;
        assert_eq!(groups, vec![vec![0], vec![1]]);
        assert!(chunk_audio_seconds(chunks[0].wav.len()) > 60.0);
        Ok(())
    }

    #[test]
    fn group_chunk_indices_without_audio_budget_keeps_one_pool() -> anyhow::Result<()> {
        let chunks = vec![
            chunk_with_seconds(1.0),
            chunk_with_seconds(2.0),
            chunk_with_seconds(3.0),
            chunk_with_seconds(4.0),
        ];
        let groups = group_chunk_indices(&[0, 1, 2, 3], &chunks, 0.0)?;
        assert_eq!(groups, vec![vec![0, 1, 2, 3]]);
        Ok(())
    }
}

#[cfg(test)]
#[path = "encoded_selection_tests.rs"]
mod encoded_selection_tests;
