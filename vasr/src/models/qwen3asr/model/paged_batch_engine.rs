//! Paged-attention batch engine: prefill + single decode steps for continuous batching.
//!
//! `greedy_generate_paged_batch` delegates here. Schedulers call `paged_batch_prefill` and
//! `paged_batch_decode_step` directly to drive batch membership across steps.

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

use crate::model::attention;
use crate::model::generation::{
    argmax_token_ids_from_logits, attention_masks_are_right_padded, count_audio_placeholders_batch,
    normalize_batch_for_paged_prefill, position_ids_for_step,
};
use crate::model::paged_cache_runtime::{PagedCacheRuntime, SharedPagedCacheRuntime};
use crate::model::paged_kv_cache::{PagedInputMetadata, PagedKvCache};
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::model::thinker::get_rope_index;
use vasr_quant::isq_linear::set_linear_is_prefill;

const DECODE_TOKEN_REPEAT_LIMIT: usize = 32;
const DECODE_PATTERN_REPEAT_LIMIT: usize = 16;
const DECODE_MAX_REPEAT_PATTERN_LEN: usize = 8;

fn has_repeated_tail(tokens: &[u32]) -> bool {
    if tokens.len() < DECODE_TOKEN_REPEAT_LIMIT {
        return false;
    }
    if let Some(&last) = tokens.last() {
        let repeated = tokens.iter().rev().take_while(|&&tok| tok == last).count();
        if repeated >= DECODE_TOKEN_REPEAT_LIMIT {
            return true;
        }
    }
    for pattern_len in 2..=DECODE_MAX_REPEAT_PATTERN_LEN {
        let needed = pattern_len.saturating_mul(DECODE_PATTERN_REPEAT_LIMIT);
        if tokens.len() < needed {
            continue;
        }
        let tail = &tokens[tokens.len() - pattern_len..];
        let repeated = (0..DECODE_PATTERN_REPEAT_LIMIT).all(|repeat_idx| {
            let start = tokens.len() - pattern_len * (repeat_idx + 1);
            &tokens[start..start + pattern_len] == tail
        });
        if repeated {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone)]
pub struct PagedBatchConfig {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    /// First `request_id` assigned to row 0; row `i` uses `request_id_base + i`.
    pub request_id_base: usize,
    /// Optional per-row total KV token capacity. When set, prefill allocates
    /// enough blocks for the full prompt + decode budget up front.
    pub token_capacities: Option<Vec<usize>>,
}

#[derive(Debug)]
pub struct PagedBatchState {
    pub batch: usize,
    pub seq_len: usize,
    pub prompt_lens_usize: Vec<usize>,
    pub prompt_lens_i64: Vec<i64>,
    pub request_ids: Vec<usize>,
    pub block_tables: Vec<Vec<usize>>,
    pub next_ids: Vec<u32>,
    pub generated: Vec<Vec<u32>>,
    pub finished: Vec<bool>,
    pub decode_step: usize,
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    pub token_capacities: Vec<usize>,
}

impl PagedBatchState {
    pub fn all_finished(&self) -> bool {
        self.finished.iter().all(|&done| done)
    }

    pub fn active_indices(&self) -> Vec<usize> {
        (0..self.batch).filter(|&i| !self.finished[i]).collect()
    }
}

/// Per-sequence decode state for continuous batching (each slot tracks its own step).
#[derive(Debug, Clone)]
pub struct PagedDecodeSlot {
    pub request_id: usize,
    pub prompt_len_usize: usize,
    pub prompt_len_i64: i64,
    pub block_table: Vec<usize>,
    pub next_id: u32,
    pub generated: Vec<u32>,
    pub finished: bool,
    pub decode_step: usize,
    pub token_capacity: usize,
}

impl PagedDecodeSlot {
    pub fn from_prefill_state(state: &PagedBatchState, row: usize) -> Result<Self> {
        Ok(Self {
            request_id: *state
                .request_ids
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing request_id for row {row}"))?,
            prompt_len_usize: *state
                .prompt_lens_usize
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing prompt_len for row {row}"))?,
            prompt_len_i64: *state
                .prompt_lens_i64
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing prompt_len_i64 for row {row}"))?,
            block_table: state
                .block_tables
                .get(row)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing block table for row {row}"))?,
            next_id: *state
                .next_ids
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing next_id for row {row}"))?,
            generated: Vec::new(),
            finished: *state.finished.get(row).unwrap_or(&false),
            decode_step: 0,
            token_capacity: *state
                .token_capacities
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("missing token capacity for row {row}"))?,
        })
    }
}

pub fn paged_batch_free_slots(runtime: &mut PagedCacheRuntime, request_ids: &[usize]) {
    runtime.manager_mut().free_many(request_ids);
}

fn decode_step_required_tokens(
    prompt_len: usize,
    generated_len_before_step: usize,
) -> Result<usize> {
    prompt_len
        .checked_add(generated_len_before_step)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("paged slot token count overflow"))
}

fn cuda_packed_prefill_enabled() -> bool {
    std::env::var_os("VASR_DISABLE_CUDA_PACKED_PREFILL").is_none()
}

pub fn paged_batch_prefill_row(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    input_ids: &[u32],
    attention_mask: &[u32],
    audio_features: Option<&Tensor>,
    request_id: usize,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
) -> Result<PagedDecodeSlot> {
    let config = PagedBatchConfig {
        max_new_tokens,
        eos_token_ids: eos_token_ids.to_vec(),
        request_id_base: request_id,
        token_capacities: None,
    };
    let state = paged_batch_prefill(
        thinker,
        device,
        runtime,
        &[input_ids],
        &[attention_mask],
        audio_features,
        &config,
    )?;
    PagedDecodeSlot::from_prefill_state(&state, 0)
}

/// Greedy decode step for `slots` at a shared `step` (all slots must share the same `decode_step`).
pub fn paged_batch_decode_slots_at_step(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    slots: &mut [PagedDecodeSlot],
    step: usize,
    eos_token_ids: &[u32],
) -> Result<()> {
    if slots.is_empty() {
        return Ok(());
    }

    let eos_fill_id = *eos_token_ids
        .first()
        .ok_or_else(|| anyhow::anyhow!("eos_token_ids is empty"))?;

    let forward_indices: Vec<usize> = (0..slots.len())
        .filter(|&i| !slots[i].finished && slots[i].decode_step == step)
        .collect();
    if forward_indices.is_empty() {
        return Ok(());
    }

    for &i in &forward_indices {
        let slot = &slots[i];
        let num_tokens = decode_step_required_tokens(slot.prompt_len_usize, slot.generated.len())?;
        if num_tokens > slot.token_capacity {
            bail!(
                "paged decode exceeded allocated KV capacity: request_id={} needed_tokens={} capacity_tokens={} prompt_tokens={} generated_tokens={}",
                slot.request_id,
                num_tokens,
                slot.token_capacity,
                slot.prompt_len_usize,
                slot.generated.len()
            );
        }
        let allocated_blocks = runtime.manager().blocks_for_request(slot.request_id);
        let required_blocks = num_tokens.div_ceil(runtime.manager().block_size());
        if allocated_blocks < required_blocks {
            bail!(
                "paged decode missing preallocated KV blocks: request_id={} needed_blocks={} allocated_blocks={} needed_tokens={} capacity_tokens={}",
                slot.request_id,
                required_blocks,
                allocated_blocks,
                num_tokens,
                slot.token_capacity
            );
        }
    }

    let mut active_indices = Vec::with_capacity(forward_indices.len());
    for &i in &forward_indices {
        let tok = slots[i].next_id;
        if eos_token_ids.contains(&tok) {
            slots[i].finished = true;
            slots[i].decode_step = slots[i].decode_step.saturating_add(1);
            continue;
        }
        slots[i].generated.push(tok);
        if has_repeated_tail(&slots[i].generated) {
            slots[i].finished = true;
            slots[i].decode_step = slots[i].decode_step.saturating_add(1);
            continue;
        }
        active_indices.push(i);
    }

    if active_indices.is_empty() {
        for i in 0..slots.len() {
            if !slots[i].finished && slots[i].decode_step == step {
                slots[i].decode_step = slots[i].decode_step.saturating_add(1);
            }
        }
        return Ok(());
    }

    let tokens_in: Vec<u32> = active_indices
        .iter()
        .map(|&i| {
            slots[i]
                .generated
                .last()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing generated token for slot {i}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let active_batch = active_indices.len();
    let input_ids_new = Tensor::from_vec(tokens_in, (active_batch, 1usize), device)?;

    let prompt_lens_active: Vec<i64> = active_indices
        .iter()
        .map(|&i| slots[i].prompt_len_i64)
        .collect();
    let position_ids_new = position_ids_for_step(prompt_lens_active.as_slice(), step, device)?;

    let request_ids: Vec<usize> = active_indices
        .iter()
        .map(|&i| slots[i].request_id)
        .collect();
    let block_tables_active = runtime.block_tables_for(&request_ids)?;
    let prompt_lens_active_usize: Vec<usize> = active_indices
        .iter()
        .map(|&i| slots[i].prompt_len_usize)
        .collect();
    let input_metadata = runtime.cache().decode_metadata_for_batch_step(
        &block_tables_active,
        prompt_lens_active_usize.as_slice(),
        step,
        device,
    )?;

    let logits_new = paged_batch_decode_forward(
        thinker,
        runtime,
        device,
        &input_ids_new,
        &position_ids_new,
        &input_metadata,
    )?;
    let next_active = argmax_token_ids_from_logits(&logits_new.squeeze(1)?, active_batch)?;
    for (slot_idx, &i) in active_indices.iter().enumerate() {
        slots[i].next_id = *next_active
            .get(slot_idx)
            .ok_or_else(|| anyhow::anyhow!("missing argmax for slot index {slot_idx}"))?;
        slots[i].decode_step = slots[i].decode_step.saturating_add(1);
    }

    // Silence unused when eos_fill would be used in non-compact paths.
    let _ = eos_fill_id;
    Ok(())
}

pub fn paged_batch_decode_slots_at_step_shared(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    slots: &mut [PagedDecodeSlot],
    step: usize,
    eos_token_ids: &[u32],
) -> Result<()> {
    if slots.is_empty() {
        return Ok(());
    }
    let eos_fill_id = *eos_token_ids
        .first()
        .ok_or_else(|| anyhow::anyhow!("eos_token_ids is empty"))?;

    let forward_indices: Vec<usize> = (0..slots.len())
        .filter(|&i| !slots[i].finished && slots[i].decode_step == step)
        .collect();
    if forward_indices.is_empty() {
        return Ok(());
    }

    let (cache, block_tables_active, prompt_lens_active_usize) = {
        let runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let block_size = runtime_guard.manager().block_size();
        for &i in &forward_indices {
            let slot = &slots[i];
            let num_tokens =
                decode_step_required_tokens(slot.prompt_len_usize, slot.generated.len())?;
            if num_tokens > slot.token_capacity {
                bail!(
                    "paged decode exceeded allocated KV capacity: request_id={} needed_tokens={} capacity_tokens={} prompt_tokens={} generated_tokens={}",
                    slot.request_id,
                    num_tokens,
                    slot.token_capacity,
                    slot.prompt_len_usize,
                    slot.generated.len()
                );
            }
            let allocated_blocks = runtime_guard.manager().blocks_for_request(slot.request_id);
            let required_blocks = num_tokens.div_ceil(block_size);
            if allocated_blocks < required_blocks {
                bail!(
                    "paged decode missing preallocated KV blocks: request_id={} needed_blocks={} allocated_blocks={} needed_tokens={} capacity_tokens={}",
                    slot.request_id,
                    required_blocks,
                    allocated_blocks,
                    num_tokens,
                    slot.token_capacity
                );
            }
        }

        let mut active_indices = Vec::with_capacity(forward_indices.len());
        for &i in &forward_indices {
            let tok = slots[i].next_id;
            if eos_token_ids.contains(&tok) {
                slots[i].finished = true;
                slots[i].decode_step = slots[i].decode_step.saturating_add(1);
                continue;
            }
            slots[i].generated.push(tok);
            if has_repeated_tail(&slots[i].generated) {
                slots[i].finished = true;
                slots[i].decode_step = slots[i].decode_step.saturating_add(1);
                continue;
            }
            active_indices.push(i);
        }

        if active_indices.is_empty() {
            for i in 0..slots.len() {
                if !slots[i].finished && slots[i].decode_step == step {
                    slots[i].decode_step = slots[i].decode_step.saturating_add(1);
                }
            }
            return Ok(());
        }

        let request_ids: Vec<usize> = active_indices
            .iter()
            .map(|&i| slots[i].request_id)
            .collect();
        let block_tables_active = runtime_guard.block_tables_for(&request_ids)?;
        let prompt_lens_active_usize: Vec<usize> = active_indices
            .iter()
            .map(|&i| slots[i].prompt_len_usize)
            .collect();
        (
            runtime_guard.cache_clone(),
            block_tables_active,
            prompt_lens_active_usize,
        )
    };

    let active_indices: Vec<usize> = (0..slots.len())
        .filter(|&i| !slots[i].finished && slots[i].decode_step == step)
        .collect();
    let tokens_in: Vec<u32> = active_indices
        .iter()
        .map(|&i| {
            slots[i]
                .generated
                .last()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing generated token for slot {i}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let active_batch = active_indices.len();
    let input_ids_new = Tensor::from_vec(tokens_in, (active_batch, 1usize), device)?;
    let prompt_lens_active: Vec<i64> = active_indices
        .iter()
        .map(|&i| slots[i].prompt_len_i64)
        .collect();
    let position_ids_new = position_ids_for_step(prompt_lens_active.as_slice(), step, device)?;
    let input_metadata = cache.decode_metadata_for_batch_step(
        &block_tables_active,
        prompt_lens_active_usize.as_slice(),
        step,
        device,
    )?;
    let logits_new = paged_batch_decode_forward_shared(
        thinker,
        runtime,
        &cache,
        &input_ids_new,
        &position_ids_new,
        &input_metadata,
    )?;
    let next_active = argmax_token_ids_from_logits(&logits_new.squeeze(1)?, active_batch)?;
    for (slot_idx, &i) in active_indices.iter().enumerate() {
        slots[i].next_id = *next_active
            .get(slot_idx)
            .ok_or_else(|| anyhow::anyhow!("missing argmax for slot index {slot_idx}"))?;
        slots[i].decode_step = slots[i].decode_step.saturating_add(1);
    }

    let _ = eos_fill_id;
    Ok(())
}

pub fn paged_batch_run(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    config: &PagedBatchConfig,
) -> Result<Vec<Vec<u32>>> {
    let batch = input_ids.len();
    if batch == 0 {
        return Ok(vec![]);
    }
    // Defensively free any stale request IDs left over from a previous failed run
    // that shares the same `request_id_base` (commonly 0 for static batches).
    {
        let request_ids: Vec<usize> = (0..batch)
            .map(|i| config.request_id_base.saturating_add(i))
            .collect();
        runtime.manager_mut().free_many(&request_ids);
    }

    let request_ids: Vec<usize> = (0..batch)
        .map(|i| config.request_id_base.saturating_add(i))
        .collect();
    let mut state = match paged_batch_prefill(
        thinker,
        device,
        runtime,
        input_ids,
        attention_mask,
        audio_features,
        config,
    ) {
        Ok(state) => state,
        Err(error) => {
            runtime.manager_mut().free_many(&request_ids);
            return Err(error);
        }
    };

    let result = (|| -> Result<Vec<Vec<u32>>> {
        while !state.all_finished() {
            let continued = paged_batch_decode_step(thinker, device, runtime, &mut state)?;
            if !continued {
                break;
            }
        }
        Ok(std::mem::take(&mut state.generated))
    })();

    runtime.manager_mut().free_many(&state.request_ids);
    result
}

pub fn paged_batch_prefill(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    config: &PagedBatchConfig,
) -> Result<PagedBatchState> {
    let batch = input_ids.len();
    if batch == 0 {
        bail!("paged batch prefill requires at least one row");
    }
    if config.eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }

    let seq_len = input_ids
        .first()
        .map(|row| row.len())
        .ok_or_else(|| anyhow::anyhow!("missing first input row"))?;
    if seq_len == 0 {
        bail!("input_ids rows are empty");
    }
    for (i, row) in input_ids.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "paged batch input_ids[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }
    for (i, row) in attention_mask.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "paged batch attention_mask[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }

    let normalized_batch = if attention_masks_are_right_padded(attention_mask) {
        None
    } else {
        Some(normalize_batch_for_paged_prefill(
            input_ids,
            attention_mask,
        )?)
    };
    let (paged_id_rows, paged_mask_rows, prompt_lens_usize): (
        Vec<&[u32]>,
        Vec<&[u32]>,
        Vec<usize>,
    ) = if let Some((ids, masks, lens)) = normalized_batch.as_ref() {
        (
            ids.iter().map(Vec::as_slice).collect(),
            masks.iter().map(Vec::as_slice).collect(),
            lens.clone(),
        )
    } else {
        (
            input_ids.to_vec(),
            attention_mask.to_vec(),
            attention_mask
                .iter()
                .map(|row| row.iter().filter(|&&v| v != 0).count())
                .collect(),
        )
    };

    let mut ids_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    let mut attn_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    for i in 0..batch {
        ids_flat.extend_from_slice(paged_id_rows[i]);
        attn_flat.extend_from_slice(paged_mask_rows[i]);
    }
    let input_ids_t = Tensor::from_vec(ids_flat, (batch, seq_len), device)?;
    let attention_mask_t = Tensor::from_vec(attn_flat, (batch, seq_len), device)?;

    let audio_placeholder_count = if audio_features.is_some() {
        count_audio_placeholders_batch(paged_id_rows.as_slice(), thinker.audio_token_id())?
    } else {
        0
    };

    let request_ids: Vec<usize> = (0..batch)
        .map(|i| config.request_id_base.saturating_add(i))
        .collect();
    let token_capacities = prefill_token_capacities(config, &prompt_lens_usize)?;
    for (i, &request_id) in request_ids.iter().enumerate() {
        runtime
            .manager_mut()
            .allocate_slots(request_id, token_capacities[i])?;
    }
    let block_tables = runtime.block_tables_for(&request_ids)?;
    let cache = runtime.cache_clone();

    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    let use_packed_cuda_prefill = device.is_cuda() && cuda_packed_prefill_enabled();
    let (last_logits, prompt_lens_i64) = if use_packed_cuda_prefill {
        let total_tokens: usize = prompt_lens_usize.iter().sum();
        let mut packed_ids = Vec::with_capacity(total_tokens);
        let mut packed_audio_features: Vec<Tensor> = Vec::new();
        let mut audio_offset = 0usize;
        for i in 0..batch {
            let len = prompt_lens_usize[i];
            packed_ids.extend_from_slice(&paged_id_rows[i][..len]);
            let row_audio_tokens = paged_id_rows[i][..len]
                .iter()
                .filter(|&&id| id == thinker.audio_token_id())
                .count();
            if let Some(af) = audio_features {
                if row_audio_tokens > 0 {
                    packed_audio_features.push(af.narrow(0, audio_offset, row_audio_tokens)?);
                }
            }
            audio_offset = audio_offset.saturating_add(row_audio_tokens);
        }
        if audio_features.is_some() && audio_offset != audio_placeholder_count {
            bail!(
                "packed prefill audio token mismatch: packed={audio_offset} placeholders={audio_placeholder_count}"
            );
        }
        let packed_input_ids_t = Tensor::from_vec(packed_ids, (1usize, total_tokens), device)?;
        let packed_audio = if audio_features.is_some() {
            if packed_audio_features.is_empty() {
                None
            } else {
                let refs = packed_audio_features.iter().collect::<Vec<_>>();
                Some(Tensor::cat(refs.as_slice(), 0)?)
            }
        } else {
            None
        };
        drop(packed_audio_features);
        let packed_inputs_embeds = thinker.inputs_embeds_with_audio_features(
            &packed_input_ids_t,
            packed_audio.as_ref(),
            audio_placeholder_count,
        )?;
        drop(packed_audio);
        drop(packed_input_ids_t);
        let mut pos_chunks = Vec::with_capacity(batch);
        for (row, &len) in prompt_lens_usize.iter().enumerate() {
            pos_chunks.push(position_ids.narrow(1, row, 1)?.narrow(2, 0, len)?);
        }
        let pos_refs = pos_chunks.iter().collect::<Vec<_>>();
        let packed_position_ids = Tensor::cat(pos_refs.as_slice(), 2)?;
        drop(pos_chunks);
        let input_metadata = cache.input_metadata_from_query_lens(
            &block_tables,
            prompt_lens_usize.as_slice(),
            device,
        )?;
        let mut last_token_indices = Vec::with_capacity(batch);
        let mut offset = 0usize;
        for (row, &len) in prompt_lens_usize.iter().enumerate() {
            if len == 0 {
                bail!("prompt length must be > 0 at batch index {row}");
            }
            let idx = offset.checked_add(len - 1).ok_or_else(|| {
                anyhow::anyhow!("packed last token index overflow at batch {row}")
            })?;
            last_token_indices.push(idx);
            offset = offset
                .checked_add(len)
                .ok_or_else(|| anyhow::anyhow!("packed prompt offset overflow at batch {row}"))?;
        }
        if offset != total_tokens {
            bail!("packed prompt total mismatch: lens_total={offset} total_tokens={total_tokens}");
        }
        let last_logits = {
            let _linear_prefill_guard = set_linear_is_prefill(true);
            thinker.forward_last_logits_packed_prefill_with_paged_cache(
                &packed_position_ids,
                &packed_inputs_embeds,
                &cache,
                &input_metadata,
                last_token_indices.as_slice(),
            )?
        };
        drop(packed_inputs_embeds);
        drop(packed_position_ids);
        drop(last_token_indices);
        let prompt_lens_i64 = input_metadata
            .query_lens
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("packed prefill metadata missing query_lens"))?
            .iter()
            .map(|&len| {
                i64::try_from(len).map_err(|_| anyhow::anyhow!("query_len overflows i64: {len}"))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(input_metadata);
        (last_logits, prompt_lens_i64)
    } else {
        let inputs_embeds = thinker.inputs_embeds_with_audio_features(
            &input_ids_t,
            audio_features,
            audio_placeholder_count,
        )?;
        let mut input_metadata = cache.input_metadata_from_attention_masks(
            &block_tables,
            paged_mask_rows.as_slice(),
            device,
        )?;
        if !input_metadata.prefill_causal_only {
            input_metadata.prefill_attention_mask = Some(attention::make_causal_mask(
                input_metadata.token_attention_mask.as_ref(),
                batch,
                seq_len,
                inputs_embeds.dtype(),
                device,
            )?);
        }

        let mut last_token_indices = Vec::with_capacity(batch);
        for (row, &len) in prompt_lens_usize.iter().enumerate() {
            if len == 0 {
                bail!("prompt length must be > 0 at batch index {row}");
            }
            let idx = row
                .checked_mul(seq_len)
                .and_then(|base| base.checked_add(len - 1))
                .ok_or_else(|| anyhow::anyhow!("last token index overflow at batch {row}"))?;
            last_token_indices.push(idx);
        }
        let last_logits = {
            let _linear_prefill_guard = set_linear_is_prefill(true);
            thinker.forward_last_logits_with_paged_cache(
                &position_ids,
                &inputs_embeds,
                &cache,
                &input_metadata,
                last_token_indices.as_slice(),
            )?
        };
        drop(inputs_embeds);
        drop(last_token_indices);
        let prompt_lens_i64 = input_metadata
            .query_lens
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paged prefill metadata missing query_lens"))?
            .iter()
            .map(|&len| {
                i64::try_from(len).map_err(|_| anyhow::anyhow!("query_len overflows i64: {len}"))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(input_metadata);
        (last_logits, prompt_lens_i64)
    };
    drop(input_ids_t);
    drop(attention_mask_t);
    drop(position_ids);

    let next_ids = argmax_token_ids_from_logits(&last_logits, batch)?;
    drop(last_logits);
    let finished: Vec<bool> = next_ids
        .iter()
        .map(|id| config.eos_token_ids.contains(id))
        .collect();

    Ok(PagedBatchState {
        batch,
        seq_len,
        prompt_lens_usize,
        prompt_lens_i64,
        request_ids,
        block_tables,
        next_ids,
        generated: vec![Vec::new(); batch],
        finished,
        decode_step: 0,
        max_new_tokens: config.max_new_tokens,
        eos_token_ids: config.eos_token_ids.clone(),
        token_capacities,
    })
}

pub fn paged_batch_prefill_shared(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    config: &PagedBatchConfig,
) -> Result<PagedBatchState> {
    let batch = input_ids.len();
    if batch == 0 {
        bail!("paged batch prefill requires at least one row");
    }

    let request_ids: Vec<usize> = (0..batch)
        .map(|i| config.request_id_base.saturating_add(i))
        .collect();
    let prompt_lens_usize: Vec<usize> = attention_mask
        .iter()
        .map(|row| row.iter().filter(|&&v| v != 0).count())
        .collect();
    let token_capacities = prefill_token_capacities(config, &prompt_lens_usize)?;
    let (cache, block_tables) = {
        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        for (i, &request_id) in request_ids.iter().enumerate() {
            if let Err(error) = runtime_guard
                .manager_mut()
                .allocate_slots(request_id, token_capacities[i])
            {
                runtime_guard
                    .manager_mut()
                    .free_many(request_ids.as_slice());
                return Err(error.into());
            }
        }
        let block_tables = runtime_guard.block_tables_for(&request_ids)?;
        (runtime_guard.cache_clone(), block_tables)
    };

    let result = paged_batch_prefill_with_allocated(
        thinker,
        device,
        &cache,
        block_tables,
        request_ids.clone(),
        token_capacities,
        input_ids,
        attention_mask,
        audio_features,
        config,
    );

    match result {
        Ok(state) => Ok(state),
        Err(error) => {
            let mut runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            runtime_guard
                .manager_mut()
                .free_many(request_ids.as_slice());
            Err(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paged_batch_prefill_with_allocated(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    cache: &PagedKvCache,
    block_tables: Vec<Vec<usize>>,
    request_ids: Vec<usize>,
    token_capacities: Vec<usize>,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    config: &PagedBatchConfig,
) -> Result<PagedBatchState> {
    let batch = input_ids.len();
    if config.eos_token_ids.is_empty() {
        bail!("eos_token_ids is empty");
    }
    let seq_len = input_ids
        .first()
        .map(|row| row.len())
        .ok_or_else(|| anyhow::anyhow!("missing first input row"))?;
    if seq_len == 0 {
        bail!("input_ids rows are empty");
    }
    for (i, row) in input_ids.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "paged batch input_ids[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }
    for (i, row) in attention_mask.iter().enumerate() {
        if row.len() != seq_len {
            bail!(
                "paged batch attention_mask[{i}] length mismatch: expected={seq_len}, got={}",
                row.len()
            );
        }
    }

    let normalized_batch = if attention_masks_are_right_padded(attention_mask) {
        None
    } else {
        Some(normalize_batch_for_paged_prefill(
            input_ids,
            attention_mask,
        )?)
    };
    let (paged_id_rows, paged_mask_rows, prompt_lens_usize): (
        Vec<&[u32]>,
        Vec<&[u32]>,
        Vec<usize>,
    ) = if let Some((ids, masks, lens)) = normalized_batch.as_ref() {
        (
            ids.iter().map(Vec::as_slice).collect(),
            masks.iter().map(Vec::as_slice).collect(),
            lens.clone(),
        )
    } else {
        (
            input_ids.to_vec(),
            attention_mask.to_vec(),
            attention_mask
                .iter()
                .map(|row| row.iter().filter(|&&v| v != 0).count())
                .collect(),
        )
    };

    if token_capacities.len() != prompt_lens_usize.len() {
        bail!(
            "token capacities length mismatch after normalization: expected={} got={}",
            prompt_lens_usize.len(),
            token_capacities.len()
        );
    }

    let mut ids_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    let mut attn_flat: Vec<u32> = Vec::with_capacity(batch.saturating_mul(seq_len));
    for i in 0..batch {
        ids_flat.extend_from_slice(paged_id_rows[i]);
        attn_flat.extend_from_slice(paged_mask_rows[i]);
    }
    let input_ids_t = Tensor::from_vec(ids_flat, (batch, seq_len), device)?;
    let attention_mask_t = Tensor::from_vec(attn_flat, (batch, seq_len), device)?;
    let audio_placeholder_count = if audio_features.is_some() {
        count_audio_placeholders_batch(paged_id_rows.as_slice(), thinker.audio_token_id())?
    } else {
        0
    };

    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    let use_packed_cuda_prefill = device.is_cuda() && cuda_packed_prefill_enabled();
    let (last_logits, prompt_lens_i64) = if use_packed_cuda_prefill {
        let total_tokens: usize = prompt_lens_usize.iter().sum();
        let mut packed_ids = Vec::with_capacity(total_tokens);
        let mut packed_audio_features: Vec<Tensor> = Vec::new();
        let mut audio_offset = 0usize;
        for i in 0..batch {
            let len = prompt_lens_usize[i];
            packed_ids.extend_from_slice(&paged_id_rows[i][..len]);
            let row_audio_tokens = paged_id_rows[i][..len]
                .iter()
                .filter(|&&id| id == thinker.audio_token_id())
                .count();
            if let Some(af) = audio_features {
                if row_audio_tokens > 0 {
                    packed_audio_features.push(af.narrow(0, audio_offset, row_audio_tokens)?);
                }
            }
            audio_offset = audio_offset.saturating_add(row_audio_tokens);
        }
        if audio_features.is_some() && audio_offset != audio_placeholder_count {
            bail!(
                "packed prefill audio token mismatch: packed={audio_offset} placeholders={audio_placeholder_count}"
            );
        }
        let packed_input_ids_t = Tensor::from_vec(packed_ids, (1usize, total_tokens), device)?;
        let packed_audio = if audio_features.is_some() {
            if packed_audio_features.is_empty() {
                None
            } else {
                let refs = packed_audio_features.iter().collect::<Vec<_>>();
                Some(Tensor::cat(refs.as_slice(), 0)?)
            }
        } else {
            None
        };
        drop(packed_audio_features);
        let packed_inputs_embeds = thinker.inputs_embeds_with_audio_features(
            &packed_input_ids_t,
            packed_audio.as_ref(),
            audio_placeholder_count,
        )?;
        drop(packed_audio);
        drop(packed_input_ids_t);
        let mut pos_chunks = Vec::with_capacity(batch);
        for (row, &len) in prompt_lens_usize.iter().enumerate() {
            pos_chunks.push(position_ids.narrow(1, row, 1)?.narrow(2, 0, len)?);
        }
        let pos_refs = pos_chunks.iter().collect::<Vec<_>>();
        let packed_position_ids = Tensor::cat(pos_refs.as_slice(), 2)?;
        drop(pos_chunks);
        let input_metadata = cache.input_metadata_from_query_lens(
            &block_tables,
            prompt_lens_usize.as_slice(),
            device,
        )?;
        let mut last_token_indices = Vec::with_capacity(batch);
        let mut offset = 0usize;
        for (row, &len) in prompt_lens_usize.iter().enumerate() {
            if len == 0 {
                bail!("prompt length must be > 0 at batch index {row}");
            }
            let idx = offset.checked_add(len - 1).ok_or_else(|| {
                anyhow::anyhow!("packed last token index overflow at batch {row}")
            })?;
            last_token_indices.push(idx);
            offset = offset
                .checked_add(len)
                .ok_or_else(|| anyhow::anyhow!("packed prompt offset overflow at batch {row}"))?;
        }
        if offset != total_tokens {
            bail!("packed prompt total mismatch: lens_total={offset} total_tokens={total_tokens}");
        }
        let last_logits = {
            let _linear_prefill_guard = set_linear_is_prefill(true);
            thinker.forward_last_logits_packed_prefill_with_paged_cache(
                &packed_position_ids,
                &packed_inputs_embeds,
                cache,
                &input_metadata,
                last_token_indices.as_slice(),
            )?
        };
        drop(packed_inputs_embeds);
        drop(packed_position_ids);
        drop(last_token_indices);
        let prompt_lens_i64 = input_metadata
            .query_lens
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("packed prefill metadata missing query_lens"))?
            .iter()
            .map(|&len| {
                i64::try_from(len).map_err(|_| anyhow::anyhow!("query_len overflows i64: {len}"))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(input_metadata);
        (last_logits, prompt_lens_i64)
    } else {
        let inputs_embeds = thinker.inputs_embeds_with_audio_features(
            &input_ids_t,
            audio_features,
            audio_placeholder_count,
        )?;
        let mut input_metadata = cache.input_metadata_from_attention_masks(
            &block_tables,
            paged_mask_rows.as_slice(),
            device,
        )?;
        if !input_metadata.prefill_causal_only {
            input_metadata.prefill_attention_mask = Some(attention::make_causal_mask(
                input_metadata.token_attention_mask.as_ref(),
                batch,
                seq_len,
                inputs_embeds.dtype(),
                device,
            )?);
        }

        let mut last_token_indices = Vec::with_capacity(batch);
        for (row, &len) in prompt_lens_usize.iter().enumerate() {
            if len == 0 {
                bail!("prompt length must be > 0 at batch index {row}");
            }
            let idx = row
                .checked_mul(seq_len)
                .and_then(|base| base.checked_add(len - 1))
                .ok_or_else(|| anyhow::anyhow!("last token index overflow at batch {row}"))?;
            last_token_indices.push(idx);
        }
        let last_logits = {
            let _linear_prefill_guard = set_linear_is_prefill(true);
            thinker.forward_last_logits_with_paged_cache(
                &position_ids,
                &inputs_embeds,
                cache,
                &input_metadata,
                last_token_indices.as_slice(),
            )?
        };
        drop(inputs_embeds);
        drop(last_token_indices);
        let prompt_lens_i64 = input_metadata
            .query_lens
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("paged prefill metadata missing query_lens"))?
            .iter()
            .map(|&len| {
                i64::try_from(len).map_err(|_| anyhow::anyhow!("query_len overflows i64: {len}"))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(input_metadata);
        (last_logits, prompt_lens_i64)
    };
    drop(input_ids_t);
    drop(attention_mask_t);
    drop(position_ids);

    let next_ids = argmax_token_ids_from_logits(&last_logits, batch)?;
    drop(last_logits);
    let finished: Vec<bool> = next_ids
        .iter()
        .map(|id| config.eos_token_ids.contains(id))
        .collect();

    Ok(PagedBatchState {
        batch,
        seq_len,
        prompt_lens_usize,
        prompt_lens_i64,
        request_ids,
        block_tables,
        next_ids,
        generated: vec![Vec::new(); batch],
        finished,
        decode_step: 0,
        max_new_tokens: config.max_new_tokens,
        eos_token_ids: config.eos_token_ids.clone(),
        token_capacities,
    })
}

fn prefill_token_capacities(
    config: &PagedBatchConfig,
    prompt_lens: &[usize],
) -> Result<Vec<usize>> {
    if let Some(capacities) = config.token_capacities.as_ref() {
        if capacities.len() != prompt_lens.len() {
            bail!(
                "token_capacities length mismatch: expected={} got={}",
                prompt_lens.len(),
                capacities.len()
            );
        }
        for (row, (&capacity, &prompt_len)) in capacities.iter().zip(prompt_lens).enumerate() {
            if capacity < prompt_len {
                bail!(
                    "token capacity shorter than prompt at row {row}: capacity={capacity} prompt_len={prompt_len}"
                );
            }
        }
        return Ok(capacities.clone());
    }
    prompt_lens
        .iter()
        .map(|&prompt_len| {
            let max_new_tokens = config.max_new_tokens.max(256);
            prompt_len
                .checked_add(max_new_tokens)
                .ok_or_else(|| anyhow::anyhow!("token capacity overflow"))
        })
        .collect()
}

/// Run one decode step. Returns `Ok(false)` when all sequences are finished.
///
/// Finished sequences are compacted out of the forward batch so only active
/// sequences consume compute.
pub fn paged_batch_decode_step(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &mut PagedCacheRuntime,
    state: &mut PagedBatchState,
) -> Result<bool> {
    if state.all_finished() {
        return Ok(false);
    }

    let step = state.decode_step;
    let eos_token_ids = state.eos_token_ids.as_slice();

    let mut forward_indices: Vec<usize> = state.active_indices();
    if forward_indices.is_empty() {
        state.decode_step = state.decode_step.saturating_add(1);
        return Ok(!state.all_finished());
    }

    for &i in &forward_indices {
        let num_tokens = state.prompt_lens_usize[i]
            .checked_add(step)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("paged batch token count overflow"))?;
        if num_tokens > state.token_capacities[i] {
            bail!(
                "paged batch decode exceeded allocated KV capacity: request_id={} needed_tokens={} capacity_tokens={} prompt_tokens={} generated_tokens={}",
                state.request_ids[i],
                num_tokens,
                state.token_capacities[i],
                state.prompt_lens_usize[i],
                state.generated[i].len()
            );
        }
        let allocated_blocks = runtime.manager().blocks_for_request(state.request_ids[i]);
        let required_blocks = num_tokens.div_ceil(runtime.manager().block_size());
        if allocated_blocks < required_blocks {
            bail!(
                "paged batch decode missing preallocated KV blocks: request_id={} needed_blocks={} allocated_blocks={} needed_tokens={} capacity_tokens={}",
                state.request_ids[i],
                required_blocks,
                allocated_blocks,
                num_tokens,
                state.token_capacities[i]
            );
        }
    }

    let mut tokens_in = Vec::with_capacity(forward_indices.len());
    for &i in &forward_indices {
        let tok = state
            .next_ids
            .get(i)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("next token id missing for batch row {i}"))?;
        if eos_token_ids.contains(&tok) {
            state.finished[i] = true;
            continue;
        }
        state.generated[i].push(tok);
        if has_repeated_tail(&state.generated[i]) {
            state.finished[i] = true;
            continue;
        }
        tokens_in.push(tok);
    }

    forward_indices.retain(|&i| !state.finished[i]);
    if forward_indices.is_empty() {
        state.decode_step = state.decode_step.saturating_add(1);
        return Ok(!state.all_finished());
    }
    tokens_in = forward_indices
        .iter()
        .map(|&i| {
            state
                .generated
                .get(i)
                .and_then(|g| g.last())
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing generated token for row {i}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let active_batch = forward_indices.len();
    let input_ids_new = Tensor::from_vec(tokens_in, (active_batch, 1usize), device)?;

    let prompt_lens_active: Vec<i64> = forward_indices
        .iter()
        .map(|&i| state.prompt_lens_i64[i])
        .collect();
    let position_ids_new = position_ids_for_step(prompt_lens_active.as_slice(), step, device)?;

    let block_tables_active: Vec<Vec<usize>> = forward_indices
        .iter()
        .map(|&i| {
            state
                .block_tables
                .get(i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing block table for row {i}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let prompt_lens_active_usize: Vec<usize> = forward_indices
        .iter()
        .map(|&i| state.prompt_lens_usize[i])
        .collect();
    let input_metadata = runtime.cache().decode_metadata_for_batch_step(
        &block_tables_active,
        prompt_lens_active_usize.as_slice(),
        step,
        device,
    )?;

    let logits_new = paged_batch_decode_forward(
        thinker,
        runtime,
        device,
        &input_ids_new,
        &position_ids_new,
        &input_metadata,
    )?;
    let next_active = argmax_token_ids_from_logits(&logits_new.squeeze(1)?, active_batch)?;
    drop(logits_new);
    drop(input_metadata);
    drop(position_ids_new);
    drop(input_ids_new);
    for (slot, &row) in forward_indices.iter().enumerate() {
        state.next_ids[row] = *next_active
            .get(slot)
            .ok_or_else(|| anyhow::anyhow!("missing argmax for active slot {slot}"))?;
    }

    state.decode_step = state.decode_step.saturating_add(1);
    Ok(!state.all_finished())
}

fn paged_batch_decode_forward(
    thinker: &ThinkerForConditionalGeneration,
    runtime: &mut PagedCacheRuntime,
    _device: &Device,
    input_ids_new: &Tensor,
    position_ids_new: &Tensor,
    input_metadata: &PagedInputMetadata,
) -> Result<Tensor> {
    #[cfg(feature = "cuda-graph")]
    {
        if runtime.cuda_decode_graph_enabled_for(input_ids_new, input_metadata)? {
            match runtime.cuda_decode_graph(
                thinker,
                input_ids_new,
                position_ids_new,
                input_metadata,
            ) {
                Ok(logits) => return Ok(logits),
                Err(err) => {
                    tracing::warn!("CUDA decode graph key disabled after decode error: {err}");
                    if let Err(disable_err) =
                        runtime.disable_cuda_decode_graph_for(input_ids_new, input_metadata)
                    {
                        tracing::warn!(
                            "failed to disable CUDA decode graph key after decode error: {disable_err}"
                        );
                    }
                }
            }
        }
    }

    paged_batch_decode_forward_without_graph(
        thinker,
        runtime.cache(),
        input_ids_new,
        position_ids_new,
        input_metadata,
    )
}

fn paged_batch_decode_forward_shared(
    thinker: &ThinkerForConditionalGeneration,
    runtime: &SharedPagedCacheRuntime,
    cache: &PagedKvCache,
    input_ids_new: &Tensor,
    position_ids_new: &Tensor,
    input_metadata: &PagedInputMetadata,
) -> Result<Tensor> {
    #[cfg(feature = "cuda-graph")]
    {
        let graph_result = {
            let mut runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            if runtime_guard.cuda_decode_graph_enabled_for(input_ids_new, input_metadata)? {
                Some(
                    match runtime_guard.cuda_decode_graph(
                        thinker,
                        input_ids_new,
                        position_ids_new,
                        input_metadata,
                    ) {
                        Ok(logits) => Ok(logits),
                        Err(err) => {
                            tracing::warn!(
                                "CUDA decode graph key disabled after decode error: {err}"
                            );
                            if let Err(disable_err) = runtime_guard
                                .disable_cuda_decode_graph_for(input_ids_new, input_metadata)
                            {
                                tracing::warn!(
                                    "failed to disable CUDA decode graph key after decode error: {disable_err}"
                                );
                            }
                            Err(err)
                        }
                    },
                )
            } else {
                None
            }
        };
        if let Some(Ok(logits)) = graph_result {
            return Ok(logits);
        }
    }

    paged_batch_decode_forward_without_graph(
        thinker,
        cache,
        input_ids_new,
        position_ids_new,
        input_metadata,
    )
}

fn paged_batch_decode_forward_without_graph(
    thinker: &ThinkerForConditionalGeneration,
    cache: &PagedKvCache,
    input_ids_new: &Tensor,
    position_ids_new: &Tensor,
    input_metadata: &PagedInputMetadata,
) -> Result<Tensor> {
    let inputs_embeds_new = thinker.embed_tokens(input_ids_new)?;
    let _linear_decode_guard = set_linear_is_prefill(false);
    thinker.forward_embeds_with_paged_cache(
        position_ids_new,
        &inputs_embeds_new,
        cache,
        input_metadata,
    )
}

#[cfg(test)]
mod tests {
    use super::{PagedBatchState, decode_step_required_tokens};

    #[test]
    fn active_indices_skips_finished_rows() {
        let state = PagedBatchState {
            batch: 4,
            seq_len: 8,
            prompt_lens_usize: vec![8; 4],
            prompt_lens_i64: vec![8; 4],
            request_ids: vec![0, 1, 2, 3],
            block_tables: vec![vec![1]; 4],
            next_ids: vec![1, 2, 3, 4],
            generated: vec![vec![]; 4],
            finished: vec![false, true, false, true],
            decode_step: 0,
            max_new_tokens: 16,
            eos_token_ids: vec![0],
            token_capacities: vec![24; 4],
        };
        assert_eq!(state.active_indices(), vec![0, 2]);
    }

    #[test]
    fn decode_step_allocation_covers_current_slot_position() -> anyhow::Result<()> {
        assert_eq!(decode_step_required_tokens(31, 0)?, 32);
        assert_eq!(decode_step_required_tokens(32, 0)?, 33);
        assert_eq!(decode_step_required_tokens(64, 31)?, 96);
        Ok(())
    }
}
