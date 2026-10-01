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
use crate::model::paged_cache_runtime::{
    PagedBlockManager, PagedCacheRuntime, SharedPagedCacheRuntime,
};
use crate::model::paged_kv_cache::{PagedInputMetadata, PagedKvCache};
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::model::thinker::get_rope_index;
#[cfg(feature = "cuda-graph")]
use vasr_paged_attn::DecodeGraphOutputKind;
use vasr_paged_attn::PagedCudaDecodeForward;
use vasr_quant::isq_linear::set_linear_is_prefill;

const DECODE_TOKEN_REPEAT_LIMIT: usize = 32;
const DECODE_PATTERN_REPEAT_LIMIT: usize = 16;
const DECODE_MAX_REPEAT_PATTERN_LEN: usize = 8;
#[cfg(not(feature = "cuda-graph"))]
const DEFAULT_PREFILL_CHUNKED_ARGMAX_SIZE: usize = 4096;

enum DecodeStepOutput {
    Logits(Tensor),
    NextIds(Tensor),
}

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

fn prefill_chunked_argmax_size(runtime: &PagedCacheRuntime) -> usize {
    #[cfg(feature = "cuda-graph")]
    {
        runtime.cuda_graph_chunked_argmax_size()
    }
    #[cfg(not(feature = "cuda-graph"))]
    {
        let _ = runtime;
        DEFAULT_PREFILL_CHUNKED_ARGMAX_SIZE
    }
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
    /// 预留已经用尽、还没遇到 EOS。槽位继续占着，等空闲 KV 补上来再解码。
    pub awaiting_kv: bool,
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
            awaiting_kv: false,
        })
    }
}

pub fn paged_batch_free_slots(runtime: &mut PagedCacheRuntime, request_ids: &[usize]) {
    runtime.manager_mut().free_many(request_ids);
}

fn with_model_execution<T>(
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if device.is_metal() {
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

fn decode_step_required_tokens(
    prompt_len: usize,
    generated_len_before_step: usize,
) -> Result<usize> {
    prompt_len
        .checked_add(generated_len_before_step)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("paged slot token count overflow"))
}

fn decode_budget_exhausted(
    prompt_len: usize,
    generated_len_after_push: usize,
    token_capacity: usize,
) -> Result<bool> {
    Ok(decode_step_required_tokens(prompt_len, generated_len_after_push)? > token_capacity)
}

/// 预留用尽、空闲池也补不出一块时，记下这条序列停在已写出的文本上。
pub(crate) fn warn_decode_budget_exhausted(
    request_id: usize,
    prompt_len: usize,
    generated_len: usize,
    token_capacity: usize,
    next_token: u32,
) {
    tracing::warn!(
        "paged decode reached token capacity before EOS | request_id={} prompt_tokens={} generated_tokens={} capacity_tokens={} last_token={}",
        request_id,
        prompt_len,
        generated_len,
        token_capacity,
        next_token
    );
}

/// 给还没生成完的序列再补一块 KV。
///
/// 最后一块里还空着的位置先用掉。不够，并且空闲池至少还有一块时，再要一块。
/// 池子是空的就返回 false，调用方继续占着槽位等别的序列结束，或者在确实无法继续时停住。
pub(crate) fn try_grant_one_kv_block(
    manager: &mut PagedBlockManager,
    request_id: usize,
    capacity: &mut usize,
    block_table: &mut Vec<usize>,
    prompt_len: usize,
    generated_len: usize,
) -> Result<bool> {
    let block_size = manager.block_size().max(1);
    let next_tokens = decode_step_required_tokens(prompt_len, generated_len)?;
    let allocated_tokens = manager
        .blocks_for_request(request_id)
        .saturating_mul(block_size);
    if allocated_tokens >= next_tokens {
        // 预留的 token 数没有铺满最后一块，空位本来就属于这条序列。
        *capacity = (*capacity).max(allocated_tokens);
        return Ok(true);
    }
    if manager.free_blocks() == 0 {
        return Ok(false);
    }
    let target = next_tokens.max(allocated_tokens.saturating_add(block_size));
    if !manager.try_allocate_slots(request_id, target)? {
        return Ok(false);
    }
    *capacity = manager
        .blocks_for_request(request_id)
        .saturating_mul(block_size);
    *block_table = manager.block_ids(request_id)?.to_vec();
    tracing::debug!(
        "paged decode granted one KV block | request_id={} prompt_tokens={} generated_tokens={} capacity_tokens={}",
        request_id,
        prompt_len,
        generated_len,
        *capacity
    );
    Ok(true)
}

fn metal_packed_prefill_flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn metal_packed_prefill_enabled(thinker: &ThinkerForConditionalGeneration) -> bool {
    static ENABLED: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    ENABLED
        .get_or_init(|| {
            std::env::var("VASR_METAL_PACKED_PREFILL")
                .ok()
                .map(|value| metal_packed_prefill_flag(Some(&value)))
        })
        .unwrap_or_else(|| thinker.metal_auto_packed_prefill())
}

fn cuda_packed_prefill_enabled() -> bool {
    std::env::var_os("VASR_DISABLE_CUDA_PACKED_PREFILL").is_none()
}

#[allow(clippy::too_many_arguments)]
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
        .filter(|&i| !slots[i].finished && !slots[i].awaiting_kv && slots[i].decode_step == step)
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
        // 重复空转直接结束，空出的显存让给别的序列，不再往下补块。
        if has_repeated_tail(&slots[i].generated) {
            slots[i].finished = true;
            slots[i].decode_step = slots[i].decode_step.saturating_add(1);
            continue;
        }
        if decode_budget_exhausted(
            slots[i].prompt_len_usize,
            slots[i].generated.len(),
            slots[i].token_capacity,
        )? {
            // 这一步还写得下。下一步之前，调度把空出来的 KV 优先补给这条序列。
            slots[i].awaiting_kv = true;
        }
        active_indices.push(i);
    }

    if active_indices.is_empty() {
        for slot in slots.iter_mut() {
            if !slot.finished && !slot.awaiting_kv && slot.decode_step == step {
                slot.decode_step = slot.decode_step.saturating_add(1);
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
    crate::profiling::record_batch("decode", active_batch, active_batch, active_batch);
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

    let decode_output = paged_batch_decode_forward(
        thinker,
        runtime,
        device,
        &input_ids_new,
        &position_ids_new,
        &input_metadata,
    )?;
    let next_active = next_token_ids_from_decode_output(decode_output, active_batch)?;
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
        .filter(|&i| !slots[i].finished && !slots[i].awaiting_kv && slots[i].decode_step == step)
        .collect();
    if forward_indices.is_empty() {
        return Ok(());
    }

    let (cache, block_tables_active, prompt_lens_active_usize, active_indices) = {
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
            // 重复空转直接结束，空出的显存让给别的序列，不再往下补块。
            if has_repeated_tail(&slots[i].generated) {
                slots[i].finished = true;
                slots[i].decode_step = slots[i].decode_step.saturating_add(1);
                continue;
            }
            if decode_budget_exhausted(
                slots[i].prompt_len_usize,
                slots[i].generated.len(),
                slots[i].token_capacity,
            )? {
                // 这一步还写得下。下一步之前，调度把空出来的 KV 优先补给这条序列。
                slots[i].awaiting_kv = true;
            }
            active_indices.push(i);
        }

        if active_indices.is_empty() {
            for slot in slots.iter_mut() {
                if !slot.finished && !slot.awaiting_kv && slot.decode_step == step {
                    slot.decode_step = slot.decode_step.saturating_add(1);
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
            active_indices,
        )
    };

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
    crate::profiling::record_batch("decode", active_batch, active_batch, active_batch);
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
    let next_active = with_model_execution(device, runtime, || {
        let decode_output = crate::profiling::measure("decode_forward", device, || {
            paged_batch_decode_forward_shared(
                thinker,
                runtime,
                &cache,
                &input_ids_new,
                &position_ids_new,
                &input_metadata,
            )
        })?;
        crate::profiling::measure("decode_argmax", device, || {
            next_token_ids_from_decode_output(decode_output, active_batch)
        })
    })?;
    for (slot_idx, &i) in active_indices.iter().enumerate() {
        slots[i].next_id = *next_active
            .get(slot_idx)
            .ok_or_else(|| anyhow::anyhow!("missing argmax for slot index {slot_idx}"))?;
        slots[i].decode_step = slots[i].decode_step.saturating_add(1);
    }

    let _ = eos_fill_id;
    Ok(())
}

pub fn paged_batch_decode_slots_mixed_steps_shared(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    slots: &mut [PagedDecodeSlot],
    slot_indices: &[usize],
    eos_token_ids: &[u32],
) -> Result<()> {
    if slot_indices.is_empty() {
        return Ok(());
    }
    let mut forward_indices = Vec::with_capacity(slot_indices.len());

    let (cache, block_tables_active, prompt_lens_active_usize, starts, context_lens) = {
        let runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let block_size = runtime_guard.manager().block_size();
        for &i in slot_indices {
            let slot = &mut slots[i];
            if slot.finished || slot.awaiting_kv {
                continue;
            }
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

            let tok = slot.next_id;
            if eos_token_ids.contains(&tok) {
                slot.finished = true;
                slot.decode_step = slot.decode_step.saturating_add(1);
                continue;
            }
            slot.generated.push(tok);
            // 重复空转直接结束，空出的显存让给别的序列，不再往下补块。
            if has_repeated_tail(&slot.generated) {
                slot.finished = true;
                slot.decode_step = slot.decode_step.saturating_add(1);
                continue;
            }
            if decode_budget_exhausted(
                slot.prompt_len_usize,
                slot.generated.len(),
                slot.token_capacity,
            )? {
                // 这一步还写得下。下一步之前，调度把空出来的 KV 优先补给这条序列。
                slot.awaiting_kv = true;
            }
            forward_indices.push(i);
        }

        if forward_indices.is_empty() {
            return Ok(());
        }

        let request_ids: Vec<usize> = forward_indices
            .iter()
            .map(|&i| slots[i].request_id)
            .collect();
        let block_tables_active = runtime_guard.block_tables_for(&request_ids)?;
        let prompt_lens_active_usize: Vec<usize> = forward_indices
            .iter()
            .map(|&i| slots[i].prompt_len_usize)
            .collect();
        let mut starts = Vec::with_capacity(forward_indices.len());
        let mut context_lens = Vec::with_capacity(forward_indices.len());
        for &i in &forward_indices {
            let start = slots[i]
                .prompt_len_usize
                .checked_add(slots[i].decode_step)
                .ok_or_else(|| anyhow::anyhow!("paged decode start overflow"))?;
            starts.push(start);
            context_lens.push(
                start
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("paged decode context length overflow"))?,
            );
        }
        (
            runtime_guard.cache_clone(),
            block_tables_active,
            prompt_lens_active_usize,
            starts,
            context_lens,
        )
    };

    let tokens_in: Vec<u32> = forward_indices
        .iter()
        .map(|&i| {
            slots[i]
                .generated
                .last()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("missing generated token for slot {i}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let active_batch = forward_indices.len();
    crate::profiling::record_batch("decode", active_batch, active_batch, active_batch);
    let input_ids_new = Tensor::from_vec(tokens_in, (active_batch, 1usize), device)?;
    let pos: Vec<i64> = forward_indices
        .iter()
        .map(|&i| {
            let step = i64::try_from(slots[i].decode_step)
                .map_err(|_| anyhow::anyhow!("decode step overflows i64"))?;
            slots[i]
                .prompt_len_i64
                .checked_add(step)
                .ok_or_else(|| anyhow::anyhow!("position id overflow"))
        })
        .collect::<Result<Vec<_>>>()?;
    let position_ids_new = Tensor::from_vec(pos, (active_batch, 1usize), device)?
        .unsqueeze(0)?
        .broadcast_as((3usize, active_batch, 1usize))?;
    // This is exclusively a one-token decode step. Reuse the dedicated builder:
    // prefill cumulative-length tensors are unused by paged decode attention.
    debug_assert!(
        starts
            .iter()
            .zip(&context_lens)
            .all(|(start, context)| start + 1 == *context)
    );
    let input_metadata = crate::profiling::measure("decode_metadata", device, || {
        cache
            .decode_metadata_for_batch_steps(&block_tables_active, &starts, 1, device)?
            .pop()
            .ok_or_else(|| candle_core::Error::Msg("missing decode metadata".into()))
    })?;
    let next_active = with_model_execution(device, runtime, || {
        let decode_output = crate::profiling::measure("decode_forward", device, || {
            paged_batch_decode_forward_shared(
                thinker,
                runtime,
                &cache,
                &input_ids_new,
                &position_ids_new,
                &input_metadata,
            )
        })?;
        crate::profiling::measure("decode_argmax", device, || {
            next_token_ids_from_decode_output(decode_output, active_batch)
        })
    })?;
    for (slot_idx, &i) in forward_indices.iter().enumerate() {
        slots[i].next_id = *next_active
            .get(slot_idx)
            .ok_or_else(|| anyhow::anyhow!("missing argmax for slot index {slot_idx}"))?;
        slots[i].decode_step = slots[i].decode_step.saturating_add(1);
    }

    let _ = prompt_lens_active_usize;
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
    let prefill_next_id_chunk_size = prefill_chunked_argmax_size(runtime);

    let (position_ids, _rope_deltas) = get_rope_index(&attention_mask_t)?;
    let use_packed_prefill = (device.is_cuda() && cuda_packed_prefill_enabled())
        || (device.is_metal() && metal_packed_prefill_enabled(thinker));
    let valid_tokens: usize = prompt_lens_usize.iter().sum();
    crate::profiling::record_batch(
        "prefill",
        batch,
        valid_tokens,
        if use_packed_prefill {
            valid_tokens
        } else {
            batch.saturating_mul(seq_len)
        },
    );
    let (next_ids, prompt_lens_i64) = if use_packed_prefill {
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
            if let Some(af) = audio_features
                && row_audio_tokens > 0
            {
                packed_audio_features.push(af.narrow(0, audio_offset, row_audio_tokens)?);
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
        let next_ids = {
            let _linear_prefill_guard = set_linear_is_prefill(true);
            if let Some(next_ids) = thinker.forward_last_next_ids_packed_prefill_with_paged_cache(
                &packed_position_ids,
                &packed_inputs_embeds,
                &cache,
                &input_metadata,
                last_token_indices.as_slice(),
                prefill_next_id_chunk_size,
            )? {
                next_token_ids_from_decode_output(DecodeStepOutput::NextIds(next_ids), batch)?
            } else {
                let last_logits = thinker.forward_last_logits_packed_prefill_with_paged_cache(
                    &packed_position_ids,
                    &packed_inputs_embeds,
                    &cache,
                    &input_metadata,
                    last_token_indices.as_slice(),
                )?;
                let next_ids = argmax_token_ids_from_logits(&last_logits, batch)?;
                drop(last_logits);
                next_ids
            }
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
        (next_ids, prompt_lens_i64)
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
        let next_ids = {
            let _linear_prefill_guard = set_linear_is_prefill(true);
            if let Some(next_ids) = thinker.forward_last_next_ids_with_paged_cache(
                &position_ids,
                &inputs_embeds,
                &cache,
                &input_metadata,
                last_token_indices.as_slice(),
                prefill_next_id_chunk_size,
            )? {
                next_token_ids_from_decode_output(DecodeStepOutput::NextIds(next_ids), batch)?
            } else {
                let last_logits = thinker.forward_last_logits_with_paged_cache(
                    &position_ids,
                    &inputs_embeds,
                    &cache,
                    &input_metadata,
                    last_token_indices.as_slice(),
                )?;
                let next_ids = argmax_token_ids_from_logits(&last_logits, batch)?;
                drop(last_logits);
                next_ids
            }
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
        (next_ids, prompt_lens_i64)
    };
    drop(input_ids_t);
    drop(attention_mask_t);
    drop(position_ids);

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
    let (cache, block_tables, prefill_next_id_chunk_size) = {
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
        (
            runtime_guard.cache_clone(),
            block_tables,
            prefill_chunked_argmax_size(&runtime_guard),
        )
    };

    let result = with_model_execution(device, runtime, || {
        paged_batch_prefill_with_allocated(
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
            prefill_next_id_chunk_size,
        )
    });

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
    prefill_next_id_chunk_size: usize,
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
    let use_packed_prefill = (device.is_cuda() && cuda_packed_prefill_enabled())
        || (device.is_metal() && metal_packed_prefill_enabled(thinker));
    let valid_tokens: usize = prompt_lens_usize.iter().sum();
    crate::profiling::record_batch(
        "prefill",
        batch,
        valid_tokens,
        if use_packed_prefill {
            valid_tokens
        } else {
            batch.saturating_mul(seq_len)
        },
    );
    let (next_ids, prompt_lens_i64) = if use_packed_prefill {
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
            if let Some(af) = audio_features
                && row_audio_tokens > 0
            {
                packed_audio_features.push(af.narrow(0, audio_offset, row_audio_tokens)?);
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
        let next_ids = {
            let _linear_prefill_guard = set_linear_is_prefill(true);
            if let Some(next_ids) = thinker.forward_last_next_ids_packed_prefill_with_paged_cache(
                &packed_position_ids,
                &packed_inputs_embeds,
                cache,
                &input_metadata,
                last_token_indices.as_slice(),
                prefill_next_id_chunk_size,
            )? {
                next_token_ids_from_decode_output(DecodeStepOutput::NextIds(next_ids), batch)?
            } else {
                let last_logits = thinker.forward_last_logits_packed_prefill_with_paged_cache(
                    &packed_position_ids,
                    &packed_inputs_embeds,
                    cache,
                    &input_metadata,
                    last_token_indices.as_slice(),
                )?;
                let next_ids = argmax_token_ids_from_logits(&last_logits, batch)?;
                drop(last_logits);
                next_ids
            }
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
        (next_ids, prompt_lens_i64)
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
        let next_ids = {
            let _linear_prefill_guard = set_linear_is_prefill(true);
            if let Some(next_ids) = thinker.forward_last_next_ids_with_paged_cache(
                &position_ids,
                &inputs_embeds,
                cache,
                &input_metadata,
                last_token_indices.as_slice(),
                prefill_next_id_chunk_size,
            )? {
                next_token_ids_from_decode_output(DecodeStepOutput::NextIds(next_ids), batch)?
            } else {
                let last_logits = thinker.forward_last_logits_with_paged_cache(
                    &position_ids,
                    &inputs_embeds,
                    cache,
                    &input_metadata,
                    last_token_indices.as_slice(),
                )?;
                let next_ids = argmax_token_ids_from_logits(&last_logits, batch)?;
                drop(last_logits);
                next_ids
            }
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
        (next_ids, prompt_lens_i64)
    };
    drop(input_ids_t);
    drop(attention_mask_t);
    drop(position_ids);

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
            prompt_len
                .checked_add(config.max_new_tokens)
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
        // 重复空转直接结束。这条路径没有别的序列会中途退块，所以能补就立刻补一块。
        if has_repeated_tail(&state.generated[i]) {
            state.finished[i] = true;
            continue;
        }
        if decode_budget_exhausted(
            state.prompt_lens_usize[i],
            state.generated[i].len(),
            state.token_capacities[i],
        )? {
            let request_id = state.request_ids[i];
            let prompt_len = state.prompt_lens_usize[i];
            let generated_len = state.generated[i].len();
            let granted = try_grant_one_kv_block(
                runtime.manager_mut(),
                request_id,
                &mut state.token_capacities[i],
                &mut state.block_tables[i],
                prompt_len,
                generated_len,
            )?;
            if !granted {
                warn_decode_budget_exhausted(
                    request_id,
                    prompt_len,
                    generated_len,
                    state.token_capacities[i],
                    tok,
                );
                state.finished[i] = true;
                continue;
            }
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
    crate::profiling::record_batch("decode", active_batch, active_batch, active_batch);
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

    let decode_output = paged_batch_decode_forward(
        thinker,
        runtime,
        device,
        &input_ids_new,
        &position_ids_new,
        &input_metadata,
    )?;
    let next_active = next_token_ids_from_decode_output(decode_output, active_batch)?;
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
) -> Result<DecodeStepOutput> {
    #[cfg(feature = "cuda-graph")]
    {
        if runtime.cuda_decode_graph_enabled_for(input_ids_new, input_metadata)? {
            match runtime.cuda_decode_graph(
                thinker,
                input_ids_new,
                position_ids_new,
                input_metadata,
            ) {
                Ok(output) => {
                    return Ok(match output.kind {
                        DecodeGraphOutputKind::Logits => DecodeStepOutput::Logits(output.tensor),
                        DecodeGraphOutputKind::NextIds | DecodeGraphOutputKind::ChunkedNextIds => {
                            DecodeStepOutput::NextIds(output.tensor)
                        }
                    });
                }
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

    if let Some(next_ids) = paged_batch_decode_next_ids_without_graph(
        thinker,
        runtime.cache(),
        input_ids_new,
        position_ids_new,
        input_metadata,
        prefill_chunked_argmax_size(runtime),
    )? {
        return Ok(DecodeStepOutput::NextIds(next_ids));
    }

    Ok(DecodeStepOutput::Logits(
        paged_batch_decode_forward_without_graph(
            thinker,
            runtime.cache(),
            input_ids_new,
            position_ids_new,
            input_metadata,
        )?,
    ))
}

fn paged_batch_decode_forward_shared(
    thinker: &ThinkerForConditionalGeneration,
    runtime: &SharedPagedCacheRuntime,
    cache: &PagedKvCache,
    input_ids_new: &Tensor,
    position_ids_new: &Tensor,
    input_metadata: &PagedInputMetadata,
) -> Result<DecodeStepOutput> {
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
                        Ok(output) => Ok(match output.kind {
                            DecodeGraphOutputKind::Logits => {
                                DecodeStepOutput::Logits(output.tensor)
                            }
                            DecodeGraphOutputKind::NextIds => {
                                DecodeStepOutput::NextIds(output.tensor)
                            }
                            DecodeGraphOutputKind::ChunkedNextIds => {
                                DecodeStepOutput::NextIds(output.tensor)
                            }
                        }),
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
        if let Some(Ok(output)) = graph_result {
            return Ok(output);
        }
    }

    let decode_next_id_chunk_size = {
        let runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        prefill_chunked_argmax_size(&runtime_guard)
    };
    if let Some(next_ids) = paged_batch_decode_next_ids_without_graph(
        thinker,
        cache,
        input_ids_new,
        position_ids_new,
        input_metadata,
        decode_next_id_chunk_size,
    )? {
        return Ok(DecodeStepOutput::NextIds(next_ids));
    }

    Ok(DecodeStepOutput::Logits(
        paged_batch_decode_forward_without_graph(
            thinker,
            cache,
            input_ids_new,
            position_ids_new,
            input_metadata,
        )?,
    ))
}

fn paged_batch_decode_next_ids_without_graph(
    thinker: &ThinkerForConditionalGeneration,
    cache: &PagedKvCache,
    input_ids_new: &Tensor,
    position_ids_new: &Tensor,
    input_metadata: &PagedInputMetadata,
    chunk_size: usize,
) -> Result<Option<Tensor>> {
    let _linear_decode_guard = set_linear_is_prefill(false);
    thinker.forward_chunked_next_ids_with_paged_cache(
        input_ids_new,
        position_ids_new,
        cache,
        input_metadata,
        chunk_size,
    )
}

fn next_token_ids_from_decode_output(
    output: DecodeStepOutput,
    active_batch: usize,
) -> Result<Vec<u32>> {
    match output {
        DecodeStepOutput::Logits(logits) => {
            argmax_token_ids_from_logits(&logits.squeeze(1)?, active_batch)
        }
        DecodeStepOutput::NextIds(next_ids) => {
            let next_ids = next_ids.to_dtype(candle_core::DType::U32)?;
            let ids = next_ids.to_vec1::<u32>()?;
            if ids.len() != active_batch {
                bail!(
                    "decode next-id output length mismatch: got {} expected {active_batch}",
                    ids.len()
                );
            }
            Ok(ids)
        }
    }
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
    use super::{
        DecodeStepOutput, PagedBatchState, decode_step_required_tokens, try_grant_one_kv_block,
    };
    use crate::model::paged_cache_runtime::PagedBlockManager;

    #[test]
    fn metal_packed_prefill_is_explicitly_opt_in() {
        assert!(!super::metal_packed_prefill_flag(None));
        for value in ["", "0", "false", "garbage"] {
            assert!(!super::metal_packed_prefill_flag(Some(value)));
        }
        for value in ["1", "true", "yes", "on", " TRUE "] {
            assert!(super::metal_packed_prefill_flag(Some(value)));
        }
    }

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

    #[test]
    fn grant_one_block_then_stops_when_the_pool_is_empty() -> anyhow::Result<()> {
        // 块 0 保留，块 1 和块 2 可分配。先占一块，下一次只该再拿到一块。
        let mut manager = PagedBlockManager::new(3, 32);
        manager.allocate_slots(7, 32)?;
        let mut capacity = 32usize;
        let mut table = manager.block_ids(7)?.to_vec();
        assert!(try_grant_one_kv_block(
            &mut manager,
            7,
            &mut capacity,
            &mut table,
            32,
            0
        )?);
        assert!(capacity >= 33);
        assert_eq!(manager.blocks_for_request(7), 2);
        assert_eq!(table.len(), 2);
        assert!(!try_grant_one_kv_block(
            &mut manager,
            7,
            &mut capacity,
            &mut table,
            32,
            32
        )?);
        Ok(())
    }

    #[test]
    fn grant_uses_slack_in_the_last_block_before_taking_another() -> anyhow::Result<()> {
        let mut manager = PagedBlockManager::new(4, 32);
        manager.allocate_slots(1, 40)?;
        let mut capacity = 40usize;
        let mut table = manager.block_ids(1)?.to_vec();
        assert!(try_grant_one_kv_block(
            &mut manager,
            1,
            &mut capacity,
            &mut table,
            40,
            0
        )?);
        assert_eq!(capacity, 64);
        assert_eq!(manager.blocks_for_request(1), 2);
        Ok(())
    }

    #[test]
    fn next_ids_decode_output_converts_to_u32_vec() -> anyhow::Result<()> {
        let next_ids = candle_core::Tensor::new(&[7f32, 11f32], &candle_core::Device::Cpu)?;
        let ids = super::next_token_ids_from_decode_output(DecodeStepOutput::NextIds(next_ids), 2)?;
        assert_eq!(ids, vec![7, 11]);
        Ok(())
    }
}
