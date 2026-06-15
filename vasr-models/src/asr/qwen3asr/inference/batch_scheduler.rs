//! ASR batch scheduler: waiting/running queues and continuous paged-attention batching.
//!
//! GPU task serialization remains in `vasr_server::InferenceScheduler`.
//!
//! `waiting` holds segment indices (typically one VAD slice each). Upstream code prepares
//! segments with [`AsrProcessor::prepare_batch_varlen`] so admit waves can pad locally and
//! prefill with per-row `query_lens` / flash varlen metadata.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

#[cfg(feature = "timing")]
use crate::model::generation::GenerationTimings;
use crate::model::paged_batch_engine::{
    PagedBatchConfig, PagedDecodeSlot, paged_batch_decode_slots_at_step, paged_batch_free_slots,
    paged_batch_prefill, paged_batch_run,
};
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::processor::asr_processor::PreparedInputs;
use vasr_paged_attn::{PagedBlockManager, PagedCacheRuntime, SharedPagedCacheRuntime};

#[cfg(feature = "timing")]
fn duration_to_us(d: std::time::Duration) -> u64 {
    let us = d.as_micros();
    if us > u128::from(u64::MAX) {
        u64::MAX
    } else {
        us as u64
    }
}

#[derive(Debug, Clone)]
pub struct AsrBatchSchedulerConfig {
    pub max_num_seqs: usize,
    explicit_decode_max_seqs: Option<usize>,
    pub continuous_batch: bool,
}

impl Default for AsrBatchSchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 8,
            explicit_decode_max_seqs: None,
            continuous_batch: continuous_batch_enabled(),
        }
    }
}

impl AsrBatchSchedulerConfig {
    pub fn from_decode_max_seqs(decode_max_seqs: usize) -> Self {
        let mut config = Self::default();
        if decode_max_seqs > 0 {
            config.max_num_seqs = decode_max_seqs;
            config.explicit_decode_max_seqs = Some(decode_max_seqs);
        }
        config
    }

    fn continuous_max_num_seqs(&self, runtime: &PagedCacheRuntime) -> usize {
        self.explicit_decode_max_seqs
            .or_else(|| runtime_decode_max_seqs(runtime))
            .unwrap_or(self.max_num_seqs)
    }
}

fn runtime_decode_max_seqs(runtime: &PagedCacheRuntime) -> Option<usize> {
    #[cfg(feature = "cuda-graph")]
    {
        let max_batch = runtime.cuda_decode_graph_max_batch();
        if max_batch > 0 {
            return Some(max_batch);
        }
    }
    let _ = runtime;
    None
}

fn continuous_batch_enabled() -> bool {
    std::env::var_os("VASR_DISABLE_CONTINUOUS_BATCH").is_none()
}

pub(crate) fn continuous_paged_batch_enabled() -> bool {
    crate::inference::utils::continuous_paged_batch_enabled()
}

#[derive(Debug)]
struct RunningSlot {
    client_index: usize,
    slot: PagedDecodeSlot,
}

#[derive(Debug, Default)]
struct ContinuousDecodeDebug {
    finished_slots_by_step: BTreeMap<usize, usize>,
    tail_active_slots: VecDeque<usize>,
    total_decode_calls: usize,
    total_active_slots: usize,
    max_active_slots: usize,
    one_slot_calls: usize,
    two_or_less_slot_calls: usize,
}

impl ContinuousDecodeDebug {
    fn record_active(&mut self, step: usize, active_slots: usize) {
        self.total_decode_calls = self.total_decode_calls.saturating_add(1);
        self.total_active_slots = self.total_active_slots.saturating_add(active_slots);
        self.max_active_slots = self.max_active_slots.max(active_slots);
        if active_slots == 1 {
            self.one_slot_calls = self.one_slot_calls.saturating_add(1);
        }
        if active_slots <= 2 {
            self.two_or_less_slot_calls = self.two_or_less_slot_calls.saturating_add(1);
        }
        self.tail_active_slots.push_back(active_slots);
        while self.tail_active_slots.len() > 50 {
            self.tail_active_slots.pop_front();
        }

        if std::env::var_os("VASR_DEBUG_CONTINUOUS_DECODE_STEPS").is_some() {
            tracing::debug!(
                target: "vasr_transcribe::pipeline",
                "continuous decode | step={step} active_slots={active_slots}",
            );
        }
    }

    fn record_finished(&mut self, running: &[Option<RunningSlot>]) {
        for entry in running.iter().filter_map(Option::as_ref) {
            if !entry.slot.finished {
                continue;
            }
            *self
                .finished_slots_by_step
                .entry(entry.slot.decode_step)
                .or_default() += 1;
        }
    }

    fn log_summary(&self, max_slots: usize) {
        if self.total_decode_calls == 0 && self.finished_slots_by_step.is_empty() {
            return;
        }

        let occupancy = if self.total_decode_calls == 0 || max_slots == 0 {
            0.0
        } else {
            self.total_active_slots as f64 / (self.total_decode_calls * max_slots) as f64
        };
        let tail_one_slot_calls = self.tail_active_slots.iter().filter(|&&n| n == 1).count();
        let tail_two_or_less_slot_calls =
            self.tail_active_slots.iter().filter(|&&n| n <= 2).count();
        let finished_steps = decode_step_distribution(&self.finished_slots_by_step);

        tracing::debug!(
            target: "vasr_transcribe::pipeline",
            "continuous decode | slots={}/{} calls={} occ={:.1}% steps=min/p50/p90/max:{}/{}/{}/{} tail50_one={} tail50_le2={}",
            self.max_active_slots,
            max_slots,
            self.total_decode_calls,
            occupancy * 100.0,
            finished_steps.min,
            finished_steps.p50,
            finished_steps.p90,
            finished_steps.max,
            tail_one_slot_calls,
            tail_two_or_less_slot_calls,
        );
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct DecodeStepDistribution {
    min: usize,
    p50: usize,
    p90: usize,
    max: usize,
}

fn decode_step_distribution(counts: &BTreeMap<usize, usize>) -> DecodeStepDistribution {
    let mut steps = Vec::new();
    for (&step, &count) in counts {
        steps.extend(std::iter::repeat(step).take(count));
    }
    if steps.is_empty() {
        return DecodeStepDistribution::default();
    }
    let len = steps.len();
    let percentile = |pct: usize| -> usize {
        let idx = len.saturating_sub(1).saturating_mul(pct).div_ceil(100);
        steps[idx.min(len - 1)]
    };
    DecodeStepDistribution {
        min: steps[0],
        p50: percentile(50),
        p90: percentile(90),
        max: steps[len - 1],
    }
}

#[derive(Debug)]
pub struct AsrBatchScheduler {
    config: AsrBatchSchedulerConfig,
    next_request_id: usize,
}

#[derive(Debug)]
pub struct PreparedAudioBatch {
    pub prepared: Vec<PreparedInputs>,
    pub audio_features: Tensor,
}

#[derive(Debug)]
pub enum PreparedAudioBatchPoll {
    Ready(PreparedAudioBatch),
    Empty,
    Closed,
}

const DYNAMIC_SOURCE_WAIT: Duration = Duration::from_millis(1);

fn free_all_running(runtime: &mut PagedCacheRuntime, running: &mut [Option<RunningSlot>]) {
    for slot in running.iter_mut() {
        if let Some(entry) = slot.take() {
            paged_batch_free_slots(runtime, &[entry.slot.request_id]);
        }
    }
}

fn prepared_prompt_len(row: &PreparedInputs) -> usize {
    row.attention_mask.iter().filter(|&&v| v != 0).count()
}

fn running_kv_blocks(running: &[Option<RunningSlot>], manager: &PagedBlockManager) -> usize {
    running
        .iter()
        .filter_map(|entry| entry.as_ref())
        .map(|entry| manager.blocks_for_request(entry.slot.request_id))
        .sum()
}

/// Whether admitting one more sequence keeps headroom for running decode growth.
fn can_admit_with_projected_kv(
    manager: &PagedBlockManager,
    kv_block_capacity: usize,
    prompt_len: usize,
    projected_used_blocks: usize,
    projected_running: usize,
) -> bool {
    let prompt_blocks = prompt_len.div_ceil(manager.block_size());
    let decode_headroom = projected_running.max(1);
    projected_used_blocks
        .saturating_add(prompt_blocks)
        .saturating_add(decode_headroom)
        <= kv_block_capacity
}

fn free_finished_running(
    running: &mut [Option<RunningSlot>],
    outputs: &mut [Vec<u32>],
    runtime: &mut PagedCacheRuntime,
) -> bool {
    let mut freed = false;
    for slot in running.iter_mut() {
        let Some(entry) = slot.as_mut() else {
            continue;
        };
        if !entry.slot.finished {
            continue;
        }
        outputs[entry.client_index] = entry.slot.generated.clone();
        paged_batch_free_slots(runtime, &[entry.slot.request_id]);
        *slot = None;
        freed = true;
    }
    freed
}

/// Right-pad selected rows to the max effective prompt length within the admit wave.
fn pad_prepared_wave_for_prefill(
    prepared: &[PreparedInputs],
    wave_indices: &[usize],
    pad_id: u32,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
    if wave_indices.is_empty() {
        bail!("pad_prepared_wave_for_prefill requires at least one row");
    }
    let max_prompt_len = wave_indices
        .iter()
        .map(|&idx| prepared_prompt_len(&prepared[idx]))
        .max()
        .unwrap_or(0);
    if max_prompt_len == 0 {
        bail!("admit wave contains zero-length prompts");
    }

    let mut ids_rows = Vec::with_capacity(wave_indices.len());
    let mut attn_rows = Vec::with_capacity(wave_indices.len());
    for &idx in wave_indices {
        let row = prepared
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("missing prepared row {idx}"))?;
        let prompt_len = prepared_prompt_len(row);
        if prompt_len > row.input_ids.len() || prompt_len > row.attention_mask.len() {
            bail!(
                "prepared[{idx}] effective prompt len {prompt_len} exceeds row capacity: ids={} mask={}",
                row.input_ids.len(),
                row.attention_mask.len()
            );
        }
        let mut ids = row.input_ids[..prompt_len].to_vec();
        let mut attn = row.attention_mask[..prompt_len].to_vec();
        if prompt_len < max_prompt_len {
            ids.extend(std::iter::repeat_n(pad_id, max_prompt_len - prompt_len));
            attn.extend(std::iter::repeat_n(0u32, max_prompt_len - prompt_len));
        }
        ids_rows.push(ids);
        attn_rows.push(attn);
    }
    Ok((ids_rows, attn_rows))
}

fn concat_prepared_audio(
    audio_features: &Tensor,
    audio_offsets: &[usize],
    row_audio_lens: &[usize],
    client_indices: &[usize],
) -> Result<Tensor> {
    if client_indices.is_empty() {
        bail!("concat_prepared_audio requires at least one row");
    }
    if client_indices.len() == 1 {
        let client_index = client_indices[0];
        return Ok(audio_features.narrow(
            0,
            audio_offsets[client_index],
            row_audio_lens[client_index],
        )?);
    }
    let parts = client_indices
        .iter()
        .map(|&client_index| {
            audio_features.narrow(0, audio_offsets[client_index], row_audio_lens[client_index])
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Tensor::cat(parts.as_slice(), 0)?)
}

fn append_prepared_audio_batch(
    audio_token_id: u32,
    prepared: &mut Vec<PreparedInputs>,
    audio_features: &mut Option<Tensor>,
    row_audio_lens: &mut Vec<usize>,
    audio_offsets: &mut Vec<usize>,
    waiting: &mut VecDeque<usize>,
    mut batch: PreparedAudioBatch,
) -> Result<()> {
    if batch.prepared.is_empty() {
        return Ok(());
    }
    let start_idx = prepared.len();
    let mut audio_cursor = audio_offsets
        .last()
        .zip(row_audio_lens.last())
        .map(|(offset, len)| offset.saturating_add(*len))
        .unwrap_or(0);
    let mut batch_audio_tokens = 0usize;

    for row in &batch.prepared {
        let len = row
            .input_ids
            .iter()
            .filter(|&&id| id == audio_token_id)
            .count();
        audio_offsets.push(audio_cursor);
        row_audio_lens.push(len);
        audio_cursor = audio_cursor.saturating_add(len);
        batch_audio_tokens = batch_audio_tokens.saturating_add(len);
    }

    let feature_tokens = batch.audio_features.dims2().map(|(n, _)| n).unwrap_or(0);
    if batch_audio_tokens != feature_tokens {
        bail!(
            "continuous batch audio token mismatch: prepared placeholders={batch_audio_tokens} audio_features={feature_tokens}"
        );
    }

    match audio_features.take() {
        Some(existing) => {
            *audio_features = Some(Tensor::cat(&[existing, batch.audio_features], 0)?);
        }
        None => {
            *audio_features = Some(batch.audio_features);
        }
    }
    let count = batch.prepared.len();
    prepared.append(&mut batch.prepared);
    waiting.extend(start_idx..start_idx.saturating_add(count));
    Ok(())
}

impl AsrBatchScheduler {
    pub fn new(config: AsrBatchSchedulerConfig) -> Self {
        Self {
            config,
            next_request_id: 0,
        }
    }

    pub fn config(&self) -> &AsrBatchSchedulerConfig {
        &self.config
    }

    /// Run one homogeneous micro-batch (same padded `seq_len` across rows).
    pub fn run_micro_batch(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        input_ids: &[&[u32]],
        attention_mask: &[&[u32]],
        audio_features: Option<&Tensor>,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let batch = input_ids.len();
        if batch > self.config.max_num_seqs {
            bail!(
                "micro-batch size {batch} exceeds scheduler max_num_seqs {}",
                self.config.max_num_seqs
            );
        }

        let request_id_base = self.next_request_id;
        self.next_request_id = self.next_request_id.saturating_add(batch.max(1));

        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let engine_config = PagedBatchConfig {
            max_new_tokens,
            eos_token_ids: eos_token_ids.to_vec(),
            request_id_base,
        };
        paged_batch_run(
            thinker,
            device,
            &mut runtime_guard,
            input_ids,
            attention_mask,
            audio_features,
            &engine_config,
        )
    }

    /// Continuous batching for variable-length prepared segments (VAD slices).
    ///
    /// `waiting` starts as all segment indices; each admit wave is right-padded locally and
    /// prefilled with per-row `query_lens` for flash varlen metadata.
    pub fn run_continuous_prepared(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        prepared: &[PreparedInputs],
        audio_features: &Tensor,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        let batch = prepared.len();
        if batch == 0 {
            return Ok(vec![]);
        }
        if max_new_tokens == 0 {
            return Ok(vec![vec![]; batch]);
        }

        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let max_num_seqs = self.config.continuous_max_num_seqs(&runtime_guard);
        let max_slots = max_num_seqs.max(1).min(batch);
        let mut waiting: VecDeque<usize> = (0..batch).collect();
        let mut running: Vec<Option<RunningSlot>> = (0..max_slots).map(|_| None).collect();
        let mut outputs: Vec<Vec<u32>> = vec![vec![]; batch];

        let audio_token_id = thinker.audio_token_id();
        let row_audio_lens: Vec<usize> = prepared
            .iter()
            .map(|row| {
                row.input_ids
                    .iter()
                    .filter(|&&id| id == audio_token_id)
                    .count()
            })
            .collect();
        let mut audio_offsets = Vec::with_capacity(batch);
        let mut audio_cursor = 0usize;
        for &len in &row_audio_lens {
            audio_offsets.push(audio_cursor);
            audio_cursor = audio_cursor.saturating_add(len);
        }
        let total_audio_tokens = audio_features.dims2().map(|(n, _)| n).unwrap_or(0);
        if audio_cursor != total_audio_tokens {
            bail!(
                "continuous batch audio token mismatch: prepared placeholders={audio_cursor} audio_features={total_audio_tokens}"
            );
        }
        let kv_block_capacity =
            PagedBlockManager::allocatable_block_capacity(runtime_guard.stats().num_blocks);

        let result = (|| -> Result<Vec<Vec<u32>>> {
            let mut decode_debug = ContinuousDecodeDebug::default();
            loop {
                decode_debug.record_finished(running.as_slice());
                free_finished_running(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                );

                let mut active_steps: Vec<(usize, usize)> = Vec::new();
                for (slot_idx, entry) in running.iter().enumerate() {
                    if let Some(entry) = entry {
                        if !entry.slot.finished {
                            active_steps.push((slot_idx, entry.slot.decode_step));
                        }
                    }
                }

                if !active_steps.is_empty() {
                    let step_groups = group_running_by_decode_step(active_steps.as_slice());
                    for (step, slot_indices) in step_groups {
                        decode_debug.record_active(step, slot_indices.len());
                        let mut batch_slots: Vec<PagedDecodeSlot> = slot_indices
                            .iter()
                            .map(|&slot_idx| {
                                running[slot_idx]
                                    .as_ref()
                                    .map(|entry| entry.slot.clone())
                                    .ok_or_else(|| {
                                        anyhow::anyhow!("missing running slot {slot_idx}")
                                    })
                            })
                            .collect::<Result<Vec<_>>>()?;

                        #[cfg(feature = "timing")]
                        let start_decode = std::time::Instant::now();
                        paged_batch_decode_slots_at_step(
                            thinker,
                            device,
                            &mut runtime_guard,
                            batch_slots.as_mut_slice(),
                            step,
                            max_new_tokens,
                            eos_token_ids,
                        )?;
                        #[cfg(feature = "timing")]
                        if let Some(t) = timings.as_deref_mut() {
                            t.decode_us = t
                                .decode_us
                                .saturating_add(duration_to_us(start_decode.elapsed()));
                            t.steps = t.steps.saturating_add(1);
                        }

                        for (&slot_idx, slot) in slot_indices.iter().zip(batch_slots.iter()) {
                            if let Some(entry) = running[slot_idx].as_mut() {
                                entry.slot = slot.clone();
                            }
                        }
                    }
                }

                for entry in running.iter_mut() {
                    let Some(entry) = entry.as_mut() else {
                        continue;
                    };
                    if !entry.slot.finished && entry.slot.decode_step >= max_new_tokens {
                        entry.slot.finished = true;
                    }
                }

                decode_debug.record_finished(running.as_slice());
                free_finished_running(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                );

                let mut admitted = false;
                let manager = runtime_guard.manager();
                let mut wave_client_indices: Vec<usize> = Vec::new();
                let mut wave_slot_indices: Vec<usize> = Vec::new();
                let mut projected_used = running_kv_blocks(running.as_slice(), manager);
                let mut projected_running = running.iter().filter(|entry| entry.is_some()).count();
                for slot_idx in 0..max_slots {
                    if running[slot_idx].is_some() {
                        continue;
                    }
                    if projected_running >= max_slots {
                        break;
                    }
                    let Some(client_index) = waiting.pop_front() else {
                        break;
                    };
                    let prompt_len = prepared_prompt_len(&prepared[client_index]);
                    let next_running = projected_running.saturating_add(1);
                    if !can_admit_with_projected_kv(
                        manager,
                        kv_block_capacity,
                        prompt_len,
                        projected_used,
                        next_running,
                    ) {
                        waiting.push_front(client_index);
                        break;
                    }
                    projected_used =
                        projected_used.saturating_add(prompt_len.div_ceil(manager.block_size()));
                    projected_running = next_running;
                    wave_client_indices.push(client_index);
                    wave_slot_indices.push(slot_idx);
                }

                if !wave_client_indices.is_empty() {
                    let wave = wave_client_indices.len();
                    let request_id_base = self.next_request_id;
                    self.next_request_id = self.next_request_id.saturating_add(wave);
                    let (wave_ids, wave_masks) = pad_prepared_wave_for_prefill(
                        prepared,
                        wave_client_indices.as_slice(),
                        pad_id,
                    )?;
                    let ids_rows: Vec<&[u32]> = wave_ids.iter().map(Vec::as_slice).collect();
                    let attn_rows: Vec<&[u32]> = wave_masks.iter().map(Vec::as_slice).collect();
                    let wave_audio = concat_prepared_audio(
                        audio_features,
                        audio_offsets.as_slice(),
                        row_audio_lens.as_slice(),
                        wave_client_indices.as_slice(),
                    )?
                    .to_device(device)?;
                    let engine_config = PagedBatchConfig {
                        max_new_tokens,
                        eos_token_ids: eos_token_ids.to_vec(),
                        request_id_base,
                    };
                    let request_ids: Vec<usize> = (0..wave)
                        .map(|i| request_id_base.saturating_add(i))
                        .collect();
                    #[cfg(feature = "timing")]
                    let start_prefill = std::time::Instant::now();
                    let state = match paged_batch_prefill(
                        thinker,
                        device,
                        &mut runtime_guard,
                        ids_rows.as_slice(),
                        attn_rows.as_slice(),
                        Some(&wave_audio),
                        &engine_config,
                    ) {
                        Ok(state) => state,
                        Err(error) => {
                            paged_batch_free_slots(&mut runtime_guard, request_ids.as_slice());
                            return Err(error);
                        }
                    };
                    #[cfg(feature = "timing")]
                    if let Some(t) = timings.as_deref_mut() {
                        let elapsed = duration_to_us(start_prefill.elapsed());
                        t.prefill_us = t.prefill_us.saturating_add(elapsed);
                        t.prefill_forward_us = t.prefill_forward_us.saturating_add(elapsed);
                        t.prefill_tokens = t
                            .prefill_tokens
                            .saturating_add(state.prompt_lens_usize.iter().copied().sum::<usize>());
                    }
                    for (wave_row, (&client_index, &slot_idx)) in wave_client_indices
                        .iter()
                        .zip(wave_slot_indices.iter())
                        .enumerate()
                    {
                        running[slot_idx] = Some(RunningSlot {
                            client_index,
                            slot: PagedDecodeSlot::from_prefill_state(&state, wave_row)?,
                        });
                    }
                    let running_count = running.iter().filter(|s| s.is_some()).count();
                    let waiting_count = waiting.len();
                    tracing::debug!(
                        target: "vasr_transcribe::pipeline",
                        "continuous | admit wave={wave} running={running_count}/{max_slots} waiting={waiting_count}",
                    );
                    admitted = true;
                }

                let has_running = running.iter().any(|entry| entry.is_some());
                if waiting.is_empty() && !has_running {
                    break;
                }
                if !admitted && active_steps.is_empty() && !has_running && !waiting.is_empty() {
                    bail!(
                        "paged KV cache exhausted: {} sequence(s) waiting, free_blocks={}",
                        waiting.len(),
                        runtime_guard.manager().free_blocks()
                    );
                }
            }

            decode_debug.log_summary(max_slots);
            Ok(outputs)
        })();

        free_all_running(&mut runtime_guard, running.as_mut_slice());
        result
    }

    pub fn run_continuous_prepared_dynamic(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        initial: PreparedAudioBatch,
        poll_source: &mut dyn FnMut(bool) -> Result<PreparedAudioBatchPoll>,
        max_new_tokens: usize,
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        if max_new_tokens == 0 {
            return Ok(vec![vec![]; initial.prepared.len()]);
        }

        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let max_num_seqs = self.config.continuous_max_num_seqs(&runtime_guard);
        let max_slots = max_num_seqs.max(1);
        let mut prepared = Vec::new();
        let mut audio_features: Option<Tensor> = None;
        let mut row_audio_lens = Vec::new();
        let mut audio_offsets = Vec::new();
        let mut waiting = VecDeque::new();
        let mut running: Vec<Option<RunningSlot>> = (0..max_slots).map(|_| None).collect();
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        let audio_token_id = thinker.audio_token_id();
        let mut source_closed = false;

        append_prepared_audio_batch(
            audio_token_id,
            &mut prepared,
            &mut audio_features,
            &mut row_audio_lens,
            &mut audio_offsets,
            &mut waiting,
            initial,
        )?;
        outputs.resize(prepared.len(), Vec::new());

        let kv_block_capacity =
            PagedBlockManager::allocatable_block_capacity(runtime_guard.stats().num_blocks);

        let result = (|| -> Result<Vec<Vec<u32>>> {
            let mut decode_debug = ContinuousDecodeDebug::default();
            loop {
                decode_debug.record_finished(running.as_slice());
                free_finished_running(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                );

                let mut active_steps: Vec<(usize, usize)> = Vec::new();
                for (slot_idx, entry) in running.iter().enumerate() {
                    if let Some(entry) = entry {
                        if !entry.slot.finished {
                            active_steps.push((slot_idx, entry.slot.decode_step));
                        }
                    }
                }

                if !active_steps.is_empty() {
                    let step_groups = group_running_by_decode_step(active_steps.as_slice());
                    for (step, slot_indices) in step_groups {
                        decode_debug.record_active(step, slot_indices.len());
                        let mut batch_slots: Vec<PagedDecodeSlot> = slot_indices
                            .iter()
                            .map(|&slot_idx| {
                                running[slot_idx]
                                    .as_ref()
                                    .map(|entry| entry.slot.clone())
                                    .ok_or_else(|| {
                                        anyhow::anyhow!("missing running slot {slot_idx}")
                                    })
                            })
                            .collect::<Result<Vec<_>>>()?;

                        #[cfg(feature = "timing")]
                        let start_decode = std::time::Instant::now();
                        paged_batch_decode_slots_at_step(
                            thinker,
                            device,
                            &mut runtime_guard,
                            batch_slots.as_mut_slice(),
                            step,
                            max_new_tokens,
                            eos_token_ids,
                        )?;
                        #[cfg(feature = "timing")]
                        if let Some(t) = timings.as_deref_mut() {
                            t.decode_us = t
                                .decode_us
                                .saturating_add(duration_to_us(start_decode.elapsed()));
                            t.steps = t.steps.saturating_add(1);
                        }

                        for (&slot_idx, slot) in slot_indices.iter().zip(batch_slots.iter()) {
                            if let Some(entry) = running[slot_idx].as_mut() {
                                entry.slot = slot.clone();
                            }
                        }
                    }
                }

                for entry in running.iter_mut() {
                    let Some(entry) = entry.as_mut() else {
                        continue;
                    };
                    if !entry.slot.finished && entry.slot.decode_step >= max_new_tokens {
                        entry.slot.finished = true;
                    }
                }

                decode_debug.record_finished(running.as_slice());
                free_finished_running(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                );

                let has_running_before_poll = running.iter().any(|entry| entry.is_some());
                if !source_closed {
                    loop {
                        match poll_source(!has_running_before_poll && waiting.is_empty())? {
                            PreparedAudioBatchPoll::Ready(batch) => {
                                append_prepared_audio_batch(
                                    audio_token_id,
                                    &mut prepared,
                                    &mut audio_features,
                                    &mut row_audio_lens,
                                    &mut audio_offsets,
                                    &mut waiting,
                                    batch,
                                )?;
                                outputs.resize(prepared.len(), Vec::new());
                            }
                            PreparedAudioBatchPoll::Empty => break,
                            PreparedAudioBatchPoll::Closed => {
                                source_closed = true;
                                break;
                            }
                        }
                    }
                }

                let mut admitted = false;
                let manager = runtime_guard.manager();
                let mut wave_client_indices: Vec<usize> = Vec::new();
                let mut wave_slot_indices: Vec<usize> = Vec::new();
                let mut projected_used = running_kv_blocks(running.as_slice(), manager);
                let mut projected_running = running.iter().filter(|entry| entry.is_some()).count();
                for slot_idx in 0..max_slots {
                    if running[slot_idx].is_some() {
                        continue;
                    }
                    if projected_running >= max_slots {
                        break;
                    }
                    let Some(client_index) = waiting.pop_front() else {
                        break;
                    };
                    let prompt_len = prepared_prompt_len(&prepared[client_index]);
                    let next_running = projected_running.saturating_add(1);
                    if !can_admit_with_projected_kv(
                        manager,
                        kv_block_capacity,
                        prompt_len,
                        projected_used,
                        next_running,
                    ) {
                        waiting.push_front(client_index);
                        break;
                    }
                    projected_used =
                        projected_used.saturating_add(prompt_len.div_ceil(manager.block_size()));
                    projected_running = next_running;
                    wave_client_indices.push(client_index);
                    wave_slot_indices.push(slot_idx);
                }

                if !wave_client_indices.is_empty() {
                    let wave = wave_client_indices.len();
                    let request_id_base = self.next_request_id;
                    self.next_request_id = self.next_request_id.saturating_add(wave);
                    let (wave_ids, wave_masks) = pad_prepared_wave_for_prefill(
                        prepared.as_slice(),
                        wave_client_indices.as_slice(),
                        pad_id,
                    )?;
                    let ids_rows: Vec<&[u32]> = wave_ids.iter().map(Vec::as_slice).collect();
                    let attn_rows: Vec<&[u32]> = wave_masks.iter().map(Vec::as_slice).collect();
                    let features = audio_features
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("missing dynamic audio features"))?;
                    let wave_audio = concat_prepared_audio(
                        features,
                        audio_offsets.as_slice(),
                        row_audio_lens.as_slice(),
                        wave_client_indices.as_slice(),
                    )?
                    .to_device(device)?;
                    let engine_config = PagedBatchConfig {
                        max_new_tokens,
                        eos_token_ids: eos_token_ids.to_vec(),
                        request_id_base,
                    };
                    let request_ids: Vec<usize> = (0..wave)
                        .map(|i| request_id_base.saturating_add(i))
                        .collect();
                    #[cfg(feature = "timing")]
                    let start_prefill = std::time::Instant::now();
                    let state = match paged_batch_prefill(
                        thinker,
                        device,
                        &mut runtime_guard,
                        ids_rows.as_slice(),
                        attn_rows.as_slice(),
                        Some(&wave_audio),
                        &engine_config,
                    ) {
                        Ok(state) => state,
                        Err(error) => {
                            paged_batch_free_slots(&mut runtime_guard, request_ids.as_slice());
                            return Err(error);
                        }
                    };
                    #[cfg(feature = "timing")]
                    if let Some(t) = timings.as_deref_mut() {
                        let elapsed = duration_to_us(start_prefill.elapsed());
                        t.prefill_us = t.prefill_us.saturating_add(elapsed);
                        t.prefill_forward_us = t.prefill_forward_us.saturating_add(elapsed);
                        t.prefill_tokens = t
                            .prefill_tokens
                            .saturating_add(state.prompt_lens_usize.iter().copied().sum::<usize>());
                    }
                    for (wave_row, (&client_index, &slot_idx)) in wave_client_indices
                        .iter()
                        .zip(wave_slot_indices.iter())
                        .enumerate()
                    {
                        running[slot_idx] = Some(RunningSlot {
                            client_index,
                            slot: PagedDecodeSlot::from_prefill_state(&state, wave_row)?,
                        });
                    }
                    let running_count = running.iter().filter(|s| s.is_some()).count();
                    let waiting_count = waiting.len();
                    tracing::debug!(
                        target: "vasr_transcribe::pipeline",
                        "continuous | admit wave={wave} running={running_count}/{max_slots} waiting={waiting_count}",
                    );
                    admitted = true;
                }

                let has_running = running.iter().any(|entry| entry.is_some());
                if waiting.is_empty() && !has_running && source_closed {
                    break;
                }
                if waiting.is_empty() && !has_running && !source_closed {
                    thread::sleep(DYNAMIC_SOURCE_WAIT);
                }
                if !admitted && active_steps.is_empty() && !has_running && !waiting.is_empty() {
                    bail!(
                        "paged KV cache exhausted: {} sequence(s) waiting, free_blocks={}",
                        waiting.len(),
                        runtime_guard.manager().free_blocks()
                    );
                }
            }

            decode_debug.log_summary(max_slots);
            Ok(outputs)
        })();

        free_all_running(&mut runtime_guard, running.as_mut_slice());
        result
    }
}

pub(crate) fn group_running_by_decode_step(active: &[(usize, usize)]) -> Vec<(usize, Vec<usize>)> {
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(slot_idx, step) in active {
        groups.entry(step).or_default().push(slot_idx);
    }
    let mut out: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
    out.sort_by_key(|(step, _)| *step);
    out
}

/// Convenience entry when a scheduler instance is not needed.
pub fn run_paged_micro_batch(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    input_ids: &[&[u32]],
    attention_mask: &[&[u32]],
    audio_features: Option<&Tensor>,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    decode_max_seqs: usize,
) -> Result<Vec<Vec<u32>>> {
    let config = AsrBatchSchedulerConfig::from_decode_max_seqs(decode_max_seqs);
    let mut scheduler = AsrBatchScheduler::new(config);
    scheduler.run_micro_batch(
        thinker,
        device,
        runtime,
        input_ids,
        attention_mask,
        audio_features,
        max_new_tokens,
        eos_token_ids,
    )
}

pub fn run_paged_prepared_batch(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    prepared: &[PreparedInputs],
    audio_features: &Tensor,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    decode_max_seqs: usize,
    pad_id: u32,
) -> Result<Vec<Vec<u32>>> {
    let config = AsrBatchSchedulerConfig::from_decode_max_seqs(decode_max_seqs);
    let mut scheduler = AsrBatchScheduler::new(config);
    if scheduler.config.continuous_batch {
        scheduler.run_continuous_prepared(
            thinker,
            device,
            runtime,
            prepared,
            audio_features,
            max_new_tokens,
            eos_token_ids,
            pad_id,
            #[cfg(feature = "timing")]
            None,
        )
    } else {
        let audio_features = audio_features.to_device(device)?;
        let ids_rows: Vec<&[u32]> = prepared.iter().map(|p| p.input_ids.as_slice()).collect();
        let attn_rows: Vec<&[u32]> = prepared
            .iter()
            .map(|p| p.attention_mask.as_slice())
            .collect();
        scheduler.run_micro_batch(
            thinker,
            device,
            runtime,
            ids_rows.as_slice(),
            attn_rows.as_slice(),
            Some(&audio_features),
            max_new_tokens,
            eos_token_ids,
        )
    }
}

#[cfg(feature = "timing")]
pub fn run_paged_prepared_batch_timed(
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    prepared: &[PreparedInputs],
    audio_features: &Tensor,
    max_new_tokens: usize,
    eos_token_ids: &[u32],
    decode_max_seqs: usize,
    pad_id: u32,
) -> Result<(Vec<Vec<u32>>, GenerationTimings)> {
    let config = AsrBatchSchedulerConfig::from_decode_max_seqs(decode_max_seqs);
    let mut scheduler = AsrBatchScheduler::new(config);
    let mut timings = GenerationTimings::default();
    let ids = if scheduler.config.continuous_batch {
        scheduler.run_continuous_prepared(
            thinker,
            device,
            runtime,
            prepared,
            audio_features,
            max_new_tokens,
            eos_token_ids,
            pad_id,
            Some(&mut timings),
        )?
    } else {
        let start = std::time::Instant::now();
        let ids = run_paged_prepared_batch(
            thinker,
            device,
            runtime,
            prepared,
            audio_features,
            max_new_tokens,
            eos_token_ids,
            decode_max_seqs,
            pad_id,
        )?;
        let elapsed = duration_to_us(start.elapsed());
        timings.prefill_us = timings.prefill_us.saturating_add(elapsed);
        ids
    };
    timings.tokens_generated = ids.iter().map(Vec::len).sum();
    Ok((ids, timings))
}

#[cfg(test)]
mod tests {
    use super::{
        AsrBatchSchedulerConfig, PagedBlockManager, can_admit_with_projected_kv,
        continuous_batch_enabled, group_running_by_decode_step, pad_prepared_wave_for_prefill,
    };
    use crate::processor::asr_processor::PreparedInputs;

    #[test]
    fn continuous_batch_defaults_on() {
        if std::env::var_os("VASR_DISABLE_CONTINUOUS_BATCH").is_none() {
            assert!(continuous_batch_enabled());
            assert!(AsrBatchSchedulerConfig::default().continuous_batch);
        }
    }

    #[test]
    fn group_running_by_decode_step_batches_same_step() {
        let groups = group_running_by_decode_step(&[(0, 2), (1, 2), (2, 0), (3, 1)]);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].0, 0);
        assert_eq!(groups[0].1, vec![2]);
        assert_eq!(groups[1].0, 1);
        assert_eq!(groups[1].1, vec![3]);
        assert_eq!(groups[2].0, 2);
        assert_eq!(groups[2].1, vec![0, 1]);
    }

    #[test]
    fn projected_kv_admission_respects_block_capacity() {
        let manager = PagedBlockManager::new(5, 32);
        let capacity = PagedBlockManager::allocatable_block_capacity(5);
        assert_eq!(capacity, 4);
        assert!(can_admit_with_projected_kv(&manager, capacity, 32, 0, 1));
        assert!(!can_admit_with_projected_kv(&manager, capacity, 128, 3, 1));
    }

    #[test]
    fn pad_prepared_wave_for_prefill_builds_varlen_masks() {
        let prepared = vec![
            PreparedInputs {
                input_ids: vec![1, 2, 3],
                attention_mask: vec![1, 1, 1],
                input_features: vec![],
                feature_attention_mask: vec![],
            },
            PreparedInputs {
                input_ids: vec![4, 5, 6, 7, 8],
                attention_mask: vec![1, 1, 1, 1, 1],
                input_features: vec![],
                feature_attention_mask: vec![],
            },
        ];
        let (ids, masks) = pad_prepared_wave_for_prefill(prepared.as_slice(), &[0, 1], 0).unwrap();
        assert_eq!(ids[0].len(), 5);
        assert_eq!(ids[1].len(), 5);
        assert_eq!(masks[0], vec![1, 1, 1, 0, 0]);
        assert_eq!(masks[1], vec![1, 1, 1, 1, 1]);
    }
}
