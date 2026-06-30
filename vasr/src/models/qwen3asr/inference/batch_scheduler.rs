//! ASR batch scheduler: waiting/running queues and continuous paged-attention batching.
//!
//! GPU task serialization remains in `vasr_server::InferenceScheduler`.
//!
//! `waiting` holds segment indices (typically one VAD slice each). Upstream code prepares
//! segments with [`AsrProcessor::prepare_batch_varlen`] so admit waves can pad locally and
//! prefill with per-row `query_lens` / flash varlen metadata.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

#[cfg(feature = "timing")]
use crate::model::generation::GenerationTimings;
use crate::model::paged_batch_engine::{
    PagedBatchConfig, PagedDecodeSlot, paged_batch_decode_slots_at_step,
    paged_batch_decode_slots_at_step_shared, paged_batch_decode_slots_mixed_steps_shared,
    paged_batch_free_slots, paged_batch_prefill, paged_batch_prefill_shared, paged_batch_run,
};
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::processor::asr_processor::PreparedInputs;
#[cfg(feature = "cuda-graph")]
use vasr_paged_attn::cuda_graph::{cuda_memory_snapshot, cuda_trim_device_mem_pools};
use vasr_paged_attn::{PagedCacheRuntime, SharedPagedCacheRuntime};

#[cfg(feature = "timing")]
fn duration_to_us(d: std::time::Duration) -> u64 {
    let us = d.as_micros();
    if us > u128::from(u64::MAX) {
        u64::MAX
    } else {
        us as u64
    }
}

#[cfg(feature = "timing")]
fn log_prefill_speed(wave: usize, tokens: usize, elapsed_us: u64) {
    let elapsed_s = elapsed_us as f64 / 1_000_000.0;
    let tok_s = tokens as f64 / elapsed_s.max(f64::EPSILON);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "prefill done | seqs={wave} tokens={tokens} wall={elapsed_s:.3}s tok/s={tok_s:.1}",
    );
}

fn log_prefill_start(
    device: &Device,
    seqs: usize,
    audio_seconds: f64,
    token_budget: usize,
    allocated_kv_blocks: usize,
    free_kv_blocks_before: usize,
) {
    let free_kv_blocks_after = free_kv_blocks_before.saturating_sub(allocated_kv_blocks);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "prefill start | seqs={seqs} audio={audio_seconds:.1}s token_budget={token_budget} allocated_kv_blocks={allocated_kv_blocks} free_kv_blocks_before={free_kv_blocks_before} free_kv_blocks_after={free_kv_blocks_after}",
    );
    log_cuda_mem_debug(device, "prefill_start");
}

fn log_cuda_mem_debug(device: &Device, label: &str) {
    if std::env::var_os("VASR_DEBUG_CUDA_MEM").is_none() {
        return;
    }
    #[cfg(feature = "cuda-graph")]
    if let Some(mem) = cuda_memory_snapshot(device) {
        tracing::info!(
            target: "vasr_transcribe::pipeline",
            "cuda_mem | {label} used={:.2}GiB free={:.2}GiB total={:.2}GiB",
            mem.used() as f64 / (1024.0 * 1024.0 * 1024.0),
            mem.free as f64 / (1024.0 * 1024.0 * 1024.0),
            mem.total as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    let _ = device;
}

fn maybe_trim_cuda_after_prefill(device: &Device, allocated_kv_blocks: usize) {
    if std::env::var_os("VASR_TRIM_CUDA_MEMPOOL_AFTER_PREFILL").is_none() {
        return;
    }
    let min_blocks = std::env::var("VASR_TRIM_CUDA_PREFILL_MIN_BLOCKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if allocated_kv_blocks < min_blocks {
        return;
    }
    #[cfg(feature = "cuda-graph")]
    cuda_trim_device_mem_pools(device);
    let _ = device;
}

fn maybe_trim_cuda_before_prefill(device: &Device, allocated_kv_blocks: usize) {
    if std::env::var_os("VASR_TRIM_CUDA_MEMPOOL_BEFORE_PREFILL").is_none() {
        return;
    }
    let min_blocks = std::env::var("VASR_TRIM_CUDA_PREFILL_MIN_BLOCKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    if allocated_kv_blocks < min_blocks {
        return;
    }
    #[cfg(feature = "cuda-graph")]
    cuda_trim_device_mem_pools(device);
    let _ = device;
}

fn maybe_trim_cuda_above_target(device: &Device, env_key: &str) {
    let Some(target_mib) = std::env::var(env_key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return;
    };
    #[cfg(feature = "cuda-graph")]
    {
        let target_bytes = target_mib.saturating_mul(1024 * 1024);
        if let Some(mem) = cuda_memory_snapshot(device) {
            if mem.used() > target_bytes {
                cuda_trim_device_mem_pools(device);
            }
        }
    }
    let _ = device;
}

#[derive(Debug, Clone)]
pub struct AsrBatchSchedulerConfig {
    pub max_num_seqs: usize,
    explicit_decode_max_seqs: Option<usize>,
    prefill_max_wave: Option<usize>,
    pub continuous_batch: bool,
    pub generated_tokens_per_sec: f64,
    pub min_generated_tokens: usize,
}

impl Default for AsrBatchSchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 8,
            explicit_decode_max_seqs: None,
            prefill_max_wave: prefill_max_wave_from_env(),
            continuous_batch: continuous_batch_enabled(),
            generated_tokens_per_sec: 4.25,
            min_generated_tokens: 128,
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

    pub fn with_decode_budget(mut self, tokens_per_sec: f64, min_tokens: usize) -> Self {
        if tokens_per_sec.is_finite() && tokens_per_sec > 0.0 {
            self.generated_tokens_per_sec = tokens_per_sec;
        }
        if min_tokens > 0 {
            self.min_generated_tokens = min_tokens;
        }
        self
    }

    pub fn prefill_max_wave(&self, runtime: &PagedCacheRuntime) -> usize {
        self.prefill_max_wave
            .unwrap_or_else(|| self.continuous_max_num_seqs(runtime))
            .max(1)
    }

    pub fn continuous_max_num_seqs(&self, runtime: &PagedCacheRuntime) -> usize {
        self.explicit_decode_max_seqs
            .or_else(|| runtime_decode_max_seqs(runtime))
            .unwrap_or(self.max_num_seqs)
    }
}

fn prefill_max_wave_from_env() -> Option<usize> {
    std::env::var("VASR_PREFILL_MAX_WAVE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
}

fn async_prefill_channel_capacity() -> usize {
    std::env::var("VASR_ASYNC_PREFILL_CHANNEL_CAP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

fn mixed_step_decode_enabled() -> bool {
    std::env::var_os("VASR_DISABLE_MIXED_STEP_DECODE").is_none()
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

#[derive(Debug)]
pub struct PrefilledSequence {
    pub client_index: usize,
    pub slot: PagedDecodeSlot,
    pub prefill_done_at: Instant,
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
    shared_next_request_id: Option<Arc<AtomicUsize>>,
}

#[derive(Debug)]
pub struct PreparedAudioBatch {
    pub prepared: Vec<PreparedInputs>,
    pub audio_features: Tensor,
    pub audio_seconds: Vec<f64>,
}

#[derive(Debug)]
pub enum PreparedAudioBatchPoll {
    Ready(PreparedAudioBatch),
    Empty,
    Closed,
}

const DYNAMIC_SOURCE_WAIT: Duration = Duration::from_millis(1);

#[derive(Debug)]
pub struct PrefillStageOutput {
    pub sequences: Vec<PrefilledSequence>,
    pub total_rows: usize,
    pub source_closed: bool,
    pub prefill_wave: usize,
    #[cfg(feature = "timing")]
    pub timings: GenerationTimings,
}

#[derive(Debug, Default)]
struct AsyncContinuousDebug {
    received_messages: usize,
    received_sequences: usize,
    max_prefilled_queue: usize,
    empty_recv_waits: usize,
    timeout_waits: usize,
}

impl AsyncContinuousDebug {
    fn record_message(&mut self, sequence_count: usize, prefilled_queue_len: usize) {
        self.received_messages = self.received_messages.saturating_add(1);
        self.received_sequences = self.received_sequences.saturating_add(sequence_count);
        self.max_prefilled_queue = self.max_prefilled_queue.max(prefilled_queue_len);
    }

    fn record_prefilled_queue(&mut self, prefilled_queue_len: usize) {
        self.max_prefilled_queue = self.max_prefilled_queue.max(prefilled_queue_len);
    }

    fn record_empty_recv_wait(&mut self) {
        self.empty_recv_waits = self.empty_recv_waits.saturating_add(1);
    }

    fn record_timeout_wait(&mut self) {
        self.timeout_waits = self.timeout_waits.saturating_add(1);
    }

    fn log_summary(&self) {
        tracing::debug!(
            target: "vasr_transcribe::pipeline",
            "continuous async summary | messages={} prefilled_seqs={} max_prefilled_queue={} empty_waits={} timeout_waits={}",
            self.received_messages,
            self.received_sequences,
            self.max_prefilled_queue,
            self.empty_recv_waits,
            self.timeout_waits,
        );
    }
}

#[cfg(feature = "timing")]
fn add_generation_timings(dst: &mut GenerationTimings, src: &GenerationTimings) {
    dst.prefill_us = dst.prefill_us.saturating_add(src.prefill_us);
    dst.prefill_forward_us = dst
        .prefill_forward_us
        .saturating_add(src.prefill_forward_us);
    dst.prefill_tokens = dst.prefill_tokens.saturating_add(src.prefill_tokens);
    dst.decode_us = dst.decode_us.saturating_add(src.decode_us);
    dst.decode_queue_us = dst.decode_queue_us.saturating_add(src.decode_queue_us);
    dst.steps = dst.steps.saturating_add(src.steps);
    dst.tokens_generated = dst.tokens_generated.saturating_add(src.tokens_generated);
}

#[derive(Debug, Default)]
struct PrefillInputBuffer {
    prepared: Vec<PreparedInputs>,
    audio_features: Option<Tensor>,
    audio_seconds: Vec<f64>,
    row_audio_lens: Vec<usize>,
    audio_offsets: Vec<usize>,
    waiting: VecDeque<usize>,
}

impl PrefillInputBuffer {
    fn append(&mut self, audio_token_id: u32, batch: PreparedAudioBatch) -> Result<()> {
        append_prepared_audio_batch(
            audio_token_id,
            &mut self.prepared,
            &mut self.audio_features,
            &mut self.audio_seconds,
            &mut self.row_audio_lens,
            &mut self.audio_offsets,
            &mut self.waiting,
            batch,
        )
    }

    fn len(&self) -> usize {
        self.prepared.len()
    }

    fn has_waiting(&self) -> bool {
        !self.waiting.is_empty()
    }

    fn waiting_len(&self) -> usize {
        self.waiting.len()
    }
}

pub struct DynamicPrefillProducer<'a> {
    scheduler: AsrBatchScheduler,
    thinker: &'a ThinkerForConditionalGeneration,
    device: &'a Device,
    runtime: &'a SharedPagedCacheRuntime,
    inputs: PrefillInputBuffer,
    poll_source: &'a mut (dyn FnMut(bool) -> Result<PreparedAudioBatchPoll> + Send),
    source_closed: bool,
    audio_token_id: u32,
    eos_token_ids: Vec<u32>,
    pad_id: u32,
}

impl<'a> DynamicPrefillProducer<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: AsrBatchSchedulerConfig,
        thinker: &'a ThinkerForConditionalGeneration,
        device: &'a Device,
        runtime: &'a SharedPagedCacheRuntime,
        initial: PreparedAudioBatch,
        poll_source: &'a mut (dyn FnMut(bool) -> Result<PreparedAudioBatchPoll> + Send),
        eos_token_ids: &[u32],
        pad_id: u32,
    ) -> Result<Self> {
        Self::with_scheduler(
            AsrBatchScheduler::new(config),
            thinker,
            device,
            runtime,
            initial,
            poll_source,
            eos_token_ids,
            pad_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_shared_request_ids(
        config: AsrBatchSchedulerConfig,
        shared_request_ids: Arc<AtomicUsize>,
        thinker: &'a ThinkerForConditionalGeneration,
        device: &'a Device,
        runtime: &'a SharedPagedCacheRuntime,
        initial: PreparedAudioBatch,
        poll_source: &'a mut (dyn FnMut(bool) -> Result<PreparedAudioBatchPoll> + Send),
        eos_token_ids: &[u32],
        pad_id: u32,
    ) -> Result<Self> {
        Self::with_scheduler(
            AsrBatchScheduler::with_shared_request_ids(config, shared_request_ids),
            thinker,
            device,
            runtime,
            initial,
            poll_source,
            eos_token_ids,
            pad_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_scheduler(
        scheduler: AsrBatchScheduler,
        thinker: &'a ThinkerForConditionalGeneration,
        device: &'a Device,
        runtime: &'a SharedPagedCacheRuntime,
        initial: PreparedAudioBatch,
        poll_source: &'a mut (dyn FnMut(bool) -> Result<PreparedAudioBatchPoll> + Send),
        eos_token_ids: &[u32],
        pad_id: u32,
    ) -> Result<Self> {
        let audio_token_id = thinker.audio_token_id();
        let mut inputs = PrefillInputBuffer::default();
        inputs.append(audio_token_id, initial)?;
        Ok(Self {
            scheduler,
            thinker,
            device,
            runtime,
            inputs,
            poll_source,
            source_closed: false,
            audio_token_id,
            eos_token_ids: eos_token_ids.to_vec(),
            pad_id,
        })
    }

    pub fn run_stage(&mut self, may_block_source: bool) -> Result<PrefillStageOutput> {
        let mut stage_prefilled = VecDeque::new();
        #[cfg(feature = "timing")]
        let mut stage_timings = GenerationTimings::default();
        #[cfg(feature = "timing")]
        let mut stage_timings_ref = Some(&mut stage_timings);
        let wave = self.scheduler.drive_dynamic_prefill_stage(
            self.thinker,
            self.device,
            self.runtime,
            &mut self.inputs,
            &mut stage_prefilled,
            self.poll_source,
            &mut self.source_closed,
            self.audio_token_id,
            self.eos_token_ids.as_slice(),
            self.pad_id,
            may_block_source,
            #[cfg(feature = "timing")]
            &mut stage_timings_ref,
        )?;
        Ok(PrefillStageOutput {
            sequences: stage_prefilled.into_iter().collect(),
            total_rows: self.inputs.len(),
            source_closed: self.source_closed,
            prefill_wave: wave,
            #[cfg(feature = "timing")]
            timings: stage_timings,
        })
    }

    pub fn is_done(&self) -> bool {
        self.source_closed && !self.inputs.has_waiting()
    }
}

fn free_all_running(runtime: &mut PagedCacheRuntime, running: &mut [Option<RunningSlot>]) {
    for slot in running.iter_mut() {
        if let Some(entry) = slot.take() {
            paged_batch_free_slots(runtime, &[entry.slot.request_id]);
        }
    }
}

fn free_all_prefilled(
    runtime: &mut PagedCacheRuntime,
    prefilled: &mut VecDeque<PrefilledSequence>,
) {
    while let Some(entry) = prefilled.pop_front() {
        paged_batch_free_slots(runtime, &[entry.slot.request_id]);
    }
}

fn prepared_prompt_len(row: &PreparedInputs) -> usize {
    row.attention_mask.iter().filter(|&&v| v != 0).count()
}

fn generated_token_budget(
    audio_seconds: f64,
    generated_tokens_per_sec: f64,
    min_generated_tokens: usize,
) -> usize {
    let estimated = if audio_seconds.is_finite() && audio_seconds > 0.0 {
        (audio_seconds * generated_tokens_per_sec).ceil() as usize
    } else {
        0
    };
    estimated.max(min_generated_tokens)
}

fn prepared_token_capacity(
    row: &PreparedInputs,
    audio_seconds: &[f64],
    row_index: usize,
    generated_tokens_per_sec: f64,
    min_generated_tokens: usize,
) -> Result<usize> {
    prepared_prompt_len(row)
        .checked_add(generated_token_budget(
            audio_seconds.get(row_index).copied().unwrap_or(0.0),
            generated_tokens_per_sec,
            min_generated_tokens,
        ))
        .ok_or_else(|| anyhow::anyhow!("prepared token capacity overflow"))
}

fn attach_prefilled_to_running(
    prefilled: &mut VecDeque<PrefilledSequence>,
    running: &mut [Option<RunningSlot>],
    #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
) -> bool {
    let mut attached = false;
    for slot in running.iter_mut() {
        if slot.is_some() {
            continue;
        }
        let Some(prefilled) = prefilled.pop_front() else {
            break;
        };
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_deref_mut() {
            t.decode_queue_us = t
                .decode_queue_us
                .saturating_add(duration_to_us(prefilled.prefill_done_at.elapsed()));
        }
        *slot = Some(RunningSlot {
            client_index: prefilled.client_index,
            slot: prefilled.slot,
        });
        attached = true;
    }
    attached
}

fn apply_prefill_stage_message(
    message: PrefillStageOutput,
    outputs: &mut Vec<Vec<u32>>,
    prefilled: &mut VecDeque<PrefilledSequence>,
    source_closed: &mut bool,
    #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
) {
    outputs.resize(outputs.len().max(message.total_rows), Vec::new());
    prefilled.extend(message.sequences);
    *source_closed = message.source_closed;
    #[cfg(feature = "timing")]
    if let Some(t) = timings.as_deref_mut() {
        add_generation_timings(t, &message.timings);
    }
}

fn collect_prefill_wave_by_kv<F>(
    waiting: &mut VecDeque<usize>,
    block_size: usize,
    mut free_blocks: usize,
    max_wave: usize,
    mut token_capacity: F,
) -> Result<Vec<usize>>
where
    F: FnMut(usize) -> Result<usize>,
{
    let mut wave = Vec::new();
    while wave.len() < max_wave {
        let mut selected = None;
        for (pos, &client_index) in waiting.iter().enumerate() {
            let required_blocks = token_capacity(client_index)?.div_ceil(block_size);
            if required_blocks <= free_blocks {
                selected = Some((pos, required_blocks));
                break;
            }
        }
        let Some((pos, required_blocks)) = selected else {
            break;
        };
        let client_index = waiting
            .remove(pos)
            .ok_or_else(|| anyhow::anyhow!("waiting queue remove failed at position {pos}"))?;
        free_blocks = free_blocks.saturating_sub(required_blocks);
        wave.push(client_index);
    }
    Ok(wave)
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

fn decode_running_step_groups(
    running: &mut [Option<RunningSlot>],
    active_steps: &[(usize, usize)],
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    eos_token_ids: &[u32],
    decode_debug: &mut ContinuousDecodeDebug,
    #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
) -> Result<()> {
    if active_steps.is_empty() {
        return Ok(());
    }
    if mixed_step_decode_enabled() {
        let slot_indices: Vec<usize> = active_steps.iter().map(|&(slot_idx, _)| slot_idx).collect();
        decode_debug.record_active(0, slot_indices.len());
        let mut batch_slots: Vec<PagedDecodeSlot> = running
            .iter()
            .map(|entry| {
                entry
                    .as_ref()
                    .map(|entry| entry.slot.clone())
                    .unwrap_or_else(|| PagedDecodeSlot {
                        request_id: usize::MAX,
                        prompt_len_usize: 0,
                        prompt_len_i64: 0,
                        block_table: Vec::new(),
                        next_id: 0,
                        generated: Vec::new(),
                        finished: true,
                        decode_step: 0,
                        token_capacity: 0,
                    })
            })
            .collect();
        #[cfg(feature = "timing")]
        let start_decode = std::time::Instant::now();
        paged_batch_decode_slots_mixed_steps_shared(
            thinker,
            device,
            runtime,
            batch_slots.as_mut_slice(),
            slot_indices.as_slice(),
            eos_token_ids,
        )?;
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_deref_mut() {
            t.decode_us = t
                .decode_us
                .saturating_add(duration_to_us(start_decode.elapsed()));
            t.steps = t.steps.saturating_add(1);
            t.tokens_generated = t.tokens_generated.saturating_add(slot_indices.len());
        }
        for &slot_idx in &slot_indices {
            if let Some(entry) = running[slot_idx].as_mut() {
                entry.slot = batch_slots[slot_idx].clone();
            }
        }
        return Ok(());
    }
    let step_groups = group_running_by_decode_step(active_steps);
    for (step, slot_indices) in step_groups {
        decode_debug.record_active(step, slot_indices.len());
        let mut batch_slots: Vec<PagedDecodeSlot> = slot_indices
            .iter()
            .map(|&slot_idx| {
                running[slot_idx]
                    .as_ref()
                    .map(|entry| entry.slot.clone())
                    .ok_or_else(|| anyhow::anyhow!("missing running slot {slot_idx}"))
            })
            .collect::<Result<Vec<_>>>()?;

        #[cfg(feature = "timing")]
        let start_decode = std::time::Instant::now();
        maybe_trim_cuda_above_target(device, "VASR_TRIM_CUDA_MEMPOOL_BEFORE_DECODE_TARGET_MIB");
        paged_batch_decode_slots_at_step_shared(
            thinker,
            device,
            runtime,
            batch_slots.as_mut_slice(),
            step,
            eos_token_ids,
        )?;
        maybe_trim_cuda_above_target(device, "VASR_TRIM_CUDA_MEMPOOL_AFTER_DECODE_TARGET_MIB");
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_deref_mut() {
            t.decode_us = t
                .decode_us
                .saturating_add(duration_to_us(start_decode.elapsed()));
            t.steps = t.steps.saturating_add(1);
            t.tokens_generated = t.tokens_generated.saturating_add(slot_indices.len());
        }

        for (&slot_idx, slot) in slot_indices.iter().zip(batch_slots.iter()) {
            if let Some(entry) = running[slot_idx].as_mut() {
                entry.slot = slot.clone();
            }
        }
    }
    Ok(())
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
    audio_seconds: &mut Vec<f64>,
    row_audio_lens: &mut Vec<usize>,
    audio_offsets: &mut Vec<usize>,
    waiting: &mut VecDeque<usize>,
    mut batch: PreparedAudioBatch,
) -> Result<()> {
    if batch.prepared.is_empty() {
        return Ok(());
    }
    if batch.audio_seconds.len() != batch.prepared.len() {
        bail!(
            "prepared audio seconds mismatch: prepared={} audio_seconds={}",
            batch.prepared.len(),
            batch.audio_seconds.len()
        );
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
    audio_seconds.append(&mut batch.audio_seconds);
    prepared.append(&mut batch.prepared);
    waiting.extend(start_idx..start_idx.saturating_add(count));
    Ok(())
}

fn dynamic_source_poll_may_block(has_running: bool, waiting_len: usize) -> bool {
    !has_running && waiting_len == 0
}

impl AsrBatchScheduler {
    pub fn new(config: AsrBatchSchedulerConfig) -> Self {
        Self {
            config,
            next_request_id: 0,
            shared_next_request_id: None,
        }
    }

    pub fn config(&self) -> &AsrBatchSchedulerConfig {
        &self.config
    }

    fn with_shared_request_ids(config: AsrBatchSchedulerConfig, shared: Arc<AtomicUsize>) -> Self {
        Self {
            config,
            next_request_id: 0,
            shared_next_request_id: Some(shared),
        }
    }

    fn allocate_request_id_base(&mut self, count: usize) -> usize {
        let count = count.max(1);
        if let Some(shared) = self.shared_next_request_id.as_ref() {
            shared.fetch_add(count, Ordering::Relaxed)
        } else {
            let base = self.next_request_id;
            self.next_request_id = self.next_request_id.saturating_add(count);
            base
        }
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

        let request_id_base = self.allocate_request_id_base(batch);

        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let engine_config = PagedBatchConfig {
            max_new_tokens,
            eos_token_ids: eos_token_ids.to_vec(),
            request_id_base,
            token_capacities: None,
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

    fn prefill_waiting_wave(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &mut PagedCacheRuntime,
        prepared: &[PreparedInputs],
        audio_features: &Tensor,
        audio_seconds: &[f64],
        row_audio_lens: &[usize],
        audio_offsets: &[usize],
        waiting: &mut VecDeque<usize>,
        prefilled: &mut VecDeque<PrefilledSequence>,
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
    ) -> Result<usize> {
        let manager = runtime.manager();
        let block_size = manager.block_size();
        let free_blocks = manager.free_blocks();
        let max_wave = self.config.prefill_max_wave(runtime);
        let wave_client_indices = collect_prefill_wave_by_kv(
            waiting,
            block_size,
            free_blocks,
            max_wave,
            |client_index| {
                prepared_token_capacity(
                    &prepared[client_index],
                    audio_seconds,
                    client_index,
                    self.config.generated_tokens_per_sec,
                    self.config.min_generated_tokens,
                )
            },
        )?;

        if wave_client_indices.is_empty() {
            return Ok(0);
        }

        let wave = wave_client_indices.len();
        let request_id_base = self.allocate_request_id_base(wave);
        let (wave_ids, wave_masks) =
            pad_prepared_wave_for_prefill(prepared, wave_client_indices.as_slice(), pad_id)?;
        let ids_rows: Vec<&[u32]> = wave_ids.iter().map(Vec::as_slice).collect();
        let attn_rows: Vec<&[u32]> = wave_masks.iter().map(Vec::as_slice).collect();
        let token_capacities = wave_client_indices
            .iter()
            .map(|&client_index| {
                prepared_token_capacity(
                    &prepared[client_index],
                    audio_seconds,
                    client_index,
                    self.config.generated_tokens_per_sec,
                    self.config.min_generated_tokens,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let allocated_kv_blocks = token_capacities
            .iter()
            .map(|&capacity| capacity.div_ceil(block_size))
            .sum::<usize>();
        let prefill_audio_seconds = wave_client_indices
            .iter()
            .map(|&client_index| audio_seconds[client_index])
            .sum::<f64>();
        let token_budget = token_capacities.iter().sum::<usize>();
        log_prefill_start(
            device,
            wave,
            prefill_audio_seconds,
            token_budget,
            allocated_kv_blocks,
            free_blocks,
        );
        maybe_trim_cuda_before_prefill(device, allocated_kv_blocks);
        let wave_audio = concat_prepared_audio(
            audio_features,
            audio_offsets,
            row_audio_lens,
            wave_client_indices.as_slice(),
        )?
        .to_device(device)?;
        let engine_config = PagedBatchConfig {
            max_new_tokens: 0,
            eos_token_ids: eos_token_ids.to_vec(),
            request_id_base,
            token_capacities: Some(token_capacities),
        };
        let request_ids: Vec<usize> = (0..wave)
            .map(|i| request_id_base.saturating_add(i))
            .collect();
        #[cfg(feature = "timing")]
        let start_prefill = std::time::Instant::now();
        let state = match paged_batch_prefill(
            thinker,
            device,
            runtime,
            ids_rows.as_slice(),
            attn_rows.as_slice(),
            Some(&wave_audio),
            &engine_config,
        ) {
            Ok(state) => state,
            Err(error) => {
                paged_batch_free_slots(runtime, request_ids.as_slice());
                return Err(error);
            }
        };
        drop(wave_audio);
        maybe_trim_cuda_after_prefill(device, allocated_kv_blocks);
        #[cfg(feature = "timing")]
        let elapsed = duration_to_us(start_prefill.elapsed());
        #[cfg(feature = "timing")]
        let prefill_tokens = state.prompt_lens_usize.iter().copied().sum::<usize>();
        #[cfg(feature = "timing")]
        log_prefill_speed(wave, prefill_tokens, elapsed);
        log_cuda_mem_debug(device, "prefill_done");
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_deref_mut() {
            t.prefill_us = t.prefill_us.saturating_add(elapsed);
            t.prefill_forward_us = t.prefill_forward_us.saturating_add(elapsed);
            t.prefill_tokens = t.prefill_tokens.saturating_add(prefill_tokens);
        }
        let prefill_done_at = Instant::now();
        for (wave_row, &client_index) in wave_client_indices.iter().enumerate() {
            prefilled.push_back(PrefilledSequence {
                client_index,
                slot: PagedDecodeSlot::from_prefill_state(&state, wave_row)?,
                prefill_done_at,
            });
        }
        Ok(wave)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_waiting_wave_shared(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        prepared: &[PreparedInputs],
        audio_features: &Tensor,
        audio_seconds: &[f64],
        row_audio_lens: &[usize],
        audio_offsets: &[usize],
        waiting: &mut VecDeque<usize>,
        prefilled: &mut VecDeque<PrefilledSequence>,
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
    ) -> Result<usize> {
        let (block_size, free_blocks, max_wave) = {
            let runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            (
                runtime_guard.manager().block_size(),
                runtime_guard.manager().free_blocks(),
                self.config.prefill_max_wave(&runtime_guard),
            )
        };
        let wave_client_indices = collect_prefill_wave_by_kv(
            waiting,
            block_size,
            free_blocks,
            max_wave,
            |client_index| {
                prepared_token_capacity(
                    &prepared[client_index],
                    audio_seconds,
                    client_index,
                    self.config.generated_tokens_per_sec,
                    self.config.min_generated_tokens,
                )
            },
        )?;

        if wave_client_indices.is_empty() {
            return Ok(0);
        }

        let wave = wave_client_indices.len();
        let request_id_base = self.allocate_request_id_base(wave);
        let (wave_ids, wave_masks) =
            pad_prepared_wave_for_prefill(prepared, wave_client_indices.as_slice(), pad_id)?;
        let ids_rows: Vec<&[u32]> = wave_ids.iter().map(Vec::as_slice).collect();
        let attn_rows: Vec<&[u32]> = wave_masks.iter().map(Vec::as_slice).collect();
        let token_capacities = wave_client_indices
            .iter()
            .map(|&client_index| {
                prepared_token_capacity(
                    &prepared[client_index],
                    audio_seconds,
                    client_index,
                    self.config.generated_tokens_per_sec,
                    self.config.min_generated_tokens,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let allocated_kv_blocks = token_capacities
            .iter()
            .map(|&capacity| capacity.div_ceil(block_size))
            .sum::<usize>();
        let prefill_audio_seconds = wave_client_indices
            .iter()
            .map(|&client_index| audio_seconds[client_index])
            .sum::<f64>();
        let token_budget = token_capacities.iter().sum::<usize>();
        log_prefill_start(
            device,
            wave,
            prefill_audio_seconds,
            token_budget,
            allocated_kv_blocks,
            free_blocks,
        );
        maybe_trim_cuda_before_prefill(device, allocated_kv_blocks);
        let wave_audio = concat_prepared_audio(
            audio_features,
            audio_offsets,
            row_audio_lens,
            wave_client_indices.as_slice(),
        )?
        .to_device(device)?;
        let engine_config = PagedBatchConfig {
            max_new_tokens: 0,
            eos_token_ids: eos_token_ids.to_vec(),
            request_id_base,
            token_capacities: Some(token_capacities),
        };
        #[cfg(feature = "timing")]
        let start_prefill = std::time::Instant::now();
        let state = match paged_batch_prefill_shared(
            thinker,
            device,
            runtime,
            ids_rows.as_slice(),
            attn_rows.as_slice(),
            Some(&wave_audio),
            &engine_config,
        ) {
            Ok(state) => state,
            Err(error) => {
                for client_index in wave_client_indices.into_iter().rev() {
                    waiting.push_front(client_index);
                }
                return Err(error);
            }
        };
        drop(wave_audio);
        maybe_trim_cuda_after_prefill(device, allocated_kv_blocks);
        #[cfg(feature = "timing")]
        let elapsed = duration_to_us(start_prefill.elapsed());
        #[cfg(feature = "timing")]
        let prefill_tokens = state.prompt_lens_usize.iter().copied().sum::<usize>();
        #[cfg(feature = "timing")]
        log_prefill_speed(wave, prefill_tokens, elapsed);
        log_cuda_mem_debug(device, "prefill_done");
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_deref_mut() {
            t.prefill_us = t.prefill_us.saturating_add(elapsed);
            t.prefill_forward_us = t.prefill_forward_us.saturating_add(elapsed);
            t.prefill_tokens = t.prefill_tokens.saturating_add(prefill_tokens);
        }
        let prefill_done_at = Instant::now();
        for (wave_row, &client_index) in wave_client_indices.iter().enumerate() {
            prefilled.push_back(PrefilledSequence {
                client_index,
                slot: PagedDecodeSlot::from_prefill_state(&state, wave_row)?,
                prefill_done_at,
            });
        }
        Ok(wave)
    }

    fn drive_dynamic_prefill_stage(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        inputs: &mut PrefillInputBuffer,
        prefilled: &mut VecDeque<PrefilledSequence>,
        poll_source: &mut (dyn FnMut(bool) -> Result<PreparedAudioBatchPoll> + Send),
        source_closed: &mut bool,
        audio_token_id: u32,
        eos_token_ids: &[u32],
        pad_id: u32,
        may_block_source: bool,
        #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
    ) -> Result<usize> {
        if !*source_closed {
            loop {
                match poll_source(may_block_source && !inputs.has_waiting())? {
                    PreparedAudioBatchPoll::Ready(batch) => {
                        inputs.append(audio_token_id, batch)?;
                    }
                    PreparedAudioBatchPoll::Empty => break,
                    PreparedAudioBatchPoll::Closed => {
                        *source_closed = true;
                        break;
                    }
                }
                if may_block_source {
                    break;
                }
            }
        }

        let Some(features) = inputs.audio_features.as_ref() else {
            return Ok(0);
        };
        self.prefill_waiting_wave_shared(
            thinker,
            device,
            runtime,
            inputs.prepared.as_slice(),
            features,
            inputs.audio_seconds.as_slice(),
            inputs.row_audio_lens.as_slice(),
            inputs.audio_offsets.as_slice(),
            &mut inputs.waiting,
            prefilled,
            eos_token_ids,
            pad_id,
            #[cfg(feature = "timing")]
            timings,
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
        audio_seconds: &[f64],
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        let batch = prepared.len();
        if batch == 0 {
            return Ok(vec![]);
        }
        if audio_seconds.len() != batch {
            bail!(
                "continuous prepared audio seconds mismatch: prepared={} audio_seconds={}",
                batch,
                audio_seconds.len()
            );
        }

        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let max_num_seqs = self.config.continuous_max_num_seqs(&runtime_guard);
        let max_slots = max_num_seqs.max(1).min(batch);
        let mut waiting: VecDeque<usize> = (0..batch).collect();
        let mut prefilled: VecDeque<PrefilledSequence> = VecDeque::new();
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

        let result = (|| -> Result<Vec<Vec<u32>>> {
            let mut decode_debug = ContinuousDecodeDebug::default();
            loop {
                decode_debug.record_finished(running.as_slice());
                free_finished_running(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                );
                attach_prefilled_to_running(
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
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
                        maybe_trim_cuda_above_target(
                            device,
                            "VASR_TRIM_CUDA_MEMPOOL_BEFORE_DECODE_TARGET_MIB",
                        );
                        paged_batch_decode_slots_at_step(
                            thinker,
                            device,
                            &mut runtime_guard,
                            batch_slots.as_mut_slice(),
                            step,
                            eos_token_ids,
                        )?;
                        maybe_trim_cuda_above_target(
                            device,
                            "VASR_TRIM_CUDA_MEMPOOL_AFTER_DECODE_TARGET_MIB",
                        );
                        #[cfg(feature = "timing")]
                        if let Some(t) = timings.as_deref_mut() {
                            t.decode_us = t
                                .decode_us
                                .saturating_add(duration_to_us(start_decode.elapsed()));
                            t.steps = t.steps.saturating_add(1);
                            t.tokens_generated =
                                t.tokens_generated.saturating_add(slot_indices.len());
                        }

                        for (&slot_idx, slot) in slot_indices.iter().zip(batch_slots.iter()) {
                            if let Some(entry) = running[slot_idx].as_mut() {
                                entry.slot = slot.clone();
                            }
                        }
                    }
                }

                decode_debug.record_finished(running.as_slice());
                free_finished_running(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                );
                attach_prefilled_to_running(
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );

                let prefill_wave = self.prefill_waiting_wave(
                    thinker,
                    device,
                    &mut runtime_guard,
                    prepared,
                    audio_features,
                    audio_seconds,
                    row_audio_lens.as_slice(),
                    audio_offsets.as_slice(),
                    &mut waiting,
                    &mut prefilled,
                    eos_token_ids,
                    pad_id,
                    #[cfg(feature = "timing")]
                    &mut timings,
                )?;
                if prefill_wave > 0 {
                    attach_prefilled_to_running(
                        &mut prefilled,
                        running.as_mut_slice(),
                        #[cfg(feature = "timing")]
                        &mut timings,
                    );
                    let running_count = running.iter().filter(|s| s.is_some()).count();
                    let waiting_count = waiting.len();
                    let prefilled_count = prefilled.len();
                    tracing::debug!(
                        target: "vasr_transcribe::pipeline",
                        "continuous | prefill wave={prefill_wave} running={running_count}/{max_slots} prefilled={prefilled_count} waiting={waiting_count}",
                    );
                }

                let has_running = running.iter().any(|entry| entry.is_some());
                if waiting.is_empty() && prefilled.is_empty() && !has_running {
                    break;
                }
                if active_steps.is_empty()
                    && !has_running
                    && prefilled.is_empty()
                    && !waiting.is_empty()
                {
                    bail!(
                        "paged KV cache exhausted: {} sequence(s) waiting, free_blocks={}",
                        waiting.len(),
                        runtime_guard.manager().free_blocks()
                    );
                }
            }

            decode_debug.log_summary(max_slots);
            log_cuda_mem_debug(device, "decode_done");
            Ok(outputs)
        })();

        free_all_running(&mut runtime_guard, running.as_mut_slice());
        free_all_prefilled(&mut runtime_guard, &mut prefilled);
        result
    }

    pub fn run_continuous_prepared_dynamic(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        initial: PreparedAudioBatch,
        poll_source: &mut (dyn FnMut(bool) -> Result<PreparedAudioBatchPoll> + Send),
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        let max_num_seqs = {
            let runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            self.config.continuous_max_num_seqs(&runtime_guard)
        };
        let max_slots = max_num_seqs.max(1);
        let mut prefill_inputs = PrefillInputBuffer::default();
        let mut prefilled: VecDeque<PrefilledSequence> = VecDeque::new();
        let mut running: Vec<Option<RunningSlot>> = (0..max_slots).map(|_| None).collect();
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        let audio_token_id = thinker.audio_token_id();
        let mut source_closed = false;

        prefill_inputs.append(audio_token_id, initial)?;
        outputs.resize(prefill_inputs.len(), Vec::new());

        let result = (|| -> Result<Vec<Vec<u32>>> {
            let mut decode_debug = ContinuousDecodeDebug::default();
            loop {
                {
                    let mut runtime_guard = runtime
                        .lock()
                        .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
                    decode_debug.record_finished(running.as_slice());
                    free_finished_running(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                    );
                }
                attach_prefilled_to_running(
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );

                let mut active_steps: Vec<(usize, usize)> = Vec::new();
                for (slot_idx, entry) in running.iter().enumerate() {
                    if let Some(entry) = entry {
                        if !entry.slot.finished {
                            active_steps.push((slot_idx, entry.slot.decode_step));
                        }
                    }
                }

                decode_running_step_groups(
                    running.as_mut_slice(),
                    active_steps.as_slice(),
                    thinker,
                    device,
                    runtime,
                    eos_token_ids,
                    &mut decode_debug,
                    #[cfg(feature = "timing")]
                    &mut timings,
                )?;

                {
                    let mut runtime_guard = runtime
                        .lock()
                        .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
                    decode_debug.record_finished(running.as_slice());
                    free_finished_running(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                    );
                }
                attach_prefilled_to_running(
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );

                let has_running_before_poll = running.iter().any(|entry| entry.is_some());
                let may_block_source = dynamic_source_poll_may_block(
                    has_running_before_poll,
                    prefill_inputs.waiting_len(),
                );
                let prefill_wave = self.drive_dynamic_prefill_stage(
                    thinker,
                    device,
                    runtime,
                    &mut prefill_inputs,
                    &mut prefilled,
                    poll_source,
                    &mut source_closed,
                    audio_token_id,
                    eos_token_ids,
                    pad_id,
                    may_block_source,
                    #[cfg(feature = "timing")]
                    &mut timings,
                )?;
                outputs.resize(prefill_inputs.len(), Vec::new());

                if prefill_wave > 0 {
                    attach_prefilled_to_running(
                        &mut prefilled,
                        running.as_mut_slice(),
                        #[cfg(feature = "timing")]
                        &mut timings,
                    );
                    let running_count = running.iter().filter(|s| s.is_some()).count();
                    let waiting_count = prefill_inputs.waiting_len();
                    let prefilled_count = prefilled.len();
                    tracing::debug!(
                        target: "vasr_transcribe::pipeline",
                        "continuous | prefill wave={prefill_wave} running={running_count}/{max_slots} prefilled={prefilled_count} waiting={waiting_count}",
                    );
                }

                let has_running = running.iter().any(|entry| entry.is_some());
                if !prefill_inputs.has_waiting()
                    && prefilled.is_empty()
                    && !has_running
                    && source_closed
                {
                    break;
                }
                if !prefill_inputs.has_waiting()
                    && prefilled.is_empty()
                    && !has_running
                    && !source_closed
                {
                    thread::sleep(DYNAMIC_SOURCE_WAIT);
                }
                if active_steps.is_empty()
                    && !has_running
                    && prefilled.is_empty()
                    && prefill_inputs.has_waiting()
                {
                    let free_blocks = {
                        let runtime_guard = runtime
                            .lock()
                            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
                        runtime_guard.manager().free_blocks()
                    };
                    bail!(
                        "paged KV cache exhausted: {} sequence(s) waiting, free_blocks={}",
                        prefill_inputs.waiting_len(),
                        free_blocks
                    );
                }
            }

            decode_debug.log_summary(max_slots);
            Ok(outputs)
        })();

        {
            let mut runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            free_all_running(&mut runtime_guard, running.as_mut_slice());
            free_all_prefilled(&mut runtime_guard, &mut prefilled);
        }
        result
    }

    pub fn run_continuous_prepared_dynamic_async(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        initial: PreparedAudioBatch,
        poll_source: &mut (dyn FnMut(bool) -> Result<PreparedAudioBatchPoll> + Send),
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        let max_num_seqs = {
            let runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            self.config.continuous_max_num_seqs(&runtime_guard)
        };
        let max_slots = max_num_seqs.max(1);
        let (message_tx, message_rx) =
            mpsc::sync_channel::<Result<PrefillStageOutput>>(async_prefill_channel_capacity());
        let shared_request_ids = Arc::new(AtomicUsize::new(self.next_request_id));
        tracing::debug!(
            target: "vasr_transcribe::pipeline",
            "continuous async | prefill_producer=true decode_consumer=true max_slots={max_slots}",
        );

        thread::scope(|scope| -> Result<Vec<Vec<u32>>> {
            let producer_runtime = runtime;
            let producer_thinker = thinker;
            let producer_device = device;
            let producer_eos = eos_token_ids;
            let producer_config = self.config.clone();
            let producer_request_ids = Arc::clone(&shared_request_ids);
            scope.spawn(move || {
                let send_message =
                    |message: PrefillStageOutput| -> bool { message_tx.send(Ok(message)).is_ok() };

                let run = (|| -> Result<()> {
                    let mut producer = DynamicPrefillProducer::with_shared_request_ids(
                        producer_config,
                        producer_request_ids,
                        producer_thinker,
                        producer_device,
                        producer_runtime,
                        initial,
                        poll_source,
                        producer_eos,
                        pad_id,
                    )?;
                    loop {
                        let message = producer.run_stage(true)?;
                        let wave = message.prefill_wave;
                        let done = producer.is_done();
                        if wave > 0 || message.source_closed {
                            if !send_message(message) {
                                return Ok(());
                            }
                        }
                        if done {
                            break;
                        }
                        if wave == 0 {
                            thread::sleep(DYNAMIC_SOURCE_WAIT);
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = run {
                    let _ = message_tx.send(Err(error));
                }
            });

            let outputs = self.run_continuous_prefilled_dynamic(
                thinker,
                device,
                runtime,
                message_rx,
                eos_token_ids,
                max_slots,
                #[cfg(feature = "timing")]
                timings.as_deref_mut(),
            )?;

            self.next_request_id = shared_request_ids.load(Ordering::Relaxed);
            Ok(outputs)
        })
    }

    pub fn run_continuous_prefilled_dynamic(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        message_rx: mpsc::Receiver<Result<PrefillStageOutput>>,
        eos_token_ids: &[u32],
        max_slots: usize,
        #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        let mut prefilled: VecDeque<PrefilledSequence> = VecDeque::new();
        let mut running: Vec<Option<RunningSlot>> = (0..max_slots.max(1)).map(|_| None).collect();
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        let mut source_closed = false;
        let mut decode_debug = ContinuousDecodeDebug::default();
        let mut async_debug = AsyncContinuousDebug::default();

        let result = (|| -> Result<Vec<Vec<u32>>> {
            loop {
                while let Ok(message) = message_rx.try_recv() {
                    let message = message?;
                    let sequence_count = message.sequences.len();
                    apply_prefill_stage_message(
                        message,
                        &mut outputs,
                        &mut prefilled,
                        &mut source_closed,
                        #[cfg(feature = "timing")]
                        &mut timings,
                    );
                    async_debug.record_message(sequence_count, prefilled.len());
                }

                {
                    let mut runtime_guard = runtime
                        .lock()
                        .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
                    decode_debug.record_finished(running.as_slice());
                    free_finished_running(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                    );
                }
                attach_prefilled_to_running(
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );
                async_debug.record_prefilled_queue(prefilled.len());

                let mut active_steps: Vec<(usize, usize)> = Vec::new();
                for (slot_idx, entry) in running.iter().enumerate() {
                    if let Some(entry) = entry {
                        if !entry.slot.finished {
                            active_steps.push((slot_idx, entry.slot.decode_step));
                        }
                    }
                }

                decode_running_step_groups(
                    running.as_mut_slice(),
                    active_steps.as_slice(),
                    thinker,
                    device,
                    runtime,
                    eos_token_ids,
                    &mut decode_debug,
                    #[cfg(feature = "timing")]
                    &mut timings,
                )?;

                {
                    let mut runtime_guard = runtime
                        .lock()
                        .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
                    decode_debug.record_finished(running.as_slice());
                    free_finished_running(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                    );
                }
                attach_prefilled_to_running(
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );
                async_debug.record_prefilled_queue(prefilled.len());

                let has_running = running.iter().any(|entry| entry.is_some());
                if source_closed && prefilled.is_empty() && !has_running {
                    match message_rx.try_recv() {
                        Ok(message) => {
                            let message = message?;
                            let sequence_count = message.sequences.len();
                            apply_prefill_stage_message(
                                message,
                                &mut outputs,
                                &mut prefilled,
                                &mut source_closed,
                                #[cfg(feature = "timing")]
                                &mut timings,
                            );
                            async_debug.record_message(sequence_count, prefilled.len());
                            continue;
                        }
                        Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => {
                            break;
                        }
                    }
                }
                if prefilled.is_empty() && !has_running {
                    async_debug.record_empty_recv_wait();
                    match message_rx.recv_timeout(DYNAMIC_SOURCE_WAIT) {
                        Ok(message) => {
                            let message = message?;
                            let sequence_count = message.sequences.len();
                            apply_prefill_stage_message(
                                message,
                                &mut outputs,
                                &mut prefilled,
                                &mut source_closed,
                                #[cfg(feature = "timing")]
                                &mut timings,
                            );
                            async_debug.record_message(sequence_count, prefilled.len());
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            async_debug.record_timeout_wait();
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            source_closed = true;
                        }
                    }
                }
            }

            decode_debug.log_summary(max_slots);
            log_cuda_mem_debug(device, "decode_done");
            async_debug.log_summary();
            Ok(std::mem::take(&mut outputs))
        })();

        let cleanup_result = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))
            .map(|mut runtime_guard| {
                let running_count = running.iter().filter(|entry| entry.is_some()).count();
                let prefilled_count = prefilled.len();
                if result.is_err() && (running_count > 0 || prefilled_count > 0) {
                    tracing::warn!(
                        target: "vasr_transcribe::pipeline",
                        "cleaning paged KV after async decode error | running={running_count} prefilled={prefilled_count}",
                    );
                }
            free_all_running(&mut runtime_guard, running.as_mut_slice());
            free_all_prefilled(&mut runtime_guard, &mut prefilled);
            });
        if let Err(error) = cleanup_result {
            if result.is_ok() {
                return Err(error);
            }
            tracing::error!(
                target: "vasr_transcribe::pipeline",
                "failed to cleanup paged KV after async decode error: {error:#}",
            );
        }
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
    audio_seconds: Option<&[f64]>,
) -> Result<Vec<Vec<u32>>> {
    let config = AsrBatchSchedulerConfig::from_decode_max_seqs(decode_max_seqs);
    let mut scheduler = AsrBatchScheduler::new(config);
    let default_audio_seconds;
    let audio_seconds = match audio_seconds {
        Some(audio_seconds) => audio_seconds,
        None => {
            default_audio_seconds = vec![0.0; prepared.len()];
            default_audio_seconds.as_slice()
        }
    };
    if scheduler.config.continuous_batch {
        scheduler.run_continuous_prepared(
            thinker,
            device,
            runtime,
            prepared,
            audio_features,
            audio_seconds,
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
    audio_seconds: Option<&[f64]>,
) -> Result<(Vec<Vec<u32>>, GenerationTimings)> {
    let config = AsrBatchSchedulerConfig::from_decode_max_seqs(decode_max_seqs);
    let mut scheduler = AsrBatchScheduler::new(config);
    let mut timings = GenerationTimings::default();
    let default_audio_seconds;
    let audio_seconds = match audio_seconds {
        Some(audio_seconds) => audio_seconds,
        None => {
            default_audio_seconds = vec![0.0; prepared.len()];
            default_audio_seconds.as_slice()
        }
    };
    let ids = if scheduler.config.continuous_batch {
        scheduler.run_continuous_prepared(
            thinker,
            device,
            runtime,
            prepared,
            audio_features,
            audio_seconds,
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
            Some(audio_seconds),
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
        AsrBatchSchedulerConfig, collect_prefill_wave_by_kv, continuous_batch_enabled,
        dynamic_source_poll_may_block, group_running_by_decode_step, pad_prepared_wave_for_prefill,
    };
    use std::collections::VecDeque;

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
    fn collect_prefill_wave_uses_free_kv_not_queue_head() {
        let mut waiting = VecDeque::from(vec![0usize, 1, 2]);
        let capacities = [128usize, 32, 64];
        let wave = collect_prefill_wave_by_kv(&mut waiting, 32, 3, 3, |idx| Ok(capacities[idx]))
            .expect("collect wave");
        assert_eq!(wave, vec![1, 2]);
        assert_eq!(waiting.into_iter().collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn generated_token_budget_uses_minimum_for_short_audio() {
        assert_eq!(super::generated_token_budget(0.1, 4.25, 128), 128);
        assert_eq!(super::generated_token_budget(5.0, 4.25, 128), 128);
        assert_eq!(super::generated_token_budget(30.0, 4.25, 128), 128);
    }

    #[test]
    fn dynamic_source_poll_blocks_only_when_decoder_is_idle() {
        assert!(!dynamic_source_poll_may_block(true, 0));
        assert!(!dynamic_source_poll_may_block(false, 1));
        assert!(!dynamic_source_poll_may_block(true, 1));
        assert!(dynamic_source_poll_may_block(false, 0));
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
