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

use crate::inference::execution::{
    ExecutionOverrides, ExecutionPolicy, METAL_AUTO_MAX_ACTIVE_SLOTS, METAL_AUTO_MAX_PREFILL_WAVE,
};
#[cfg(feature = "timing")]
use crate::model::generation::GenerationTimings;
use crate::model::paged_batch_engine::{
    PagedBatchConfig, PagedDecodeSlot, paged_batch_decode_slots_at_step,
    paged_batch_decode_slots_at_step_shared, paged_batch_decode_slots_mixed_steps_shared,
    paged_batch_free_slots, paged_batch_prefill, paged_batch_prefill_shared, paged_batch_run,
    try_grant_one_kv_block, warn_decode_budget_exhausted,
};
use crate::model::thinker::ThinkerForConditionalGeneration;
use crate::processor::asr_processor::PreparedInputs;
#[cfg(feature = "cuda-graph")]
use vasr_paged_attn::cuda_graph::cuda_memory_snapshot;
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

fn log_cuda_mem_debug(device: &Device, label: &str, enabled: bool) {
    if !enabled {
        return;
    }
    let _ = label;
    #[cfg(feature = "cuda-graph")]
    if let Some(mem) = cuda_memory_snapshot(device) {
        let current_pool = match (mem.pool_used_current, mem.pool_reserved_current) {
            (Some(used), Some(reserved)) => format!(
                " pool_current={:.2}/{:.2}GiB",
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                reserved as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
            _ => String::new(),
        };
        let default_pool = match (
            mem.default_pool_used_current,
            mem.default_pool_reserved_current,
        ) {
            (Some(used), Some(reserved)) => format!(
                " pool_default={:.2}/{:.2}GiB",
                used as f64 / (1024.0 * 1024.0 * 1024.0),
                reserved as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
            _ => String::new(),
        };
        tracing::info!(
            target: "vasr_transcribe::pipeline",
            "cuda_mem | {label} used={:.2}GiB free={:.2}GiB total={:.2}GiB{}{}",
            mem.used() as f64 / (1024.0 * 1024.0 * 1024.0),
            mem.free as f64 / (1024.0 * 1024.0 * 1024.0),
            mem.total as f64 / (1024.0 * 1024.0 * 1024.0),
            current_pool,
            default_pool,
        );
    }
    let _ = device;
}

/// 记录模型 forward 执行锁的累计等待统计。
///
/// Metal 上 encoder、prefill、decode 共用一个 forward 锁，这个日志让性能报告能直接
/// 看到锁等待是否成为瓶颈，而不必从各阶段耗时反推。默认仅在 `VASR_DEBUG_LOCK_WAIT`
/// 开启时输出，避免日常启动刷屏。
fn log_model_execution_lock_wait(runtime: &SharedPagedCacheRuntime, enabled: bool) {
    if !enabled {
        return;
    }
    let Ok(guard) = runtime.lock() else {
        return;
    };
    let (acquisitions, waited_seconds) = guard.model_execution_wait_stats();
    if acquisitions == 0 {
        return;
    }
    tracing::info!(
        target: "vasr_transcribe::pipeline",
        "model_forward_lock | acquisitions={} waited={:.3}s avg_wait_ms={:.3}",
        acquisitions,
        waited_seconds,
        waited_seconds * 1000.0 / acquisitions as f64,
    );
}

#[derive(Debug, Clone)]
struct SchedulerEnvironment {
    execution: ExecutionOverrides,
    async_prefill_capacity: usize,
    length_schedule: bool,
    debug_cuda_mem: bool,
    debug_lock_wait: bool,
    debug_request_mem: bool,
    debug_decode_steps: bool,
}
impl SchedulerEnvironment {
    fn capture() -> Self {
        Self {
            execution: ExecutionOverrides::capture(),
            async_prefill_capacity: async_prefill_channel_capacity(),
            length_schedule: std::env::var_os("VASR_DISABLE_LENGTH_SCHEDULE").is_none(),
            debug_cuda_mem: std::env::var_os("VASR_DEBUG_CUDA_MEM").is_some(),
            debug_lock_wait: std::env::var_os("VASR_DEBUG_LOCK_WAIT").is_some(),
            debug_request_mem: std::env::var_os("VASR_DEBUG_REQUEST_MEM").is_some(),
            debug_decode_steps: std::env::var_os("VASR_DEBUG_CONTINUOUS_DECODE_STEPS").is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AsrBatchSchedulerConfig {
    pub max_num_seqs: usize,
    /// Per-sequence output cap used by continuous decode; mirrors the public ASR default.
    pub max_new_tokens: usize,
    explicit_decode_max_seqs: Option<usize>,
    prefill_max_wave: Option<usize>,
    pub continuous_batch: bool,
    pub generated_tokens_per_sec: f64,
    /// 启动时解析一次的统一执行策略。`None` 时按设备与容量自行推导。
    execution: Option<ExecutionPolicy>,
    environment: SchedulerEnvironment,
}

impl Default for AsrBatchSchedulerConfig {
    fn default() -> Self {
        let environment = SchedulerEnvironment::capture();
        Self {
            max_num_seqs: 8,
            max_new_tokens: 256,
            explicit_decode_max_seqs: None,
            prefill_max_wave: environment.execution.prefill_wave,
            continuous_batch: continuous_batch_enabled(),
            generated_tokens_per_sec: crate::models::qwen3asr::QWEN3_ASR_DECODE_TOKENS_PER_SEC,
            execution: None,
            environment,
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

    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = if max_new_tokens == 0 {
            256
        } else {
            max_new_tokens
        };
        self
    }

    /// 附加启动时解析出的统一执行策略。
    #[must_use]
    pub fn with_execution_policy(mut self, execution: Option<ExecutionPolicy>) -> Self {
        self.execution = execution;
        self
    }

    /// 解析本次调度实际使用的执行策略。
    ///
    /// 优先使用调用方传入的策略；未传入时按设备与容量推导。无论哪条路径，结果都会被
    /// 约束在本次解析出的安全容量以内，保证不会越过内存预算。
    fn effective_execution(&self, runtime: &PagedCacheRuntime, device: &Device) -> ExecutionPolicy {
        let capacity = self.continuous_max_num_seqs(runtime).max(1);
        let explicit_capacity =
            self.explicit_decode_max_seqs.is_some() || runtime_decode_max_seqs(runtime).is_some();
        let policy = self.execution.unwrap_or_else(|| {
            self.environment.execution.resolve_backend(
                device.is_metal(),
                capacity,
                explicit_capacity,
                METAL_AUTO_MAX_ACTIVE_SLOTS,
                METAL_AUTO_MAX_PREFILL_WAVE,
            )
        });
        policy.clamped_to(capacity)
    }

    pub fn prefill_max_wave(&self, runtime: &PagedCacheRuntime) -> usize {
        // 有统一策略时以策略为准（其中已包含环境变量覆盖）；否则退回旧行为。
        let capacity = self.continuous_max_num_seqs(runtime).max(1);
        if let Some(policy) = self.execution {
            return policy.prefill_wave.clamp(1, capacity);
        }
        self.prefill_max_wave.unwrap_or(capacity).max(1)
    }

    pub fn continuous_max_num_seqs(&self, runtime: &PagedCacheRuntime) -> usize {
        self.explicit_decode_max_seqs
            .or_else(|| runtime_decode_max_seqs(runtime))
            .unwrap_or(self.max_num_seqs)
    }

    pub(crate) fn execution_max_num_seqs(
        &self,
        runtime: &PagedCacheRuntime,
        device: &Device,
    ) -> usize {
        self.effective_execution(runtime, device).active_slots
    }
}

fn async_prefill_channel_capacity() -> usize {
    std::env::var("VASR_ASYNC_PREFILL_CHANNEL_CAP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
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
    /// 按 4.25 token/秒预留的生成长度，供空闲工位挑选下一条。
    pub predicted_generated: usize,
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
    debug_steps: bool,
}

impl ContinuousDecodeDebug {
    fn with_debug(debug_steps: bool) -> Self {
        Self {
            debug_steps,
            ..Self::default()
        }
    }
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

        if self.debug_steps {
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
        steps.extend(std::iter::repeat_n(step, count));
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
    /// 每段相对普通语速的倍率。缺省或长度不够时按 1 处理。
    pub speech_rate_ratios: Vec<f64>,
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
    pub reason: PrefillBatchReason,
    #[cfg(feature = "timing")]
    pub timings: GenerationTimings,
}

#[derive(Debug, Clone, Copy)]
pub enum PrefillBatchReason {
    MaxSlots,
    KvBlocks,
    WaitingEmpty,
}

impl PrefillBatchReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MaxSlots => "max_slots",
            Self::KvBlocks => "kv_blocks",
            Self::WaitingEmpty => "waiting_empty",
        }
    }
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
    base_index: usize,
    prepared: Vec<PreparedInputs>,
    audio_features: Option<Tensor>,
    audio_seconds: Vec<f64>,
    speech_rate_ratios: Vec<f64>,
    row_audio_lens: Vec<usize>,
    audio_offsets: Vec<usize>,
    waiting: VecDeque<usize>,
}

impl PrefillInputBuffer {
    fn append(&mut self, audio_token_id: u32, batch: PreparedAudioBatch) -> Result<()> {
        append_prepared_audio_batch(
            audio_token_id,
            self.base_index,
            &mut self.prepared,
            &mut self.audio_features,
            &mut self.audio_seconds,
            &mut self.speech_rate_ratios,
            &mut self.row_audio_lens,
            &mut self.audio_offsets,
            &mut self.waiting,
            batch,
        )
    }

    fn len(&self) -> usize {
        self.base_index.saturating_add(self.prepared.len())
    }

    fn has_waiting(&self) -> bool {
        !self.waiting.is_empty()
    }

    fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    fn compact_completed_prefix(&mut self) -> Result<()> {
        let Some(first_waiting) = self.waiting.iter().min().copied() else {
            self.base_index = self.len();
            self.prepared.clear();
            self.audio_features = None;
            self.audio_seconds.clear();
            self.speech_rate_ratios.clear();
            self.row_audio_lens.clear();
            self.audio_offsets.clear();
            return Ok(());
        };
        if first_waiting == self.base_index {
            return Ok(());
        }
        let drop_rows = first_waiting.saturating_sub(self.base_index);
        if drop_rows == 0 || drop_rows > self.prepared.len() {
            return Ok(());
        }

        let drop_tokens = self
            .audio_offsets
            .get(drop_rows)
            .copied()
            .or_else(|| {
                self.audio_offsets
                    .last()
                    .zip(self.row_audio_lens.last())
                    .map(|(offset, len)| offset.saturating_add(*len))
            })
            .unwrap_or(0);

        self.prepared.drain(..drop_rows);
        self.audio_seconds.drain(..drop_rows);
        if self.speech_rate_ratios.len() >= drop_rows {
            self.speech_rate_ratios.drain(..drop_rows);
        } else {
            self.speech_rate_ratios.clear();
        }
        self.row_audio_lens.drain(..drop_rows);
        self.audio_offsets.drain(..drop_rows);
        for offset in &mut self.audio_offsets {
            *offset = offset.saturating_sub(drop_tokens);
        }
        self.base_index = first_waiting;

        if let Some(features) = self.audio_features.take() {
            let (tokens, _) = features.dims2()?;
            let keep_tokens = tokens.saturating_sub(drop_tokens);
            self.audio_features = if keep_tokens == 0 {
                None
            } else {
                Some(features.narrow(0, drop_tokens, keep_tokens)?.copy()?)
            };
        }
        Ok(())
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
        let (wave, reason) = self.scheduler.drive_dynamic_prefill_stage(
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
            reason,
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

/// 取这一行的语速倍率。没有单独估计时按普通语速 1。
fn speech_rate_ratio_at(ratios: &[f64], row_index: usize) -> f64 {
    ratios
        .get(row_index)
        .copied()
        .filter(|ratio| ratio.is_finite() && *ratio > 0.0)
        .unwrap_or(1.0)
}

fn prepared_generated_tokens(
    audio_seconds: &[f64],
    speech_rate_ratios: &[f64],
    row_index: usize,
    generated_tokens_per_sec: f64,
) -> usize {
    super::speech_rate::predicted_generated_tokens(
        audio_seconds.get(row_index).copied().unwrap_or(0.0),
        generated_tokens_per_sec,
        speech_rate_ratio_at(speech_rate_ratios, row_index),
    )
}

fn prepared_token_capacity(
    row: &PreparedInputs,
    audio_seconds: &[f64],
    speech_rate_ratios: &[f64],
    row_index: usize,
    generated_tokens_per_sec: f64,
) -> Result<usize> {
    let generated = prepared_generated_tokens(
        audio_seconds,
        speech_rate_ratios,
        row_index,
        generated_tokens_per_sec,
    );
    prepared_prompt_len(row)
        .checked_add(generated)
        .ok_or_else(|| anyhow::anyhow!("prepared token capacity overflow"))
}

/// 空闲工位出现时，领预测生成长度最长的那条。关闭长度规划时仍按到达顺序。
fn pop_scheduled_prefilled(
    prefilled: &mut VecDeque<PrefilledSequence>,
    length_schedule: bool,
) -> Option<PrefilledSequence> {
    if !length_schedule
        || prefilled.len() <= 1
        || prefilled.iter().all(|seq| seq.predicted_generated == 0)
    {
        return prefilled.pop_front();
    }
    let index = prefilled
        .iter()
        .enumerate()
        .max_by_key(|(_, seq)| (seq.predicted_generated, usize::MAX - seq.client_index))
        .map(|(index, _)| index)?;
    prefilled.remove(index)
}

/// 两个工位的子集和划分。返回两侧作业下标，使长度和尽量接近总长的一半。
#[cfg(test)]
fn partition_two_slots(lengths: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let total = lengths.iter().copied().sum::<usize>();
    let limit = total / 2;
    let mut reachable = vec![false; limit.saturating_add(1)];
    let mut previous = vec![None; limit.saturating_add(1)];
    let mut chosen = vec![None; limit.saturating_add(1)];
    reachable[0] = true;
    for (item, &length) in lengths.iter().enumerate() {
        if length == 0 || length > limit {
            continue;
        }
        for sum in (0..=limit.saturating_sub(length)).rev() {
            let next = sum.saturating_add(length);
            if reachable[sum] && !reachable[next] {
                reachable[next] = true;
                previous[next] = Some(sum);
                chosen[next] = Some(item);
            }
        }
    }
    let best = (0..=limit).rev().find(|&sum| reachable[sum]).unwrap_or(0);
    let mut side_a = Vec::new();
    let mut cursor = best;
    while cursor != 0 {
        let Some(item) = chosen[cursor] else {
            break;
        };
        side_a.push(item);
        cursor = previous[cursor].unwrap_or(0);
    }
    side_a.sort_unstable();
    let side_b = (0..lengths.len())
        .filter(|item| !side_a.contains(item))
        .collect();
    (side_a, side_b)
}

fn attach_prefilled_to_running(
    length_schedule: bool,
    prefilled: &mut VecDeque<PrefilledSequence>,
    running: &mut [Option<RunningSlot>],
    #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
) -> bool {
    let mut attached = false;
    for slot in running.iter_mut() {
        if slot.is_some() {
            continue;
        }
        let Some(prefilled) = pop_scheduled_prefilled(prefilled, length_schedule) else {
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
    retain_outputs: bool,
    #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
) {
    if retain_outputs {
        outputs.resize(outputs.len().max(message.total_rows), Vec::new());
    }
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
) -> Result<(Vec<usize>, PrefillBatchReason)>
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
    let reason = if wave.len() >= max_wave {
        PrefillBatchReason::MaxSlots
    } else if waiting.is_empty() {
        PrefillBatchReason::WaitingEmpty
    } else {
        PrefillBatchReason::KvBlocks
    };
    Ok((wave, reason))
}

fn free_finished_running(
    running: &mut [Option<RunningSlot>],
    outputs: &mut [Vec<u32>],
    runtime: &mut PagedCacheRuntime,
    mut on_complete: Option<&mut dyn FnMut(usize, Vec<u32>) -> Result<()>>,
    retain_outputs: bool,
) -> Result<bool> {
    let mut freed = false;
    for slot in running.iter_mut() {
        let Some(entry) = slot.as_mut() else {
            continue;
        };
        if !entry.slot.finished {
            continue;
        }
        let generated = entry.slot.generated.clone();
        if retain_outputs {
            outputs[entry.client_index] = generated.clone();
        }
        if let Some(on_complete) = on_complete.as_deref_mut() {
            on_complete(entry.client_index, generated)?;
        }
        paged_batch_free_slots(runtime, &[entry.slot.request_id]);
        *slot = None;
        freed = true;
    }
    Ok(freed)
}

/// 结束已经写完的序列，再把退回来的 KV 优先补给还没生成完的序列。
///
/// 每条只补一块，剩下的空闲块留给后面的 prefill。补上之后这条序列继续占着原来的槽位解码。
/// 一块都补不到、旁边也没有还能往下写的序列时，把还在等的序列标成结束。
/// 返回 true 时调用方要再退一次它们的显存。
fn free_finished_and_grant_kv(
    running: &mut [Option<RunningSlot>],
    outputs: &mut [Vec<u32>],
    runtime: &mut PagedCacheRuntime,
    on_complete: Option<&mut dyn FnMut(usize, Vec<u32>) -> Result<()>>,
    retain_outputs: bool,
    prefilled_len: usize,
) -> Result<bool> {
    free_finished_running(running, outputs, runtime, on_complete, retain_outputs)?;
    grant_freed_kv_to_awaiting(running, runtime)?;
    Ok(release_awaiting_if_no_progress(running, prefilled_len))
}

/// 按槽位顺序，给每条预留已用尽的序列补一块 KV。
fn grant_freed_kv_to_awaiting(
    running: &mut [Option<RunningSlot>],
    runtime: &mut PagedCacheRuntime,
) -> Result<()> {
    let pending: Vec<usize> = running
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| {
            slot.as_ref().and_then(|entry| {
                if entry.slot.awaiting_kv && !entry.slot.finished {
                    Some(index)
                } else {
                    None
                }
            })
        })
        .collect();
    for index in pending {
        let Some(entry) = running[index].as_ref() else {
            continue;
        };
        let request_id = entry.slot.request_id;
        let prompt_len = entry.slot.prompt_len_usize;
        let generated_len = entry.slot.generated.len();
        let mut capacity = entry.slot.token_capacity;
        let mut block_table = entry.slot.block_table.clone();
        let granted = try_grant_one_kv_block(
            runtime.manager_mut(),
            request_id,
            &mut capacity,
            &mut block_table,
            prompt_len,
            generated_len,
        )?;
        if !granted {
            continue;
        }
        let Some(entry) = running[index].as_mut() else {
            continue;
        };
        entry.slot.token_capacity = capacity;
        entry.slot.block_table = block_table;
        entry.slot.awaiting_kv = false;
    }
    Ok(())
}

/// 没有序列还能往下写、也没有已经 prefill 好等着进空槽的序列时，结束还在等 KV 的序列。
///
/// 返回 true 表示有序列被标成结束，调用方要把它们的显存退回池子。
fn release_awaiting_if_no_progress(
    running: &mut [Option<RunningSlot>],
    prefilled_len: usize,
) -> bool {
    let runnable = running.iter().any(|slot| {
        slot.as_ref()
            .is_some_and(|entry| !entry.slot.finished && !entry.slot.awaiting_kv)
    });
    if runnable {
        return false;
    }
    // 还有空槽，并且 prefill 已经准备好下一条，先让那条跑起来，等它结束再把块补给还没写完的序列。
    let free_slot = running.iter().any(|slot| slot.is_none());
    if free_slot && prefilled_len > 0 {
        return false;
    }
    let mut released = false;
    for slot in running.iter_mut() {
        let Some(entry) = slot.as_mut() else {
            continue;
        };
        if entry.slot.finished || !entry.slot.awaiting_kv {
            continue;
        }
        warn_decode_budget_exhausted(
            entry.slot.request_id,
            entry.slot.prompt_len_usize,
            entry.slot.generated.len(),
            entry.slot.token_capacity,
            entry.slot.generated.last().copied().unwrap_or(0),
        );
        entry.slot.finished = true;
        entry.slot.awaiting_kv = false;
        released = true;
    }
    released
}

#[allow(clippy::too_many_arguments)]
fn decode_running_step_groups(
    running: &mut [Option<RunningSlot>],
    active_steps: &[(usize, usize)],
    max_new_tokens: usize,
    execution: &ExecutionPolicy,
    thinker: &ThinkerForConditionalGeneration,
    device: &Device,
    runtime: &SharedPagedCacheRuntime,
    eos_token_ids: &[u32],
    decode_debug: &mut ContinuousDecodeDebug,
    #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
) -> Result<()> {
    let active_steps = filter_sequences_at_token_limit(running, active_steps, max_new_tokens);
    if active_steps.is_empty() {
        return Ok(());
    }
    if execution.mixed_step {
        let slot_indices: Vec<usize> = active_steps.iter().map(|&(slot_idx, _)| slot_idx).collect();
        let batch_size = execution.decode_rows.min(slot_indices.len()).max(1);
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
                        awaiting_kv: false,
                    })
            })
            .collect();
        for slot_group in slot_indices.chunks(batch_size) {
            decode_debug.record_active(0, slot_group.len());
            #[cfg(feature = "timing")]
            let start_decode = std::time::Instant::now();
            paged_batch_decode_slots_mixed_steps_shared(
                thinker,
                device,
                runtime,
                batch_slots.as_mut_slice(),
                slot_group,
                eos_token_ids,
            )?;
            #[cfg(feature = "timing")]
            if let Some(t) = timings.as_deref_mut() {
                t.decode_us = t
                    .decode_us
                    .saturating_add(duration_to_us(start_decode.elapsed()));
                t.steps = t.steps.saturating_add(1);
                t.tokens_generated = t.tokens_generated.saturating_add(slot_group.len());
            }
        }
        for &slot_idx in &slot_indices {
            if let Some(entry) = running[slot_idx].as_mut() {
                entry.slot = batch_slots[slot_idx].clone();
            }
        }
        return Ok(());
    }
    let step_groups = group_running_by_decode_step(&active_steps);
    for (step, slot_indices) in step_groups {
        let batch_size = execution.decode_rows.min(slot_indices.len()).max(1);
        for slot_group in slot_indices.chunks(batch_size) {
            decode_debug.record_active(step, slot_group.len());
            let mut batch_slots: Vec<PagedDecodeSlot> = slot_group
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
            paged_batch_decode_slots_at_step_shared(
                thinker,
                device,
                runtime,
                batch_slots.as_mut_slice(),
                step,
                eos_token_ids,
            )?;
            #[cfg(feature = "timing")]
            if let Some(t) = timings.as_deref_mut() {
                t.decode_us = t
                    .decode_us
                    .saturating_add(duration_to_us(start_decode.elapsed()));
                t.steps = t.steps.saturating_add(1);
                t.tokens_generated = t.tokens_generated.saturating_add(slot_group.len());
            }

            for (&slot_idx, slot) in slot_group.iter().zip(batch_slots.iter()) {
                if let Some(entry) = running[slot_idx].as_mut() {
                    entry.slot = slot.clone();
                }
            }
        }
    }
    Ok(())
}

/// Finish sequences that have already emitted their full output budget before another
/// decoder step is submitted. Scan all slots (including those waiting for KV) so a capped
/// sequence cannot wait indefinitely for another block.
fn filter_sequences_at_token_limit(
    running: &mut [Option<RunningSlot>],
    active_steps: &[(usize, usize)],
    max_new_tokens: usize,
) -> Vec<(usize, usize)> {
    for entry in running.iter_mut().flatten() {
        if !entry.slot.finished && entry.slot.generated.len() >= max_new_tokens {
            entry.slot.finished = true;
            entry.slot.awaiting_kv = false;
        }
    }
    active_steps
        .iter()
        .copied()
        .filter(|&(slot_idx, _)| {
            running
                .get(slot_idx)
                .and_then(Option::as_ref)
                .is_some_and(|entry| !entry.slot.finished && !entry.slot.awaiting_kv)
        })
        .collect()
}

/// 右对齐填充后的一批 token 行，分别对应输入 id 与标签 id。
type PaddedWaveTokens = (Vec<Vec<u32>>, Vec<Vec<u32>>);

/// Right-pad selected rows to the max effective prompt length within the admit wave.
fn pad_prepared_wave_for_prefill(
    prepared: &[PreparedInputs],
    wave_indices: &[usize],
    pad_id: u32,
) -> Result<PaddedWaveTokens> {
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

#[allow(clippy::too_many_arguments)]
fn append_prepared_audio_batch(
    audio_token_id: u32,
    base_index: usize,
    prepared: &mut Vec<PreparedInputs>,
    audio_features: &mut Option<Tensor>,
    audio_seconds: &mut Vec<f64>,
    speech_rate_ratios: &mut Vec<f64>,
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
    if !batch.speech_rate_ratios.is_empty()
        && batch.speech_rate_ratios.len() != batch.prepared.len()
    {
        bail!(
            "prepared speech-rate ratio mismatch: prepared={} ratios={}",
            batch.prepared.len(),
            batch.speech_rate_ratios.len()
        );
    }
    let start_idx = base_index.saturating_add(prepared.len());
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
    if batch.speech_rate_ratios.is_empty() {
        speech_rate_ratios.extend(std::iter::repeat_n(1.0, count));
    } else {
        speech_rate_ratios.append(&mut batch.speech_rate_ratios);
    }
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
    #[allow(clippy::too_many_arguments)]
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

        // The legacy static micro-batch path calls the non-shared paged engine
        // directly, so it must hold the same Metal gate across prefill and decode.
        // Fetch the gate and release the cache mutex before waiting on it to keep
        // lock ordering consistent with the shared staged path.
        let execution_lock = if device.is_metal() {
            Some(
                runtime
                    .lock()
                    .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?
                    .model_execution_lock(),
            )
        } else {
            None
        };
        let _execution_guard = execution_lock
            .as_ref()
            .map(|lock| {
                lock.lock()
                    .map_err(|_| anyhow::anyhow!("Metal model execution lock poisoned"))
            })
            .transpose()?;

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

    #[allow(clippy::too_many_arguments)]
    fn prefill_waiting_wave(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &mut PagedCacheRuntime,
        prepared: &[PreparedInputs],
        audio_features: &Tensor,
        audio_seconds: &[f64],
        speech_rate_ratios: &[f64],
        row_audio_lens: &[usize],
        audio_offsets: &[usize],
        waiting: &mut VecDeque<usize>,
        prefilled: &mut VecDeque<PrefilledSequence>,
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
    ) -> Result<(usize, PrefillBatchReason)> {
        let manager = runtime.manager();
        let block_size = manager.block_size();
        let free_blocks = manager.free_blocks();
        let max_wave = self.config.prefill_max_wave(runtime);
        let (wave_client_indices, reason) = collect_prefill_wave_by_kv(
            waiting,
            block_size,
            free_blocks,
            max_wave,
            |client_index| {
                prepared_token_capacity(
                    &prepared[client_index],
                    audio_seconds,
                    speech_rate_ratios,
                    client_index,
                    self.config.generated_tokens_per_sec,
                )
            },
        )?;

        if wave_client_indices.is_empty() {
            return Ok((0, reason));
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
                    speech_rate_ratios,
                    client_index,
                    self.config.generated_tokens_per_sec,
                )
            })
            .collect::<Result<Vec<_>>>()?;
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
        #[cfg(feature = "timing")]
        let elapsed = duration_to_us(start_prefill.elapsed());
        #[cfg(feature = "timing")]
        let prefill_tokens = state.prompt_lens_usize.iter().copied().sum::<usize>();
        log_cuda_mem_debug(
            device,
            "prefill_done",
            self.config.environment.debug_cuda_mem,
        );
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
                predicted_generated: prepared_generated_tokens(
                    audio_seconds,
                    speech_rate_ratios,
                    client_index,
                    self.config.generated_tokens_per_sec,
                ),
                prefill_done_at,
            });
        }
        Ok((wave, reason))
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
        speech_rate_ratios: &[f64],
        row_audio_lens: &[usize],
        audio_offsets: &[usize],
        waiting: &mut VecDeque<usize>,
        prefilled: &mut VecDeque<PrefilledSequence>,
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] timings: &mut Option<&mut GenerationTimings>,
        base_index: usize,
    ) -> Result<(usize, PrefillBatchReason)> {
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
        let (wave_client_indices, reason) = collect_prefill_wave_by_kv(
            waiting,
            block_size,
            free_blocks,
            max_wave,
            |client_index| {
                let local_index = client_index.checked_sub(base_index).ok_or_else(|| {
                    anyhow::anyhow!(
                        "dynamic prefill index underflow: client={client_index} base={base_index}"
                    )
                })?;
                prepared_token_capacity(
                    &prepared[local_index],
                    audio_seconds,
                    speech_rate_ratios,
                    local_index,
                    self.config.generated_tokens_per_sec,
                )
            },
        )?;

        if wave_client_indices.is_empty() {
            return Ok((0, reason));
        }

        let wave = wave_client_indices.len();
        let wave_local_indices = wave_client_indices
            .iter()
            .map(|&client_index| {
                client_index.checked_sub(base_index).ok_or_else(|| {
                    anyhow::anyhow!(
                        "dynamic prefill index underflow: client={client_index} base={base_index}"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let request_id_base = self.allocate_request_id_base(wave);
        let (wave_ids, wave_masks) =
            pad_prepared_wave_for_prefill(prepared, wave_local_indices.as_slice(), pad_id)?;
        let ids_rows: Vec<&[u32]> = wave_ids.iter().map(Vec::as_slice).collect();
        let attn_rows: Vec<&[u32]> = wave_masks.iter().map(Vec::as_slice).collect();
        let token_capacities = wave_local_indices
            .iter()
            .map(|&local_index| {
                prepared_token_capacity(
                    &prepared[local_index],
                    audio_seconds,
                    speech_rate_ratios,
                    local_index,
                    self.config.generated_tokens_per_sec,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let wave_audio = concat_prepared_audio(
            audio_features,
            audio_offsets,
            row_audio_lens,
            wave_local_indices.as_slice(),
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
        #[cfg(feature = "timing")]
        let elapsed = duration_to_us(start_prefill.elapsed());
        #[cfg(feature = "timing")]
        let prefill_tokens = state.prompt_lens_usize.iter().copied().sum::<usize>();
        log_cuda_mem_debug(
            device,
            "prefill_done",
            self.config.environment.debug_cuda_mem,
        );
        #[cfg(feature = "timing")]
        if let Some(t) = timings.as_deref_mut() {
            t.prefill_us = t.prefill_us.saturating_add(elapsed);
            t.prefill_forward_us = t.prefill_forward_us.saturating_add(elapsed);
            t.prefill_tokens = t.prefill_tokens.saturating_add(prefill_tokens);
        }
        let prefill_done_at = Instant::now();
        for (wave_row, (&client_index, &local_index)) in wave_client_indices
            .iter()
            .zip(wave_local_indices.iter())
            .enumerate()
        {
            prefilled.push_back(PrefilledSequence {
                client_index,
                slot: PagedDecodeSlot::from_prefill_state(&state, wave_row)?,
                predicted_generated: prepared_generated_tokens(
                    audio_seconds,
                    speech_rate_ratios,
                    local_index,
                    self.config.generated_tokens_per_sec,
                ),
                prefill_done_at,
            });
        }
        Ok((wave, reason))
    }

    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(usize, PrefillBatchReason)> {
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
            return Ok((0, PrefillBatchReason::WaitingEmpty));
        };
        let result = self.prefill_waiting_wave_shared(
            thinker,
            device,
            runtime,
            inputs.prepared.as_slice(),
            features,
            inputs.audio_seconds.as_slice(),
            inputs.speech_rate_ratios.as_slice(),
            inputs.row_audio_lens.as_slice(),
            inputs.audio_offsets.as_slice(),
            &mut inputs.waiting,
            prefilled,
            eos_token_ids,
            pad_id,
            #[cfg(feature = "timing")]
            timings,
            inputs.base_index,
        );
        if result.as_ref().is_ok_and(|(wave, _)| *wave > 0) {
            inputs.compact_completed_prefix()?;
        }
        if self.config.environment.debug_request_mem {
            tracing::info!(
                target: "vasr_transcribe::pipeline",
                "metadata_lifetime | stage=prefill_buffer prepared={} waiting={} audio_tokens={}",
                inputs.prepared.len(), inputs.waiting.len(),
                inputs.audio_features.as_ref().map(|features| features.dims()[0]).unwrap_or(0),
            );
        }
        result
    }

    /// Continuous batching for variable-length prepared segments (VAD slices).
    ///
    /// `waiting` starts as all segment indices; each admit wave is right-padded locally and
    /// prefilled with per-row `query_lens` for flash varlen metadata.
    #[allow(clippy::too_many_arguments)]
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

        // Unlike the asynchronous shared scheduler, this offline path calls the
        // non-shared paged engine while retaining the runtime mutex. Acquire the
        // device gate first to preserve a single lock order across all paths.
        let execution_lock = if device.is_metal() {
            Some(
                runtime
                    .lock()
                    .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?
                    .model_execution_lock(),
            )
        } else {
            None
        };
        let _execution_guard = execution_lock
            .as_ref()
            .map(|lock| {
                lock.lock()
                    .map_err(|_| anyhow::anyhow!("Metal model execution lock poisoned"))
            })
            .transpose()?;

        let mut runtime_guard = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
        let max_num_seqs = self.config.execution_max_num_seqs(&runtime_guard, device);
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
            let mut decode_debug =
                ContinuousDecodeDebug::with_debug(self.config.environment.debug_decode_steps);
            loop {
                decode_debug.record_finished(running.as_slice());
                if free_finished_and_grant_kv(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                    None,
                    true,
                    prefilled.len(),
                )? {
                    free_finished_running(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                        None,
                        true,
                    )?;
                }
                attach_prefilled_to_running(
                    self.config.environment.length_schedule,
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );

                let mut active_steps: Vec<(usize, usize)> = Vec::new();
                for (slot_idx, entry) in running.iter().enumerate() {
                    if let Some(entry) = entry
                        && !entry.slot.finished
                        && !entry.slot.awaiting_kv
                    {
                        active_steps.push((slot_idx, entry.slot.decode_step));
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
                            eos_token_ids,
                        )?;
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
                if free_finished_and_grant_kv(
                    running.as_mut_slice(),
                    outputs.as_mut_slice(),
                    &mut runtime_guard,
                    None,
                    true,
                    prefilled.len(),
                )? {
                    free_finished_running(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                        None,
                        true,
                    )?;
                }
                attach_prefilled_to_running(
                    self.config.environment.length_schedule,
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );

                let (prefill_wave, _prefill_reason) = self.prefill_waiting_wave(
                    thinker,
                    device,
                    &mut runtime_guard,
                    prepared,
                    audio_features,
                    audio_seconds,
                    &[],
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
                        self.config.environment.length_schedule,
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
            log_cuda_mem_debug(
                device,
                "decode_done",
                self.config.environment.debug_cuda_mem,
            );
            Ok(outputs)
        })();

        free_all_running(&mut runtime_guard, running.as_mut_slice());
        free_all_prefilled(&mut runtime_guard, &mut prefilled);
        result
    }

    #[allow(clippy::too_many_arguments)]
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
        let (max_num_seqs, execution) = {
            let runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            (
                self.config.execution_max_num_seqs(&runtime_guard, device),
                self.config.effective_execution(&runtime_guard, device),
            )
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
            let mut decode_debug =
                ContinuousDecodeDebug::with_debug(self.config.environment.debug_decode_steps);
            loop {
                {
                    let mut runtime_guard = runtime
                        .lock()
                        .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
                    decode_debug.record_finished(running.as_slice());
                    if free_finished_and_grant_kv(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                        None,
                        true,
                        prefilled.len(),
                    )? {
                        free_finished_running(
                            running.as_mut_slice(),
                            outputs.as_mut_slice(),
                            &mut runtime_guard,
                            None,
                            true,
                        )?;
                    }
                }
                attach_prefilled_to_running(
                    self.config.environment.length_schedule,
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );

                let mut active_steps: Vec<(usize, usize)> = Vec::new();
                for (slot_idx, entry) in running.iter().enumerate() {
                    if let Some(entry) = entry
                        && !entry.slot.finished
                        && !entry.slot.awaiting_kv
                    {
                        active_steps.push((slot_idx, entry.slot.decode_step));
                    }
                }

                decode_running_step_groups(
                    running.as_mut_slice(),
                    active_steps.as_slice(),
                    self.config.max_new_tokens,
                    &execution,
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
                    if free_finished_and_grant_kv(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                        None,
                        true,
                        prefilled.len(),
                    )? {
                        free_finished_running(
                            running.as_mut_slice(),
                            outputs.as_mut_slice(),
                            &mut runtime_guard,
                            None,
                            true,
                        )?;
                    }
                }
                attach_prefilled_to_running(
                    self.config.environment.length_schedule,
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
                let (prefill_wave, _prefill_reason) = self.drive_dynamic_prefill_stage(
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
                        self.config.environment.length_schedule,
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

    #[allow(clippy::too_many_arguments)]
    pub fn run_continuous_prepared_dynamic_async(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        initial: PreparedAudioBatch,
        poll_source: &mut (dyn FnMut(bool) -> Result<PreparedAudioBatchPoll> + Send),
        eos_token_ids: &[u32],
        pad_id: u32,
        #[cfg(feature = "timing")] timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        let max_num_seqs = {
            let runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            self.config.execution_max_num_seqs(&runtime_guard, device)
        };
        let max_slots = max_num_seqs.max(1);
        let (message_tx, message_rx) = mpsc::sync_channel::<Result<PrefillStageOutput>>(
            self.config.environment.async_prefill_capacity,
        );
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
                        if (wave > 0 || message.source_closed) && !send_message(message) {
                            return Ok(());
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
                timings,
            )?;

            self.next_request_id = shared_request_ids.load(Ordering::Relaxed);
            Ok(outputs)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_continuous_prefilled_dynamic(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        message_rx: mpsc::Receiver<Result<PrefillStageOutput>>,
        eos_token_ids: &[u32],
        max_slots: usize,
        #[cfg(feature = "timing")] timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        self.run_continuous_prefilled_dynamic_with_completed(
            thinker,
            device,
            runtime,
            message_rx,
            eos_token_ids,
            max_slots,
            &mut |_, _| Ok(()),
            #[cfg(feature = "timing")]
            timings,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_continuous_prefilled_dynamic_with_completed(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        message_rx: mpsc::Receiver<Result<PrefillStageOutput>>,
        eos_token_ids: &[u32],
        max_slots: usize,
        on_complete: &mut dyn FnMut(usize, Vec<u32>) -> Result<()>,
        #[cfg(feature = "timing")] timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        self.run_continuous_prefilled_dynamic_with_completed_impl(
            thinker,
            device,
            runtime,
            message_rx,
            eos_token_ids,
            max_slots,
            on_complete,
            true,
            true,
            None,
            #[cfg(feature = "timing")]
            timings,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_global_continuous_prefilled_dynamic_with_completed(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        message_rx: mpsc::Receiver<Result<PrefillStageOutput>>,
        eos_token_ids: &[u32],
        max_slots: usize,
        on_complete: &mut dyn FnMut(usize, Vec<u32>) -> Result<()>,
        on_tick: &mut dyn FnMut() -> Result<()>,
        #[cfg(feature = "timing")] timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        self.run_continuous_prefilled_dynamic_with_completed_impl(
            thinker,
            device,
            runtime,
            message_rx,
            eos_token_ids,
            max_slots,
            on_complete,
            false,
            false,
            Some(on_tick),
            #[cfg(feature = "timing")]
            timings,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_continuous_prefilled_dynamic_with_completed_impl(
        &mut self,
        thinker: &ThinkerForConditionalGeneration,
        device: &Device,
        runtime: &SharedPagedCacheRuntime,
        message_rx: mpsc::Receiver<Result<PrefillStageOutput>>,
        eos_token_ids: &[u32],
        max_slots: usize,
        on_complete: &mut dyn FnMut(usize, Vec<u32>) -> Result<()>,
        retain_outputs: bool,
        close_on_message_source_closed: bool,
        mut on_tick: Option<&mut dyn FnMut() -> Result<()>>,
        #[cfg(feature = "timing")] mut timings: Option<&mut GenerationTimings>,
    ) -> Result<Vec<Vec<u32>>> {
        // 解析本次连续解码实际使用的执行策略（活跃槽位、decode 行数、mixed-step）。
        let execution = {
            let runtime_guard = runtime
                .lock()
                .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
            self.config.effective_execution(&runtime_guard, device)
        };
        let mut prefilled: VecDeque<PrefilledSequence> = VecDeque::new();
        let mut running: Vec<Option<RunningSlot>> = (0..max_slots.max(1)).map(|_| None).collect();
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        let mut source_closed = false;
        let mut decode_debug =
            ContinuousDecodeDebug::with_debug(self.config.environment.debug_decode_steps);
        let mut async_debug = AsyncContinuousDebug::default();
        // A global service keeps this loop alive across requests, so waiting
        // until loop exit would never report lock statistics during a benchmark.
        let mut lock_wait_reported_at = self.config.environment.debug_lock_wait.then(Instant::now);

        let result = (|| -> Result<Vec<Vec<u32>>> {
            loop {
                if let Some(last) = &mut lock_wait_reported_at {
                    if last.elapsed() >= Duration::from_secs(5) {
                        log_model_execution_lock_wait(
                            runtime,
                            self.config.environment.debug_lock_wait,
                        );
                        *last = Instant::now();
                    }
                }
                if let Some(on_tick) = on_tick.as_deref_mut() {
                    on_tick()?;
                }
                while let Ok(message) = message_rx.try_recv() {
                    let message = message?;
                    let sequence_count = message.sequences.len();
                    let message_source_closed = message.source_closed;
                    apply_prefill_stage_message(
                        message,
                        &mut outputs,
                        &mut prefilled,
                        &mut source_closed,
                        retain_outputs,
                        #[cfg(feature = "timing")]
                        &mut timings,
                    );
                    if !close_on_message_source_closed && message_source_closed {
                        source_closed = false;
                    }
                    async_debug.record_message(sequence_count, prefilled.len());
                }

                {
                    let mut runtime_guard = runtime
                        .lock()
                        .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))?;
                    decode_debug.record_finished(running.as_slice());
                    if free_finished_and_grant_kv(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                        Some(&mut *on_complete),
                        retain_outputs,
                        prefilled.len(),
                    )? {
                        free_finished_running(
                            running.as_mut_slice(),
                            outputs.as_mut_slice(),
                            &mut runtime_guard,
                            Some(&mut *on_complete),
                            retain_outputs,
                        )?;
                    }
                }
                attach_prefilled_to_running(
                    self.config.environment.length_schedule,
                    &mut prefilled,
                    running.as_mut_slice(),
                    #[cfg(feature = "timing")]
                    &mut timings,
                );
                async_debug.record_prefilled_queue(prefilled.len());

                let mut active_steps: Vec<(usize, usize)> = Vec::new();
                for (slot_idx, entry) in running.iter().enumerate() {
                    if let Some(entry) = entry
                        && !entry.slot.finished
                        && !entry.slot.awaiting_kv
                    {
                        active_steps.push((slot_idx, entry.slot.decode_step));
                    }
                }

                decode_running_step_groups(
                    running.as_mut_slice(),
                    active_steps.as_slice(),
                    self.config.max_new_tokens,
                    &execution,
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
                    if free_finished_and_grant_kv(
                        running.as_mut_slice(),
                        outputs.as_mut_slice(),
                        &mut runtime_guard,
                        Some(&mut *on_complete),
                        retain_outputs,
                        prefilled.len(),
                    )? {
                        free_finished_running(
                            running.as_mut_slice(),
                            outputs.as_mut_slice(),
                            &mut runtime_guard,
                            Some(&mut *on_complete),
                            retain_outputs,
                        )?;
                    }
                }
                attach_prefilled_to_running(
                    self.config.environment.length_schedule,
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
                            let message_source_closed = message.source_closed;
                            apply_prefill_stage_message(
                                message,
                                &mut outputs,
                                &mut prefilled,
                                &mut source_closed,
                                retain_outputs,
                                #[cfg(feature = "timing")]
                                &mut timings,
                            );
                            if !close_on_message_source_closed && message_source_closed {
                                source_closed = false;
                            }
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
                            let message_source_closed = message.source_closed;
                            apply_prefill_stage_message(
                                message,
                                &mut outputs,
                                &mut prefilled,
                                &mut source_closed,
                                retain_outputs,
                                #[cfg(feature = "timing")]
                                &mut timings,
                            );
                            if !close_on_message_source_closed && message_source_closed {
                                source_closed = false;
                            }
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
            log_cuda_mem_debug(
                device,
                "decode_done",
                self.config.environment.debug_cuda_mem,
            );
            async_debug.log_summary();
            log_model_execution_lock_wait(runtime, self.config.environment.debug_lock_wait);
            Ok(std::mem::take(&mut outputs))
        })();

        let cleanup_result = runtime
            .lock()
            .map_err(|_| anyhow::anyhow!("paged cache runtime lock poisoned"))
            .map(|mut runtime_guard| {
                let running_count = running.iter().filter(|entry| entry.is_some()).count();
                let prefilled_count = prefilled.len();
                if result.is_err() && (running_count > 0 || prefilled_count > 0) {
                    if let Err(error) = &result {
                        tracing::warn!(
                            target: "vasr_transcribe::pipeline",
                            running = running_count,
                            prefilled = prefilled_count,
                            error = %format!("{error:#}"),
                            "cleaning paged KV after async decode error"
                        );
                    }
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

#[allow(clippy::too_many_arguments)]
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
    let config = AsrBatchSchedulerConfig::from_decode_max_seqs(decode_max_seqs)
        .with_max_new_tokens(max_new_tokens);
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
#[allow(clippy::too_many_arguments)]
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
    let config = AsrBatchSchedulerConfig::from_decode_max_seqs(decode_max_seqs)
        .with_max_new_tokens(max_new_tokens);
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
        AsrBatchSchedulerConfig, PrefillBatchReason, PrefillInputBuffer, PreparedAudioBatch,
        RunningSlot, collect_prefill_wave_by_kv, continuous_batch_enabled,
        dynamic_source_poll_may_block, filter_sequences_at_token_limit,
        group_running_by_decode_step, pad_prepared_wave_for_prefill,
    };
    use crate::inference::execution::ExecutionPolicy;
    use std::collections::VecDeque;

    use candle_core::{Device, Tensor};

    use crate::processor::asr_processor::PreparedInputs;

    #[test]
    fn streaming_prefill_does_not_retain_historical_output_rows() {
        use super::{PrefillStageOutput, apply_prefill_stage_message};
        for retain in [false, true] {
            let rows = if retain { 5 } else { 1_000_000 };
            let message = PrefillStageOutput {
                sequences: Vec::new(),
                total_rows: rows,
                source_closed: true,
                prefill_wave: 0,
                reason: PrefillBatchReason::WaitingEmpty,
                #[cfg(feature = "timing")]
                timings: super::GenerationTimings::default(),
            };
            let mut outputs = Vec::new();
            let mut prefilled = VecDeque::new();
            let mut closed = false;
            #[cfg(feature = "timing")]
            let mut timings = None;
            apply_prefill_stage_message(
                message,
                &mut outputs,
                &mut prefilled,
                &mut closed,
                retain,
                #[cfg(feature = "timing")]
                &mut timings,
            );
            assert_eq!(outputs.len(), if retain { rows } else { 0 });
            assert!(closed);
        }
    }

    #[test]
    fn continuous_batch_defaults_on() {
        if std::env::var_os("VASR_DISABLE_CONTINUOUS_BATCH").is_none() {
            assert!(continuous_batch_enabled());
            assert!(AsrBatchSchedulerConfig::default().continuous_batch);
        }
    }

    #[test]
    fn continuous_scheduler_uses_standard_default_token_limit() {
        assert_eq!(AsrBatchSchedulerConfig::default().max_new_tokens, 256);
        assert_eq!(
            AsrBatchSchedulerConfig::default()
                .with_max_new_tokens(0)
                .max_new_tokens,
            256
        );
        assert_eq!(
            AsrBatchSchedulerConfig::default()
                .with_max_new_tokens(17)
                .max_new_tokens,
            17
        );
    }

    #[test]
    fn continuous_decode_finishes_exactly_at_token_limit() {
        let slot = |generated: usize| super::PagedDecodeSlot {
            request_id: 1,
            prompt_len_usize: 4,
            prompt_len_i64: 4,
            block_table: vec![],
            next_id: 42,
            generated: vec![7; generated],
            finished: false,
            decode_step: generated,
            token_capacity: 1024,
            awaiting_kv: false,
        };
        let mut running = vec![
            Some(RunningSlot {
                client_index: 0,
                slot: slot(3),
            }),
            Some(RunningSlot {
                client_index: 1,
                slot: slot(4),
            }),
        ];
        running[1].as_mut().unwrap().slot.awaiting_kv = true;
        let active = vec![(0, 3), (1, 4)];

        let remaining = filter_sequences_at_token_limit(&mut running, &active, 4);

        assert_eq!(remaining, vec![(0, 3)]);
        assert!(!running[0].as_ref().unwrap().slot.finished);
        assert!(running[1].as_ref().unwrap().slot.finished);
        assert!(!running[1].as_ref().unwrap().slot.awaiting_kv);
        assert_eq!(running[1].as_ref().unwrap().client_index, 1);
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
    fn metal_policy_enables_mixed_step_and_uses_validated_slots() {
        // 自动模式下 Metal 不再退回“1 槽 / 1 行 / mixed-step 关闭”。
        let policy = ExecutionPolicy::resolve_backend(true, 8, false);
        assert!(policy.mixed_step);
        assert_eq!(policy.active_slots, 8);
        assert_eq!(policy.decode_rows, policy.active_slots);
        assert!(policy.prefill_wave <= policy.active_slots);
    }

    #[test]
    fn policy_decode_rows_never_exceeds_active_slots() {
        let policy = ExecutionPolicy::resolve_backend(true, 4, true);
        assert_eq!(policy.active_slots, 4);
        assert!(policy.decode_rows <= policy.active_slots);
        assert!(policy.prefill_wave <= policy.active_slots);
    }

    #[test]
    fn non_metal_policy_keeps_capacity_as_execution_width() {
        let policy = ExecutionPolicy::resolve_backend(false, 8, false);
        assert_eq!(policy.active_slots, 8);
        assert_eq!(policy.decode_rows, 8);
        assert_eq!(policy.prefill_wave, 8);
    }

    #[test]
    fn collect_prefill_wave_uses_free_kv_not_queue_head() {
        let mut waiting = VecDeque::from(vec![0usize, 1, 2]);
        let capacities = [128usize, 32, 64];
        let (wave, reason) =
            collect_prefill_wave_by_kv(&mut waiting, 32, 3, 3, |idx| Ok(capacities[idx]))
                .expect("collect wave");
        assert_eq!(wave, vec![1, 2]);
        assert!(matches!(reason, PrefillBatchReason::KvBlocks));
        assert_eq!(waiting.into_iter().collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn generated_tokens_follow_duration_without_minimum() {
        // ceil(秒 × 4.25 × 倍率)，没有 128 下限。
        assert_eq!(
            super::super::speech_rate::predicted_generated_tokens(0.1, 4.25, 1.0),
            1
        );
        assert_eq!(
            super::super::speech_rate::predicted_generated_tokens(5.0, 4.25, 1.0),
            22
        );
        assert_eq!(
            super::super::speech_rate::predicted_generated_tokens(30.0, 4.25, 1.0),
            128
        );
    }

    #[test]
    fn two_slots_partition_balances_the_example() {
        let (side_a, side_b) = super::partition_two_slots(&[8, 7, 6, 5, 4]);
        let sum = |items: &[usize]| {
            items
                .iter()
                .map(|&index| [8, 7, 6, 5, 4][index])
                .sum::<usize>()
        };
        assert_eq!(sum(&side_a), 15);
        assert_eq!(sum(&side_b), 15);
    }

    #[test]
    fn dynamic_source_poll_blocks_only_when_decoder_is_idle() {
        assert!(!dynamic_source_poll_may_block(true, 0));
        assert!(!dynamic_source_poll_may_block(false, 1));
        assert!(!dynamic_source_poll_may_block(true, 1));
        assert!(dynamic_source_poll_may_block(false, 0));
    }

    #[test]
    fn prefill_input_buffer_compacts_completed_prefix() -> anyhow::Result<()> {
        let prepared = (0..3)
            .map(|_| PreparedInputs {
                input_ids: vec![10, 99, 99],
                attention_mask: vec![1, 1, 1],
                input_features: vec![],
                feature_attention_mask: vec![],
            })
            .collect();
        let audio_features = Tensor::new(
            &[
                [1f32, 10.],
                [2., 20.],
                [3., 30.],
                [4., 40.],
                [5., 50.],
                [6., 60.],
            ],
            &Device::Cpu,
        )?;
        let mut inputs = PrefillInputBuffer::default();
        inputs.append(
            99,
            PreparedAudioBatch {
                prepared,
                audio_features,
                audio_seconds: vec![1.0, 2.0, 3.0],
                speech_rate_ratios: vec![1.0, 1.0, 1.0],
            },
        )?;
        assert_eq!(inputs.waiting.pop_front(), Some(0));
        inputs.compact_completed_prefix()?;

        assert_eq!(inputs.base_index, 1);
        assert_eq!(inputs.len(), 3);
        assert_eq!(
            inputs.waiting.iter().copied().collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(inputs.audio_offsets, vec![0, 2]);
        assert_eq!(inputs.row_audio_lens, vec![2, 2]);
        assert_eq!(
            inputs
                .audio_features
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing compacted features"))?
                .to_vec2::<f32>()?,
            vec![vec![3., 30.], vec![4., 40.], vec![5., 50.], vec![6., 60.]]
        );
        Ok(())
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
