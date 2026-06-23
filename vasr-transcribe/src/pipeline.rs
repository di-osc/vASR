//! Async parallel transcribe: loader, VAD, and ASR stages overlap via channels.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use anyhow::{Result, bail};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::EncodedAsrBatchPoll;
#[cfg(feature = "timing")]
use vasr::models::qwen3asr::TranscribeTimings;
use vasr::models::qwen3asr::{AudioFeatureEncodeTimings, EncodedAsrBatch, Qwen3ASR};
use vasr_data::{Annotation, AudioSource, DurationMs, Timeline, Waveform};
use vasr_data::{AudioLoadOptions, AudioLoader};

use crate::runtime::pipeline::{OfflinePipeline, VadPreparation, VadPrepared, offset_annotations};
use crate::runtime::{AsrOptions, ParallelTranscribeOptions, VadOptions};
#[cfg(feature = "paged-attn")]
use vasr::qwen_outputs_to_timelines;

const FALLBACK_STAGE_PARALLELISM: usize = 20;
const DEFAULT_LOADED_QUEUE_MEMORY_FRACTION: f64 = 0.20;
const BYTES_PER_MEMORY_PERMIT: usize = 1024 * 1024;
const MIN_LOADED_QUEUE_MEMORY_PERMITS: usize = 1;
const ASR_MICROBATCH_COLLECT_WAIT_MS: u64 = 10;
const GLOBAL_DECODE_IDLE_FLUSH_MS: u64 = 10;

#[derive(Debug, Default)]
struct LoaderStageStats {
    jobs: usize,
    failed: usize,
    audio_seconds: f64,
    compute_seconds: f64,
    wall_seconds: f64,
}

#[derive(Debug, Default)]
struct LoaderTaskStats {
    jobs: usize,
    failed: usize,
    audio_seconds: f64,
    compute_seconds: f64,
}

#[derive(Debug, Default)]
struct VadStageStats {
    jobs: usize,
    failed: usize,
    no_speech: usize,
    segments: usize,
    audio_seconds: f64,
    speech_seconds: f64,
    compute_seconds: f64,
    send_wait_seconds: f64,
    wall_seconds: f64,
}

#[derive(Debug, Default)]
struct VadTaskStats {
    jobs: usize,
    failed: usize,
    no_speech: usize,
    segments: usize,
    audio_seconds: f64,
    speech_seconds: f64,
    compute_seconds: f64,
    send_wait_seconds: f64,
    wall_seconds: f64,
}

#[derive(Debug, Default)]
struct AsrStageStats {
    batches: usize,
    jobs: usize,
    segments: usize,
    audio_seconds: f64,
    wall_seconds: f64,
    encode_batches: usize,
    encode_segments: usize,
    encode_audio_seconds: f64,
    encode_seconds: f64,
    decode_batches: usize,
    decode_segments: usize,
    decode_audio_seconds: f64,
    decode_seconds: f64,
    audio_encoder_seconds: f64,
    prefill_seconds: f64,
    prefill_tokens: usize,
    decode_model_seconds: f64,
    decode_tokens: usize,
    decode_steps: usize,
}
#[derive(Clone)]
pub struct AsyncTranscribePipeline {
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    offline: Arc<OfflinePipeline>,
    parallel_options: ParallelTranscribeOptions,
}

#[derive(Debug, Clone)]
pub struct TranscribeInput {
    pub index: usize,
    pub source: AudioSource,
}

#[derive(Debug)]
pub struct TranscribeItemOutcome {
    pub index: usize,
    pub result: Result<Timeline>,
    pub audio_seconds: f64,
    pub bad_component: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PipelineRequestId(u64);

#[derive(Debug, Clone, Copy)]
struct RequestItemRef {
    request_id: PipelineRequestId,
    item_index: usize,
}

#[derive(Debug)]
struct RoutedTranscribeItemOutcome {
    request_item: Option<RequestItemRef>,
    outcome: TranscribeItemOutcome,
}

impl From<TranscribeItemOutcome> for RoutedTranscribeItemOutcome {
    fn from(outcome: TranscribeItemOutcome) -> Self {
        Self {
            request_item: None,
            outcome,
        }
    }
}

impl RoutedTranscribeItemOutcome {
    fn into_public_outcome(self) -> TranscribeItemOutcome {
        self.outcome
    }

    fn for_request(request_item: RequestItemRef, mut outcome: TranscribeItemOutcome) -> Self {
        outcome.index = request_item.item_index;
        Self {
            request_item: Some(request_item),
            outcome,
        }
    }
}

fn route_outcome(
    request_item: Option<RequestItemRef>,
    outcome: TranscribeItemOutcome,
) -> RoutedTranscribeItemOutcome {
    match request_item {
        Some(request_item) => RoutedTranscribeItemOutcome::for_request(request_item, outcome),
        None => outcome.into(),
    }
}

impl AsyncTranscribePipeline {
    pub fn new(
        loader: AudioLoader,
        offline: Arc<OfflinePipeline>,
        asr_options: AsrOptions,
    ) -> Self {
        Self {
            loader,
            load_options: AudioLoadOptions::default(),
            offline,
            parallel_options: ParallelTranscribeOptions::from(asr_options),
        }
    }

    pub fn with_vad_options(mut self, vad_options: VadOptions) -> Self {
        self.parallel_options.vad_options = vad_options;
        self
    }

    pub async fn run_source(&self, index: usize, source: AudioSource) -> TranscribeItemOutcome {
        let outcomes = self.run(vec![TranscribeInput { index, source }]).await;
        outcomes
            .into_iter()
            .next()
            .unwrap_or(TranscribeItemOutcome {
                index,
                result: Err(anyhow::anyhow!("run returned no outcome")),
                audio_seconds: 0.0,
                bad_component: Some("recognizer"),
            })
    }

    pub async fn run(&self, inputs: Vec<TranscribeInput>) -> Vec<TranscribeItemOutcome> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let pipeline_start = Instant::now();
        let job_count = inputs.len();
        let DeduplicatedTranscribeInputs {
            unique_inputs,
            duplicate_indices,
        } = deduplicate_transcribe_inputs(inputs);
        let unique_count = unique_inputs.len();

        let stage_parallelism = default_stage_parallelism();
        let loaded_queue_memory = loaded_queue_memory_budget();
        let vad_to_asr_queue_memory = loaded_queue_memory_budget();
        tracing::info!(
            target: "vasr_transcribe::pipeline",
            "pipeline config | loader_workers={} vad_workers={} loaded_queue_mem={:.1}GiB segment_queue_mem={:.1}GiB encoder_request_queue=unbounded feature_queue=unbounded",
            stage_parallelism,
            stage_parallelism,
            permits_to_gib(loaded_queue_memory.capacity_permits),
            permits_to_gib(vad_to_asr_queue_memory.capacity_permits)
        );
        tracing::debug!(
            target: "vasr_transcribe::pipeline",
            "pipeline queues | loaded_queue_memory={}MiB vad_to_asr_queue_memory={}MiB queue_memory_source={} permit_size={}MiB",
            loaded_queue_memory.capacity_permits,
            vad_to_asr_queue_memory.capacity_permits,
            loaded_queue_memory.source,
            BYTES_PER_MEMORY_PERMIT / (1024 * 1024)
        );
        let (loader_to_vad_tx, loader_to_vad_rx) = mpsc::unbounded_channel();
        let (vad_to_asr_tx, vad_to_asr_rx) = mpsc::unbounded_channel();
        let (asr_to_result_tx, mut asr_to_result_rx) = mpsc::channel(unique_inputs.len());
        let loader_stats = Arc::new(Mutex::new(LoaderStageStats::default()));
        let vad_stats = Arc::new(Mutex::new(VadStageStats::default()));
        let asr_stats = Arc::new(Mutex::new(AsrStageStats::default()));

        let loader = self.loader.clone();
        let load_options = self.load_options.clone();
        let loader_handle = spawn_loader_worker(
            unique_inputs,
            loader,
            load_options,
            stage_parallelism,
            loaded_queue_memory,
            loader_to_vad_tx,
            Arc::clone(&loader_stats),
        );

        let offline = Arc::clone(&self.offline);
        let vad_options = self.parallel_options.vad_options.clone();
        let vad_handle = spawn_vad_worker(
            offline,
            vad_options,
            stage_parallelism,
            loader_to_vad_rx,
            vad_to_asr_tx,
            vad_to_asr_queue_memory,
            Arc::clone(&vad_stats),
        );

        let offline = Arc::clone(&self.offline);
        let asr_options = self.parallel_options.asr_options.clone();
        let max_batch_audio_sec = asr_options.max_batch_audio_sec;
        let asr_handle = spawn_asr_worker(
            offline,
            asr_options,
            max_batch_audio_sec,
            vad_to_asr_rx,
            asr_to_result_tx,
            Arc::clone(&asr_stats),
            None,
        );

        let mut outcomes = Vec::with_capacity(job_count);
        while let Some(outcome) = asr_to_result_rx.recv().await {
            outcomes.push(outcome.into_public_outcome());
            if outcomes.len() >= unique_count {
                break;
            }
        }

        loader_handle.abort();
        vad_handle.abort();
        asr_handle.abort();
        let _ = loader_handle.await;
        let _ = vad_handle.await;
        let _ = asr_handle.await;

        if let Ok(stats) = asr_stats.lock() {
            log_asr_summary(&stats);
        }
        let outcomes = expand_duplicate_outcomes(outcomes, &duplicate_indices);
        log_pipeline(job_count, &outcomes, pipeline_start.elapsed().as_secs_f64());
        outcomes
    }
}

#[derive(Clone)]
pub struct GlobalAsyncTranscribePipeline {
    submit_tx: mpsc::UnboundedSender<GlobalTranscribeRequest>,
}

struct GlobalTranscribeRequest {
    inputs: Vec<TranscribeInput>,
    response_tx: oneshot::Sender<Vec<TranscribeItemOutcome>>,
}

#[derive(Debug)]
struct GlobalLoadJob {
    internal_index: usize,
    request_item: RequestItemRef,
    source: AudioSource,
}

struct PendingGlobalRequest {
    remaining: usize,
    duplicate_indices: Vec<(usize, usize)>,
    internal_indices: Vec<usize>,
    outcomes: Vec<TranscribeItemOutcome>,
    response_tx: oneshot::Sender<Vec<TranscribeItemOutcome>>,
}

impl GlobalAsyncTranscribePipeline {
    pub fn start(
        loader: AudioLoader,
        offline: Arc<OfflinePipeline>,
        asr_options: AsrOptions,
        vad_options: VadOptions,
    ) -> Self {
        let parallel_options = ParallelTranscribeOptions {
            asr_options,
            vad_options,
        };
        let load_options = AudioLoadOptions::default();
        let stage_parallelism = default_stage_parallelism();
        let loaded_queue_memory = loaded_queue_memory_budget();
        let vad_to_asr_queue_memory = loaded_queue_memory_budget();
        tracing::info!(
            target: "vasr_transcribe::pipeline",
            "global pipeline config | mode=serve loader_workers={} vad_workers={} loaded_queue_mem={:.1}GiB segment_queue_mem={:.1}GiB encoder_request_queue=unbounded feature_queue=unbounded decode_idle_flush={}ms",
            stage_parallelism,
            stage_parallelism,
            permits_to_gib(loaded_queue_memory.capacity_permits),
            permits_to_gib(vad_to_asr_queue_memory.capacity_permits),
            GLOBAL_DECODE_IDLE_FLUSH_MS
        );

        let (submit_tx, submit_rx) = mpsc::unbounded_channel();
        let (load_job_tx, load_job_rx) = mpsc::unbounded_channel();
        let (loader_to_vad_tx, loader_to_vad_rx) = mpsc::unbounded_channel();
        let (vad_to_asr_tx, vad_to_asr_rx) = mpsc::unbounded_channel();
        let (asr_to_result_tx, asr_to_result_rx) = mpsc::channel(1024);
        let pending_requests = Arc::new(Mutex::new(HashMap::new()));
        let job_routes = Arc::new(Mutex::new(HashMap::new()));
        let next_request_id = Arc::new(AtomicU64::new(1));
        let next_job_id = Arc::new(AtomicU64::new(1));
        let loader_stats = Arc::new(Mutex::new(LoaderStageStats::default()));
        let vad_stats = Arc::new(Mutex::new(VadStageStats::default()));
        let asr_stats = Arc::new(Mutex::new(AsrStageStats::default()));

        spawn_global_submit_worker(
            submit_rx,
            load_job_tx,
            Arc::clone(&pending_requests),
            Arc::clone(&job_routes),
            Arc::clone(&next_request_id),
            Arc::clone(&next_job_id),
        );
        spawn_global_loader_worker(
            loader,
            load_options,
            stage_parallelism,
            loaded_queue_memory,
            load_job_rx,
            loader_to_vad_tx,
            loader_stats,
        );
        spawn_vad_worker(
            Arc::clone(&offline),
            parallel_options.vad_options.clone(),
            stage_parallelism,
            loader_to_vad_rx,
            vad_to_asr_tx,
            vad_to_asr_queue_memory,
            vad_stats,
        );
        spawn_asr_worker(
            offline,
            parallel_options.asr_options.clone(),
            parallel_options.asr_options.max_batch_audio_sec,
            vad_to_asr_rx,
            asr_to_result_tx,
            asr_stats,
            Some(std::time::Duration::from_millis(
                GLOBAL_DECODE_IDLE_FLUSH_MS,
            )),
        );
        spawn_global_result_router(asr_to_result_rx, pending_requests, job_routes);

        Self { submit_tx }
    }

    pub async fn run(&self, inputs: Vec<TranscribeInput>) -> Vec<TranscribeItemOutcome> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let (response_tx, response_rx) = oneshot::channel();
        if self
            .submit_tx
            .send(GlobalTranscribeRequest {
                inputs,
                response_tx,
            })
            .is_err()
        {
            return vec![TranscribeItemOutcome {
                index: 0,
                result: Err(anyhow::anyhow!("global transcribe pipeline stopped")),
                audio_seconds: 0.0,
                bad_component: Some("recognizer"),
            }];
        }
        response_rx.await.unwrap_or_else(|_| {
            vec![TranscribeItemOutcome {
                index: 0,
                result: Err(anyhow::anyhow!(
                    "global transcribe pipeline response dropped"
                )),
                audio_seconds: 0.0,
                bad_component: Some("recognizer"),
            }]
        })
    }
}

fn spawn_global_submit_worker(
    mut submit_rx: mpsc::UnboundedReceiver<GlobalTranscribeRequest>,
    load_job_tx: mpsc::UnboundedSender<GlobalLoadJob>,
    pending_requests: Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: Arc<Mutex<HashMap<usize, RequestItemRef>>>,
    next_request_id: Arc<AtomicU64>,
    next_job_id: Arc<AtomicU64>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(request) = submit_rx.recv().await {
            let request_id = PipelineRequestId(next_request_id.fetch_add(1, Ordering::Relaxed));
            let DeduplicatedTranscribeInputs {
                unique_inputs,
                duplicate_indices,
            } = deduplicate_transcribe_inputs(request.inputs);
            let unique_count = unique_inputs.len();
            if unique_count == 0 {
                let _ = request.response_tx.send(Vec::new());
                continue;
            }
            let mut load_jobs = Vec::with_capacity(unique_count);
            let mut internal_indices = Vec::with_capacity(unique_count);
            for input in unique_inputs {
                let internal_index = next_job_id.fetch_add(1, Ordering::Relaxed) as usize;
                let request_item = RequestItemRef {
                    request_id,
                    item_index: input.index,
                };
                internal_indices.push(internal_index);
                load_jobs.push(GlobalLoadJob {
                    internal_index,
                    request_item,
                    source: input.source,
                });
            }
            if let Ok(mut routes) = job_routes.lock() {
                for job in &load_jobs {
                    routes.insert(job.internal_index, job.request_item);
                }
            }
            if let Ok(mut pending) = pending_requests.lock() {
                pending.insert(
                    request_id,
                    PendingGlobalRequest {
                        remaining: unique_count,
                        duplicate_indices,
                        internal_indices,
                        outcomes: Vec::with_capacity(unique_count),
                        response_tx: request.response_tx,
                    },
                );
            }
            for job in load_jobs {
                if load_job_tx.send(job).is_err() {
                    break;
                }
            }
        }
    })
}

fn spawn_global_loader_worker(
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    max_in_flight: usize,
    loaded_queue_memory: LoadedQueueMemoryBudget,
    mut load_job_rx: mpsc::UnboundedReceiver<GlobalLoadJob>,
    loader_to_vad_tx: mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
    stats: Arc<Mutex<LoaderStageStats>>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        loop {
            tokio::select! {
                job = load_job_rx.recv(), if tasks.len() < max_in_flight => {
                    match job {
                        Some(job) => spawn_global_loader_task(&mut tasks, loader.clone(), load_options.clone(), job),
                        None => break,
                    }
                }
                result = tasks.join_next(), if !tasks.is_empty() => {
                    if let Some(result) = result {
                        send_global_loaded_result(result, &loaded_queue_memory, &loader_to_vad_tx, &stats).await?;
                    }
                }
            }
        }
        while let Some(result) = tasks.join_next().await {
            send_global_loaded_result(result, &loaded_queue_memory, &loader_to_vad_tx, &stats)
                .await?;
        }
        Ok(())
    })
}

fn spawn_global_loader_task(
    tasks: &mut JoinSet<(Result<LoadedJob, TranscribeItemOutcome>, LoaderTaskStats)>,
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    job: GlobalLoadJob,
) {
    tasks.spawn(async move {
        let start = Instant::now();
        let loaded = tokio::task::spawn_blocking({
            let source = job.source.clone();
            move || loader.load(&source, &load_options)
        })
        .await;
        let compute_seconds = start.elapsed().as_secs_f64();
        let mut stats = LoaderTaskStats {
            compute_seconds,
            ..Default::default()
        };
        let message = match loaded {
            Ok(Ok(waveform)) => {
                stats.jobs = 1;
                stats.audio_seconds = waveform.duration_seconds();
                Ok(LoadedJob {
                    index: job.internal_index,
                    request_item: Some(job.request_item),
                    waveform,
                    _queue_memory_permit: None,
                })
            }
            Ok(Err(error)) => {
                stats.failed = 1;
                Err(TranscribeItemOutcome {
                    index: job.internal_index,
                    result: Err(error),
                    audio_seconds: 0.0,
                    bad_component: Some("loader"),
                })
            }
            Err(error) => {
                stats.failed = 1;
                Err(TranscribeItemOutcome {
                    index: job.internal_index,
                    result: Err(anyhow::anyhow!("loader join error: {error}")),
                    audio_seconds: 0.0,
                    bad_component: Some("loader"),
                })
            }
        };
        (message, stats)
    });
}

async fn send_global_loaded_result(
    result: std::result::Result<
        (Result<LoadedJob, TranscribeItemOutcome>, LoaderTaskStats),
        tokio::task::JoinError,
    >,
    loaded_queue_memory: &LoadedQueueMemoryBudget,
    loader_to_vad_tx: &mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
    stats: &Arc<Mutex<LoaderStageStats>>,
) -> Result<()> {
    let (mut message, task_stats) = result.unwrap_or_else(|error| {
        (
            Err(TranscribeItemOutcome {
                index: 0,
                result: Err(anyhow::anyhow!("loader task join error: {error}")),
                audio_seconds: 0.0,
                bad_component: Some("loader"),
            }),
            LoaderTaskStats {
                failed: 1,
                ..Default::default()
            },
        )
    });
    record_loader_task_stats(stats, &task_stats);
    if let Ok(job) = &mut message {
        job._queue_memory_permit = Some(acquire_loaded_job_memory(loaded_queue_memory, job).await?);
    }
    loader_to_vad_tx
        .send(message)
        .map_err(|_| anyhow::anyhow!("loaded channel closed"))
}

fn spawn_global_result_router(
    mut asr_to_result_rx: mpsc::Receiver<RoutedTranscribeItemOutcome>,
    pending_requests: Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: Arc<Mutex<HashMap<usize, RequestItemRef>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(routed) = asr_to_result_rx.recv().await {
            let request_item = match routed.request_item {
                Some(request_item) => request_item,
                None => {
                    let Ok(routes) = job_routes.lock() else {
                        continue;
                    };
                    let Some(request_item) = routes.get(&routed.outcome.index).copied() else {
                        continue;
                    };
                    request_item
                }
            };
            let routed = route_outcome(Some(request_item), routed.outcome);
            let Some(request_item) = routed.request_item else {
                continue;
            };
            let mut completed = None;
            if let Ok(mut pending) = pending_requests.lock() {
                if let Some(request) = pending.get_mut(&request_item.request_id) {
                    request.outcomes.push(routed.outcome);
                    request.remaining = request.remaining.saturating_sub(1);
                    if request.remaining == 0 {
                        completed = pending.remove(&request_item.request_id);
                    }
                }
            }
            if let Some(request) = completed {
                if let Ok(mut routes) = job_routes.lock() {
                    for internal_index in &request.internal_indices {
                        routes.remove(internal_index);
                    }
                }
                let outcomes =
                    expand_duplicate_outcomes(request.outcomes, &request.duplicate_indices);
                let _ = request.response_tx.send(outcomes);
            }
        }
    })
}

fn default_stage_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(FALLBACK_STAGE_PARALLELISM)
}

fn permits_to_gib(permits: usize) -> f64 {
    permits as f64 * BYTES_PER_MEMORY_PERMIT as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[derive(Clone)]
struct LoadedQueueMemoryBudget {
    semaphore: Arc<Semaphore>,
    capacity_permits: usize,
    source: &'static str,
}

fn loaded_queue_memory_budget() -> LoadedQueueMemoryBudget {
    let LoadedQueueMemoryBudgetConfig { permits, source } = loaded_queue_memory_budget_permits(
        std::env::var("VASR_LOADED_QUEUE_MEMORY_MB").ok().as_deref(),
        std::env::var("VASR_LOADED_QUEUE_MEMORY_FRACTION")
            .ok()
            .as_deref(),
        total_memory_bytes(),
    );
    LoadedQueueMemoryBudget {
        semaphore: Arc::new(Semaphore::new(permits)),
        capacity_permits: permits,
        source,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LoadedQueueMemoryBudgetConfig {
    permits: usize,
    source: &'static str,
}

fn loaded_queue_memory_budget_permits(
    memory_mb: Option<&str>,
    fraction: Option<&str>,
    total_memory_bytes: Option<u64>,
) -> LoadedQueueMemoryBudgetConfig {
    if let Some(value) = memory_mb {
        match value.parse::<usize>() {
            Ok(mb) if mb > 0 => {
                return LoadedQueueMemoryBudgetConfig {
                    permits: mb,
                    source: "env:VASR_LOADED_QUEUE_MEMORY_MB",
                };
            }
            _ => tracing::warn!(
                target: "vasr_transcribe::pipeline",
                "ignoring invalid VASR_LOADED_QUEUE_MEMORY_MB={value:?}; expected a positive integer"
            ),
        }
    }

    let fraction_from_env = fraction.is_some();
    let memory_fraction = match fraction {
        Some(value) => match value.parse::<f64>() {
            Ok(n) if n.is_finite() && n > 0.0 => n,
            _ => {
                tracing::warn!(
                    target: "vasr_transcribe::pipeline",
                    "ignoring invalid VASR_LOADED_QUEUE_MEMORY_FRACTION={value:?}; expected a positive number"
                );
                DEFAULT_LOADED_QUEUE_MEMORY_FRACTION
            }
        },
        None => DEFAULT_LOADED_QUEUE_MEMORY_FRACTION,
    };

    let permits = total_memory_bytes
        .and_then(|bytes| {
            let permits =
                ((bytes as f64 * memory_fraction) / BYTES_PER_MEMORY_PERMIT as f64).floor();
            if permits.is_finite() && permits >= 1.0 && permits <= usize::MAX as f64 {
                Some(permits as usize)
            } else {
                None
            }
        })
        .filter(|&permits: &usize| permits > 0)
        .unwrap_or_else(default_stage_parallelism)
        .max(MIN_LOADED_QUEUE_MEMORY_PERMITS);
    let source = if total_memory_bytes.is_some() {
        if fraction_from_env {
            "env:VASR_LOADED_QUEUE_MEMORY_FRACTION"
        } else {
            "system-memory-fraction"
        }
    } else {
        "fallback:stage_parallelism"
    };
    LoadedQueueMemoryBudgetConfig { permits, source }
}

fn loaded_job_memory_permits(job: &LoadedJob) -> usize {
    let bytes = job
        .waveform
        .samples
        .len()
        .saturating_mul(std::mem::size_of::<f32>());
    bytes
        .saturating_add(BYTES_PER_MEMORY_PERMIT - 1)
        .saturating_div(BYTES_PER_MEMORY_PERMIT)
        .max(1)
}

fn vad_to_asr_message_memory_permits(
    message: &Result<AsrPipelineMessage, TranscribeItemOutcome>,
) -> usize {
    let bytes = match message {
        Ok(AsrPipelineMessage::Job(job)) => job
            .waveform
            .samples
            .len()
            .saturating_mul(std::mem::size_of::<f32>()),
        Ok(AsrPipelineMessage::Segment(segment)) => segment
            .waveform
            .samples
            .len()
            .saturating_mul(std::mem::size_of::<f32>()),
        Err(_) => 0,
    };
    if bytes == 0 {
        return 0;
    }
    bytes
        .saturating_add(BYTES_PER_MEMORY_PERMIT - 1)
        .saturating_div(BYTES_PER_MEMORY_PERMIT)
        .max(1)
}

async fn acquire_loaded_job_memory(
    budget: &LoadedQueueMemoryBudget,
    job: &LoadedJob,
) -> Result<OwnedSemaphorePermit> {
    let permits = loaded_job_memory_permits(job).min(budget.capacity_permits);
    let permits = u32::try_from(permits)
        .map_err(|_| anyhow::anyhow!("loaded queue memory permit count overflows u32"))?;
    budget
        .semaphore
        .clone()
        .acquire_many_owned(permits)
        .await
        .map_err(|_| anyhow::anyhow!("loaded queue memory limiter closed"))
}

async fn acquire_vad_to_asr_message_memory(
    budget: &LoadedQueueMemoryBudget,
    message: &Result<AsrPipelineMessage, TranscribeItemOutcome>,
) -> Result<Option<OwnedSemaphorePermit>> {
    let permits = vad_to_asr_message_memory_permits(message).min(budget.capacity_permits);
    if permits == 0 {
        return Ok(None);
    }
    let permits = u32::try_from(permits)
        .map_err(|_| anyhow::anyhow!("VAD to ASR queue memory permit count overflows u32"))?;
    let permit = budget
        .semaphore
        .clone()
        .acquire_many_owned(permits)
        .await
        .map_err(|_| anyhow::anyhow!("VAD to ASR queue memory limiter closed"))?;
    Ok(Some(permit))
}

fn total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        total_memory_bytes_from_proc_meminfo("/proc/meminfo").ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn total_memory_bytes_from_proc_meminfo(path: impl AsRef<Path>) -> Result<u64> {
    let text = fs::read_to_string(path)?;
    parse_meminfo_kib(&text, "MemTotal:")
        .and_then(|kib| kib.checked_mul(1024))
        .ok_or_else(|| anyhow::anyhow!("MemTotal not found in /proc/meminfo"))
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(text: &str, key: &str) -> Option<u64> {
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let value = parts.next()?.parse::<u64>().ok()?;
        return Some(value);
    }
    None
}

#[derive(Debug)]
struct DeduplicatedTranscribeInputs {
    unique_inputs: Vec<TranscribeInput>,
    duplicate_indices: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AudioSourceDedupKey {
    Path(PathBuf),
    Url(String),
    Base64(String),
    Bytes(Vec<u8>),
    LocalFileContent { len: u64, hash: u64 },
}

fn deduplicate_transcribe_inputs(inputs: Vec<TranscribeInput>) -> DeduplicatedTranscribeInputs {
    let mut seen = HashMap::with_capacity(inputs.len());
    let mut unique_inputs = Vec::with_capacity(inputs.len());
    let mut duplicate_indices = Vec::new();

    for input in inputs {
        if let Some(key) = audio_source_dedup_key(&input.source) {
            if let Some(&canonical_index) = seen.get(&key) {
                duplicate_indices.push((input.index, canonical_index));
                continue;
            }
            seen.insert(key, input.index);
        }
        unique_inputs.push(input);
    }

    DeduplicatedTranscribeInputs {
        unique_inputs,
        duplicate_indices,
    }
}

fn audio_source_dedup_key(source: &AudioSource) -> Option<AudioSourceDedupKey> {
    match source {
        AudioSource::Path(path) => Some(
            local_file_content_key(path).unwrap_or_else(|| AudioSourceDedupKey::Path(path.clone())),
        ),
        AudioSource::Url(url) => {
            if let Some(path) = local_path_from_url(url) {
                return Some(
                    local_file_content_key(&path)
                        .unwrap_or_else(|| AudioSourceDedupKey::Url(url.clone())),
                );
            }
            Some(AudioSourceDedupKey::Url(url.clone()))
        }
        AudioSource::Base64(encoded) => Some(AudioSourceDedupKey::Base64(encoded.clone())),
        AudioSource::Bytes(bytes) => Some(AudioSourceDedupKey::Bytes(bytes.clone())),
        AudioSource::Waveform(_) => None,
    }
}

fn local_file_content_key(path: &Path) -> Option<AudioSourceDedupKey> {
    let bytes = fs::read(path).ok()?;
    Some(bytes_content_key(&bytes))
}

fn bytes_content_key(bytes: &[u8]) -> AudioSourceDedupKey {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    AudioSourceDedupKey::LocalFileContent {
        len: bytes.len() as u64,
        hash: hasher.finish(),
    }
}

fn local_path_from_url(value: &str) -> Option<PathBuf> {
    value
        .strip_prefix("file://")
        .map(|rest| PathBuf::from(percent_decode_path(rest)))
}

fn percent_decode_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn expand_duplicate_outcomes(
    mut outcomes: Vec<TranscribeItemOutcome>,
    duplicate_indices: &[(usize, usize)],
) -> Vec<TranscribeItemOutcome> {
    if duplicate_indices.is_empty() {
        outcomes.sort_unstable_by_key(|outcome| outcome.index);
        return outcomes;
    }

    let duplicate_outcomes = duplicate_indices
        .iter()
        .filter_map(|&(duplicate_index, canonical_index)| {
            outcomes
                .iter()
                .find(|outcome| outcome.index == canonical_index)
                .map(|outcome| duplicate_outcome_from(outcome, duplicate_index))
        })
        .collect::<Vec<_>>();
    outcomes.extend(duplicate_outcomes);
    outcomes.sort_unstable_by_key(|outcome| outcome.index);
    outcomes
}

fn duplicate_outcome_from(
    outcome: &TranscribeItemOutcome,
    duplicate_index: usize,
) -> TranscribeItemOutcome {
    let result = match &outcome.result {
        Ok(timeline) => Ok(timeline.clone()),
        Err(error) => Err(anyhow::anyhow!("{}", error)),
    };
    TranscribeItemOutcome {
        index: duplicate_index,
        result,
        audio_seconds: outcome.audio_seconds,
        bad_component: outcome.bad_component,
    }
}

#[derive(Debug)]
struct LoadedJob {
    index: usize,
    request_item: Option<RequestItemRef>,
    waveform: Waveform,
    _queue_memory_permit: Option<OwnedSemaphorePermit>,
}

#[derive(Debug)]
struct AsrJob {
    index: usize,
    request_item: Option<RequestItemRef>,
    prepared: VadPreparation,
    waveform: Waveform,
}

/// One VAD speech segment enqueued for ASR continuous micro-batching.
#[derive(Debug, Clone)]
struct AsrSegmentJob {
    job_index: usize,
    request_item: Option<RequestItemRef>,
    job_audio_seconds: f64,
    speech_annotations: Vec<Annotation>,
    segment_index: usize,
    segment_count: usize,
    offset: DurationMs,
    waveform: Waveform,
}

#[derive(Debug)]
enum AsrPipelineMessage {
    Job(AsrJob),
    Segment(AsrSegmentJob),
}

#[derive(Debug)]
struct VadToAsrMessage {
    message: Result<AsrPipelineMessage, TranscribeItemOutcome>,
    _queue_memory_permit: Option<OwnedSemaphorePermit>,
}

#[derive(Debug)]
struct SegmentJobState {
    index: usize,
    request_item: Option<RequestItemRef>,
    audio_seconds: f64,
    speech_annotations: Vec<Annotation>,
    segment_annotations: Vec<Option<Vec<Annotation>>>,
    segment_count: usize,
    segments_done: usize,
}

#[derive(Default)]
struct SegmentAsrState {
    jobs: HashMap<usize, SegmentJobState>,
}

fn speech_job_to_segments(job: AsrJob) -> Result<Vec<AsrSegmentJob>> {
    let VadPreparation::Speech(prepared) = job.prepared else {
        bail!("speech_job_to_segments requires Speech preparation");
    };
    let VadPrepared {
        speech_annotations,
        segments,
        slices,
    } = prepared;
    if segments.len() != slices.len() {
        bail!(
            "VAD prepared mismatch: segments={} slices={}",
            segments.len(),
            slices.len()
        );
    }
    let segment_count = segments.len();
    Ok(segments
        .into_iter()
        .zip(slices)
        .enumerate()
        .map(|(segment_index, (segment, waveform))| AsrSegmentJob {
            job_index: job.index,
            request_item: job.request_item,
            job_audio_seconds: job.waveform.duration_seconds(),
            speech_annotations: if segment_index == 0 {
                speech_annotations.clone()
            } else {
                Vec::new()
            },
            segment_index,
            segment_count,
            offset: segment.range.start,
            waveform,
        })
        .collect())
}

fn spawn_loader_worker(
    inputs: Vec<TranscribeInput>,
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    max_in_flight: usize,
    loaded_queue_memory: LoadedQueueMemoryBudget,
    loader_to_vad_tx: mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
    stats: Arc<Mutex<LoaderStageStats>>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let wall_start = Instant::now();
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        for input in inputs {
            spawn_loader_task(&mut tasks, loader.clone(), load_options.clone(), input);
            if tasks.len() >= max_in_flight {
                send_next_loaded(&mut tasks, &loaded_queue_memory, &loader_to_vad_tx, &stats)
                    .await?;
            }
        }
        while !tasks.is_empty() {
            send_next_loaded(&mut tasks, &loaded_queue_memory, &loader_to_vad_tx, &stats).await?;
        }
        if let Ok(mut stats) = stats.lock() {
            stats.wall_seconds = wall_start.elapsed().as_secs_f64();
            log_loader_summary(&stats);
        }
        Ok(())
    })
}

fn spawn_loader_task(
    tasks: &mut JoinSet<(Result<LoadedJob, TranscribeItemOutcome>, LoaderTaskStats)>,
    loader: AudioLoader,
    load_options: AudioLoadOptions,
    input: TranscribeInput,
) {
    tasks.spawn(async move {
        let start = Instant::now();
        let loaded = tokio::task::spawn_blocking({
            let source = input.source.clone();
            move || loader.load(&source, &load_options)
        })
        .await;

        let compute_seconds = start.elapsed().as_secs_f64();
        let mut stats = LoaderTaskStats {
            compute_seconds,
            ..Default::default()
        };
        let message = match loaded {
            Ok(Ok(waveform)) => {
                stats.jobs = 1;
                stats.audio_seconds = waveform.duration_seconds();
                Ok(LoadedJob {
                    index: input.index,
                    request_item: None,
                    waveform,
                    _queue_memory_permit: None,
                })
            }
            Ok(Err(error)) => {
                stats.failed = 1;
                Err(TranscribeItemOutcome {
                    index: input.index,
                    result: Err(error),
                    audio_seconds: 0.0,
                    bad_component: Some("loader"),
                })
            }
            Err(error) => {
                stats.failed = 1;
                Err(TranscribeItemOutcome {
                    index: input.index,
                    result: Err(anyhow::anyhow!("loader join error: {error}")),
                    audio_seconds: 0.0,
                    bad_component: Some("loader"),
                })
            }
        };
        (message, stats)
    });
}

async fn send_next_loaded(
    tasks: &mut JoinSet<(Result<LoadedJob, TranscribeItemOutcome>, LoaderTaskStats)>,
    loaded_queue_memory: &LoadedQueueMemoryBudget,
    loader_to_vad_tx: &mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
    stats: &Arc<Mutex<LoaderStageStats>>,
) -> Result<()> {
    let (mut message, task_stats) = match tasks.join_next().await {
        Some(Ok(result)) => result,
        Some(Err(error)) => (
            Err(TranscribeItemOutcome {
                index: 0,
                result: Err(anyhow::anyhow!("loader task join error: {error}")),
                audio_seconds: 0.0,
                bad_component: Some("loader"),
            }),
            LoaderTaskStats {
                failed: 1,
                ..Default::default()
            },
        ),
        None => return Ok(()),
    };
    record_loader_task_stats(stats, &task_stats);
    if let Ok(job) = &mut message {
        job._queue_memory_permit = Some(acquire_loaded_job_memory(loaded_queue_memory, job).await?);
    }
    loader_to_vad_tx
        .send(message)
        .map_err(|_| anyhow::anyhow!("loaded channel closed"))
}

fn spawn_vad_worker(
    offline: Arc<OfflinePipeline>,
    vad_options: VadOptions,
    max_in_flight: usize,
    mut loader_to_vad_rx: mpsc::UnboundedReceiver<Result<LoadedJob, TranscribeItemOutcome>>,
    vad_to_asr_tx: mpsc::UnboundedSender<VadToAsrMessage>,
    vad_to_asr_queue_memory: LoadedQueueMemoryBudget,
    stats: Arc<Mutex<VadStageStats>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let wall_start = Instant::now();
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        while let Some(message) = loader_to_vad_rx.recv().await {
            match message {
                Ok(job) => {
                    spawn_vad_task(
                        &mut tasks,
                        Arc::clone(&offline),
                        vad_options.clone(),
                        vad_to_asr_tx.clone(),
                        vad_to_asr_queue_memory.clone(),
                        job,
                    );
                    if tasks.len() >= max_in_flight {
                        let _ = send_next_vad_task(&mut tasks, &stats).await;
                    }
                }
                Err(outcome) => {
                    let mut send_wait_seconds = 0.0;
                    let _ = send_vad_to_asr_timed(
                        &vad_to_asr_tx,
                        &vad_to_asr_queue_memory,
                        Err(outcome),
                        &mut send_wait_seconds,
                    )
                    .await;
                }
            }
        }
        while !tasks.is_empty() {
            let _ = send_next_vad_task(&mut tasks, &stats).await;
        }
        if let Ok(mut stats) = stats.lock() {
            stats.wall_seconds = wall_start.elapsed().as_secs_f64();
            log_vad_summary(&stats);
        }
    })
}

fn spawn_vad_task(
    tasks: &mut JoinSet<Result<VadTaskStats, TranscribeItemOutcome>>,
    offline: Arc<OfflinePipeline>,
    vad_options: VadOptions,
    vad_to_asr_tx: mpsc::UnboundedSender<VadToAsrMessage>,
    vad_to_asr_queue_memory: LoadedQueueMemoryBudget,
    job: LoadedJob,
) {
    tasks.spawn(async move {
        let stage_start = Instant::now();
        let job_index = job.index;
        let request_item = job.request_item;
        let audio_seconds = job.waveform.duration_seconds();
        let prepared = tokio::task::spawn_blocking({
            let offline = Arc::clone(&offline);
            let waveform = job.waveform.clone();
            let vad_options = vad_options.clone();
            move || offline.prepare_vad_stage(&waveform, &vad_options)
        })
        .await;
        let compute_seconds = stage_start.elapsed().as_secs_f64();
        let mut send_wait_seconds = 0.0;

        match prepared {
            Ok(Ok(prepared)) => {
                let (segments, speech_seconds) = vad_preparation_stats(&prepared);
                let no_speech = matches!(prepared, VadPreparation::NoSpeech);
                let asr_job = AsrJob {
                    index: job_index,
                    request_item,
                    prepared,
                    waveform: job.waveform,
                };
                match asr_job.prepared {
                    VadPreparation::Speech(_) => {
                        let segments = match speech_job_to_segments(asr_job) {
                            Ok(segments) => segments,
                            Err(error) => {
                                send_vad_to_asr_timed(
                                    &vad_to_asr_tx,
                                    &vad_to_asr_queue_memory,
                                    Err(TranscribeItemOutcome {
                                        index: job_index,
                                        result: Err(error),
                                        audio_seconds,
                                        bad_component: Some("vad"),
                                    }),
                                    &mut send_wait_seconds,
                                )
                                .await
                                .map_err(|error| {
                                    TranscribeItemOutcome {
                                        index: job_index,
                                        result: Err(error),
                                        audio_seconds,
                                        bad_component: Some("vad"),
                                    }
                                })?;
                                log_vad(
                                    job_index,
                                    audio_seconds,
                                    segments,
                                    speech_seconds,
                                    compute_seconds,
                                    send_wait_seconds,
                                    stage_start.elapsed().as_secs_f64(),
                                );
                                return Ok(VadTaskStats {
                                    jobs: 1,
                                    failed: 1,
                                    segments,
                                    audio_seconds,
                                    speech_seconds,
                                    compute_seconds,
                                    send_wait_seconds,
                                    wall_seconds: stage_start.elapsed().as_secs_f64(),
                                    ..Default::default()
                                });
                            }
                        };
                        for segment in segments {
                            send_vad_to_asr_timed(
                                &vad_to_asr_tx,
                                &vad_to_asr_queue_memory,
                                Ok(AsrPipelineMessage::Segment(segment)),
                                &mut send_wait_seconds,
                            )
                            .await
                            .map_err(|error| {
                                TranscribeItemOutcome {
                                    index: job_index,
                                    result: Err(error),
                                    audio_seconds,
                                    bad_component: Some("asr"),
                                }
                            })?;
                        }
                    }
                    _ => {
                        send_vad_to_asr_timed(
                            &vad_to_asr_tx,
                            &vad_to_asr_queue_memory,
                            Ok(AsrPipelineMessage::Job(asr_job)),
                            &mut send_wait_seconds,
                        )
                        .await
                        .map_err(|error| TranscribeItemOutcome {
                            index: job_index,
                            result: Err(error),
                            audio_seconds,
                            bad_component: Some("asr"),
                        })?;
                    }
                }
                log_vad(
                    job_index,
                    audio_seconds,
                    segments,
                    speech_seconds,
                    compute_seconds,
                    send_wait_seconds,
                    stage_start.elapsed().as_secs_f64(),
                );
                Ok(VadTaskStats {
                    jobs: 1,
                    no_speech: usize::from(no_speech),
                    segments,
                    audio_seconds,
                    speech_seconds,
                    compute_seconds,
                    send_wait_seconds,
                    wall_seconds: stage_start.elapsed().as_secs_f64(),
                    ..Default::default()
                })
            }
            Ok(Err(error)) => {
                send_vad_to_asr_timed(
                    &vad_to_asr_tx,
                    &vad_to_asr_queue_memory,
                    Err(TranscribeItemOutcome {
                        index: job_index,
                        result: Err(error),
                        audio_seconds,
                        bad_component: Some("vad"),
                    }),
                    &mut send_wait_seconds,
                )
                .await
                .map_err(|error| TranscribeItemOutcome {
                    index: job_index,
                    result: Err(error),
                    audio_seconds,
                    bad_component: Some("vad"),
                })?;
                log_vad(
                    job_index,
                    audio_seconds,
                    0,
                    0.0,
                    compute_seconds,
                    send_wait_seconds,
                    stage_start.elapsed().as_secs_f64(),
                );
                Ok(VadTaskStats {
                    jobs: 1,
                    failed: 1,
                    audio_seconds,
                    compute_seconds,
                    send_wait_seconds,
                    wall_seconds: stage_start.elapsed().as_secs_f64(),
                    ..Default::default()
                })
            }
            Err(error) => {
                send_vad_to_asr_timed(
                    &vad_to_asr_tx,
                    &vad_to_asr_queue_memory,
                    Err(TranscribeItemOutcome {
                        index: job_index,
                        result: Err(anyhow::anyhow!("VAD join error: {error}")),
                        audio_seconds,
                        bad_component: Some("vad"),
                    }),
                    &mut send_wait_seconds,
                )
                .await
                .map_err(|error| TranscribeItemOutcome {
                    index: job_index,
                    result: Err(error),
                    audio_seconds,
                    bad_component: Some("vad"),
                })?;
                log_vad(
                    job_index,
                    audio_seconds,
                    0,
                    0.0,
                    compute_seconds,
                    send_wait_seconds,
                    stage_start.elapsed().as_secs_f64(),
                );
                Ok(VadTaskStats {
                    jobs: 1,
                    failed: 1,
                    audio_seconds,
                    compute_seconds,
                    send_wait_seconds,
                    wall_seconds: stage_start.elapsed().as_secs_f64(),
                    ..Default::default()
                })
            }
        }
    });
}

async fn send_vad_to_asr_timed(
    tx: &mpsc::UnboundedSender<VadToAsrMessage>,
    budget: &LoadedQueueMemoryBudget,
    message: Result<AsrPipelineMessage, TranscribeItemOutcome>,
    wait_seconds: &mut f64,
) -> Result<()> {
    let send_start = Instant::now();
    let permit = acquire_vad_to_asr_message_memory(budget, &message).await?;
    let result = tx
        .send(VadToAsrMessage {
            message,
            _queue_memory_permit: permit,
        })
        .map_err(|_| anyhow::anyhow!("ASR stage channel closed"));
    *wait_seconds += send_start.elapsed().as_secs_f64();
    result
}

async fn send_next_vad_task(
    tasks: &mut JoinSet<Result<VadTaskStats, TranscribeItemOutcome>>,
    stats: &Arc<Mutex<VadStageStats>>,
) -> Result<(), TranscribeItemOutcome> {
    match tasks.join_next().await {
        Some(Ok(result)) => {
            if let Ok(task_stats) = &result {
                record_vad_task_stats(stats, task_stats);
            }
            result.map(|_| ())
        }
        Some(Err(error)) => Err(TranscribeItemOutcome {
            index: 0,
            result: Err(anyhow::anyhow!("VAD task join error: {error}")),
            audio_seconds: 0.0,
            bad_component: Some("vad"),
        }),
        None => Ok(()),
    }
}

#[derive(Debug)]
struct SegmentEncodeRequest {
    segment: AsrSegmentJob,
    stage_start: Instant,
}

#[derive(Debug)]
struct SegmentEncodeBatch {
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
}

#[derive(Debug)]
struct SegmentEncodedMessage {
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    batch: Result<SegmentEncodedAsrBatch>,
}

fn spawn_segment_encoder_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    max_batch_audio_sec: f32,
    mut request_rx: mpsc::UnboundedReceiver<SegmentEncodeRequest>,
    encoded_tx: mpsc::UnboundedSender<SegmentEncodedMessage>,
    stats: Arc<Mutex<AsrStageStats>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut pending = VecDeque::new();
        loop {
            let Some(request) = (match pending.pop_front() {
                Some(request) => Some(request),
                None => request_rx.recv().await,
            }) else {
                break;
            };
            let request = collect_encoder_segment_batch(
                request,
                max_batch_audio_sec,
                &mut request_rx,
                &mut pending,
            );
            let encode_start = Instant::now();
            let segments_for_result = request.segments.clone();
            let segment_audio_seconds = segments_for_result
                .iter()
                .map(|segment| segment.waveform.duration_seconds())
                .sum::<f64>();
            let batch = tokio::task::spawn_blocking({
                let offline = Arc::clone(&offline);
                let asr_options = asr_options.clone();
                let segments = request.segments;
                move || run_segment_asr_encode(&offline, &segments, &asr_options)
            })
            .await
            .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR audio encoder join error: {error}")));
            let encode_wall_seconds = encode_start.elapsed().as_secs_f64();
            let encode_timings = batch
                .as_ref()
                .map(|batch| batch.timings)
                .unwrap_or_default();
            log_asr_audio_encoder_batch(
                segments_for_result.len(),
                segment_audio_seconds,
                encode_wall_seconds,
                encode_timings,
            );
            record_asr_encode_stats(
                &stats,
                segments_for_result.len(),
                segment_audio_seconds,
                encode_wall_seconds,
            );
            let message = SegmentEncodedMessage {
                segments: segments_for_result,
                stage_start: request.stage_start,
                batch,
            };
            if encoded_tx.send(message).is_err() {
                break;
            }
        }
    })
}

fn collect_encoder_segment_batch(
    first: SegmentEncodeRequest,
    max_batch_audio_sec: f32,
    request_rx: &mut mpsc::UnboundedReceiver<SegmentEncodeRequest>,
    pending: &mut VecDeque<SegmentEncodeRequest>,
) -> SegmentEncodeBatch {
    let mut audio_sec = first.segment.waveform.duration_seconds() as f32;
    let mut stage_start = first.stage_start;
    let mut segments = vec![first.segment];
    while max_batch_audio_sec <= 0.0 || audio_sec < max_batch_audio_sec {
        match request_rx.try_recv() {
            Ok(request) => {
                let next_audio_sec = request.segment.waveform.duration_seconds() as f32;
                if max_batch_audio_sec > 0.0
                    && audio_sec + next_audio_sec > max_batch_audio_sec + f32::EPSILON
                {
                    pending.push_back(request);
                    break;
                }
                audio_sec += next_audio_sec;
                stage_start = stage_start.min(request.stage_start);
                segments.push(request.segment);
            }
            Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                break;
            }
        }
    }
    segments.sort_by_key(|segment| (segment.job_index, segment.offset.0));
    SegmentEncodeBatch {
        segments,
        stage_start,
    }
}

fn spawn_segment_decoder_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    #[cfg_attr(feature = "paged-attn", allow(unused_mut))] mut encoded_rx: mpsc::UnboundedReceiver<
        SegmentEncodedMessage,
    >,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    #[cfg_attr(not(feature = "paged-attn"), allow(unused_variables))] decode_idle_flush: Option<
        std::time::Duration,
    >,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut segment_state = SegmentAsrState::default();
        #[cfg(feature = "paged-attn")]
        {
            let encoded_rx = Arc::new(tokio::sync::Mutex::new(encoded_rx));
            loop {
                let first = {
                    let mut encoded_rx = encoded_rx.lock().await;
                    encoded_rx.recv().await
                };
                let Some(first) = first else {
                    break;
                };
                let SegmentEncodedMessage {
                    segments,
                    stage_start,
                    batch,
                } = first;
                let initial = match batch {
                    Ok(batch) => batch,
                    Err(error) => {
                        let outcomes = finish_segment_asr_batch(
                            &mut segment_state,
                            &segments,
                            stage_start,
                            Err(error),
                            Some(&stats),
                        );
                        for outcome in outcomes {
                            let _ = asr_to_result_tx.send(outcome.into()).await;
                        }
                        continue;
                    }
                };
                handle_dynamic_segment_encoded_message(
                    segments,
                    stage_start,
                    initial,
                    Arc::clone(&encoded_rx),
                    &offline,
                    &asr_options,
                    &mut segment_state,
                    &asr_to_result_tx,
                    &stats,
                    decode_idle_flush,
                )
                .await;
                if decode_idle_flush.is_none() {
                    break;
                }
            }
        }
        #[cfg(not(feature = "paged-attn"))]
        while let Some(first) = encoded_rx.recv().await {
            handle_ready_segment_encoded_messages(
                first,
                &mut encoded_rx,
                &offline,
                &asr_options,
                &mut segment_state,
                &asr_to_result_tx,
                &stats,
            )
            .await;
        }
    })
}

#[cfg(feature = "paged-attn")]
#[derive(Debug)]
struct SegmentEncodedFailure {
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    error: anyhow::Error,
}

#[cfg(feature = "paged-attn")]
#[derive(Debug)]
struct DynamicSegmentAsrDecode {
    consumed_segments: Vec<AsrSegmentJob>,
    inference: SegmentAsrInference,
    timelines: Vec<Timeline>,
    encoded_failures: Vec<SegmentEncodedFailure>,
    #[cfg(feature = "timing")]
    timings: TranscribeTimings,
}

#[cfg(feature = "paged-attn")]
async fn handle_dynamic_segment_encoded_message(
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    initial: SegmentEncodedAsrBatch,
    encoded_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncodedMessage>>>,
    offline: &OfflinePipeline,
    asr_options: &AsrOptions,
    segment_state: &mut SegmentAsrState,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
    decode_idle_flush: Option<std::time::Duration>,
) {
    let decode_start = Instant::now();
    let batch = tokio::task::spawn_blocking({
        let asr = Arc::clone(&offline.asr);
        let asr_options = asr_options.clone();
        let segments = segments.clone();
        let runtime_handle = tokio::runtime::Handle::current();
        move || {
            run_segment_asr_decode_dynamic(
                asr.as_ref(),
                segments,
                initial,
                encoded_rx,
                runtime_handle,
                &asr_options,
                decode_idle_flush,
            )
        }
    })
    .await
    .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR dynamic decoder join error: {error}")));

    #[cfg(feature = "timing")]
    let (consumed_segments, batch, encoded_failures, timings) = match batch {
        Ok(decoded) => (
            decoded.consumed_segments,
            Ok((decoded.inference, decoded.timelines)),
            decoded.encoded_failures,
            Some(decoded.timings),
        ),
        Err(error) => (segments, Err(error), Vec::new(), None),
    };
    #[cfg(not(feature = "timing"))]
    let (consumed_segments, batch, encoded_failures) = match batch {
        Ok(decoded) => (
            decoded.consumed_segments,
            Ok((decoded.inference, decoded.timelines)),
            decoded.encoded_failures,
        ),
        Err(error) => (segments, Err(error), Vec::new()),
    };
    #[cfg(feature = "timing")]
    if let Some(timings) = timings.as_ref() {
        record_asr_timing_stats(stats, timings);
    }
    record_asr_decode_stats(
        stats,
        consumed_segments.len(),
        consumed_segments
            .iter()
            .map(|segment| segment.waveform.duration_seconds())
            .sum(),
        decode_start.elapsed().as_secs_f64(),
    );
    let outcomes = finish_segment_asr_batch(
        segment_state,
        &consumed_segments,
        stage_start,
        batch,
        Some(stats),
    );
    for outcome in outcomes {
        let _ = asr_to_result_tx.send(outcome.into()).await;
    }
    for failure in encoded_failures {
        let outcomes = finish_segment_asr_batch(
            segment_state,
            failure.segments.as_slice(),
            failure.stage_start,
            Err(failure.error),
            Some(stats),
        );
        for outcome in outcomes {
            let _ = asr_to_result_tx.send(outcome.into()).await;
        }
    }
}

#[cfg(not(feature = "paged-attn"))]
async fn handle_segment_encoded_messages(
    offline: &OfflinePipeline,
    asr_options: &AsrOptions,
    segment_state: &mut SegmentAsrState,
    encoded_messages: Vec<SegmentEncodedMessage>,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) {
    let mut ok_batches = Vec::new();
    for encoded in encoded_messages {
        match encoded.batch {
            Ok(batch) => ok_batches.push((encoded.segments, encoded.stage_start, batch)),
            Err(error) => {
                let outcomes = finish_segment_asr_batch(
                    segment_state,
                    &encoded.segments,
                    encoded.stage_start,
                    Err(error),
                    Some(stats),
                );
                for outcome in outcomes {
                    let _ = asr_to_result_tx.send(outcome.into()).await;
                }
            }
        }
    }
    if ok_batches.is_empty() {
        return;
    }

    let mut segments = Vec::new();
    let mut stage_start = ok_batches[0].1;
    let mut batches = Vec::with_capacity(ok_batches.len());
    for (mut batch_segments, batch_start, batch) in ok_batches {
        if batch_start < stage_start {
            stage_start = batch_start;
        }
        segments.append(&mut batch_segments);
        batches.push(batch);
    }
    let batch = SegmentEncodedAsrBatch::merge(batches);
    handle_merged_segment_encoded_batch(
        offline,
        asr_options,
        segment_state,
        segments,
        stage_start,
        batch,
        asr_to_result_tx,
        stats,
    )
    .await;
}

#[cfg(not(feature = "paged-attn"))]
async fn handle_merged_segment_encoded_batch(
    offline: &OfflinePipeline,
    asr_options: &AsrOptions,
    segment_state: &mut SegmentAsrState,
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    batch: Result<SegmentEncodedAsrBatch>,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) {
    let decode_start = Instant::now();
    #[cfg(feature = "timing")]
    let batch = match batch {
        Ok(batch) => tokio::task::spawn_blocking({
            let asr = Arc::clone(&offline.asr);
            let asr_options = asr_options.clone();
            move || run_segment_asr_decode(asr.as_ref(), batch, &asr_options)
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR decoder join error: {error}"))),
        Err(error) => Err(error),
    };
    #[cfg(feature = "timing")]
    if let Ok((_, _, timings)) = &batch {
        record_asr_timing_stats(stats, timings);
    }
    #[cfg(not(feature = "timing"))]
    let batch = match batch {
        Ok(batch) => tokio::task::spawn_blocking({
            let asr = Arc::clone(&offline.asr);
            let asr_options = asr_options.clone();
            move || run_segment_asr_decode(asr.as_ref(), batch, &asr_options)
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR decoder join error: {error}"))),
        Err(error) => Err(error),
    };
    record_asr_decode_stats(
        stats,
        segments.len(),
        segments
            .iter()
            .map(|segment| segment.waveform.duration_seconds())
            .sum(),
        decode_start.elapsed().as_secs_f64(),
    );
    #[cfg(feature = "timing")]
    let batch = batch.map(|(inference, timelines, _timings)| (inference, timelines));
    let outcomes =
        finish_segment_asr_batch(segment_state, &segments, stage_start, batch, Some(stats));
    for outcome in outcomes {
        let _ = asr_to_result_tx.send(outcome.into()).await;
    }
}

#[cfg(not(feature = "paged-attn"))]
async fn handle_ready_segment_encoded_messages(
    first: SegmentEncodedMessage,
    encoder_to_decoder_rx: &mut mpsc::UnboundedReceiver<SegmentEncodedMessage>,
    offline: &OfflinePipeline,
    asr_options: &AsrOptions,
    segment_state: &mut SegmentAsrState,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) {
    let mut messages = vec![first];
    while let Ok(message) = encoder_to_decoder_rx.try_recv() {
        messages.push(message);
    }
    if messages.len() > 1 {
        let segs: usize = messages.iter().map(|message| message.segments.len()).sum();
        tracing::debug!(
            target: "vasr_transcribe::pipeline",
            "asr audio_features decode merge | batches={} segs={}",
            messages.len(),
            segs,
        );
    }
    handle_segment_encoded_messages(
        offline,
        asr_options,
        segment_state,
        messages,
        asr_to_result_tx,
        stats,
    )
    .await;
}

fn send_segment_encode_request(
    asr_to_encoder_tx: &mpsc::UnboundedSender<SegmentEncodeRequest>,
    request: SegmentEncodeRequest,
) {
    if asr_to_encoder_tx.send(request).is_err() {
        tracing::warn!(
            target: "vasr_transcribe::pipeline",
            "ASR audio feature encoder closed"
        );
    }
}

fn spawn_asr_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    max_batch_audio_sec: f32,
    mut vad_to_asr_rx: mpsc::UnboundedReceiver<VadToAsrMessage>,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    decode_idle_flush: Option<std::time::Duration>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut fallback_segment_state = SegmentAsrState::default();
        let mut pending = VecDeque::new();
        let (asr_to_encoder_tx, asr_to_encoder_rx) = mpsc::unbounded_channel();
        let (encoder_to_decoder_tx, encoder_to_decoder_rx) = mpsc::unbounded_channel();
        let segment_encoder_handle = spawn_segment_encoder_worker(
            Arc::clone(&offline),
            asr_options.clone(),
            max_batch_audio_sec,
            asr_to_encoder_rx,
            encoder_to_decoder_tx,
            Arc::clone(&stats),
        );
        let segment_decoder_handle = spawn_segment_decoder_worker(
            Arc::clone(&offline),
            asr_options.clone(),
            encoder_to_decoder_rx,
            asr_to_result_tx.clone(),
            Arc::clone(&stats),
            decode_idle_flush,
        );

        while let Some(envelope) = match pending.pop_front() {
            Some(envelope) => Some(envelope),
            None => vad_to_asr_rx.recv().await,
        } {
            let message = envelope.message;
            match message {
                Err(outcome) => {
                    let _ = asr_to_result_tx.send(outcome.into()).await;
                }
                Ok(AsrPipelineMessage::Job(job)) => {
                    let messages = collect_job_microbatch(
                        job,
                        max_batch_audio_sec,
                        &mut vad_to_asr_rx,
                        &mut pending,
                    )
                    .await;
                    let mut jobs = Vec::new();
                    for message in messages {
                        match message {
                            Err(outcome) => {
                                let _ = asr_to_result_tx.send(outcome.into()).await;
                            }
                            Ok(AsrPipelineMessage::Job(job)) => jobs.push(job),
                            Ok(AsrPipelineMessage::Segment(_)) => {
                                tracing::warn!(
                                    "ASR worker received segment while collecting job micro-batch"
                                );
                            }
                        }
                    }
                    if jobs.is_empty() {
                        continue;
                    }
                    let outcomes = tokio::task::spawn_blocking({
                        let offline = Arc::clone(&offline);
                        let asr_options = asr_options.clone();
                        let stats = Arc::clone(&stats);
                        move || run_asr_jobs_batch(&offline, jobs, &asr_options, Some(&stats))
                    })
                    .await
                    .unwrap_or_else(|error| {
                        vec![
                            TranscribeItemOutcome {
                                index: 0,
                                result: Err(anyhow::anyhow!("ASR join error: {error}")),
                                audio_seconds: 0.0,
                                bad_component: Some("recognizer"),
                            }
                            .into(),
                        ]
                    });
                    for outcome in outcomes {
                        let _ = asr_to_result_tx.send(outcome).await;
                    }
                }
                Ok(AsrPipelineMessage::Segment(segment)) => {
                    let stage_start = Instant::now();
                    if offline.asr.as_any().is::<Qwen3ASR>() {
                        let request = SegmentEncodeRequest {
                            segment,
                            stage_start,
                        };
                        send_segment_encode_request(&asr_to_encoder_tx, request);
                    } else {
                        let segments = collect_segment_microbatch(
                            segment,
                            max_batch_audio_sec,
                            &mut vad_to_asr_rx,
                            &mut pending,
                        )
                        .await;
                        let batch = tokio::task::spawn_blocking({
                            let offline = Arc::clone(&offline);
                            let asr_options = asr_options.clone();
                            let stats = Arc::clone(&stats);
                            let segments = segments.clone();
                            move || {
                                let stage_start = Instant::now();
                                let result =
                                    run_segment_asr_inference(&offline, &segments, &asr_options);
                                if let Ok((inference, _)) = &result {
                                    record_asr_batch_stats(
                                        &stats,
                                        inference.active_job_count,
                                        segments.len(),
                                        inference.segment_audio_seconds,
                                        stage_start.elapsed().as_secs_f64(),
                                    );
                                }
                                result
                            }
                        })
                        .await
                        .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR join error: {error}")));
                        let outcomes = finish_segment_asr_batch(
                            &mut fallback_segment_state,
                            &segments,
                            stage_start,
                            batch,
                            None,
                        );
                        for outcome in outcomes {
                            let _ = asr_to_result_tx.send(outcome.into()).await;
                        }
                    }
                }
            }
        }

        drop(asr_to_encoder_tx);
        let _ = segment_encoder_handle.await;
        let _ = segment_decoder_handle.await;
    })
}

async fn collect_job_microbatch(
    first: AsrJob,
    max_batch_audio_sec: f32,
    vad_to_asr_rx: &mut mpsc::UnboundedReceiver<VadToAsrMessage>,
    pending: &mut VecDeque<VadToAsrMessage>,
) -> Vec<Result<AsrPipelineMessage, TranscribeItemOutcome>> {
    let mut audio_sec = first.waveform.duration_seconds() as f32;
    let mut messages = vec![Ok(AsrPipelineMessage::Job(first))];
    while max_batch_audio_sec <= 0.0 || audio_sec < max_batch_audio_sec {
        match vad_to_asr_rx.try_recv() {
            Ok(envelope) => match envelope.message {
                Ok(AsrPipelineMessage::Segment(_)) => {
                    pending.push_back(envelope);
                }
                Ok(AsrPipelineMessage::Job(job)) => {
                    let next_audio_sec = job.waveform.duration_seconds() as f32;
                    if max_batch_audio_sec > 0.0
                        && audio_sec + next_audio_sec > max_batch_audio_sec + f32::EPSILON
                    {
                        pending.push_back(VadToAsrMessage {
                            message: Ok(AsrPipelineMessage::Job(job)),
                            _queue_memory_permit: envelope._queue_memory_permit,
                        });
                        break;
                    }
                    audio_sec += next_audio_sec;
                    messages.push(Ok(AsrPipelineMessage::Job(job)));
                }
                message => messages.push(message),
            },
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => return messages,
        }
    }
    while max_batch_audio_sec <= 0.0 || audio_sec < max_batch_audio_sec {
        match tokio::time::timeout(
            std::time::Duration::from_millis(ASR_MICROBATCH_COLLECT_WAIT_MS),
            vad_to_asr_rx.recv(),
        )
        .await
        {
            Ok(Some(envelope)) => match envelope.message {
                Ok(AsrPipelineMessage::Segment(_)) => {
                    pending.push_back(envelope);
                }
                Ok(AsrPipelineMessage::Job(job)) => {
                    let next_audio_sec = job.waveform.duration_seconds() as f32;
                    if max_batch_audio_sec > 0.0
                        && audio_sec + next_audio_sec > max_batch_audio_sec + f32::EPSILON
                    {
                        pending.push_back(VadToAsrMessage {
                            message: Ok(AsrPipelineMessage::Job(job)),
                            _queue_memory_permit: envelope._queue_memory_permit,
                        });
                        break;
                    }
                    audio_sec += next_audio_sec;
                    messages.push(Ok(AsrPipelineMessage::Job(job)));
                }
                message => messages.push(message),
            },
            Ok(None) | Err(tokio::time::error::Elapsed { .. }) => break,
        }
    }
    messages
}

async fn collect_segment_microbatch(
    first: AsrSegmentJob,
    max_batch_audio_sec: f32,
    vad_to_asr_rx: &mut mpsc::UnboundedReceiver<VadToAsrMessage>,
    pending: &mut VecDeque<VadToAsrMessage>,
) -> Vec<AsrSegmentJob> {
    let mut audio_sec = first.waveform.duration_seconds() as f32;
    let mut segments = vec![first];
    // Phase 1: non-blocking drain of everything already in the channel.
    while max_batch_audio_sec <= 0.0 || audio_sec < max_batch_audio_sec {
        match vad_to_asr_rx.try_recv() {
            Ok(envelope) => match envelope.message {
                Ok(AsrPipelineMessage::Segment(segment)) => {
                    let next_audio_sec = segment.waveform.duration_seconds() as f32;
                    if max_batch_audio_sec > 0.0
                        && audio_sec + next_audio_sec > max_batch_audio_sec + f32::EPSILON
                    {
                        pending.push_back(VadToAsrMessage {
                            message: Ok(AsrPipelineMessage::Segment(segment)),
                            _queue_memory_permit: envelope._queue_memory_permit,
                        });
                        break;
                    }
                    audio_sec += next_audio_sec;
                    segments.push(segment);
                }
                message => {
                    // Stash non-segment messages (e.g. Job) so we can
                    // keep collecting segments behind them.
                    pending.push_back(VadToAsrMessage {
                        message,
                        _queue_memory_permit: envelope._queue_memory_permit,
                    });
                }
            },
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => return segments,
        }
    }
    // Phase 2: briefly wait for more segments so the continuous scheduler sees
    // larger admit waves without adding noticeable tail latency.
    while max_batch_audio_sec <= 0.0 || audio_sec < max_batch_audio_sec {
        match tokio::time::timeout(
            std::time::Duration::from_millis(ASR_MICROBATCH_COLLECT_WAIT_MS),
            vad_to_asr_rx.recv(),
        )
        .await
        {
            Ok(Some(envelope)) => match envelope.message {
                Ok(AsrPipelineMessage::Segment(segment)) => {
                    let next_audio_sec = segment.waveform.duration_seconds() as f32;
                    if max_batch_audio_sec > 0.0
                        && audio_sec + next_audio_sec > max_batch_audio_sec + f32::EPSILON
                    {
                        pending.push_back(VadToAsrMessage {
                            message: Ok(AsrPipelineMessage::Segment(segment)),
                            _queue_memory_permit: envelope._queue_memory_permit,
                        });
                        break;
                    }
                    audio_sec += next_audio_sec;
                    segments.push(segment);
                }
                message => {
                    // Same as Phase 1: stash non-segment, keep looking.
                    pending.push_back(VadToAsrMessage {
                        message,
                        _queue_memory_permit: envelope._queue_memory_permit,
                    });
                }
            },
            Ok(None) | Err(tokio::time::error::Elapsed { .. }) => break,
        }
    }
    segments.sort_by_key(|segment| (segment.job_index, segment.offset.0));
    segments
}

fn prepare_segment_job_state(state: &mut SegmentAsrState, segments: &[AsrSegmentJob]) {
    for segment in segments {
        let entry = state
            .jobs
            .entry(segment.job_index)
            .or_insert_with(|| SegmentJobState {
                index: segment.job_index,
                request_item: segment.request_item,
                audio_seconds: segment.job_audio_seconds,
                speech_annotations: segment.speech_annotations.clone(),
                segment_annotations: vec![None; segment.segment_count],
                segment_count: segment.segment_count,
                segments_done: 0,
            });
        if !segment.speech_annotations.is_empty() {
            entry.speech_annotations = segment.speech_annotations.clone();
        }
        if entry.segment_count != segment.segment_count {
            tracing::warn!(
                target: "vasr_transcribe::pipeline",
                "VAD segment count changed for job={} previous={} new={}",
                segment.job_index,
                entry.segment_count,
                segment.segment_count
            );
            entry.segment_count = segment.segment_count;
            entry
                .segment_annotations
                .resize_with(segment.segment_count, || None);
        }
    }
}

fn finish_segment_job(job: SegmentJobState) -> TranscribeItemOutcome {
    let mut timeline = Timeline::new("offline_audio");
    timeline.annotations.extend(job.speech_annotations);
    for annotations in job.segment_annotations.into_iter().flatten() {
        timeline.annotations.extend(annotations);
    }
    TranscribeItemOutcome {
        index: job.index,
        result: Ok(timeline),
        audio_seconds: job.audio_seconds,
        bad_component: None,
    }
}

fn routed_segment_job_outcome(job: SegmentJobState) -> RoutedTranscribeItemOutcome {
    let request_item = job.request_item;
    let outcome = finish_segment_job(job);
    match request_item {
        Some(request_item) => RoutedTranscribeItemOutcome::for_request(request_item, outcome),
        None => outcome.into(),
    }
}

#[derive(Debug)]
struct SegmentAsrInference {
    targets: Vec<(usize, usize, DurationMs)>,
    segment_audio_seconds: f64,
    active_job_count: usize,
}

#[derive(Debug)]
struct SegmentEncodedAsrBatch {
    inference: SegmentAsrInference,
    waveforms: Vec<Waveform>,
    encoded: EncodedAsrBatch,
    timings: AudioFeatureEncodeTimings,
}

impl SegmentEncodedAsrBatch {
    #[cfg(not(feature = "paged-attn"))]
    fn merge(batches: Vec<Self>) -> Result<Self> {
        let mut targets = Vec::new();
        let mut waveforms = Vec::new();
        let mut encoded_batches = Vec::with_capacity(batches.len());
        let mut segment_audio_seconds = 0.0;
        let mut active_jobs = HashSet::new();

        for mut batch in batches {
            segment_audio_seconds += batch.inference.segment_audio_seconds;
            for (job_index, _, _) in &batch.inference.targets {
                active_jobs.insert(*job_index);
            }
            targets.append(&mut batch.inference.targets);
            waveforms.append(&mut batch.waveforms);
            encoded_batches.push(batch.encoded);
        }

        let encoded = EncodedAsrBatch::merge(encoded_batches)?;
        if encoded.len() != waveforms.len() {
            bail!(
                "merged staged ASR batch mismatch: encoded={} waveforms={}",
                encoded.len(),
                waveforms.len()
            );
        }

        Ok(Self {
            inference: SegmentAsrInference {
                targets,
                segment_audio_seconds,
                active_job_count: active_jobs.len(),
            },
            waveforms,
            timings: encoded.audio_feature_encode_timings(),
            encoded,
        })
    }

    #[cfg(feature = "paged-attn")]
    fn into_parts(self) -> (SegmentAsrInference, Vec<Waveform>, EncodedAsrBatch) {
        (self.inference, self.waveforms, self.encoded)
    }
}

fn run_segment_asr_inference(
    offline: &OfflinePipeline,
    segments: &[AsrSegmentJob],
    asr_options: &AsrOptions,
) -> Result<(SegmentAsrInference, Vec<Timeline>)> {
    if offline.asr.as_any().is::<Qwen3ASR>() {
        let encoded = run_segment_asr_encode(offline, segments, asr_options)?;
        #[cfg(feature = "timing")]
        {
            let (inference, timelines, _timings) =
                run_segment_asr_decode(offline.asr.as_ref(), encoded, asr_options)?;
            return Ok((inference, timelines));
        }
        #[cfg(not(feature = "timing"))]
        return run_segment_asr_decode(offline.asr.as_ref(), encoded, asr_options);
    }
    let slices: Vec<Waveform> = segments
        .iter()
        .map(|segment| segment.waveform.clone())
        .collect();
    let targets: Vec<(usize, usize, DurationMs)> = segments
        .iter()
        .map(|segment| (segment.job_index, segment.segment_index, segment.offset))
        .collect();
    let active_job_count = segments
        .iter()
        .map(|segment| segment.job_index)
        .collect::<HashSet<_>>()
        .len();
    let segment_audio_seconds: f64 = slices.iter().map(Waveform::duration_seconds).sum();
    let timelines = offline.asr.transcribe(slices.as_slice(), asr_options)?;
    Ok((
        SegmentAsrInference {
            targets,
            segment_audio_seconds,
            active_job_count,
        },
        timelines,
    ))
}

fn run_segment_asr_encode(
    offline: &OfflinePipeline,
    segments: &[AsrSegmentJob],
    asr_options: &AsrOptions,
) -> Result<SegmentEncodedAsrBatch> {
    let qwen3 = offline
        .asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("staged ASR requires Qwen3ASR"))?;
    let waveforms: Vec<Waveform> = segments
        .iter()
        .map(|segment| segment.waveform.clone())
        .collect();
    let targets: Vec<(usize, usize, DurationMs)> = segments
        .iter()
        .map(|segment| (segment.job_index, segment.segment_index, segment.offset))
        .collect();
    let active_job_count = segments
        .iter()
        .map(|segment| segment.job_index)
        .collect::<HashSet<_>>()
        .len();
    let segment_audio_seconds: f64 = waveforms.iter().map(Waveform::duration_seconds).sum();
    let encoded =
        qwen3.encode_waveforms_audio_features_to_cpu(waveforms.as_slice(), asr_options)?;
    let timings = encoded.audio_feature_encode_timings();
    Ok(SegmentEncodedAsrBatch {
        inference: SegmentAsrInference {
            targets,
            segment_audio_seconds,
            active_job_count,
        },
        waveforms,
        encoded,
        timings,
    })
}

#[cfg(feature = "timing")]
fn run_segment_asr_decode(
    asr: &dyn crate::runtime::model::AsrModel,
    encoded: SegmentEncodedAsrBatch,
    asr_options: &AsrOptions,
) -> Result<(SegmentAsrInference, Vec<Timeline>, TranscribeTimings)> {
    let qwen3 = asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("staged ASR decode requires Qwen3ASR"))?;
    let (timelines, timings) = qwen3.decode_cpu_audio_features_to_timelines_timed(
        encoded.encoded,
        encoded.waveforms.as_slice(),
        asr_options,
    )?;
    Ok((encoded.inference, timelines, timings))
}

#[cfg(not(feature = "timing"))]
fn run_segment_asr_decode(
    asr: &dyn crate::runtime::model::AsrModel,
    encoded: SegmentEncodedAsrBatch,
    asr_options: &AsrOptions,
) -> Result<(SegmentAsrInference, Vec<Timeline>)> {
    let qwen3 = asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("staged ASR decode requires Qwen3ASR"))?;
    let timelines = qwen3.decode_cpu_audio_features_to_timelines(
        encoded.encoded,
        encoded.waveforms.as_slice(),
        asr_options,
    )?;
    Ok((encoded.inference, timelines))
}

#[cfg(feature = "paged-attn")]
fn run_segment_asr_decode_dynamic(
    asr: &dyn crate::runtime::model::AsrModel,
    initial_segments: Vec<AsrSegmentJob>,
    initial: SegmentEncodedAsrBatch,
    encoded_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncodedMessage>>>,
    runtime_handle: tokio::runtime::Handle,
    asr_options: &AsrOptions,
    decode_idle_flush: Option<std::time::Duration>,
) -> Result<DynamicSegmentAsrDecode> {
    let qwen3 = asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("dynamic staged ASR decode requires Qwen3ASR"))?;
    let (mut inference, mut waveforms, encoded) = initial.into_parts();
    let mut consumed_segments = initial_segments;
    let mut encoded_failures = Vec::new();
    let mut source_closed = false;
    let mut poll_encoded = |may_block: bool| -> Result<EncodedAsrBatchPoll> {
        if decode_idle_flush.is_some() && !may_block {
            return Ok(EncodedAsrBatchPoll::Empty);
        }
        loop {
            if source_closed {
                return Ok(EncodedAsrBatchPoll::Closed);
            }
            let message = if may_block {
                if let Some(timeout) = decode_idle_flush {
                    match runtime_handle.block_on(async {
                        let mut encoded_rx = encoded_rx.lock().await;
                        tokio::time::timeout(timeout, encoded_rx.recv()).await
                    }) {
                        Ok(Some(message)) => Some(message),
                        Ok(None) => {
                            source_closed = true;
                            return Ok(EncodedAsrBatchPoll::Closed);
                        }
                        Err(tokio::time::error::Elapsed { .. }) => {
                            source_closed = true;
                            return Ok(EncodedAsrBatchPoll::Closed);
                        }
                    }
                } else {
                    match runtime_handle.block_on(async {
                        let mut encoded_rx = encoded_rx.lock().await;
                        encoded_rx.recv().await
                    }) {
                        Some(message) => Some(message),
                        None => {
                            source_closed = true;
                            return Ok(EncodedAsrBatchPoll::Closed);
                        }
                    }
                }
            } else {
                match runtime_handle.block_on(async {
                    let mut encoded_rx = encoded_rx.lock().await;
                    encoded_rx.try_recv()
                }) {
                    Ok(message) => Some(message),
                    Err(mpsc::error::TryRecvError::Empty) => None,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        source_closed = true;
                        return Ok(EncodedAsrBatchPoll::Closed);
                    }
                }
            };
            let Some(message) = message else {
                return Ok(EncodedAsrBatchPoll::Empty);
            };
            match message.batch {
                Ok(batch) => {
                    consumed_segments.extend(message.segments);
                    let (next_inference, mut next_waveforms, next_encoded) = batch.into_parts();
                    inference.targets.extend(next_inference.targets);
                    inference.segment_audio_seconds += next_inference.segment_audio_seconds;
                    inference.active_job_count = inference
                        .targets
                        .iter()
                        .map(|(job_index, _, _)| *job_index)
                        .collect::<HashSet<_>>()
                        .len();
                    waveforms.append(&mut next_waveforms);
                    return Ok(EncodedAsrBatchPoll::Ready(next_encoded));
                }
                Err(error) => {
                    encoded_failures.push(SegmentEncodedFailure {
                        segments: message.segments,
                        stage_start: message.stage_start,
                        error,
                    });
                }
            }
        }
    };
    #[cfg(feature = "timing")]
    let (outputs, timings) = qwen3.decode_cpu_audio_features_dynamic_outputs_timed(
        encoded,
        &mut poll_encoded,
        asr_options,
    )?;
    #[cfg(not(feature = "timing"))]
    let outputs =
        qwen3.decode_cpu_audio_features_dynamic_outputs(encoded, &mut poll_encoded, asr_options)?;
    #[cfg(all(feature = "cuda", feature = "paged-attn"))]
    if decode_idle_flush.is_some() {
        vasr::vasr_paged_attn::cuda_graph::cuda_trim_device_mem_pools(qwen3.device());
    }
    let mut timelines = Vec::with_capacity(waveforms.len());
    let mut waveform_offset = 0usize;
    for outputs in outputs {
        let end = waveform_offset.saturating_add(outputs.len());
        if end > waveforms.len() {
            bail!(
                "dynamic ASR waveform split mismatch: end={} waveforms={}",
                end,
                waveforms.len()
            );
        }
        timelines.extend(qwen_outputs_to_timelines(
            outputs,
            &waveforms[waveform_offset..end],
        )?);
        waveform_offset = end;
    }
    if waveform_offset != waveforms.len() {
        bail!(
            "dynamic ASR unused waveforms: used={} waveforms={}",
            waveform_offset,
            waveforms.len()
        );
    }
    Ok(DynamicSegmentAsrDecode {
        consumed_segments,
        inference,
        timelines,
        encoded_failures,
        #[cfg(feature = "timing")]
        timings,
    })
}

fn finish_segment_asr_batch(
    state: &mut SegmentAsrState,
    segments: &[AsrSegmentJob],
    stage_start: Instant,
    batch: Result<(SegmentAsrInference, Vec<Timeline>)>,
    stats: Option<&Arc<Mutex<AsrStageStats>>>,
) -> Vec<RoutedTranscribeItemOutcome> {
    if segments.is_empty() {
        return Vec::new();
    }
    prepare_segment_job_state(state, segments);

    let mut completed = Vec::new();
    match batch {
        Ok((inference, timelines)) => {
            let elapsed = stage_start.elapsed().as_secs_f64();
            if stats.is_some() {
                log_asr_wave(
                    "vad-segment",
                    inference.active_job_count,
                    segments.len(),
                    inference.segment_audio_seconds,
                    elapsed,
                );
            } else {
                log_asr(
                    "vad-segment",
                    inference.active_job_count,
                    segments.len(),
                    inference.segment_audio_seconds,
                    elapsed,
                );
            }
            if let Some(stats) = stats {
                record_asr_batch_stats(
                    stats,
                    inference.active_job_count,
                    segments.len(),
                    inference.segment_audio_seconds,
                    elapsed,
                );
            }
            if timelines.len() == inference.targets.len() {
                for ((job_index, segment_index, offset), asr_timeline) in
                    inference.targets.into_iter().zip(timelines)
                {
                    if let Some(job) = state.jobs.get_mut(&job_index) {
                        if segment_index >= job.segment_annotations.len() {
                            tracing::warn!(
                                target: "vasr_transcribe::pipeline",
                                "VAD segment index out of range for job={} segment_index={} segment_count={}",
                                job_index,
                                segment_index,
                                job.segment_annotations.len()
                            );
                            continue;
                        }
                        if job.segment_annotations[segment_index].is_none() {
                            job.segments_done = job.segments_done.saturating_add(1);
                        }
                        job.segment_annotations[segment_index] =
                            Some(offset_annotations(asr_timeline.annotations, offset));
                        if job.segments_done >= job.segment_count {
                            if let Some(job) = state.jobs.remove(&job_index) {
                                completed.push(routed_segment_job_outcome(job));
                            }
                        }
                    }
                }
            } else {
                let err = format!(
                    "ASR segment batch returned {} timelines for {} segments",
                    timelines.len(),
                    segments.len()
                );
                let affected: HashSet<usize> =
                    segments.iter().map(|segment| segment.job_index).collect();
                for job_index in affected {
                    if let Some(job) = state.jobs.remove(&job_index) {
                        let request_item = job.request_item;
                        let outcome = outcome_from_result(
                            job.index,
                            job.audio_seconds,
                            Err(anyhow::anyhow!(err.clone())),
                        );
                        completed.push(route_outcome(request_item, outcome));
                    }
                }
            }
        }
        Err(error) => {
            let reason = error.to_string();
            let affected: HashSet<usize> =
                segments.iter().map(|segment| segment.job_index).collect();
            for job_index in affected {
                if let Some(job) = state.jobs.remove(&job_index) {
                    let request_item = job.request_item;
                    let outcome = outcome_from_result(
                        job.index,
                        job.audio_seconds,
                        Err(anyhow::anyhow!(reason.clone())),
                    );
                    completed.push(route_outcome(request_item, outcome));
                }
            }
        }
    }

    completed
}

fn run_segment_microbatch(
    offline: &OfflinePipeline,
    segments: Vec<AsrSegmentJob>,
    asr_options: &AsrOptions,
    state: &mut SegmentAsrState,
) -> Vec<TranscribeItemOutcome> {
    if segments.is_empty() {
        return Vec::new();
    }

    let stage_start = Instant::now();
    let batch = run_segment_asr_inference(offline, &segments, asr_options);
    finish_segment_asr_batch(state, &segments, stage_start, batch, None)
        .into_iter()
        .map(RoutedTranscribeItemOutcome::into_public_outcome)
        .collect()
}

fn run_asr_jobs_batch(
    offline: &OfflinePipeline,
    jobs: Vec<AsrJob>,
    asr_options: &AsrOptions,
    stats: Option<&Arc<Mutex<AsrStageStats>>>,
) -> Vec<RoutedTranscribeItemOutcome> {
    let mut outcomes = Vec::with_capacity(jobs.len());
    let mut raw_jobs: Vec<(AsrJob, Vec<Annotation>)> = Vec::new();
    let mut prepared_jobs = Vec::new();

    for mut job in jobs {
        let prepared = std::mem::replace(&mut job.prepared, VadPreparation::Disabled);
        match prepared {
            VadPreparation::Speech(prepared) => {
                job.prepared = VadPreparation::Speech(prepared);
                prepared_jobs.push(job);
            }
            VadPreparation::NoSpeech => {
                let request_item = job.request_item;
                let outcome = TranscribeItemOutcome {
                    index: job.index,
                    result: Ok(Timeline::new("offline_audio")),
                    audio_seconds: job.waveform.duration_seconds(),
                    bad_component: None,
                };
                outcomes.push(route_outcome(request_item, outcome));
            }
            VadPreparation::Disabled => raw_jobs.push((job, Vec::new())),
        }
    }

    if !raw_jobs.is_empty() {
        let indices: Vec<usize> = raw_jobs.iter().map(|(job, _)| job.index).collect();
        let request_items: Vec<Option<RequestItemRef>> =
            raw_jobs.iter().map(|(job, _)| job.request_item).collect();
        let audio_seconds: Vec<f64> = raw_jobs
            .iter()
            .map(|(job, _)| job.waveform.duration_seconds())
            .collect();
        let speech_annotations = raw_jobs
            .iter()
            .map(|(_, annotations)| annotations.clone())
            .collect::<Vec<_>>();
        let total_audio_seconds: f64 = audio_seconds.iter().sum();
        let waveforms: Vec<Waveform> = raw_jobs.into_iter().map(|(job, _)| job.waveform).collect();
        let stage_start = Instant::now();
        let result = offline.asr.transcribe(&waveforms, asr_options);
        let elapsed = stage_start.elapsed().as_secs_f64();
        if stats.is_some() {
            log_asr_wave(
                "raw",
                indices.len(),
                waveforms.len(),
                total_audio_seconds,
                elapsed,
            );
        } else {
            log_asr(
                "raw",
                indices.len(),
                waveforms.len(),
                total_audio_seconds,
                elapsed,
            );
        }
        if let Some(stats) = stats {
            record_asr_batch_stats(
                stats,
                indices.len(),
                waveforms.len(),
                total_audio_seconds,
                elapsed,
            );
        }
        match result {
            Ok(timelines) if timelines.len() == indices.len() => {
                for ((((index, request_item), audio_seconds), mut timeline), mut annotations) in
                    indices
                        .into_iter()
                        .zip(request_items)
                        .zip(audio_seconds)
                        .zip(timelines)
                        .zip(speech_annotations)
                {
                    if !annotations.is_empty() {
                        annotations.extend(timeline.annotations);
                        timeline.annotations = annotations;
                    }
                    outcomes.push(route_outcome(
                        request_item,
                        TranscribeItemOutcome {
                            index,
                            result: Ok(timeline),
                            audio_seconds,
                            bad_component: None,
                        },
                    ));
                }
            }
            Ok(timelines) => {
                let err = anyhow::anyhow!(
                    "ASR batch returned {} timelines for {} jobs",
                    timelines.len(),
                    indices.len()
                );
                for ((index, request_item), audio_seconds) in
                    indices.into_iter().zip(request_items).zip(audio_seconds)
                {
                    let outcome =
                        outcome_from_result(index, audio_seconds, Err(anyhow::anyhow!("{err}")));
                    outcomes.push(route_outcome(request_item, outcome));
                }
            }
            Err(error) => {
                let reason = error.to_string();
                for ((index, request_item), audio_seconds) in
                    indices.into_iter().zip(request_items).zip(audio_seconds)
                {
                    let outcome =
                        outcome_from_result(index, audio_seconds, Err(anyhow::anyhow!("{reason}")));
                    outcomes.push(route_outcome(request_item, outcome));
                }
            }
        }
    }

    if !prepared_jobs.is_empty() {
        let mut segment_state = SegmentAsrState::default();
        let mut segment_jobs = Vec::new();
        for job in prepared_jobs {
            match speech_job_to_segments(job) {
                Ok(mut segments) => segment_jobs.append(&mut segments),
                Err(error) => outcomes.push(
                    TranscribeItemOutcome {
                        index: 0,
                        result: Err(error),
                        audio_seconds: 0.0,
                        bad_component: Some("vad"),
                    }
                    .into(),
                ),
            }
        }
        outcomes.extend(
            run_segment_microbatch(offline, segment_jobs, asr_options, &mut segment_state)
                .into_iter()
                .map(Into::into),
        );
    }

    outcomes.sort_unstable_by_key(|outcome| outcome.outcome.index);
    outcomes
}

#[derive(Debug, Clone, Copy)]
struct StageMetrics {
    speedup: f64,
    rtf: f64,
}

fn stage_metrics(audio_seconds: f64, wall_seconds: f64) -> StageMetrics {
    if audio_seconds <= 0.0 || wall_seconds <= 0.0 {
        return StageMetrics {
            speedup: 0.0,
            rtf: 0.0,
        };
    }
    StageMetrics {
        speedup: audio_seconds / wall_seconds,
        rtf: wall_seconds / audio_seconds,
    }
}

fn vad_preparation_stats(prepared: &VadPreparation) -> (usize, f64) {
    match prepared {
        VadPreparation::Speech(prepared) => {
            let speech_seconds = prepared
                .segments
                .iter()
                .map(|segment| segment.range.duration().0 as f64 / 1000.0)
                .sum();
            (prepared.segments.len(), speech_seconds)
        }
        VadPreparation::Disabled | VadPreparation::NoSpeech => (0, 0.0),
    }
}

fn log_vad(
    index: usize,
    audio_seconds: f64,
    segments: usize,
    speech_seconds: f64,
    compute_seconds: f64,
    send_wait_seconds: f64,
    wall_seconds: f64,
) {
    let metrics = stage_metrics(audio_seconds, wall_seconds);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "vad | job={index} | audio={audio_seconds:.2}s | segments={segments} | speech={speech_seconds:.2}s | compute={compute_seconds:.2}s | send_wait={send_wait_seconds:.2}s | wall={wall_seconds:.2}s | speed={:.2}x | rtf={:.3}",
        metrics.speedup,
        metrics.rtf
    );
}

fn record_loader_task_stats(stats: &Arc<Mutex<LoaderStageStats>>, task: &LoaderTaskStats) {
    if let Ok(mut stats) = stats.lock() {
        stats.jobs += task.jobs;
        stats.failed += task.failed;
        stats.audio_seconds += task.audio_seconds;
        stats.compute_seconds += task.compute_seconds;
    }
}

fn record_vad_task_stats(stats: &Arc<Mutex<VadStageStats>>, task: &VadTaskStats) {
    if let Ok(mut stats) = stats.lock() {
        stats.jobs += task.jobs;
        stats.failed += task.failed;
        stats.no_speech += task.no_speech;
        stats.segments += task.segments;
        stats.audio_seconds += task.audio_seconds;
        stats.speech_seconds += task.speech_seconds;
        stats.compute_seconds += task.compute_seconds;
        stats.send_wait_seconds += task.send_wait_seconds;
        stats.wall_seconds += task.wall_seconds;
    }
}

fn record_asr_encode_stats(
    stats: &Arc<Mutex<AsrStageStats>>,
    segments: usize,
    audio_seconds: f64,
    wall_seconds: f64,
) {
    if let Ok(mut stats) = stats.lock() {
        stats.encode_batches += 1;
        stats.encode_segments += segments;
        stats.encode_audio_seconds += audio_seconds;
        stats.encode_seconds += wall_seconds;
    }
}

#[cfg(feature = "timing")]
fn record_asr_timing_stats(stats: &Arc<Mutex<AsrStageStats>>, timings: &TranscribeTimings) {
    if let Ok(mut stats) = stats.lock() {
        stats.audio_encoder_seconds += timings.audio_encoder_us as f64 / 1_000_000.0;
        stats.prefill_seconds += timings.generation.prefill_us as f64 / 1_000_000.0;
        stats.prefill_tokens = stats
            .prefill_tokens
            .saturating_add(timings.generation.prefill_tokens);
        stats.decode_model_seconds += timings.generation.decode_us as f64 / 1_000_000.0;
        stats.decode_tokens = stats
            .decode_tokens
            .saturating_add(timings.generation.tokens_generated);
        stats.decode_steps = stats.decode_steps.saturating_add(timings.generation.steps);
    }
}

fn record_asr_decode_stats(
    stats: &Arc<Mutex<AsrStageStats>>,
    segments: usize,
    audio_seconds: f64,
    wall_seconds: f64,
) {
    if let Ok(mut stats) = stats.lock() {
        stats.decode_batches += 1;
        stats.decode_segments += segments;
        stats.decode_audio_seconds += audio_seconds;
        stats.decode_seconds += wall_seconds;
    }
}

fn record_asr_batch_stats(
    stats: &Arc<Mutex<AsrStageStats>>,
    jobs: usize,
    segments: usize,
    audio_seconds: f64,
    wall_seconds: f64,
) {
    if let Ok(mut stats) = stats.lock() {
        stats.batches += 1;
        stats.jobs += jobs;
        stats.segments += segments;
        stats.audio_seconds += audio_seconds;
        stats.wall_seconds += wall_seconds;
    }
}

fn log_loader_summary(stats: &LoaderStageStats) {
    let metrics = stage_metrics(stats.audio_seconds, stats.wall_seconds);
    tracing::info!(
        target: "vasr_transcribe::pipeline",
        "loader | jobs={} failed={} audio={:.1}s wall={:.1}s compute={:.1}s speed={:.1}x",
        stats.jobs,
        stats.failed,
        stats.audio_seconds,
        stats.wall_seconds,
        stats.compute_seconds,
        metrics.speedup
    );
}

fn log_vad_summary(stats: &VadStageStats) {
    let metrics = stage_metrics(stats.audio_seconds, stats.wall_seconds);
    tracing::info!(
        target: "vasr_transcribe::pipeline",
        "vad | jobs={} failed={} no_speech={} segments={} speech={:.1}s audio={:.1}s wall={:.1}s compute={:.1}s send_wait={:.1}s speed={:.1}x",
        stats.jobs,
        stats.failed,
        stats.no_speech,
        stats.segments,
        stats.speech_seconds,
        stats.audio_seconds,
        stats.wall_seconds,
        stats.compute_seconds,
        stats.send_wait_seconds,
        metrics.speedup
    );
}

fn log_asr_audio_encoder_batch(
    segments: usize,
    audio_s: f64,
    wall_s: f64,
    timings: AudioFeatureEncodeTimings,
) {
    let speedup = audio_s / wall_s.max(f64::EPSILON);
    let prepare_s = timing_us_to_seconds(timings.prepare_us);
    let stack_s = timing_us_to_seconds(timings.stack_us);
    let forward_s = timing_us_to_seconds(timings.forward_us);
    let d2h_s = timing_us_to_seconds(timings.d2h_us);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "asr audio_encoder | segs={segments} audio={audio_s:.1}s wall={wall_s:.2}s speed={speedup:.1}x prepare={prepare_s:.3}s stack={stack_s:.3}s forward={forward_s:.3}s d2h={d2h_s:.3}s"
    );
}

fn timing_us_to_seconds(us: u64) -> f64 {
    us as f64 / 1_000_000.0
}

fn log_asr_summary(stats: &AsrStageStats) {
    if stats.jobs == 0 && stats.segments == 0 {
        return;
    }
    let metrics = stage_metrics(stats.audio_seconds, stats.wall_seconds);
    let audio_encoder_seconds = if stats.audio_encoder_seconds > 0.0 {
        stats.audio_encoder_seconds
    } else {
        stats.encode_seconds
    };
    let audio_encoder_speed = stats.encode_audio_seconds / audio_encoder_seconds.max(f64::EPSILON);
    let prefill_ktok_s =
        stats.prefill_tokens as f64 / stats.prefill_seconds.max(f64::EPSILON) / 1000.0;
    let decode_ktok_s =
        stats.decode_tokens as f64 / stats.decode_model_seconds.max(f64::EPSILON) / 1000.0;
    tracing::info!(
        target: "vasr_transcribe::pipeline",
        "asr | jobs={} segs={} audio={:.1}s wall={:.1}s speed={:.1}x audio_encoder_compute={:.1}s/{:.1}x prefill_compute={:.1}s tokens={} {:.1}ktok/s decode_compute={:.1}s tokens={} {:.1}ktok/s decode_calls={}",
        stats.jobs,
        stats.segments,
        stats.audio_seconds,
        stats.wall_seconds,
        metrics.speedup,
        audio_encoder_seconds,
        audio_encoder_speed,
        stats.prefill_seconds,
        stats.prefill_tokens,
        prefill_ktok_s,
        stats.decode_model_seconds,
        stats.decode_tokens,
        decode_ktok_s,
        stats.decode_steps
    );
}

fn log_asr(kind: &'static str, jobs: usize, segments: usize, audio_s: f64, wall_s: f64) {
    let speedup = audio_s / wall_s.max(f64::EPSILON);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "asr | {kind} | jobs={jobs} segs={segments} | audio={audio_s:.1}s | wall={wall_s:.1}s | speed={speedup:.1}x"
    );
}

fn log_asr_wave(kind: &'static str, jobs: usize, segments: usize, audio_s: f64, wall_s: f64) {
    let speedup = audio_s / wall_s.max(f64::EPSILON);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "asr wave | mode=serve | source={kind} | jobs={jobs} segs={segments} | audio={audio_s:.1}s | wall={wall_s:.1}s | speed={speedup:.1}x"
    );
}

fn log_pipeline(job_count: usize, outcomes: &[TranscribeItemOutcome], wall_seconds: f64) {
    let audio_seconds: f64 = outcomes.iter().map(|outcome| outcome.audio_seconds).sum();
    let bad = outcomes
        .iter()
        .filter(|outcome| outcome.result.is_err())
        .count();
    let metrics = stage_metrics(audio_seconds, wall_seconds);
    for outcome in outcomes {
        if let Err(error) = &outcome.result {
            tracing::error!(
                target: "vasr_transcribe::pipeline",
                index = outcome.index,
                component = outcome.bad_component.unwrap_or("unknown"),
                error = %error,
                audio_s = outcome.audio_seconds,
                "Pipeline job failed"
            );
        }
    }
    tracing::info!(
        target: "vasr_transcribe::pipeline",
        "pipeline | batch={} | returned={} | audio={:.2}s | spent={:.2}s | speed={:.2}x | rtf={:.3} | bad={}",
        job_count,
        outcomes.len(),
        audio_seconds,
        wall_seconds,
        metrics.speedup,
        metrics.rtf,
        bad
    );
}

fn outcome_from_result(
    index: usize,
    audio_seconds: f64,
    result: Result<Timeline>,
) -> TranscribeItemOutcome {
    let bad_component = if result.is_err() {
        Some("recognizer")
    } else {
        None
    };
    TranscribeItemOutcome {
        index,
        result,
        audio_seconds,
        bad_component,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{AsrModel, VadSegment};
    use std::sync::{Arc, Mutex};
    use vasr_data::{
        Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, DurationMs, TextSpan,
        TimeRange,
    };

    #[test]
    fn default_stage_parallelism_is_nonzero() {
        assert!(default_stage_parallelism() >= 1);
    }

    #[test]
    fn loaded_queue_memory_budget_defaults_to_twenty_percent_of_total_memory() {
        assert_eq!(
            loaded_queue_memory_budget_permits(None, None, Some(64 * 1024 * 1024 * 1024)),
            LoadedQueueMemoryBudgetConfig {
                permits: 13_107,
                source: "system-memory-fraction"
            }
        );
    }

    #[test]
    fn loaded_queue_memory_budget_accepts_explicit_mb_override() {
        assert_eq!(
            loaded_queue_memory_budget_permits(
                Some("512"),
                Some("0.5"),
                Some(64 * 1024 * 1024 * 1024),
            ),
            LoadedQueueMemoryBudgetConfig {
                permits: 512,
                source: "env:VASR_LOADED_QUEUE_MEMORY_MB"
            }
        );
    }

    #[test]
    fn loaded_queue_memory_budget_accepts_fraction_override() {
        assert_eq!(
            loaded_queue_memory_budget_permits(None, Some("0.10"), Some(64 * 1024 * 1024 * 1024)),
            LoadedQueueMemoryBudgetConfig {
                permits: 6_553,
                source: "env:VASR_LOADED_QUEUE_MEMORY_FRACTION"
            }
        );
    }

    #[test]
    fn loaded_job_memory_permits_rounds_up_to_megabytes() {
        let job = LoadedJob {
            index: 0,
            waveform: Waveform::new(vec![0.0; 1024 * 1024 + 1], 16_000),
            _queue_memory_permit: None,
        };
        assert_eq!(loaded_job_memory_permits(&job), 5);
    }

    #[test]
    fn vad_to_asr_segment_memory_permits_rounds_up_to_megabytes() {
        let message = Ok(AsrPipelineMessage::Segment(AsrSegmentJob {
            job_index: 0,
            job_audio_seconds: 1.0,
            speech_annotations: Vec::new(),
            segment_index: 0,
            segment_count: 1,
            offset: DurationMs(0),
            waveform: Waveform::new(vec![0.0; 1024 * 1024 + 1], 16_000),
        }));
        assert_eq!(vad_to_asr_message_memory_permits(&message), 5);
    }

    #[test]
    fn vad_to_asr_error_messages_do_not_need_memory_permits() {
        let message = Err(TranscribeItemOutcome {
            index: 0,
            result: Err(anyhow::anyhow!("failed")),
            audio_seconds: 0.0,
            bad_component: Some("vad"),
        });
        assert_eq!(vad_to_asr_message_memory_permits(&message), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn parses_mem_total_from_proc_meminfo() {
        let text = "MemAvailable:    98765 kB\nMemTotal:       123456 kB\n";
        assert_eq!(parse_meminfo_kib(text, "MemTotal:"), Some(123_456));
    }

    #[derive(Default)]
    struct BatchRecordingAsr {
        seen_batch_sizes: Mutex<Vec<usize>>,
    }

    impl AsrModel for BatchRecordingAsr {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn transcribe(
            &self,
            waveforms: &[Waveform],
            _options: &AsrOptions,
        ) -> Result<Vec<Timeline>> {
            self.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned")
                .push(waveforms.len());
            Ok(waveforms
                .iter()
                .map(|_| {
                    let mut timeline = Timeline::new("fake_batch_asr");
                    timeline.push(Annotation::new(
                        Default::default(),
                        AnnotationPayload::Transcription(TextSpan::new("ok")),
                        AnnotationSource::Model("fake_batch_asr".to_string()),
                        AnnotationStatus::Final,
                    ));
                    timeline
                })
                .collect())
        }
    }

    #[test]
    fn vad_segments_from_multiple_jobs_are_transcribed_as_one_batch() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        };
        let jobs = vec![
            AsrJob {
                index: 0,
                prepared: VadPreparation::Speech(prepared_with_segments(&[(0, 100), (250, 400)])),
                waveform: Waveform::new(vec![0.1; 8_000], 16_000),
            },
            AsrJob {
                index: 1,
                prepared: VadPreparation::Speech(prepared_with_segments(&[(50, 200)])),
                waveform: Waveform::new(vec![0.2; 8_000], 16_000),
            },
        ];

        let outcomes = run_asr_jobs_batch(&pipeline, jobs, &AsrOptions::default(), None);

        assert_eq!(outcomes.len(), 2);
        assert!(outcomes.iter().all(|outcome| outcome.result.is_ok()));
        assert_eq!(
            *asr.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned"),
            vec![3]
        );
        Ok(())
    }

    #[test]
    fn segment_batch_error_after_partial_success_returns_job_error() -> Result<()> {
        let mut state = SegmentAsrState::default();
        let segments = vec![
            AsrSegmentJob {
                job_index: 7,
                job_audio_seconds: 1.0,
                speech_annotations: Vec::new(),
                segment_index: 0,
                segment_count: 2,
                offset: DurationMs(0),
                waveform: Waveform::new(vec![0.1; 160], 16_000),
            },
            AsrSegmentJob {
                job_index: 7,
                job_audio_seconds: 1.0,
                speech_annotations: Vec::new(),
                segment_index: 1,
                segment_count: 2,
                offset: DurationMs(100),
                waveform: Waveform::new(vec![0.2; 160], 16_000),
            },
        ];
        prepare_segment_job_state(&mut state, segments.as_slice());

        let mut ok_timeline = Timeline::new("fake_batch_asr");
        ok_timeline.push(Annotation::new(
            Default::default(),
            AnnotationPayload::Transcription(TextSpan::new("ok")),
            AnnotationSource::Model("fake_batch_asr".to_string()),
            AnnotationStatus::Final,
        ));
        let ok = finish_segment_asr_batch(
            &mut state,
            &segments[..1],
            Instant::now(),
            Ok((
                SegmentAsrInference {
                    targets: vec![(7, 0, DurationMs(0))],
                    segment_audio_seconds: 0.01,
                    active_job_count: 1,
                },
                vec![ok_timeline],
            )),
            None,
        );
        assert!(ok.is_empty());

        let failed = finish_segment_asr_batch(
            &mut state,
            &segments[1..],
            Instant::now(),
            Err(anyhow::anyhow!("encoded feature failed")),
            None,
        );

        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].index, 7);
        let err = failed[0]
            .result
            .as_ref()
            .expect_err("partial segment failure should fail the whole job")
            .to_string();
        assert!(err.contains("encoded feature failed"));
        assert!(!state.jobs.contains_key(&7));
        Ok(())
    }

    #[test]
    fn segment_batch_error_registers_missing_job_state() -> Result<()> {
        let mut state = SegmentAsrState::default();
        let segments = vec![AsrSegmentJob {
            job_index: 11,
            job_audio_seconds: 2.5,
            speech_annotations: Vec::new(),
            segment_index: 0,
            segment_count: 1,
            offset: DurationMs(0),
            waveform: Waveform::new(vec![0.1; 160], 16_000),
        }];

        let failed = finish_segment_asr_batch(
            &mut state,
            segments.as_slice(),
            Instant::now(),
            Err(anyhow::anyhow!("first encoded batch failed")),
            None,
        );

        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].index, 11);
        assert_eq!(failed[0].audio_seconds, 2.5);
        let err = failed[0]
            .result
            .as_ref()
            .expect_err("missing-state encode failure should still return an outcome")
            .to_string();
        assert!(err.contains("first encoded batch failed"));
        assert!(state.jobs.is_empty());
        Ok(())
    }

    #[test]
    fn no_speech_jobs_are_returned_without_asr() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        };
        let jobs = vec![AsrJob {
            index: 0,
            prepared: VadPreparation::NoSpeech,
            waveform: Waveform::new(vec![0.1; 160], 16_000),
        }];

        let outcomes = run_asr_jobs_batch(&pipeline, jobs, &AsrOptions::default(), None);

        assert_eq!(outcomes.len(), 1);
        assert!(
            outcomes[0]
                .result
                .as_ref()
                .is_ok_and(|timeline| timeline.annotations.is_empty())
        );
        assert!(
            asr.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned")
                .is_empty()
        );
        Ok(())
    }

    #[tokio::test]
    async fn asr_worker_coalesces_slightly_staggered_jobs() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = Arc::new(OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        });
        let (vad_to_asr_tx, vad_to_asr_rx) = mpsc::unbounded_channel();
        let (asr_to_result_tx, mut asr_to_result_rx) = mpsc::channel(4);
        let stats = Arc::new(Mutex::new(AsrStageStats::default()));
        let handle = spawn_asr_worker(
            pipeline,
            AsrOptions::default(),
            60.0,
            vad_to_asr_rx,
            asr_to_result_tx,
            stats,
        );

        for index in 0..3usize {
            vad_to_asr_tx.send(test_vad_to_asr_message(Ok(AsrPipelineMessage::Job(
                AsrJob {
                    index,
                    prepared: VadPreparation::Disabled,
                    waveform: Waveform::new(vec![index as f32; 160], 16_000),
                },
            ))))?;
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        }
        drop(vad_to_asr_tx);

        let mut outcomes = Vec::new();
        while let Some(outcome) = asr_to_result_rx.recv().await {
            outcomes.push(outcome);
        }
        handle.await?;

        assert_eq!(outcomes.len(), 3);
        assert_eq!(
            *asr.seen_batch_sizes
                .lock()
                .expect("seen batch sizes poisoned"),
            vec![3]
        );
        Ok(())
    }

    #[tokio::test]
    async fn asr_worker_does_not_drop_job_after_segment_microbatch() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = Arc::new(OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        });
        let (vad_to_asr_tx, vad_to_asr_rx) = mpsc::unbounded_channel();
        let (asr_to_result_tx, mut asr_to_result_rx) = mpsc::channel(4);
        let stats = Arc::new(Mutex::new(AsrStageStats::default()));
        let handle = spawn_asr_worker(
            pipeline,
            AsrOptions::default(),
            60.0,
            vad_to_asr_rx,
            asr_to_result_tx,
            stats,
        );

        vad_to_asr_tx.send(test_vad_to_asr_message(Ok(AsrPipelineMessage::Segment(
            AsrSegmentJob {
                job_index: 0,
                job_audio_seconds: 0.01,
                speech_annotations: Vec::new(),
                segment_index: 0,
                segment_count: 1,
                offset: DurationMs(0),
                waveform: Waveform::new(vec![0.1; 160], 16_000),
            },
        ))))?;
        tokio::task::yield_now().await;
        vad_to_asr_tx.send(test_vad_to_asr_message(Ok(AsrPipelineMessage::Job(
            AsrJob {
                index: 1,
                prepared: VadPreparation::Disabled,
                waveform: Waveform::new(vec![0.2; 160], 16_000),
            },
        ))))?;
        drop(vad_to_asr_tx);

        let mut outcomes = Vec::new();
        while let Some(outcome) = asr_to_result_rx.recv().await {
            outcomes.push(outcome);
        }
        handle.await?;
        outcomes.sort_unstable_by_key(|outcome| outcome.index);

        assert_eq!(
            outcomes
                .iter()
                .map(|outcome| outcome.index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(outcomes.iter().all(|outcome| outcome.result.is_ok()));
        Ok(())
    }

    #[tokio::test]
    async fn asr_worker_does_not_drop_segment_after_job_microbatch() -> Result<()> {
        let asr = Arc::new(BatchRecordingAsr::default());
        let pipeline = Arc::new(OfflinePipeline {
            vad: None,
            asr: asr.clone(),
        });
        let (vad_to_asr_tx, vad_to_asr_rx) = mpsc::unbounded_channel();
        let (asr_to_result_tx, mut asr_to_result_rx) = mpsc::channel(4);
        let stats = Arc::new(Mutex::new(AsrStageStats::default()));
        let handle = spawn_asr_worker(
            pipeline,
            AsrOptions::default(),
            60.0,
            vad_to_asr_rx,
            asr_to_result_tx,
            stats,
        );

        vad_to_asr_tx.send(test_vad_to_asr_message(Ok(AsrPipelineMessage::Job(
            AsrJob {
                index: 0,
                prepared: VadPreparation::Disabled,
                waveform: Waveform::new(vec![0.2; 160], 16_000),
            },
        ))))?;
        tokio::task::yield_now().await;
        vad_to_asr_tx.send(test_vad_to_asr_message(Ok(AsrPipelineMessage::Segment(
            AsrSegmentJob {
                job_index: 1,
                job_audio_seconds: 0.01,
                speech_annotations: Vec::new(),
                segment_index: 0,
                segment_count: 1,
                offset: DurationMs(0),
                waveform: Waveform::new(vec![0.1; 160], 16_000),
            },
        ))))?;
        drop(vad_to_asr_tx);

        let mut outcomes = Vec::new();
        while let Some(outcome) = asr_to_result_rx.recv().await {
            outcomes.push(outcome);
        }
        handle.await?;
        outcomes.sort_unstable_by_key(|outcome| outcome.index);

        assert_eq!(
            outcomes
                .iter()
                .map(|outcome| outcome.index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(outcomes.iter().all(|outcome| outcome.result.is_ok()));
        Ok(())
    }

    #[test]
    fn duplicate_transcribe_inputs_keep_one_canonical_job() {
        let inputs = vec![
            TranscribeInput {
                index: 10,
                source: AudioSource::Url("file:///same.wav".to_string()),
            },
            TranscribeInput {
                index: 11,
                source: AudioSource::Url("file:///other.wav".to_string()),
            },
            TranscribeInput {
                index: 12,
                source: AudioSource::Url("file:///same.wav".to_string()),
            },
        ];

        let deduped = deduplicate_transcribe_inputs(inputs);

        assert_eq!(
            deduped
                .unique_inputs
                .iter()
                .map(|input| input.index)
                .collect::<Vec<_>>(),
            vec![10, 11]
        );
        assert_eq!(deduped.duplicate_indices, vec![(12, 10)]);
    }

    #[test]
    fn local_file_urls_with_same_content_keep_one_canonical_job() {
        let dir =
            std::env::temp_dir().join(format!("vasr-dedup-test-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let first = dir.join("first.wav");
        let second = dir.join("second.wav");
        std::fs::write(&first, b"same audio bytes").expect("write first file");
        std::fs::write(&second, b"same audio bytes").expect("write second file");

        let inputs = vec![
            TranscribeInput {
                index: 0,
                source: AudioSource::Url(format!("file://{}", first.display())),
            },
            TranscribeInput {
                index: 1,
                source: AudioSource::Url(format!("file://{}", second.display())),
            },
        ];

        let deduped = deduplicate_transcribe_inputs(inputs);

        assert_eq!(deduped.unique_inputs.len(), 1);
        assert_eq!(deduped.unique_inputs[0].index, 0);
        assert_eq!(deduped.duplicate_indices, vec![(1, 0)]);

        std::fs::remove_dir_all(dir).expect("remove temp dir");
    }

    #[test]
    fn duplicate_transcribe_outcomes_are_expanded_to_original_indices() {
        let mut timeline = Timeline::new("canonical");
        timeline.push(Annotation::new(
            Default::default(),
            AnnotationPayload::Transcription(TextSpan::new("same result")),
            AnnotationSource::Model("fake_batch_asr".to_string()),
            AnnotationStatus::Final,
        ));
        let outcomes = vec![
            TranscribeItemOutcome {
                index: 10,
                result: Ok(timeline),
                audio_seconds: 1.25,
                bad_component: None,
            },
            TranscribeItemOutcome {
                index: 11,
                result: Err(anyhow::anyhow!("loader failed")),
                audio_seconds: 0.0,
                bad_component: Some("loader"),
            },
        ];

        let expanded = expand_duplicate_outcomes(outcomes, &[(12, 10), (13, 11)]);

        assert_eq!(
            expanded
                .iter()
                .map(|outcome| outcome.index)
                .collect::<Vec<_>>(),
            vec![10, 11, 12, 13]
        );
        assert_eq!(expanded[2].audio_seconds, 1.25);
        assert!(
            expanded[2]
                .result
                .as_ref()
                .is_ok_and(|timeline| { timeline.transcript().text.trim() == "same result" })
        );
        assert_eq!(expanded[3].bad_component, Some("loader"));
        assert!(expanded[3].result.is_err());
    }

    #[test]
    fn stage_metrics_calculate_speedup_and_rtf() {
        let metrics = stage_metrics(10.0, 2.0);
        assert_eq!(metrics.speedup, 5.0);
        assert_eq!(metrics.rtf, 0.2);

        let empty_audio = stage_metrics(0.0, 2.0);
        assert_eq!(empty_audio.speedup, 0.0);
        assert_eq!(empty_audio.rtf, 0.0);
    }

    fn prepared_with_segments(ranges: &[(u64, u64)]) -> VadPrepared {
        let mut speech_annotations = Vec::new();
        let mut segments = Vec::new();
        let mut slices = Vec::new();
        for &(start, end) in ranges {
            let range = TimeRange::new(DurationMs(start), DurationMs(end));
            speech_annotations.push(Annotation::new(
                range,
                AnnotationPayload::Speech,
                AnnotationSource::Model("fake_vad".to_string()),
                AnnotationStatus::Final,
            ));
            segments.push(VadSegment {
                range,
                probability: 0.9,
            });
            let sample_count = ((end - start) as usize).max(1) * 16;
            slices.push(Waveform::new(vec![0.1; sample_count], 16_000));
        }
        VadPrepared {
            speech_annotations,
            segments,
            slices,
        }
    }

    fn test_vad_to_asr_message(
        message: Result<AsrPipelineMessage, TranscribeItemOutcome>,
    ) -> VadToAsrMessage {
        VadToAsrMessage {
            message,
            _queue_memory_permit: None,
        }
    }
}
