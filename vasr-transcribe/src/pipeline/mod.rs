//! Async parallel transcribe: loader, VAD, and ASR stages overlap via channels.

mod dedup;
mod load;
mod memory;
mod registry;
mod result_router;
mod runtime;
#[cfg(feature = "paged-attn")]
mod segment_decode;
#[cfg(feature = "paged-attn")]
mod segment_encode;
#[cfg(feature = "paged-attn")]
mod segment_prefill;
mod stages;
mod stats;
mod submit;
mod vad;

use self::dedup::*;
use self::load::*;
use self::memory::*;
use self::registry::*;
use self::result_router::*;
use self::runtime::*;
pub(crate) use self::runtime::{PipelineLimits, SourceReservation, source_memory_budget};
#[cfg(feature = "paged-attn")]
use self::segment_decode::*;
#[cfg(feature = "paged-attn")]
use self::segment_encode::*;
#[cfg(feature = "paged-attn")]
use self::segment_prefill::*;
use self::stages::*;
use self::stats::*;
use self::submit::*;
use self::vad::*;

use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Result;
#[cfg(feature = "paged-attn")]
use anyhow::bail;
use asr_data::{AudioEvent, AudioSource, Timeline, Waveform};
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use vasr::AudioLoader;
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::AudioFeatureEncodeTimings;
#[cfg(feature = "timing")]
use vasr::models::qwen3asr::TranscribeTimings;
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::model::paged_cache_runtime::PagedCacheStats;
use vasr::models::qwen3asr::{
    AudioInput, Qwen3ASR, qwen_outputs_to_timelines, qwen_transcribe_options,
};

use self::vad::VadPreparation;
#[cfg(feature = "paged-attn")]
use self::vad::VadPrepared;
use crate::table::render_titled_table;
use vasr::{AsrOptions, OfflineAsrModel as AsrModel};
use vasr_vad::{VadModel, VadOptions};

const FALLBACK_STAGE_PARALLELISM: usize = 20;
const DEFAULT_LOADED_QUEUE_MEMORY_FRACTION: f64 = 0.20;
const BYTES_PER_MEMORY_PERMIT: usize = 1024 * 1024;
const MIN_LOADED_QUEUE_MEMORY_PERMITS: usize = 1;
const ASR_MICROBATCH_COLLECT_WAIT_MS: u64 = 10;
#[cfg(feature = "paged-attn")]
const ASR_ENCODER_BATCH_COLLECT_WAIT_MS: u64 = 10;
#[cfg(feature = "paged-attn")]
const ASR_PREFILL_COALESCE_WAIT_MS: u64 = 20;
const GLOBAL_DECODE_IDLE_FLUSH_MS: u64 = 10;
const COMPONENT_THROUGHPUT_LOG_INTERVAL: Duration = Duration::from_secs(5);
pub const DEFAULT_ASR_ENCODER_BATCH_MAX_SEGMENTS: usize = 32;
pub const DEFAULT_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS: f64 = 30.0;

#[cfg(feature = "paged-attn")]
fn asr_prefill_coalesce_wait() -> Duration {
    let millis = std::env::var("VASR_ASR_PREFILL_COALESCE_WAIT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(ASR_PREFILL_COALESCE_WAIT_MS);
    Duration::from_millis(millis)
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineWorkerOptions {
    pub num_load_workers: usize,
    pub num_vad_workers: usize,
    pub profile: bool,
    pub profile_window: Duration,
    pub encoder_batch_max_segments: usize,
    pub encoder_batch_max_audio_seconds: f64,
}

impl Default for PipelineWorkerOptions {
    fn default() -> Self {
        Self {
            num_load_workers: 1,
            num_vad_workers: 1,
            profile: false,
            profile_window: COMPONENT_THROUGHPUT_LOG_INTERVAL,
            encoder_batch_max_segments: DEFAULT_ASR_ENCODER_BATCH_MAX_SEGMENTS,
            encoder_batch_max_audio_seconds: DEFAULT_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS,
        }
    }
}

impl PipelineWorkerOptions {
    fn resolved(self) -> ResolvedPipelineWorkerOptions {
        ResolvedPipelineWorkerOptions {
            load_workers: resolve_worker_count(self.num_load_workers),
            vad_workers: resolve_worker_count(self.num_vad_workers),
            encoder_batch_max_segments: self.encoder_batch_max_segments,
            encoder_batch_max_audio_seconds: self.encoder_batch_max_audio_seconds,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedPipelineWorkerOptions {
    load_workers: usize,
    vad_workers: usize,
    encoder_batch_max_segments: usize,
    encoder_batch_max_audio_seconds: f64,
}

#[derive(Debug, Clone)]
struct ParallelTranscribeOptions {
    asr_options: AsrOptions,
    vad_options: VadOptions,
}

impl From<AsrOptions> for ParallelTranscribeOptions {
    fn from(asr_options: AsrOptions) -> Self {
        Self {
            asr_options,
            vad_options: VadOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TranscribeItemTiming {
    pub loader_seconds: f64,
    pub vad_seconds: f64,
    pub asr_seconds: f64,
    pub audio_encoder_seconds: f64,
    pub audio_encoder_prepare_seconds: f64,
    pub audio_encoder_stack_seconds: f64,
    pub audio_encoder_forward_seconds: f64,
    pub audio_encoder_d2h_seconds: f64,
    pub prefill_seconds: f64,
    pub prefill_inputs_seconds: f64,
    pub prefill_rope_seconds: f64,
    pub prefill_metadata_seconds: f64,
    pub prefill_mask_seconds: f64,
    pub prefill_forward_seconds: f64,
    pub prefill_gather_seconds: f64,
    pub prefill_decode_setup_seconds: f64,
    pub prefill_argmax_seconds: f64,
    pub decode_seconds: f64,
    pub decode_token_tensor_seconds: f64,
    pub decode_embed_seconds: f64,
    pub decode_position_seconds: f64,
    pub decode_metadata_seconds: f64,
    pub decode_graph_replay_seconds: f64,
    pub decode_forward_seconds: f64,
    pub decode_pre_argmax_sync_seconds: f64,
    pub decode_argmax_seconds: f64,
    pub tokenizer_decode_seconds: f64,
}

impl TranscribeItemTiming {
    pub(crate) fn add_scaled(&mut self, other: TranscribeItemTiming, scale: f64) {
        self.loader_seconds += other.loader_seconds * scale;
        self.vad_seconds += other.vad_seconds * scale;
        self.asr_seconds += other.asr_seconds * scale;
        self.audio_encoder_seconds += other.audio_encoder_seconds * scale;
        self.audio_encoder_prepare_seconds += other.audio_encoder_prepare_seconds * scale;
        self.audio_encoder_stack_seconds += other.audio_encoder_stack_seconds * scale;
        self.audio_encoder_forward_seconds += other.audio_encoder_forward_seconds * scale;
        self.audio_encoder_d2h_seconds += other.audio_encoder_d2h_seconds * scale;
        self.prefill_seconds += other.prefill_seconds * scale;
        self.prefill_inputs_seconds += other.prefill_inputs_seconds * scale;
        self.prefill_rope_seconds += other.prefill_rope_seconds * scale;
        self.prefill_metadata_seconds += other.prefill_metadata_seconds * scale;
        self.prefill_mask_seconds += other.prefill_mask_seconds * scale;
        self.prefill_forward_seconds += other.prefill_forward_seconds * scale;
        self.prefill_gather_seconds += other.prefill_gather_seconds * scale;
        self.prefill_decode_setup_seconds += other.prefill_decode_setup_seconds * scale;
        self.prefill_argmax_seconds += other.prefill_argmax_seconds * scale;
        self.decode_seconds += other.decode_seconds * scale;
        self.decode_token_tensor_seconds += other.decode_token_tensor_seconds * scale;
        self.decode_embed_seconds += other.decode_embed_seconds * scale;
        self.decode_position_seconds += other.decode_position_seconds * scale;
        self.decode_metadata_seconds += other.decode_metadata_seconds * scale;
        self.decode_graph_replay_seconds += other.decode_graph_replay_seconds * scale;
        self.decode_forward_seconds += other.decode_forward_seconds * scale;
        self.decode_pre_argmax_sync_seconds += other.decode_pre_argmax_sync_seconds * scale;
        self.decode_argmax_seconds += other.decode_argmax_seconds * scale;
        self.tokenizer_decode_seconds += other.tokenizer_decode_seconds * scale;
    }
}

#[derive(Clone)]
pub struct AsyncTranscribePipeline {
    loader: AudioLoader,
    vad: Option<Arc<dyn VadModel>>,
    asr: Arc<dyn AsrModel>,
    parallel_options: ParallelTranscribeOptions,
    worker_options: PipelineWorkerOptions,
}

#[derive(Debug, Clone)]
pub struct TranscribeInput {
    pub index: usize,
    pub source: AudioSource,
    pub mono: bool,
}

#[derive(Debug)]
pub struct TranscribeItemOutcome {
    pub index: usize,
    pub channel: &'static str,
    pub result: Result<Timeline>,
    pub audio_seconds: f64,
    pub timing: TranscribeItemTiming,
    pub spans: TranscribeItemSpans,
    pub bad_component: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TranscribeItemSpans {
    pub loader: Option<StageSpan>,
    pub vad: Option<StageSpan>,
    pub asr: Option<StageSpan>,
    pub audio_encoder: Option<StageSpan>,
    pub prefill: Option<StageSpan>,
    pub decode: Option<StageSpan>,
}

#[derive(Debug, Clone, Copy)]
pub struct StageSpan {
    pub start: Instant,
    pub end: Instant,
}

impl StageSpan {
    pub(crate) fn new(start: Instant, end: Instant) -> Self {
        Self { start, end }
    }

    pub(crate) fn try_new(start: Instant, end: Instant) -> Option<Self> {
        (end >= start).then_some(Self { start, end })
    }

    pub(crate) fn seconds(self) -> f64 {
        self.end.duration_since(self.start).as_secs_f64()
    }
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

impl AsyncTranscribePipeline {
    pub fn new(
        loader: AudioLoader,
        vad: Option<Arc<dyn VadModel>>,
        asr: Arc<dyn AsrModel>,
        asr_options: AsrOptions,
    ) -> Self {
        Self {
            loader,
            vad,
            asr,
            parallel_options: ParallelTranscribeOptions::from(asr_options),
            worker_options: PipelineWorkerOptions::default(),
        }
    }

    pub fn with_vad_options(mut self, vad_options: VadOptions) -> Self {
        self.parallel_options.vad_options = vad_options;
        self
    }

    pub fn with_worker_options(mut self, worker_options: PipelineWorkerOptions) -> Self {
        self.worker_options = worker_options;
        self
    }

    pub async fn run_source(&self, index: usize, source: AudioSource) -> TranscribeItemOutcome {
        let outcomes = self
            .run(vec![TranscribeInput {
                index,
                source,
                mono: true,
            }])
            .await;
        outcomes
            .into_iter()
            .next()
            .unwrap_or(TranscribeItemOutcome {
                index,
                channel: "mono",
                result: Err(anyhow::anyhow!("run returned no outcome")),
                audio_seconds: 0.0,
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
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

        let worker_options = self.worker_options.resolved();
        let loader_parallelism = worker_options.load_workers;
        let vad_parallelism = worker_options.vad_workers;
        let loaded_queue_memory = loaded_queue_memory_budget();
        let vad_to_asr_queue_memory = loaded_queue_memory_budget();
        tracing::info!(
            target: "vasr_transcribe::pipeline",
            "pipeline config | loader_workers={} vad_workers={} encoder_batch_segments={} encoder_batch_audio_seconds={:.1} loaded_queue_mem={:.1}GiB segment_queue_mem={:.1}GiB encoder_request_queue=unbounded feature_queue=unbounded",
            loader_parallelism,
            vad_parallelism,
            worker_options.encoder_batch_max_segments,
            worker_options.encoder_batch_max_audio_seconds,
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

        let loader = self.loader;
        let loader_handle = spawn_loader_worker(
            unique_inputs,
            loader,
            loader_parallelism,
            loaded_queue_memory,
            loader_to_vad_tx,
            Arc::clone(&loader_stats),
        );

        let vad = self.vad.clone();
        let vad_options = self.parallel_options.vad_options.clone();
        let vad_handle = spawn_vad_worker(
            vad,
            vad_options,
            vad_parallelism,
            loader_to_vad_rx,
            vad_to_asr_tx,
            vad_to_asr_queue_memory,
            None,
            Arc::clone(&vad_stats),
        );

        let asr = Arc::clone(&self.asr);
        let asr_options = self.parallel_options.asr_options.clone();
        let asr_handle = spawn_asr_worker(
            asr,
            asr_options,
            vad_to_asr_rx,
            asr_to_result_tx,
            Arc::clone(&asr_stats),
            None,
            None,
            None,
            None,
            worker_options.encoder_batch_max_segments,
            worker_options.encoder_batch_max_audio_seconds,
        );

        let mut outcomes = Vec::with_capacity(job_count);
        while let Some(outcome) = asr_to_result_rx.recv().await {
            outcomes.push(outcome.into_public_outcome());
        }

        loader_handle.abort();
        vad_handle.abort();
        asr_handle.abort();
        let _ = loader_handle.await;
        let _ = vad_handle.await;
        let _ = asr_handle.await;

        let outcomes = expand_duplicate_outcomes(outcomes, &duplicate_indices);
        log_pipeline(job_count, &outcomes, pipeline_start.elapsed().as_secs_f64());
        outcomes
    }
}

#[derive(Clone)]
pub struct GlobalAsyncTranscribePipeline {
    submit_tx: mpsc::UnboundedSender<GlobalTranscribeRequest>,
    failed: Arc<AtomicBool>,
    fatal_error_tx: watch::Sender<Option<String>>,
    #[cfg(any(feature = "paged-attn", feature = "cuda-graph"))]
    asr: Arc<dyn AsrModel>,
    service_id: Arc<Mutex<String>>,
    supervisor: Arc<PipelineSupervisor>,
    next_request_id: Arc<AtomicU64>,
}

impl Drop for GlobalAsyncTranscribePipeline {
    fn drop(&mut self) {
        // Only public pipeline clones own submit senders. Stage tasks may
        // temporarily retain the supervisor while waiting for admission.
        if self.submit_tx.strong_count() == 1 {
            self.supervisor.stop();
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PipelineMemorySnapshot {
    #[cfg(feature = "paged-attn")]
    pub paged_cache: Option<PagedCacheStats>,
    #[cfg(all(feature = "paged-attn", feature = "cuda-graph"))]
    pub cuda_decode_graphs: Option<usize>,
    #[cfg(feature = "cuda-graph")]
    pub cuda_memory: Option<PipelineCudaMemorySnapshot>,
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "cuda-graph")]
pub struct PipelineCudaMemorySnapshot {
    pub used_bytes: usize,
    pub free_bytes: usize,
    pub total_bytes: usize,
    pub pool_used_bytes: Option<usize>,
    pub pool_reserved_bytes: Option<usize>,
    pub default_pool_used_bytes: Option<usize>,
    pub default_pool_reserved_bytes: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
#[cfg(feature = "cuda-graph")]
pub struct PipelineCudaTrimSnapshot {
    pub before: Option<PipelineCudaMemorySnapshot>,
    pub after: Option<PipelineCudaMemorySnapshot>,
}

struct PendingGlobalRequest {
    log_id: String,
    remaining: usize,
    expected: std::collections::HashSet<(usize, &'static str)>,
    duplicate_indices: Vec<(usize, usize)>,
    internal_indices: Vec<usize>,
    item_routes: HashMap<(usize, &'static str), usize>,
    outcomes: Vec<TranscribeItemOutcome>,
    stages: RequestStageProgress,
    response_tx: oneshot::Sender<Vec<TranscribeItemOutcome>>,
}

fn global_pipeline_failure_outcomes(
    indices: impl IntoIterator<Item = usize>,
    error: &str,
) -> Vec<TranscribeItemOutcome> {
    indices
        .into_iter()
        .map(|index| TranscribeItemOutcome {
            index,
            channel: "mono",
            result: Err(anyhow::anyhow!(error.to_owned())),
            audio_seconds: 0.0,
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            bad_component: Some("recognizer"),
        })
        .collect()
}

#[cfg(feature = "cuda-graph")]
fn snapshot_qwen_cuda_memory(model: &Qwen3ASR) -> Option<PipelineCudaMemorySnapshot> {
    vasr::vasr_paged_attn::cuda_graph::cuda_memory_snapshot(model.device()).map(|memory| {
        PipelineCudaMemorySnapshot {
            used_bytes: memory.used(),
            free_bytes: memory.free,
            total_bytes: memory.total,
            pool_used_bytes: memory.pool_used_current,
            pool_reserved_bytes: memory.pool_reserved_current,
            default_pool_used_bytes: memory.default_pool_used_current,
            default_pool_reserved_bytes: memory.default_pool_reserved_current,
        }
    })
}

impl GlobalAsyncTranscribePipeline {
    pub fn start(
        loader: AudioLoader,
        vad: Option<Arc<dyn VadModel>>,
        asr: Arc<dyn AsrModel>,
        asr_options: AsrOptions,
        vad_options: VadOptions,
        worker_options: PipelineWorkerOptions,
    ) -> Self {
        // A zero override means automatic, matching the scheduler's default
        // window. CLI startup supplies its measured limits separately.
        let capacity = if asr_options.decode_max_seqs == 0 {
            8
        } else {
            asr_options.decode_max_seqs
        };
        let slots = asr_options
            .execution
            .map(|policy| policy.active_slots)
            .unwrap_or(capacity)
            .clamp(1, capacity);
        Self::start_with_limits(
            loader,
            vad,
            asr,
            asr_options,
            vad_options,
            worker_options,
            PipelineLimits {
                active_slots: slots,
                safe_capacity: capacity,
                ..Default::default()
            },
        )
    }

    pub(crate) fn start_with_limits(
        loader: AudioLoader,
        vad: Option<Arc<dyn VadModel>>,
        asr: Arc<dyn AsrModel>,
        asr_options: AsrOptions,
        vad_options: VadOptions,
        worker_options: PipelineWorkerOptions,
        limits: PipelineLimits,
    ) -> Self {
        let parallel_options = ParallelTranscribeOptions {
            asr_options,
            vad_options,
        };
        let profile = worker_options.profile;
        let profile_window = worker_options.profile_window;
        let worker_options = worker_options.resolved();
        let loader_parallelism = worker_options.load_workers;
        let vad_parallelism = worker_options.vad_workers;
        let loaded_queue_memory = loaded_queue_memory_budget();
        let vad_to_asr_queue_memory = loaded_queue_memory_budget();
        tracing::debug!(
            target: "vasr_transcribe::pipeline",
            "global pipeline config | mode=serve loader_workers={} vad_workers={} encoder_batch_segments={} encoder_batch_audio_seconds={:.1} loaded_queue_mem={:.1}GiB segment_queue_mem={:.1}GiB encoder_request_queue=unbounded feature_queue=unbounded decode_idle_flush={}ms",
            loader_parallelism,
            vad_parallelism,
            worker_options.encoder_batch_max_segments,
            worker_options.encoder_batch_max_audio_seconds,
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
        let failed = Arc::new(AtomicBool::new(false));
        let (fatal_error_tx, _fatal_error_rx) = watch::channel(None);
        let service_id = Arc::new(Mutex::new(String::from("service")));
        let registry = RequestRegistry::new(pending_requests.clone(), job_routes.clone());
        let supervisor =
            PipelineSupervisor::new(registry, failed.clone(), fatal_error_tx.clone(), limits);
        if supervisor.debug_enabled() {
            let weak = Arc::downgrade(&supervisor);
            let reporter = tokio::spawn(async move {
                let mut ticks = tokio::time::interval(Duration::from_secs(1));
                loop {
                    ticks.tick().await;
                    let Some(supervisor) = weak.upgrade() else {
                        break;
                    };
                    supervisor.log_snapshot();
                }
            });
            supervisor.supervise("lifecycle reporter", reporter);
        }
        let loader_stats = Arc::new(Mutex::new(LoaderStageStats::default()));
        let vad_stats = Arc::new(Mutex::new(VadStageStats::default()));
        let asr_stats = Arc::new(Mutex::new(AsrStageStats::default()));

        if profile {
            let reporter = spawn_component_throughput_reporter(
                Arc::clone(&service_id),
                Arc::clone(&loader_stats),
                Arc::clone(&vad_stats),
                Arc::clone(&asr_stats),
                profile_window,
            );
            supervisor.supervise("profile reporter", reporter);
        }

        let submit_handle = spawn_supervised_submit_worker(
            submit_rx,
            load_job_tx,
            Arc::clone(&pending_requests),
            Arc::clone(&job_routes),
            Arc::clone(&next_request_id),
            Arc::clone(&next_job_id),
            Arc::downgrade(&supervisor),
        );
        supervisor.supervise("submit", submit_handle);
        let loader_handle = spawn_global_loader_worker(
            loader,
            loader_parallelism,
            loaded_queue_memory,
            load_job_rx,
            loader_to_vad_tx,
            Arc::clone(&pending_requests),
            Arc::clone(&job_routes),
            Arc::clone(&next_job_id),
            loader_stats,
        );
        supervisor.supervise("load", loader_handle);
        let vad_handle = spawn_vad_worker(
            vad,
            parallel_options.vad_options.clone(),
            vad_parallelism,
            loader_to_vad_rx,
            vad_to_asr_tx,
            vad_to_asr_queue_memory,
            Some(Arc::clone(&pending_requests)),
            vad_stats,
        );
        supervisor.supervise("VAD", vad_handle);
        let asr_handle = spawn_supervised_asr_worker(
            Arc::clone(&asr),
            parallel_options.asr_options.clone(),
            vad_to_asr_rx,
            asr_to_result_tx,
            asr_stats,
            Some(std::time::Duration::from_millis(
                GLOBAL_DECODE_IDLE_FLUSH_MS,
            )),
            Some(Arc::clone(&pending_requests)),
            Some(Arc::clone(&failed)),
            Some(fatal_error_tx.clone()),
            worker_options.encoder_batch_max_segments,
            worker_options.encoder_batch_max_audio_seconds,
            Some(Arc::downgrade(&supervisor)),
        );
        supervisor.supervise("ASR", asr_handle);
        let router_handle =
            spawn_result_router_with_registry(asr_to_result_rx, supervisor.registry.clone());
        supervisor.supervise("result router", router_handle);

        Self {
            submit_tx,
            failed,
            fatal_error_tx,
            #[cfg(any(feature = "paged-attn", feature = "cuda-graph"))]
            asr,
            service_id,
            supervisor,
            next_request_id,
        }
    }

    pub(crate) async fn admit_request(&self) -> Result<OwnedSemaphorePermit> {
        self.supervisor
            .requests
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| anyhow::anyhow!("global transcribe pipeline stopped"))
    }
    pub(crate) async fn reserve_source(&self, bytes: usize) -> Result<Arc<SourceReservation>> {
        self.supervisor.reserve_source(bytes).await
    }
    pub(crate) fn shutdown(&self) {
        self.supervisor.stop();
    }

    pub fn set_service_id(&self, service_id: String) {
        if let Ok(mut current) = self.service_id.lock() {
            *current = service_id;
        }
    }

    /// Subscribe to unrecoverable background pipeline failures.
    ///
    /// The global decoder cannot be restarted in place. HTTP servers should
    /// terminate with a non-zero status when this channel publishes an error,
    /// allowing their process supervisor to start a clean instance.
    pub fn subscribe_fatal_errors(&self) -> watch::Receiver<Option<String>> {
        self.fatal_error_tx.subscribe()
    }

    pub fn memory_snapshot(&self) -> PipelineMemorySnapshot {
        #[cfg(any(feature = "paged-attn", feature = "cuda-graph"))]
        let qwen = self.asr.as_any().downcast_ref::<Qwen3ASR>();
        PipelineMemorySnapshot {
            #[cfg(feature = "paged-attn")]
            paged_cache: qwen.and_then(Qwen3ASR::paged_cache_stats),
            #[cfg(all(feature = "paged-attn", feature = "cuda-graph"))]
            cuda_decode_graphs: qwen.and_then(Qwen3ASR::cuda_decode_graph_count),
            #[cfg(feature = "cuda-graph")]
            cuda_memory: qwen.and_then(snapshot_qwen_cuda_memory),
        }
    }

    #[cfg(feature = "cuda-graph")]
    pub fn trim_cuda_mempool(&self) -> Option<PipelineCudaTrimSnapshot> {
        let qwen = self.asr.as_any().downcast_ref::<Qwen3ASR>()?;
        let before = snapshot_qwen_cuda_memory(qwen);
        vasr::vasr_quant::trim_cuda_workspaces();
        vasr::vasr_paged_attn::cuda_graph::cuda_trim_device_mem_pools(qwen.device());
        let after = snapshot_qwen_cuda_memory(qwen);
        Some(PipelineCudaTrimSnapshot { before, after })
    }

    pub async fn run(&self, inputs: Vec<TranscribeInput>) -> Vec<TranscribeItemOutcome> {
        self.run_with_log_id(String::new(), inputs).await
    }

    pub async fn run_with_log_id(
        &self,
        log_id: String,
        mut inputs: Vec<TranscribeInput>,
    ) -> Vec<TranscribeItemOutcome> {
        // Caller indices need not be unique. Only this compatibility wrapper
        // remaps aliases; the HTTP path already assigns unique input positions.
        let mut seen = std::collections::HashSet::with_capacity(inputs.len());
        let aliased = inputs.iter().any(|input| !seen.insert(input.index));
        let original_indices = aliased.then(|| {
            inputs
                .iter_mut()
                .enumerate()
                .map(|(position, input)| {
                    let original = input.index;
                    input.index = position;
                    original
                })
                .collect::<Vec<_>>()
        });
        let mut outcomes = self.run_with_source_admission(log_id, inputs).await;
        if let Some(original_indices) = original_indices {
            for outcome in &mut outcomes {
                outcome.index = original_indices[outcome.index];
            }
            outcomes.sort_by_key(|outcome| outcome.index);
        }
        outcomes
    }

    async fn run_with_source_admission(
        &self,
        log_id: String,
        inputs: Vec<TranscribeInput>,
    ) -> Vec<TranscribeItemOutcome> {
        let indices = inputs.iter().map(|input| input.index).collect::<Vec<_>>();
        let _admission = match self.admit_request().await {
            Ok(permit) => permit,
            Err(error) => return global_pipeline_failure_outcomes(indices, &error.to_string()),
        };
        let Some(bytes) = inputs.iter().try_fold(0usize, |bytes, input| {
            bytes.checked_add(source_size(&input.source))
        }) else {
            return global_pipeline_failure_outcomes(
                indices,
                "request encoded source budget size overflow",
            );
        };
        let reservation = match self.reserve_source(bytes).await {
            Ok(reservation) => reservation,
            Err(error) => return global_pipeline_failure_outcomes(indices, &error.to_string()),
        };
        self.run_admitted(log_id, inputs, reservation).await
    }

    pub(crate) async fn run_admitted(
        &self,
        log_id: String,
        inputs: Vec<TranscribeInput>,
        source_memory: Arc<SourceReservation>,
    ) -> Vec<TranscribeItemOutcome> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let indices = inputs.iter().map(|input| input.index).collect::<Vec<_>>();
        if self.failed.load(Ordering::Acquire) {
            return global_pipeline_failure_outcomes(indices, "global transcribe pipeline stopped");
        }
        let request_id = PipelineRequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let cancellation = Arc::new(CancellationSignal::default());
        let _guard = RequestCancellation {
            registry: self.supervisor.registry.clone(),
            id: request_id,
            signal: cancellation.clone(),
        };
        // The response receiver's closure is visible before registration as well
        // as after it, so dropping a queued request never resurrects metadata.
        let (response_tx, response_rx) = oneshot::channel();
        if self
            .submit_tx
            .send(GlobalTranscribeRequest {
                log_id,
                inputs,
                response_tx,
                request_id: Some(request_id),
                source_memory: Some(source_memory),
                cancellation: Some(cancellation),
            })
            .is_err()
        {
            return global_pipeline_failure_outcomes(indices, "global transcribe pipeline stopped");
        }
        response_rx.await.unwrap_or_else(|_| {
            global_pipeline_failure_outcomes(indices, "global transcribe pipeline response dropped")
        })
    }
}

fn load_transcribe_waveforms(
    loader: &AudioLoader,
    source: &AudioSource,
    mono: bool,
) -> Result<Vec<(&'static str, Waveform)>> {
    let waveform = loader.load_raw(source)?;
    if mono || waveform.channels <= 1 {
        return Ok(vec![("mono", waveform.to_mono()?)]);
    }
    Ok(vec![
        ("left", waveform.channel(0)?),
        ("right", waveform.channel(1)?),
    ])
}

#[derive(Debug)]
struct LoadedJob {
    _item_permit: Option<Arc<OwnedSemaphorePermit>>,
    index: usize,
    request_item: Option<RequestItemRef>,
    channel: &'static str,
    waveform: Waveform,
    timing: TranscribeItemTiming,
    spans: TranscribeItemSpans,
    _queue_memory_permit: Option<OwnedSemaphorePermit>,
}

#[derive(Debug)]
struct AsrJob {
    _item_permit: Option<Arc<OwnedSemaphorePermit>>,
    index: usize,
    request_item: Option<RequestItemRef>,
    channel: &'static str,
    timing: TranscribeItemTiming,
    spans: TranscribeItemSpans,
    prepared: VadPreparation,
    waveform: Waveform,
}

/// One VAD speech segment enqueued for ASR continuous micro-batching.
#[derive(Debug, Clone)]
#[cfg(feature = "paged-attn")]
struct AsrSegmentJob {
    _item_permit: Option<Arc<OwnedSemaphorePermit>>,
    job_index: usize,
    request_item: Option<RequestItemRef>,
    channel: &'static str,
    timing: TranscribeItemTiming,
    spans: TranscribeItemSpans,
    job_audio_seconds: f64,
    speech_annotations: Vec<AudioEvent>,
    segment_index: usize,
    segment_count: usize,
    offset: usize,
    waveform: Waveform,
}

#[derive(Debug)]
enum AsrPipelineMessage {
    Job(AsrJob),
    #[cfg(feature = "paged-attn")]
    Segment(AsrSegmentJob),
}

#[derive(Debug)]
struct VadToAsrMessage {
    message: Result<AsrPipelineMessage, TranscribeItemOutcome>,
    _queue_memory_permit: Option<OwnedSemaphorePermit>,
}

#[cfg(feature = "paged-attn")]
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
            _item_permit: job._item_permit.clone(),
            job_index: job.index,
            request_item: job.request_item,
            channel: job.channel,
            timing: job.timing,
            spans: job.spans,
            job_audio_seconds: job.waveform.duration_seconds(),
            speech_annotations: if segment_index == 0 {
                speech_annotations.clone()
            } else {
                Vec::new()
            },
            segment_index,
            segment_count,
            offset: segment.range.start_ms,
            waveform,
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn spawn_asr_worker(
    asr: Arc<dyn AsrModel>,
    asr_options: AsrOptions,
    vad_to_asr_rx: mpsc::UnboundedReceiver<VadToAsrMessage>,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    decode_idle_flush: Option<std::time::Duration>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    failed: Option<Arc<AtomicBool>>,
    fatal_error_tx: Option<watch::Sender<Option<String>>>,
    encoder_batch_max_segments: usize,
    encoder_batch_max_audio_seconds: f64,
) -> JoinHandle<Result<()>> {
    spawn_supervised_asr_worker(
        asr,
        asr_options,
        vad_to_asr_rx,
        asr_to_result_tx,
        stats,
        decode_idle_flush,
        pending_requests,
        failed,
        fatal_error_tx,
        encoder_batch_max_segments,
        encoder_batch_max_audio_seconds,
        None,
    )
}
#[allow(clippy::too_many_arguments)]
fn spawn_supervised_asr_worker(
    asr: Arc<dyn AsrModel>,
    asr_options: AsrOptions,
    mut vad_to_asr_rx: mpsc::UnboundedReceiver<VadToAsrMessage>,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    decode_idle_flush: Option<std::time::Duration>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    _failed: Option<Arc<AtomicBool>>,
    _fatal_error_tx: Option<watch::Sender<Option<String>>>,
    encoder_batch_max_segments: usize,
    encoder_batch_max_audio_seconds: f64,
    supervisor: Option<std::sync::Weak<PipelineSupervisor>>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        #[cfg(not(feature = "paged-attn"))]
        let _ = (
            decode_idle_flush,
            &pending_requests,
            &_failed,
            &_fatal_error_tx,
            &supervisor,
            encoder_batch_max_segments,
            encoder_batch_max_audio_seconds,
        );
        let mut pending = VecDeque::new();
        #[cfg(feature = "paged-attn")]
        let (asr_to_encoder_tx, asr_to_encoder_rx) = mpsc::unbounded_channel();
        #[cfg(feature = "paged-attn")]
        let (encoder_to_prefill_tx, encoder_to_prefill_rx) = mpsc::unbounded_channel();
        #[cfg(feature = "paged-attn")]
        let nested_handles = if asr.as_any().is::<Qwen3ASR>() {
            let encoder = spawn_segment_encoder_worker(
                Arc::clone(&asr),
                asr_options.clone(),
                asr_to_encoder_rx,
                encoder_to_prefill_tx,
                Arc::clone(&stats),
                pending_requests.clone(),
                encoder_batch_max_segments,
                encoder_batch_max_audio_seconds,
            );
            let (prefill_to_decoder_tx, prefill_to_decoder_rx) = mpsc::unbounded_channel();
            let prefill = spawn_segment_prefill_worker(
                Arc::clone(&asr),
                asr_options.clone(),
                encoder_to_prefill_rx,
                prefill_to_decoder_tx,
                decode_idle_flush,
                pending_requests.clone(),
            );
            let decoder = spawn_segment_decoder_worker(
                Arc::clone(&asr),
                asr_options.clone(),
                prefill_to_decoder_rx,
                asr_to_result_tx.clone(),
                Arc::clone(&stats),
                pending_requests.clone(),
            );
            [
                monitor_nested(&supervisor, "encoder", encoder),
                monitor_nested(&supervisor, "prefill", prefill),
                monitor_nested(&supervisor, "decode", decoder),
            ]
        } else {
            [None, None, None]
        };
        while let Some(envelope) = match pending.pop_front() {
            Some(envelope) => Some(envelope),
            None => vad_to_asr_rx.recv().await,
        } {
            let message = envelope.message;
            match message {
                Err(outcome) => {
                    asr_to_result_tx
                        .send(outcome.into())
                        .await
                        .map_err(|_| anyhow::anyhow!("result channel closed"))?;
                }
                Ok(AsrPipelineMessage::Job(job)) => {
                    let messages =
                        collect_job_microbatch(job, &mut vad_to_asr_rx, &mut pending).await;
                    let mut jobs = Vec::new();
                    for message in messages {
                        match message {
                            Err(outcome) => {
                                asr_to_result_tx
                                    .send(outcome.into())
                                    .await
                                    .map_err(|_| anyhow::anyhow!("result channel closed"))?;
                            }
                            Ok(AsrPipelineMessage::Job(job))
                                if request_is_live(&pending_requests, job.request_item) =>
                            {
                                jobs.push(job)
                            }
                            Ok(AsrPipelineMessage::Job(_)) => {}
                            #[cfg(feature = "paged-attn")]
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
                    let outcomes = blocking_result(
                        tokio::task::spawn_blocking({
                            let asr = Arc::clone(&asr);
                            let asr_options = asr_options.clone();
                            let stats = Arc::clone(&stats);
                            move || {
                                run_asr_jobs_batch(asr.as_ref(), jobs, &asr_options, Some(&stats))
                            }
                        })
                        .await,
                    )?;
                    for outcome in outcomes {
                        asr_to_result_tx
                            .send(outcome)
                            .await
                            .map_err(|_| anyhow::anyhow!("result channel closed"))?;
                    }
                }
                #[cfg(feature = "paged-attn")]
                Ok(AsrPipelineMessage::Segment(segment)) => {
                    if !request_is_live(&pending_requests, segment.request_item) {
                        continue;
                    }
                    let stage_start = Instant::now();
                    let request = SegmentEncodeRequest {
                        segment,
                        stage_start,
                    };
                    send_segment_encode_request(&asr_to_encoder_tx, request)?;
                }
            }
        }

        #[cfg(feature = "paged-attn")]
        drop(asr_to_encoder_tx);
        #[cfg(feature = "paged-attn")]
        for handle in nested_handles.into_iter().flatten() {
            handle
                .await
                .map_err(|error| anyhow::anyhow!("ASR child join failed: {error}"))??;
        }
        Ok(())
    })
}

async fn collect_job_microbatch(
    first: AsrJob,
    vad_to_asr_rx: &mut mpsc::UnboundedReceiver<VadToAsrMessage>,
    _pending: &mut VecDeque<VadToAsrMessage>,
) -> Vec<Result<AsrPipelineMessage, TranscribeItemOutcome>> {
    let mut messages = vec![Ok(AsrPipelineMessage::Job(first))];
    loop {
        match vad_to_asr_rx.try_recv() {
            Ok(envelope) => match envelope.message {
                #[cfg(feature = "paged-attn")]
                Ok(AsrPipelineMessage::Segment(_)) => {
                    _pending.push_back(envelope);
                }
                Ok(AsrPipelineMessage::Job(job)) => {
                    messages.push(Ok(AsrPipelineMessage::Job(job)));
                }
                message => messages.push(message),
            },
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => return messages,
        }
    }
    while let Ok(Some(envelope)) = tokio::time::timeout(
        std::time::Duration::from_millis(ASR_MICROBATCH_COLLECT_WAIT_MS),
        vad_to_asr_rx.recv(),
    )
    .await
    {
        match envelope.message {
            #[cfg(feature = "paged-attn")]
            Ok(AsrPipelineMessage::Segment(_)) => {
                _pending.push_back(envelope);
            }
            Ok(AsrPipelineMessage::Job(job)) => {
                messages.push(Ok(AsrPipelineMessage::Job(job)));
            }
            message => messages.push(message),
        }
    }
    messages
}

fn run_raw_asr_inference(
    asr: &dyn AsrModel,
    waveforms: &[Waveform],
    asr_options: &AsrOptions,
) -> Result<(Vec<Timeline>, Option<TranscribeItemTiming>)> {
    #[cfg(feature = "timing")]
    if let Some(qwen3) = asr.as_any().downcast_ref::<Qwen3ASR>() {
        let result = (|| -> Result<(Vec<Timeline>, Option<TranscribeItemTiming>)> {
            let runtime_options = qwen_transcribe_options(asr_options);
            let audio = waveforms
                .iter()
                .map(|waveform| AudioInput::Waveform {
                    samples: &waveform.samples,
                    sample_rate: waveform.sample_rate,
                })
                .collect::<Vec<_>>();
            let (outputs, timings) = Qwen3ASR::transcribe_timed(qwen3, audio, runtime_options)?;
            let timing = transcribe_timing_seconds(&timings);
            Ok((qwen_outputs_to_timelines(outputs, waveforms)?, Some(timing)))
        })();
        return result;
    }

    asr.transcribe(waveforms, asr_options)
        .map(|timelines| (timelines, None))
}

fn run_asr_jobs_batch(
    asr: &dyn AsrModel,
    jobs: Vec<AsrJob>,
    asr_options: &AsrOptions,
    stats: Option<&Arc<Mutex<AsrStageStats>>>,
) -> Vec<RoutedTranscribeItemOutcome> {
    let mut outcomes = Vec::with_capacity(jobs.len());
    let mut raw_jobs: Vec<(AsrJob, Vec<AudioEvent>)> = Vec::new();

    for mut job in jobs {
        let prepared = std::mem::replace(&mut job.prepared, VadPreparation::Disabled);
        match prepared {
            VadPreparation::Speech(prepared) => {
                let request_item = job.request_item;
                outcomes.push(route_outcome(
                    request_item,
                    outcome_from_result(
                        job.index,
                        job.channel,
                        job.waveform.duration_seconds(),
                        job.timing,
                        job.spans,
                        Err(anyhow::anyhow!(
                            "speech jobs must enter ASR as segment messages"
                        )),
                    ),
                ));
                drop(prepared);
            }
            VadPreparation::NoSpeech => {
                let request_item = job.request_item;
                let outcome = TranscribeItemOutcome {
                    index: job.index,
                    channel: job.channel,
                    result: Ok(Timeline::new(
                        "offline_audio",
                        job.waveform.duration_ms().round() as usize,
                    )),
                    audio_seconds: job.waveform.duration_seconds(),
                    timing: job.timing,
                    spans: job.spans,
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
        let channels: Vec<&'static str> = raw_jobs.iter().map(|(job, _)| job.channel).collect();
        let timings: Vec<TranscribeItemTiming> =
            raw_jobs.iter().map(|(job, _)| job.timing).collect();
        let mut spans: Vec<TranscribeItemSpans> =
            raw_jobs.iter().map(|(job, _)| job.spans).collect();
        let audio_seconds: Vec<f64> = raw_jobs
            .iter()
            .map(|(job, _)| job.waveform.duration_seconds())
            .collect();
        let speech_annotations = raw_jobs
            .iter()
            .map(|(_, annotations)| annotations.clone())
            .collect::<Vec<_>>();
        let total_audio_seconds: f64 = audio_seconds.iter().sum();
        // Extracting the waveforms consumes the jobs. Keep their canonical
        // admission leases until the blocking forward has completed.
        let _item_permits = raw_jobs
            .iter()
            .filter_map(|(job, _)| job._item_permit.clone())
            .collect::<Vec<_>>();
        let waveforms: Vec<Waveform> = raw_jobs.into_iter().map(|(job, _)| job.waveform).collect();
        let stage_start = Instant::now();
        let result = run_raw_asr_inference(asr, &waveforms, asr_options);
        let asr_span = StageSpan::new(stage_start, Instant::now());
        for spans in &mut spans {
            spans.asr = Some(asr_span);
        }
        let elapsed = stage_start.elapsed().as_secs_f64();
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
            Ok((timelines, batch_timing)) if timelines.len() == indices.len() => {
                for (
                    (
                        (
                            ((((index, request_item), channel), audio_seconds), mut timing),
                            mut spans,
                        ),
                        mut timeline,
                    ),
                    annotations,
                ) in indices
                    .into_iter()
                    .zip(request_items)
                    .zip(channels)
                    .zip(audio_seconds)
                    .zip(timings)
                    .zip(spans)
                    .zip(timelines)
                    .zip(speech_annotations)
                {
                    timing.asr_seconds += allocated_batch_seconds(
                        audio_seconds,
                        total_audio_seconds,
                        waveforms.len(),
                        elapsed,
                    );
                    if let Some(batch_timing) = batch_timing {
                        let scale =
                            allocation_scale(audio_seconds, total_audio_seconds, waveforms.len());
                        timing.add_scaled(batch_timing, scale);
                        spans.prefill = Some(merge_optional_span(
                            spans.prefill,
                            prefill_span_from_decode(asr_span, Some(batch_timing)),
                        ));
                    }
                    if !annotations.is_empty()
                        && let Err(err) = vasr::extend_predictions(&mut timeline, annotations)
                    {
                        outcomes.push(route_outcome(
                            request_item,
                            TranscribeItemOutcome {
                                index,
                                channel,
                                result: Err(err),
                                audio_seconds,
                                timing,
                                spans,
                                bad_component: Some("recognizer"),
                            },
                        ));
                        continue;
                    }
                    outcomes.push(route_outcome(
                        request_item,
                        TranscribeItemOutcome {
                            index,
                            channel,
                            result: Ok(timeline),
                            audio_seconds,
                            timing,
                            spans,
                            bad_component: None,
                        },
                    ));
                }
            }
            Ok((timelines, batch_timing)) => {
                let err = anyhow::anyhow!(
                    "ASR batch returned {} timelines for {} jobs",
                    timelines.len(),
                    indices.len()
                );
                for (((((index, request_item), channel), audio_seconds), mut timing), mut spans) in
                    indices
                        .into_iter()
                        .zip(request_items)
                        .zip(channels)
                        .zip(audio_seconds)
                        .zip(timings)
                        .zip(spans)
                {
                    timing.asr_seconds += allocated_batch_seconds(
                        audio_seconds,
                        total_audio_seconds,
                        waveforms.len(),
                        elapsed,
                    );
                    if let Some(batch_timing) = batch_timing {
                        let scale =
                            allocation_scale(audio_seconds, total_audio_seconds, waveforms.len());
                        timing.add_scaled(batch_timing, scale);
                        spans.prefill = Some(merge_optional_span(
                            spans.prefill,
                            prefill_span_from_decode(asr_span, Some(batch_timing)),
                        ));
                    }
                    let outcome = outcome_from_result(
                        index,
                        channel,
                        audio_seconds,
                        timing,
                        spans,
                        Err(anyhow::anyhow!("{err}")),
                    );
                    outcomes.push(route_outcome(request_item, outcome));
                }
            }
            Err(error) => {
                let reason = error.to_string();
                for (((((index, request_item), channel), audio_seconds), mut timing), spans) in
                    indices
                        .into_iter()
                        .zip(request_items)
                        .zip(channels)
                        .zip(audio_seconds)
                        .zip(timings)
                        .zip(spans)
                {
                    timing.asr_seconds += allocated_batch_seconds(
                        audio_seconds,
                        total_audio_seconds,
                        waveforms.len(),
                        elapsed,
                    );
                    let outcome = outcome_from_result(
                        index,
                        channel,
                        audio_seconds,
                        timing,
                        spans,
                        Err(anyhow::anyhow!("{reason}")),
                    );
                    outcomes.push(route_outcome(request_item, outcome));
                }
            }
        }
    }

    outcomes.sort_unstable_by_key(|outcome| outcome.outcome.index);
    outcomes
}

fn outcome_from_result(
    index: usize,
    channel: &'static str,
    audio_seconds: f64,
    timing: TranscribeItemTiming,
    spans: TranscribeItemSpans,
    result: Result<Timeline>,
) -> TranscribeItemOutcome {
    let bad_component = if result.is_err() {
        Some("recognizer")
    } else {
        None
    };
    TranscribeItemOutcome {
        index,
        channel,
        result,
        audio_seconds,
        timing,
        spans,
        bad_component,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lifecycle_tests;

fn source_size(source: &AudioSource) -> usize {
    match source {
        AudioSource::Base64(value) | AudioSource::Url(value) => value.len(),
        AudioSource::EncodedBytes(bytes) | AudioSource::PcmS16Le { bytes, .. } => bytes.len(),
        AudioSource::Path(path) => path.as_os_str().len(),
        AudioSource::ModelScope {
            repo_id,
            file_path,
            revision,
        } => repo_id.len() + file_path.len() + revision.len(),
    }
}
