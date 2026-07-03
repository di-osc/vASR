//! Async parallel transcribe: loader, VAD, and ASR stages overlap via channels.

#[cfg(feature = "paged-attn")]
use std::cell::RefCell;
#[cfg(feature = "paged-attn")]
use std::collections::BTreeMap;
#[cfg(feature = "paged-attn")]
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
#[cfg(feature = "paged-attn")]
use std::sync::mpsc as std_mpsc;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::Result;
#[cfg(feature = "paged-attn")]
use anyhow::bail;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
#[cfg(feature = "timing")]
use vasr::models::qwen3asr::TranscribeTimings;
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::model::paged_cache_runtime::PagedCacheStats;
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::{AudioFeatureEncodeTimings, EncodedAsrBatch};
use vasr::models::qwen3asr::{
    AudioInput, Qwen3ASR, qwen_outputs_to_timelines, qwen_transcribe_options,
};
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::{
    EncodedAsrBatchMetadata, PrefillStageOutput, PreparedAudioBatch, PreparedAudioBatchPoll,
    StreamingMetadataDecoder,
};
use vasr_data::AudioLoader;
#[cfg(feature = "paged-attn")]
use vasr_data::DurationMs;
use vasr_data::{Annotation, AudioSource, Timeline, Waveform};

use crate::runtime::pipeline::{OfflinePipeline, VadPreparation};
#[cfg(feature = "paged-attn")]
use crate::runtime::pipeline::{VadPrepared, offset_annotations};
use crate::runtime::{AsrOptions, ParallelTranscribeOptions, VadOptions};
use crate::table::render_titled_table;

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
}

impl Default for PipelineWorkerOptions {
    fn default() -> Self {
        Self {
            num_load_workers: 1,
            num_vad_workers: 1,
            profile: false,
            profile_window: COMPONENT_THROUGHPUT_LOG_INTERVAL,
        }
    }
}

impl PipelineWorkerOptions {
    fn resolved(self) -> ResolvedPipelineWorkerOptions {
        ResolvedPipelineWorkerOptions {
            load_workers: resolve_worker_count(self.num_load_workers),
            vad_workers: resolve_worker_count(self.num_vad_workers),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedPipelineWorkerOptions {
    load_workers: usize,
    vad_workers: usize,
}

#[derive(Debug, Default)]
struct LoaderStageStats {
    jobs: usize,
    failed: usize,
    audio_seconds: f64,
    compute_seconds: f64,
    worker_compute_seconds: HashMap<usize, f64>,
    wall_seconds: f64,
}

#[derive(Debug, Default)]
struct LoaderTaskStats {
    jobs: usize,
    failed: usize,
    audio_seconds: f64,
    compute_seconds: f64,
    worker_id: usize,
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

#[derive(Debug, Default)]
struct VadStageStats {
    jobs: usize,
    failed: usize,
    no_speech: usize,
    segments: usize,
    audio_seconds: f64,
    speech_seconds: f64,
    compute_seconds: f64,
    worker_compute_seconds: HashMap<usize, f64>,
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
    worker_id: usize,
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

#[derive(Debug, Clone, Default)]
struct LoaderStageStatsSnapshot {
    jobs: usize,
    failed: usize,
    audio_seconds: f64,
    compute_seconds: f64,
    worker_compute_seconds: HashMap<usize, f64>,
}

#[derive(Debug, Clone, Default)]
struct VadStageStatsSnapshot {
    jobs: usize,
    failed: usize,
    no_speech: usize,
    segments: usize,
    audio_seconds: f64,
    speech_seconds: f64,
    compute_seconds: f64,
    worker_compute_seconds: HashMap<usize, f64>,
    send_wait_seconds: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct AsrStageStatsSnapshot {
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
    offline: Arc<OfflinePipeline>,
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
            offline,
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
            "pipeline config | loader_workers={} vad_workers={} loaded_queue_mem={:.1}GiB segment_queue_mem={:.1}GiB encoder_request_queue=unbounded feature_queue=unbounded",
            loader_parallelism,
            vad_parallelism,
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
        let loader_handle = spawn_loader_worker(
            unique_inputs,
            loader,
            loader_parallelism,
            loaded_queue_memory,
            loader_to_vad_tx,
            Arc::clone(&loader_stats),
        );

        let offline = Arc::clone(&self.offline);
        let vad_options = self.parallel_options.vad_options.clone();
        let vad_handle = spawn_vad_worker(
            offline,
            vad_options,
            vad_parallelism,
            loader_to_vad_rx,
            vad_to_asr_tx,
            vad_to_asr_queue_memory,
            None,
            Arc::clone(&vad_stats),
        );

        let offline = Arc::clone(&self.offline);
        let asr_options = self.parallel_options.asr_options.clone();
        let asr_handle = spawn_asr_worker(
            offline,
            asr_options,
            vad_to_asr_rx,
            asr_to_result_tx,
            Arc::clone(&asr_stats),
            None,
            None,
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
    #[cfg(any(feature = "paged-attn", feature = "cuda-graph"))]
    offline: Arc<OfflinePipeline>,
    service_id: Arc<Mutex<String>>,
}

#[derive(Debug, Clone)]
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

struct GlobalTranscribeRequest {
    log_id: String,
    inputs: Vec<TranscribeInput>,
    response_tx: oneshot::Sender<Vec<TranscribeItemOutcome>>,
}

#[derive(Debug)]
struct GlobalLoadJob {
    internal_index: usize,
    request_item: RequestItemRef,
    source: AudioSource,
    mono: bool,
}

struct PendingGlobalRequest {
    log_id: String,
    remaining: usize,
    duplicate_indices: Vec<(usize, usize)>,
    internal_indices: Vec<usize>,
    outcomes: Vec<TranscribeItemOutcome>,
    stages: RequestStageProgress,
    response_tx: oneshot::Sender<Vec<TranscribeItemOutcome>>,
}

#[derive(Debug, Default)]
struct RequestStageProgress {
    load_remaining: usize,
    vad_remaining: usize,
    encode_remaining: usize,
    decode_remaining: usize,
    load_logged: bool,
    vad_logged: bool,
    encode_logged: bool,
    decode_logged: bool,
    load: StageProgress,
    vad: StageProgress,
    encode: StageProgress,
    prefill: StageProgress,
    decode: StageProgress,
}

#[derive(Debug, Default)]
struct StageProgress {
    run_span: Option<StageSpan>,
    queue_span: Option<StageSpan>,
    active_spans: Vec<StageSpan>,
    queue_spans: Vec<StageSpan>,
    audio_seconds: f64,
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
        offline: Arc<OfflinePipeline>,
        asr_options: AsrOptions,
        vad_options: VadOptions,
        worker_options: PipelineWorkerOptions,
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
            "global pipeline config | mode=serve loader_workers={} vad_workers={} loaded_queue_mem={:.1}GiB segment_queue_mem={:.1}GiB encoder_request_queue=unbounded feature_queue=unbounded decode_idle_flush={}ms",
            loader_parallelism,
            vad_parallelism,
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
        let service_id = Arc::new(Mutex::new(String::from("service")));
        let loader_stats = Arc::new(Mutex::new(LoaderStageStats::default()));
        let vad_stats = Arc::new(Mutex::new(VadStageStats::default()));
        let asr_stats = Arc::new(Mutex::new(AsrStageStats::default()));

        if profile {
            spawn_component_throughput_reporter(
                Arc::clone(&service_id),
                Arc::clone(&loader_stats),
                Arc::clone(&vad_stats),
                Arc::clone(&asr_stats),
                profile_window,
            );
        }

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
            loader_parallelism,
            loaded_queue_memory,
            load_job_rx,
            loader_to_vad_tx,
            Arc::clone(&pending_requests),
            Arc::clone(&job_routes),
            Arc::clone(&next_job_id),
            loader_stats,
        );
        spawn_vad_worker(
            Arc::clone(&offline),
            parallel_options.vad_options.clone(),
            vad_parallelism,
            loader_to_vad_rx,
            vad_to_asr_tx,
            vad_to_asr_queue_memory,
            Some(Arc::clone(&pending_requests)),
            vad_stats,
        );
        spawn_asr_worker(
            Arc::clone(&offline),
            parallel_options.asr_options.clone(),
            vad_to_asr_rx,
            asr_to_result_tx,
            asr_stats,
            Some(std::time::Duration::from_millis(
                GLOBAL_DECODE_IDLE_FLUSH_MS,
            )),
            Some(Arc::clone(&pending_requests)),
        );
        spawn_global_result_router(asr_to_result_rx, pending_requests, job_routes);

        Self {
            submit_tx,
            #[cfg(any(feature = "paged-attn", feature = "cuda-graph"))]
            offline,
            service_id,
        }
    }

    pub fn set_service_id(&self, service_id: String) {
        if let Ok(mut current) = self.service_id.lock() {
            *current = service_id;
        }
    }

    pub fn memory_snapshot(&self) -> PipelineMemorySnapshot {
        #[cfg(any(feature = "paged-attn", feature = "cuda-graph"))]
        let qwen = self.offline.asr.as_any().downcast_ref::<Qwen3ASR>();
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
        let qwen = self.offline.asr.as_any().downcast_ref::<Qwen3ASR>()?;
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
        inputs: Vec<TranscribeInput>,
    ) -> Vec<TranscribeItemOutcome> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let (response_tx, response_rx) = oneshot::channel();
        if self
            .submit_tx
            .send(GlobalTranscribeRequest {
                log_id,
                inputs,
                response_tx,
            })
            .is_err()
        {
            return vec![TranscribeItemOutcome {
                index: 0,
                channel: "mono",
                result: Err(anyhow::anyhow!("global transcribe pipeline stopped")),
                audio_seconds: 0.0,
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
                bad_component: Some("recognizer"),
            }];
        }
        response_rx.await.unwrap_or_else(|_| {
            vec![TranscribeItemOutcome {
                index: 0,
                channel: "mono",
                result: Err(anyhow::anyhow!(
                    "global transcribe pipeline response dropped"
                )),
                audio_seconds: 0.0,
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
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
            let GlobalTranscribeRequest {
                log_id,
                inputs,
                response_tx,
            } = request;
            let request_id = PipelineRequestId(next_request_id.fetch_add(1, Ordering::Relaxed));
            let DeduplicatedTranscribeInputs {
                unique_inputs,
                duplicate_indices,
            } = deduplicate_transcribe_inputs(inputs);
            let unique_count = unique_inputs.len();
            if unique_count == 0 {
                let _ = response_tx.send(Vec::new());
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
                    mono: input.mono,
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
                        log_id,
                        remaining: unique_count,
                        duplicate_indices,
                        internal_indices,
                        outcomes: Vec::with_capacity(unique_count),
                        stages: RequestStageProgress {
                            load_remaining: unique_count,
                            vad_remaining: unique_count,
                            ..Default::default()
                        },
                        response_tx,
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
    max_in_flight: usize,
    loaded_queue_memory: LoadedQueueMemoryBudget,
    mut load_job_rx: mpsc::UnboundedReceiver<GlobalLoadJob>,
    loader_to_vad_tx: mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
    pending_requests: Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: Arc<Mutex<HashMap<usize, RequestItemRef>>>,
    next_job_id: Arc<AtomicU64>,
    stats: Arc<Mutex<LoaderStageStats>>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        let mut free_workers = (0..max_in_flight).rev().collect::<Vec<_>>();
        loop {
            tokio::select! {
                job = load_job_rx.recv(), if tasks.len() < max_in_flight && !free_workers.is_empty() => {
                    match job {
                        Some(job) => {
                            let worker_id = free_workers.pop().unwrap_or(0);
                            spawn_global_loader_task(&mut tasks, loader.clone(), Arc::clone(&next_job_id), job, worker_id);
                        }
                        None => break,
                    }
                }
                result = tasks.join_next(), if !tasks.is_empty() => {
                    if let Some(result) = result {
                        let worker_id = send_global_loaded_result(result, &loaded_queue_memory, &loader_to_vad_tx, &pending_requests, &job_routes, &stats).await?;
                        free_workers.push(worker_id);
                    }
                }
            }
        }
        while let Some(result) = tasks.join_next().await {
            let worker_id = send_global_loaded_result(
                result,
                &loaded_queue_memory,
                &loader_to_vad_tx,
                &pending_requests,
                &job_routes,
                &stats,
            )
            .await?;
            free_workers.push(worker_id);
        }
        Ok(())
    })
}

fn spawn_global_loader_task(
    tasks: &mut JoinSet<(
        Result<Vec<LoadedJob>, TranscribeItemOutcome>,
        LoaderTaskStats,
    )>,
    loader: AudioLoader,
    next_job_id: Arc<AtomicU64>,
    job: GlobalLoadJob,
    worker_id: usize,
) {
    tasks.spawn(async move {
        let start = Instant::now();
        let loaded = tokio::task::spawn_blocking({
            let source = job.source.clone();
            let mono = job.mono;
            move || load_transcribe_waveforms(&loader, &source, mono)
        })
        .await;
        let compute_seconds = start.elapsed().as_secs_f64();
        let loader_span = StageSpan::new(start, Instant::now());
        let mut stats = LoaderTaskStats {
            compute_seconds,
            worker_id,
            ..Default::default()
        };
        let message = match loaded {
            Ok(Ok(mut waveforms)) => {
                stats.jobs = waveforms.len();
                stats.audio_seconds = waveforms
                    .iter()
                    .map(|(_, waveform)| waveform.duration_seconds())
                    .sum();
                Ok(waveforms
                    .drain(..)
                    .enumerate()
                    .map(|(offset, (channel, waveform))| LoadedJob {
                        index: if offset == 0 {
                            job.internal_index
                        } else {
                            next_job_id.fetch_add(1, Ordering::Relaxed) as usize
                        },
                        request_item: Some(job.request_item),
                        channel,
                        waveform,
                        timing: TranscribeItemTiming {
                            loader_seconds: compute_seconds,
                            ..Default::default()
                        },
                        spans: TranscribeItemSpans {
                            loader: Some(loader_span),
                            ..Default::default()
                        },
                        _queue_memory_permit: None,
                    })
                    .collect())
            }
            Ok(Err(error)) => {
                stats.failed = 1;
                Err(TranscribeItemOutcome {
                    index: job.internal_index,
                    channel: "mono",
                    result: Err(error),
                    audio_seconds: 0.0,
                    timing: TranscribeItemTiming {
                        loader_seconds: compute_seconds,
                        ..Default::default()
                    },
                    spans: TranscribeItemSpans {
                        loader: Some(loader_span),
                        ..Default::default()
                    },
                    bad_component: Some("loader"),
                })
            }
            Err(error) => {
                stats.failed = 1;
                Err(TranscribeItemOutcome {
                    index: job.internal_index,
                    channel: "mono",
                    result: Err(anyhow::anyhow!("loader join error: {error}")),
                    audio_seconds: 0.0,
                    timing: TranscribeItemTiming {
                        loader_seconds: compute_seconds,
                        ..Default::default()
                    },
                    spans: TranscribeItemSpans {
                        loader: Some(loader_span),
                        ..Default::default()
                    },
                    bad_component: Some("loader"),
                })
            }
        };
        (message, stats)
    });
}

async fn send_global_loaded_result(
    result: std::result::Result<
        (
            Result<Vec<LoadedJob>, TranscribeItemOutcome>,
            LoaderTaskStats,
        ),
        tokio::task::JoinError,
    >,
    loaded_queue_memory: &LoadedQueueMemoryBudget,
    loader_to_vad_tx: &mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    job_routes: &Arc<Mutex<HashMap<usize, RequestItemRef>>>,
    stats: &Arc<Mutex<LoaderStageStats>>,
) -> Result<usize> {
    let (mut message, task_stats) = result.unwrap_or_else(|error| {
        (
            Err(TranscribeItemOutcome {
                index: 0,
                channel: "mono",
                result: Err(anyhow::anyhow!("loader task join error: {error}")),
                audio_seconds: 0.0,
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
                bad_component: Some("loader"),
            }),
            LoaderTaskStats {
                failed: 1,
                ..Default::default()
            },
        )
    });
    let worker_id = task_stats.worker_id;
    record_loader_task_stats(stats, &task_stats);
    if let Ok(jobs) = &mut message {
        if jobs.is_empty() {
            return Ok(worker_id);
        }
        if let Some(request_item) = jobs[0].request_item {
            let span = jobs[0].spans.loader.unwrap_or_else(|| {
                let now = Instant::now();
                StageSpan::new(now, now)
            });
            let audio_seconds = jobs.iter().map(|job| job.waveform.duration_seconds()).sum();
            record_request_load_stage(
                pending_requests,
                request_item.request_id,
                span,
                audio_seconds,
                jobs.len(),
            );
        }
        if jobs.len() > 1 {
            let request_item = jobs[0].request_item;
            if let Some(request_item) = request_item {
                if let Ok(mut pending) = pending_requests.lock() {
                    if let Some(request) = pending.get_mut(&request_item.request_id) {
                        request.remaining += jobs.len() - 1;
                        request
                            .internal_indices
                            .extend(jobs.iter().skip(1).map(|job| job.index));
                    }
                }
                if let Ok(mut routes) = job_routes.lock() {
                    for job in jobs.iter().skip(1) {
                        routes.insert(job.index, request_item);
                    }
                }
            }
        }
        for job in jobs {
            job._queue_memory_permit =
                Some(acquire_loaded_job_memory(loaded_queue_memory, job).await?);
        }
    }
    match message {
        Ok(jobs) => {
            for job in jobs {
                loader_to_vad_tx
                    .send(Ok(job))
                    .map_err(|_| anyhow::anyhow!("loaded channel closed"))?;
            }
            Ok(worker_id)
        }
        Err(outcome) => loader_to_vad_tx
            .send(Err(outcome))
            .map_err(|_| anyhow::anyhow!("loaded channel closed"))
            .map(|_| worker_id),
    }
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

fn resolve_worker_count(configured: usize) -> usize {
    if configured == 0 {
        default_stage_parallelism()
    } else {
        configured
    }
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
        #[cfg(feature = "paged-attn")]
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
struct DeduplicatedTranscribeInputs {
    unique_inputs: Vec<TranscribeInput>,
    duplicate_indices: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TranscribeInputDedupKey {
    source: AudioSourceDedupKey,
    mono: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AudioSourceDedupKey {
    Path(PathBuf),
    Url(String),
    Base64(String),
    EncodedBytes(Vec<u8>),
    LocalFileContent { len: u64, hash: u64 },
}

fn deduplicate_transcribe_inputs(inputs: Vec<TranscribeInput>) -> DeduplicatedTranscribeInputs {
    let mut seen = HashMap::with_capacity(inputs.len());
    let mut unique_inputs = Vec::with_capacity(inputs.len());
    let mut duplicate_indices = Vec::new();

    for input in inputs {
        if let Some(source_key) = audio_source_dedup_key(&input.source) {
            let key = TranscribeInputDedupKey {
                source: source_key,
                mono: input.mono,
            };
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
        AudioSource::EncodedBytes(bytes) => Some(AudioSourceDedupKey::EncodedBytes(bytes.clone())),
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

fn record_request_load_stage(
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    request_id: PipelineRequestId,
    span: StageSpan,
    audio_seconds: f64,
    loaded_jobs: usize,
) {
    if let Ok(mut pending) = pending_requests.lock() {
        let Some(request) = pending.get_mut(&request_id) else {
            return;
        };
        request.stages.load.run_span = Some(merge_optional_span(
            request.stages.load.run_span,
            Some(span),
        ));
        request.stages.load.active_spans.push(span);
        request.stages.load.audio_seconds += audio_seconds;
        if loaded_jobs > 1 {
            request.stages.vad_remaining += loaded_jobs - 1;
        }
        request.stages.load_remaining = request.stages.load_remaining.saturating_sub(1);
        if request.stages.load_remaining == 0 && !request.stages.load_logged {
            request.stages.load_logged = true;
            log_request_stage(&request.log_id, "load", &request.stages.load);
        }
    }
}

#[cfg(feature = "paged-attn")]
#[derive(Debug, Clone, Copy)]
struct SegmentLogItem {
    request_item: Option<RequestItemRef>,
    audio_seconds: f64,
}

#[cfg(feature = "paged-attn")]
fn segment_log_items(segments: &[AsrSegmentJob]) -> Vec<SegmentLogItem> {
    segments
        .iter()
        .map(|segment| SegmentLogItem {
            request_item: segment.request_item,
            audio_seconds: segment.waveform.duration_seconds(),
        })
        .collect()
}

#[cfg(feature = "paged-attn")]
fn segment_batch_request_count(segments: &[AsrSegmentJob]) -> usize {
    let mut request_ids = HashSet::new();
    for segment in segments {
        if let Some(request_item) = segment.request_item {
            request_ids.insert(request_item.request_id);
        }
    }
    request_ids.len()
}

#[cfg(feature = "paged-attn")]
fn log_prefill_batch_done(
    all_segments: &[SegmentLogItem],
    output: &PrefillStageOutput,
    elapsed_seconds: f64,
) {
    let mut segments = Vec::new();
    for sequence in &output.sequences {
        if let Some(&segment) = all_segments.get(sequence.client_index) {
            segments.push(segment);
        }
    }
    if segments.is_empty() {
        return;
    }
    let audio_seconds = segments
        .iter()
        .map(|segment| segment.audio_seconds)
        .sum::<f64>();
    let request_count = segments
        .iter()
        .filter_map(|segment| segment.request_item.map(|item| item.request_id))
        .collect::<HashSet<_>>()
        .len();
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "prefill batch DONE | reqs={} seg={} audio={} elapsed={} speed={} reason={}",
        request_count,
        segments.len(),
        format_seconds(audio_seconds),
        format_seconds(elapsed_seconds),
        format_speed(audio_seconds, elapsed_seconds),
        output.reason.as_str(),
    );
}

fn record_request_vad_stage(
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    request_id: PipelineRequestId,
    span: StageSpan,
    queue_span: Option<StageSpan>,
    audio_seconds: f64,
    segment_count: usize,
) {
    if let Ok(mut pending) = pending_requests.lock() {
        let Some(request) = pending.get_mut(&request_id) else {
            return;
        };
        request.stages.vad.run_span =
            Some(merge_optional_span(request.stages.vad.run_span, Some(span)));
        request.stages.vad.queue_span = merge_stage_span(request.stages.vad.queue_span, queue_span);
        request.stages.vad.active_spans.push(span);
        if let Some(queue_span) = queue_span {
            request.stages.vad.queue_spans.push(queue_span);
        }
        request.stages.vad.audio_seconds += audio_seconds;
        request.stages.encode_remaining += segment_count;
        request.stages.decode_remaining += segment_count;
        request.stages.vad_remaining = request.stages.vad_remaining.saturating_sub(1);
        if request.stages.vad_remaining == 0 && !request.stages.vad_logged {
            request.stages.vad_logged = true;
            log_request_stage(&request.log_id, "vad", &request.stages.vad);
            log_ready_downstream_stages(request);
        }
    }
}

#[cfg(feature = "paged-attn")]
fn record_request_encode_stage(
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    request_id: PipelineRequestId,
    span: StageSpan,
    queue_span: Option<StageSpan>,
    audio_seconds: f64,
    segment_count: usize,
) {
    if let Ok(mut pending) = pending_requests.lock() {
        let Some(request) = pending.get_mut(&request_id) else {
            return;
        };
        request.stages.encode.run_span = Some(merge_optional_span(
            request.stages.encode.run_span,
            Some(span),
        ));
        request.stages.encode.queue_span =
            merge_stage_span(request.stages.encode.queue_span, queue_span);
        request.stages.encode.active_spans.push(span);
        if let Some(queue_span) = queue_span {
            request.stages.encode.queue_spans.push(queue_span);
        }
        request.stages.encode.audio_seconds += audio_seconds;
        request.stages.encode_remaining = request
            .stages
            .encode_remaining
            .saturating_sub(segment_count);
        log_ready_downstream_stages(request);
    }
}

#[cfg(feature = "paged-attn")]
fn record_request_encode_stages(
    pending_requests: &Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    segments: &[AsrSegmentJob],
    span: StageSpan,
) {
    let Some(pending_requests) = pending_requests else {
        return;
    };
    let mut by_request: HashMap<PipelineRequestId, (f64, usize, Option<StageSpan>)> =
        HashMap::new();
    for segment in segments {
        let Some(request_item) = segment.request_item else {
            continue;
        };
        let entry = by_request.entry(request_item.request_id).or_default();
        entry.0 += segment.waveform.duration_seconds();
        entry.1 += 1;
        let queue_span = segment
            .spans
            .vad
            .and_then(|vad| StageSpan::try_new(vad.end, span.start));
        entry.2 = merge_stage_span(entry.2, queue_span);
    }
    for (request_id, (audio_seconds, segment_count, queue_span)) in by_request {
        record_request_encode_stage(
            pending_requests,
            request_id,
            span,
            queue_span,
            audio_seconds,
            segment_count,
        );
    }
}

#[allow(dead_code)]
fn record_request_decode_stage(
    pending_requests: &Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>,
    request_id: PipelineRequestId,
    decode_span: StageSpan,
    prefill_span: Option<StageSpan>,
    prefill_queue_span: Option<StageSpan>,
    decode_queue_span: Option<StageSpan>,
    audio_seconds: f64,
    segment_count: usize,
) {
    if let Ok(mut pending) = pending_requests.lock() {
        let Some(request) = pending.get_mut(&request_id) else {
            return;
        };
        if let Some(prefill_span) = prefill_span {
            request.stages.prefill.run_span = Some(merge_optional_span(
                request.stages.prefill.run_span,
                Some(prefill_span),
            ));
            request.stages.prefill.queue_span =
                merge_stage_span(request.stages.prefill.queue_span, prefill_queue_span);
            request.stages.prefill.active_spans.push(prefill_span);
            if let Some(prefill_queue_span) = prefill_queue_span {
                request.stages.prefill.queue_spans.push(prefill_queue_span);
            }
            request.stages.prefill.audio_seconds += audio_seconds;
        }
        request.stages.decode.run_span = Some(merge_optional_span(
            request.stages.decode.run_span,
            Some(decode_span),
        ));
        request.stages.decode.queue_span =
            merge_stage_span(request.stages.decode.queue_span, decode_queue_span);
        request.stages.decode.active_spans.push(decode_span);
        if let Some(decode_queue_span) = decode_queue_span {
            request.stages.decode.queue_spans.push(decode_queue_span);
        }
        request.stages.decode.audio_seconds += audio_seconds;
        request.stages.decode_remaining = request
            .stages
            .decode_remaining
            .saturating_sub(segment_count);
        log_ready_downstream_stages(request);
    }
}

fn log_ready_downstream_stages(request: &mut PendingGlobalRequest) {
    if !request.stages.vad_logged {
        return;
    }
    if request.stages.encode_remaining == 0
        && request.stages.encode.run_span.is_some()
        && !request.stages.encode_logged
    {
        request.stages.encode_logged = true;
        log_request_stage(&request.log_id, "encode", &request.stages.encode);
    }
    if request.stages.decode_remaining == 0
        && request.stages.decode.run_span.is_some()
        && !request.stages.decode_logged
    {
        request.stages.decode_logged = true;
        if request.stages.prefill.run_span.is_some() {
            log_request_stage(&request.log_id, "prefill", &request.stages.prefill);
        }
        log_request_stage(&request.log_id, "decode", &request.stages.decode);
    }
}

#[cfg(feature = "paged-attn")]
#[allow(dead_code)]
fn record_request_decode_stages(
    pending_requests: &Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    segments: &[AsrSegmentJob],
    decode_span: StageSpan,
    prefill_span: Option<StageSpan>,
) {
    let Some(pending_requests) = pending_requests else {
        return;
    };
    let mut by_request: HashMap<
        PipelineRequestId,
        (f64, usize, Option<StageSpan>, Option<StageSpan>),
    > = HashMap::new();
    for segment in segments {
        let Some(request_item) = segment.request_item else {
            continue;
        };
        let entry = by_request.entry(request_item.request_id).or_default();
        entry.0 += segment.waveform.duration_seconds();
        entry.1 += 1;
        let (prefill_queue_span, decode_queue_span) = if let Some(prefill_span) = prefill_span {
            (
                segment
                    .spans
                    .audio_encoder
                    .and_then(|encode| StageSpan::try_new(encode.end, prefill_span.start)),
                StageSpan::try_new(prefill_span.end, decode_span.start),
            )
        } else {
            (
                None,
                segment
                    .spans
                    .audio_encoder
                    .and_then(|encode| StageSpan::try_new(encode.end, decode_span.start)),
            )
        };
        entry.2 = merge_stage_span(entry.2, prefill_queue_span);
        entry.3 = merge_stage_span(entry.3, decode_queue_span);
    }
    for (request_id, (audio_seconds, segment_count, prefill_queue_span, decode_queue_span)) in
        by_request
    {
        record_request_decode_stage(
            pending_requests,
            request_id,
            decode_span,
            prefill_span,
            prefill_queue_span,
            decode_queue_span,
            audio_seconds,
            segment_count,
        );
    }
}

fn log_request_stage(request_id: &str, stage: &'static str, progress: &StageProgress) {
    let active_seconds = merged_spans_seconds(progress.active_spans.clone());
    let wall_seconds = merge_stage_span(progress.queue_span, progress.run_span)
        .map(|span| span.end.duration_since(span.start).as_secs_f64())
        .unwrap_or_default();
    let queue_seconds = (wall_seconds - active_seconds).max(0.0);
    let elapsed_seconds = wall_seconds;
    let speed = progress.audio_seconds / active_seconds.max(f64::EPSILON);
    let e2e_speed = progress.audio_seconds / elapsed_seconds.max(f64::EPSILON);
    let audio_label = match stage {
        "load" | "vad" => "input_audio",
        _ => "speech_audio",
    };
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "{} {:<7} {:<5} | {}={:>6.1}s | elapsed={:>6.3}s | active={:>6.3}s | queue={:>6.3}s wall={:>6.3}s | speed={:>6.1}x e2e={:>6.1}x",
        request_id,
        stage,
        "DONE",
        audio_label,
        progress.audio_seconds,
        elapsed_seconds,
        active_seconds,
        queue_seconds,
        wall_seconds,
        speed,
        e2e_speed
    );
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
        .flat_map(|&(duplicate_index, canonical_index)| {
            outcomes
                .iter()
                .filter(move |outcome| outcome.index == canonical_index)
                .map(move |outcome| duplicate_outcome_from(outcome, duplicate_index))
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
        channel: outcome.channel,
        result,
        audio_seconds: outcome.audio_seconds,
        timing: outcome.timing,
        spans: outcome.spans,
        bad_component: outcome.bad_component,
    }
}

#[derive(Debug)]
struct LoadedJob {
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
    job_index: usize,
    request_item: Option<RequestItemRef>,
    channel: &'static str,
    timing: TranscribeItemTiming,
    spans: TranscribeItemSpans,
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
    #[cfg(feature = "paged-attn")]
    Segment(AsrSegmentJob),
}

#[derive(Debug)]
struct VadToAsrMessage {
    message: Result<AsrPipelineMessage, TranscribeItemOutcome>,
    _queue_memory_permit: Option<OwnedSemaphorePermit>,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
struct SegmentJobState {
    index: usize,
    request_item: Option<RequestItemRef>,
    channel: &'static str,
    timing: TranscribeItemTiming,
    spans: TranscribeItemSpans,
    audio_seconds: f64,
    speech_annotations: Vec<Annotation>,
    segment_annotations: Vec<Option<Vec<Annotation>>>,
    segment_count: usize,
    segments_done: usize,
}

#[derive(Default)]
#[cfg(feature = "paged-attn")]
struct SegmentAsrState {
    jobs: HashMap<usize, SegmentJobState>,
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
            offset: segment.range.start,
            waveform,
        })
        .collect())
}

fn spawn_loader_worker(
    inputs: Vec<TranscribeInput>,
    loader: AudioLoader,
    max_in_flight: usize,
    loaded_queue_memory: LoadedQueueMemoryBudget,
    loader_to_vad_tx: mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
    stats: Arc<Mutex<LoaderStageStats>>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        let wall_start = Instant::now();
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        let mut free_workers = (0..max_in_flight).rev().collect::<Vec<_>>();
        for input in inputs {
            if tasks.len() >= max_in_flight {
                if let Some(worker_id) =
                    send_next_loaded(&mut tasks, &loaded_queue_memory, &loader_to_vad_tx, &stats)
                        .await?
                {
                    free_workers.push(worker_id);
                }
            }
            let worker_id = free_workers.pop().unwrap_or(tasks.len());
            spawn_loader_task(&mut tasks, loader.clone(), input, worker_id);
        }
        while !tasks.is_empty() {
            if let Some(worker_id) =
                send_next_loaded(&mut tasks, &loaded_queue_memory, &loader_to_vad_tx, &stats)
                    .await?
            {
                free_workers.push(worker_id);
            }
        }
        if let Ok(mut stats) = stats.lock() {
            stats.wall_seconds = wall_start.elapsed().as_secs_f64();
            log_loader_summary(&stats);
        }
        Ok(())
    })
}

fn spawn_loader_task(
    tasks: &mut JoinSet<(
        Result<Vec<LoadedJob>, TranscribeItemOutcome>,
        LoaderTaskStats,
    )>,
    loader: AudioLoader,
    input: TranscribeInput,
    worker_id: usize,
) {
    tasks.spawn(async move {
        let start = Instant::now();
        let loaded = tokio::task::spawn_blocking({
            let source = input.source.clone();
            let mono = input.mono;
            move || load_transcribe_waveforms(&loader, &source, mono)
        })
        .await;

        let compute_seconds = start.elapsed().as_secs_f64();
        let loader_span = StageSpan::new(start, Instant::now());
        let mut stats = LoaderTaskStats {
            compute_seconds,
            worker_id,
            ..Default::default()
        };
        let message = match loaded {
            Ok(Ok(mut waveforms)) => {
                stats.jobs = waveforms.len();
                stats.audio_seconds = waveforms
                    .iter()
                    .map(|(_, waveform)| waveform.duration_seconds())
                    .sum();
                Ok(waveforms
                    .drain(..)
                    .enumerate()
                    .map(|(offset, (channel, waveform))| LoadedJob {
                        index: input.index + offset,
                        request_item: None,
                        channel,
                        waveform,
                        timing: TranscribeItemTiming {
                            loader_seconds: compute_seconds,
                            ..Default::default()
                        },
                        spans: TranscribeItemSpans {
                            loader: Some(loader_span),
                            ..Default::default()
                        },
                        _queue_memory_permit: None,
                    })
                    .collect())
            }
            Ok(Err(error)) => {
                stats.failed = 1;
                Err(TranscribeItemOutcome {
                    index: input.index,
                    channel: "mono",
                    result: Err(error),
                    audio_seconds: 0.0,
                    timing: TranscribeItemTiming {
                        loader_seconds: compute_seconds,
                        ..Default::default()
                    },
                    spans: TranscribeItemSpans {
                        loader: Some(loader_span),
                        ..Default::default()
                    },
                    bad_component: Some("loader"),
                })
            }
            Err(error) => {
                stats.failed = 1;
                Err(TranscribeItemOutcome {
                    index: input.index,
                    channel: "mono",
                    result: Err(anyhow::anyhow!("loader join error: {error}")),
                    audio_seconds: 0.0,
                    timing: TranscribeItemTiming {
                        loader_seconds: compute_seconds,
                        ..Default::default()
                    },
                    spans: TranscribeItemSpans {
                        loader: Some(loader_span),
                        ..Default::default()
                    },
                    bad_component: Some("loader"),
                })
            }
        };
        (message, stats)
    });
}

async fn send_next_loaded(
    tasks: &mut JoinSet<(
        Result<Vec<LoadedJob>, TranscribeItemOutcome>,
        LoaderTaskStats,
    )>,
    loaded_queue_memory: &LoadedQueueMemoryBudget,
    loader_to_vad_tx: &mpsc::UnboundedSender<Result<LoadedJob, TranscribeItemOutcome>>,
    stats: &Arc<Mutex<LoaderStageStats>>,
) -> Result<Option<usize>> {
    let (mut message, task_stats) = match tasks.join_next().await {
        Some(Ok(result)) => result,
        Some(Err(error)) => (
            Err(TranscribeItemOutcome {
                index: 0,
                channel: "mono",
                result: Err(anyhow::anyhow!("loader task join error: {error}")),
                audio_seconds: 0.0,
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
                bad_component: Some("loader"),
            }),
            LoaderTaskStats {
                failed: 1,
                ..Default::default()
            },
        ),
        None => return Ok(None),
    };
    let worker_id = task_stats.worker_id;
    record_loader_task_stats(stats, &task_stats);
    if let Ok(jobs) = &mut message {
        for job in jobs {
            job._queue_memory_permit =
                Some(acquire_loaded_job_memory(loaded_queue_memory, job).await?);
        }
    }
    match message {
        Ok(jobs) => {
            for job in jobs {
                loader_to_vad_tx
                    .send(Ok(job))
                    .map_err(|_| anyhow::anyhow!("loaded channel closed"))?;
            }
            Ok(Some(worker_id))
        }
        Err(outcome) => loader_to_vad_tx
            .send(Err(outcome))
            .map_err(|_| anyhow::anyhow!("loaded channel closed"))
            .map(|_| Some(worker_id)),
    }
}

fn spawn_vad_worker(
    offline: Arc<OfflinePipeline>,
    vad_options: VadOptions,
    max_in_flight: usize,
    mut loader_to_vad_rx: mpsc::UnboundedReceiver<Result<LoadedJob, TranscribeItemOutcome>>,
    vad_to_asr_tx: mpsc::UnboundedSender<VadToAsrMessage>,
    vad_to_asr_queue_memory: LoadedQueueMemoryBudget,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    stats: Arc<Mutex<VadStageStats>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let wall_start = Instant::now();
        let max_in_flight = max_in_flight.max(1);
        let mut tasks = JoinSet::new();
        let mut free_workers = (0..max_in_flight).rev().collect::<Vec<_>>();
        while let Some(message) = loader_to_vad_rx.recv().await {
            match message {
                Ok(job) => {
                    if tasks.len() >= max_in_flight {
                        if let Ok(Some(worker_id)) = send_next_vad_task(&mut tasks, &stats).await {
                            free_workers.push(worker_id);
                        }
                    }
                    let worker_id = free_workers.pop().unwrap_or(tasks.len());
                    spawn_vad_task(
                        &mut tasks,
                        Arc::clone(&offline),
                        vad_options.clone(),
                        vad_to_asr_tx.clone(),
                        vad_to_asr_queue_memory.clone(),
                        pending_requests.clone(),
                        job,
                        worker_id,
                    );
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
            if let Ok(Some(worker_id)) = send_next_vad_task(&mut tasks, &stats).await {
                free_workers.push(worker_id);
            }
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
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
    job: LoadedJob,
    worker_id: usize,
) {
    tasks.spawn(async move {
        let stage_start = Instant::now();
        let request_item = job.request_item;
        let mut timing = job.timing;
        let audio_seconds = job.waveform.duration_seconds();
        let job_index = job.index;
        let job_channel = job.channel;
        let job_timing = job.timing;
        let mut spans = job.spans;
        let vad_queue_span = spans
            .loader
            .and_then(|loader| StageSpan::try_new(loader.end, stage_start));
        let prepared = tokio::task::spawn_blocking({
            let offline = Arc::clone(&offline);
            let waveform = job.waveform.clone();
            let vad_options = vad_options.clone();
            move || offline.prepare_vad_stage(&waveform, &vad_options)
        })
        .await;
        let compute_seconds = stage_start.elapsed().as_secs_f64();
        let vad_span = StageSpan::new(stage_start, Instant::now());
        spans.vad = Some(vad_span);
        timing.vad_seconds += compute_seconds;
        let mut send_wait_seconds = 0.0;

        match prepared {
            Ok(Ok(prepared)) => {
                let (segments, speech_seconds) = vad_preparation_stats(&prepared);
                let no_speech = matches!(prepared, VadPreparation::NoSpeech);
                let asr_job = AsrJob {
                    index: job_index,
                    request_item,
                    channel: job.channel,
                    timing,
                    spans,
                    prepared,
                    waveform: job.waveform,
                };
                match asr_job.prepared {
                    #[cfg(feature = "paged-attn")]
                    VadPreparation::Speech(_) => {
                        let segment_jobs = match speech_job_to_segments(asr_job) {
                            Ok(segments) => segments,
                            Err(error) => {
                                send_vad_to_asr_timed(
                                    &vad_to_asr_tx,
                                    &vad_to_asr_queue_memory,
                                    Err(TranscribeItemOutcome {
                                        index: job_index,
                                        channel: job.channel,
                                        result: Err(error),
                                        audio_seconds,
                                        timing,
                                        spans,
                                        bad_component: Some("vad"),
                                    }),
                                    &mut send_wait_seconds,
                                )
                                .await
                                .map_err(|error| {
                                    TranscribeItemOutcome {
                                        index: job_index,
                                        channel: job.channel,
                                        result: Err(error),
                                        audio_seconds,
                                        timing,
                                        spans,
                                        bad_component: Some("vad"),
                                    }
                                })?;
                                Vec::new()
                            }
                        };
                        for segment in segment_jobs {
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
                                    channel: job.channel,
                                    result: Err(error),
                                    audio_seconds,
                                    timing,
                                    spans,
                                    bad_component: Some("asr"),
                                }
                            })?;
                        }
                    }
                    #[cfg(not(feature = "paged-attn"))]
                    VadPreparation::Speech(_) => {
                        send_vad_to_asr_timed(
                            &vad_to_asr_tx,
                            &vad_to_asr_queue_memory,
                            Err(TranscribeItemOutcome {
                                index: job_index,
                                channel: job.channel,
                                result: Err(anyhow::anyhow!("speech ASR requires paged-attn")),
                                audio_seconds,
                                timing,
                                spans,
                                bad_component: Some("asr"),
                            }),
                            &mut send_wait_seconds,
                        )
                        .await
                        .map_err(|error| TranscribeItemOutcome {
                            index: job_index,
                            channel: job.channel,
                            result: Err(error),
                            audio_seconds,
                            timing,
                            spans,
                            bad_component: Some("asr"),
                        })?;
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
                            channel: job.channel,
                            result: Err(error),
                            audio_seconds,
                            timing,
                            spans,
                            bad_component: Some("asr"),
                        })?;
                    }
                }
                if let (Some(pending_requests), Some(request_item)) =
                    (&pending_requests, request_item)
                {
                    record_request_vad_stage(
                        pending_requests,
                        request_item.request_id,
                        vad_span,
                        vad_queue_span,
                        audio_seconds,
                        segments,
                    );
                }
                let wall_seconds = stage_start.elapsed().as_secs_f64();
                Ok(VadTaskStats {
                    jobs: 1,
                    no_speech: usize::from(no_speech),
                    segments,
                    audio_seconds,
                    speech_seconds,
                    compute_seconds,
                    worker_id,
                    send_wait_seconds,
                    wall_seconds,
                    ..Default::default()
                })
            }
            Ok(Err(error)) => {
                send_vad_to_asr_timed(
                    &vad_to_asr_tx,
                    &vad_to_asr_queue_memory,
                    Err(TranscribeItemOutcome {
                        index: job_index,
                        channel: job.channel,
                        result: Err(error),
                        audio_seconds,
                        timing,
                        spans,
                        bad_component: Some("vad"),
                    }),
                    &mut send_wait_seconds,
                )
                .await
                .map_err(|error| TranscribeItemOutcome {
                    index: job_index,
                    channel: job.channel,
                    result: Err(error),
                    audio_seconds,
                    timing,
                    spans,
                    bad_component: Some("vad"),
                })?;
                if let (Some(pending_requests), Some(request_item)) =
                    (&pending_requests, request_item)
                {
                    record_request_vad_stage(
                        pending_requests,
                        request_item.request_id,
                        vad_span,
                        vad_queue_span,
                        audio_seconds,
                        0,
                    );
                }
                let wall_seconds = stage_start.elapsed().as_secs_f64();
                Ok(VadTaskStats {
                    jobs: 1,
                    failed: 1,
                    audio_seconds,
                    compute_seconds,
                    worker_id,
                    send_wait_seconds,
                    wall_seconds,
                    ..Default::default()
                })
            }
            Err(error) => {
                send_vad_to_asr_timed(
                    &vad_to_asr_tx,
                    &vad_to_asr_queue_memory,
                    Err(TranscribeItemOutcome {
                        index: job_index,
                        channel: job_channel,
                        result: Err(anyhow::anyhow!("VAD join error: {error}")),
                        audio_seconds,
                        timing: job_timing,
                        spans,
                        bad_component: Some("vad"),
                    }),
                    &mut send_wait_seconds,
                )
                .await
                .map_err(|error| TranscribeItemOutcome {
                    index: job_index,
                    channel: job_channel,
                    result: Err(error),
                    audio_seconds,
                    timing: job_timing,
                    spans,
                    bad_component: Some("vad"),
                })?;
                if let (Some(pending_requests), Some(request_item)) =
                    (&pending_requests, request_item)
                {
                    record_request_vad_stage(
                        pending_requests,
                        request_item.request_id,
                        vad_span,
                        vad_queue_span,
                        audio_seconds,
                        0,
                    );
                }
                let wall_seconds = stage_start.elapsed().as_secs_f64();
                Ok(VadTaskStats {
                    jobs: 1,
                    failed: 1,
                    audio_seconds,
                    compute_seconds,
                    worker_id,
                    send_wait_seconds,
                    wall_seconds,
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
) -> Result<Option<usize>, TranscribeItemOutcome> {
    match tasks.join_next().await {
        Some(Ok(result)) => {
            let worker_id = result.as_ref().ok().map(|task_stats| task_stats.worker_id);
            if let Ok(task_stats) = &result {
                record_vad_task_stats(stats, task_stats);
            }
            result.map(|_| worker_id)
        }
        Some(Err(error)) => Err(TranscribeItemOutcome {
            index: 0,
            channel: "mono",
            result: Err(anyhow::anyhow!("VAD task join error: {error}")),
            audio_seconds: 0.0,
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            bad_component: Some("vad"),
        }),
        None => Ok(None),
    }
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
struct SegmentEncodeRequest {
    segment: AsrSegmentJob,
    stage_start: Instant,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
struct SegmentEncodeBatch {
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    reason: &'static str,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
struct SegmentEncodedMessage {
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    encoder_timing: TranscribeItemTiming,
    batch: Result<SegmentEncodedAsrBatch>,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
enum SegmentEncodeMessage {
    Segment(SegmentEncodeRequest),
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
enum SegmentEncoderOutput {
    Encoded(SegmentEncodedMessage),
}

#[cfg(feature = "paged-attn")]
type SegmentDecoderReceiver = mpsc::UnboundedReceiver<SegmentPrefillMessage>;

#[cfg(feature = "paged-attn")]
fn spawn_segment_encoder_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    mut request_rx: mpsc::UnboundedReceiver<SegmentEncodeMessage>,
    encoded_tx: mpsc::UnboundedSender<SegmentEncoderOutput>,
    stats: Arc<Mutex<AsrStageStats>>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
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
            let request = match request {
                SegmentEncodeMessage::Segment(request) => request,
            };
            let request =
                collect_encoder_segment_batch(request, &mut request_rx, &mut pending).await;
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
            let encode_span = StageSpan::new(encode_start, Instant::now());
            let encode_timings = batch
                .as_ref()
                .map(|batch| batch.timings)
                .unwrap_or_default();
            let encoder_timing = audio_encoder_timing_seconds(encode_timings);
            record_asr_encode_stats(
                &stats,
                segments_for_result.len(),
                segment_audio_seconds,
                encode_wall_seconds,
            );
            let encode_failed = batch.is_err();
            let mut segments_for_result = segments_for_result;
            for segment in &mut segments_for_result {
                segment.spans.audio_encoder = Some(encode_span);
            }
            record_request_encode_stages(&pending_requests, &segments_for_result, encode_span);
            let request_count = segment_batch_request_count(&segments_for_result);
            tracing::debug!(
                target: "vasr_transcribe::pipeline",
                "encode batch DONE | jobs={} reqs={} seg={} audio={} elapsed={} speed={} reason={} bad={}",
                segments_for_result
                    .iter()
                    .map(|segment| segment.job_index)
                    .collect::<HashSet<_>>()
                    .len(),
                request_count,
                segments_for_result.len(),
                format_seconds(segment_audio_seconds),
                format_seconds(encode_wall_seconds),
                format_speed(segment_audio_seconds, encode_wall_seconds),
                request.reason,
                usize::from(encode_failed),
            );
            let message = SegmentEncodedMessage {
                segments: segments_for_result,
                stage_start: request.stage_start,
                encoder_timing,
                batch,
            };
            if encoded_tx
                .send(SegmentEncoderOutput::Encoded(message))
                .is_err()
            {
                break;
            }
        }
    })
}

#[cfg(feature = "paged-attn")]
enum SegmentPrefillMessage {
    Prefill {
        producer_id: usize,
        result: Result<PrefillStageOutput>,
    },
    Prepared {
        producer_id: usize,
        result: Result<SegmentPreparedEvent, SegmentEncodedFailure>,
    },
    Failure(SegmentEncodedFailure),
}

#[cfg(feature = "paged-attn")]
enum SegmentPreparedEvent {
    Prepared(SegmentPreparedMetadata),
}

#[cfg(feature = "paged-attn")]
struct PrefillCoalescedPrepared {
    prepared: PreparedAudioBatch,
}

#[cfg(feature = "paged-attn")]
enum PrefillCoalescedPoll {
    Ready(PrefillCoalescedPrepared),
    Empty,
    Closed,
}

#[cfg(feature = "paged-attn")]
fn spawn_segment_prefill_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    encoded_rx: mpsc::UnboundedReceiver<SegmentEncoderOutput>,
    prefilled_tx: mpsc::UnboundedSender<SegmentPrefillMessage>,
    decode_idle_flush: Option<std::time::Duration>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let encoded_rx = Arc::new(tokio::sync::Mutex::new(encoded_rx));
        let mut next_producer_id = 0usize;
        loop {
            let first = {
                let mut encoded_rx = encoded_rx.lock().await;
                encoded_rx.recv().await
            };
            let Some(first) = first else {
                break;
            };
            let first = match first {
                SegmentEncoderOutput::Encoded(message) => message,
            };
            let SegmentEncodedMessage {
                mut segments,
                stage_start,
                encoder_timing,
                batch,
            } = first;
            apply_batch_timing_to_segments(&mut segments, encoder_timing);
            let initial = match batch {
                Ok(batch) => batch,
                Err(error) => {
                    if prefilled_tx
                        .send(SegmentPrefillMessage::Failure(SegmentEncodedFailure {
                            segments,
                            stage_start,
                            error,
                        }))
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
            };
            let result = tokio::task::spawn_blocking({
                let producer_id = next_producer_id;
                next_producer_id = next_producer_id.saturating_add(1);
                let offline = Arc::clone(&offline);
                let asr_options = asr_options.clone();
                let encoded_rx = Arc::clone(&encoded_rx);
                let prefilled_tx = prefilled_tx.clone();
                let runtime_handle = tokio::runtime::Handle::current();
                move || {
                    run_global_segment_prefill(
                        offline,
                        segments,
                        stage_start,
                        initial,
                        encoded_rx,
                        runtime_handle,
                        &asr_options,
                        decode_idle_flush,
                        prefilled_tx,
                        producer_id,
                    )
                }
            })
            .await
            .unwrap_or_else(|error| {
                Err(anyhow::anyhow!(
                    "ASR dynamic prefill worker join error: {error}"
                ))
            });
            if let Err(error) = result {
                tracing::warn!(
                    target: "vasr_transcribe::pipeline",
                    "ASR dynamic prefill failed: {error}"
                );
            }
            if decode_idle_flush.is_none() {
                break;
            }
        }
    })
}

#[cfg(feature = "paged-attn")]
async fn collect_encoder_segment_batch(
    first: SegmentEncodeRequest,
    request_rx: &mut mpsc::UnboundedReceiver<SegmentEncodeMessage>,
    _pending: &mut VecDeque<SegmentEncodeMessage>,
) -> SegmentEncodeBatch {
    let mut stage_start = first.stage_start;
    let mut segments = vec![first.segment];
    let mut reason = "wait_timeout";
    loop {
        match request_rx.try_recv() {
            Ok(SegmentEncodeMessage::Segment(request)) => {
                stage_start = stage_start.min(request.stage_start);
                segments.push(request.segment);
            }
            Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                break;
            }
        }
    }
    loop {
        match tokio::time::timeout(
            std::time::Duration::from_millis(ASR_ENCODER_BATCH_COLLECT_WAIT_MS),
            request_rx.recv(),
        )
        .await
        {
            Ok(Some(SegmentEncodeMessage::Segment(request))) => {
                stage_start = stage_start.min(request.stage_start);
                segments.push(request.segment);
            }
            Ok(None) => {
                reason = "source_closed";
                break;
            }
            Err(tokio::time::error::Elapsed { .. }) => break,
        }
    }
    segments.sort_by_key(|segment| (segment.job_index, segment.offset.0));
    SegmentEncodeBatch {
        segments,
        stage_start,
        reason,
    }
}

#[cfg(feature = "paged-attn")]
fn spawn_segment_decoder_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    decoder_rx: SegmentDecoderReceiver,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut segment_state = SegmentAsrState::default();
        let result = tokio::task::spawn_blocking({
            let asr = Arc::clone(&offline.asr);
            let asr_options = asr_options.clone();
            let stats = Arc::clone(&stats);
            let pending_requests = pending_requests.clone();
            let asr_to_result_tx = asr_to_result_tx.clone();
            move || {
                run_global_segment_decode_worker(
                    asr.as_ref(),
                    asr_options,
                    decoder_rx,
                    &mut segment_state,
                    asr_to_result_tx,
                    stats,
                    pending_requests,
                )
            }
        })
        .await
        .unwrap_or_else(|error| Err(anyhow::anyhow!("ASR global decoder join error: {error}")));
        if let Err(error) = result {
            tracing::error!(
                target: "vasr_transcribe::pipeline",
                "ASR global decoder failed: {error:#}"
            );
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
struct SegmentPreparedMetadata {
    segments: Vec<AsrSegmentJob>,
    waveforms: Vec<Waveform>,
    meta: EncodedAsrBatchMetadata,
    stage_start: Instant,
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn run_global_segment_prefill(
    offline: Arc<OfflinePipeline>,
    segments: Vec<AsrSegmentJob>,
    stage_start: Instant,
    initial: SegmentEncodedAsrBatch,
    encoded_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: tokio::runtime::Handle,
    asr_options: &AsrOptions,
    decode_idle_flush: Option<std::time::Duration>,
    decoder_tx: mpsc::UnboundedSender<SegmentPrefillMessage>,
    producer_id: usize,
) -> Result<()> {
    let qwen3 = offline
        .asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("dynamic staged ASR prefill requires Qwen3ASR"))?;
    let (_inference, waveforms, encoded) = initial.into_parts();
    let (initial_prepared, initial_meta) = qwen3.prepare_encoded_audio_features(encoded)?;
    let prefill_log_segments = segment_log_items(&segments);
    decoder_tx
        .send(SegmentPrefillMessage::Prepared {
            producer_id,
            result: Ok(SegmentPreparedEvent::Prepared(SegmentPreparedMetadata {
                segments,
                waveforms,
                meta: initial_meta,
                stage_start,
            })),
        })
        .map_err(|_| anyhow::anyhow!("ASR global decoder closed"))?;

    let runtime_options = vasr::models::qwen3asr::qwen_transcribe_options(asr_options);
    run_segment_asr_prefill_dynamic_worker(
        qwen3,
        initial_prepared,
        encoded_rx,
        runtime_handle,
        runtime_options,
        decode_idle_flush,
        decoder_tx,
        producer_id,
        prefill_log_segments,
    );
    Ok(())
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn run_global_segment_decode_worker(
    asr: &dyn crate::runtime::model::AsrModel,
    asr_options: AsrOptions,
    mut decoder_rx: SegmentDecoderReceiver,
    segment_state: &mut SegmentAsrState,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    _pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
) -> Result<()> {
    let qwen3 = asr
        .as_any()
        .downcast_ref::<Qwen3ASR>()
        .ok_or_else(|| anyhow::anyhow!("dynamic staged ASR decode requires Qwen3ASR"))?;
    let runtime_options = vasr::models::qwen3asr::qwen_transcribe_options(&asr_options);
    let (prefill_tx, prefill_rx) = std_mpsc::channel();
    let (meta_tx, meta_rx) = std_mpsc::channel();
    let mut stream_decoder = StreamingMetadataDecoder::new(Vec::new(), Vec::new());
    let mut stream_segments = Vec::new();
    let mut stream_waveforms = Vec::new();
    let mut stream_batch_lens = Vec::new();
    let mut stream_stage_starts = Vec::new();
    let mut stream_chunk_base = 0usize;
    let pending_chunks = RefCell::new(BTreeMap::new());

    let forward_thread = std::thread::spawn(move || {
        let mut next_global_chunk = 0usize;
        let mut producer_chunk_ranges: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
        let mut producer_chunk_counts: HashMap<usize, usize> = HashMap::new();
        while let Some(message) = decoder_rx.blocking_recv() {
            match message {
                SegmentPrefillMessage::Prefill {
                    producer_id,
                    result,
                } => {
                    let result = result.map(|mut output| {
                        if let Some(ranges) = producer_chunk_ranges.get(&producer_id) {
                            for seq in &mut output.sequences {
                                let local = seq.client_index;
                                if let Some((local_start, global_start, _len)) =
                                    ranges.iter().find(|(local_start, _, len)| {
                                        local >= *local_start
                                            && local < local_start.saturating_add(*len)
                                    })
                                {
                                    seq.client_index =
                                        global_start.saturating_add(local - local_start);
                                }
                            }
                            output.total_rows = next_global_chunk;
                        }
                        output
                    });
                    if prefill_tx.send(result).is_err() {
                        break;
                    }
                }
                SegmentPrefillMessage::Prepared {
                    producer_id,
                    result,
                } => {
                    if let Ok(SegmentPreparedEvent::Prepared(prepared)) = &result {
                        let chunk_count = prepared.meta.chunk_len();
                        let local_start = producer_chunk_counts.entry(producer_id).or_default();
                        producer_chunk_ranges.entry(producer_id).or_default().push((
                            *local_start,
                            next_global_chunk,
                            chunk_count,
                        ));
                        *local_start = local_start.saturating_add(chunk_count);
                        next_global_chunk = next_global_chunk.saturating_add(chunk_count);
                    }
                    if meta_tx
                        .send(SegmentPrefillMessage::Prepared {
                            producer_id,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
                SegmentPrefillMessage::Failure(failure) => {
                    if meta_tx
                        .send(SegmentPrefillMessage::Failure(failure))
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }
    });

    let mut on_complete = |chunk_index: usize, ids: Vec<u32>| -> Result<()> {
        pending_chunks.borrow_mut().insert(chunk_index, ids);
        Ok(())
    };
    let mut on_tick = || -> Result<()> {
        drain_global_decoder_meta(
            &meta_rx,
            &mut stream_decoder,
            segment_state,
            &mut stream_segments,
            &mut stream_waveforms,
            &mut stream_batch_lens,
            &mut stream_stage_starts,
            &asr_to_result_tx,
            &stats,
        )?;
        {
            let mut pending_chunks = pending_chunks.borrow_mut();
            drain_completed_chunks(
                qwen3,
                &mut pending_chunks,
                &mut stream_decoder,
                segment_state,
                stream_segments.as_mut_slice(),
                stream_waveforms.as_mut_slice(),
                stream_batch_lens.as_slice(),
                stream_stage_starts.as_slice(),
                stream_chunk_base,
                &asr_to_result_tx,
            )?;
        }
        reset_stream_decode_state_if_idle(
            &pending_chunks.borrow(),
            &mut stream_decoder,
            segment_state,
            &mut stream_segments,
            &mut stream_waveforms,
            &mut stream_batch_lens,
            &mut stream_stage_starts,
            &mut stream_chunk_base,
        );
        Ok(())
    };

    #[cfg(feature = "timing")]
    let mut timings = TranscribeTimings::default();
    let decode_result = qwen3.decode_global_prefilled_dynamic_ids_streaming(
        prefill_rx,
        runtime_options,
        &mut on_complete,
        &mut on_tick,
        #[cfg(feature = "timing")]
        Some(&mut timings.generation),
    );
    drop(on_tick);
    let _ = forward_thread.join();
    drain_global_decoder_meta(
        &meta_rx,
        &mut stream_decoder,
        segment_state,
        &mut stream_segments,
        &mut stream_waveforms,
        &mut stream_batch_lens,
        &mut stream_stage_starts,
        &asr_to_result_tx,
        &stats,
    )?;
    drain_completed_chunks(
        qwen3,
        &mut pending_chunks.borrow_mut(),
        &mut stream_decoder,
        segment_state,
        stream_segments.as_mut_slice(),
        stream_waveforms.as_mut_slice(),
        stream_batch_lens.as_slice(),
        stream_stage_starts.as_slice(),
        stream_chunk_base,
        &asr_to_result_tx,
    )?;

    decode_result?;
    if !pending_chunks.borrow().is_empty() {
        bail!(
            "dynamic ASR streaming metadata missing for {} completed chunks",
            pending_chunks.borrow().len()
        );
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
fn handle_global_decoder_message(
    message: SegmentPrefillMessage,
    decoder: &mut StreamingMetadataDecoder,
    state: &mut SegmentAsrState,
    segments: &mut Vec<AsrSegmentJob>,
    waveforms: &mut Vec<Waveform>,
    batch_lens: &mut Vec<usize>,
    stage_starts: &mut Vec<Instant>,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) -> Result<()> {
    match message {
        SegmentPrefillMessage::Prefill { .. } => Ok(()),
        SegmentPrefillMessage::Prepared {
            result: Ok(SegmentPreparedEvent::Prepared(prepared)),
            ..
        } => {
            let segment_count = prepared.segments.len();
            decoder.push_metadata(prepared.meta, prepared.waveforms.len());
            prepare_segment_job_state(state, prepared.segments.as_slice());
            batch_lens.push(prepared.waveforms.len());
            waveforms.extend(prepared.waveforms);
            segments.extend(prepared.segments);
            stage_starts.extend(std::iter::repeat_n(prepared.stage_start, segment_count));
            Ok(())
        }
        SegmentPrefillMessage::Prepared {
            result: Err(failure),
            ..
        }
        | SegmentPrefillMessage::Failure(failure) => {
            let outcomes = finish_segment_asr_batch(
                state,
                failure.segments.as_slice(),
                failure.stage_start,
                Err(failure.error),
                None,
                Some(stats),
            );
            for outcome in outcomes {
                if asr_to_result_tx.blocking_send(outcome.into()).is_err() {
                    bail!("ASR result router closed");
                }
            }
            Ok(())
        }
    }
}

#[cfg(feature = "paged-attn")]
fn drain_global_decoder_meta(
    meta_rx: &std_mpsc::Receiver<SegmentPrefillMessage>,
    decoder: &mut StreamingMetadataDecoder,
    state: &mut SegmentAsrState,
    segments: &mut Vec<AsrSegmentJob>,
    waveforms: &mut Vec<Waveform>,
    batch_lens: &mut Vec<usize>,
    stage_starts: &mut Vec<Instant>,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: &Arc<Mutex<AsrStageStats>>,
) -> Result<()> {
    for message in meta_rx.try_iter() {
        handle_global_decoder_message(
            message,
            decoder,
            state,
            segments,
            waveforms,
            batch_lens,
            stage_starts,
            asr_to_result_tx,
            stats,
        )?;
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
fn drain_completed_chunks(
    qwen3: &Qwen3ASR,
    pending_chunks: &mut BTreeMap<usize, Vec<u32>>,
    decoder: &mut StreamingMetadataDecoder,
    state: &mut SegmentAsrState,
    segments: &mut [AsrSegmentJob],
    waveforms: &mut [Waveform],
    batch_lens: &[usize],
    stage_starts: &[Instant],
    chunk_base: usize,
    asr_to_result_tx: &mpsc::Sender<RoutedTranscribeItemOutcome>,
) -> Result<()> {
    let chunk_end = chunk_base.saturating_add(decoder.chunk_count());
    let ready: Vec<usize> = pending_chunks
        .range(..chunk_end)
        .map(|(chunk_index, _)| *chunk_index)
        .collect();
    for chunk_index in ready {
        let Some(ids) = pending_chunks.remove(&chunk_index) else {
            continue;
        };
        let local_chunk_index = chunk_index.saturating_sub(chunk_base);
        if let Some(outcomes) = accept_completed_chunk(
            qwen3,
            decoder,
            state,
            segments,
            waveforms,
            batch_lens,
            stage_starts,
            local_chunk_index,
            ids,
        )? {
            for outcome in outcomes {
                if asr_to_result_tx.blocking_send(outcome.into()).is_err() {
                    bail!("ASR result router closed");
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn reset_stream_decode_state_if_idle(
    pending_chunks: &BTreeMap<usize, Vec<u32>>,
    decoder: &mut StreamingMetadataDecoder,
    state: &SegmentAsrState,
    segments: &mut Vec<AsrSegmentJob>,
    waveforms: &mut Vec<Waveform>,
    batch_lens: &mut Vec<usize>,
    stage_starts: &mut Vec<Instant>,
    chunk_base: &mut usize,
) {
    if !pending_chunks.is_empty() || !state.jobs.is_empty() || decoder.chunk_count() == 0 {
        return;
    }
    *chunk_base = chunk_base.saturating_add(decoder.chunk_count());
    *decoder = StreamingMetadataDecoder::new(Vec::new(), Vec::new());
    segments.clear();
    waveforms.clear();
    batch_lens.clear();
    stage_starts.clear();
}

#[cfg(feature = "paged-attn")]
fn accept_completed_chunk(
    qwen3: &Qwen3ASR,
    decoder: &mut StreamingMetadataDecoder,
    state: &mut SegmentAsrState,
    segments: &mut [AsrSegmentJob],
    waveforms: &mut [Waveform],
    batch_lens: &[usize],
    stage_starts: &[Instant],
    chunk_index: usize,
    ids: Vec<u32>,
) -> Result<Option<Vec<RoutedTranscribeItemOutcome>>> {
    let Some((batch_index, audio_index, output)) =
        decoder.accept_chunk(qwen3.processor(), chunk_index, ids)?
    else {
        return Ok(None);
    };
    let segment_index = batch_lens
        .iter()
        .take(batch_index)
        .sum::<usize>()
        .saturating_add(audio_index);
    let Some(segment) = segments.get(segment_index) else {
        bail!(
            "dynamic ASR streaming segment index out of range: batch={} audio={} segment_index={} segments={}",
            batch_index,
            audio_index,
            segment_index,
            segments.len()
        );
    };
    let stage_start = stage_starts
        .get(segment_index)
        .copied()
        .unwrap_or_else(Instant::now);
    let mut timelines =
        qwen_outputs_to_timelines(vec![output], &waveforms[segment_index..segment_index + 1])?;
    let timeline = timelines
        .pop()
        .ok_or_else(|| anyhow::anyhow!("missing streaming ASR timeline"))?;
    let batch = SegmentAsrInference {
        targets: vec![(segment.job_index, segment.segment_index, segment.offset)],
        segment_audio_seconds: segment.waveform.duration_seconds(),
        active_job_count: 1,
    };
    let outcomes = finish_segment_asr_batch(
        state,
        std::slice::from_ref(segment),
        stage_start,
        Ok((batch, vec![timeline])),
        None,
        None,
    );
    release_stream_segment_audio(segments, waveforms, segment_index);
    Ok(Some(outcomes))
}

#[cfg(feature = "paged-attn")]
fn release_stream_segment_audio(
    segments: &mut [AsrSegmentJob],
    waveforms: &mut [Waveform],
    segment_index: usize,
) {
    if let Some(segment) = segments.get_mut(segment_index) {
        segment.waveform.samples = Vec::new();
        segment.waveform.encoded_data = None;
    }
    if let Some(waveform) = waveforms.get_mut(segment_index) {
        waveform.samples = Vec::new();
        waveform.encoded_data = None;
    }
}

#[cfg(feature = "paged-attn")]
#[allow(clippy::too_many_arguments)]
fn run_segment_asr_prefill_dynamic_worker(
    qwen3: &Qwen3ASR,
    initial_prepared: PreparedAudioBatch,
    encoded_rx: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: tokio::runtime::Handle,
    runtime_options: vasr::models::qwen3asr::TranscribeOptions,
    decode_idle_flush: Option<std::time::Duration>,
    decoder_tx: mpsc::UnboundedSender<SegmentPrefillMessage>,
    producer_id: usize,
    prefill_log_segments: Vec<SegmentLogItem>,
) {
    let run = (|| -> Result<()> {
        let prefill_log_segments = Mutex::new(prefill_log_segments);
        let mut source_closed = false;
        let mut poll_prepared = |may_block: bool| -> Result<PreparedAudioBatchPoll> {
            if decode_idle_flush.is_some() && !may_block {
                return Ok(PreparedAudioBatchPoll::Empty);
            }
            loop {
                if source_closed {
                    return Ok(PreparedAudioBatchPoll::Closed);
                }
                let message = if may_block {
                    recv_segment_encoded_message_blocking(
                        &encoded_rx,
                        &runtime_handle,
                        decode_idle_flush,
                        &mut source_closed,
                    )
                } else {
                    try_recv_segment_encoded_message(
                        &encoded_rx,
                        &runtime_handle,
                        &mut source_closed,
                    )
                };
                let Some(message) = message else {
                    return Ok(PreparedAudioBatchPoll::Empty);
                };
                match prepare_coalesced_segment_prefill_batch(
                    qwen3,
                    message,
                    &encoded_rx,
                    &runtime_handle,
                    &decoder_tx,
                    producer_id,
                    &prefill_log_segments,
                )? {
                    PrefillCoalescedPoll::Ready(coalesced) => {
                        return Ok(PreparedAudioBatchPoll::Ready(coalesced.prepared));
                    }
                    PrefillCoalescedPoll::Empty => return Ok(PreparedAudioBatchPoll::Empty),
                    PrefillCoalescedPoll::Closed => return Ok(PreparedAudioBatchPoll::Closed),
                }
            }
        };
        let mut producer = qwen3.dynamic_prefill_producer(
            initial_prepared,
            &mut poll_prepared,
            runtime_options,
        )?;
        loop {
            let stage_start = Instant::now();
            let message = producer.run_stage(true)?;
            #[cfg(feature = "timing")]
            let elapsed_seconds = message.timings.prefill_us as f64 / 1_000_000.0;
            #[cfg(not(feature = "timing"))]
            let elapsed_seconds = stage_start.elapsed().as_secs_f64();
            #[cfg(feature = "timing")]
            let _ = stage_start;
            let wave = message.prefill_wave;
            let done = producer.is_done();
            if wave > 0 {
                log_prefill_batch_done(
                    &prefill_log_segments
                        .lock()
                        .map_err(|_| anyhow::anyhow!("prefill log segments lock poisoned"))?,
                    &message,
                    elapsed_seconds,
                );
            }
            if wave > 0
                && decoder_tx
                    .send(SegmentPrefillMessage::Prefill {
                        producer_id,
                        result: Ok(message),
                    })
                    .is_err()
            {
                return Ok(());
            }
            if done {
                break;
            }
            if wave == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        Ok(())
    })();
    if let Err(error) = run {
        let _ = decoder_tx.send(SegmentPrefillMessage::Prefill {
            producer_id,
            result: Err(error),
        });
    }
}

#[cfg(feature = "paged-attn")]
fn prepare_coalesced_segment_prefill_batch(
    qwen3: &Qwen3ASR,
    first: SegmentEncoderOutput,
    encoded_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: &tokio::runtime::Handle,
    decoder_tx: &mpsc::UnboundedSender<SegmentPrefillMessage>,
    producer_id: usize,
    prefill_log_segments: &Mutex<Vec<SegmentLogItem>>,
) -> Result<PrefillCoalescedPoll> {
    let mut prepared_batches = Vec::new();

    let mut current = Some(first);
    let deadline = std::time::Instant::now() + asr_prefill_coalesce_wait();
    loop {
        let Some(message) = current.take() else {
            break;
        };
        if !prepare_segment_encoded_for_prefill(
            qwen3,
            message,
            decoder_tx,
            producer_id,
            prefill_log_segments,
            &mut prepared_batches,
        )? {
            return Ok(PrefillCoalescedPoll::Closed);
        }

        if std::time::Instant::now() >= deadline {
            break;
        }
        current =
            try_recv_segment_encoded_message_for_coalesce(encoded_rx, runtime_handle, deadline);
        if current.is_none() {
            break;
        }
    }

    if prepared_batches.is_empty() {
        return Ok(PrefillCoalescedPoll::Empty);
    }

    Ok(PrefillCoalescedPoll::Ready(PrefillCoalescedPrepared {
        prepared: merge_prepared_audio_batches(prepared_batches)?,
    }))
}

#[cfg(feature = "paged-attn")]
fn prepare_segment_encoded_for_prefill(
    qwen3: &Qwen3ASR,
    message: SegmentEncoderOutput,
    decoder_tx: &mpsc::UnboundedSender<SegmentPrefillMessage>,
    producer_id: usize,
    prefill_log_segments: &Mutex<Vec<SegmentLogItem>>,
    prepared_batches: &mut Vec<PreparedAudioBatch>,
) -> Result<bool> {
    let SegmentEncoderOutput::Encoded(message) = message;
    match message.batch {
        Ok(batch) => {
            let (_next_inference, next_waveforms, next_encoded) = batch.into_parts();
            let (prepared, meta) = qwen3.prepare_encoded_audio_features(next_encoded)?;
            prefill_log_segments
                .lock()
                .map_err(|_| anyhow::anyhow!("prefill log segments lock poisoned"))?
                .extend(segment_log_items(&message.segments));
            if decoder_tx
                .send(SegmentPrefillMessage::Prepared {
                    producer_id,
                    result: Ok(SegmentPreparedEvent::Prepared(SegmentPreparedMetadata {
                        segments: message.segments,
                        waveforms: next_waveforms,
                        meta,
                        stage_start: message.stage_start,
                    })),
                })
                .is_err()
            {
                return Ok(false);
            }
            prepared_batches.push(prepared);
        }
        Err(error) => {
            if decoder_tx
                .send(SegmentPrefillMessage::Prepared {
                    producer_id,
                    result: Err(SegmentEncodedFailure {
                        segments: message.segments,
                        stage_start: message.stage_start,
                        error,
                    }),
                })
                .is_err()
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(feature = "paged-attn")]
fn try_recv_segment_encoded_message_for_coalesce(
    encoded_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: &tokio::runtime::Handle,
    deadline: std::time::Instant,
) -> Option<SegmentEncoderOutput> {
    let now = std::time::Instant::now();
    if now >= deadline {
        return None;
    }
    let remaining = deadline.saturating_duration_since(now);
    match runtime_handle.block_on(async {
        let mut encoded_rx = encoded_rx.lock().await;
        tokio::time::timeout(remaining, encoded_rx.recv()).await
    }) {
        Ok(Some(message)) => Some(message),
        Ok(None) | Err(tokio::time::error::Elapsed { .. }) => None,
    }
}

#[cfg(feature = "paged-attn")]
fn merge_prepared_audio_batches(batches: Vec<PreparedAudioBatch>) -> Result<PreparedAudioBatch> {
    let mut prepared = Vec::new();
    let mut audio_seconds = Vec::new();
    let mut audio_features = Vec::new();
    for batch in batches {
        prepared.extend(batch.prepared);
        audio_seconds.extend(batch.audio_seconds);
        audio_features.push(batch.audio_features);
    }

    let audio_features = if audio_features.len() == 1 {
        audio_features
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing prepared audio features"))?
    } else {
        let first_dims = audio_features
            .first()
            .map(|tensor| tensor.dims().to_vec())
            .unwrap_or_default();
        if first_dims.len() != 2 {
            bail!("prepared audio features must be rank-2, got dims={first_dims:?}");
        }
        candle_core::Tensor::cat(audio_features.as_slice(), 0)?
    };

    Ok(PreparedAudioBatch {
        prepared,
        audio_features,
        audio_seconds,
    })
}

#[cfg(feature = "paged-attn")]
fn recv_segment_encoded_message_blocking(
    encoded_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: &tokio::runtime::Handle,
    timeout: Option<std::time::Duration>,
    source_closed: &mut bool,
) -> Option<SegmentEncoderOutput> {
    if let Some(timeout) = timeout {
        match runtime_handle.block_on(async {
            let mut encoded_rx = encoded_rx.lock().await;
            tokio::time::timeout(timeout, encoded_rx.recv()).await
        }) {
            Ok(Some(message)) => Some(message),
            Ok(None) | Err(tokio::time::error::Elapsed { .. }) => {
                *source_closed = true;
                None
            }
        }
    } else {
        match runtime_handle.block_on(async {
            let mut encoded_rx = encoded_rx.lock().await;
            encoded_rx.recv().await
        }) {
            Some(message) => Some(message),
            None => {
                *source_closed = true;
                None
            }
        }
    }
}

#[cfg(feature = "paged-attn")]
fn try_recv_segment_encoded_message(
    encoded_rx: &Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<SegmentEncoderOutput>>>,
    runtime_handle: &tokio::runtime::Handle,
    source_closed: &mut bool,
) -> Option<SegmentEncoderOutput> {
    match runtime_handle.block_on(async {
        let mut encoded_rx = encoded_rx.lock().await;
        encoded_rx.try_recv()
    }) {
        Ok(message) => Some(message),
        Err(mpsc::error::TryRecvError::Empty) => None,
        Err(mpsc::error::TryRecvError::Disconnected) => {
            *source_closed = true;
            None
        }
    }
}

#[cfg(feature = "paged-attn")]
fn send_segment_encode_request(
    asr_to_encoder_tx: &mpsc::UnboundedSender<SegmentEncodeMessage>,
    request: SegmentEncodeRequest,
) {
    if asr_to_encoder_tx
        .send(SegmentEncodeMessage::Segment(request))
        .is_err()
    {
        tracing::warn!(
            target: "vasr_transcribe::pipeline",
            "ASR audio feature encoder closed"
        );
    }
}

fn spawn_asr_worker(
    offline: Arc<OfflinePipeline>,
    asr_options: AsrOptions,
    mut vad_to_asr_rx: mpsc::UnboundedReceiver<VadToAsrMessage>,
    asr_to_result_tx: mpsc::Sender<RoutedTranscribeItemOutcome>,
    stats: Arc<Mutex<AsrStageStats>>,
    decode_idle_flush: Option<std::time::Duration>,
    pending_requests: Option<Arc<Mutex<HashMap<PipelineRequestId, PendingGlobalRequest>>>>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        #[cfg(not(feature = "paged-attn"))]
        let _ = (decode_idle_flush, &pending_requests);
        let mut pending = VecDeque::new();
        #[cfg(feature = "paged-attn")]
        let (asr_to_encoder_tx, asr_to_encoder_rx) = mpsc::unbounded_channel();
        #[cfg(feature = "paged-attn")]
        let (encoder_to_prefill_tx, encoder_to_prefill_rx) = mpsc::unbounded_channel();
        #[cfg(feature = "paged-attn")]
        let segment_encoder_handle = spawn_segment_encoder_worker(
            Arc::clone(&offline),
            asr_options.clone(),
            asr_to_encoder_rx,
            encoder_to_prefill_tx,
            Arc::clone(&stats),
            pending_requests.clone(),
        );
        #[cfg(feature = "paged-attn")]
        let segment_prefill_and_decoder_handles = {
            let (prefill_to_decoder_tx, prefill_to_decoder_rx) = mpsc::unbounded_channel();
            let prefill_handle = spawn_segment_prefill_worker(
                Arc::clone(&offline),
                asr_options.clone(),
                encoder_to_prefill_rx,
                prefill_to_decoder_tx,
                decode_idle_flush,
            );
            let decoder_handle = spawn_segment_decoder_worker(
                Arc::clone(&offline),
                asr_options.clone(),
                prefill_to_decoder_rx,
                asr_to_result_tx.clone(),
                Arc::clone(&stats),
                pending_requests.clone(),
            );
            (prefill_handle, decoder_handle)
        };
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
                    let messages =
                        collect_job_microbatch(job, &mut vad_to_asr_rx, &mut pending).await;
                    let mut jobs = Vec::new();
                    for message in messages {
                        match message {
                            Err(outcome) => {
                                let _ = asr_to_result_tx.send(outcome.into()).await;
                            }
                            Ok(AsrPipelineMessage::Job(job)) => jobs.push(job),
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
                                channel: "mono",
                                result: Err(anyhow::anyhow!("ASR join error: {error}")),
                                audio_seconds: 0.0,
                                timing: TranscribeItemTiming::default(),
                                spans: TranscribeItemSpans::default(),
                                bad_component: Some("recognizer"),
                            }
                            .into(),
                        ]
                    });
                    for outcome in outcomes {
                        let _ = asr_to_result_tx.send(outcome).await;
                    }
                }
                #[cfg(feature = "paged-attn")]
                Ok(AsrPipelineMessage::Segment(segment)) => {
                    let stage_start = Instant::now();
                    let request = SegmentEncodeRequest {
                        segment,
                        stage_start,
                    };
                    send_segment_encode_request(&asr_to_encoder_tx, request);
                }
            }
        }

        #[cfg(feature = "paged-attn")]
        drop(asr_to_encoder_tx);
        #[cfg(feature = "paged-attn")]
        let _ = segment_encoder_handle.await;
        #[cfg(feature = "paged-attn")]
        {
            let (segment_prefill_handle, segment_decoder_handle) =
                segment_prefill_and_decoder_handles;
            let _ = segment_prefill_handle.await;
            let _ = segment_decoder_handle.await;
        }
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
    loop {
        match tokio::time::timeout(
            std::time::Duration::from_millis(ASR_MICROBATCH_COLLECT_WAIT_MS),
            vad_to_asr_rx.recv(),
        )
        .await
        {
            Ok(Some(envelope)) => match envelope.message {
                #[cfg(feature = "paged-attn")]
                Ok(AsrPipelineMessage::Segment(_)) => {
                    _pending.push_back(envelope);
                }
                Ok(AsrPipelineMessage::Job(job)) => {
                    messages.push(Ok(AsrPipelineMessage::Job(job)));
                }
                message => messages.push(message),
            },
            Ok(None) | Err(tokio::time::error::Elapsed { .. }) => break,
        }
    }
    messages
}

#[cfg(feature = "paged-attn")]
fn prepare_segment_job_state(state: &mut SegmentAsrState, segments: &[AsrSegmentJob]) {
    for segment in segments {
        let entry = state
            .jobs
            .entry(segment.job_index)
            .or_insert_with(|| SegmentJobState {
                index: segment.job_index,
                request_item: segment.request_item,
                channel: segment.channel,
                timing: segment.timing,
                spans: segment.spans,
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

#[cfg(feature = "paged-attn")]
fn finish_segment_job(job: SegmentJobState) -> TranscribeItemOutcome {
    let mut timeline = Timeline::new("offline_audio");
    timeline.annotations.extend(job.speech_annotations);
    for annotations in job.segment_annotations.into_iter().flatten() {
        timeline.annotations.extend(annotations);
    }
    TranscribeItemOutcome {
        index: job.index,
        channel: job.channel,
        result: Ok(timeline),
        audio_seconds: job.audio_seconds,
        timing: job.timing,
        spans: job.spans,
        bad_component: None,
    }
}

#[cfg(feature = "paged-attn")]
fn routed_segment_job_outcome(job: SegmentJobState) -> RoutedTranscribeItemOutcome {
    let request_item = job.request_item;
    let outcome = finish_segment_job(job);
    match request_item {
        Some(request_item) => RoutedTranscribeItemOutcome::for_request(request_item, outcome),
        None => outcome.into(),
    }
}

#[derive(Debug, Clone)]
#[cfg(feature = "paged-attn")]
struct SegmentAsrInference {
    targets: Vec<(usize, usize, DurationMs)>,
    segment_audio_seconds: f64,
    active_job_count: usize,
}

#[derive(Debug)]
#[cfg(feature = "paged-attn")]
struct SegmentEncodedAsrBatch {
    inference: SegmentAsrInference,
    waveforms: Vec<Waveform>,
    encoded: EncodedAsrBatch,
    timings: AudioFeatureEncodeTimings,
}

#[cfg(feature = "paged-attn")]
impl SegmentEncodedAsrBatch {
    fn into_parts(self) -> (SegmentAsrInference, Vec<Waveform>, EncodedAsrBatch) {
        (self.inference, self.waveforms, self.encoded)
    }
}

fn run_raw_asr_inference(
    offline: &OfflinePipeline,
    waveforms: &[Waveform],
    asr_options: &AsrOptions,
) -> Result<(Vec<Timeline>, Option<TranscribeItemTiming>)> {
    #[cfg(feature = "timing")]
    if let Some(qwen3) = offline.asr.as_any().downcast_ref::<Qwen3ASR>() {
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

    offline
        .asr
        .transcribe(waveforms, asr_options)
        .map(|timelines| (timelines, None))
}

#[cfg(feature = "paged-attn")]
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

#[cfg(feature = "paged-attn")]
fn finish_segment_asr_batch(
    state: &mut SegmentAsrState,
    segments: &[AsrSegmentJob],
    stage_start: Instant,
    batch: Result<(SegmentAsrInference, Vec<Timeline>)>,
    asr_timing: Option<TranscribeItemTiming>,
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
            let asr_span = StageSpan::new(stage_start, Instant::now());
            let prefill_span = prefill_span_from_decode(asr_span, asr_timing);
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
                let target_count = inference.targets.len();
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
                        let segment_audio_seconds = segments
                            .iter()
                            .find(|segment| {
                                segment.job_index == job_index
                                    && segment.segment_index == segment_index
                            })
                            .map(|segment| segment.waveform.duration_seconds())
                            .unwrap_or(0.0);
                        job.timing.asr_seconds += allocated_batch_seconds(
                            segment_audio_seconds,
                            inference.segment_audio_seconds,
                            target_count,
                            elapsed,
                        );
                        if let Some(asr_timing) = asr_timing {
                            let scale = allocation_scale(
                                segment_audio_seconds,
                                inference.segment_audio_seconds,
                                target_count,
                            );
                            job.timing.add_scaled(asr_timing, scale);
                        }
                        job.spans.asr = Some(merge_optional_span(job.spans.asr, Some(asr_span)));
                        job.spans.prefill =
                            Some(merge_optional_span(job.spans.prefill, prefill_span));
                        job.spans.decode =
                            Some(merge_optional_span(job.spans.decode, Some(asr_span)));
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
                let affected_count = affected.len();
                for job_index in affected {
                    if let Some(job) = state.jobs.remove(&job_index) {
                        let request_item = job.request_item;
                        let mut timing = job.timing;
                        timing.asr_seconds += allocated_batch_seconds(
                            segments
                                .iter()
                                .filter(|segment| segment.job_index == job_index)
                                .map(|segment| segment.waveform.duration_seconds())
                                .sum(),
                            inference.segment_audio_seconds,
                            affected_count,
                            elapsed,
                        );
                        if let Some(asr_timing) = asr_timing {
                            let item_audio_seconds = segments
                                .iter()
                                .filter(|segment| segment.job_index == job_index)
                                .map(|segment| segment.waveform.duration_seconds())
                                .sum();
                            let scale = allocation_scale(
                                item_audio_seconds,
                                inference.segment_audio_seconds,
                                affected_count,
                            );
                            timing.add_scaled(asr_timing, scale);
                        }
                        let mut spans = job.spans;
                        spans.asr = Some(merge_optional_span(spans.asr, Some(asr_span)));
                        spans.prefill = Some(merge_optional_span(spans.prefill, prefill_span));
                        spans.decode = Some(merge_optional_span(spans.decode, Some(asr_span)));
                        let outcome = outcome_from_result(
                            job.index,
                            job.channel,
                            job.audio_seconds,
                            timing,
                            spans,
                            Err(anyhow::anyhow!(err.clone())),
                        );
                        completed.push(route_outcome(request_item, outcome));
                    }
                }
            }
        }
        Err(error) => {
            let elapsed = stage_start.elapsed().as_secs_f64();
            let asr_span = StageSpan::new(stage_start, Instant::now());
            let prefill_span = prefill_span_from_decode(asr_span, asr_timing);
            let reason = error.to_string();
            let affected: HashSet<usize> =
                segments.iter().map(|segment| segment.job_index).collect();
            let total_audio_seconds: f64 = segments
                .iter()
                .map(|segment| segment.waveform.duration_seconds())
                .sum();
            for job_index in affected {
                if let Some(job) = state.jobs.remove(&job_index) {
                    let request_item = job.request_item;
                    let mut timing = job.timing;
                    let mut spans = job.spans;
                    spans.asr = Some(merge_optional_span(spans.asr, Some(asr_span)));
                    spans.prefill = Some(merge_optional_span(spans.prefill, prefill_span));
                    spans.decode = Some(merge_optional_span(spans.decode, Some(asr_span)));
                    timing.asr_seconds += allocated_batch_seconds(
                        segments
                            .iter()
                            .filter(|segment| segment.job_index == job_index)
                            .map(|segment| segment.waveform.duration_seconds())
                            .sum(),
                        total_audio_seconds,
                        1,
                        elapsed,
                    );
                    let outcome = outcome_from_result(
                        job.index,
                        job.channel,
                        job.audio_seconds,
                        timing,
                        spans,
                        Err(anyhow::anyhow!(reason.clone())),
                    );
                    completed.push(route_outcome(request_item, outcome));
                }
            }
        }
    }

    completed
}

fn run_asr_jobs_batch(
    offline: &OfflinePipeline,
    jobs: Vec<AsrJob>,
    asr_options: &AsrOptions,
    stats: Option<&Arc<Mutex<AsrStageStats>>>,
) -> Vec<RoutedTranscribeItemOutcome> {
    let mut outcomes = Vec::with_capacity(jobs.len());
    let mut raw_jobs: Vec<(AsrJob, Vec<Annotation>)> = Vec::new();

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
                    result: Ok(Timeline::new("offline_audio")),
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
        let waveforms: Vec<Waveform> = raw_jobs.into_iter().map(|(job, _)| job.waveform).collect();
        let stage_start = Instant::now();
        let result = run_raw_asr_inference(offline, &waveforms, asr_options);
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
                    mut annotations,
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
                    if !annotations.is_empty() {
                        annotations.extend(timeline.annotations);
                        timeline.annotations = annotations;
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

fn allocated_batch_seconds(
    item_audio_seconds: f64,
    total_audio_seconds: f64,
    item_count: usize,
    elapsed_seconds: f64,
) -> f64 {
    if elapsed_seconds <= 0.0 {
        return 0.0;
    }
    if total_audio_seconds > 0.0 && item_audio_seconds > 0.0 {
        return elapsed_seconds * item_audio_seconds / total_audio_seconds;
    }
    if item_count > 0 {
        return elapsed_seconds / item_count as f64;
    }
    elapsed_seconds
}

fn merge_optional_span(left: Option<StageSpan>, right: Option<StageSpan>) -> StageSpan {
    match (left, right) {
        (Some(left), Some(right)) => StageSpan {
            start: left.start.min(right.start),
            end: left.end.max(right.end),
        },
        (Some(span), None) | (None, Some(span)) => span,
        (None, None) => {
            let now = Instant::now();
            StageSpan::new(now, now)
        }
    }
}

fn merge_stage_span(left: Option<StageSpan>, right: Option<StageSpan>) -> Option<StageSpan> {
    match (left, right) {
        (Some(left), Some(right)) => Some(StageSpan {
            start: left.start.min(right.start),
            end: left.end.max(right.end),
        }),
        (Some(span), None) | (None, Some(span)) => Some(span),
        (None, None) => None,
    }
}

fn merged_spans_seconds(mut spans: Vec<StageSpan>) -> f64 {
    if spans.is_empty() {
        return 0.0;
    }
    spans.sort_by_key(|span| span.start);
    let mut total = 0.0;
    let mut current = spans[0];
    for span in spans.into_iter().skip(1) {
        if span.start <= current.end {
            if span.end > current.end {
                current.end = span.end;
            }
        } else {
            total += current.seconds();
            current = span;
        }
    }
    total + current.seconds()
}

fn prefill_span_from_decode(
    decode_span: StageSpan,
    timing: Option<TranscribeItemTiming>,
) -> Option<StageSpan> {
    let prefill_seconds = timing?.prefill_seconds;
    if prefill_seconds <= 0.0 {
        return None;
    }
    let prefill_duration = Duration::from_secs_f64(prefill_seconds);
    let decode_duration = decode_span.end.duration_since(decode_span.start);
    let duration = prefill_duration.min(decode_duration);
    Some(StageSpan::new(
        decode_span.start,
        decode_span.start + duration,
    ))
}

#[cfg(feature = "paged-attn")]
fn apply_batch_timing_to_segments(segments: &mut [AsrSegmentJob], timing: TranscribeItemTiming) {
    if segments.is_empty() {
        return;
    }
    let total_audio_seconds: f64 = segments
        .iter()
        .map(|segment| segment.waveform.duration_seconds())
        .sum();
    let item_count = segments.len();
    for segment in segments {
        let scale = allocation_scale(
            segment.waveform.duration_seconds(),
            total_audio_seconds,
            item_count,
        );
        segment.timing.add_scaled(timing, scale);
    }
}

fn allocation_scale(item_audio_seconds: f64, total_audio_seconds: f64, item_count: usize) -> f64 {
    if total_audio_seconds > 0.0 && item_audio_seconds > 0.0 {
        item_audio_seconds / total_audio_seconds
    } else if item_count > 0 {
        1.0 / item_count as f64
    } else {
        1.0
    }
}

#[cfg(feature = "paged-attn")]
fn audio_encoder_timing_seconds(timings: AudioFeatureEncodeTimings) -> TranscribeItemTiming {
    TranscribeItemTiming {
        audio_encoder_seconds: timing_us_to_seconds(
            timings
                .prepare_us
                .saturating_add(timings.stack_us)
                .saturating_add(timings.forward_us)
                .saturating_add(timings.d2h_us),
        ),
        audio_encoder_prepare_seconds: timing_us_to_seconds(timings.prepare_us),
        audio_encoder_stack_seconds: timing_us_to_seconds(timings.stack_us),
        audio_encoder_forward_seconds: timing_us_to_seconds(timings.forward_us),
        audio_encoder_d2h_seconds: timing_us_to_seconds(timings.d2h_us),
        ..Default::default()
    }
}

#[cfg(feature = "timing")]
fn transcribe_timing_seconds(timings: &TranscribeTimings) -> TranscribeItemTiming {
    let generation = &timings.generation;
    TranscribeItemTiming {
        audio_encoder_seconds: timing_us_to_seconds(timings.audio_encoder_us),
        prefill_seconds: timing_us_to_seconds(generation.prefill_us),
        prefill_inputs_seconds: timing_us_to_seconds(generation.prefill_inputs_us),
        prefill_rope_seconds: timing_us_to_seconds(generation.prefill_rope_us),
        prefill_metadata_seconds: timing_us_to_seconds(generation.prefill_metadata_us),
        prefill_mask_seconds: timing_us_to_seconds(generation.prefill_mask_us),
        prefill_forward_seconds: timing_us_to_seconds(generation.prefill_forward_us),
        prefill_gather_seconds: timing_us_to_seconds(generation.prefill_gather_us),
        prefill_decode_setup_seconds: timing_us_to_seconds(generation.prefill_decode_setup_us),
        prefill_argmax_seconds: timing_us_to_seconds(generation.prefill_argmax_us),
        decode_seconds: timing_us_to_seconds(generation.decode_us),
        decode_token_tensor_seconds: timing_us_to_seconds(generation.decode_token_tensor_us),
        decode_embed_seconds: timing_us_to_seconds(generation.decode_embed_us),
        decode_position_seconds: timing_us_to_seconds(generation.decode_position_us),
        decode_metadata_seconds: timing_us_to_seconds(generation.decode_metadata_us),
        decode_graph_replay_seconds: timing_us_to_seconds(generation.decode_graph_replay_us),
        decode_forward_seconds: timing_us_to_seconds(generation.decode_forward_us),
        decode_pre_argmax_sync_seconds: timing_us_to_seconds(generation.decode_pre_argmax_sync_us),
        decode_argmax_seconds: timing_us_to_seconds(generation.decode_argmax_us),
        tokenizer_decode_seconds: timing_us_to_seconds(timings.tokenizer_decode_us),
        ..Default::default()
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

fn record_loader_task_stats(stats: &Arc<Mutex<LoaderStageStats>>, task: &LoaderTaskStats) {
    if let Ok(mut stats) = stats.lock() {
        stats.jobs += task.jobs;
        stats.failed += task.failed;
        stats.audio_seconds += task.audio_seconds;
        stats.compute_seconds += task.compute_seconds;
        *stats
            .worker_compute_seconds
            .entry(task.worker_id)
            .or_default() += task.compute_seconds;
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
        *stats
            .worker_compute_seconds
            .entry(task.worker_id)
            .or_default() += task.compute_seconds;
        stats.send_wait_seconds += task.send_wait_seconds;
        stats.wall_seconds += task.wall_seconds;
    }
}

#[cfg(feature = "paged-attn")]
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
#[allow(dead_code)]
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

#[allow(dead_code)]
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

fn spawn_component_throughput_reporter(
    service_id: Arc<Mutex<String>>,
    loader_stats: Arc<Mutex<LoaderStageStats>>,
    vad_stats: Arc<Mutex<VadStageStats>>,
    asr_stats: Arc<Mutex<AsrStageStats>>,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker.tick().await;
        let mut last_instant = Instant::now();
        let mut last_loader = snapshot_loader_stats(&loader_stats);
        let mut last_vad = snapshot_vad_stats(&vad_stats);
        let mut last_asr = snapshot_asr_stats(&asr_stats);
        loop {
            ticker.tick().await;
            let now = Instant::now();
            let window_seconds = now.duration_since(last_instant).as_secs_f64();
            last_instant = now;

            let loader = snapshot_loader_stats(&loader_stats);
            let vad = snapshot_vad_stats(&vad_stats);
            let asr = snapshot_asr_stats(&asr_stats);

            let loader_delta = loader_delta(last_loader.clone(), loader.clone());
            let vad_delta = vad_delta(last_vad.clone(), vad.clone());
            let asr_delta = asr_delta(last_asr, asr);
            last_loader = loader;
            last_vad = vad;
            last_asr = asr;

            if loader_delta.jobs == 0
                && loader_delta.failed == 0
                && vad_delta.jobs == 0
                && vad_delta.failed == 0
                && asr_delta.encode_batches == 0
                && asr_delta.decode_batches == 0
                && asr_delta.batches == 0
            {
                continue;
            }

            let service_id = service_id
                .lock()
                .map(|service_id| service_id.clone())
                .unwrap_or_else(|_| String::from("service"));
            log_component_throughput_window(
                &service_id,
                window_seconds,
                loader_delta,
                vad_delta,
                asr_delta,
            );
        }
    })
}

fn snapshot_loader_stats(stats: &Arc<Mutex<LoaderStageStats>>) -> LoaderStageStatsSnapshot {
    stats
        .lock()
        .map(|stats| LoaderStageStatsSnapshot {
            jobs: stats.jobs,
            failed: stats.failed,
            audio_seconds: stats.audio_seconds,
            compute_seconds: stats.compute_seconds,
            worker_compute_seconds: stats.worker_compute_seconds.clone(),
        })
        .unwrap_or_default()
}

fn snapshot_vad_stats(stats: &Arc<Mutex<VadStageStats>>) -> VadStageStatsSnapshot {
    stats
        .lock()
        .map(|stats| VadStageStatsSnapshot {
            jobs: stats.jobs,
            failed: stats.failed,
            no_speech: stats.no_speech,
            segments: stats.segments,
            audio_seconds: stats.audio_seconds,
            speech_seconds: stats.speech_seconds,
            compute_seconds: stats.compute_seconds,
            worker_compute_seconds: stats.worker_compute_seconds.clone(),
            send_wait_seconds: stats.send_wait_seconds,
        })
        .unwrap_or_default()
}

fn snapshot_asr_stats(stats: &Arc<Mutex<AsrStageStats>>) -> AsrStageStatsSnapshot {
    stats
        .lock()
        .map(|stats| AsrStageStatsSnapshot {
            batches: stats.batches,
            jobs: stats.jobs,
            segments: stats.segments,
            audio_seconds: stats.audio_seconds,
            wall_seconds: stats.wall_seconds,
            encode_batches: stats.encode_batches,
            encode_segments: stats.encode_segments,
            encode_audio_seconds: stats.encode_audio_seconds,
            encode_seconds: stats.encode_seconds,
            decode_batches: stats.decode_batches,
            decode_segments: stats.decode_segments,
            decode_audio_seconds: stats.decode_audio_seconds,
            decode_seconds: stats.decode_seconds,
            audio_encoder_seconds: stats.audio_encoder_seconds,
            prefill_seconds: stats.prefill_seconds,
            prefill_tokens: stats.prefill_tokens,
            decode_model_seconds: stats.decode_model_seconds,
            decode_tokens: stats.decode_tokens,
            decode_steps: stats.decode_steps,
        })
        .unwrap_or_default()
}

fn loader_delta(
    before: LoaderStageStatsSnapshot,
    after: LoaderStageStatsSnapshot,
) -> LoaderStageStatsSnapshot {
    LoaderStageStatsSnapshot {
        jobs: after.jobs.saturating_sub(before.jobs),
        failed: after.failed.saturating_sub(before.failed),
        audio_seconds: (after.audio_seconds - before.audio_seconds).max(0.0),
        compute_seconds: (after.compute_seconds - before.compute_seconds).max(0.0),
        worker_compute_seconds: worker_compute_delta(
            before.worker_compute_seconds,
            after.worker_compute_seconds,
        ),
    }
}

fn vad_delta(before: VadStageStatsSnapshot, after: VadStageStatsSnapshot) -> VadStageStatsSnapshot {
    VadStageStatsSnapshot {
        jobs: after.jobs.saturating_sub(before.jobs),
        failed: after.failed.saturating_sub(before.failed),
        no_speech: after.no_speech.saturating_sub(before.no_speech),
        segments: after.segments.saturating_sub(before.segments),
        audio_seconds: (after.audio_seconds - before.audio_seconds).max(0.0),
        speech_seconds: (after.speech_seconds - before.speech_seconds).max(0.0),
        compute_seconds: (after.compute_seconds - before.compute_seconds).max(0.0),
        worker_compute_seconds: worker_compute_delta(
            before.worker_compute_seconds,
            after.worker_compute_seconds,
        ),
        send_wait_seconds: (after.send_wait_seconds - before.send_wait_seconds).max(0.0),
    }
}

fn worker_compute_delta(
    before: HashMap<usize, f64>,
    after: HashMap<usize, f64>,
) -> HashMap<usize, f64> {
    after
        .into_iter()
        .map(|(worker_id, seconds)| {
            let before_seconds = before.get(&worker_id).copied().unwrap_or_default();
            (worker_id, (seconds - before_seconds).max(0.0))
        })
        .collect()
}

fn max_worker_compute_seconds(worker_seconds: &HashMap<usize, f64>) -> f64 {
    worker_seconds.values().copied().fold(0.0, f64::max)
}

fn asr_delta(before: AsrStageStatsSnapshot, after: AsrStageStatsSnapshot) -> AsrStageStatsSnapshot {
    AsrStageStatsSnapshot {
        batches: after.batches.saturating_sub(before.batches),
        jobs: after.jobs.saturating_sub(before.jobs),
        segments: after.segments.saturating_sub(before.segments),
        audio_seconds: (after.audio_seconds - before.audio_seconds).max(0.0),
        wall_seconds: (after.wall_seconds - before.wall_seconds).max(0.0),
        encode_batches: after.encode_batches.saturating_sub(before.encode_batches),
        encode_segments: after.encode_segments.saturating_sub(before.encode_segments),
        encode_audio_seconds: (after.encode_audio_seconds - before.encode_audio_seconds).max(0.0),
        encode_seconds: (after.encode_seconds - before.encode_seconds).max(0.0),
        decode_batches: after.decode_batches.saturating_sub(before.decode_batches),
        decode_segments: after.decode_segments.saturating_sub(before.decode_segments),
        decode_audio_seconds: (after.decode_audio_seconds - before.decode_audio_seconds).max(0.0),
        decode_seconds: (after.decode_seconds - before.decode_seconds).max(0.0),
        audio_encoder_seconds: (after.audio_encoder_seconds - before.audio_encoder_seconds)
            .max(0.0),
        prefill_seconds: (after.prefill_seconds - before.prefill_seconds).max(0.0),
        prefill_tokens: after.prefill_tokens.saturating_sub(before.prefill_tokens),
        decode_model_seconds: (after.decode_model_seconds - before.decode_model_seconds).max(0.0),
        decode_tokens: after.decode_tokens.saturating_sub(before.decode_tokens),
        decode_steps: after.decode_steps.saturating_sub(before.decode_steps),
    }
}

fn log_component_throughput_window(
    service_id: &str,
    window_seconds: f64,
    loader: LoaderStageStatsSnapshot,
    vad: VadStageStatsSnapshot,
    asr: AsrStageStatsSnapshot,
) {
    let decode_audio = asr.decode_audio_seconds;
    let loader_elapsed = max_worker_compute_seconds(&loader.worker_compute_seconds);
    let vad_elapsed = max_worker_compute_seconds(&vad.worker_compute_seconds);
    let table = render_titled_table(
        format!("Service {service_id} Perf | window={window_seconds:.1}s"),
        ["stage", "count", "audio", "elapsed", "speed", "detail"],
        [
            vec![
                "load".to_string(),
                format!("job={}", loader.jobs),
                format_seconds(loader.audio_seconds),
                format_seconds(loader_elapsed),
                format_speed(loader.audio_seconds, loader_elapsed),
                format!("bad={}", loader.failed),
            ],
            vec![
                "vad".to_string(),
                format!("job={}", vad.jobs),
                format_seconds(vad.audio_seconds),
                format_seconds(vad_elapsed),
                format_speed(vad.audio_seconds, vad_elapsed),
                format!(
                    "speech={} ratio={} seg={} bad={} no_speech={}",
                    format_seconds(vad.speech_seconds),
                    format_percent(vad.speech_seconds, vad.audio_seconds),
                    vad.segments,
                    vad.failed,
                    vad.no_speech
                ),
            ],
            vec![
                "encode".to_string(),
                format!("bat={}", asr.encode_batches),
                format_seconds(asr.encode_audio_seconds),
                format_seconds(asr.encode_seconds),
                format_speed(asr.encode_audio_seconds, asr.encode_seconds),
                format!("seg={}", asr.encode_segments),
            ],
            vec![
                "prefill".to_string(),
                format!("bat={}", asr.decode_batches),
                format_seconds(decode_audio),
                format_seconds(asr.prefill_seconds),
                format_speed(decode_audio, asr.prefill_seconds),
                format!(
                    "tok={} tok/s={} seg={}",
                    asr.prefill_tokens,
                    format_rate(asr.prefill_tokens, asr.prefill_seconds),
                    asr.decode_segments
                ),
            ],
            vec![
                "decode".to_string(),
                format!("bat={}", asr.decode_batches),
                format_seconds(decode_audio),
                format_seconds(asr.decode_model_seconds),
                format_speed(decode_audio, asr.decode_model_seconds),
                format!(
                    "tok={} tok/s={} step={} seg={}",
                    asr.decode_tokens,
                    format_rate(asr.decode_tokens, asr.decode_model_seconds),
                    asr.decode_steps,
                    asr.decode_segments
                ),
            ],
        ],
    );

    tracing::info!(
        target: "vasr_transcribe::pipeline",
        "\n{}",
        table
    );
}

fn format_seconds(seconds: f64) -> String {
    format!("{seconds:.2}s")
}

fn format_speed(audio_seconds: f64, compute_seconds: f64) -> String {
    format!("{:.1}x", audio_seconds / compute_seconds.max(f64::EPSILON))
}

fn format_percent(numerator: f64, denominator: f64) -> String {
    format!("{:.1}%", numerator * 100.0 / denominator.max(f64::EPSILON))
}

fn format_rate(count: usize, seconds: f64) -> String {
    format!("{:.0}", count as f64 / seconds.max(f64::EPSILON))
}

fn log_loader_summary(stats: &LoaderStageStats) {
    let metrics = stage_metrics(stats.audio_seconds, stats.wall_seconds);
    tracing::debug!(
        target: "vasr_transcribe::pipeline",
        "load | jobs={} failed={} audio={:.1}s wall={:.1}s compute={:.1}s speed={:.1}x",
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
    tracing::debug!(
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

fn timing_us_to_seconds(us: u64) -> f64 {
    us as f64 / 1_000_000.0
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
mod tests {
    use super::*;
    use crate::runtime::AsrModel;
    use std::sync::{Arc, Mutex};
    use vasr_data::{
        Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, DurationMs, TextSpan,
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
            request_item: None,
            channel: "mono",
            waveform: Waveform::new(vec![0.0; 1024 * 1024 + 1], 16_000),
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            _queue_memory_permit: None,
        };
        assert_eq!(loaded_job_memory_permits(&job), 5);
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn vad_to_asr_segment_memory_permits_rounds_up_to_megabytes() {
        let message = Ok(AsrPipelineMessage::Segment(AsrSegmentJob {
            job_index: 0,
            request_item: None,
            channel: "mono",
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
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
            channel: "mono",
            result: Err(anyhow::anyhow!("failed")),
            audio_seconds: 0.0,
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
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

    #[cfg(feature = "paged-attn")]
    #[test]
    fn segment_batch_error_after_partial_success_returns_job_error() -> Result<()> {
        let mut state = SegmentAsrState::default();
        let segments = vec![
            AsrSegmentJob {
                job_index: 7,
                request_item: None,
                channel: "mono",
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
                job_audio_seconds: 1.0,
                speech_annotations: Vec::new(),
                segment_index: 0,
                segment_count: 2,
                offset: DurationMs(0),
                waveform: Waveform::new(vec![0.1; 160], 16_000),
            },
            AsrSegmentJob {
                job_index: 7,
                request_item: None,
                channel: "mono",
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
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
            None,
        );
        assert!(ok.is_empty());

        let failed = finish_segment_asr_batch(
            &mut state,
            &segments[1..],
            Instant::now(),
            Err(anyhow::anyhow!("encoded feature failed")),
            None,
            None,
        );

        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].outcome.index, 7);
        let err = failed[0]
            .outcome
            .result
            .as_ref()
            .expect_err("partial segment failure should fail the whole job")
            .to_string();
        assert!(err.contains("encoded feature failed"));
        assert!(!state.jobs.contains_key(&7));
        Ok(())
    }

    #[cfg(feature = "paged-attn")]
    #[test]
    fn segment_batch_error_registers_missing_job_state() -> Result<()> {
        let mut state = SegmentAsrState::default();
        let segments = vec![AsrSegmentJob {
            job_index: 11,
            request_item: None,
            channel: "mono",
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
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
            None,
        );

        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].outcome.index, 11);
        assert_eq!(failed[0].outcome.audio_seconds, 2.5);
        let err = failed[0]
            .outcome
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
            request_item: None,
            channel: "mono",
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            prepared: VadPreparation::NoSpeech,
            waveform: Waveform::new(vec![0.1; 160], 16_000),
        }];

        let outcomes = run_asr_jobs_batch(&pipeline, jobs, &AsrOptions::default(), None);

        assert_eq!(outcomes.len(), 1);
        assert!(
            outcomes[0]
                .outcome
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
            vad_to_asr_rx,
            asr_to_result_tx,
            stats,
            None,
            None,
        );

        for index in 0..3usize {
            vad_to_asr_tx.send(test_vad_to_asr_message(Ok(AsrPipelineMessage::Job(
                AsrJob {
                    index,
                    request_item: None,
                    channel: "mono",
                    timing: TranscribeItemTiming::default(),
                    spans: TranscribeItemSpans::default(),
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

    #[test]
    fn duplicate_transcribe_inputs_keep_one_canonical_job() {
        let inputs = vec![
            TranscribeInput {
                index: 10,
                source: AudioSource::Url("file:///same.wav".to_string()),
                mono: true,
            },
            TranscribeInput {
                index: 11,
                source: AudioSource::Url("file:///other.wav".to_string()),
                mono: true,
            },
            TranscribeInput {
                index: 12,
                source: AudioSource::Url("file:///same.wav".to_string()),
                mono: true,
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
    fn duplicate_transcribe_inputs_include_mono_flag_in_key() {
        let inputs = vec![
            TranscribeInput {
                index: 10,
                source: AudioSource::Url("file:///same.wav".to_string()),
                mono: true,
            },
            TranscribeInput {
                index: 11,
                source: AudioSource::Url("file:///same.wav".to_string()),
                mono: false,
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
        assert!(deduped.duplicate_indices.is_empty());
    }

    #[test]
    fn load_transcribe_waveform_downmixes_stereo_when_mono_is_requested() -> anyhow::Result<()> {
        let waveform = Waveform::new_with_channels(
            vec![
                1.0, 3.0, // frame 0
                2.0, 4.0, // frame 1
            ],
            16_000,
            2,
        );

        let loaded =
            load_transcribe_waveforms(&AudioLoader, &AudioSource::Waveform(waveform), true)?;

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "mono");
        assert_eq!(loaded[0].1.channels, 1);
        assert_eq!(loaded[0].1.samples, vec![2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn load_transcribe_waveforms_preserves_source_sample_rate() -> anyhow::Result<()> {
        let waveform = Waveform::new_with_channels(
            vec![
                1.0, 3.0, // frame 0
                2.0, 4.0, // frame 1
            ],
            48_000,
            2,
        );

        let loaded =
            load_transcribe_waveforms(&AudioLoader, &AudioSource::Waveform(waveform), true)?;

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].1.sample_rate, 48_000);
        assert_eq!(loaded[0].1.channels, 1);
        Ok(())
    }

    #[test]
    fn load_transcribe_waveforms_splits_stereo_when_mono_is_false() -> anyhow::Result<()> {
        let waveform = Waveform::new_with_channels(
            vec![
                1.0, 3.0, // frame 0
                2.0, 4.0, // frame 1
            ],
            16_000,
            2,
        );

        let loaded =
            load_transcribe_waveforms(&AudioLoader, &AudioSource::Waveform(waveform), false)?;

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "left");
        assert_eq!(loaded[0].1.channels, 1);
        assert_eq!(loaded[0].1.samples, vec![1.0, 2.0]);
        assert_eq!(loaded[1].0, "right");
        assert_eq!(loaded[1].1.channels, 1);
        assert_eq!(loaded[1].1.samples, vec![3.0, 4.0]);
        Ok(())
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
                mono: true,
            },
            TranscribeInput {
                index: 1,
                source: AudioSource::Url(format!("file://{}", second.display())),
                mono: true,
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
                channel: "mono",
                result: Ok(timeline),
                audio_seconds: 1.25,
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
                bad_component: None,
            },
            TranscribeItemOutcome {
                index: 11,
                channel: "mono",
                result: Err(anyhow::anyhow!("loader failed")),
                audio_seconds: 0.0,
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
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

    #[test]
    fn merged_spans_seconds_counts_overlaps_once() {
        let base = Instant::now();
        let spans = vec![
            StageSpan::new(base, base + Duration::from_secs(2)),
            StageSpan::new(base + Duration::from_secs(1), base + Duration::from_secs(3)),
            StageSpan::new(base + Duration::from_secs(5), base + Duration::from_secs(6)),
        ];

        assert_eq!(merged_spans_seconds(spans), 4.0);
    }

    #[test]
    fn downstream_stage_logging_waits_for_vad_completion() {
        let (response_tx, _rx) = oneshot::channel();
        let mut request = PendingGlobalRequest {
            log_id: "test".to_string(),
            remaining: 1,
            duplicate_indices: Vec::new(),
            internal_indices: Vec::new(),
            outcomes: Vec::new(),
            stages: RequestStageProgress {
                vad_remaining: 1,
                encode_remaining: 0,
                decode_remaining: 0,
                encode: StageProgress {
                    run_span: Some(test_span()),
                    audio_seconds: 1.0,
                    ..Default::default()
                },
                decode: StageProgress {
                    run_span: Some(test_span()),
                    audio_seconds: 1.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            response_tx,
        };

        log_ready_downstream_stages(&mut request);
        assert!(!request.stages.encode_logged);
        assert!(!request.stages.decode_logged);

        request.stages.vad_logged = true;
        log_ready_downstream_stages(&mut request);
        assert!(request.stages.encode_logged);
        assert!(request.stages.decode_logged);
    }

    fn test_span() -> StageSpan {
        let start = Instant::now();
        StageSpan::new(start, start)
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
