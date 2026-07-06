//! Async parallel transcribe: loader, VAD, and ASR stages overlap via channels.

mod dedup;
mod load;
mod memory;
mod result_router;
#[cfg(feature = "paged-attn")]
mod segment_decode;
#[cfg(feature = "paged-attn")]
mod segment_encode;
#[cfg(feature = "paged-attn")]
mod segment_prefill;
mod stages;
mod submit;
mod vad;

use self::dedup::*;
use self::load::*;
use self::memory::*;
use self::result_router::*;
#[cfg(feature = "paged-attn")]
use self::segment_decode::*;
#[cfg(feature = "paged-attn")]
use self::segment_encode::*;
#[cfg(feature = "paged-attn")]
use self::segment_prefill::*;
use self::stages::*;
use self::submit::*;
use self::vad::*;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, atomic::AtomicU64};
use std::time::{Duration, Instant};

use anyhow::Result;
#[cfg(feature = "paged-attn")]
use anyhow::bail;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::AudioFeatureEncodeTimings;
#[cfg(feature = "timing")]
use vasr::models::qwen3asr::TranscribeTimings;
#[cfg(feature = "paged-attn")]
use vasr::models::qwen3asr::model::paged_cache_runtime::PagedCacheStats;
use vasr::models::qwen3asr::{
    AudioInput, Qwen3ASR, qwen_outputs_to_timelines, qwen_transcribe_options,
};
use vasr_data::AudioLoader;
#[cfg(feature = "paged-attn")]
use vasr_data::DurationMs;
use vasr_data::{Annotation, AudioSource, Timeline, Waveform};

#[cfg(feature = "paged-attn")]
use crate::engine::offline::VadPrepared;
use crate::engine::offline::{OfflinePipeline, VadPreparation};
use crate::engine::{AsrOptions, ParallelTranscribeOptions, VadOptions};
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

struct PendingGlobalRequest {
    log_id: String,
    remaining: usize,
    duplicate_indices: Vec<(usize, usize)>,
    internal_indices: Vec<usize>,
    outcomes: Vec<TranscribeItemOutcome>,
    stages: RequestStageProgress,
    response_tx: oneshot::Sender<Vec<TranscribeItemOutcome>>,
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
    use crate::engine::AsrModel;
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
