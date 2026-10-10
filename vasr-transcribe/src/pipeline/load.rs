use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::{JoinHandle, JoinSet};
use vasr::AudioLoader;

use super::{
    GlobalLoadJob, LoadedJob, LoadedQueueMemoryBudget, LoaderStageStats, LoaderTaskStats,
    PendingGlobalRequest, PipelineRequestId, RequestItemRef, StageSpan, TranscribeInput,
    TranscribeItemOutcome, TranscribeItemSpans, TranscribeItemTiming, acquire_loaded_job_memory,
    load_transcribe_waveforms, log_loader_summary, record_loader_task_stats,
    record_request_load_stage,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_global_loader_worker(
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
                            if !pending_requests.lock().map(|pending| pending.contains_key(&job.request_item.request_id)).unwrap_or(false) { continue; }
                            let worker_id = free_workers.pop().unwrap_or(0);
                            spawn_global_loader_task(&mut tasks, loader, job, worker_id);
                        }
                        None => break,
                    }
                }
                result = tasks.join_next(), if !tasks.is_empty() => {
                    if let Some(result) = result {
                        let worker_id = send_global_loaded_result(result, &loaded_queue_memory, &loader_to_vad_tx, &pending_requests, &job_routes, &next_job_id, &stats).await?;
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
                &next_job_id,
                &stats,
            )
            .await?;
            free_workers.push(worker_id);
        }
        Ok(())
    })
}

pub(super) fn spawn_loader_worker(
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
            if tasks.len() >= max_in_flight
                && let Some(worker_id) =
                    send_next_loaded(&mut tasks, &loaded_queue_memory, &loader_to_vad_tx, &stats)
                        .await?
            {
                free_workers.push(worker_id);
            }
            let worker_id = free_workers.pop().unwrap_or(tasks.len());
            spawn_loader_task(&mut tasks, loader, input, worker_id);
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
        let loaded = super::blocking_result(
            tokio::task::spawn_blocking({
                let source = input.source.clone();
                let mono = input.mono;
                move || load_transcribe_waveforms(&loader, &source, mono)
            })
            .await,
        )
        .expect("loader blocking task must finish");

        let compute_seconds = start.elapsed().as_secs_f64();
        let loader_span = StageSpan::new(start, Instant::now());
        let mut stats = LoaderTaskStats {
            compute_seconds,
            worker_id,
            ..Default::default()
        };
        let message = match Ok::<_, tokio::task::JoinError>(loaded) {
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
                        _item_permit: None,
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
    let (message, task_stats) = match tasks.join_next().await {
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
    match message {
        Ok(jobs) => {
            for mut job in jobs {
                // Publish each channel before reserving memory for the next one.
                job._queue_memory_permit =
                    Some(acquire_loaded_job_memory(loaded_queue_memory, &job).await?);
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

fn spawn_global_loader_task(
    tasks: &mut JoinSet<(
        Result<Vec<LoadedJob>, TranscribeItemOutcome>,
        LoaderTaskStats,
    )>,
    loader: AudioLoader,
    job: GlobalLoadJob,
    worker_id: usize,
) {
    tasks.spawn(async move {
        let start = Instant::now();
        let loaded = super::blocking_result(
            tokio::task::spawn_blocking({
                let source = job.source;
                let mono = job.mono;
                let source_memory = job._source_memory;
                let item_permit = job._item_permit.clone();
                move || {
                    let _source_memory = source_memory;
                    let _item_permit = item_permit;
                    load_transcribe_waveforms(&loader, &source, mono)
                }
            })
            .await,
        )
        .expect("loader blocking task must finish");
        let compute_seconds = start.elapsed().as_secs_f64();
        let loader_span = StageSpan::new(start, Instant::now());
        let mut stats = LoaderTaskStats {
            compute_seconds,
            worker_id,
            ..Default::default()
        };
        let message = match Ok::<_, tokio::task::JoinError>(loaded) {
            Ok(Ok(mut waveforms)) => {
                stats.jobs = waveforms.len();
                stats.audio_seconds = waveforms
                    .iter()
                    .map(|(_, waveform)| waveform.duration_seconds())
                    .sum();
                Ok(waveforms
                    .drain(..)
                    .map(|(channel, waveform)| LoadedJob {
                        _item_permit: job._item_permit.clone(),
                        // Extra channel IDs are assigned atomically with live
                        // request registration when the result is published.
                        index: job.internal_index,
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

#[allow(clippy::too_many_arguments)]
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
    next_job_id: &AtomicU64,
    stats: &Arc<Mutex<LoaderStageStats>>,
) -> Result<usize> {
    let (mut message, task_stats) =
        result.map_err(|error| anyhow::anyhow!("loader task join failed: {error}"))?;
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
        // Register every channel before publishing any: downstream may finish immediately.
        if jobs.len() > 1 {
            let request_item = jobs[0].request_item;
            if let Some(request_item) = request_item {
                let mut pending = pending_requests
                    .lock()
                    .map_err(|_| anyhow::anyhow!("pending request lock poisoned"))?;
                if let Some(request) = pending.get_mut(&request_item.request_id) {
                    // Cancellation uses this same lock. Every issued extra ID
                    // is registered before cancellation can retire the request.
                    for job in jobs.iter_mut().skip(1) {
                        job.index = next_job_id.fetch_add(1, Ordering::Relaxed) as usize;
                    }
                    request.remaining += jobs.len() - 1;
                    request.expected.remove(&(request_item.item_index, "mono"));
                    request
                        .item_routes
                        .remove(&(request_item.item_index, "mono"));
                    request.item_routes.extend(
                        jobs.iter()
                            .map(|job| ((request_item.item_index, job.channel), job.index)),
                    );
                    request.expected.extend(
                        jobs.iter()
                            .map(|job| (request_item.item_index, job.channel)),
                    );
                    request
                        .internal_indices
                        .extend(jobs.iter().skip(1).map(|job| job.index));
                    let mut routes = job_routes
                        .lock()
                        .map_err(|_| anyhow::anyhow!("request route lock poisoned"))?;
                    for job in jobs.iter().skip(1) {
                        routes.insert(job.index, request_item);
                    }
                } else {
                    return Ok(worker_id);
                }
            }
        }
    }
    match message {
        Ok(jobs) => {
            for mut job in jobs {
                if !super::request_is_live(&Some(pending_requests.clone()), job.request_item) {
                    continue;
                }
                job._queue_memory_permit =
                    Some(acquire_loaded_job_memory(loaded_queue_memory, &job).await?);
                if !super::request_is_live(&Some(pending_requests.clone()), job.request_item) {
                    continue;
                }
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

#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;
