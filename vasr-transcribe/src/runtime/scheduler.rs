use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicU64, Ordering as AtomicOrdering},
};
use std::thread;

use anyhow::{Result, anyhow};
use vasr_data::{Timeline, Waveform};

use crate::runtime::model::{AsrModel, AsrOptions};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferencePriority {
    Transcribe,
}

impl InferencePriority {
    fn value(self) -> u8 {
        match self {
            Self::Transcribe => 10,
        }
    }
}

#[derive(Clone)]
pub struct InferenceScheduler {
    inner: Arc<SchedulerInner>,
}

struct SchedulerInner {
    label: String,
    queue: Mutex<BinaryHeap<QueuedTask>>,
    available: Condvar,
    sequence: AtomicU64,
}

struct QueuedTask {
    priority: InferencePriority,
    sequence: u64,
    label: String,
    run: Box<dyn FnOnce() + Send + 'static>,
}

impl InferenceScheduler {
    pub fn start(label: impl Into<String>) -> Self {
        let scheduler = Self {
            inner: Arc::new(SchedulerInner {
                label: label.into(),
                queue: Mutex::new(BinaryHeap::new()),
                available: Condvar::new(),
                sequence: AtomicU64::new(0),
            }),
        };
        scheduler.spawn_worker();
        scheduler
    }

    pub fn submit_blocking<T, F>(
        &self,
        label: impl Into<String>,
        priority: InferencePriority,
        task: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let label = label.into();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let queued = QueuedTask {
            priority,
            sequence: self.inner.sequence.fetch_add(1, AtomicOrdering::Relaxed),
            label,
            run: Box::new(move || {
                let _ = tx.send(task());
            }),
        };

        {
            let mut queue = self.inner.queue.lock().expect("inference queue poisoned");
            queue.push(queued);
        }
        self.inner.available.notify_one();

        rx.recv()
            .map_err(|_| anyhow!("inference scheduler worker stopped"))?
    }

    fn spawn_worker(&self) {
        let inner = Arc::clone(&self.inner);
        thread::Builder::new()
            .name(format!("{}-inference-worker", inner.label))
            .spawn(move || worker_loop(inner))
            .expect("failed to spawn inference scheduler worker");
    }
}

fn worker_loop(inner: Arc<SchedulerInner>) {
    loop {
        let task = {
            let mut queue = inner.queue.lock().expect("inference queue poisoned");
            loop {
                if let Some(task) = queue.pop() {
                    break task;
                }
                queue = inner
                    .available
                    .wait(queue)
                    .expect("inference queue poisoned");
            }
        };
        let _label = task.label;
        (task.run)();
    }
}

impl Ord for QueuedTask {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .priority
            .value()
            .cmp(&self.priority.value())
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for QueuedTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for QueuedTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl Eq for QueuedTask {}

pub struct ScheduledAsrModel {
    inner: Arc<dyn AsrModel>,
    scheduler: InferenceScheduler,
}

impl ScheduledAsrModel {
    pub fn new(inner: Arc<dyn AsrModel>, scheduler: InferenceScheduler) -> Self {
        Self { inner, scheduler }
    }
}

impl AsrModel for ScheduledAsrModel {
    fn as_any(&self) -> &dyn std::any::Any {
        self.inner.as_any()
    }

    fn transcribe(&self, waveforms: &[Waveform], options: &AsrOptions) -> Result<Vec<Timeline>> {
        let inner = Arc::clone(&self.inner);
        let waveforms = waveforms.to_vec();
        let options = options.clone();
        self.scheduler
            .submit_blocking("asr.transcribe", InferencePriority::Transcribe, move || {
                inner.transcribe(&waveforms, &options)
            })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use vasr_data::{Timeline, Waveform};

    use super::{InferencePriority, InferenceScheduler, ScheduledAsrModel};
    use crate::runtime::model::{AsrModel, AsrOptions};

    #[test]
    fn scheduled_asr_model_preserves_batch_calls() {
        struct BatchOnlyAsr {
            batch_calls: Arc<AtomicUsize>,
        }

        impl AsrModel for BatchOnlyAsr {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn transcribe(
                &self,
                waveforms: &[Waveform],
                _options: &AsrOptions,
            ) -> Result<Vec<Timeline>> {
                self.batch_calls.fetch_add(1, Ordering::SeqCst);
                Ok(waveforms.iter().map(|_| Timeline::new("batch")).collect())
            }
        }

        let batch_calls = Arc::new(AtomicUsize::new(0));
        let model = ScheduledAsrModel::new(
            Arc::new(BatchOnlyAsr {
                batch_calls: Arc::clone(&batch_calls),
            }),
            InferenceScheduler::start("batch-test"),
        );
        let waveforms = vec![
            Waveform::new(vec![0.0; 16_000], 16_000),
            Waveform::new(vec![0.0; 8_000], 16_000),
        ];

        let timelines = model
            .transcribe(&waveforms, &AsrOptions::default())
            .expect("batch transcribe");

        assert_eq!(timelines.len(), 2);
        assert_eq!(batch_calls.load(Ordering::SeqCst), 1);
    }
}
