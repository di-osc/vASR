use std::sync::Arc;

#[derive(Clone, Default)]
pub struct Progress {
    sink: Option<Arc<dyn ProgressSink>>,
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Progress")
            .field("enabled", &self.sink.is_some())
            .finish()
    }
}

impl Progress {
    pub fn disabled() -> Self {
        Self { sink: None }
    }

    pub fn new(sink: Arc<dyn ProgressSink>) -> Self {
        Self { sink: Some(sink) }
    }

    pub fn start(&self, event: ProgressEvent) -> ProgressHandle {
        match &self.sink {
            Some(sink) => sink.start(event),
            None => ProgressHandle::disabled(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProgressEvent {
    pub model: &'static str,
    pub phase: &'static str,
    pub total: Option<u64>,
    pub unit: ProgressUnit,
}

impl ProgressEvent {
    pub fn layers(model: &'static str, phase: &'static str, total: usize) -> Self {
        Self {
            model,
            phase,
            total: Some(total as u64),
            unit: ProgressUnit::Layer,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressUnit {
    Layer,
    Tensor,
    Shard,
    File,
    Byte,
}

pub trait ProgressSink: Send + Sync {
    fn start(&self, event: ProgressEvent) -> ProgressHandle;
}

pub trait ProgressTask: Send + Sync {
    fn inc(&self, n: u64);
    fn finish(&self);
}

pub struct ProgressHandle {
    inner: Option<Box<dyn ProgressTask>>,
}

impl ProgressHandle {
    pub fn new(inner: Box<dyn ProgressTask>) -> Self {
        Self { inner: Some(inner) }
    }

    pub fn disabled() -> Self {
        Self { inner: None }
    }

    pub fn inc(&self, n: u64) {
        if let Some(inner) = &self.inner {
            inner.inc(n);
        }
    }

    pub fn finish(self) {
        if let Some(inner) = self.inner {
            inner.finish();
        }
    }
}
