use anyhow::Result;
use vasr_data::{AudioChunk, Waveform};

pub use vasr::{AsrOptions, StreamEvent, StreamingAsrModel, StreamingAsrSession};
pub use vasr_vad::{VadOptions, VadSegment};

pub use vasr::StreamingAsrModel as AsrModel;

pub trait VadModel: Send + Sync {
    fn detect(&self, waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>>;

    fn new_stream(&self, options: &VadOptions) -> Result<Box<dyn StreamingVadSession>>;
}

pub trait StreamingVadSession: Send {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<StreamEvent>>;

    fn finish(&mut self) -> Result<Vec<StreamEvent>>;
}
