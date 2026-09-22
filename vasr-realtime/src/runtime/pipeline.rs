use anyhow::Result;
use vasr::StreamEvent;

use crate::runtime::model::{StreamingAsrSession, StreamingVadSession};

pub struct RealtimePipeline {
    pub vad: Box<dyn StreamingVadSession>,
    pub asr: Box<dyn StreamingAsrSession>,
}

impl RealtimePipeline {
    pub fn push_chunk(&mut self, chunk: &vasr_data::AudioChunk) -> Result<Vec<StreamEvent>> {
        let mut events = self.vad.push_chunk(chunk)?;
        events.extend(self.asr.push_chunk(chunk)?);
        Ok(events)
    }

    pub fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        let mut events = self.finish_vad()?;
        events.extend(self.finish_asr()?);
        Ok(events)
    }

    pub fn finish_vad(&mut self) -> Result<Vec<StreamEvent>> {
        self.vad.finish()
    }

    pub fn finish_asr(&mut self) -> Result<Vec<StreamEvent>> {
        self.asr.finish()
    }
}
