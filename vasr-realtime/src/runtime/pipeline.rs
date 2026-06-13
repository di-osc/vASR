use anyhow::Result;
use vasr_data::Annotation;

use crate::runtime::model::{StreamingAsrSession, StreamingVadSession};

pub struct RealtimePipeline {
    pub vad: Box<dyn StreamingVadSession>,
    pub asr: Box<dyn StreamingAsrSession>,
}

impl RealtimePipeline {
    pub fn push_chunk(&mut self, chunk: &vasr_data::AudioChunk) -> Result<Vec<Annotation>> {
        let mut annotations = self.vad.push_chunk(chunk)?;
        annotations.extend(self.asr.push_chunk(chunk)?);
        Ok(annotations)
    }

    pub fn finish(&mut self) -> Result<Vec<Annotation>> {
        let mut annotations = self.vad.finish()?;
        annotations.extend(self.asr.finish()?);
        Ok(annotations)
    }
}
