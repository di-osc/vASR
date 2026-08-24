use anyhow::Result;
use asr_data::Annotation;

use crate::runtime::model::{StreamingAsrSession, StreamingVadSession};

pub struct RealtimePipeline {
    pub vad: Box<dyn StreamingVadSession>,
    pub asr: Box<dyn StreamingAsrSession>,
}

impl RealtimePipeline {
    pub fn push_chunk(&mut self, chunk: &asr_data::AudioChunk) -> Result<Vec<Annotation>> {
        let mut annotations = self.vad.push_chunk(chunk)?;
        annotations.extend(self.asr.push_chunk(chunk)?);
        Ok(annotations)
    }

    pub fn finish(&mut self) -> Result<Vec<Annotation>> {
        let mut annotations = self.finish_vad()?;
        annotations.extend(self.finish_asr()?);
        Ok(annotations)
    }

    pub fn finish_vad(&mut self) -> Result<Vec<Annotation>> {
        self.vad.finish()
    }

    pub fn finish_asr(&mut self) -> Result<Vec<Annotation>> {
        self.asr.finish()
    }
}
