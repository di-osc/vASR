use anyhow::Result;
use vasr_data::Waveform;

use crate::aligner::ForcedAlignResult;

pub trait AlignerModel: Send + Sync {
    fn align(
        &self,
        waveforms: &[Waveform],
        text: &[String],
        language: &[String],
    ) -> Result<Vec<ForcedAlignResult>>;
}
