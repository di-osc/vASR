use std::sync::Arc;

use anyhow::Result;
use vasr_data::{Annotation, DurationMs, Timeline, Waveform};

use crate::runtime::model::{AsrModel, AsrOptions, VadModel, VadOptions, VadSegment};

pub struct OfflinePipeline {
    pub vad: Option<Arc<dyn VadModel>>,
    pub asr: Arc<dyn AsrModel>,
}

/// Output of the VAD stage, ready for batched ASR.
#[derive(Debug, Clone)]
pub struct VadPrepared {
    pub speech_annotations: Vec<Annotation>,
    pub segments: Vec<VadSegment>,
    pub slices: Vec<Waveform>,
}

/// Result of the VAD stage. `Disabled` is the only case that should fall back
/// to raw ASR; `NoSpeech` means VAD ran and found nothing to recognize.
#[derive(Debug, Clone)]
pub enum VadPreparation {
    Disabled,
    NoSpeech,
    Speech(VadPrepared),
}

impl OfflinePipeline {
    pub fn transcribe(&self, waveform: &Waveform, options: &AsrOptions) -> Result<Timeline> {
        self.transcribe_with_vad_options(waveform, options, &VadOptions::default())
    }

    pub fn transcribe_with_vad_options(
        &self,
        waveform: &Waveform,
        options: &AsrOptions,
        vad_options: &VadOptions,
    ) -> Result<Timeline> {
        let mut timeline = Timeline::new("offline_audio");
        match self.prepare_vad_stage(waveform, vad_options)? {
            VadPreparation::Speech(prepared) => {
                timeline
                    .annotations
                    .extend(self.transcribe_prepared(prepared, options)?);
            }
            VadPreparation::NoSpeech => {}
            VadPreparation::Disabled => {
                let asr_timeline = self.asr.transcribe(waveform, options)?;
                timeline.annotations.extend(asr_timeline.annotations);
            }
        }
        Ok(timeline)
    }

    /// Run VAD and slice the waveform, preserving whether VAD was disabled or
    /// ran and found no speech.
    pub fn prepare_vad_stage(
        &self,
        waveform: &Waveform,
        vad_options: &VadOptions,
    ) -> Result<VadPreparation> {
        let Some(vad) = &self.vad else {
            return Ok(VadPreparation::Disabled);
        };
        let segments = vad.detect(waveform, vad_options)?;
        if segments.is_empty() {
            return Ok(VadPreparation::NoSpeech);
        }
        let speech_annotations = segments
            .iter()
            .map(|segment| {
                vasr_data::Annotation::new(
                    segment.range,
                    vasr_data::AnnotationPayload::Speech,
                    vasr_data::AnnotationSource::Model("vad".to_string()),
                    vasr_data::AnnotationStatus::Final,
                )
            })
            .collect();
        let slices = segments
            .iter()
            .map(|segment| waveform.slice_ms(segment.range.start.0, segment.range.end.0))
            .collect();
        Ok(VadPreparation::Speech(VadPrepared {
            speech_annotations,
            segments,
            slices,
        }))
    }

    /// Run VAD and slice the waveform. Returns `None` when VAD is disabled or finds no speech.
    pub fn prepare_vad(
        &self,
        waveform: &Waveform,
        vad_options: &VadOptions,
    ) -> Result<Option<VadPrepared>> {
        match self.prepare_vad_stage(waveform, vad_options)? {
            VadPreparation::Speech(prepared) => Ok(Some(prepared)),
            VadPreparation::Disabled | VadPreparation::NoSpeech => Ok(None),
        }
    }

    /// Run ASR on VAD slices and offset segment annotations back to the original timeline.
    pub fn transcribe_prepared(
        &self,
        prepared: VadPrepared,
        options: &AsrOptions,
    ) -> Result<Vec<Annotation>> {
        let VadPrepared {
            speech_annotations,
            segments,
            slices,
        } = prepared;
        let asr_timelines = self.asr.transcribe_batch(&slices, options)?;
        let mut annotations = speech_annotations;
        for (segment, asr_timeline) in segments.into_iter().zip(asr_timelines) {
            annotations.extend(offset_annotations(
                asr_timeline.annotations,
                segment.range.start,
            ));
        }
        Ok(annotations)
    }
}

pub fn offset_annotations(annotations: Vec<Annotation>, offset: DurationMs) -> Vec<Annotation> {
    annotations
        .into_iter()
        .map(|mut annotation| {
            annotation.range.start.0 = annotation.range.start.0.saturating_add(offset.0);
            annotation.range.end.0 = annotation.range.end.0.saturating_add(offset.0);
            annotation
        })
        .collect()
}
