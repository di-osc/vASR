use std::sync::Arc;

use anyhow::Result;
use vasr_data::{
    Annotation, AudioInput, DurationMs, SAMPLE_RATE_HZ, Timeline, Waveform, normalize_audio_input,
};

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
                let mut asr_timelines = self
                    .asr
                    .transcribe(std::slice::from_ref(waveform), options)?;
                let asr_timeline = asr_timelines
                    .pop()
                    .ok_or_else(|| anyhow::anyhow!("ASR returned no transcription"))?;
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
        let vad_waveform = prepare_waveform_for_vad(waveform)?;
        let segments = vad.detect(&vad_waveform, vad_options)?;
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
        let asr_timelines = self.asr.transcribe(&slices, options)?;
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

pub(crate) fn prepare_waveform_for_vad(waveform: &Waveform) -> Result<Waveform> {
    let mono = waveform.to_mono()?;
    if mono.sample_rate == SAMPLE_RATE_HZ && mono.is_normalized {
        return Ok(mono);
    }
    let samples = normalize_audio_input(&AudioInput::Waveform {
        samples: &mono.samples,
        sample_rate: mono.sample_rate,
    })?;
    Ok(Waveform::new(samples, SAMPLE_RATE_HZ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use std::sync::Mutex;
    use vasr_data::TimeRange;

    #[derive(Default)]
    struct RecordingVad {
        seen: Mutex<Vec<(u32, u16)>>,
    }

    impl VadModel for RecordingVad {
        fn detect(&self, waveform: &Waveform, _options: &VadOptions) -> Result<Vec<VadSegment>> {
            self.seen
                .lock()
                .unwrap()
                .push((waveform.sample_rate, waveform.channels));
            Ok(vec![VadSegment {
                range: TimeRange::new(DurationMs(0), DurationMs(500)),
                probability: 0.9,
            }])
        }

        fn new_stream(
            &self,
            _options: &VadOptions,
        ) -> Result<Box<dyn crate::runtime::model::StreamingVadSession>> {
            anyhow::bail!("streaming VAD is not used in this test")
        }
    }

    struct NoopAsr;

    impl AsrModel for NoopAsr {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn transcribe(
            &self,
            waveforms: &[Waveform],
            _options: &AsrOptions,
        ) -> Result<Vec<Timeline>> {
            Ok(waveforms
                .iter()
                .map(|_| Timeline::new("test_audio"))
                .collect())
        }
    }

    #[test]
    fn vad_stage_resamples_for_vad_but_slices_original_waveform() -> Result<()> {
        let vad = Arc::new(RecordingVad::default());
        let pipeline = OfflinePipeline {
            vad: Some(vad.clone()),
            asr: Arc::new(NoopAsr),
        };
        let waveform = Waveform::new_with_channels(vec![0.0; 48_000 * 2], 48_000, 2);

        let prepared = pipeline.prepare_vad_stage(&waveform, &VadOptions::default())?;

        assert_eq!(vad.seen.lock().unwrap().as_slice(), &[(SAMPLE_RATE_HZ, 1)]);
        let VadPreparation::Speech(prepared) = prepared else {
            panic!("expected speech");
        };
        assert_eq!(prepared.slices.len(), 1);
        assert_eq!(prepared.slices[0].sample_rate, 48_000);
        assert_eq!(prepared.slices[0].channels, 2);
        assert_eq!(prepared.slices[0].duration_ms(), 500.0);
        Ok(())
    }
}
