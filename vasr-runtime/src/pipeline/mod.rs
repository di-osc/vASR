use std::sync::Arc;

use anyhow::Result;
use vasr_data::{Annotation, DurationMs, TimeRange, Timeline, Waveform};

use crate::model::{
    AsrModel, AsrOptions, StreamingAsrModel, StreamingVadModel, VadModel, VadOptions, VadSegment,
};

const ASR_VAD_CONTEXT_PAD_MS: u64 = 500;
const FULL_WAVEFORM_COVERAGE_GAP_MS: u64 = 20;
const VAD_REFINE_SILENCE_MS: [u64; 2] = [500, 300];

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
    FullSpeech { speech_annotations: Vec<Annotation> },
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
            VadPreparation::FullSpeech { speech_annotations } => {
                timeline.annotations.extend(speech_annotations);
                let asr_timeline = self.asr.transcribe(waveform, options)?;
                timeline.annotations.extend(asr_timeline.annotations);
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
        let asr_segments =
            split_vad_segments_for_asr(vad.as_ref(), waveform, segments.as_slice(), vad_options)?;
        let asr_segments = asr_segments
            .into_iter()
            .map(|mut segment| {
                segment.range = padded_asr_range(waveform, &segment);
                segment
            })
            .collect::<Vec<_>>();
        if covers_full_waveform(waveform, asr_segments.as_slice()) {
            return Ok(VadPreparation::FullSpeech { speech_annotations });
        }
        let slices = asr_segments
            .iter()
            .map(|segment| waveform.slice_ms(segment.range.start.0, segment.range.end.0))
            .collect();
        Ok(VadPreparation::Speech(VadPrepared {
            speech_annotations,
            segments: asr_segments,
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
            VadPreparation::Disabled
            | VadPreparation::NoSpeech
            | VadPreparation::FullSpeech { .. } => Ok(None),
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

fn split_vad_segments_for_asr(
    vad: &dyn VadModel,
    waveform: &Waveform,
    segments: &[VadSegment],
    vad_options: &VadOptions,
) -> Result<Vec<VadSegment>> {
    let max_segment_ms = vad_options.max_segment_ms;
    if segments.is_empty() || max_segment_ms == 0 {
        return Ok(segments.to_vec());
    }

    let mut split = Vec::with_capacity(segments.len());
    for segment in segments {
        if segment_duration_ms(segment) <= max_segment_ms {
            split.push(segment.clone());
            continue;
        }

        let refined = refine_long_vad_segment(vad, waveform, segment, vad_options, max_segment_ms)?;
        split.extend(refined);
    }

    Ok(split)
}

fn refine_long_vad_segment(
    vad: &dyn VadModel,
    waveform: &Waveform,
    segment: &VadSegment,
    vad_options: &VadOptions,
    max_segment_ms: u64,
) -> Result<Vec<VadSegment>> {
    let mut current = vec![segment.clone()];
    for silence_ms in VAD_REFINE_SILENCE_MS {
        if vad_options.min_silence_ms <= silence_ms {
            continue;
        }

        let mut refined = Vec::new();
        let mut changed = false;
        for candidate in current {
            if segment_duration_ms(&candidate) <= max_segment_ms {
                refined.push(candidate);
                continue;
            }

            let local =
                detect_vad_with_min_silence(vad, waveform, &candidate, vad_options, silence_ms)?;
            if local.is_empty() {
                refined.push(candidate);
                continue;
            }
            changed = true;
            refined.extend(local);
        }

        current = refined;
        if current
            .iter()
            .all(|candidate| segment_duration_ms(candidate) <= max_segment_ms)
        {
            return Ok(current);
        }
        if !changed {
            break;
        }
    }

    Ok(current
        .into_iter()
        .flat_map(|candidate| hard_split_vad_segment(&candidate, max_segment_ms))
        .collect())
}

fn detect_vad_with_min_silence(
    vad: &dyn VadModel,
    waveform: &Waveform,
    segment: &VadSegment,
    vad_options: &VadOptions,
    min_silence_ms: u64,
) -> Result<Vec<VadSegment>> {
    let local_waveform = waveform.slice_ms(segment.range.start.0, segment.range.end.0);
    let mut refined_options = vad_options.clone();
    refined_options.min_silence_ms = min_silence_ms;
    let local_segments = vad.detect(&local_waveform, &refined_options)?;
    Ok(local_segments
        .into_iter()
        .map(|mut local| {
            local.range.start.0 = local.range.start.0.saturating_add(segment.range.start.0);
            local.range.end.0 = local
                .range
                .end
                .0
                .saturating_add(segment.range.start.0)
                .min(segment.range.end.0);
            local
        })
        .filter(|local| local.range.end.0 > local.range.start.0)
        .collect())
}

fn hard_split_vad_segment(segment: &VadSegment, max_segment_ms: u64) -> Vec<VadSegment> {
    if max_segment_ms == 0 || segment_duration_ms(segment) <= max_segment_ms {
        return vec![segment.clone()];
    }

    let mut split = Vec::new();
    let mut start = segment.range.start.0;
    let end = segment.range.end.0;
    while start < end {
        let next = start.saturating_add(max_segment_ms).min(end);
        split.push(VadSegment {
            range: TimeRange::new(DurationMs(start), DurationMs(next)),
            probability: segment.probability,
        });
        start = next;
    }
    split
}

fn segment_duration_ms(segment: &VadSegment) -> u64 {
    segment.range.end.0.saturating_sub(segment.range.start.0)
}

pub(crate) fn padded_asr_range(waveform: &Waveform, segment: &VadSegment) -> TimeRange {
    let audio_end = waveform.duration_ms().ceil().max(0.0) as u64;
    let start = segment.range.start.0.saturating_sub(ASR_VAD_CONTEXT_PAD_MS);
    let end = segment
        .range
        .end
        .0
        .saturating_add(ASR_VAD_CONTEXT_PAD_MS)
        .min(audio_end);
    TimeRange::new(DurationMs(start), DurationMs(end))
}

fn covers_full_waveform(waveform: &Waveform, segments: &[VadSegment]) -> bool {
    if segments.len() != 1 {
        return false;
    }
    let audio_end = waveform.duration_ms().ceil().max(0.0) as u64;
    let range = segments[0].range;
    range.start.0 <= FULL_WAVEFORM_COVERAGE_GAP_MS
        && range.end.0.saturating_add(FULL_WAVEFORM_COVERAGE_GAP_MS) >= audio_end
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

pub struct RealtimePipeline {
    pub vad: Box<dyn StreamingVadModel>,
    pub asr: Box<dyn StreamingAsrModel>,
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

#[cfg(feature = "async")]
pub mod r#async;
