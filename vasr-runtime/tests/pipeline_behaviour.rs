use anyhow::Result;
use std::sync::{Arc, Mutex};
use vasr_data::{
    Annotation, AnnotationPayload, AnnotationSource, AnnotationStatus, AudioChunk, DurationMs,
    TextSpan, TimeRange, Timeline, Waveform,
};
use vasr_runtime::{
    AsrModel, AsrOptions, OfflinePipeline, StreamingAsrModel, StreamingVadModel, VadModel,
    VadOptions, VadSegment,
};

#[derive(Default)]
struct FakeAsr {
    seen_durations_ms: Mutex<Vec<u64>>,
    seen_batch_durations_ms: Mutex<Vec<Vec<u64>>>,
}

impl AsrModel for FakeAsr {
    fn transcribe(&self, waveform: &Waveform, _options: &AsrOptions) -> Result<Timeline> {
        self.seen_durations_ms
            .lock()
            .expect("seen durations poisoned")
            .push(waveform.duration_ms().round() as u64);
        let mut timeline = Timeline::new("fake_audio");
        timeline.push(Annotation::new(
            TimeRange::new(
                DurationMs(0),
                DurationMs(waveform.duration_ms().round() as u64),
            ),
            AnnotationPayload::Transcription(TextSpan::new("hello world")),
            AnnotationSource::Model("fake_asr".to_string()),
            AnnotationStatus::Final,
        ));
        Ok(timeline)
    }

    fn transcribe_batch(
        &self,
        waveforms: &[Waveform],
        _options: &AsrOptions,
    ) -> Result<Vec<Timeline>> {
        self.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .push(
                waveforms
                    .iter()
                    .map(|waveform| waveform.duration_ms().round() as u64)
                    .collect(),
            );
        Ok(waveforms
            .iter()
            .map(|waveform| {
                let mut timeline = Timeline::new("fake_audio_batch");
                timeline.push(Annotation::new(
                    TimeRange::new(
                        DurationMs(0),
                        DurationMs(waveform.duration_ms().round() as u64),
                    ),
                    AnnotationPayload::Transcription(TextSpan::new("hello batch")),
                    AnnotationSource::Model("fake_asr".to_string()),
                    AnnotationStatus::Final,
                ));
                timeline
            })
            .collect())
    }

    fn start_stream(&self, _options: &AsrOptions) -> Result<Box<dyn StreamingAsrModel>> {
        Ok(Box::new(FakeStream))
    }
}

struct FakeStream;

impl StreamingAsrModel for FakeStream {
    fn push_chunk(&mut self, chunk: &AudioChunk) -> Result<Vec<Annotation>> {
        Ok(vec![Annotation::new(
            chunk.range,
            AnnotationPayload::Transcription(TextSpan::new("partial")),
            AnnotationSource::Model("fake_asr".to_string()),
            AnnotationStatus::Partial,
        )])
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        Ok(vec![Annotation::new(
            TimeRange::default(),
            AnnotationPayload::Transcription(TextSpan::new("final")),
            AnnotationSource::Model("fake_asr".to_string()),
            AnnotationStatus::Final,
        )])
    }
}

#[derive(Default)]
struct FakeVad {
    segments: Vec<VadSegment>,
    refine_segments: Vec<(u64, Vec<VadSegment>)>,
    seen_min_silence_ms: Mutex<Vec<u64>>,
}

impl VadModel for FakeVad {
    fn detect(&self, _waveform: &Waveform, options: &VadOptions) -> Result<Vec<VadSegment>> {
        let mut seen = self
            .seen_min_silence_ms
            .lock()
            .expect("seen min silence poisoned");
        let is_initial_detect = seen.is_empty();
        seen.push(options.min_silence_ms);
        drop(seen);
        if let Some((_, segments)) = self
            .refine_segments
            .iter()
            .find(|(min_silence_ms, _)| *min_silence_ms == options.min_silence_ms)
        {
            return Ok(segments.clone());
        }
        if is_initial_detect {
            Ok(self.segments.clone())
        } else {
            Ok(Vec::new())
        }
    }

    fn start_stream(&self, _options: &VadOptions) -> Result<Box<dyn StreamingVadModel>> {
        Ok(Box::new(FakeVadStream))
    }
}

struct FakeVadStream;

impl StreamingVadModel for FakeVadStream {
    fn push_chunk(&mut self, _chunk: &AudioChunk) -> Result<Vec<Annotation>> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<Annotation>> {
        Ok(Vec::new())
    }
}

#[test]
fn offline_pipeline_merges_vad_and_asr_annotations() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 160_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let pipeline = OfflinePipeline {
        vad: Some(Arc::new(FakeVad {
            segments: vec![VadSegment {
                range: TimeRange::new(
                    DurationMs(0),
                    DurationMs(waveform.duration_ms().round() as u64),
                ),
                probability: 0.9,
            }],
            ..FakeVad::default()
        })),
        asr: asr.clone(),
    };

    let timeline = pipeline.transcribe(&waveform, &AsrOptions::default())?;

    assert!(timeline.annotations.len() >= 2);
    assert_eq!(timeline.transcript().text, "hello world");
    Ok(())
}

#[test]
fn offline_pipeline_keeps_vad_segments_separate_before_asr() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 160_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let pipeline = OfflinePipeline {
        vad: Some(Arc::new(FakeVad {
            segments: vec![
                VadSegment {
                    range: TimeRange::new(DurationMs(1_000), DurationMs(2_500)),
                    probability: 0.9,
                },
                VadSegment {
                    range: TimeRange::new(DurationMs(4_000), DurationMs(5_000)),
                    probability: 0.8,
                },
            ],
            ..FakeVad::default()
        })),
        asr: asr.clone(),
    };

    let timeline = pipeline.transcribe(&waveform, &AsrOptions::default())?;

    assert_eq!(
        asr.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .as_slice(),
        &[vec![2_500, 2_000]]
    );
    assert!(
        asr.seen_durations_ms
            .lock()
            .expect("seen durations poisoned")
            .is_empty()
    );
    let final_segments = timeline
        .annotations
        .iter()
        .filter(|annotation| matches!(annotation.payload, AnnotationPayload::Transcription(_)))
        .collect::<Vec<_>>();
    assert_eq!(final_segments.len(), 2);
    assert_eq!(final_segments[0].range.start, DurationMs(500));
    assert_eq!(final_segments[0].range.end, DurationMs(3_000));
    assert_eq!(final_segments[1].range.start, DurationMs(3_500));
    assert_eq!(final_segments[1].range.end, DurationMs(5_500));
    assert_eq!(timeline.transcript().text, "hello batch hello batch");
    Ok(())
}

#[test]
fn offline_pipeline_splits_long_vad_segments_before_asr() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 80_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let vad = Arc::new(FakeVad {
        segments: vec![VadSegment {
            range: TimeRange::new(DurationMs(1_000), DurationMs(3_500)),
            probability: 0.9,
        }],
        ..FakeVad::default()
    });
    let pipeline = OfflinePipeline {
        vad: Some(vad.clone()),
        asr: asr.clone(),
    };
    let options = VadOptions {
        max_segment_ms: 1_000,
        ..VadOptions::default()
    };

    let timeline =
        pipeline.transcribe_with_vad_options(&waveform, &AsrOptions::default(), &options)?;

    assert_eq!(
        *vad.seen_min_silence_ms
            .lock()
            .expect("seen min silence poisoned"),
        vec![500, 300]
    );
    assert_eq!(
        asr.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .as_slice(),
        &[vec![2_000, 2_000, 1_500]]
    );
    let final_segments = timeline
        .annotations
        .iter()
        .filter(|annotation| matches!(annotation.payload, AnnotationPayload::Transcription(_)))
        .collect::<Vec<_>>();
    assert_eq!(final_segments.len(), 3);
    assert_eq!(final_segments[0].range.start, DurationMs(500));
    assert_eq!(final_segments[0].range.end, DurationMs(2_500));
    assert_eq!(final_segments[1].range.start, DurationMs(1_500));
    assert_eq!(final_segments[1].range.end, DurationMs(3_500));
    assert_eq!(final_segments[2].range.start, DurationMs(2_500));
    assert_eq!(final_segments[2].range.end, DurationMs(4_000));
    Ok(())
}

#[test]
fn offline_pipeline_refines_long_vad_segments_with_lower_silence() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 96_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let vad = Arc::new(FakeVad {
        segments: vec![VadSegment {
            range: TimeRange::new(DurationMs(1_000), DurationMs(5_000)),
            probability: 0.9,
        }],
        refine_segments: vec![(
            500,
            vec![
                VadSegment {
                    range: TimeRange::new(DurationMs(0), DurationMs(1_200)),
                    probability: 0.9,
                },
                VadSegment {
                    range: TimeRange::new(DurationMs(1_400), DurationMs(2_600)),
                    probability: 0.8,
                },
                VadSegment {
                    range: TimeRange::new(DurationMs(2_800), DurationMs(4_000)),
                    probability: 0.7,
                },
            ],
        )],
        ..FakeVad::default()
    });
    let pipeline = OfflinePipeline {
        vad: Some(vad.clone()),
        asr: asr.clone(),
    };
    let options = VadOptions {
        min_silence_ms: 800,
        max_segment_ms: 1_500,
        ..VadOptions::default()
    };

    let timeline =
        pipeline.transcribe_with_vad_options(&waveform, &AsrOptions::default(), &options)?;

    assert_eq!(
        *vad.seen_min_silence_ms
            .lock()
            .expect("seen min silence poisoned"),
        vec![800, 500]
    );
    assert_eq!(
        asr.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .as_slice(),
        &[vec![2_200, 2_200, 2_200]]
    );
    let final_segments = timeline
        .annotations
        .iter()
        .filter(|annotation| matches!(annotation.payload, AnnotationPayload::Transcription(_)))
        .collect::<Vec<_>>();
    assert_eq!(final_segments.len(), 3);
    assert_eq!(final_segments[0].range.start, DurationMs(500));
    assert_eq!(final_segments[0].range.end, DurationMs(2_700));
    assert_eq!(final_segments[1].range.start, DurationMs(1_900));
    assert_eq!(final_segments[1].range.end, DurationMs(4_100));
    assert_eq!(final_segments[2].range.start, DurationMs(3_300));
    assert_eq!(final_segments[2].range.end, DurationMs(5_500));
    Ok(())
}

#[test]
fn offline_pipeline_refines_long_vad_segments_twice_before_hard_split() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 112_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let vad = Arc::new(FakeVad {
        segments: vec![VadSegment {
            range: TimeRange::new(DurationMs(1_000), DurationMs(6_000)),
            probability: 0.9,
        }],
        refine_segments: vec![
            (
                500,
                vec![VadSegment {
                    range: TimeRange::new(DurationMs(0), DurationMs(5_000)),
                    probability: 0.9,
                }],
            ),
            (
                300,
                vec![
                    VadSegment {
                        range: TimeRange::new(DurationMs(0), DurationMs(2_000)),
                        probability: 0.9,
                    },
                    VadSegment {
                        range: TimeRange::new(DurationMs(2_200), DurationMs(5_000)),
                        probability: 0.8,
                    },
                ],
            ),
        ],
        ..FakeVad::default()
    });
    let pipeline = OfflinePipeline {
        vad: Some(vad.clone()),
        asr: asr.clone(),
    };
    let options = VadOptions {
        min_silence_ms: 700,
        max_segment_ms: 1_500,
        ..VadOptions::default()
    };

    let timeline =
        pipeline.transcribe_with_vad_options(&waveform, &AsrOptions::default(), &options)?;

    assert_eq!(
        *vad.seen_min_silence_ms
            .lock()
            .expect("seen min silence poisoned"),
        vec![700, 500, 300]
    );
    assert_eq!(
        asr.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .as_slice(),
        &[vec![2_500, 1_500, 2_500, 2_300]]
    );
    let final_segments = timeline
        .annotations
        .iter()
        .filter(|annotation| matches!(annotation.payload, AnnotationPayload::Transcription(_)))
        .collect::<Vec<_>>();
    assert_eq!(final_segments.len(), 4);
    Ok(())
}

#[test]
fn offline_pipeline_uses_next_lower_refine_silence() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 96_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let vad = Arc::new(FakeVad {
        segments: vec![VadSegment {
            range: TimeRange::new(DurationMs(1_000), DurationMs(5_000)),
            probability: 0.9,
        }],
        refine_segments: vec![(
            300,
            vec![
                VadSegment {
                    range: TimeRange::new(DurationMs(0), DurationMs(1_500)),
                    probability: 0.9,
                },
                VadSegment {
                    range: TimeRange::new(DurationMs(1_700), DurationMs(4_000)),
                    probability: 0.8,
                },
            ],
        )],
        ..FakeVad::default()
    });
    let pipeline = OfflinePipeline {
        vad: Some(vad.clone()),
        asr: asr.clone(),
    };
    let options = VadOptions {
        min_silence_ms: 400,
        max_segment_ms: 1_500,
        ..VadOptions::default()
    };

    let _ = pipeline.transcribe_with_vad_options(&waveform, &AsrOptions::default(), &options)?;

    assert_eq!(
        *vad.seen_min_silence_ms
            .lock()
            .expect("seen min silence poisoned"),
        vec![400, 300]
    );
    Ok(())
}

#[test]
fn offline_pipeline_does_not_refine_when_silence_already_at_floor() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 80_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let vad = Arc::new(FakeVad {
        segments: vec![VadSegment {
            range: TimeRange::new(DurationMs(1_000), DurationMs(3_500)),
            probability: 0.9,
        }],
        ..FakeVad::default()
    });
    let pipeline = OfflinePipeline {
        vad: Some(vad.clone()),
        asr: asr.clone(),
    };
    let options = VadOptions {
        min_silence_ms: 300,
        max_segment_ms: 1_000,
        ..VadOptions::default()
    };

    let _ = pipeline.transcribe_with_vad_options(&waveform, &AsrOptions::default(), &options)?;

    assert_eq!(
        *vad.seen_min_silence_ms
            .lock()
            .expect("seen min silence poisoned"),
        vec![300]
    );
    Ok(())
}

#[test]
fn offline_pipeline_pads_vad_slice_for_asr_context() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 80_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let pipeline = OfflinePipeline {
        vad: Some(Arc::new(FakeVad {
            segments: vec![VadSegment {
                range: TimeRange::new(DurationMs(150), DurationMs(4_180)),
                probability: 0.9,
            }],
            ..FakeVad::default()
        })),
        asr: asr.clone(),
    };

    let timeline = pipeline.transcribe(&waveform, &AsrOptions::default())?;

    assert_eq!(
        asr.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .as_slice(),
        &[vec![4_680]]
    );
    let final_segments = timeline
        .annotations
        .iter()
        .filter(|annotation| matches!(annotation.payload, AnnotationPayload::Transcription(_)))
        .collect::<Vec<_>>();
    assert_eq!(final_segments.len(), 1);
    assert_eq!(final_segments[0].range.start, DurationMs(0));
    assert_eq!(final_segments[0].range.end, DurationMs(4_680));
    Ok(())
}

#[test]
fn offline_pipeline_uses_full_waveform_when_padded_vad_covers_audio() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 67_264], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let pipeline = OfflinePipeline {
        vad: Some(Arc::new(FakeVad {
            segments: vec![VadSegment {
                range: TimeRange::new(DurationMs(150), DurationMs(4_180)),
                probability: 0.9,
            }],
            ..FakeVad::default()
        })),
        asr: asr.clone(),
    };

    let timeline = pipeline.transcribe(&waveform, &AsrOptions::default())?;

    assert_eq!(
        asr.seen_durations_ms
            .lock()
            .expect("seen durations poisoned")
            .as_slice(),
        &[4_204]
    );
    assert_eq!(
        asr.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .as_slice(),
        &[] as &[Vec<u64>]
    );
    let final_segments = timeline
        .annotations
        .iter()
        .filter(|annotation| matches!(annotation.payload, AnnotationPayload::Transcription(_)))
        .collect::<Vec<_>>();
    assert_eq!(final_segments.len(), 1);
    assert_eq!(final_segments[0].range.start, DurationMs(0));
    assert_eq!(final_segments[0].range.end, DurationMs(4_204));
    Ok(())
}

#[test]
fn offline_pipeline_skips_asr_when_vad_finds_no_speech() -> Result<()> {
    let waveform = Waveform::new(vec![0.05; 160_000], 16_000);
    let asr = Arc::new(FakeAsr::default());
    let pipeline = OfflinePipeline {
        vad: Some(Arc::new(FakeVad {
            segments: Vec::new(),
            ..FakeVad::default()
        })),
        asr: asr.clone(),
    };

    let timeline = pipeline.transcribe(&waveform, &AsrOptions::default())?;

    assert!(timeline.annotations.is_empty());
    assert!(
        asr.seen_durations_ms
            .lock()
            .expect("seen durations poisoned")
            .is_empty()
    );
    assert!(
        asr.seen_batch_durations_ms
            .lock()
            .expect("seen batch durations poisoned")
            .is_empty()
    );
    Ok(())
}
