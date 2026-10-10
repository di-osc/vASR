use super::*;
use asr_data::TimeRange;
use std::sync::{Arc, Mutex};
use vasr::OfflineAsrModel as AsrModel;

fn test_transcription_timeline(audio_id: &str, text: &str, source: &str) -> Timeline {
    vasr::timeline_from_spans(
        audio_id,
        0,
        [vasr::transcription_span(
            TimeRange::default(),
            text,
            None,
            source,
        )],
    )
    .expect("test timeline")
}

#[test]
fn default_stage_parallelism_is_nonzero() {
    assert!(default_stage_parallelism() >= 1);
}

#[test]
fn loaded_queue_memory_budget_defaults_to_twenty_percent_of_total_memory() {
    assert_eq!(
        loaded_queue_memory_budget_permits(None, None, Some(64 * 1024 * 1024 * 1024)),
        LoadedQueueMemoryBudgetConfig {
            permits: 13_107,
            source: "system-memory-fraction"
        }
    );
}

#[test]
fn loaded_queue_memory_budget_accepts_explicit_mb_override() {
    assert_eq!(
        loaded_queue_memory_budget_permits(Some("512"), Some("0.5"), Some(64 * 1024 * 1024 * 1024),),
        LoadedQueueMemoryBudgetConfig {
            permits: 512,
            source: "env:VASR_LOADED_QUEUE_MEMORY_MB"
        }
    );
}

#[test]
fn loaded_queue_memory_budget_accepts_fraction_override() {
    assert_eq!(
        loaded_queue_memory_budget_permits(None, Some("0.10"), Some(64 * 1024 * 1024 * 1024)),
        LoadedQueueMemoryBudgetConfig {
            permits: 6_553,
            source: "env:VASR_LOADED_QUEUE_MEMORY_FRACTION"
        }
    );
}

#[test]
fn loaded_job_memory_permits_rounds_up_to_megabytes() {
    let job = LoadedJob {
        _item_permit: None,
        index: 0,
        request_item: None,
        channel: "mono",
        waveform: Waveform::new(vec![0.0; 1024 * 1024 + 1], 16_000),
        timing: TranscribeItemTiming::default(),
        spans: TranscribeItemSpans::default(),
        _queue_memory_permit: None,
    };
    assert_eq!(loaded_job_memory_permits(&job), 5);
}

#[cfg(feature = "paged-attn")]
#[test]
fn vad_to_asr_segment_memory_permits_rounds_up_to_megabytes() {
    let message = Ok(AsrPipelineMessage::Segment(AsrSegmentJob {
        _item_permit: None,
        job_index: 0,
        request_item: None,
        channel: "mono",
        timing: TranscribeItemTiming::default(),
        spans: TranscribeItemSpans::default(),
        job_audio_seconds: 1.0,
        speech_annotations: Vec::new(),
        segment_index: 0,
        segment_count: 1,
        offset: 0,
        waveform: Waveform::new(vec![0.0; 1024 * 1024 + 1], 16_000),
    }));
    assert_eq!(vad_to_asr_message_memory_permits(&message), 5);
}

#[test]
fn vad_to_asr_error_messages_do_not_need_memory_permits() {
    let message = Err(TranscribeItemOutcome {
        index: 0,
        channel: "mono",
        result: Err(anyhow::anyhow!("failed")),
        audio_seconds: 0.0,
        timing: TranscribeItemTiming::default(),
        spans: TranscribeItemSpans::default(),
        bad_component: Some("vad"),
    });
    assert_eq!(vad_to_asr_message_memory_permits(&message), 0);
}

#[cfg(target_os = "linux")]
#[test]
fn parses_mem_total_from_proc_meminfo() {
    let text = "MemAvailable:    98765 kB\nMemTotal:       123456 kB\n";
    assert_eq!(parse_meminfo_kib(text, "MemTotal:"), Some(123_456));
}

#[derive(Default)]
struct BatchRecordingAsr {
    seen_batch_sizes: Mutex<Vec<usize>>,
}

impl AsrModel for BatchRecordingAsr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn transcribe(&self, waveforms: &[Waveform], _options: &AsrOptions) -> Result<Vec<Timeline>> {
        self.seen_batch_sizes
            .lock()
            .expect("seen batch sizes poisoned")
            .push(waveforms.len());
        Ok(waveforms
            .iter()
            .map(|_| test_transcription_timeline("fake_batch_asr", "ok", "fake_batch_asr"))
            .collect())
    }
}

#[cfg(feature = "paged-attn")]
#[test]
fn segment_batch_error_after_partial_success_returns_job_error() -> Result<()> {
    let mut state = SegmentAsrState::default();
    let segments = vec![
        AsrSegmentJob {
            _item_permit: None,
            job_index: 7,
            request_item: None,
            channel: "mono",
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            job_audio_seconds: 1.0,
            speech_annotations: Vec::new(),
            segment_index: 0,
            segment_count: 2,
            offset: 0,
            waveform: Waveform::new(vec![0.1; 160], 16_000),
        },
        AsrSegmentJob {
            _item_permit: None,
            job_index: 7,
            request_item: None,
            channel: "mono",
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            job_audio_seconds: 1.0,
            speech_annotations: Vec::new(),
            segment_index: 1,
            segment_count: 2,
            offset: 100,
            waveform: Waveform::new(vec![0.2; 160], 16_000),
        },
    ];
    prepare_segment_job_state(&mut state, segments.as_slice());

    let ok_timeline = test_transcription_timeline("fake_batch_asr", "ok", "fake_batch_asr");
    let ok = finish_segment_asr_batch(
        &mut state,
        &segments[..1],
        Instant::now(),
        Ok((
            SegmentAsrInference {
                targets: vec![(7, 0, 0)],
                segment_audio_seconds: 0.01,
                active_job_count: 1,
            },
            vec![ok_timeline],
        )),
        None,
        None,
    );
    assert!(ok.is_empty());

    let failed = finish_segment_asr_batch(
        &mut state,
        &segments[1..],
        Instant::now(),
        Err(anyhow::anyhow!("encoded feature failed")),
        None,
        None,
    );

    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].outcome.index, 7);
    let err = failed[0]
        .outcome
        .result
        .as_ref()
        .expect_err("partial segment failure should fail the whole job")
        .to_string();
    assert!(err.contains("encoded feature failed"));
    assert!(!state.jobs.contains_key(&7));
    Ok(())
}

#[cfg(feature = "paged-attn")]
#[test]
fn segment_batch_error_registers_missing_job_state() -> Result<()> {
    let mut state = SegmentAsrState::default();
    let segments = vec![AsrSegmentJob {
        _item_permit: None,
        job_index: 11,
        request_item: None,
        channel: "mono",
        timing: TranscribeItemTiming::default(),
        spans: TranscribeItemSpans::default(),
        job_audio_seconds: 2.5,
        speech_annotations: Vec::new(),
        segment_index: 0,
        segment_count: 1,
        offset: 0,
        waveform: Waveform::new(vec![0.1; 160], 16_000),
    }];

    let failed = finish_segment_asr_batch(
        &mut state,
        segments.as_slice(),
        Instant::now(),
        Err(anyhow::anyhow!("first encoded batch failed")),
        None,
        None,
    );

    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].outcome.index, 11);
    assert_eq!(failed[0].outcome.audio_seconds, 2.5);
    let err = failed[0]
        .outcome
        .result
        .as_ref()
        .expect_err("missing-state encode failure should still return an outcome")
        .to_string();
    assert!(err.contains("first encoded batch failed"));
    assert!(state.jobs.is_empty());
    Ok(())
}

#[test]
fn no_speech_jobs_are_returned_without_asr() -> Result<()> {
    let asr = Arc::new(BatchRecordingAsr::default());
    let jobs = vec![AsrJob {
        _item_permit: None,
        index: 0,
        request_item: None,
        channel: "mono",
        timing: TranscribeItemTiming::default(),
        spans: TranscribeItemSpans::default(),
        prepared: VadPreparation::NoSpeech,
        waveform: Waveform::new(vec![0.1; 160], 16_000),
    }];

    let outcomes = run_asr_jobs_batch(asr.as_ref(), jobs, &AsrOptions::default(), None);

    assert_eq!(outcomes.len(), 1);
    assert!(
        outcomes[0]
            .outcome
            .result
            .as_ref()
            .is_ok_and(|timeline| timeline.span_count() == 0)
    );
    assert!(
        asr.seen_batch_sizes
            .lock()
            .expect("seen batch sizes poisoned")
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn asr_worker_coalesces_slightly_staggered_jobs() -> Result<()> {
    let asr = Arc::new(BatchRecordingAsr::default());
    let (vad_to_asr_tx, vad_to_asr_rx) = mpsc::unbounded_channel();
    let (asr_to_result_tx, mut asr_to_result_rx) = mpsc::channel(4);
    let stats = Arc::new(Mutex::new(AsrStageStats::default()));
    let handle = spawn_asr_worker(
        asr.clone(),
        AsrOptions::default(),
        vad_to_asr_rx,
        asr_to_result_tx,
        stats,
        None,
        None,
        None,
        None,
        DEFAULT_ASR_ENCODER_BATCH_MAX_SEGMENTS,
        DEFAULT_ASR_ENCODER_BATCH_MAX_AUDIO_SECONDS,
    );

    for index in 0..3usize {
        vad_to_asr_tx.send(test_vad_to_asr_message(Ok(AsrPipelineMessage::Job(
            AsrJob {
                _item_permit: None,
                index,
                request_item: None,
                channel: "mono",
                timing: TranscribeItemTiming::default(),
                spans: TranscribeItemSpans::default(),
                prepared: VadPreparation::Disabled,
                waveform: Waveform::new(vec![index as f32; 160], 16_000),
            },
        ))))?;
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    drop(vad_to_asr_tx);

    let mut outcomes = Vec::new();
    while let Some(outcome) = asr_to_result_rx.recv().await {
        outcomes.push(outcome);
    }
    handle.await??;

    assert_eq!(outcomes.len(), 3);
    assert_eq!(
        *asr.seen_batch_sizes
            .lock()
            .expect("seen batch sizes poisoned"),
        vec![3]
    );
    Ok(())
}

#[test]
fn duplicate_transcribe_inputs_keep_one_canonical_job() {
    let inputs = vec![
        TranscribeInput {
            index: 10,
            source: AudioSource::Url("file:///same.wav".to_string()),
            mono: true,
        },
        TranscribeInput {
            index: 11,
            source: AudioSource::Url("file:///other.wav".to_string()),
            mono: true,
        },
        TranscribeInput {
            index: 12,
            source: AudioSource::Url("file:///same.wav".to_string()),
            mono: true,
        },
    ];

    let deduped = deduplicate_transcribe_inputs(inputs);

    assert_eq!(
        deduped
            .unique_inputs
            .iter()
            .map(|input| input.index)
            .collect::<Vec<_>>(),
        vec![10, 11]
    );
    assert_eq!(deduped.duplicate_indices, vec![(12, 10)]);
}

#[test]
fn duplicate_transcribe_inputs_include_mono_flag_in_key() {
    let inputs = vec![
        TranscribeInput {
            index: 10,
            source: AudioSource::Url("file:///same.wav".to_string()),
            mono: true,
        },
        TranscribeInput {
            index: 11,
            source: AudioSource::Url("file:///same.wav".to_string()),
            mono: false,
        },
    ];

    let deduped = deduplicate_transcribe_inputs(inputs);

    assert_eq!(
        deduped
            .unique_inputs
            .iter()
            .map(|input| input.index)
            .collect::<Vec<_>>(),
        vec![10, 11]
    );
    assert!(deduped.duplicate_indices.is_empty());
}

#[test]
fn load_transcribe_waveform_downmixes_stereo_when_mono_is_requested() -> anyhow::Result<()> {
    let waveform = Waveform::new_with_channels(
        vec![
            0.25, 0.75, // frame 0
            0.5, -0.5, // frame 1
        ],
        16_000,
        2,
    );

    let source = AudioSource::from_pcm_s16le(
        waveform
            .to_i16_pcm()
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>(),
        waveform.sample_rate,
        waveform.channels,
    );
    let loaded = load_transcribe_waveforms(&AudioLoader, &source, true)?;

    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, "mono");
    assert_eq!(loaded[0].1.channels, 1);
    assert_eq!(loaded[0].1.samples, vec![0.5, 0.0]);
    Ok(())
}

#[test]
fn load_transcribe_waveforms_preserves_source_sample_rate() -> anyhow::Result<()> {
    let waveform = Waveform::new_with_channels(
        vec![
            0.25, 0.75, // frame 0
            0.5, -0.5, // frame 1
        ],
        48_000,
        2,
    );

    let source = AudioSource::from_pcm_s16le(
        waveform
            .to_i16_pcm()
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>(),
        waveform.sample_rate,
        waveform.channels,
    );
    let loaded = load_transcribe_waveforms(&AudioLoader, &source, true)?;

    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].1.sample_rate, 48_000);
    assert_eq!(loaded[0].1.channels, 1);
    Ok(())
}

#[test]
fn load_transcribe_waveforms_splits_stereo_when_mono_is_false() -> anyhow::Result<()> {
    let waveform = Waveform::new_with_channels(
        vec![
            0.25, 0.75, // frame 0
            0.5, -0.5, // frame 1
        ],
        16_000,
        2,
    );

    let source = AudioSource::from_pcm_s16le(
        waveform
            .to_i16_pcm()
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>(),
        waveform.sample_rate,
        waveform.channels,
    );
    let loaded = load_transcribe_waveforms(&AudioLoader, &source, false)?;

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].0, "left");
    assert_eq!(loaded[0].1.channels, 1);
    assert_eq!(loaded[0].1.samples, vec![0.25, 0.5]);
    assert_eq!(loaded[1].0, "right");
    assert_eq!(loaded[1].1.channels, 1);
    assert_eq!(loaded[1].1.samples, vec![0.75, -0.5]);
    Ok(())
}

#[test]
fn local_file_urls_with_same_content_keep_one_canonical_job() {
    let dir =
        std::env::temp_dir().join(format!("vasr-dedup-test-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let first = dir.join("first.wav");
    let second = dir.join("second.wav");
    std::fs::write(&first, b"same audio bytes").expect("write first file");
    std::fs::write(&second, b"same audio bytes").expect("write second file");

    let inputs = vec![
        TranscribeInput {
            index: 0,
            source: AudioSource::Url(format!("file://{}", first.display())),
            mono: true,
        },
        TranscribeInput {
            index: 1,
            source: AudioSource::Url(format!("file://{}", second.display())),
            mono: true,
        },
    ];

    let deduped = deduplicate_transcribe_inputs(inputs);

    assert_eq!(deduped.unique_inputs.len(), 1);
    assert_eq!(deduped.unique_inputs[0].index, 0);
    assert_eq!(deduped.duplicate_indices, vec![(1, 0)]);

    std::fs::remove_dir_all(dir).expect("remove temp dir");
}

#[test]
fn duplicate_transcribe_outcomes_are_expanded_to_original_indices() {
    let timeline = test_transcription_timeline("canonical", "same result", "fake_batch_asr");
    let outcomes = vec![
        TranscribeItemOutcome {
            index: 10,
            channel: "mono",
            result: Ok(timeline),
            audio_seconds: 1.25,
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            bad_component: None,
        },
        TranscribeItemOutcome {
            index: 11,
            channel: "mono",
            result: Err(anyhow::anyhow!("loader failed")),
            audio_seconds: 0.0,
            timing: TranscribeItemTiming::default(),
            spans: TranscribeItemSpans::default(),
            bad_component: Some("loader"),
        },
    ];

    let expanded = expand_duplicate_outcomes(outcomes, &[(12, 10), (13, 11)]);

    assert_eq!(
        expanded
            .iter()
            .map(|outcome| outcome.index)
            .collect::<Vec<_>>(),
        vec![10, 11, 12, 13]
    );
    assert_eq!(expanded[2].audio_seconds, 1.25);
    assert!(
        expanded[2]
            .result
            .as_ref()
            .is_ok_and(|timeline| vasr::prediction_text(timeline).trim() == "same result")
    );
    assert_eq!(expanded[3].bad_component, Some("loader"));
    assert!(expanded[3].result.is_err());
}

#[test]
fn stage_metrics_calculate_speedup_and_rtf() {
    let metrics = stage_metrics(10.0, 2.0);
    assert_eq!(metrics.speedup, 5.0);
    assert_eq!(metrics.rtf, 0.2);

    let empty_audio = stage_metrics(0.0, 2.0);
    assert_eq!(empty_audio.speedup, 0.0);
    assert_eq!(empty_audio.rtf, 0.0);
}

#[test]
fn merged_spans_seconds_counts_overlaps_once() {
    let base = Instant::now();
    let spans = vec![
        StageSpan::new(base, base + Duration::from_secs(2)),
        StageSpan::new(base + Duration::from_secs(1), base + Duration::from_secs(3)),
        StageSpan::new(base + Duration::from_secs(5), base + Duration::from_secs(6)),
    ];

    assert_eq!(merged_spans_seconds(spans), 4.0);
}

#[test]
fn downstream_stage_logging_waits_for_vad_completion() {
    let (response_tx, _rx) = oneshot::channel();
    let mut request = PendingGlobalRequest {
        item_routes: HashMap::new(),
        log_id: "test".to_string(),
        remaining: 1,
        expected: [(0, "mono")].into(),
        duplicate_indices: Vec::new(),
        internal_indices: Vec::new(),
        outcomes: Vec::new(),
        stages: RequestStageProgress {
            vad_remaining: 1,
            encode_remaining: 0,
            decode_remaining: 0,
            encode: StageProgress {
                run_span: Some(test_span()),
                audio_seconds: 1.0,
                ..Default::default()
            },
            decode: StageProgress {
                run_span: Some(test_span()),
                audio_seconds: 1.0,
                ..Default::default()
            },
            ..Default::default()
        },
        response_tx,
    };

    log_ready_downstream_stages(&mut request);
    assert!(!request.stages.encode_logged);
    assert!(!request.stages.decode_logged);

    request.stages.vad_logged = true;
    log_ready_downstream_stages(&mut request);
    assert!(request.stages.encode_logged);
    assert!(request.stages.decode_logged);
}

fn test_span() -> StageSpan {
    let start = Instant::now();
    StageSpan::new(start, start)
}

fn test_vad_to_asr_message(
    message: Result<AsrPipelineMessage, TranscribeItemOutcome>,
) -> VadToAsrMessage {
    VadToAsrMessage {
        message,
        _queue_memory_permit: None,
    }
}
