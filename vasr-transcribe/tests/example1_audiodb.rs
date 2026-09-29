use std::path::PathBuf;

use vasr_data::{Annotation, AudioChannel, AudioDb, AudioDbMode, AudioQuery, AudioSource};

const REFERENCE: &str = "我要修改商品帖子的价格";

#[test]
fn example1_audiodb_contains_embedded_audio_and_reference_transcript() {
    let db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../datasets/example1.vasr");
    let db = AudioDb::open(&db_path, AudioDbMode::ReadOnly)
        .unwrap_or_else(|error| panic!("open {}: {error}", db_path.display()));
    let audios = db.query(&AudioQuery::default()).expect("query AudioDb");

    assert_eq!(audios.len(), 1, "fixture should contain exactly one clip");
    let audio = &audios[0];
    let embedded_bytes = match &audio.source {
        AudioSource::EncodedBytes(bytes) => bytes,
        source => panic!("fixture should embed the WAV bytes, got {source:?}"),
    };
    let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/example1.wav");
    assert_eq!(
        embedded_bytes,
        &std::fs::read(source_path).expect("read source WAV")
    );
    assert_eq!(audio.info.sample_rate, 16_000);
    assert_eq!(audio.info.channels, 1);

    let timeline = audio
        .timeline(AudioChannel::Mono)
        .expect("valid mono channel")
        .expect("mono reference timeline");
    let reference_texts: Vec<&str> = timeline
        .reference
        .iter()
        .filter_map(|span| match &span.annotation {
            Annotation::Transcription(transcription) => Some(transcription.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reference_texts, [REFERENCE]);
}
