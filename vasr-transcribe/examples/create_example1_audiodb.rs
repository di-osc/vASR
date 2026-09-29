use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use vasr_data::{Annotation, Audio, AudioChannel, AudioDb, AudioSource, Transcription};

fn main() -> Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("vasr-transcribe has no repository parent")?
        .to_path_buf();
    let source_path = repo_root.join("assets/example1.wav");
    let output_path = repo_root.join("datasets/example1.vasr");
    if output_path.exists() {
        bail!("refusing to overwrite {}", output_path.display());
    }

    let bytes =
        std::fs::read(&source_path).with_context(|| format!("read {}", source_path.display()))?;
    let mut audio = Audio::with_id("example1", AudioSource::from_encoded_bytes(bytes))?;
    let duration_ms = audio.timeline_duration().context("audio has no duration")?;
    audio
        .ensure_timeline(AudioChannel::Mono, Some(duration_ms))?
        .annotate_span(
            0,
            duration_ms,
            Annotation::Transcription(Transcription::new("我要修改商品帖子的价格")),
        )?;

    let db = AudioDb::create(&output_path)?;
    db.insert(&audio)?;
    println!("created {} ({duration_ms} ms)", output_path.display());
    Ok(())
}
