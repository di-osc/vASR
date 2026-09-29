use std::path::PathBuf;

use anyhow::Context as _;
use candle_core::{DType, Device};
use vasr::{AudioInput, Qwen3ASR, models::qwen3asr::Batch};
use vasr_data::{
    Annotation, AudioChannel, AudioDb, AudioDbMode, AudioQuery, Transcription, compute_cer,
    normalize_for_cer,
};

const MODEL_ID: &str = "Qwen/Qwen3-ASR-0.6B";

/// Run explicitly after model weights are available. For CPU, use
/// `VASR_TEST_DEVICE=cpu cargo test -p vasr-transcribe --no-default-features --test \
/// example1_model_inference -- --ignored --nocapture`. For Metal, build with `--features metal`.
#[test]
#[ignore = "loads Qwen3-ASR-0.6B; run explicitly for the model regression check"]
fn qwen3_asr_transcribes_example1_to_its_reference() -> anyhow::Result<()> {
    let db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../datasets/example1.vasr");
    let db = AudioDb::open(&db_path, AudioDbMode::ReadOnly)?;
    let mut audios = db.query(&AudioQuery::default())?;
    anyhow::ensure!(audios.len() == 1, "expected exactly one fixture audio");

    let audio = &mut audios[0];
    let reference = audio
        .timeline(AudioChannel::Mono)?
        .and_then(|timeline| {
            timeline
                .reference
                .iter()
                .find_map(|span| match &span.annotation {
                    Annotation::Transcription(Transcription { text, .. }) => Some(text.clone()),
                    _ => None,
                })
        })
        .context("fixture is missing a mono transcription reference")?;
    let waveform = audio.as_waveform()?;

    let device = resolve_test_device()?;
    #[allow(unused_mut)]
    let mut load_options = vasr::models::qwen3asr::LoadOptions {
        dtype: if device.is_cpu() {
            DType::F32
        } else {
            DType::BF16
        },
        progress: vasr::utils::progress::Progress::disabled(),
        ..vasr::models::qwen3asr::LoadOptions::default()
    };
    #[cfg(feature = "paged-attn")]
    if device.is_metal() {
        load_options.paged_cache = Some(vasr::vasr_paged_attn::PagedCacheConfig {
            memory: vasr::vasr_paged_attn::PagedCacheMemory::ContextSize(4_096),
            ..vasr::vasr_paged_attn::PagedCacheConfig::default()
        });
    }

    println!("inference device: {:?}", device.location());
    let model = Qwen3ASR::from_pretrained(MODEL_ID, &device, &load_options)?;
    let mut options = vasr::models::qwen3asr::TranscribeOptions::default();
    options.language = Batch::one(Some("Chinese".to_string()));
    let input = AudioInput::Waveform {
        samples: &waveform.samples,
        sample_rate: waveform.sample_rate,
    };
    let output = model.transcribe(vec![input], options.clone())?;
    let batch_output = model.transcribe(vec![input, input], options)?;
    anyhow::ensure!(output.len() == 1, "expected one single-item result");
    anyhow::ensure!(batch_output.len() == 2, "expected two batched results");
    assert_eq!(batch_output[0].text, output[0].text);
    assert_eq!(batch_output[1].text, output[0].text);

    let hypothesis = &output[0].text;
    let normalized_reference = normalize_for_cer(&reference, true);
    let normalized_hypothesis = normalize_for_cer(hypothesis, true);
    let stats = compute_cer(&normalized_reference, &normalized_hypothesis);
    println!("reference: {reference}");
    println!("hypothesis: {hypothesis}");
    println!("CER: {:.4}", stats.cer());
    assert_eq!(
        stats.edits(),
        0,
        "model output differs from the fixture reference after CER normalization"
    );
    Ok(())
}

fn resolve_test_device() -> anyhow::Result<Device> {
    match std::env::var("VASR_TEST_DEVICE").as_deref() {
        Ok("cpu") => Ok(Device::Cpu),
        Ok("metal") => metal_test_device(),
        Err(std::env::VarError::NotPresent) => metal_test_device(),
        Ok(other) => anyhow::bail!("unsupported VASR_TEST_DEVICE={other:?}; use cpu or metal"),
        Err(error) => anyhow::bail!("read VASR_TEST_DEVICE: {error}"),
    }
}

#[cfg(feature = "metal")]
fn metal_test_device() -> anyhow::Result<Device> {
    let result = std::panic::catch_unwind(|| Device::new_metal(0)).map_err(|_| {
        anyhow::anyhow!(
            "Candle panicked while enumerating Metal devices; set VASR_TEST_DEVICE=cpu and build without default features for the CPU check"
        )
    })?;
    Ok(result?)
}

#[cfg(not(feature = "metal"))]
fn metal_test_device() -> anyhow::Result<Device> {
    anyhow::bail!("Metal test requires building with --features metal")
}
