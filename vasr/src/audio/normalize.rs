//! Audio normalization to mono 16k float32 in [-1, 1].

use anyhow::Result;

use super::input::AudioInput;
use super::loader::{SAMPLE_RATE_HZ, peak_normalize};
use asr_data::Waveform;

pub fn normalize_audio_input(input: &AudioInput<'_>) -> Result<Vec<f32>> {
    let mut waveform = match input {
        AudioInput::Waveform {
            samples,
            sample_rate,
        } => Waveform::new(samples.to_vec(), *sample_rate),
        AudioInput::Path(path) => Waveform::from_path(path)?,
        AudioInput::Url(url) => Waveform::from_url(*url)?,
        AudioInput::Base64(b64) => Waveform::from_base64(*b64)?,
    };
    if waveform.channels != 1 {
        waveform = waveform.to_mono()?;
    }
    if waveform.sample_rate != SAMPLE_RATE_HZ {
        waveform = waveform.resample(SAMPLE_RATE_HZ)?;
    }
    peak_normalize(&mut waveform.samples);
    Ok(waveform.samples)
}
