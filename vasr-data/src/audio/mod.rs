//! Audio input decoding, loading, normalization, and resampling.

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

use crate::{AudioSource, Waveform};

pub mod chunking;
pub mod decode;
pub mod input;
pub mod normalize;
pub mod resample;

pub use input::AudioInput;
pub use normalize::normalize_audio_input;

/// Target ASR/VAD sample rate used by the offline pipeline.
pub const SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Debug, Clone)]
pub struct AudioLoadOptions {
    pub target_sample_rate: u32,
    pub target_channels: u16,
    pub normalize: bool,
}

impl Default for AudioLoadOptions {
    fn default() -> Self {
        Self {
            target_sample_rate: SAMPLE_RATE_HZ,
            target_channels: 1,
            normalize: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct AudioLoader;

impl AudioLoader {
    pub fn load(&self, source: &AudioSource, options: &AudioLoadOptions) -> Result<Waveform> {
        let waveform = match source {
            AudioSource::Path(path) => decode::decode_path_waveform(path)?,
            AudioSource::Url(url) => {
                if let Some(path) = local_path_from_urlish(url) {
                    decode::decode_path_waveform(&path)?
                } else {
                    decode::decode_url_waveform(url)?
                }
            }
            AudioSource::Base64(b64) => decode::decode_base64_waveform(b64)?,
            AudioSource::Bytes(bytes) => {
                let mut pcm = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    pcm.push(i16::from_le_bytes([chunk[0], chunk[1]]));
                }
                Waveform::from_i16_pcm(&pcm, SAMPLE_RATE_HZ)
            }
            AudioSource::Waveform(waveform) => waveform.clone(),
        };
        normalize_loaded_waveform(waveform, options)
    }
}

fn normalize_loaded_waveform(
    mut waveform: Waveform,
    options: &AudioLoadOptions,
) -> Result<Waveform> {
    if waveform.channels == 0 {
        bail!("invalid channel count: 0");
    }
    if options.normalize && !waveform.samples.is_empty() {
        let peak = waveform
            .samples
            .iter()
            .fold(0.0f32, |m, &x| if x.abs() > m { x.abs() } else { m });
        if peak.is_finite() && peak > 1.0 {
            for x in &mut waveform.samples {
                *x /= peak;
            }
        }
        for x in &mut waveform.samples {
            *x = (*x).clamp(-1.0, 1.0);
        }
        waveform.is_normalized = true;
    }
    if waveform.sample_rate == options.target_sample_rate {
        return Ok(waveform);
    }
    if waveform.sample_rate == 0 || options.target_sample_rate == 0 {
        bail!(
            "invalid sample rate conversion: {} -> {}",
            waveform.sample_rate,
            options.target_sample_rate
        );
    }
    resample_waveform(waveform, options.target_sample_rate)
}

fn resample_waveform(waveform: Waveform, target_sample_rate: u32) -> Result<Waveform> {
    if waveform.channels == 1 {
        let samples = resample::resample_mono_f32(
            &waveform.samples,
            waveform.sample_rate,
            target_sample_rate,
        )?;
        return Ok(Waveform::new_with_channels(samples, target_sample_rate, 1));
    }

    let mut channel_samples = Vec::with_capacity(usize::from(waveform.channels));
    for channel in 0..waveform.channels {
        let channel_waveform = waveform.channel(channel)?;
        channel_samples.push(resample::resample_mono_f32(
            &channel_waveform.samples,
            waveform.sample_rate,
            target_sample_rate,
        )?);
    }
    let frames = channel_samples
        .iter()
        .map(Vec::len)
        .min()
        .unwrap_or_default();
    let channels = channel_samples.len();
    let mut samples = Vec::with_capacity(frames * channels);
    for frame in 0..frames {
        for channel in &channel_samples {
            samples.push(channel[frame]);
        }
    }
    Ok(Waveform::new_with_channels(
        samples,
        target_sample_rate,
        waveform.channels,
    ))
}

fn local_path_from_urlish(value: &str) -> Option<PathBuf> {
    if let Some(rest) = value.strip_prefix("file://") {
        return Some(PathBuf::from(percent_decode_path(rest)));
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        return None;
    }
    Some(Path::new(value).to_path_buf())
}

fn percent_decode_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::local_path_from_urlish;
    use std::path::Path;

    #[test]
    fn urlish_local_path_detection_supports_file_urls_and_plain_paths() {
        assert_eq!(
            local_path_from_urlish("file:///tmp/audio.wav").as_deref(),
            Some(Path::new("/tmp/audio.wav"))
        );
        assert_eq!(
            local_path_from_urlish("file:///tmp/audio%20%281%29.wav").as_deref(),
            Some(Path::new("/tmp/audio (1).wav"))
        );
        assert_eq!(
            local_path_from_urlish("/tmp/audio.wav").as_deref(),
            Some(Path::new("/tmp/audio.wav"))
        );
        assert!(local_path_from_urlish("https://example.com/audio.wav").is_none());
    }
}
