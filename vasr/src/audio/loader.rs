use anyhow::{Result, bail};
use asr_data::{AudioSource, Waveform};

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

#[derive(Debug, Default, Clone, Copy)]
pub struct AudioLoader;

impl AudioLoader {
    pub fn load_raw(&self, source: &AudioSource) -> Result<Waveform> {
        Waveform::from_source(source)
    }

    pub fn load(&self, source: &AudioSource, options: &AudioLoadOptions) -> Result<Waveform> {
        normalize_loaded_waveform(self.load_raw(source)?, options)
    }
}

pub(crate) fn peak_normalize(samples: &mut [f32]) {
    let peak = samples
        .iter()
        .fold(0.0f32, |max, sample| sample.abs().max(max));
    if peak.is_finite() && peak > 1.0 {
        for sample in samples.iter_mut() {
            *sample /= peak;
        }
    }
    for sample in samples {
        *sample = sample.clamp(-1.0, 1.0);
    }
}

fn normalize_loaded_waveform(
    mut waveform: Waveform,
    options: &AudioLoadOptions,
) -> Result<Waveform> {
    if waveform.channels == 0 {
        bail!("invalid channel count: 0");
    }
    if options.target_channels == 0 {
        bail!("invalid target channel count: 0");
    }
    if waveform.channels != options.target_channels {
        if options.target_channels == 1 {
            waveform = waveform.to_mono()?;
        } else {
            bail!(
                "unsupported channel conversion: {} -> {}",
                waveform.channels,
                options.target_channels
            );
        }
    }
    if waveform.sample_rate != options.target_sample_rate {
        waveform = waveform.resample(options.target_sample_rate)?;
    }
    if options.normalize {
        peak_normalize(&mut waveform.samples);
    }
    Ok(waveform)
}
