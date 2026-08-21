use std::ffi::CStr;
use std::path::Path;

use anyhow::{Context, Result, bail};
use kaldi_fbank_rust_kautism::{
    FbankOptions, FrameExtractionOptions, MelBanksOptions, OnlineFbank,
};

const SAMPLE_RATE: f32 = 16_000.0;
const N_MELS: usize = 80;
const LFR_M: usize = 7;
const LFR_N: usize = 6;
const FEATURE_DIM: usize = N_MELS * LFR_M;

pub struct SenseVoiceFrontend {
    means: Vec<f32>,
    inverse_stddev: Vec<f32>,
}

impl SenseVoiceFrontend {
    pub fn new(cmvn_path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(cmvn_path)
            .with_context(|| format!("failed to read CMVN file {}", cmvn_path.display()))?;
        let means = extract_cmvn_vector(&text, "<AddShift>")?;
        let inverse_stddev = extract_cmvn_vector(&text, "<Rescale>")?;
        if means.len() != FEATURE_DIM || inverse_stddev.len() != FEATURE_DIM {
            bail!(
                "SenseVoice CMVN dimension must be {FEATURE_DIM}, got means={} vars={}",
                means.len(),
                inverse_stddev.len()
            );
        }
        Ok(Self {
            means,
            inverse_stddev,
        })
    }

    pub fn feature_dim(&self) -> usize {
        FEATURE_DIM
    }

    pub fn extract(&self, normalized_samples: &[f32], is_final: bool) -> Result<Vec<Vec<f32>>> {
        let waveform = normalized_samples
            .iter()
            .map(|sample| sample.clamp(-1.0, 1.0) * 32768.0)
            .collect::<Vec<_>>();
        let options = FbankOptions {
            frame_opts: FrameExtractionOptions {
                samp_freq: SAMPLE_RATE,
                window_type: CStr::from_bytes_with_nul(b"hamming\0")?.as_ptr(),
                dither: 0.0,
                frame_shift_ms: 10.0,
                frame_length_ms: 25.0,
                snip_edges: true,
                ..Default::default()
            },
            mel_opts: MelBanksOptions {
                num_bins: N_MELS as i32,
                ..Default::default()
            },
            energy_floor: 0.0,
            ..Default::default()
        };
        let mut fbank = OnlineFbank::new(options);
        fbank.accept_waveform(SAMPLE_RATE, &waveform);
        let frame_count = fbank.num_ready_frames().max(0) as usize;
        if frame_count == 0 {
            return Ok(Vec::new());
        }
        let mut frames = Vec::with_capacity(frame_count);
        for index in 0..frame_count {
            let frame = fbank
                .get_frame(index as i32)
                .ok_or_else(|| anyhow::anyhow!("missing SenseVoice fbank frame {index}"))?;
            frames.push(frame.to_vec());
        }
        Ok(self.apply_lfr_cmvn(&frames, is_final))
    }

    fn apply_lfr_cmvn(&self, frames: &[Vec<f32>], is_final: bool) -> Vec<Vec<f32>> {
        let count = frames.len();
        let output_frames = if is_final {
            count.div_ceil(LFR_N)
        } else if count >= LFR_M.div_ceil(2) {
            (count - LFR_M.div_ceil(2)) / LFR_N + 1
        } else {
            0
        };
        let left_padding = (LFR_M - 1) / 2;
        let mut output = Vec::with_capacity(output_frames);
        for output_index in 0..output_frames {
            let padded_start = output_index * LFR_N;
            let mut feature = Vec::with_capacity(FEATURE_DIM);
            for offset in 0..LFR_M {
                let padded_index = padded_start + offset;
                let source_index = if padded_index < left_padding {
                    0
                } else {
                    (padded_index - left_padding).min(count - 1)
                };
                feature.extend_from_slice(&frames[source_index]);
            }
            for (index, value) in feature.iter_mut().enumerate() {
                *value = (*value + self.means[index]) * self.inverse_stddev[index];
            }
            output.push(feature);
        }
        output
    }
}

fn extract_cmvn_vector(text: &str, section: &str) -> Result<Vec<f32>> {
    let lines = text.lines().collect::<Vec<_>>();
    for pair in lines.windows(2) {
        if pair[0].split_whitespace().next() != Some(section) {
            continue;
        }
        let fields = pair[1].split_whitespace().collect::<Vec<_>>();
        if fields.first().copied() != Some("<LearnRateCoef>") || fields.len() < 5 {
            continue;
        }
        return fields[3..fields.len() - 1]
            .iter()
            .map(|value| {
                value
                    .parse::<f32>()
                    .with_context(|| format!("invalid CMVN value {value:?}"))
            })
            .collect();
    }
    bail!("missing {section} vector in SenseVoice CMVN")
}

#[cfg(test)]
mod tests {
    use super::SenseVoiceFrontend;

    #[test]
    fn online_lfr_waits_for_right_context() {
        let frontend = SenseVoiceFrontend {
            means: vec![0.0; 560],
            inverse_stddev: vec![1.0; 560],
        };
        let frames = (0..9)
            .map(|index| vec![index as f32; 80])
            .collect::<Vec<_>>();
        assert_eq!(frontend.apply_lfr_cmvn(&frames[..3], false).len(), 0);
        assert_eq!(frontend.apply_lfr_cmvn(&frames[..4], false).len(), 1);
        assert_eq!(frontend.apply_lfr_cmvn(&frames, false).len(), 1);
        assert_eq!(frontend.apply_lfr_cmvn(&frames, true).len(), 2);
    }
}
