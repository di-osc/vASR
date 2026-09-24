//! 生成长度按固定 4.25 token/秒预留。
//!
//! 声学语速估过一圈，和真实解码 token 率对不上，倍率固定为 1。
//! 实际生成超出预留时，不在这里加长，而由解码阶段把空出来的 KV 补给这条序列。

/// 相对 4.25 token/秒的倍率。固定为 1，不再按波形改长度。
///
/// 参数保留是为了调用点不用改。波形和采样率都不参与计算。
pub fn speech_rate_ratio(_samples: &[f32], _sample_rate: u32) -> f64 {
    1.0
}

/// 按固定倍率估算生成 token 数，不套固定下限。
///
/// 公式是 `ceil(秒数 × 每秒 token 数 × 倍率)`。秒数、token 率或倍率无效时返回 1。
pub fn predicted_generated_tokens(audio_seconds: f64, tokens_per_sec: f64, ratio: f64) -> usize {
    if !audio_seconds.is_finite()
        || audio_seconds <= 0.0
        || !tokens_per_sec.is_finite()
        || tokens_per_sec <= 0.0
        || !ratio.is_finite()
        || ratio <= 0.0
    {
        return 1;
    }
    // 向上取整，避免把不足 1 个 token 的短段预留成 0。
    (audio_seconds * tokens_per_sec * ratio).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::{predicted_generated_tokens, speech_rate_ratio};

    #[test]
    fn ratio_stays_at_one() {
        assert!((speech_rate_ratio(&[], 16_000) - 1.0).abs() < 1.0e-9);
        assert!((speech_rate_ratio(&[0.0; 16_000], 16_000) - 1.0).abs() < 1.0e-9);
        let tone: Vec<f32> = (0..16_000)
            .map(|index| (index as f32 * 0.05).sin())
            .collect();
        assert!((speech_rate_ratio(&tone, 16_000) - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn one_second_reserves_five_tokens() {
        assert_eq!(predicted_generated_tokens(1.0, 4.25, 1.0), 5);
        assert_eq!(predicted_generated_tokens(2.0, 4.25, 1.0), 9);
    }

    #[test]
    fn invalid_inputs_reserve_one_token() {
        assert_eq!(predicted_generated_tokens(0.0, 4.25, 1.0), 1);
        assert_eq!(predicted_generated_tokens(1.0, 0.0, 1.0), 1);
        assert_eq!(predicted_generated_tokens(f64::NAN, 4.25, 1.0), 1);
    }
}
