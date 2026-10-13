use super::{
    AudioAttention, AudioTower, cu_seqlens_from_aftercnn_len, feature_lens_from_mask,
    sinusoidal_positional_embedding,
};
use crate::config::AudioEncoderConfig;
use crate::utils::progress::Progress;
use candle_nn::Linear;
use vasr_quant::isq_linear::IsqLinear;

/// 确定性伪随机权重后端，用于验证多段 encoder 在非零权重下仍然 batch 不变。
///
/// `VarBuilder::zeros` 会把所有中间量压成 0，padding / 窗口边界写错也看不出来；
/// 这里按张量名做稳定哈希，生成非零、可复现且幅度适中的权重。
struct SeededBackend;

/// FNV-1a，把张量名映射成一个稳定的 64 位种子。
fn fingerprint(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(feature = "metal-paged-attn")]
#[test]
#[ignore = "requires exclusive Metal GPU"]
fn encoder_fc1_fusion_keeps_dense_projection_values() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    let device = Device::new_metal(0)?;
    let cfg = test_config();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let vb =
            candle_nn::VarBuilder::from_backend(Box::new(SeededBackend), dtype, device.clone());
        let mut layer = super::AudioEncoderLayer::load(&cfg, vb, false, None)?;
        for rows in [1, 17, 129] {
            let x = Tensor::from_vec(
                (0..rows * cfg.d_model)
                    .map(|i| ((i * 13 % 193) as f32 - 96.) / 23.)
                    .collect::<Vec<_>>(),
                (rows, cfg.d_model),
                &device,
            )?
            .to_dtype(dtype)?;
            layer.fused_fc1_gelu = false;
            let expected = layer.fc1_gelu(&x)?;
            layer.fused_fc1_gelu = true;
            let actual = layer.fc1_gelu(&x)?;
            assert_tensor_close(&actual, &expected, 0.)?;
        }
    }
    Ok(())
}

impl candle_nn::var_builder::SimpleBackend for SeededBackend {
    fn get(
        &self,
        shape: candle_core::Shape,
        name: &str,
        _init: candle_nn::Init,
        dtype: candle_core::DType,
        device: &candle_core::Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        let count = shape.elem_count();
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ fingerprint(name);
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            // xorshift64*：稳定、快速，不引入外部 RNG 依赖。
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let x = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            // 映射到 [-0.5, 0.5)，避免随机权重把激活值推到溢出。
            values.push(((x >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32);
        }
        candle_core::Tensor::from_vec(values, shape, device)?.to_dtype(dtype)
    }

    fn get_unchecked(
        &self,
        _name: &str,
        _dtype: candle_core::DType,
        _device: &candle_core::Device,
    ) -> candle_core::Result<candle_core::Tensor> {
        candle_core::bail!("SeededBackend 需要形状，请使用 get 而不是 get_unchecked")
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

/// 构造一个用于数值对比的测试配置。
fn test_config() -> AudioEncoderConfig {
    AudioEncoderConfig {
        num_mel_bins: 4,
        encoder_layers: 1,
        encoder_attention_heads: 1,
        encoder_ffn_dim: 8,
        d_model: 4,
        dropout: 0.0,
        attention_dropout: 0.0,
        activation_function: "gelu".to_string(),
        activation_dropout: 0.0,
        scale_embedding: false,
        initializer_range: 0.02,
        max_source_positions: 64,
        n_window: 5,
        output_dim: 6,
        n_window_infer: 20,
        conv_chunksize: 8,
        downsample_hidden_size: 2,
    }
}

/// 用给定权重后端加载 tower，并按 batch-vs-per-sample 方式比较输出。
fn assert_batched_matches_per_sample(
    device: &candle_core::Device,
    dtype: candle_core::DType,
    zeros: bool,
    tolerance: f32,
) -> anyhow::Result<()> {
    let cfg = test_config();
    let vb = if zeros {
        candle_nn::VarBuilder::zeros(dtype, device)
    } else {
        candle_nn::VarBuilder::from_backend(Box::new(SeededBackend), dtype, device.clone())
    };
    let tower = AudioTower::load(&cfg, vb, false, None, &Progress::disabled())?;

    let batch = 3usize;
    let frames = 10usize;
    let lens = [10usize, 7usize, 4usize];
    let feats: Vec<f32> = (0..batch * cfg.num_mel_bins * frames)
        .map(|i| (i as f32) / 100.0)
        .collect();
    let input_features =
        candle_core::Tensor::from_vec(feats, (batch, cfg.num_mel_bins, frames), device)?;

    let batched = tower.forward_with_lens(&input_features, &lens)?;
    let weight_dtype = tower.conv2d1.weight().dtype();
    let mut per_sample = Vec::with_capacity(batch);
    for (idx, &len) in lens.iter().enumerate() {
        let input = input_features
            .narrow(0, idx, 1)?
            .squeeze(0)?
            .narrow(1, 0, len)?;
        // `forward_one` 是测试辅助路径，不负责 dtype 转换，这里手动对齐权重 dtype。
        let input = super::cast_input_features_to_weight_dtype(&input, weight_dtype)?;
        per_sample.push(tower.forward_one(&input)?);
    }
    let expected = candle_core::Tensor::cat(&per_sample.iter().collect::<Vec<_>>(), 0)?;

    if batched.dims() != expected.dims() {
        anyhow::bail!(
            "batched dims mismatch: batched={:?} expected={:?}",
            batched.dims(),
            expected.dims()
        );
    }
    let got = batched
        .to_dtype(candle_core::DType::F32)?
        .to_vec2::<f32>()?;
    let exp = expected
        .to_dtype(candle_core::DType::F32)?
        .to_vec2::<f32>()?;
    let mut max_abs = 0f32;
    for (row_g, row_e) in got.iter().zip(exp.iter()) {
        for (&a, &b) in row_g.iter().zip(row_e.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    if max_abs > tolerance {
        anyhow::bail!("batched forward mismatch: max_abs={max_abs} tolerance={tolerance}");
    }
    Ok(())
}

#[test]
fn fused_conv_policy_is_automatic_and_respects_explicit_off() -> anyhow::Result<()> {
    let mut enabled = super::ConvExecutionPolicy::from_values(Some("1"), None)?;
    enabled.bias_gelu_override = Some(true);
    assert!(enabled.fused_bias_gelu());
    enabled.nhwc_override = Some(false);
    enabled.auto = true;
    assert!(!enabled.fused_bias_gelu());
    let mut automatic = super::ConvExecutionPolicy::from_values(None, None)?;
    automatic.auto = true;
    assert!(automatic.fused_bias_gelu());
    automatic.bias_gelu_override = Some(false);
    assert!(!automatic.fused_bias_gelu());
    Ok(())
}

#[test]
fn conv_execution_policy_preserves_explicit_reference_override() -> anyhow::Result<()> {
    let mut policy = super::ConvExecutionPolicy::from_values(None, None)?;
    assert_eq!(policy.resolve(), (false, false));
    policy.auto = true;
    assert_eq!(policy.resolve(), (true, true));
    let mut disabled = super::ConvExecutionPolicy::from_values(Some("0"), Some("1"))?;
    disabled.auto = true;
    assert_eq!(disabled.resolve(), (false, false));
    let mut layout_only = super::ConvExecutionPolicy::from_values(Some("1"), Some("0"))?;
    layout_only.auto = true;
    assert_eq!(layout_only.resolve(), (true, false));
    assert!(super::ConvExecutionPolicy::from_values(Some("invalid"), None).is_err());
    Ok(())
}

fn assert_cross_sample_conv_matches_reference(
    device: &candle_core::Device,
    dtype: candle_core::DType,
    tolerance: f32,
) -> anyhow::Result<()> {
    let mut cfg = test_config();
    cfg.n_window = 50;
    cfg.n_window_infer = 200;
    let vb = candle_nn::VarBuilder::from_backend(Box::new(SeededBackend), dtype, device.clone());
    let mut tower = AudioTower::load(&cfg, vb, false, None, &Progress::disabled())?;
    // Force multiple microbatches, including a boundary in the middle of a sample.
    tower.conv_micro_batch = 2;
    let lens = [0usize, 1, 99, 100, 101, 399, 663, 0];
    let frames = 663;
    let values = (0..lens.len() * cfg.num_mel_bins * frames)
        .map(|i| ((i * 19 % 113) as f32 - 56.0) / 61.0)
        .collect::<Vec<_>>();
    let features =
        candle_core::Tensor::from_vec(values, (lens.len(), cfg.num_mel_bins, frames), device)?;
    let features = super::cast_input_features_to_weight_dtype(&features, dtype)?;
    let window = cfg.n_window * 2;
    let got = tower.forward_conv_cross_sample(&features, &lens, window)?;
    assert_eq!(got.len(), lens.len());
    for (index, (&len, got)) in lens.iter().zip(&got).enumerate() {
        let input = features
            .narrow(0, index, 1)?
            .squeeze(0)?
            .narrow(1, 0, len)?;
        let reference = tower.forward_conv_sample(&input, window)?;
        assert_eq!(
            (got.aftercnn_total, got.max_t, got.chunks),
            (reference.aftercnn_total, reference.max_t, reference.chunks)
        );
        assert_tensor_close(&got.hidden_states, &reference.hidden_states, tolerance)?;
        if got.hidden_states.elem_count() > 0 {
            let values = got
                .hidden_states
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(values.iter().all(|value| value.is_finite()));
        }
    }
    tower.conv_cross_sample = false;
    let reference = tower.forward_with_lens(&features, &lens)?;
    tower.conv_cross_sample = true;
    let routed = tower.forward_with_lens(&features, &lens)?;
    let encoder_tolerance = if dtype == candle_core::DType::BF16 {
        0.1
    } else {
        tolerance
    };
    assert_tensor_close(&routed, &reference, encoder_tolerance)?;
    let empty = tower.forward_conv_cross_sample(&features, &[0; 8], window)?;
    assert!(empty.iter().all(|sample| sample.aftercnn_total == 0
        && sample.max_t == 0
        && sample.chunks == 0
        && sample.hidden_states.dims() == [0, cfg.d_model]));
    Ok(())
}

#[test]
fn test_cross_sample_conv_nonzero_cpu() -> anyhow::Result<()> {
    assert_cross_sample_conv_matches_reference(
        &candle_core::Device::Cpu,
        candle_core::DType::F32,
        1e-4,
    )
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires exclusive Metal GPU; run after stopping soak/benchmark server"]
fn test_cross_sample_conv_nonzero_metal_f32() -> anyhow::Result<()> {
    assert_cross_sample_conv_matches_reference(
        &candle_core::Device::new_metal(0)?,
        candle_core::DType::F32,
        1e-3,
    )
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires exclusive Metal GPU; run after stopping soak/benchmark server"]
fn test_cross_sample_conv_nonzero_metal_bf16() -> anyhow::Result<()> {
    assert_cross_sample_conv_matches_reference(
        &candle_core::Device::new_metal(0)?,
        candle_core::DType::BF16,
        0.025,
    )
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an idle CUDA GPU"]
fn test_cross_sample_conv_nonzero_cuda_bf16() -> anyhow::Result<()> {
    assert_cross_sample_conv_matches_reference(
        &candle_core::Device::new_cuda(0)?,
        candle_core::DType::BF16,
        0.025,
    )
}

fn assert_attention_modes(
    device: &candle_core::Device,
    dtype: candle_core::DType,
    tolerance: f32,
) -> anyhow::Result<()> {
    use super::AudioAttentionMode;
    let mut cfg = test_config();
    cfg.encoder_attention_heads = 2;
    let vb = candle_nn::VarBuilder::from_backend(Box::new(SeededBackend), dtype, device.clone());
    let mut attention = AudioAttention::load(&cfg, vb, false, None)?;
    attention.mode = AudioAttentionMode::Reference;
    // Unequal tails, repeated lengths, empty window, and multiple microbatches.
    let lengths = [3usize, 1, 3, 2, 0, 3, 3, 3, 3, 3, 3, 3, 3, 2];
    let mut cu = vec![0];
    for len in lengths {
        cu.push(cu.last().copied().unwrap_or(0) + len);
    }
    let seq = cu.last().copied().unwrap_or(0);
    let values: Vec<f32> = (0..seq * cfg.d_model)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) / 29.0)
        .collect();
    let xs = candle_core::Tensor::from_vec(values, (seq, cfg.d_model), device)?.to_dtype(dtype)?;
    let expected = attention.forward(xs.clone(), &cu)?;
    for mode in [
        AudioAttentionMode::ProjectOnce,
        AudioAttentionMode::Grouped,
        AudioAttentionMode::GroupedNative,
    ] {
        attention.mode = mode;
        let got = attention.forward(xs.clone(), &cu)?;
        assert_tensor_close(&got, &expected, tolerance)?;
        let mut isolated = Vec::new();
        for window in cu.windows(2) {
            let len = window[1] - window[0];
            if len > 0 {
                isolated.push(attention.forward(xs.narrow(0, window[0], len)?, &[0, len])?);
            }
        }
        let isolated = candle_core::Tensor::cat(&isolated.iter().collect::<Vec<_>>(), 0)?;
        assert_tensor_close(&got, &isolated, tolerance)?;
    }
    Ok(())
}

fn assert_tensor_close(
    got: &candle_core::Tensor,
    expected: &candle_core::Tensor,
    tolerance: f32,
) -> anyhow::Result<()> {
    assert_eq!(got.dims(), expected.dims());
    // Empty shapes already establish equality. Candle Metal cannot dispatch
    // a zero-element cast (linear_split divides by zero while encoding).
    if got.elem_count() == 0 {
        return Ok(());
    }
    let got = got
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(
        got.iter().chain(&expected).all(|value| value.is_finite()),
        "non-finite attention values"
    );
    let max_abs = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs <= tolerance,
        "attention mismatch: max_abs={max_abs} tolerance={tolerance}"
    );
    Ok(())
}

#[test]
fn test_attention_optimized_modes_preserve_values_and_isolation_cpu() -> anyhow::Result<()> {
    assert_attention_modes(&candle_core::Device::Cpu, candle_core::DType::F32, 1e-5)
}

#[cfg(feature = "metal")]
#[test]
fn test_attention_optimized_modes_preserve_values_and_isolation_metal_f32() -> anyhow::Result<()> {
    assert_attention_modes(
        &candle_core::Device::new_metal(0)?,
        candle_core::DType::F32,
        1e-3,
    )
}

#[cfg(feature = "metal")]
#[test]
fn test_attention_optimized_modes_preserve_values_and_isolation_metal_bf16() -> anyhow::Result<()> {
    assert_attention_modes(
        &candle_core::Device::new_metal(0)?,
        candle_core::DType::BF16,
        0.025,
    )
}

#[test]
fn test_sinusoidal_positional_embedding_shape() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let pe = sinusoidal_positional_embedding(10, 8, &device)?;
    if pe.dims() != vec![10, 8] {
        anyhow::bail!("unexpected pe dims: {:?}", pe.dims());
    }
    Ok(())
}

#[test]
fn test_feature_lens_from_mask_counts_nonzero() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let mask =
        candle_core::Tensor::from_vec(vec![1u32, 1, 0, 0, 1, 1, 1, 0], (2usize, 4usize), &device)?;
    let lens = feature_lens_from_mask(&mask, 4)?;
    if lens != vec![2usize, 3usize] {
        anyhow::bail!("unexpected lens: {lens:?}");
    }
    Ok(())
}

#[test]
fn test_audio_tower_forward_with_lens_matches_mask() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let cfg = AudioEncoderConfig {
        num_mel_bins: 4,
        encoder_layers: 1,
        encoder_attention_heads: 1,
        encoder_ffn_dim: 8,
        d_model: 4,
        dropout: 0.0,
        attention_dropout: 0.0,
        activation_function: "gelu".to_string(),
        activation_dropout: 0.0,
        scale_embedding: false,
        initializer_range: 0.02,
        max_source_positions: 64,
        n_window: 5,
        output_dim: 6,
        n_window_infer: 20,
        conv_chunksize: 8,
        downsample_hidden_size: 2,
    };
    let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
    let tower = AudioTower::load(&cfg, vb, false, None, &Progress::disabled())?;

    let batch = 2usize;
    let mel = cfg.num_mel_bins;
    let frames = 10usize;
    let feats: Vec<f32> = (0..batch * mel * frames).map(|i| i as f32).collect();
    let input_features = candle_core::Tensor::from_vec(feats, (batch, mel, frames), &device)?;

    let len1 = 6usize;
    let mut mask: Vec<u32> = Vec::with_capacity(batch * frames);
    mask.extend(std::iter::repeat_n(1u32, frames));
    mask.extend(std::iter::repeat_n(1u32, len1));
    mask.extend(std::iter::repeat_n(0u32, frames - len1));
    let mask = candle_core::Tensor::from_vec(mask, (batch, frames), &device)?;

    let out_mask = tower.forward(&input_features, Some(&mask))?;
    let out_lens = tower.forward_with_lens(&input_features, &[frames, len1])?;

    if out_mask.dims() != out_lens.dims() {
        anyhow::bail!(
            "output dims mismatch: mask={:?} lens={:?}",
            out_mask.dims(),
            out_lens.dims()
        );
    }

    let got = out_mask.to_vec2::<f32>()?;
    let exp = out_lens.to_vec2::<f32>()?;
    let mut max_abs = 0f32;
    for (row_g, row_e) in got.iter().zip(exp.iter()) {
        for (&a, &b) in row_g.iter().zip(row_e.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    if max_abs > 1e-6 {
        anyhow::bail!("forward mismatch: max_abs={max_abs}");
    }
    Ok(())
}

#[test]
fn test_audio_tower_batched_forward_matches_per_sample_forward() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let cfg = AudioEncoderConfig {
        num_mel_bins: 4,
        encoder_layers: 1,
        encoder_attention_heads: 1,
        encoder_ffn_dim: 8,
        d_model: 4,
        dropout: 0.0,
        attention_dropout: 0.0,
        activation_function: "gelu".to_string(),
        activation_dropout: 0.0,
        scale_embedding: false,
        initializer_range: 0.02,
        max_source_positions: 64,
        n_window: 5,
        output_dim: 6,
        n_window_infer: 20,
        conv_chunksize: 8,
        downsample_hidden_size: 2,
    };
    let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
    let tower = AudioTower::load(&cfg, vb, false, None, &Progress::disabled())?;

    let batch = 3usize;
    let mel = cfg.num_mel_bins;
    let frames = 10usize;
    let lens = [10usize, 7usize, 4usize];
    let feats: Vec<f32> = (0..batch * mel * frames)
        .map(|i| (i as f32) / 100.0)
        .collect();
    let input_features = candle_core::Tensor::from_vec(feats, (batch, mel, frames), &device)?;

    let batched = tower.forward_with_lens(&input_features, &lens)?;
    let weight_dtype = tower.conv2d1.weight().dtype();
    let mut per_sample = Vec::with_capacity(batch);
    for (idx, &len) in lens.iter().enumerate() {
        let input = input_features
            .narrow(0, idx, 1)?
            .squeeze(0)?
            .narrow(1, 0, len)?;
        // `forward_one` 是测试辅助路径，不负责 dtype 转换，这里手动对齐权重 dtype。
        let input = super::cast_input_features_to_weight_dtype(&input, weight_dtype)?;
        per_sample.push(tower.forward_one(&input)?);
    }
    let expected = candle_core::Tensor::cat(&per_sample.iter().collect::<Vec<_>>(), 0)?;

    if batched.dims() != expected.dims() {
        anyhow::bail!(
            "batched dims mismatch: batched={:?} expected={:?}",
            batched.dims(),
            expected.dims()
        );
    }

    let got = batched.to_vec2::<f32>()?;
    let exp = expected.to_vec2::<f32>()?;
    let mut max_abs = 0f32;
    for (row_g, row_e) in got.iter().zip(exp.iter()) {
        for (&a, &b) in row_g.iter().zip(row_e.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    if max_abs > 1e-5 {
        anyhow::bail!("batched forward mismatch: max_abs={max_abs}");
    }
    Ok(())
}

#[test]
fn test_nonzero_weights_batched_forward_matches_per_sample() -> anyhow::Result<()> {
    // 零权重测试无法暴露 padding / 窗口边界错误；这里用非零权重复验多段拼接。
    assert_batched_matches_per_sample(
        &candle_core::Device::Cpu,
        candle_core::DType::F32,
        false,
        1e-4,
    )
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_nonzero_weights_batched_forward_matches_per_sample_f32() -> anyhow::Result<()> {
    let device = candle_core::Device::new_metal(0)?;
    assert_batched_matches_per_sample(&device, candle_core::DType::F32, false, 1e-3)
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_nonzero_weights_batched_forward_matches_per_sample_bf16() -> anyhow::Result<()> {
    let device = candle_core::Device::new_metal(0)?;
    // BF16 精度较低，放宽到 1e-1；重点验证多段 padding / 边界是否一致。
    assert_batched_matches_per_sample(&device, candle_core::DType::BF16, false, 1e-1)
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_input_features_can_be_cast_to_bf16() -> anyhow::Result<()> {
    let device = candle_core::Device::new_metal(0)?;
    let input = candle_core::Tensor::from_vec(vec![1f32, 2.0], (1usize, 1usize, 2usize), &device)?;

    let got = super::cast_input_features_to_weight_dtype(&input, candle_core::DType::BF16)?;

    assert_eq!(got.dtype(), candle_core::DType::BF16);
    assert!(got.device().is_metal());
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_bf16_transpose_can_be_materialized_contiguous() -> anyhow::Result<()> {
    let device = candle_core::Device::new_metal(0)?;
    let input =
        candle_core::Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2usize, 2usize), &device)?;
    let input = super::cast_input_features_to_weight_dtype(&input, candle_core::DType::BF16)?;

    let got = input.transpose(0, 1)?.contiguous()?;

    assert_eq!(got.dtype(), candle_core::DType::BF16);
    assert!(got.device().is_metal());
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "exclusive Metal chunk preparation diagnostic"]
fn diagnose_native_bf16_chunk_preparation() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    let device = Device::new_metal(0)?;
    for frames in [5usize, 105, 805, 3005] {
        let mel = 128;
        let input = Tensor::from_vec(
            (0..mel * (frames + 7))
                .map(|i| ((i % 193) as f32 - 96.) / 11.)
                .collect::<Vec<_>>(),
            (mel, frames + 7),
            &device,
        )?
        .to_dtype(DType::BF16)?
        .narrow(1, 7, frames)?
        .transpose(0, 1)?;
        let mut reference = Vec::new();
        let mut native = Vec::new();
        for start in (0..frames).step_by(100) {
            let len = (frames - start).min(100);
            let chunk = input.narrow(0, start, len)?;
            reference.push(super::pad_audio_chunk(&chunk, 0, 0, 100 - len, false)?);
            native.push(super::pad_audio_chunk(&chunk, 0, 0, 100 - len, true)?);
        }
        let ref_stack = super::stack_audio_chunks(&reference, false)?;
        let native_stack = super::stack_audio_chunks(&native, true)?;
        assert_tensor_close(&native_stack, &ref_stack, 0.)?;
        for phase in ["compat", "native"] {
            let mut times = Vec::new();
            for iteration in 0..24 {
                device.synchronize()?;
                let start_time = std::time::Instant::now();
                let chunks = (0..frames)
                    .step_by(100)
                    .map(|start| {
                        let len = (frames - start).min(100);
                        let chunk = input.narrow(0, start, len)?;
                        if phase == "compat" {
                            super::pad_audio_chunk(&chunk, 0, 0, 100 - len, false)
                        } else {
                            chunk.pad_with_zeros(0, 0, 100 - len)
                        }
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;
                let output = if phase == "compat" {
                    super::stack_audio_chunks(&chunks, false)?
                } else {
                    Tensor::stack(&chunks.iter().collect::<Vec<_>>(), 0)?
                };
                device.synchronize()?;
                std::hint::black_box(output);
                if iteration >= 4 {
                    times.push(start_time.elapsed().as_secs_f64() * 1e6);
                }
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "chunk_prep frames={frames} phase={phase} median_us={:.3} min_us={:.3} max_us={:.3}",
                (times[9] + times[10]) / 2.,
                times[0],
                times[19]
            );
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_bf16_audio_chunk_padding_avoids_strided_copy_kernel() -> anyhow::Result<()> {
    let device = candle_core::Device::new_metal(0)?;
    let mel = 128usize;
    let frames = 1505usize;
    let input = candle_core::Tensor::from_vec(
        (0..mel * frames).map(|i| i as f32).collect::<Vec<_>>(),
        (1usize, mel, frames),
        &device,
    )?;
    let input = super::cast_input_features_to_weight_dtype(&input, candle_core::DType::BF16)?;
    let input = input.narrow(0, 0, 1)?.squeeze(0)?;

    let input_t = input.transpose(0, 1)?;
    let chunk = input_t.narrow(0, 1500, 5)?;
    let padded = super::pad_audio_chunk(&chunk, 0, 0, 95, false)?;

    assert_eq!(padded.dtype(), candle_core::DType::BF16);
    assert!(padded.device().is_metal());
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_bf16_audio_chunks_can_be_stacked() -> anyhow::Result<()> {
    let device = candle_core::Device::new_metal(0)?;
    let mel = 128usize;
    let frames = 1505usize;
    let input = candle_core::Tensor::from_vec(
        (0..mel * frames).map(|i| i as f32).collect::<Vec<_>>(),
        (1usize, mel, frames),
        &device,
    )?;
    let input = super::cast_input_features_to_weight_dtype(&input, candle_core::DType::BF16)?;
    let input = input.narrow(0, 0, 1)?.squeeze(0)?;

    let input_t = input.transpose(0, 1)?;
    let mut chunks = Vec::new();
    for offset in (0..frames).step_by(100) {
        let len = (frames - offset).min(100);
        let chunk = input_t.narrow(0, offset, len)?;
        chunks.push(super::pad_audio_chunk(&chunk, 0, 0, 100 - len, false)?);
    }
    let stacked = super::stack_audio_chunks(&chunks, false)?;

    assert_eq!(stacked.dtype(), candle_core::DType::BF16);
    assert!(stacked.device().is_metal());
    Ok(())
}

#[test]
fn test_cu_seqlens_from_aftercnn_len_splits_into_windows() -> anyhow::Result<()> {
    // Mirrors the official computation:
    // window_aftercnn = max_chunk_len_aftercnn * (n_window_infer / (n_window*2))
    let aftercnn_len = 100usize;
    let max_chunk_len_aftercnn = 26usize;
    let window = 200usize; // n_window*2
    let n_window_infer = 400usize;
    let cu =
        cu_seqlens_from_aftercnn_len(aftercnn_len, max_chunk_len_aftercnn, window, n_window_infer)?;
    if cu != vec![0usize, 52, 100] {
        anyhow::bail!("unexpected cu_seqlens: {cu:?}");
    }
    Ok(())
}

#[test]
fn test_audio_attention_respects_cu_seqlens_when_eager() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let embed_dim = 4usize;
    let seq_len = 5usize;

    let mut eye = vec![0f32; embed_dim * embed_dim];
    for i in 0..embed_dim {
        eye[i * embed_dim + i] = 1.0;
    }
    let w = candle_core::Tensor::from_vec(eye, (embed_dim, embed_dim), &device)?;
    let q_proj = IsqLinear::new(Linear::new(w.clone(), None), None, &device)?;
    let k_proj = IsqLinear::new(Linear::new(w.clone(), None), None, &device)?;
    let v_proj = IsqLinear::new(Linear::new(w.clone(), None), None, &device)?;
    let out_proj = IsqLinear::new(Linear::new(w, None), None, &device)?;

    let attn = AudioAttention {
        mode: super::AudioAttentionMode::Reference,
        use_flash_attn: false,
        num_heads: 1,
        head_dim: embed_dim,
        scaling: (embed_dim as f64).powf(-0.5),
        q_proj,
        k_proj,
        v_proj,
        out_proj,
    };

    let xs: Vec<f32> = (0..seq_len * embed_dim).map(|i| i as f32).collect();
    let hidden_states = candle_core::Tensor::from_vec(xs, (seq_len, embed_dim), &device)?;

    let out_blocked = attn.forward(hidden_states.clone(), &[0usize, 2, 5])?;
    let out_full = attn.forward(hidden_states, &[0usize, 5])?;

    let got = out_blocked.to_vec2::<f32>()?;
    let exp = out_full.to_vec2::<f32>()?;
    let mut max_abs = 0f32;
    for (row_g, row_e) in got.iter().zip(exp.iter()) {
        for (&a, &b) in row_g.iter().zip(row_e.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    if max_abs <= 1e-4 {
        anyhow::bail!("segmented attention unexpectedly matched full attention");
    }
    Ok(())
}
