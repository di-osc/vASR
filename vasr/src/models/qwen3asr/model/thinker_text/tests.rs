use super::load_multimodal_rope;
#[cfg(feature = "paged-attn")]
use super::unpack_gathered_kv_for_attention;

#[test]
fn legacy_rope_parsers_keep_case_insensitive_boolean_contract() -> anyhow::Result<()> {
    for parse in [
        super::parse_prefill_gate_up_policy,
        super::parse_prepared_mrope_policy,
        super::parse_packed_rope_policy,
    ] {
        assert!(!parse(None)?);
        for value in ["0", "false", "off", " OFF "] {
            assert!(!parse(Some(value))?);
        }
        for value in ["1", "true", "on", " TRUE "] {
            assert!(parse(Some(value))?);
        }
        for value in ["", "yes", "no", "invalid"] {
            assert!(parse(Some(value)).is_err());
        }
    }
    Ok(())
}

fn dense_fixture(
    out: usize,
    input: usize,
    bias: bool,
    seed: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> candle_core::Result<super::IsqLinear> {
    let values = (0..out * input)
        .map(|i| (((i * 17 + seed * 23) % 97) as f32 - 48.0) / 193.0)
        .collect::<Vec<_>>();
    let weight = candle_core::Tensor::from_vec(values, (out, input), device)?.to_dtype(dtype)?;
    let bias = if bias {
        Some(
            candle_core::Tensor::from_vec(
                (0..out)
                    .map(|i| (i as f32 - 3.0) / 53.0)
                    .collect::<Vec<_>>(),
                out,
                device,
            )?
            .to_dtype(dtype)?,
        )
    } else {
        None
    };
    Ok(super::IsqLinear::Linear(candle_nn::Linear::new(
        weight, bias,
    )))
}

#[test]
fn test_metal_dense_projection_fusion_leaves_cpu_unchanged() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let gate = dense_fixture(12, 8, false, 1, candle_core::DType::F32, &device)?;
    let up = dense_fixture(12, 8, false, 2, candle_core::DType::F32, &device)?;
    assert!(super::MetalDenseProjection::new(&[&gate, &up], true)?.is_none());
    Ok(())
}

#[test]
fn measured_rope_defaults_require_validated_profile_and_keep_overrides() {
    use super::resolve_rope_defaults as resolve;
    assert_eq!(resolve(None, None, false), (false, false));
    assert_eq!(resolve(None, None, true), (true, true));
    assert_eq!(resolve(Some(false), None, true), (false, false));
    assert_eq!(resolve(None, Some(false), true), (false, false));
    assert_eq!(resolve(Some(false), Some(true), true), (false, false));
    assert_eq!(resolve(Some(true), Some(false), true), (true, false));
    assert_eq!(resolve(None, Some(true), false), (true, true));
    assert_eq!(resolve(Some(true), None, false), (true, false));
}

#[test]
fn packed_rope_policy_requires_explicit_boolean() -> anyhow::Result<()> {
    assert!(!super::parse_packed_rope_policy(None)?);
    assert!(super::parse_packed_rope_policy(Some(" TRUE "))?);
    assert!(!super::parse_packed_rope_policy(Some("off"))?);
    assert!(super::parse_packed_rope_policy(Some("invalid")).is_err());
    Ok(())
}

#[test]
fn prepared_mrope_candidate_requires_explicit_opt_in() -> anyhow::Result<()> {
    assert!(!super::parse_prepared_mrope_policy(None)?);
    for value in ["0", "false", "off", " FALSE "] {
        assert!(!super::parse_prepared_mrope_policy(Some(value))?);
    }
    for value in ["1", "true", "on", " TRUE "] {
        assert!(super::parse_prepared_mrope_policy(Some(value))?);
    }
    assert!(super::parse_prepared_mrope_policy(Some("invalid")).is_err());
    Ok(())
}

#[test]
fn prefill_gate_up_candidate_requires_explicit_opt_in() -> anyhow::Result<()> {
    assert!(!super::parse_prefill_gate_up_policy(None)?);
    for value in ["0", "false", "off"] {
        assert!(!super::parse_prefill_gate_up_policy(Some(value))?);
    }
    for value in ["1", "true", "on"] {
        assert!(super::parse_prefill_gate_up_policy(Some(value))?);
    }
    assert!(super::parse_prefill_gate_up_policy(Some("invalid")).is_err());
    Ok(())
}

#[cfg(feature = "metal-paged-attn")]
#[test]
#[ignore = "exclusive Metal prefill-only MLP numerical comparison"]
fn prefill_gate_up_candidate_preserves_decode_and_shared_weights() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::Module;
    use vasr_quant::{
        isq_linear::set_linear_is_prefill,
        metal_gemm::{GemmTile, set_prefill_tile},
    };
    let device = Device::new_metal(0)?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let gate = dense_fixture(96, 32, false, 1, dtype, &device)?;
        let up = dense_fixture(96, 32, false, 2, dtype, &device)?;
        let down = dense_fixture(32, 96, false, 3, dtype, &device)?;
        let projection = super::MetalDenseProjection::new(&[&gate, &up], true)?.unwrap();
        let mut shared = projection.shared_linears()?.into_iter();
        let shared_gate = shared.next().unwrap();
        let shared_up = shared.next().unwrap();
        for (index, linear) in [&shared_gate, &shared_up].into_iter().enumerate() {
            let super::IsqLinear::Linear(linear) = linear else {
                panic!("dense view expected")
            };
            assert!(linear.weight().is_contiguous());
            assert_eq!(linear.weight().dims(), [96, 32]);
            let (_storage, layout) = linear.weight().storage_and_layout();
            assert_eq!(layout.start_offset(), index * 96 * 32);
        }
        let reference = super::ThinkerTextMlp {
            fused_gate_up: None,
            prefill_gate_up: None,
            prefill_fused_silu_mul: false,
            fused_silu_mul: false,
            gate_proj: gate,
            up_proj: up,
            down_proj: down.clone(),
            hidden_act: "silu".into(),
        };
        let candidate = super::ThinkerTextMlp {
            fused_gate_up: None,
            prefill_gate_up: Some(projection),
            prefill_fused_silu_mul: true,
            fused_silu_mul: false,
            gate_proj: shared_gate,
            up_proj: shared_up,
            down_proj: down,
            hidden_act: "silu".into(),
        };
        for rows in [1, 17, 65, 257] {
            let input = Tensor::from_vec(
                (0..(rows + 1) * 32)
                    .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                    .collect::<Vec<_>>(),
                (1, rows + 1, 32),
                &device,
            )?
            .to_dtype(dtype)?
            .narrow(1, 1, rows)?;
            let _tiles = set_prefill_tile(Some(GemmTile::Adaptive));
            {
                let _decode = set_linear_is_prefill(false);
                assert!(candidate.prefill_projection().is_none());
                assert_dense_close(&candidate.forward(&input)?, &reference.forward(&input)?, 0.)?;
            }
            let _prefill = set_linear_is_prefill(true);
            assert!(candidate.prefill_projection().is_some());
            let tolerance = match dtype {
                DType::F32 => 1e-4,
                DType::F16 => 0.005,
                _ => 0.025,
            };
            assert_dense_close(
                &candidate.forward(&input)?,
                &reference.forward(&input)?,
                tolerance,
            )?;
            let mut disabled_activation = candidate.clone();
            disabled_activation.prefill_fused_silu_mul = false;
            assert_dense_close(
                &disabled_activation.forward(&input)?,
                &reference.forward(&input)?,
                tolerance,
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "metal-paged-attn")]
#[test]
#[ignore = "requires exclusive Metal GPU"]
fn test_metal_joined_gate_up_views_preserve_values() -> anyhow::Result<()> {
    let device = candle_core::Device::new_metal(0)?;
    for dtype in [candle_core::DType::F32, candle_core::DType::BF16] {
        let gate = dense_fixture(12, 8, false, 1, dtype, &device)?;
        let up = dense_fixture(12, 8, false, 2, dtype, &device)?;
        let projection = super::MetalDenseProjection::new(&[&gate, &up], true)?.unwrap();
        for (batch, seq) in [(1usize, 1usize), (32, 1), (3, 17)] {
            let x = candle_core::Tensor::from_vec(
                (0..batch * seq * 8)
                    .map(|i| ((i * 11 % 83) as f32 - 41.) / 71.)
                    .collect::<Vec<_>>(),
                (batch, seq, 8),
                &device,
            )?
            .to_dtype(dtype)?;
            let copied = projection.forward(&x)?;
            let views = projection.forward_views(&x)?;
            assert_eq!(views.len(), 2);
            for (view, copied) in views.iter().zip(&copied) {
                if batch * seq > 1 {
                    assert!(!view.is_contiguous());
                }
                assert_dense_close(view, copied, 0.)?;
            }
            let hidden = vasr_paged_attn::metal::pointwise::silu_mul(&views[0], &views[1])?;
            let reference = copied[0].silu()?.mul(&copied[1])?;
            assert_dense_close(&hidden, &reference, 0.)?;
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn assert_metal_dense_projection_fusion(
    dtype: candle_core::DType,
    tolerance: f32,
) -> anyhow::Result<()> {
    use candle_nn::Module;
    let device = candle_core::Device::new_metal(0)?;
    let gate = dense_fixture(12, 8, false, 1, dtype, &device)?;
    let up = dense_fixture(12, 8, false, 2, dtype, &device)?;
    let down = dense_fixture(8, 12, false, 3, dtype, &device)?;
    let reference = super::ThinkerTextMlp {
        prefill_gate_up: None,
        prefill_fused_silu_mul: false,
        fused_silu_mul: false,
        gate_proj: gate.clone(),
        up_proj: up.clone(),
        down_proj: down.clone(),
        hidden_act: "silu".into(),
        fused_gate_up: None,
    };
    let fused = super::ThinkerTextMlp {
        prefill_gate_up: None,
        prefill_fused_silu_mul: false,
        fused_silu_mul: true,
        gate_proj: gate.clone(),
        up_proj: up.clone(),
        down_proj: down,
        hidden_act: "silu".into(),
        fused_gate_up: super::MetalDenseProjection::new(&[&gate, &up], true)?,
    };
    assert!(fused.fused_gate_up.is_some());
    assert!(super::MetalDenseProjection::new(&[&gate, &up], false)?.is_none());
    for batch in [1usize, 8, 32] {
        for seq in [1usize, 5] {
            let xs = candle_core::Tensor::from_vec(
                (0..batch * seq * 8)
                    .map(|i| ((i * 11 % 83) as f32 - 41.0) / 71.0)
                    .collect::<Vec<_>>(),
                (batch, seq, 8),
                &device,
            )?
            .to_dtype(dtype)?;
            assert_dense_close(&fused.forward(&xs)?, &reference.forward(&xs)?, tolerance)?;
            // Q has more output channels than K/V; exercise bias and no bias.
            for bias in [false, true] {
                let q = dense_fixture(8, 8, bias, 4, dtype, &device)?;
                let k = dense_fixture(4, 8, bias, 5, dtype, &device)?;
                let v = dense_fixture(4, 8, bias, 6, dtype, &device)?;
                let projection = super::MetalDenseProjection::new(&[&q, &k, &v], true)?
                    .ok_or_else(|| anyhow::anyhow!("missing fused dense QKV"))?;
                let got = projection.forward(&xs)?;
                for (output, reference) in got.iter().zip([&q, &k, &v]) {
                    assert_dense_close(output, &reference.forward(&xs)?, tolerance)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn assert_dense_close(
    got: &candle_core::Tensor,
    expected: &candle_core::Tensor,
    tolerance: f32,
) -> anyhow::Result<()> {
    assert_eq!(got.dims(), expected.dims());
    let got = got
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(got.iter().chain(&expected).all(|x| x.is_finite()));
    let max_abs = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_abs <= tolerance,
        "dense fusion mismatch max_abs={max_abs}, tolerance={tolerance}"
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires exclusive Metal GPU; run explicitly after stopping benchmark server"]
fn test_metal_dense_projection_fusion_f32() -> anyhow::Result<()> {
    assert_metal_dense_projection_fusion(candle_core::DType::F32, 1e-4)
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires exclusive Metal GPU; run explicitly after stopping benchmark server"]
fn test_metal_dense_projection_fusion_bf16() -> anyhow::Result<()> {
    assert_metal_dense_projection_fusion(candle_core::DType::BF16, 0.025)
}

#[cfg(feature = "metal-paged-attn")]
#[test]
#[ignore = "exclusive Metal prefill gate/up projection microbenchmark"]
fn diagnose_prefill_gate_up_projection() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use vasr_quant::metal_gemm::{GemmTile, matmul_nt, set_prefill_tile};
    let device = Device::new_metal(0)?;
    let gate = dense_fixture(3072, 1024, false, 1, DType::BF16, &device)?;
    let up = dense_fixture(3072, 1024, false, 2, DType::BF16, &device)?;
    let joined = super::MetalDenseProjection::new(&[&gate, &up], true)?.unwrap();
    let _policy = set_prefill_tile(Some(GemmTile::Adaptive));
    for rows in [65, 256, 640, 1000, 2000] {
        let input = Tensor::from_vec(
            (0..rows * 1024)
                .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                .collect::<Vec<_>>(),
            (1, rows, 1024),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let run = |index| -> candle_core::Result<Tensor> {
            match index {
                0 => {
                    let gate = candle_nn::ops::silu(&gate.forward(&input)?)?;
                    gate.broadcast_mul(&up.forward(&input)?)
                }
                1 => {
                    let outputs = joined.forward(&input)?;
                    candle_nn::ops::silu(&outputs[0])?.broadcast_mul(&outputs[1])
                }
                2 => {
                    let views = joined.forward_views(&input)?;
                    vasr_paged_attn::metal::pointwise::silu_mul(&views[0], &views[1])
                }
                _ => {
                    let joined = matmul_nt(&input, joined.linear.weight(), GemmTile::Adaptive)?;
                    vasr_paged_attn::metal::pointwise::silu_mul(
                        &joined.narrow(2, 0, 3072)?,
                        &joined.narrow(2, 3072, 3072)?,
                    )
                }
            }
        };
        let reference = run(0)?;
        for index in 1..4 {
            assert_dense_close(&run(index)?, &reference, 0.125)?;
        }
        let mut times: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        for iteration in 0..60 {
            for offset in 0..4 {
                let index = if iteration % 2 == 0 {
                    (iteration + offset) % 4
                } else {
                    (iteration + 4 - offset) % 4
                };
                device.synchronize()?;
                let start = std::time::Instant::now();
                let output = run(index)?;
                device.synchronize()?;
                std::hint::black_box(output);
                if iteration >= 12 {
                    times[index].push(start.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        for (mode, times) in [
            "independent_adaptive",
            "joined_candle_copy",
            "joined_candle_views",
            "joined_adaptive_views",
        ]
        .into_iter()
        .zip(&mut times)
        {
            times.sort_by(f64::total_cmp);
            eprintln!(
                "gate_up rows={rows} mode={mode} median_us={:.3} min_us={:.3} max_us={:.3}",
                (times[23] + times[24]) / 2.,
                times[0],
                times[47]
            );
        }
    }
    Ok(())
}

#[cfg(feature = "metal-paged-attn")]
#[test]
#[ignore = "exclusive Metal Qwen3-ASR-0.6B QKV projection microbenchmark"]
fn diagnose_prefill_qkv_projection() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use vasr_quant::isq_linear::set_linear_is_prefill;
    use vasr_quant::metal_gemm::{GemmTile, set_prefill_tile};
    let device = Device::new_metal(0)?;
    // Actual 0.6B geometry: 16 query heads / 8 KV heads, head_dim=128.
    // Materialize each joined slice, as project_qkv consumers reshape it.
    let q = dense_fixture(2048, 1024, false, 4, DType::BF16, &device)?;
    let k = dense_fixture(1024, 1024, false, 5, DType::BF16, &device)?;
    let v = dense_fixture(1024, 1024, false, 6, DType::BF16, &device)?;
    let joined = super::MetalDenseProjection::new(&[&q, &k, &v], true)?.unwrap();
    let _policy = set_prefill_tile(Some(GemmTile::Adaptive));
    let _prefill = set_linear_is_prefill(true);
    for rows in [65, 256, 640, 1000, 2000] {
        let input = Tensor::from_vec(
            (0..(rows + 1) * 1024)
                .map(|i| ((i * 17 % 193) as f32 - 96.) / 151.)
                .collect::<Vec<_>>(),
            (1, rows + 1, 1024),
            &device,
        )?
        .to_dtype(DType::BF16)?
        .narrow(1, 1, rows)?;
        let run = |index| -> candle_core::Result<Vec<Tensor>> {
            match index {
                0 => Ok(vec![
                    q.forward(&input)?,
                    k.forward(&input)?,
                    v.forward(&input)?,
                ]),
                1 => joined.forward(&input),
                _ => joined
                    .forward_prefill_views(&input)?
                    .into_iter()
                    .map(|view| view.contiguous())
                    .collect(),
            }
        };
        let reference = run(0)?;
        for index in 1..3 {
            for (got, expected) in run(index)?.iter().zip(&reference) {
                assert!(got.is_contiguous());
                assert_dense_close(got, expected, 0.125)?;
            }
        }
        let mut times: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        for iteration in 0..60 {
            for offset in 0..3 {
                let index = if iteration % 2 == 0 {
                    (iteration + offset) % 3
                } else {
                    (iteration + 3 - offset) % 3
                };
                device.synchronize()?;
                let start = std::time::Instant::now();
                let output = run(index)?;
                device.synchronize()?;
                std::hint::black_box(output);
                if iteration >= 12 {
                    times[index].push(start.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        for (mode, times) in [
            "independent_adaptive",
            "joined_candle_copy",
            "joined_adaptive_copy",
        ]
        .into_iter()
        .zip(&mut times)
        {
            times.sort_by(f64::total_cmp);
            eprintln!(
                "qkv rows={rows} mode={mode} median_us={:.3} min_us={:.3} max_us={:.3}",
                (times[23] + times[24]) / 2.,
                times[0],
                times[47]
            );
        }
    }
    Ok(())
}

#[test]
fn test_rotary_embedding_loads_without_rope_scaling() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let cfg = crate::config::TextConfig::default();
    let _ = load_multimodal_rope(&cfg, &device)?;
    Ok(())
}

#[cfg(feature = "paged-attn")]
#[test]
fn test_unpack_gathered_kv_for_attention_repeats_groups() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let packed = candle_core::Tensor::from_vec(
        vec![1f32, 10.0, 2.0, 20.0, 3.0, 30.0],
        (3usize, 2usize, 1usize),
        &device,
    )?;
    let out = unpack_gathered_kv_for_attention(&packed, &[2, 1], 2, 2, 1, &device)?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let expected = vec![1.0, 2.0, 10.0, 20.0, 3.0, 0.0, 30.0, 0.0];
    if got != expected {
        anyhow::bail!("unexpected unpacked kv: expected={expected:?} got={got:?}");
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
#[test]
fn test_unpack_gathered_kv_for_attention_equal_length_fast_path() -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let packed = candle_core::Tensor::from_vec(
        vec![1f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        (4usize, 2usize, 1usize),
        &device,
    )?;
    let out = unpack_gathered_kv_for_attention(&packed, &[2, 2], 2, 2, 1, &device)?;
    let got = out.flatten_all()?.to_vec1::<f32>()?;
    let expected = vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
    if got != expected {
        anyhow::bail!("unexpected equal-length unpacked kv: expected={expected:?} got={got:?}");
    }
    Ok(())
}
