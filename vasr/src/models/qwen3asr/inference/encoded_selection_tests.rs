use super::*;
use crate::processor::feat_lengths::feat_extract_output_length;

fn fixture() -> Result<EncodedAsrBatch> {
    let audios = [0, 1, 0, 2, 1];
    let frames = [1, 9, 100, 3, 17];
    let mut groups = Vec::new();
    for indices in [vec![3, 0, 4], vec![2, 1]] {
        let mut prepared = Vec::new();
        let mut values = Vec::new();
        for &index in &indices {
            let mut mask = vec![1; frames[index]];
            mask.extend([0, 0, 0]);
            prepared.push(PreparedInputs {
                input_ids: vec![index as u32],
                attention_mask: vec![1],
                input_features: vec![vec![index as f32; frames[index] + 3]],
                feature_attention_mask: mask,
            });
            for row in 0..feat_extract_output_length(frames[index]) {
                values.extend([
                    index as f32 * 100. + row as f32 + 0.25,
                    index as f32 * 100. + row as f32 + 0.75,
                ]);
            }
        }
        groups.push(EncodedPreparedGroup {
            chunk_indices: indices,
            prepared,
            audio_features_cpu: Tensor::from_vec(
                values.clone(),
                (values.len() / 2, 2),
                &Device::Cpu,
            )?,
        });
    }
    Ok(EncodedAsrBatch {
        audio_len: 3,
        chunks: audios
            .into_iter()
            .enumerate()
            .map(|(index, orig_index)| ChunkItem {
                orig_index,
                wav: vec![index as f32 + 1.; 16 * (index + 1)],
                offset_sec: index as f32,
            })
            .collect(),
        chunk_audio_seconds: vec![1., 2., 3., 4., 5.],
        forced_langs_norm: vec![Some("English".into()), None, Some("Japanese".into())],
        groups,
        timings: AudioFeatureEncodeTimings {
            audio_tokens: 20,
            forward_us: 7,
            ..Default::default()
        },
    })
}
#[test]
fn selecting_interleaved_audios_preserves_exact_feature_rows_and_remaps_all_metadata() -> Result<()>
{
    let selected = fixture()?.select_audios(&[2, 0])?;
    assert_eq!(selected.audio_len, 2);
    assert_eq!(
        selected.forced_langs_norm,
        [Some("Japanese".into()), Some("English".into())]
    );
    assert_eq!(
        selected
            .chunks
            .iter()
            .map(|chunk| chunk.orig_index)
            .collect::<Vec<_>>(),
        [1, 1, 0]
    );
    assert_eq!(
        selected
            .chunks
            .iter()
            .map(|chunk| chunk.offset_sec)
            .collect::<Vec<_>>(),
        [0., 2., 3.]
    );
    assert_eq!(selected.chunk_audio_seconds, [1., 3., 4.]);
    assert_eq!(selected.groups[0].chunk_indices, [2, 0]);
    assert_eq!(selected.groups[1].chunk_indices, [1]);
    assert_eq!(
        selected.groups[0]
            .prepared
            .iter()
            .map(|prepared| prepared.input_ids[0])
            .collect::<Vec<_>>(),
        [3, 0]
    );
    assert_eq!(
        selected.groups[0].audio_features_cpu.to_vec2::<f32>()?,
        [[300.25, 300.75], [0.25, 0.75]]
    );
    assert_eq!(
        selected.groups[1].audio_features_cpu.to_vec2::<f32>()?,
        (0..13)
            .map(|row| vec![200.25 + row as f32, 200.75 + row as f32])
            .collect::<Vec<_>>()
    );
    assert_eq!(selected.timings.forward_us, 7);
    Ok(())
}
#[test]
fn retaining_all_audios_is_an_allocation_preserving_noop_and_none_drops_every_group() -> Result<()>
{
    let batch = fixture()?;
    let ids = batch
        .groups
        .iter()
        .map(|group| group.audio_features_cpu.id())
        .collect::<Vec<_>>();
    let chunk_ptr = batch.chunks.as_ptr();
    let batch = batch.retain_audios(&[true, true, true])?;
    assert_eq!(batch.chunks.as_ptr(), chunk_ptr);
    assert_eq!(
        batch
            .groups
            .iter()
            .map(|group| group.audio_features_cpu.id())
            .collect::<Vec<_>>(),
        ids
    );
    let none = batch.retain_audios(&[false, false, false])?;
    assert!(
        none.is_empty()
            && none.chunks.is_empty()
            && none.groups.is_empty()
            && none.forced_langs_norm.is_empty()
    );
    Ok(())
}
#[test]
fn selecting_audios_rejects_bad_masks_indices_and_misaligned_encoder_token_rows() -> Result<()> {
    assert!(fixture()?.retain_audios(&[true]).is_err());
    assert!(fixture()?.select_audios(&[3]).is_err());
    assert!(fixture()?.select_audios(&[1, 1]).is_err());
    let mut batch = fixture()?;
    batch.groups[0].audio_features_cpu =
        Tensor::zeros((1, 2), candle_core::DType::F32, &Device::Cpu)?;
    assert!(batch.select_audios(&[0]).is_err());
    let mut batch = fixture()?;
    batch.groups[0].prepared.pop();
    assert!(batch.select_audios(&[0]).is_err());
    Ok(())
}
#[test]
fn selecting_one_audio_drops_whole_cancelled_groups_without_changing_survivor_storage() -> Result<()>
{
    let batch = fixture()?.retain_audios(&[false, false, true])?;
    assert_eq!(batch.groups.len(), 1);
    assert_eq!(
        batch.groups[0].audio_features_cpu.to_vec2::<f32>()?,
        [[300.25, 300.75]]
    );
    Ok(())
}

#[cfg(feature = "paged-attn")]
#[test]
fn staged_conversion_keeps_each_feature_row_attached_to_its_original_chunk_metadata() -> Result<()>
{
    let (prepared, metadata) = fixture()?.into_prepared_audio_batch_and_metadata()?;
    let chunk_audio = [0, 1, 0, 2, 1];
    let produced_audio = prepared
        .prepared
        .iter()
        .map(|row| chunk_audio[row.input_ids[0] as usize])
        .collect::<Vec<_>>();
    assert_eq!(
        produced_audio,
        metadata
            .chunk_audio_metadata()
            .map(|(audio, _)| audio)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        prepared
            .prepared
            .iter()
            .map(|row| row.input_ids[0])
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4]
    );
    assert_eq!(prepared.audio_seconds, [1., 2., 3., 4., 5.]);
    let expected = fixture()?
        .groups
        .into_iter()
        .flat_map(|group| group.chunk_indices.into_iter().zip(group.prepared))
        .map(|(index, row)| {
            (
                index,
                feat_extract_output_length(
                    row.feature_attention_mask
                        .iter()
                        .filter(|&&v| v != 0)
                        .count(),
                ),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut cursor = 0;
    for chunk in 0..5 {
        let rows = expected[&chunk];
        let values = prepared
            .audio_features
            .narrow(0, cursor, rows)?
            .to_vec2::<f32>()?;
        assert_eq!(
            values,
            (0..rows)
                .map(|row| vec![
                    chunk as f32 * 100. + row as f32 + 0.25,
                    chunk as f32 * 100. + row as f32 + 0.75
                ])
                .collect::<Vec<_>>()
        );
        cursor += rows;
    }
    Ok(())
}

#[cfg(feature = "paged-attn")]
#[test]
fn selected_staged_conversion_preserves_original_chunk_order_after_index_remapping() -> Result<()> {
    let (prepared, metadata) = fixture()?
        .select_audios(&[2, 0])?
        .into_prepared_audio_batch_and_metadata()?;
    assert_eq!(
        prepared
            .prepared
            .iter()
            .map(|row| row.input_ids[0])
            .collect::<Vec<_>>(),
        [0, 2, 3]
    );
    assert_eq!(
        metadata.chunk_audio_metadata().collect::<Vec<_>>(),
        [(1, 1.), (1, 3.), (0, 4.)]
    );
    assert_eq!(prepared.audio_seconds, [1., 3., 4.]);
    let values = prepared.audio_features.to_vec2::<f32>()?;
    assert_eq!(values[0], [0.25, 0.75]);
    assert_eq!(
        values[1..14],
        (0..13)
            .map(|row| vec![200.25 + row as f32, 200.75 + row as f32])
            .collect::<Vec<_>>()
    );
    assert_eq!(values[14], [300.25, 300.75]);
    Ok(())
}
