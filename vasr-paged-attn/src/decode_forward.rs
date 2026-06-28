//! Model forward hook used by CUDA decode graph capture/replay.

use anyhow::Result;
use candle_core::Tensor;

use crate::{PagedInputMetadata, PagedKvCache};

pub trait PagedCudaDecodeForward {
    fn forward_input_ids_with_paged_cache(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        paged_cache: &PagedKvCache,
        input_metadata: &PagedInputMetadata,
    ) -> Result<Tensor>;

    fn forward_next_ids_with_paged_cache(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        paged_cache: &PagedKvCache,
        input_metadata: &PagedInputMetadata,
    ) -> Result<Tensor> {
        let logits = self.forward_input_ids_with_paged_cache(
            input_ids,
            position_ids,
            paged_cache,
            input_metadata,
        )?;
        Ok(logits.squeeze(1)?.argmax(1usize)?)
    }

    fn forward_next_id_candidates_with_paged_cache(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        paged_cache: &PagedKvCache,
        input_metadata: &PagedInputMetadata,
    ) -> Result<Option<Tensor>> {
        let _ = (input_ids, position_ids, paged_cache, input_metadata);
        Ok(None)
    }
}
