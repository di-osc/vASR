//! In-situ quantization (ISQ) and decode matmul fast paths for VASR.

pub mod isq_linear;
#[cfg(feature = "cuda")]
mod q8_mmvq;
#[cfg(feature = "cuda")]
mod rotary;

#[cfg(feature = "cuda")]
pub use q8_mmvq::FastPathStats as Q8FastPathStats;
#[cfg(feature = "cuda")]
pub use rotary::apply_rotary_qk;

#[cfg(not(feature = "cuda"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Q8FastPathStats {
    pub plain_prefill_calls: u64,
    pub plain_decode_calls: u64,
    pub plain_prefill_rows: u64,
    pub plain_decode_rows: u64,
    pub plain_audio_calls: u64,
    pub plain_audio_rows: u64,
    pub plain_lm_head_calls: u64,
    pub plain_lm_head_rows: u64,
    pub mmq_prefill_calls: u64,
    pub mmq_prefill_rows: u64,
    pub mmq_decode_calls: u64,
    pub mmq_decode_rows: u64,
    pub mmq_lm_head_calls: u64,
    pub mmq_lm_head_rows: u64,
    pub prefill_lm_head_mmq_skip_calls: u64,
    pub prefill_lm_head_mmq_skip_rows: u64,
    pub prefill_next_id_calls: u64,
    pub prefill_next_id_rows: u64,
    pub decode_next_id_calls: u64,
    pub decode_next_id_rows: u64,
    pub fused_glu_decode_calls: u64,
    pub fused_glu_decode_rows: u64,
    pub fused_act_calls: u64,
    pub fused_act_rows: u64,
    pub fused_qkv_calls: u64,
    pub fused_qkv_rows: u64,
    pub mmq_fused_qkv_calls: u64,
    pub mmq_fused_qkv_rows: u64,
}

#[cfg(not(feature = "cuda"))]
impl Q8FastPathStats {
    pub fn saturating_sub(self, baseline: Self) -> Self {
        Self {
            plain_prefill_calls: self
                .plain_prefill_calls
                .saturating_sub(baseline.plain_prefill_calls),
            plain_decode_calls: self
                .plain_decode_calls
                .saturating_sub(baseline.plain_decode_calls),
            plain_prefill_rows: self
                .plain_prefill_rows
                .saturating_sub(baseline.plain_prefill_rows),
            plain_decode_rows: self
                .plain_decode_rows
                .saturating_sub(baseline.plain_decode_rows),
            plain_audio_calls: self
                .plain_audio_calls
                .saturating_sub(baseline.plain_audio_calls),
            plain_audio_rows: self
                .plain_audio_rows
                .saturating_sub(baseline.plain_audio_rows),
            plain_lm_head_calls: self
                .plain_lm_head_calls
                .saturating_sub(baseline.plain_lm_head_calls),
            plain_lm_head_rows: self
                .plain_lm_head_rows
                .saturating_sub(baseline.plain_lm_head_rows),
            mmq_prefill_calls: self
                .mmq_prefill_calls
                .saturating_sub(baseline.mmq_prefill_calls),
            mmq_prefill_rows: self
                .mmq_prefill_rows
                .saturating_sub(baseline.mmq_prefill_rows),
            mmq_decode_calls: self
                .mmq_decode_calls
                .saturating_sub(baseline.mmq_decode_calls),
            mmq_decode_rows: self
                .mmq_decode_rows
                .saturating_sub(baseline.mmq_decode_rows),
            mmq_lm_head_calls: self
                .mmq_lm_head_calls
                .saturating_sub(baseline.mmq_lm_head_calls),
            mmq_lm_head_rows: self
                .mmq_lm_head_rows
                .saturating_sub(baseline.mmq_lm_head_rows),
            prefill_lm_head_mmq_skip_calls: self
                .prefill_lm_head_mmq_skip_calls
                .saturating_sub(baseline.prefill_lm_head_mmq_skip_calls),
            prefill_lm_head_mmq_skip_rows: self
                .prefill_lm_head_mmq_skip_rows
                .saturating_sub(baseline.prefill_lm_head_mmq_skip_rows),
            prefill_next_id_calls: self
                .prefill_next_id_calls
                .saturating_sub(baseline.prefill_next_id_calls),
            prefill_next_id_rows: self
                .prefill_next_id_rows
                .saturating_sub(baseline.prefill_next_id_rows),
            decode_next_id_calls: self
                .decode_next_id_calls
                .saturating_sub(baseline.decode_next_id_calls),
            decode_next_id_rows: self
                .decode_next_id_rows
                .saturating_sub(baseline.decode_next_id_rows),
            fused_glu_decode_calls: self
                .fused_glu_decode_calls
                .saturating_sub(baseline.fused_glu_decode_calls),
            fused_glu_decode_rows: self
                .fused_glu_decode_rows
                .saturating_sub(baseline.fused_glu_decode_rows),
            fused_act_calls: self
                .fused_act_calls
                .saturating_sub(baseline.fused_act_calls),
            fused_act_rows: self.fused_act_rows.saturating_sub(baseline.fused_act_rows),
            fused_qkv_calls: self
                .fused_qkv_calls
                .saturating_sub(baseline.fused_qkv_calls),
            fused_qkv_rows: self.fused_qkv_rows.saturating_sub(baseline.fused_qkv_rows),
            mmq_fused_qkv_calls: self
                .mmq_fused_qkv_calls
                .saturating_sub(baseline.mmq_fused_qkv_calls),
            mmq_fused_qkv_rows: self
                .mmq_fused_qkv_rows
                .saturating_sub(baseline.mmq_fused_qkv_rows),
        }
    }
}

#[cfg(feature = "cuda")]
pub fn trim_cuda_workspaces() {
    q8_mmvq::trim_cuda_workspace();
}

#[cfg(not(feature = "cuda"))]
pub fn trim_cuda_workspaces() {}

#[cfg(feature = "cuda")]
pub fn reset_q8_fast_path_stats() {
    q8_mmvq::reset_fast_path_stats();
}

#[cfg(not(feature = "cuda"))]
pub fn reset_q8_fast_path_stats() {}

#[cfg(feature = "cuda")]
pub fn q8_fast_path_stats() -> Q8FastPathStats {
    q8_mmvq::fast_path_stats()
}

#[cfg(not(feature = "cuda"))]
pub fn q8_fast_path_stats() -> Q8FastPathStats {
    Q8FastPathStats::default()
}

#[cfg(feature = "cuda")]
pub fn set_q8_prefill_lm_head_mmq_enabled(enabled: bool) {
    q8_mmvq::set_prefill_lm_head_mmq_enabled(enabled);
}

#[cfg(not(feature = "cuda"))]
pub fn set_q8_prefill_lm_head_mmq_enabled(_enabled: bool) {}

#[cfg(feature = "cuda")]
pub fn set_q8_prefill_lm_head_mmq_max_logits_bytes(bytes: usize) {
    q8_mmvq::set_prefill_lm_head_mmq_max_logits_bytes(bytes);
}

#[cfg(not(feature = "cuda"))]
pub fn set_q8_prefill_lm_head_mmq_max_logits_bytes(_bytes: usize) {}
