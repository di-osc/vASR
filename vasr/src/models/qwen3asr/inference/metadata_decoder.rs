//! Per-audio streaming results and completion ownership.
#[cfg(test)]
use super::ChunkItem;
use super::{
    AsrProcessor, AsrTranscription, EncodedAsrBatchMetadata, SAMPLE_RATE_HZ, merge_languages,
    parse_asr_output,
};
use anyhow::{Result, bail};

pub struct StreamingMetadataDecoder {
    merged: EncodedAsrBatchMetadata,
    batch_lens: Vec<usize>,
    audio_progress: Vec<AudioDecodeProgress>,
    completed_chunks: Vec<bool>,
}

struct DecodedChunk {
    index: usize,
    language: String,
    text: String,
}

#[derive(Default)]
struct AudioDecodeProgress {
    chunks: Vec<DecodedChunk>,
    expected_chunks: usize,
    emitted: bool,
}

impl AudioDecodeProgress {
    fn transcription(&self) -> AsrTranscription {
        let mut ordered: Vec<_> = self.chunks.iter().collect();
        ordered.sort_unstable_by_key(|chunk| chunk.index);
        AsrTranscription {
            language: merge_languages(
                ordered
                    .iter()
                    .map(|chunk| chunk.language.clone())
                    .collect::<Vec<_>>(),
            ),
            text: ordered
                .into_iter()
                .map(|chunk| chunk.text.as_str())
                .collect(),
            timestamps: None,
        }
    }
}

impl StreamingMetadataDecoder {
    pub fn new(metas: Vec<EncodedAsrBatchMetadata>, batch_lens: Vec<usize>) -> Self {
        let mut decoder = Self {
            merged: EncodedAsrBatchMetadata {
                audio_len: 0,
                chunks: Vec::new(),
                chunk_audio_seconds: Vec::new(),
                forced_langs_norm: Vec::new(),
            },
            batch_lens: Vec::new(),
            audio_progress: Vec::new(),
            completed_chunks: Vec::new(),
        };
        for (meta, len) in metas.into_iter().zip(batch_lens) {
            decoder.push_metadata(meta, len);
        }
        decoder
    }

    pub fn push_metadata(&mut self, mut meta: EncodedAsrBatchMetadata, batch_len: usize) {
        let audio_offset = self.merged.audio_len;
        for chunk in &mut meta.chunks {
            chunk.orig_index = chunk.orig_index.saturating_add(audio_offset);
        }
        self.merged.audio_len = self.merged.audio_len.saturating_add(meta.audio_len);
        self.audio_progress
            .resize_with(self.merged.audio_len, AudioDecodeProgress::default);
        for chunk in &meta.chunks {
            if let Some(audio) = self.audio_progress.get_mut(chunk.orig_index) {
                audio.expected_chunks = audio.expected_chunks.saturating_add(1);
            }
        }
        self.completed_chunks
            .extend(std::iter::repeat_n(false, meta.chunks.len()));
        self.merged.chunks.extend(meta.chunks);
        self.merged
            .chunk_audio_seconds
            .extend(meta.chunk_audio_seconds);
        self.merged.forced_langs_norm.extend(meta.forced_langs_norm);
        self.batch_lens.push(batch_len);
    }

    pub fn accept_chunk(
        &mut self,
        processor: &AsrProcessor,
        chunk_index: usize,
        ids: Vec<u32>,
    ) -> Result<Option<(usize, usize, AsrTranscription)>> {
        let is_completed = self.completed_chunks.get(chunk_index).ok_or_else(|| {
            anyhow::anyhow!("dynamic staged ASR completion index out of range: {chunk_index}")
        })?;
        if *is_completed {
            bail!("dynamic staged ASR chunk completed more than once: {chunk_index}");
        }
        let raw = processor.tokenizer.decode(ids.as_slice())?;
        // 只在显式打开时记下原文，用来核对特别长的生成。
        if std::env::var_os("VASR_DEBUG_RAW_ASR").is_some() {
            let audio_seconds =
                self.merged.chunks[chunk_index].wav.len() as f64 / f64::from(SAMPLE_RATE_HZ);
            tracing::debug!(
                target: "vasr::inference::transcribe",
                tokens = ids.len(),
                audio_seconds,
                raw = raw.as_str(),
                "raw ASR decode"
            );
        }
        let audio_index = self.merged.chunks[chunk_index].orig_index;
        let forced = self
            .merged
            .forced_langs_norm
            .get(audio_index)
            .and_then(|x| x.as_deref());
        let (lang, text) = parse_asr_output(Some(raw.as_str()), forced);
        self.accept_decoded_chunk(chunk_index, lang, text)
    }

    fn accept_decoded_chunk(
        &mut self,
        chunk_index: usize,
        lang: String,
        text: String,
    ) -> Result<Option<(usize, usize, AsrTranscription)>> {
        let audio_index = self
            .merged
            .chunks
            .get(chunk_index)
            .map(|chunk| chunk.orig_index)
            .ok_or_else(|| {
                anyhow::anyhow!("dynamic staged ASR chunk index out of range: {chunk_index}")
            })?;
        if *self.completed_chunks.get(chunk_index).ok_or_else(|| {
            anyhow::anyhow!("dynamic staged ASR completion index out of range: {chunk_index}")
        })? {
            bail!("dynamic staged ASR chunk completed more than once: {chunk_index}");
        }
        // Tokenization and optional raw-output logging have consumed the only
        // remaining waveform use. Drop its allocation even if sibling chunks
        // or earlier requests are still decoding.
        self.merged.chunks[chunk_index].wav = Vec::new();
        self.completed_chunks[chunk_index] = true;
        let audio = &mut self.audio_progress[audio_index];
        audio.chunks.push(DecodedChunk {
            index: chunk_index,
            language: lang,
            text,
        });
        if audio.emitted || audio.chunks.len() < audio.expected_chunks {
            return Ok(None);
        }
        audio.emitted = true;
        let transcription = audio.transcription();
        let (batch_index, batch_audio_index) = self.batch_audio_index(audio_index)?;
        Ok(Some((batch_index, batch_audio_index, transcription)))
    }

    pub fn chunk_count(&self) -> usize {
        self.merged.chunks.len()
    }

    pub fn all_chunks_completed(&self) -> bool {
        self.audio_progress
            .iter()
            .all(|audio| audio.chunks.len() >= audio.expected_chunks)
    }

    fn batch_audio_index(&self, audio_index: usize) -> Result<(usize, usize)> {
        let mut cursor = 0usize;
        for (batch_index, &len) in self.batch_lens.iter().enumerate() {
            let end = cursor.saturating_add(len);
            if audio_index < end {
                return Ok((batch_index, audio_index - cursor));
            }
            cursor = end;
        }
        bail!("dynamic staged ASR audio index out of range: {audio_index}")
    }

    pub fn finish(self) -> Result<Vec<Vec<AsrTranscription>>> {
        let mut outputs = Vec::with_capacity(self.merged.audio_len);
        for (audio_index, audio) in self.audio_progress.iter().enumerate() {
            if audio.chunks.len() != audio.expected_chunks {
                bail!(
                    "dynamic staged ASR incomplete audio output: audio={} chunks={}/{}",
                    audio_index,
                    audio.chunks.len(),
                    audio.expected_chunks
                );
            }
            outputs.push(audio.transcription());
        }
        let mut grouped = Vec::with_capacity(self.batch_lens.len());
        let mut cursor = 0usize;
        for len in self.batch_lens {
            let end = cursor.saturating_add(len);
            if end > outputs.len() {
                bail!(
                    "dynamic staged ASR batch split mismatch: end={} outputs={}",
                    end,
                    outputs.len()
                );
            }
            grouped.push(outputs[cursor..end].to_vec());
            cursor = end;
        }
        if cursor != outputs.len() {
            bail!(
                "dynamic staged ASR unused outputs: used={} outputs={}",
                cursor,
                outputs.len()
            );
        }
        Ok(grouped)
    }
}

#[cfg(test)]
mod streaming_metadata_order_tests {
    use super::{AudioDecodeProgress, DecodedChunk};

    fn metadata(audios: &[usize]) -> super::EncodedAsrBatchMetadata {
        let chunks = audios
            .iter()
            .enumerate()
            .flat_map(|(audio, &count)| {
                (0..count).map(move |_| super::ChunkItem {
                    orig_index: audio,
                    wav: vec![0.25; 1600],
                    offset_sec: 0.0,
                })
            })
            .collect::<Vec<_>>();
        super::EncodedAsrBatchMetadata {
            audio_len: audios.len(),
            chunk_audio_seconds: vec![0.1; chunks.len()],
            chunks,
            forced_langs_norm: vec![None; audios.len()],
        }
    }

    #[test]
    fn completed_chunks_release_waveforms_and_keep_finish_text_order() -> anyhow::Result<()> {
        let mut decoder = super::StreamingMetadataDecoder::new(vec![metadata(&[3])], vec![1]);
        assert!(
            decoder
                .accept_decoded_chunk(2, "Chinese".into(), "third".into())?
                .is_none()
        );
        assert_eq!(decoder.merged.chunks[2].wav.capacity(), 0);
        assert!(!decoder.merged.chunks[0].wav.is_empty());
        assert!(
            decoder
                .accept_decoded_chunk(2, "Chinese".into(), "duplicate".into())
                .is_err()
        );
        decoder.accept_decoded_chunk(0, "Chinese".into(), "first".into())?;
        let (_, _, output) = decoder
            .accept_decoded_chunk(1, "Chinese".into(), "second".into())?
            .expect("all chunks completed");
        assert_eq!(output.text, "firstsecondthird");
        assert!(
            decoder
                .merged
                .chunks
                .iter()
                .all(|chunk| chunk.wav.capacity() == 0)
        );
        assert_eq!(decoder.finish()?[0][0].text, "firstsecondthird");
        Ok(())
    }

    #[test]
    fn metadata_append_keeps_cross_batch_mapping_and_early_results() -> anyhow::Result<()> {
        let mut decoder = super::StreamingMetadataDecoder::new(vec![metadata(&[2, 1])], vec![2]);
        let early = decoder
            .accept_decoded_chunk(2, "Chinese".into(), "b".into())?
            .unwrap();
        assert_eq!((early.0, early.1), (0, 1));
        assert_eq!(early.2.text, "b");
        decoder.push_metadata(metadata(&[1, 2]), 2);
        assert!(
            decoder
                .accept_decoded_chunk(5, "Chinese".into(), "d2".into())?
                .is_none()
        );
        assert!(
            decoder
                .accept_decoded_chunk(0, "Chinese".into(), "a1".into())?
                .is_none()
        );
        let middle = decoder
            .accept_decoded_chunk(3, "Chinese".into(), "c".into())?
            .unwrap();
        assert_eq!((middle.0, middle.1), (1, 0));
        assert_eq!(middle.2.text, "c");
        decoder.accept_decoded_chunk(1, "Chinese".into(), "a2".into())?;
        decoder.accept_decoded_chunk(4, "Chinese".into(), "d1".into())?;
        assert!(decoder.all_chunks_completed());
        let outputs = decoder.finish()?;
        assert_eq!(
            outputs
                .iter()
                .map(|batch| batch
                    .iter()
                    .map(|audio| audio.text.as_str())
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            vec![vec!["a1a2", "b"], vec!["c", "d1d2"]]
        );
        Ok(())
    }
    #[test]
    fn metadata_finish_checks_missing_chunks_and_empty_audio() -> anyhow::Result<()> {
        let incomplete = super::StreamingMetadataDecoder::new(vec![metadata(&[2])], vec![1]);
        assert!(incomplete.finish().is_err());
        let empty = super::StreamingMetadataDecoder::new(vec![metadata(&[0])], vec![1]);
        assert!(empty.all_chunks_completed());
        assert_eq!(empty.finish()?[0][0].text, "");
        Ok(())
    }

    #[test]
    fn chunk_texts_are_joined_in_audio_order_not_completion_order() {
        let completed = vec![
            (2usize, "third".to_string()),
            (0usize, "first".to_string()),
            (1usize, "second".to_string()),
        ];

        assert_eq!(
            AudioDecodeProgress {
                chunks: completed
                    .into_iter()
                    .map(|(index, text)| DecodedChunk {
                        index,
                        language: "Chinese".into(),
                        text
                    })
                    .collect(),
                expected_chunks: 3,
                emitted: false
            }
            .transcription()
            .text,
            "firstsecondthird"
        );
    }
}
