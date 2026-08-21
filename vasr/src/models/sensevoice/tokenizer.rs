use std::path::Path;

use anyhow::{Context, Result};

pub struct SenseVoiceTokenizer {
    pieces: Vec<String>,
}

impl SenseVoiceTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read SenseVoice tokens {}", path.display()))?;
        let pieces: Vec<String> = serde_json::from_str(&text)
            .with_context(|| format!("failed to parse SenseVoice tokens {}", path.display()))?;
        Ok(Self { pieces })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn decode_piece(&self, id: u32) -> &str {
        self.pieces
            .get(id as usize)
            .map(String::as_str)
            .unwrap_or("")
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut text = String::new();
        let mut bytes = Vec::new();
        for &id in ids {
            let piece = self.decode_piece(id);
            if let Some(hex) = piece
                .strip_prefix("<0x")
                .and_then(|value| value.strip_suffix('>'))
                && let Ok(byte) = u8::from_str_radix(hex, 16)
            {
                bytes.push(byte);
                continue;
            }
            flush_bytes(&mut text, &mut bytes);
            if piece.starts_with('<') && piece.ends_with('>') {
                continue;
            }
            text.push_str(&piece.replace('▁', " "));
        }
        flush_bytes(&mut text, &mut bytes);
        text.trim().to_string()
    }
}

fn flush_bytes(text: &mut String, bytes: &mut Vec<u8>) {
    if bytes.is_empty() {
        return;
    }
    text.push_str(&String::from_utf8_lossy(bytes));
    bytes.clear();
}

#[cfg(test)]
mod tests {
    use super::SenseVoiceTokenizer;

    #[test]
    fn sentencepiece_markers_become_spaces() {
        let tokenizer = SenseVoiceTokenizer {
            pieces: vec!["<unk>".into(), "▁hello".into(), "世界".into()],
        };
        assert_eq!(tokenizer.decode(&[1, 2]), "hello世界");
    }
}
