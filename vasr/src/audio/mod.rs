//! Audio input helpers used by Qwen3-ASR inference.

pub mod chunking;
pub mod input;
pub mod loader;
pub mod normalize;
pub mod stream;

pub use vasr_data::audio::{
    AudioChannel, AudioChunk, AudioEncoding, AudioError, AudioFormat, AudioInfo, AudioSource,
    Waveform, decode,
};

pub use input::AudioInput;
pub use loader::{AudioLoadOptions, AudioLoader, SAMPLE_RATE_HZ};
pub use normalize::normalize_audio_input;
pub use stream::{AudioBytesStream, AudioChunkList};
