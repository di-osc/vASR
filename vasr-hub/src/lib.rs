//! Model cache and download helpers for vASR.

pub mod modelscope;

/// Backward-compatible access to the ModelScope download module.
pub use modelscope::download;

/// Directory where vASR models are cached.
///
/// Respects the `VASR_MODEL_DIR` environment variable. When set, it is used
/// directly (no further path composition). Otherwise defaults to
/// `$HOME/.cache/vasr` (or `/tmp/.cache/vasr` when `$HOME` is unset).
pub fn cache_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("VASR_MODEL_DIR") {
        return std::path::PathBuf::from(dir);
    }
    std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
        .join(".cache")
        .join("vasr")
}

/// Backward-compatible name for the model cache directory.
pub fn modelscope_cache_dir() -> std::path::PathBuf {
    cache_dir()
}
