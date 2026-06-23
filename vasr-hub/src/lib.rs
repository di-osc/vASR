//! Model cache and download helpers for vASR.

use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};

pub mod modelscope;

/// Backward-compatible access to the ModelScope download module.
pub use modelscope::download;

static CACHE_DIR_OVERRIDE: OnceLock<RwLock<Option<PathBuf>>> = OnceLock::new();

/// Override the model cache directory for the current process.
pub fn set_cache_dir(dir: impl Into<PathBuf>) {
    let lock = CACHE_DIR_OVERRIDE.get_or_init(|| RwLock::new(None));
    if let Ok(mut value) = lock.write() {
        *value = Some(dir.into());
    }
}

/// Directory where vASR models are cached.
///
/// Respects the `VASR_MODEL_DIR` environment variable. When set, it is used
/// directly (no further path composition). Otherwise defaults to
/// `$HOME/.cache/vasr` (or `/tmp/.cache/vasr` when `$HOME` is unset).
pub fn cache_dir() -> PathBuf {
    if let Some(lock) = CACHE_DIR_OVERRIDE.get()
        && let Ok(value) = lock.read()
        && let Some(dir) = value.as_ref()
    {
        return dir.clone();
    }
    if let Ok(dir) = std::env::var("VASR_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
        .join(".cache")
        .join("vasr")
}

/// Backward-compatible name for the model cache directory.
pub fn modelscope_cache_dir() -> PathBuf {
    cache_dir()
}
