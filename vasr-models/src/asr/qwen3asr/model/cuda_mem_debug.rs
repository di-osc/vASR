use candle_core::Device;

#[cfg(feature = "cuda")]
static LAST_LOGGED_CUDA_USED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(u64::MAX);

#[cfg(feature = "cuda")]
pub(crate) fn log_cuda_mem(device: &Device, stage: &str) {
    if std::env::var_os("VASR_DEBUG_CUDA_MEM").is_none() {
        return;
    }
    let Some(cuda) = device.as_cuda_device().ok() else {
        return;
    };
    if std::env::var_os("VASR_DEBUG_CUDA_MEM_SYNC").is_some() {
        let _ = device.synchronize();
    }
    if let Ok((free, total)) = cuda.cuda_stream().context().mem_get_info() {
        let used = total.saturating_sub(free);
        let used_u64 = used.min(u64::MAX as usize) as u64;
        let previous = LAST_LOGGED_CUDA_USED.swap(used_u64, std::sync::atomic::Ordering::Relaxed);
        let verbose = std::env::var_os("VASR_DEBUG_CUDA_MEM_VERBOSE").is_some();
        if !verbose && previous == used_u64 {
            return;
        }
        let delta_mib = if previous == u64::MAX {
            0.0
        } else {
            (used_u64 as i128 - previous as i128) as f64 / (1024.0 * 1024.0)
        };
        tracing::info!(
            target: "vasr_models::cuda_mem",
            "cuda_mem | stage={} | used={:.2}MiB | delta={:+.2}MiB | free={:.2}MiB | total={:.2}MiB",
            stage,
            used as f64 / (1024.0 * 1024.0),
            delta_mib,
            free as f64 / (1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0),
        );
    }
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn log_cuda_mem(_device: &Device, _stage: &str) {}
