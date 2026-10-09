//! Narrow auto policy for the hardware covered by the throughput report.
use candle_core::Device;
use objc2_metal::MTLDevice;

/// Bound temporary Objective-C ownership during a Metal forward. Disabled callers
/// and other devices keep the ordinary path; returned tensors own retained buffers.
pub fn with_autorelease_scope<T>(
    enabled: bool,
    device: &Device,
    operation: impl FnOnce() -> T,
) -> T {
    if enabled && device.is_metal() {
        objc2::rc::autoreleasepool(|_| operation())
    } else {
        operation()
    }
}

fn validated_hardware(name: &str, bytes: u64) -> bool {
    name == "Apple M4 Pro" && bytes == 48u64 << 30
}

pub fn validated_m4_pro_48gb(device: &Device) -> bool {
    let Ok(metal) = device.as_metal_device() else {
        return false;
    };
    let name = metal.device().as_ref().name().to_string();
    let memory = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|out| out.trim().parse::<u64>().ok())
        .unwrap_or(0);
    validated_hardware(&name, memory)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_measured_hardware_enables_auto_policy() {
        assert!(validated_hardware("Apple M4 Pro", 48 << 30));
        for (name, mem) in [
            ("Apple M4", 48 << 30),
            ("Apple M4 Pro", 24 << 30),
            ("Apple M4 Max", 48 << 30),
            ("Apple M4 Pro", 0),
        ] {
            assert!(!validated_hardware(name, mem));
        }
    }

    #[test]
    #[ignore = "exclusive Metal Objective-C lifetime check"]
    fn autorelease_scope_reclaims_temporary_ownership() -> anyhow::Result<()> {
        use objc2::rc::{Retained, autoreleasepool};
        use objc2::runtime::NSObject;
        let device = Device::new_metal(0)?;
        autoreleasepool(|_| {
            let object = NSObject::new();
            // Only test the known ownership of this fresh object, never a Metal
            // resource's implementation-specific retain count.
            let before: usize = unsafe { objc2::msg_send![&*object, retainCount] };
            with_autorelease_scope(true, &device, || {
                let _pointer = Retained::autorelease_ptr(object.clone());
                let during: usize = unsafe { objc2::msg_send![&*object, retainCount] };
                assert_eq!(during, before + 1);
            });
            let after: usize = unsafe { objc2::msg_send![&*object, retainCount] };
            assert_eq!(
                after, before,
                "the forward scope must release its temporary owner"
            );
        });
        Ok(())
    }

    #[test]
    #[ignore = "exclusive Metal returned-buffer lifetime check"]
    fn autorelease_scope_keeps_returned_tensors_alive() -> anyhow::Result<()> {
        use candle_core::{DType, Tensor};
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let input = Tensor::from_vec(
                (0..9 * 17)
                    .map(|i| (i as f32 - 71.) / 113.)
                    .collect::<Vec<_>>(),
                (9, 17),
                &device,
            )?
            .to_dtype(dtype)?
            .narrow(0, 1, 7)?;
            let expected = input.affine(0.75, 0.1)?;
            let actual = with_autorelease_scope(true, &device, || input.affine(0.75, 0.1))?;
            // Completion occurs after the pool has drained, as in an async forward.
            device.synchronize()?;
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(actual.dims(), expected.dims());
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            );
        }
        Ok(())
    }
}
