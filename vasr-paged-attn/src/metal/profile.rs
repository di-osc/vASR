//! Narrow auto policy for the hardware covered by the throughput report.
use candle_core::Device;
use objc2_metal::MTLDevice;

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
}
