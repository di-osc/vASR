#[cfg(feature = "cuda")]
mod cuda_build {
    use anyhow::Context;
    use cudaforge::KernelBuilder;
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    const CUTLASS_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";
    const CUTLASS_COMMIT_ENV: &str = "VASR_CUTLASS_COMMIT";

    const KERNEL_FILES: &[&str] = &[
        "kernels/flash_api.cu",
        "kernels/flash_fwd_hdim64_bf16_sm80.cu",
        "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
    ];

    fn header_hash(dir: &Path) -> anyhow::Result<u64> {
        fn update(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= u64::from(*byte);
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }

        fn visit(path: &Path, hash: &mut u64) -> anyhow::Result<()> {
            if path.is_dir() {
                let mut entries = fs::read_dir(path)?
                    .map(|entry| entry.map(|entry| entry.path()))
                    .collect::<std::io::Result<Vec<_>>>()?;
                entries.sort();
                for entry in entries {
                    visit(&entry, hash)?;
                }
                return Ok(());
            }

            let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
                return Ok(());
            };
            if !matches!(ext, "cuh" | "h") {
                return Ok(());
            }

            println!("cargo:rerun-if-changed={}", path.display());
            update(hash, path.to_string_lossy().as_bytes());
            update(hash, &fs::read(path)?);
            Ok(())
        }

        let mut hash = 0xcbf29ce484222325;
        visit(dir, &mut hash)?;
        Ok(hash)
    }

    pub fn build() -> anyhow::Result<()> {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=VASR_FLASH_ATTN_BUILD_DIR");
        println!("cargo:rerun-if-env-changed={CUTLASS_COMMIT_ENV}");

        for kernel_file in KERNEL_FILES {
            println!("cargo:rerun-if-changed={kernel_file}");
        }

        let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
        let build_dir = std::env::var("VASR_FLASH_ATTN_BUILD_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| out_dir.clone());
        fs::create_dir_all(&build_dir)?;
        let build_dir = build_dir.canonicalize().with_context(|| {
            format!(
                "failed to canonicalize VASR_FLASH_ATTN_BUILD_DIR {}",
                build_dir.display()
            )
        })?;

        let header_hash_arg = format!(
            "-DVASR_FLASH_ATTN_HEADER_HASH=0x{:016x}",
            header_hash(Path::new("kernels"))?
        );
        let cutlass_commit =
            std::env::var(CUTLASS_COMMIT_ENV).unwrap_or_else(|_| CUTLASS_COMMIT.to_string());
        let builder = KernelBuilder::new()
            .source_files(KERNEL_FILES)
            .out_dir(&build_dir)
            .with_cutlass(Some(&cutlass_commit))
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-U__CUDA_NO_HALF_OPERATORS__")
            .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
            .arg("-U__CUDA_NO_HALF2_OPERATORS__")
            .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
            .arg("--expt-relaxed-constexpr")
            .arg("--expt-extended-lambda")
            .arg("--use_fast_math")
            .arg("--verbose")
            .arg(&header_hash_arg)
            .thread_percentage(0.5);

        let target = std::env::var("TARGET").unwrap_or_default();
        let is_msvc = target.contains("msvc");
        let out_file = if is_msvc {
            build_dir.join("vasrflashattention.lib")
        } else {
            build_dir.join("libvasrflashattention.a")
        };
        builder.build_lib(out_file)?;

        println!("cargo:rustc-link-search={}", build_dir.display());
        println!("cargo:rustc-link-lib=vasrflashattention");
        println!("cargo:rustc-link-lib=dylib=cudart");
        if !is_msvc {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    cuda_build::build()?;
    Ok(())
}
