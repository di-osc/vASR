#[cfg(all(feature = "cuda", target_family = "unix"))]
fn cuda_header_hash(dir: &str) -> Result<u64, String> {
    use std::path::Path;

    fn update(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= u64::from(*byte);
            *hash = hash.wrapping_mul(0x100000001b3);
        }
    }

    fn visit(path: &Path, hash: &mut u64) -> Result<(), String> {
        if path.is_dir() {
            let mut entries = std::fs::read_dir(path)
                .map_err(|e| e.to_string())?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<Vec<_>>>()
                .map_err(|e| e.to_string())?;
            entries.sort();
            for entry in entries {
                visit(&entry, hash)?;
            }
            return Ok(());
        }

        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            return Ok(());
        };
        if ext != "cuh" && ext != "h" {
            return Ok(());
        }

        println!("cargo:rerun-if-changed={}", path.display());
        update(hash, path.to_string_lossy().as_bytes());
        update(hash, &std::fs::read(path).map_err(|e| e.to_string())?);
        Ok(())
    }

    let mut hash = 0xcbf29ce484222325;
    visit(Path::new(dir), &mut hash)?;
    Ok(hash)
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn compile_cuda() -> Result<(), String> {
    use std::path::PathBuf;

    const CUDA_NVCC_FLAGS: Option<&'static str> = option_env!("CUDA_NVCC_FLAGS");
    const CUDA_SOURCES: &[&str] = &[
        "src/cuda/gather_kv_cache_kernel.cu",
        "src/cuda/pagedattention_v1_bf16.cu",
        "src/cuda/pagedattention_v1_f16.cu",
        "src/cuda/pagedattention_v1_f32.cu",
        "src/cuda/pagedattention_v2_bf16.cu",
        "src/cuda/pagedattention_v2_f16.cu",
        "src/cuda/pagedattention_v2_f32.cu",
        "src/cuda/reshape_and_cache_kernel.cu",
    ];

    println!("cargo::rustc-check-cfg=cfg(has_fp8)");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
    println!("cargo:rerun-if-changed=src/cuda/pagedattention.cuh");
    for source in CUDA_SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }

    let header_hash_arg = format!(
        "-DVASR_CUDA_HEADER_HASH={:016x}",
        cuda_header_hash("src/cuda")?
    );

    let mut builder = cudaforge::KernelBuilder::new()
        .source_files(CUDA_SOURCES)
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
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg(&header_hash_arg);

    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    if compute_cap >= 80 {
        builder = builder.arg("-DENABLE_FP8");
        println!("cargo:rustc-cfg=has_fp8");
    }

    if let Some(cuda_nvcc_flags_env) = CUDA_NVCC_FLAGS {
        builder = builder.arg("--compiler-options");
        builder = builder.arg(cuda_nvcc_flags_env);
    }

    let target = std::env::var("TARGET").map_err(|e| e.to_string())?;
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").map_err(|e| e.to_string())?);
    let out_file = if target.contains("msvc") {
        build_dir.join("vasrpagedattention.lib")
    } else {
        build_dir.join("libvasrpagedattention.a")
    };
    builder
        .build_lib(out_file)
        .expect("Build paged attention lib failed!");

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=vasrpagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    Ok(())
}

#[cfg(feature = "metal")]
fn compile_metal() -> Result<(), String> {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/metal/kernels/prefill_paged_attn.metal");
    println!("cargo:rerun-if-env-changed=VASR_METAL_PRECOMPILE");

    let skip = env::var("VASR_METAL_PRECOMPILE")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(|_| "OUT_DIR not set")?);
    let lib_path = out_dir.join("vasr_varlen_prefill.metallib");

    if skip {
        std::fs::write(&lib_path, []).map_err(|e| e.to_string())?;
        return Ok(());
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(|_| "CARGO_MANIFEST_DIR")?);
    let source = manifest_dir.join("src/metal/kernels/prefill_paged_attn.metal");
    let working_directory = out_dir.to_string_lossy().to_string();
    let air_path = out_dir.join("prefill_paged_attn.air");

    let mut compile_air = Command::new("xcrun");
    compile_air
        .arg("--sdk")
        .arg("macosx")
        .arg("metal")
        .arg("-std=metal3.1")
        .arg(format!("-working-directory={working_directory}"))
        .arg("-O3")
        .arg("-c")
        .arg("-w")
        .arg(&source);
    let status = compile_air.status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!(
            "compiling varlen prefill metal -> air failed: {status}"
        ));
    }

    let mut compile_lib = Command::new("xcrun");
    compile_lib
        .arg("metal")
        .arg("-o")
        .arg(&lib_path)
        .arg(&air_path);
    let status = compile_lib.status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!(
            "compiling varlen prefill air -> metallib failed: {status}"
        ));
    }
    Ok(())
}

fn main() -> Result<(), String> {
    println!("cargo::rerun-if-changed=build.rs");
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    compile_cuda()?;
    #[cfg(feature = "metal")]
    compile_metal()?;
    Ok(())
}
