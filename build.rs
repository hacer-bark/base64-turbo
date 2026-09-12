//! Resolves the backend selection into `cfg` aliases, so the source never has
//! to repeat the target/override matrix at every gate.

fn main() {
    for alias in [
        "b64_avx2",
        "b64_avx512",
        "b64_neon",
        "x86_simd",
        "unsafe_simd",
        "arithmetic_simd",
    ] {
        println!("cargo::rustc-check-cfg=cfg({alias})");
    }
    println!(
        "cargo::rustc-check-cfg=cfg(base64_turbo_backend, \
         values(\"soft\", \"avx2\", \"avx512\", \"neon\"))"
    );

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let x86 = matches!(arch.as_str(), "x86" | "x86_64");
    let aarch64 = arch == "aarch64";

    // Absent from the environment unless the user passed the `--cfg`.
    let backend = std::env::var("CARGO_CFG_BASE64_TURBO_BACKEND").ok();

    let (avx2, avx512, neon) = match backend.as_deref() {
        None => (x86, x86, aarch64),
        Some("soft") => (false, false, false),
        Some("avx2") => (x86, false, false),
        Some("avx512") => (false, x86, false),
        Some("neon") => (false, false, aarch64),
        Some(other) => {
            println!(
                "cargo::error=unknown base64_turbo_backend {other:?}; \
                 expected \"soft\", \"avx2\", \"avx512\" or \"neon\""
            );
            return;
        }
    };

    for (enabled, alias) in [
        (avx2, "b64_avx2"),
        (avx512, "b64_avx512"),
        (neon, "b64_neon"),
        (avx2 || avx512, "x86_simd"),
        (avx2 || avx512 || neon, "unsafe_simd"),
        (avx2 || neon, "arithmetic_simd"),
    ] {
        if enabled {
            println!("cargo::rustc-cfg={alias}");
        }
    }
}
