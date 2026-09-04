//! Emits three convenience `cfg` aliases so the source never has to repeat the
//! SIMD feature matrix at every gate:
//!
//! * `unsafe_simd` — at least one kernel that uses `unsafe` is compiled in
//!   (any x86 AVX kernel, or NEON on aarch64). When it is absent the crate is
//!   pure safe scalar Rust and carries `#![forbid(unsafe_code)]`.
//! * `x86_simd` — at least one x86 AVX kernel is compiled in, i.e. runtime CPU
//!   detection is needed.
//! * `arithmetic_simd` — at least one kernel that derives characters
//!   arithmetically from the RFC 4648 layout (AVX2 or NEON) is compiled in.
//!   Those two cannot serve a custom `Alphabet`; the scalar and AVX-512 VBMI
//!   kernels, being pure table lookups, can.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(unsafe_simd)");
    println!("cargo::rustc-check-cfg=cfg(x86_simd)");
    println!("cargo::rustc-check-cfg=cfg(arithmetic_simd)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let feat = |name: &str| std::env::var_os(name).is_some();

    let x86 = matches!(arch.as_str(), "x86" | "x86_64");
    let x86_simd = x86 && (feat("CARGO_FEATURE_AVX2") || feat("CARGO_FEATURE_AVX512_VBMI"));
    let neon = arch == "aarch64" && feat("CARGO_FEATURE_NEON");
    let unsafe_simd = x86_simd || neon;
    let arithmetic_simd = (x86 && feat("CARGO_FEATURE_AVX2")) || neon;

    if x86_simd {
        println!("cargo::rustc-cfg=x86_simd");
    }
    if unsafe_simd {
        println!("cargo::rustc-cfg=unsafe_simd");
    }
    if arithmetic_simd {
        println!("cargo::rustc-cfg=arithmetic_simd");
    }
}
