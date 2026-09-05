//! Compiles Turbo-Base64 from a checkout the user supplies; no source is vendored here.
//!
//! The per-file architecture flags mirror upstream's own `makefile` exactly, so the C
//! competitor is built the way its author builds it. Every translation unit ends up in a
//! single archive because `tb64ini` in the base unit references the vector kernels and the
//! vector kernels reference the base tables; splitting them across archives would make the
//! link order significant.

use std::path::{Path, PathBuf};

/// Where the Turbo-Base64 checkout is expected when `TB64_SRC` is unset.
const DEFAULT_SRC: &str = "../../target/tb64-src";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(tb64)");
    println!("cargo::rerun-if-env-changed=TB64_SRC");

    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    println!(
        "cargo::rerun-if-changed={}",
        manifest.join("src/ffi_floor.c").display()
    );

    let Some(src) = locate(&manifest) else {
        skip(
            "Turbo-Base64 source not found. Run \
             `git clone --depth 1 https://github.com/powturbo/Turbo-Base64 target/tb64-src` \
             or point TB64_SRC at an existing checkout",
        );
        return;
    };
    println!(
        "cargo::rerun-if-changed={}",
        src.join("turbob64.h").display()
    );

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    // Upstream builds `turbob64v128.c` twice on x86_64 (SSSE3 and AVX), which is why the
    // same file appears under two group names: each group gets its own object directory.
    let groups: &[(&str, &[&str], &[&str])] = match arch.as_str() {
        "x86_64" => &[
            ("v128", &["turbob64v128.c"], &["-mssse3"]),
            (
                "v128a",
                &["turbob64v128.c"],
                &["-march=corei7-avx", "-mtune=corei7-avx", "-mno-aes"],
            ),
            ("v256", &["turbob64v256.c"], &["-march=haswell"]),
            (
                "v512",
                &["turbob64v512.c"],
                &["-march=skylake-avx512", "-mavx512vbmi"],
            ),
        ],
        "aarch64" => &[("v128", &["turbob64v128.c"], &["-march=armv8-a"])],
        other => {
            skip(&format!(
                "Turbo-Base64 has no kernel set wired up for target arch `{other}`"
            ));
            return;
        }
    };

    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let mut objects = compile(&src, &out, "base", &["turbob64c.c", "turbob64d.c"], &[]);
    for (name, files, flags) in groups {
        objects.extend(compile(&src, &out, name, files, flags));
    }
    // The FFI floor control is ours, and is deliberately built with the same compiler and
    // flags as the base unit so its call costs what a tb64 call costs.
    objects.extend(compile(&manifest, &out, "floor", &["src/ffi_floor.c"], &[]));

    cc::Build::new().objects(objects).compile("tb64");
    println!("cargo::rustc-cfg=tb64");
}

/// `TB64_SRC`, else the default checkout path, else nothing.
fn locate(manifest: &Path) -> Option<PathBuf> {
    let candidate =
        std::env::var_os("TB64_SRC").map_or_else(|| manifest.join(DEFAULT_SRC), PathBuf::from);
    candidate.join("turbob64.h").is_file().then_some(candidate)
}

fn skip(reason: &str) {
    println!("cargo::warning={reason}; the tb64 benchmark candidates are disabled.");
}

fn compile(dir: &Path, out: &Path, name: &str, files: &[&str], flags: &[&str]) -> Vec<PathBuf> {
    let mut build = cc::Build::new();
    build
        .out_dir(out.join(name))
        .include(dir)
        .opt_level(3)
        .define("NDEBUG", None)
        .flag("-fstrict-aliasing")
        .warnings(false)
        .cargo_metadata(false)
        .files(files.iter().map(|f| dir.join(f)));

    // Upstream adds this for gcc only; clang rejects it.
    if !build.get_compiler().is_like_clang() {
        build.flag("-falign-loops");
    }
    for flag in flags {
        build.flag(flag);
    }
    build.compile_intermediates()
}
