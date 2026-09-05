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

    warn_on_inherited_cflags();

    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let groups: Vec<(Vec<PathBuf>, &[&str])> = match arch.as_str() {
        "x86_64" => vec![
            // `-mno-avx` is not upstream's flag, it is insurance: this unit is the only
            // one that defines `tb64ini`, `_tb64e`, `_tb64d`, `cpuini` and `cpustr`, and
            // it guards them with `#ifndef __AVX__`. Anything that leaks AVX into this
            // compile -- an inherited `CFLAGS=-march=native`, say -- makes all five
            // vanish and the bench fails to link.
            (vec![src.join("turbob64v128.c")], &["-mssse3", "-mno-avx"]),
            // Upstream compiles turbob64v128.c a second time as an AVX build; the file
            // renames its own symbols under `__AVX__`. It is copied to a distinct name
            // first because two archive members may not share one: the linker resolves
            // the symbol index by member name, and would silently pick the AVX object,
            // which omits everything guarded by `#ifndef __AVX__` — `tb64ini`, `_tb64e`,
            // `_tb64d`, `cpuini`, `cpustr`.
            (
                vec![copy_to(
                    &src.join("turbob64v128.c"),
                    &out.join("turbob64v128a.c"),
                )],
                &["-march=corei7-avx", "-mtune=corei7-avx", "-mno-aes"],
            ),
            (vec![src.join("turbob64v256.c")], &["-march=haswell"]),
            (
                vec![src.join("turbob64v512.c")],
                &["-march=skylake-avx512", "-mavx512vbmi"],
            ),
        ],
        "aarch64" => vec![(vec![src.join("turbob64v128.c")], &["-march=armv8-a"])],
        other => {
            skip(&format!(
                "Turbo-Base64 has no kernel set wired up for target arch `{other}`"
            ));
            return;
        }
    };

    let mut objects = compile(
        &src,
        &[src.join("turbob64c.c"), src.join("turbob64d.c")],
        &[],
    );
    for (files, flags) in &groups {
        objects.extend(compile(&src, files, flags));
    }
    // The FFI floor control is ours, and is deliberately built with the same compiler and
    // flags as the base unit so its call costs what a tb64 call costs.
    objects.extend(compile(&src, &[manifest.join("src/ffi_floor.c")], &[]));

    assert_unique(&objects);
    cc::Build::new().objects(objects).compile("tb64");
    println!("cargo::rustc-cfg=tb64");
}

/// `cc` picks up `CFLAGS` from the environment, which would silently build the C
/// competitor with flags its author never uses and make the comparison something other
/// than what it claims to be.
fn warn_on_inherited_cflags() {
    for var in [
        "CFLAGS",
        "TARGET_CFLAGS",
        "HOST_CFLAGS",
        "CFLAGS_x86_64_unknown_linux_gnu",
    ] {
        println!("cargo::rerun-if-env-changed={var}");
        if let Some(value) = std::env::var_os(var) {
            println!(
                "cargo::warning={var}={value:?} is set. Turbo-Base64 is meant to be built \
                 with its own per-file flags; inherited flags make the benchmark comparison \
                 something other than upstream's own build."
            );
        }
    }
}

/// Copies `from` to `to`, returning `to`. Used to give the second build of a source file
/// an object name of its own.
fn copy_to(from: &Path, to: &Path) -> PathBuf {
    std::fs::copy(from, to)
        .unwrap_or_else(|e| panic!("cannot copy {} to {}: {e}", from.display(), to.display()));
    to.to_path_buf()
}

/// Two archive members sharing a file name link in a way that depends on the linker, so
/// refuse to build one rather than debug it later.
fn assert_unique(objects: &[PathBuf]) {
    let mut names: Vec<_> = objects.iter().filter_map(|o| o.file_name()).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(
        before,
        names.len(),
        "duplicate object file names in libtb64.a"
    );
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

fn compile(include: &Path, files: &[PathBuf], flags: &[&str]) -> Vec<PathBuf> {
    let mut build = cc::Build::new();
    build
        .include(include)
        .opt_level(3)
        .define("NDEBUG", None)
        .flag("-fstrict-aliasing")
        .warnings(false)
        .cargo_metadata(false)
        .files(files);

    // Upstream adds this for gcc only; clang rejects it.
    if !build.get_compiler().is_like_clang() {
        build.flag("-falign-loops");
    }
    for flag in flags {
        build.flag(flag);
    }
    build.compile_intermediates()
}
