# Ecosystem Comparison

This project references and benchmarks against several external Base64 libraries. Below is an objective analysis of the current landscape, detailing performance characteristics, implementation details, and safety guarantees.

## Quick Feature Matrix

| Library | Language | SIMD | Verified Safety | Est. Throughput (AVX2) | Source |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **base64-turbo** | Rust | ✅ | ✅ (Kani/MIRI/MSan) | **~12.1 GiB/s** | our `cargo bench` |
| [base64-simd](https://crates.io/crates/base64-simd) | Rust | ✅ | ❌ | ~8.0 GiB/s | our `cargo bench` |
| [base64 (std)](https://crates.io/crates/base64) | Rust | ❌ | ✅ (Compiler) | ~1.6 GiB/s | our `cargo bench` |
| [Turbo-Base64](https://github.com/powturbo/Turbo-Base64) | C | ✅ | ❌ | **~29.0 GiB/s** | vendor-reported, not independently verified |
| [fastbase64](https://github.com/lemire/fastbase64) | C | ✅ | ❌ | ~23.0 GiB/s | vendor-reported, not independently verified |

The Rust rows come from our own [benchmark suite](./benchmarks) (`BENCH_TARGET=all cargo bench`), so you can reproduce them. The C rows are not wired into our benchmark harness — those figures are as published by their respective projects, and we have not verified them ourselves.

## The Rust Ecosystem

### 1. [base64](https://crates.io/crates/base64) (Standard)
The de facto standard library for Rust.
*   **Pros:** Rock-solid stability. Uses 100% Safe Rust. Zero `unsafe` blocks.
*   **Cons:** Low performance. Relies on scalar lookup tables.
*   **Verdict:** Use this if you absolutely cannot have `unsafe` code in your dependency tree and do not care about throughput.

### 2. [base64-simd](https://crates.io/crates/base64-simd)
A well-established SIMD-accelerated Base64 crate for Rust.
*   **Pros:** Significantly faster than standard. Native Rust.
*   **Cons:** Slower than `base64-turbo` in our benchmarks. Uses `unsafe` logic (specifically `core::simd`) that, as far as we're aware, has not been checked by Kani or MIRI.
*   **Verdict:** A strong library. `base64-turbo` measured faster in our benchmarks and additionally carries Kani/MIRI verification, which we could not find published for `base64-simd`.

### 3. [vb64](https://crates.io/crates/vb64) (Experimental)
*   **Status:** Broken / Unmaintained.
*   **Details:** Relies on the unstable `core::simd` nightly API. Because the nightly API changes frequently, this crate currently fails to compile on modern Rust versions. Benchmarks (when it worked) indicated it was slower than `base64-simd`.

### 4. [base-d](https://crates.io/crates/base-d)
*   **Focus:** Flexibility (Supports 33+ alphabets).
*   **Performance:** Uses SIMD for decoding only. Generally slower than `base64-simd`.
*   **Verdict:** Good if you need obscure custom alphabets, not for raw speed.

### 5. [webbuf](https://crates.io/crates/webbuf)
*   **Focus:** WebAssembly compatibility and convenience (whitespace stripping).
*   **Performance:** Prioritizes utility over raw hardware acceleration.

### 6. [baste64](https://crates.io/crates/baste64)
*   **Details:** Uses WASM-based SIMD instructions.
*   **Verdict:** Not benchmarked due to maintainability issues. Generally, the overhead of WASM SIMD makes it slower than native intrinsics.

## The C Ecosystem (Raw Speed)

### 1. [Turbo-Base64](https://github.com/powturbo/Turbo-Base64) (PowTurbo)
One of the fastest Base64 implementations available in any language.
*   **Pros:** Very high throughput — the project's own benchmarks report ~29 GiB/s on AVX2. We have not independently benchmarked it ourselves (it isn't wired into our `cargo bench` suite), so treat that figure as vendor-reported, not verified by us.
*   **Cons:** **Unsafe.** Written in C. Relies on unchecked pointer arithmetic and memory manipulation, with no published formal verification. Harder to build in Rust toolchains (requires a C toolchain).
*   **Verdict:** Use only if you need the theoretical maximum speed and are willing to own the risk of segfaults/buffer overflows and C build complexity yourself.

### 2. [fastbase64](https://github.com/lemire/fastbase64) (Lemire)
A research-oriented library by Daniel Lemire.
*   **Pros:** Excellent performance (vendor-reported ~23 GiB/s, not independently verified by us). Pioneered many SIMD techniques used today.
*   **Cons:** C-based safety risks, no published formal verification.

### 3. [base64](https://github.com/aklomp/base64) (aklomp)
A highly optimized C library by Alfred Klomp.
*   **Pros:** Very fast (vendor-reported ~25 GiB/s, not independently verified by us).
*   **Cons:** C-based safety risks, no published formal verification.

---

> **Final Safety Note:**
> With the exception of the standard `base64` crate (which uses Safe Rust with zero `unsafe`), none of the alternative libraries listed above publish Kani or MIRI verification for their `unsafe`/SIMD code, as far as we could find. To our knowledge, `base64-turbo` is the only crate in this comparison that pairs SIMD-accelerated (AVX512-class) throughput with Kani + MIRI formal verification of its `unsafe` paths — if you know of one we've missed, please open an issue so we can correct this.
