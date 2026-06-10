# ⚖️ Ecosystem Comparison

This project references and benchmarks against several external Base64 libraries. Below is an objective analysis of the current landscape, detailing performance characteristics, implementation details, and safety guarantees.

## 📊 Quick Feature Matrix

| Library | Language | SIMD | Verified Safety | Est. Throughput (AVX2) |
| :--- | :---: | :---: | :---: | :--- |
| **base64-turbo** | Rust | ✅ | ✅ (Kani/MIRI/MSan) | **~12.1 GiB/s** |
| [base64-simd](https://crates.io/crates/base64-simd) | Rust | ✅ | ❌ | ~8.0 GiB/s |
| [base64 (std)](https://crates.io/crates/base64) | Rust | ❌ | ✅ (Compiler) | ~1.6 GiB/s |
| [Turbo-Base64](https://github.com/powturbo/Turbo-Base64) | C | ✅ | ❌ | **~29.0 GiB/s** |
| [fastbase64](https://github.com/lemire/fastbase64) | C | ✅ | ❌ | ~23.0 GiB/s |

## The Rust Ecosystem

### 1. [base64](https://crates.io/crates/base64) (Standard)
The de facto standard library for Rust.
*   **Pros:** Rock-solid stability. Uses 100% Safe Rust. Zero `unsafe` blocks.
*   **Cons:** Low performance. Relies on scalar lookup tables.
*   **Verdict:** Use this if you absolutely cannot have `unsafe` code in your dependency tree and do not care about throughput.

### 2. [base64-simd](https://crates.io/crates/base64-simd)
The previous speed leader in the Rust ecosystem.
*   **Pros:** Significantly faster than standard. Native Rust.
*   **Cons:** Slower than `base64-turbo`. Uses `unsafe` logic (specifically `core::simd`) that has not been formally audited by Kani or MIRI.
*   **Verdict:** A strong library, but `base64-turbo` supersedes it in both throughput and verification guarantees.

### 3. [vb64](https://crates.io/crates/vb64) (Experimental)
*   **Status:** ⚠️ Broken / Unmaintained.
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
The current "Speed of Light" for Base64.
*   **Pros:** Extreme velocity (~29GB/s AVX2, ~70GB/s AVX512).
*   **Cons:** **Unsafe.** Written in C. Relies on unchecked pointer arithmetic and memory manipulation. Harder to build in Rust chains (requires C toolchain).
*   **Verdict:** Use only if you need theoretical maximum speed and are willing to accept the risk of Segfaults/Buffer Overflows and C build complexity.

### 2. [fastbase64](https://github.com/lemire/fastbase64) (Lemire)
A research-oriented library by Daniel Lemire.
*   **Pros:** Excellent performance (~23 GB/s). Pioneered many SIMD techniques used today.
*   **Cons:** C-based safety risks. Slower than `Turbo-Base64`.

### 3. [base64](https://github.com/aklomp/base64) (aklomp)
A highly optimized C library by Alfred Klomp.
*   **Pros:** Very fast (~25 GB/s).
*   **Cons:** C-based safety risks.

---

> **🛡️ Final Safety Note:**
> With the exception of the standard `base64` crate (which uses Safe Rust), **none** of the alternative libraries listed above offer verified guarantees against Undefined Behavior (UB). `base64-turbo` is unique in bridging the gap between C-level speed and Rust-level formal safety.
