# Base64 Turbo

[![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![License](https://img.shields.io/crates/l/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/verification.yml?label=Kani%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/verification.yml)
[![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)

**The fastest Base64 implementation we're aware of that is formally verified memory-safe.**

`base64-turbo` is a production-grade library engineered for **High Frequency Trading (HFT)**, **Mission-Critical Servers**, and **Embedded Systems** where CPU cycles are scarce and Undefined Behavior (UB) is unacceptable.

It aligns with **modern hardware reality** without sacrificing portability. It automatically detects the best algorithm at runtime:
*   **x86_64:** Uses AVX512 or AVX2.
*   **ARM (aarch64):** Uses NEON.
*   **Other:** Falls back to a highly optimized Scalar kernel.

### What we actually mean by "fastest" and "memory-safe"

We define **memory-safe** narrowly and concretely: the `unsafe` code paths are checked by the [Kani model checker](https://github.com/model-checking/kani) (mathematical proof, not testing) and [MIRI](https://github.com/rust-lang/miri) (a strict Undefined Behavior interpreter), on top of MemorySanitizer audits and continuous fuzzing. See [Safety & Verification](docs/verification.md) for exactly what is and isn't covered per architecture.

We are **not** claiming to be faster than unchecked C/assembly implementations — we aren't, and we don't try to be (see [Ecosystem Comparison](docs/ecosystem_comparison.md) for real numbers). Our claim is narrower and, we believe, fully defensible: **within the set of crates that combine SIMD-accelerated Base64 with Kani + MIRI formal verification, we are not aware of another one that reaches AVX512 speeds.** That's a statement about the current ecosystem as far as we know it, not a universal superlative — if you know of one, please open an issue.

## Quick Start

### Encoding

```rust
use base64_turbo::STANDARD;

fn main() {
    let data = b"Speed and Safety";
    let encoded = STANDARD.encode(data);
    assert_eq!(encoded, "U3BlZWQgYW5kIFNhZmV0eQ==");
}
```

### Decoding

```rust
use base64_turbo::STANDARD;

fn main() {
    let encoded = "U3BlZWQgYW5kIFNhZmV0eQ==";
    
    // Returns Result<Vec<u8>, Error>
    let decoded = STANDARD.decode(encoded).unwrap();
    
    assert_eq!(decoded, b"Speed and Safety");
}
```

### Zero-Allocation (Stack)

For scenarios where heap allocation is too slow (e.g., HFT hot paths), write directly to stack buffers:

```rust
use base64_turbo::STANDARD;

fn main() {
    let input = b"Low Latency";
    let mut output = [0u8; 64];

    // Returns Result<usize, Error>
    let len = STANDARD.encode_into(input, &mut output).unwrap();

    assert_eq!(&output[..len], b"TG93IExhdGVuY3kK");
}
```

## Compatibility & Stability

### Minimum Supported Rust Version (MSRV)
**This crate requires Rust 1.89.0 or newer.**
We rely on recently stabilized AVX512 intrinsics in the standard library to guarantee safety without external dependencies.
*   We **do not** plan to lower this requirement in the future.
*   We **do not** plan to support older compilers via feature flags.

### Public API Stability
The public API (traits, structs, and error types) is considered **Stable**.
*   We adhere to **Semantic Versioning**.
*   The current API surface will remain valid and backward-compatible throughout the `0.2.x` lifecycle.

## Performance

**Claim:** `base64-turbo` outperforms the current Rust standard by approximately **2x** in raw throughput and offers **1.8x lower latency**.

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_intel.png?raw=true)

**Benchmark Summary (Intel Xeon Platinum 8488C):**

| Metric | `base64-turbo` | `base64-simd` | Improvement |
| :--- | :--- | :--- | :--- |
| **Decode Throughput** | **~21.1 GiB/s** | ~10.0 GiB/s | **+111%** |
| **Encode Throughput** | **~12.5 GiB/s** | ~10.5 GiB/s | **+20%** |
| **Latency (32B)** | **~10ns** | ~18 ns | **1.8x Lower** |

**[See Full Benchmark Reports](https://github.com/hacer-bark/base64-turbo/tree/main/docs/benchmarks)**

## Safety & Verification

Achieving maximum throughput must not cost memory safety. We leverage `unsafe` intrinsics for SIMD, so instead of relying on manual review alone, we run every `unsafe` path through multiple independent verification layers. Each layer proves or checks a specific, narrow property — see the matrix below for exactly which layers cover which architecture.

*   **Kani Verified:** For the architectures marked below, Kani mathematically proves the encode/decode kernels never read or write out of bounds and never panic, for all possible inputs of the induction length (see [Safety & Verification](docs/verification.md) for the proof strategy).
*   **MIRI Verified:** MIRI's interpreter checks that no Undefined Behavior (UB) — invalid pointer provenance, misaligned access, data races, etc. — occurs on the test inputs it's run against.
*   **MSan Audited:** MemorySanitizer confirms our test runs never branch on or output uninitialized memory.
*   **Fuzz Tested:** Over 2.5 billion `cargo-fuzz` iterations across all code paths with zero crashes found to date.

**Verified Architectures:**

| Architecture | MIRI | MSan | Kani | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Scalar** | ✅ | ✅ | ✅ | **Formally Verified** |
| **AVX2** | ✅ | ✅ | ✅ | **Formally Verified** |
| **AVX512** | ✅ | ✅ | ✅ | **Formally Verified** |
| **AVX512-VBMI** | ✅ | ✅ | ❌ | **MIRI Verified** |
| **NEON** | ✅ | ✅ | ❌  | **MIRI Verified** |

AVX512-VBMI (the `vpermb`/`vpermi2b` fast path, gated separately from plain AVX512) has no Kani proof — see [Safety & Verification](docs/verification.md#kani-coverage-avx512-vs-avx512-vbmi) for why, and note that the AVX512 Kani proof itself runs locally rather than in CI (also explained there).

**[Read the Verification Audit](https://github.com/hacer-bark/base64-turbo/blob/main/docs/verification.md)**

## Ecosystem Comparison

Here is how we stack up against the fastest C library we benchmarked against. **We are slower — on purpose.**

**vs. C (`turbo-base64`)**
`turbo-base64` is one of the fastest known Base64 implementations, C or otherwise. It gets there via unchecked pointer arithmetic with no safety verification. `base64-turbo` trades some of that raw throughput for formally-verified memory safety.

| Feature | `base64-turbo` (This Crate) | `turbo-base64` (C Library) |
| :--- | :--- | :--- |
| **Throughput** | ~12-20 GiB/s (Safe Slices) | **~29 GiB/s** (Unchecked Pointers) |
| **Memory Safety** | ✅ Kani + MIRI verified `unsafe` paths | ❌ Unaudited raw C pointers |
| **Formal Verification** | ✅ Kani proofs (see coverage matrix above) | ❌ None published |
| **Fuzzing** | ✅ 2.5B+ `cargo-fuzz` iterations, no crashes found | ❌ Not stated |
| **License** | ✅ MIT or Apache-2.0 | ❌ GPLv3 / Commercial |

**Verdict:** Choose `base64-turbo` if you want to stay close to C-level throughput while keeping every `unsafe` path Kani/MIRI-verified. Choose the C library if you need the absolute fastest Base64 available and can own the safety risk yourself.

## Feature Flags

| Feature | Default | Description |
| :--- | :---: | :--- |
| `std` | ✅ | Enables `String` and `Vec` support. Disable for `no_std` |
| `simd` | ✅ | Enables runtime detection for AVX512 and AVX2 (x86/x86_64) |
| `neon` | ✅ | Enables NEON SIMD acceleration on aarch64 (ARM64). No `std` required. |
| `unstable` | ❌ | Exposes raw `unsafe` internal functions (e.g., `encode_avx2`, `encode_neon`) |

## Documentation

*   [**Safety & Verification**](https://github.com/hacer-bark/base64-turbo/blob/main/docs/verification.md) - Proofs, MIRI logs, and audit strategy.
*   [**Benchmarks & Methodology**](https://github.com/hacer-bark/base64-turbo/tree/main/docs/benchmarks) - Hardware specs and reproduction steps.
*   [**Architecture & Design**](https://github.com/hacer-bark/base64-turbo/blob/main/docs/design.md) - Internal data flow and SIMD selection logic.
*   [**Ecosystem Comparison**](https://github.com/hacer-bark/base64-turbo/blob/main/docs/ecosystem_comparison.md) - Comparison of top Rust and C libs.
*   [**FAQ**](https://github.com/hacer-bark/base64-turbo/blob/main/docs/faq.md) - Common questions about `no_std`, NEON, and embedded support.

## Acknowledgements

The encode/decode kernels build directly on techniques published by other Base64 implementations, all under permissive licenses:

*   **[Alfred Klomp](https://github.com/aklomp) — [`aklomp/base64`](https://github.com/aklomp/base64) (BSD-2-Clause).** Our decoder's nibble-lookup validation (`lut_lo`/`lut_hi`/`lut_roll`) and our encoder's offset-load loop (avoiding a per-iteration lane permute) and single-LUT character mapping are direct ports of techniques from this library. The URL-safe alphabet variants of these tables aren't published anywhere we could find, upstream or otherwise — we re-derived them ourselves following the same construction method and verified them exhaustively (see `src/simd/avx2.rs`).
*   **[Daniel Lemire](https://github.com/lemire) and Wojciech Muła — [`lemire/fastbase64`](https://github.com/lemire/fastbase64) (BSD-2-Clause).** This library's `fastavxbase64.c` independently documents and credits the same nibble-lookup decode algorithm (originated by Muła, with the `+`/`/` disambiguation trick credited there to `@aqrit`), which we cross-referenced against `aklomp/base64` while implementing our version.
*   **[`base64-simd`](https://crates.io/crates/base64-simd) (MIT).** The existing Rust SIMD Base64 crate that raised the bar before we did — see [Ecosystem Comparison](docs/ecosystem_comparison.md) for how we stack up against it. Its existence, benchmarks, and API design were a useful reference point throughout.

**We love open source. None of this would exist without people willing to share what they figured out.**

## License

This project licensed under either the [MIT License](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-MIT) or the [Apache License, Version 2.0](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-APACHE) at your option.
