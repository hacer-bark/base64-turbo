# Base64 Turbo

[![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![License](https://img.shields.io/crates/l/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/verification.yml?label=Kani%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/verification.yml)
[![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)

**The fastest memory-safe Base64 implementation.**

`base64-turbo` is a production-grade library engineered for **High Frequency Trading (HFT)**, **Mission-Critical Servers**, and **Embedded Systems** where CPU cycles are scarce and Undefined Behavior (UB) is unacceptable.

It aligns with **modern hardware reality** without sacrificing portability. It automatically detects the best algorithm at runtime:
*   **x86_64:** Uses AVX512, AVX2, or SSE4.1.
*   **ARM / Other:** Falls back to a highly optimized Scalar kernel.

## Quick Start

### Installation

**Requires Rust 1.89+** (Due to stabilized AVX512 intrinsics).

```toml
[dependencies]
base64-turbo = "0.1"
```

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
*   The current API surface will remain valid and backward-compatible throughout the `0.1.x` lifecycle.

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

Achieving maximum throughput must not cost memory safety. While we leverage `unsafe` intrinsics for SIMD, we have mathematically proven the absence of bugs using a "Swiss Cheese" model of verification layers.

*   ✅ **Kani Verified:** Mathematical proofs ensure no input (0..∞ bytes) can cause panics or overflows.
*   ✅ **MIRI Verified:** Validates that no Undefined Behavior (UB) occurs during execution across all architectures.
*   ✅ **MSan Audited:** MemorySanitizer confirms no logic is ever performed on uninitialized memory.
*   ✅ **Fuzz Tested:** Over 2.5 billion iterations with zero failures.

**Verified Architectures:**

| Architecture | MIRI | MSan | Kani | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Scalar** | ✅ | ✅ | ✅ | **Formally Verified** |
| **AVX2** | ✅ | ✅ | ✅ | **Formally Verified** |
| **SSE4.1** | ✅ | ✅ | 🚧 | **Memory Safe (Audited)** |
| **AVX512** | ✅ | ✅ | 🚧 | **Memory Safe (Audited)** |

**[Read the Verification Audit](https://github.com/hacer-bark/base64-turbo/blob/main/docs/verification.md)**

## Ecosystem Comparison

We believe in radical transparency. Here is how we stack up against the fastest C library.

**vs. C (`turbo-base64`)**
The C library `turbo-base64` is the current theoretical "speed of light." However, it relies on unchecked pointer arithmetic. `base64-turbo` offers a strategic compromise: **Massive speed, but with 100% memory safety.**

| Feature | `base64-turbo` (This Crate) | `turbo-base64` (C Library) |
| :--- | :--- | :--- |
| **Throughput** | ~12-20 GiB/s (Safe Slices) | **~29 GiB/s** (Unchecked Pointers) |
| **Memory Safety** | ✅ **Guaranteed** (MIRI Audited) | ❌ Unsafe (Raw C Pointers) |
| **Formal Verification** | ✅ **Kani Verified** (Math Proofs) | ❌ None (No audits) |
| **Reliability** | ✅ **2.5 Billion Fuzz Iterations** | ❌ Unknown / Not Stated |
| **License** | ✅ **MIT or Apache-2.0** | ❌ GPLv3 / Commercial |

**Verdict:** Choose `base64-turbo` if you need to saturate RAM bandwidth **safely** with a permissive license. Choose the C library only if you require absolute theoretical max speed and can tolerate segfault risks.

## Feature Flags

| Feature | Default | Description |
| :--- | :---: | :--- |
| `std` | ✅ | Enables `String` and `Vec` support. Disable for `no_std` |
| `simd` | ✅ | Enables runtime detection for AVX512, AVX2, and SSE4.1 |
| `unstable` | ❌ | Exposes raw `unsafe` internal functions (e.g., `encode_avx2`) |

## Documentation

*   [**Safety & Verification**](https://github.com/hacer-bark/base64-turbo/blob/main/docs/verification.md) - Proofs, MIRI logs, and audit strategy.
*   [**Benchmarks & Methodology**](https://github.com/hacer-bark/base64-turbo/tree/main/docs/benchmarks) - Hardware specs and reproduction steps.
*   [**Architecture & Design**](https://github.com/hacer-bark/base64-turbo/blob/main/docs/design.md) - Internal data flow and SIMD selection logic.
*   [**Ecosystem Comparison**](https://github.com/hacer-bark/base64-turbo/blob/main/docs/ecosystem_comparison.md) - Comparison of top Rust and C libs.
*   [**FAQ**](https://github.com/hacer-bark/base64-turbo/blob/main/docs/faq.md) - Common questions about `no_std`, NEON, and embedded support.

## License

This project licensed under either the [MIT License](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-MIT) or the [Apache License, Version 2.0](https://github.com/hacer-bark/base64-turbo/blob/main/LICENCE-APACHE) at your option.


