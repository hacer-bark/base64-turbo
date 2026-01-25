# Base64 Turbo

[![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![Documentation](https://docs.rs/base64-turbo/badge.svg)](https://docs.rs/base64-turbo)
[![License](https://img.shields.io/github/license/hacer-bark/base64-turbo)](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE)
[![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/verification.yml?label=Kani%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/verification.yml)
[![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)
[![Logic Tests](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/tests.yml?label=Logic%20Tests)](https://github.com/hacer-bark/base64-turbo/actions/workflows/tests.yml)

**The fastest memory-safe Base64 implementation.**

`base64-turbo` is a production-grade library engineered for **High Frequency Trading (HFT)**, **Mission-Critical Servers**, and **Embedded Systems** where CPU cycles are scarce and Undefined Behavior (UB) is unacceptable.

It aligns with **modern hardware reality** without sacrificing portability. Whether running on an embedded ARM microcontroller or a Zen 4 node, it automatically selects the fastest, safest SIMD algorithm for your specific architecture or very fast Scalar fallback.

## Quick Start

```toml
[dependencies]
base64-turbo = "0.1"
```

```rust
use base64_turbo::STANDARD;

fn main() {
    let data = b"Speed and Safety";
    
    // Auto-selects AVX2 / SSE4.1 / Scalar based on hardware
    let encoded = STANDARD.encode(data); 
    assert_eq!(encoded, "U3BlZWQgYW5kIFNhZmV0eQ==");
}
```

## Performance Summary

**Claim:** `base64-turbo` outperforms the current Rust standard by approximately **2x** in raw throughput and offers **1.8x lower latency**.

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_intel.png?raw=true)

**Benchmark Summary:**

| Metric | `base64-turbo` | `base64-simd` | Improvement |
| :--- | :--- | :--- | :--- |
| **Decode Throughput** | **~21.1 GiB/s** | ~10.0 GiB/s | **+111%** |
| **Encode Throughput** | **~12.5 GiB/s** | ~10.5 GiB/s | **+20%** |
| **Latency (32B)** | **~10ns** | ~18 ns | **1.8x Lower** |

**[See More Benchmark Reports](docs/benchmarks/README.md)**: Includes methodology, hardware specs, and reproduction scripts.

## Safety & Verification

Achieving maximum throughput must not cost memory safety. While we leverage `unsafe` intrinsics for SIMD, we have mathematically proven the absence of bugs.

*   ✅ **Kani Verified:** Mathematical proofs ensure no input can cause panics or overflows.
*   ✅ **MIRI Verified:** Validates that no Undefined Behavior (UB) occurs during execution.
*   ✅ **Fuzz Tested:** Over 2.5 billion iterations with zero failures.

**[See Verification Proofs](docs/verification.md)**: Details on our threat model and formal verification strategy.

## Ecosystem Comparison

We believe in radical transparency. Here is how we stack up against the C ecosystem.

**vs. C (`turbo-base64`)**
The C library `turbo-base64` is the current "speed of light." However, it relies on unchecked pointer arithmetic and restrictive licensing. `base64-turbo` offers a strategic compromise: **50% of the speed of C, but with 100% memory safety.**

| Feature | `base64-turbo` (This Crate) | `turbo-base64` (C Library) |
| :--- | :--- | :--- |
| **Throughput (AVX2)** | ~12 GiB/s (Safe Slices) | **~29 GiB/s** (Unchecked Pointers) |
| **Memory Safety** | ✅ **Guaranteed** (MIRI Audited) | ❌ Unsafe (Raw C Pointers) |
| **Formal Verification** | ✅ **Kani Verified** (Math Proofs) | ❌ None (No audits) |
| **Reliability** | ✅ **2.5 Billion Fuzz Iterations** | ❌ Unknown / Not Stated |
| **License** | ✅ **MIT** (Permissive) | ❌ GPLv3 / Commercial |

**Verdict:** Choose `base64-turbo` if you need to saturate RAM bandwidth safely with a permissive license. Choose the C library only if you require absolute theoretical max speed and can tolerate segfault risks.

## Documentation

For detailed implementation data, please refer to our internal docs:

*   [**Safety & Formal Verification**](docs/verification.md) - MIRI/Kani proofs.
*   [**Architecture & Design**](docs/design.md) - Internal data flow and SIMD selection logic.
*   [**FAQ**](docs/faq.md) - Common questions about `no_std` and embedded support.

## License

MIT License. Copyright (c) 2026.
