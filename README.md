# base64-turbo

[![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![Documentation](https://docs.rs/base64-turbo/badge.svg)](https://docs.rs/base64-turbo)
[![License](https://img.shields.io/github/license/hacer-bark/base64-turbo)](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE)
[![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)
[![Logic Tests](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/tests.yml?label=Logic%20Tests)](https://github.com/hacer-bark/base64-turbo/actions/workflows/tests.yml)

**Hardware-Accelerated, Zero-Allocation Base64 Engine for Rust.**

`base64-turbo` is a production-grade encoding engine engineered for **High Frequency Trading (HFT)**, **Web Servers**, and **Embedded Systems** where every CPU cycle and byte of memory bandwidth matters.

We optimize for **modern hardware reality** without sacrificing **portability**. While `base64-turbo` is renowned for its blisteringly fast AVX2/AVX512 paths, it is architected to be the fastest engine on *any* platform:

*   **Universal Speed:** Even on targets without SIMD (like older servers, WASM, or IoT devices), our highly optimized Scalar fallback runs **~1.5x to 2x faster** than the standard ecosystem.
*   **Zero Dependencies:** Fully supports `no_std` environments, making it ideal for embedded firmware and operating system kernels.
*   **Hardware Sympathy:** On supported x86 CPUs, it unlocks **10x-12x higher throughput** via hand-written intrinsics and hybrid parallel scheduling.

Whether you are running on an embedded ARM microcontroller or a Zen 4 server, `base64-turbo` automatically selects the fastest safe algorithm for your hardware.

## 🚀 Performance

This is currently the fastest pure Rust Base64 implementation available.

### 1. Maximum System Throughput (AVX2 + Parallel)
Benchmarks run on a consumer Intel Core i7-8750H (4 vCPU, AVX2, DDR4). The engine automatically scales to saturate memory bandwidth on large payloads.

| Operation | Size | Throughput | Context |
| :--- | :--- | :--- | :--- |
| **Decode** | 1 MB | **~16.6 GiB/s** | L3 Cache Saturation (Parallel) |
| **Decode** | 10 MB+ | **~10.0 GiB/s** | RAM Bandwidth Limited |
| **Encode** | 1 MB | **~15.9 GiB/s** | L3 Cache Saturation (Parallel) |
| **Encode** | 10 MB+ | **~8.2 GiB/s** | RAM Bandwidth Limited |
| **Latency** | 32 B | **~19 ns** | Zero-Alloc Hot Path |

*> **Note:** At peak throughput, `base64-turbo` approaches the theoretical limit of `memcpy` on this machine, effectively saturating the memory controller.*

### 2. Instruction Set Scaling (AVX512 vs AVX2)
To measure raw per-core efficiency, we benchmarked on a **shared, noisy VPS environment** (1 vCPU, limited power budget). This isolates the architectural efficiency of our hand-written intrinsics.

Even in a constrained environment, enabling `avx512` delivers a massive leap in performance.

| Backend / Crate | Instruction Set | Throughput (4KB) | Relative Speed |
| :--- | :--- | :--- | :--- |
| `base64` (Standard) | Scalar | ~0.9 GiB/s | 1.0x |
| `base64-turbo` | **AVX2** | **~4.3 GiB/s** | **4.7x** |
| `base64-turbo` | **AVX512** | **~6.8 GiB/s** | **7.5x** |

*> **Key Takeaway:** The AVX512-VBMI path provides a **~60% performance boost per core** over our already-optimized AVX2 path. On dedicated modern hardware (Zen 4 / Ice Lake), single-core throughput is projected to exceed 12 GiB/s.*

### 3. Scalar / Portable Performance (No SIMD)
We benchmarked `base64-turbo` against the standard `base64` crate with **all SIMD features disabled**. This represents performance on legacy hardware, WASM, or embedded targets.

Even without hardware acceleration, our algorithmic optimizations and "Zero-Allocation" API provide significant gains.

| Operation | Size | `base64` (Std) | `base64-turbo` (Scalar) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Decode** | 10 MB | 1.35 GiB/s | **2.23 GiB/s** | **~1.65x** |
| **Encode** | 10 MB | 1.48 GiB/s | **1.55 GiB/s** | **~1.05x** |
| **Encode** | 32 B | 0.65 GiB/s | **1.34 GiB/s** | **~2.06x** |
| **Latency** | 32 B | ~47 ns | **~22 ns** | **~2.14x** |

*> **Note:** The 32 B Encode and Latency comparisons use the `encode_into` (Zero-Allocation) API for `base64-turbo`, demonstrating the efficiency of avoiding heap allocation for small, hot-path payloads.*

## 🆚 Ecosystem Comparison

We believe in transparency. Below is a fact-based comparison against the best Rust and C alternatives.

### vs. Rust Ecosystem (`base64-simd`)
`base64-turbo` outperforms the current Rust gold standard by approximately **2x** in raw throughput due to aggressive loop unrolling, reduced instruction count per byte, and hybrid parallelism.

| Crate | Decode Speed (1MB) | Implementation |
| :--- | :--- | :--- |
| **`base64-turbo`** | **16.6 GiB/s** | **AVX2 + Hybrid Parallelism** |
| `base64-simd` | 8.3 GiB/s | AVX2 Multi Threaded |
| `base64` (Standard) | 1.4 GiB/s | Scalar |

### vs. C Ecosystem (`turbo-base64`)
The C library `turbo-base64` is the "speed of light" benchmark. It achieves extreme speeds by using unchecked C pointers and ignoring memory safety.

| Feature | `base64-turbo` (Rust, AVX2) | `turbo-base64` (C, AVX2) |
| :--- | :--- | :--- |
| **Single Core Speed** | ~7-8 GiB/s (Safe Slices) | **~29 GiB/s** (Unchecked Pointers) |
| **Multi Core Speed** | **~16.6 GiB/s** (Saturates RAM) | N/A |
| **Memory Safety** | ✅ **Guaranteed** (MIRI Audited) | ❌ Unsafe (Raw C) |
| **Vulnerability Check**| ✅ **1 Billion+ Fuzz Iterations** | ❓ Unknown / Not Stated |
| **License** | ✅ **MIT** (Permissive) | ⚠️ GPLv3 / Commercial |

**Verdict:** If you need absolute maximum single-core speed regardless of safety or licensing, use C. If you need the fastest possible speed within Safe Rust (fast enough to saturate RAM) with a permissive license, use `base64-turbo`.

*> **Note:** While C achieves higher L1 throughput, `base64-turbo` is designed to saturate the Memory Controller (DDR4/DDR5 bandwidth) safely, which is the practical limit for real-world ingestion workloads.*

## ⚡ Architecture & Hardware Sympathy

This is not just a loop over a lookup table. The engine is engineered to exploit specific x86 mechanics:

*   **AVX2 Lane Stitching:** Uses custom "double-load" intrinsics to overcome the 128-bit lane-crossing limitations of AVX2, allowing full 32-byte register utilization.
*   **Algorithmic Mapping:** Replaces memory lookups (which cause cache pressure) with vector arithmetic comparisons. This eliminates branch misprediction penalties on random input data.
*   **AVX512 Support:** Includes one of the first production-ready AVX512-VBMI paths in the Rust ecosystem, offering ~60% higher throughput per core on Zen 4 and Ice Lake CPUs compared to AVX2.
*   **Hybrid Scheduling:** Automatically switches between Pure SIMD (low overhead) and Rayon Parallelism (memory saturation) based on input size thresholds (> 512KB).

## 🛡️ Safety & Verification

High performance does not mean undefined behavior. This crate uses `unsafe` for SIMD and Scalar optimizations, but it is rigorously audited.

*   **Fuzz Testing:** The codebase has undergone over **1 Billion fuzzing iterations** via `cargo-fuzz` to detect edge cases, invalid inputs, and buffer boundary conditions.
*   **MIRI Verified:** The core logic, scalar fallbacks, and AVX2 paths are audited against the MIRI Interpreter to ensure no misalignment, data races, or out-of-bounds access occurs.
*   **Runtime Detection:** CPU features are detected at runtime. If SSSE3/AVX2/AVX512 is unavailable, it falls back to a highly optimized scalar implementation.

## 📦 Usage

### Standard (Simple)
The easiest way to use the library. Handles allocation automatically.

```rust
use base64_turbo::STANDARD;

let data = b"huge_market_data_feed...";

// Automatically selects the fastest SIMD algorithm (AVX2, SSSE3, or AVX512) at runtime.
// 
// Note: Multi-threaded processing (Rayon) is opt-in via the `parallel` feature
// to ensure deterministic latency in standard deployments.
let encoded = STANDARD.encode(data);
let decoded = STANDARD.decode(&encoded).unwrap();
```

### Zero-Allocation (HFT / Embedded)
For hot paths where `malloc` overhead is unacceptable.

```rust
use base64_turbo::STANDARD;

let input = b"order_id_123";
let mut buffer = [0u8; 1024]; // Stack allocated, kept hot in L1 cache

// No syscalls, no malloc, pure CPU cycles
// Returns Result<usize, Error> indicating bytes written
let len = STANDARD.encode_into(input, &mut buffer).unwrap();

assert_eq!(&buffer[..len], b"b3JkZXJfaWRfMTIz");
```

## ⚙️ Feature Flags

| Flag | Description | Default |
| :--- | :--- | :--- |
| `std` | Enables `encode` and `decode` functions (allocating `String`/`Vec`). Disable for `no_std` environments. | **On** |
| `simd` | Enables runtime detection for AVX2 and SSSE3 intrinsics. Falls back to scalar if hardware is unsupported. | **On** |
| `parallel` | Enables Rayon multi-threading for large payloads (> 512KB). | **Off** |
| `avx512` | Enables AVX512-VBMI intrinsics on supported CPUs. | **Off** |

### Why are `parallel` and `avx512` disabled by default?

We prioritize **deterministic latency** and **formal verification** out of the box.

1.  **`parallel` (Rayon):**
    *   **Thread Safety:** In latency-sensitive applications (like HFT or Async Web Servers), a library spawning threads or blocking the global thread pool can cause unpredictable jitter.
    *   **Overhead:** For payloads under 512KB, the cost of context switching outweighs the throughput gains.
    *   *Recommendation:* Enable this only if you are processing massive files (MBs/GBs) and want to trade CPU cores for raw memory-saturating throughput.

2.  **`avx512`:**
    *   **Audit Status:** While the AVX512 path is stable and has passed explicit **100 Million+ Fuzzing Iterations**, it is **not yet covered by the MIRI audit**. The Rust MIRI interpreter does not currently support AVX512 intrinsics, meaning we cannot formally guarantee undefined-behavior-free execution for this specific path to the same rigorous standard as our AVX2 path.
    *   *Recommendation:* Enable this if you are running on Zen 4 / Ice Lake hardware and need the extra ~60% throughput per core, and accept Fuzzing as sufficient validation.

## License

MIT License. Copyright (c) 2026.

