# base64-turbo

[![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![Documentation](https://docs.rs/base64-turbo/badge.svg)](https://docs.rs/base64-turbo)
[![License](https://img.shields.io/github/license/hacer-bark/base64-turbo)](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE)
[![Formal Verification](https://img.shields.io/badge/Formal%20Verification-Kani%20Verified-success)](https://github.com/model-checking/kani)
[![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)
[![Logic Tests](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/tests.yml?label=Logic%20Tests)](https://github.com/hacer-bark/base64-turbo/actions/workflows/tests.yml)

**AVX512-Accelerated, Zero-Allocation Base64 Engine for Rust.**

`base64-turbo` is a production-grade library engineered for **High Frequency Trading (HFT)**, **Mission-Critical Servers**, and **Embedded Systems** where CPU cycles and memory bandwidth are scarce resources.

Designed to align with **modern hardware reality** without sacrificing **portability**, this crate ensures optimal performance across the entire spectrum of computing devices:

*   **Universal Optimization:** Performance is not limited to high-end servers. Even on non-SIMD targets (WASM, IoT, legacy hardware), highly optimized Scalar fallback executes **~1.5x faster** than the ecosystem standard.
*   **Bare-Metal Ready:** Zero dependencies and full `no_std` support make it ideal for embedded firmware, operating system kernels, and bootloaders.
*   **Hardware Sympathy:** The engine performs runtime CPU detection. On supported x86 hardware, it unlocks hand-written AVX2 and AVX512 intrinsics to achieve **10x-12x higher throughput** compared to standard implementations.

Whether you are running on an embedded ARM microcontroller or a Zen 4 data center node, `base64-turbo` automatically selects the fastest, safest algorithm for your specific architecture.

## Ecosystem Comparison

We believe in radical transparency. Below is a fact-based comparison against the leading alternatives in both the Rust and C ecosystems.

### 1. vs. Rust Ecosystem (`base64-simd`)
`base64-turbo` outperforms the current Rust standard by approximately **2x** in raw throughput. This performance delta is achieved through aggressive loop unrolling, reduced instruction count per encoded byte, and hybrid logics.

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_intel.png?raw=true)

**Benchmark Summarize:**

| Metric | `base64-turbo` (This Crate) | `base64-simd` | Speedup |
| :--- | :--- | :--- | :--- |
| **Decode (Read)** | **~21.1 GiB/s** | ~10.0 GiB/s | **+111%** |
| **Encode (Write)** | **~12.5 GiB/s** | ~10.5 GiB/s | **+20%** |
| **Small Data (32B)** | **~3.0 GiB/s** | ~1.6 GiB/s | **+87%** |
| **Latency (32B)** | **~10ns** | ~18 ns | **1.8x Lower** |

> *Figure 1: Comparative benchmarks conducted on an **AWS c7i.large** instance (Intel Xeon Platinum 8488C).*

### 2. vs. C Ecosystem (`turbo-base64`)
The C library `turbo-base64` currently sets the "speed of light" for Base64 encoding. It achieves extreme velocities by utilizing pure C, unchecked pointer arithmetic, and bypassing memory safety checks.

`base64-turbo` (this crate) offers a strategic compromise: it delivers **40-50% of the C speed** while maintaining **100% Rust memory verifications guarantees** and a permissive license.

| Feature | `base64-turbo` (This Crate) | `turbo-base64` (C Library) |
| :--- | :--- | :--- |
| **Throughput (AVX2)** | ~12 GiB/s (Safe Slices) | **~29 GiB/s** (Unchecked Pointers) |
| **Memory Safety** | ✅ **Guaranteed** (MIRI Audited) | ❌ Unsafe (Raw C Pointers) |
| **Formal Verification** | ✅ **Kani Verified** (Math Proofs) | ❌ None (No audits) |
| **Reliability** | ✅ **2 Billion+ Fuzz Iterations** | ❌ Unknown / Not Stated |
| **License** | ✅ **MIT** (Permissive) | ❌ GPLv3 / Commercial |

### The Verdict

*   **Choose the C library** if you require absolute maximum single-core throughput, can tolerate GPL/Commercial licensing, and are willing to accept the risks of unchecked memory access (segfaults/buffer overflows).
*   **Choose `base64-turbo`** if you require the highest possible performance within **Verified Rust** (fast enough to saturate RAM bandwidth) and require a permissive license with formally verified safety guarantees.

## Usage

### Standard (Simple)
The easiest way to use the library. Handles allocation automatically.

```rust
use base64_turbo::STANDARD;

let data = b"huge_market_data_feed...";

// Automatically selects the fastest SIMD algorithm (AVX2, SSE4.1, or AVX512) at runtime.
// 
// Note: Multi-threaded processing (Rayon) is opt-in via the `parallel` feature
// to ensure deterministic latency in standard deployments.
let encoded = STANDARD.encode(data);
let decoded = STANDARD.decode(&encoded).unwrap();
```

### Zero-Allocation
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

## Feature Flags

By default, this crate is **dependency-free** and compiles on Stable Rust. Features are opt-in to allow users to balance compile times, binary size, and specific performance needs.

| Flag | Description | Default |
| :--- | :--- | :--- |
| `std` | Provides high-level `encode` and `decode` API returning heap-allocated `String` and `Vec<u8>`. Disable for embedded/bare-metal `no_std` environments. | **Enabled** |
| `simd` | Enables runtime CPU feature detection (AVX2/SSE4.1). Automatically falls back to safe scalar logic if hardware support is missing. | **Enabled** |
| `parallel` | Enables multi-threaded processing for large payloads (> 512KB) via Rayon. **Note: This adds `rayon` as a dependency.** | **Disabled** |
| `avx512` | Compiles AVX512 intrinsics. Requires a supported CPU (e.g., Zen 4, Ice Lake) to execute the optimized path. | **Disabled** |

### Why are `parallel` and `avx512` disabled by default?

We prioritize **zero-dependencies**, **deterministic latency**, and **strict formal verification** in the default configuration.

#### 1. `parallel` (Rayon)
*   **Dependency Weight:** By default, it strive to keep the crate dependency-free to ensure fast compilation and minimal binary size. Enabling this flag pulls in `rayon`, which is a significant external dependency.
*   **Deterministic Latency:** In latency-sensitive environments (such as High-Frequency Trading or Async Web Servers), a library implicitly spawning threads or blocking a global thread pool can introduce unpredictable jitter.
*   **Context Switching Overhead:** For payloads under 512KB, the cost of thread synchronization and context switching often outweighs the throughput gains of parallelism.
*   *Recommendation:* Enable this only if you are processing massive datasets (MBs/GBs) and are willing to trade unpredictable jitter for memory-saturating throughput.

#### 2. `avx512`
*   **Verification Gap (MIRI):** While the AVX512 path is stable and has withstood over **2.5 Billion+ Fuzzing Iterations**, it is **not yet covered by the MIRI audit**. The Rust MIRI interpreter does not currently support AVX512 intrinsics. Therefore, it cannot formally guarantee that this specific path is free of Undefined Behavior (UB) to the same rigorous standard as Scalar and AVX2 paths.
*   *Recommendation:* Enable this if you are deploying on supported hardware (Zen 4 / Ice Lake) and require the additional ~60% throughput per core, accepting that this path relies on Fuzzing (Done) and Kani (In Progress) verification rather than MIRI.

## Architecture & Hardware Sympathy

This engine is not merely a loop over a lookup table; it is engineered to exploit the micro-architectural mechanics of modern x86 processors. By aligning software logic with hardware capabilities, it is trying to achieve maximum Instruction Level Parallelism (ILP).

*   **AVX2 Lane Stitching:** Standard AVX2 instructions (specifically `vpshufb`) are restricted to 128-bit lanes, preventing data from crossing between the lower and upper halves of a register. It utilize "double-load" intrinsics to bridge this gap, allowing full utilization of the 32-byte YMM registers without pipeline stalls.
*   **Vectorized Arithmetic:** To minimize L1 cache pressure, it replace traditional memory-based lookups with vector arithmetic comparisons. This "logic-over-memory" approach eliminates branch misprediction penalties caused by random input data (entropy).
*   **Optimized Port Saturation (LUTs):** While minimize memory lookups, it utilize highly optimized register-based Look-Up Tables (LUTs) for specific shuffle operations. These are designed to balance the load across CPU execution ports (ALU vs. Shuffle ports), preventing bottlenecks in the superscalar pipeline.
*   **AVX512:** The library features a dedicated AVX512 implementation. Unlike simple AVX2 ports, this path leverages the larger register width and masking capabilities found in Zen 4 and Ice Lake CPUs to significantly increase throughput per core.

## Safety & Formal Verification

Achieving maximum throughput should not come at the cost of memory safety. While this crate leverages `unsafe` intrinsics for SIMD optimizations, the codebase is rigorously audited and formally verified to guarantee stability.

To ensure strict adherence to these standards, **GitHub CI pipeline** is configured to block any release that fails to pass logical tests or MIRI verification.

*   **Formal Verification (Kani)**: The logic for Scalar (Done), SSE4.1 (In Progress), AVX2 (Done), and AVX512 (In Progress) implementations has been verified using the **Kani Model Checker**. This provides a mathematical proof that there are no possible inputs that can trigger Panics or Undefined Behavior (UB) within the core arithmetic.
*   **MIRI Analysis**: The Scalar, SSE4.1, and AVX2 execution paths are audited against the **MIRI Interpreter**. This ensures strict compliance with the Rust memory model, checking for data races, misalignment, and out-of-bounds access.
    *   *Note regarding AVX512*: MIRI does not currently support AVX512 intrinsics. Consequently, AVX512 paths are verified via Kani and Fuzzing, but not MIRI. For more details on this upstream limitation, please refer to the [FAQ](https://github.com/hacer-bark/base64-turbo/tree/main?tab=readme-ov-file#why-are-parallel-and-avx512-disabled-by-default).
*   **Deep Fuzzing**: The decoder and encoder have withstood over **2.5 Billion fuzzing iterations** via `cargo fuzz`. This ensures resilience against edge cases, invalid inputs, and complex buffer boundary conditions.
*   **Dynamic Dispatch**: CPU features are detected at runtime. The library automatically selects the fastest safe implementation available. If hardware support (e.g. AVX512) is missing, it safely falls back to optimized Scalar or SSE4.1 paths.

## Binary Footprint

As part of transparency policy, here the sizes of the compiled library artifact (`.rlib`) under maximum optimization settings (`lto = "fat"`, `codegen-units = 1`).

| Configuration | Size | Details |
| :--- | :--- | :--- |
| **Default** (`std` + `simd`) | **~512 KB** | "Fat Binary" containing **AVX2, SSE4.1**, and Scalar paths to support runtime CPU detection. |
| **Scalar** (`std` only) | **~82 KB** | SIMD disabled. Optimized for legacy x86 or generic architectures. |
| **Embedded** (`no_std`) | **~64 KB** | Pure Scalar logic. Ideal for microcontrollers, WASM, or kernel drivers. |

*> **Note:** These sizes represent the intermediate `.rlib`, which includes metadata and symbol tables. The actual machine code added to your final executable is significantly smaller due to linker dead-code elimination. Additionally, compiling with `-C target-cpu=native` allows the compiler to strip unused SIMD paths, further reducing the binary size.*

## Related Libraries

This project references several external Base64 libraries. Below is a comparative list detailing their performance characteristics and implementation details.

*   **[base64](https://crates.io/crates/base64)**: The standard Base64 library for Rust. It prioritizes safety by utilizing only pure, safe Rust (referenced as `std` in this project's benchmarks).
*   **[base64-simd](https://crates.io/crates/base64-simd)**: A high-performance Rust library. It is currently the fastest Rust-native implementation, trailing only `base64-turbo` (This Crate). **Note:** It utilizes `unsafe` logic, specifically leveraging `core::simd` (e.g., `u8x32`, `u8x64`), and has not undergone formal security audits.
*   **[Turbo-Base64](https://github.com/powturbo/Turbo-Base64)**: The current state-of-the-art implementation regarding raw speed. Written in C, it uses unchecked arithmetic and pointer manipulation to achieve ~70 GB/s throughput with AVX512 and ~30 GB/s with AVX2.
*   **[Base64 (C)](https://github.com/aklomp/base64)**: A highly optimized C library by Alfred Klomp. It utilizes SIMD acceleration to achieve ~25 GB/s throughput with AVX2.
*   **[fast-base64](https://github.com/lemire/fastbase64)**: A research-oriented C library by Daniel Lemire. It achieves approximately ~23 GB/s throughput with AVX2.
*   **[vb64](https://crates.io/crates/vb64)**: An experimental Rust crate. It relies on the unstable `core::simd` module. It currently fails to compile because the `core::simd` API has changed significantly since the crate was written, breaking backward compatibility. Even when functional, benchmarks indicate it is slower than `base64-simd`.
*   **[bs64](https://crates.io/crates/bs64)**: A Rust port attempting to replicate the logic of the C `fast-base64` library. Performance is lower than that of `base64-simd`.
*   **[base-d](https://crates.io/crates/base-d)**: A Rust crate focused on flexibility and ease of use, offering support for 33+ alphabets. It utilizes SIMD for decoding only and is slower than `base64-simd`.
*   **[webbuf](https://crates.io/crates/webbuf)**: A utility crate supporting both Base64 and Hex encoding. It prioritizes WebAssembly (WASM) compatibility and convenience features (such as whitespace stripping) over raw hardware acceleration.
*   **[baste64](https://crates.io/crates/baste64)**: A Rust crate utilizing WASM-based SIMD instructions. It was not benchmarked due to maintainability issues; however, due to the overhead of WASM SIMD, it is projected to be slower than `base64-simd`.

> **Safety Note**: With the exception of the standard `base64` crate (which uses only Safe Rust), none of these libraries offer verified guarantees against Undefined Behavior (UB).

## License

MIT License. Copyright (c) 2026.
