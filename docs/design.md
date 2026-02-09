# 🏗️ Architecture & Design

This document details the internal engineering of `base64-turbo`. The design goal is to maximize throughput for **Memory-Safe Rust** by leveraging SIMD (Single Instruction, Multiple Data) and minimizing CPU pipeline stalls, while acknowledging the inherent performance gap compared to unchecked C/Assembly.

## Design Philosophy: Instruction Level Parallelism (ILP)

Traditional Base64 implementations iterate byte-by-byte using lookup tables. This causes two primary bottlenecks:
1.  **Data Dependency:** Each iteration depends on the previous state, preventing the CPU from executing instructions in parallel.
2.  **Branch Misprediction:** Frequent checks for padding (`=`) or invalid characters inside the hot loop flush the pipeline.

`base64-turbo` replaces byte-level logic with **Vectorized Data Movement**:
*   **Batch Processing:** We load 32 or 64 bytes into vector registers and process them simultaneously.
*   **Branchless Logic:** Padding and error detection are handled via bitmasks *after* the vector operation, removing conditional jumps from the hot path.

## 1. Scalar Implementation (SWAR)

For architectures without supported SIMD (e.g., ARM NEON, older x86), we utilize a **SWAR (SIMD Within A Register)** approach.

*   **Wide Loads:** Instead of processing `u8`, we cast pointers to `u64`. This allows us to move 8 bytes of data with a single instruction.
*   **Bitwise Logic:** We construct the Base64 indices using bitwise shifts (`<<`, `>>`) and masks (`&`) on 64-bit integers, rather than individual byte lookups.
*   **Loop Unrolling:** The compiler is guided to unroll loops to minimize increment/compare overhead.
*   **Safety:** While this relies on `unsafe` pointer casts, the bounds are verified via Kani Model Checking to ensure we never read beyond the input slice, even when reading 8 bytes near the end of a buffer.

## 2. AVX2 Implementation

The AVX2 path is the primary engine for modern x86_64 CPUs. The algorithm is tuned for **Execution Port balancing**.

### Port Pressure Optimization
Modern Intel/AMD CPUs have specific "Ports" for different operations.
*   **Shuffle Operations (`vpshufb`):** Typically execute on **Port 5**.
*   **Bitwise/Math Operations:** Can often execute on **Ports 0, 1, or 5**.

If an algorithm relies too heavily on shuffling (common in naive Base64 SIMD), Port 5 becomes the bottleneck while other ports sit idle. We distribute the workload by mixing logical operations (AND/OR/SHIFT) with shuffles to saturate multiple ports simultaneously.

### The "Lane Stitching" Strategy
AVX2 `ymm` registers are 256-bit, but many intrinsics treat them as two separate 128-bit "lanes." Data cannot easily cross from the lower 128 bits to the upper 128 bits.
*   **Problem:** Base64 is a sliding stream; bits must flow continuously across the register.
*   **Solution:** We use a permutation strategy to "stitch" the lanes. By loading data with an offset and permuting it, we bridge the 128-bit boundary without falling back to scalar code.

## 3. AVX512 Implementation

The AVX512 implementation targets Zen 4 and newer Intel CPUs. It offers two distinct advantages over AVX2:

1.  **Bit-Level Masking (`k` registers):**
    AVX512 allows us to apply operations to specific bytes using mask registers. This is critical for the "tail" of the data (the last 1-31 bytes). In AVX2, the tail usually falls back to Scalar code. In AVX512, we generate a mask for the remaining bytes and process the tail in a single vector instruction.

2.  **Larger Register File:**
    With 32 `zmm` registers (vs 16 `ymm` in AVX2), we can unroll loops more aggressively, keeping all lookup tables and constants resident in registers to avoid L1 cache latency.

## 4. Runtime Dispatch

To ensure portability, the library compiles multiple execution paths into a single binary. Feature detection occurs at runtime (during the first call).

**Priority Order:**
1.  **AVX512** (Detected & Verified Safe)
2.  **AVX2**
3.  **SSE4.1**
4.  **Scalar** (Fallback)

> **Note:** This detection prevents `SIGILL` (Illegal Instruction) errors. An ARM CPU or older Intel CPU will simply take the Scalar path.

## 5. Performance Characteristics

Performance is dictated by the complexity of the bit-shuffling required and the available hardware bandwidth.

### Decoder (~20 GiB/s)
Decoding is often **Memory Bound**.
*   The transformation from 4 bytes -> 3 bytes shrinks the data.
*   On high-end test machines (e.g., Xeon Platinum), the decoder achieves ~20+ GiB/s, effectively saturating the RAM bandwidth of the test harness.

### Encoder (~12 GiB/s)
Encoding is **Compute Bound**.
*   The expansion from 3 bytes -> 4 bytes requires complex bit-interleaving.
*   Even with AVX512, the CPU ALUs are the bottleneck.
*   While unchecked C libraries may achieve higher throughput by skipping bounds checks and padding logic, ~12 GiB/s represents the practical limit for a **Memory-Safe** implementation on current hardware.

## Summary

`base64-turbo` is not designed to beat handwritten Assembly or unsafe C in raw theoretical throughput. Its architecture is designed to provide the **maximum possible speed within the constraints of Rust's memory safety guarantees**.
