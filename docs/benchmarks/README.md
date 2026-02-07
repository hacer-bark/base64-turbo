# ⚡ Benchmarks & Methodology

This directory contains detailed performance reports for `base64-turbo` across various hardware architectures.

## 📊 Benchmark Reports

### ☁️ Server / Data Center (HFT & Cloud)
High-performance environments where AVX512 and AVX2 are typically available.

*   **[Intel Xeon Platinum 8488C](./intel_xeon_8488c.md)**
    *   **Environment:** AWS `c7i.large`
    *   **Features:** AVX512, AVX2
    *   **Context:** Modern HFT/Cloud standard.

*   **[AMD EPYC Genoa](./amd_epyc_genoa.md)**
    *   **Environment:** Vultr `voc-c-1c-2gb-25s`
    *   **Features:** AVX512, AVX2
    *   **Context:** Modern AMD Cloud.

### 💻 Consumer / Runtime Scaling Analysis
To demonstrate our **Runtime Dispatch** system, we ran benchmarks on the *same* Intel i7 processor while forcibly disabling instruction sets. This shows exactly how the library scales down from AVX2 to Scalar.

*   **[Intel Core i7-8750H (AVX2 Mode)](./intel_i7_avx2.md)**
    *   **Mode:** Normal operation (Best available).
    *   **Result:** Maximum throughput for this chip.

*   **[Intel Core i7-8750H (SSE4.1 Mode)](./intel_i7_sse41.md)**
    *   **WARNING:** THESE BENCHES NOT YET READY!
    *   **Mode:** AVX2 disabled.
    *   **Context:** Simulates older hardware (e.g., circa 2010-2012) or lower-end CPUs.

*   **[Intel Core i7-8750H (Scalar Mode)](./intel_i7_scalar.md)**
    *   **Mode:** All SIMD disabled.
    *   **Context:** Simulates non-x86 fallback or embedded targets without vector units.

## 🧪 Methodology

All benchmarks were conducted using [criterion.rs](https://github.com/bheisler/criterion.rs) to ensure statistical significance, utilizing a rigorous configuration to filter out OS noise.

### 1. Test Configuration
To ensure high confidence intervals, we utilize longer-than-average test durations and dynamic sampling:

*   **Warm-up Time:** **5 seconds** (Primes branch predictor and L1/L2 caches).
*   **Measurement Time:** **15 seconds** per group.
*   **Noise Threshold:** **0.05** (5%).
*   **Dynamic Sampling:**
    *   **Small Inputs (< 1MB):** 250 samples (High granularity).
    *   **Large Inputs (> 1MB):** 50 samples (To maintain reasonable runtime).

### 2. Input Scaling
We benchmark against a logarithmic spread of data sizes to capture performance characteristics across different cache levels (L1 vs L2 vs RAM).

| Size | Use Case |
| :--- | :--- |
| **32 B** | Small strings, headers, keys. |
| **512 B** | API responses, JSON snippets. |
| **4 KB** | Typical OS page size. |
| **64 KB** | Medium binary blobs. |
| **512 KB** | Large documents/images. |
| **1 MB** | Data streams. |
| **10 MB** | Heavy bulk processing. |

> **Note:** Benchmark plots use a **Logarithmic Scale** on the X-axis to visualize the massive range between 32B and 10MB.

### 3. Comparison Targets (`BENCH_TARGET`)
Our benchmark suite is controlled via the `BENCH_TARGET` environment variable. This allows isolated comparisons between specific implementations. You can provide a comma-separated list of targets.

| Target | Description |
| :--- | :--- |
| `turbo` | **(Default)** The standard `base64-turbo` API (allocating). |
| `turbo-buff` | The `base64-turbo` zero-allocation API (writing to pre-allocated buffer). |
| `simd` | The `base64-simd` crate (Current Rust standard). |
| `std` | The classic `base64` crate (referred to as "std" or "base64" in reports). |
| `all` | Runs all of the above. |

### 4. Reproduction
You can reproduce these results locally using the following commands:

```bash
# Run comparison: base64-turbo vs base64-simd
BENCH_TARGET=turbo,simd cargo bench

# Run EVERYTHING (Warning: Takes a long time)
BENCH_TARGET=all cargo bench

# Bench ONLY the zero-allocation performance
BENCH_TARGET=turbo-buff cargo bench
```
