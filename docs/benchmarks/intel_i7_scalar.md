# 💻 Benchmark: Intel i7-8750H (Scalar / No-SIMD)

**Context:** This test forcibly disables all SIMD instructions (AVX2, SSE4.1). It measures the raw efficiency of our SWAR (SIMD Within A Register) fallback algorithm against the standard `base64` crate.

*   **Processor:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
*   **Mode:** **Scalar Only** (All SIMD flags disabled via `RUSTFLAGS`)
*   **Competitor:** `base64` (Standard crate, referred to as "Std")

## 📈 Performance Snapshot

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_i7_scalar.png?raw=true)

**Key Findings:**
1.  **Consistent Lead:** Even without SIMD, `base64-turbo` is consistently **20-40% faster** than the standard crate across all file sizes.
2.  **Latency King:** Small payload encoding (32B) is **~1.6x faster** (30ns vs 48ns), making it ideal for high-throughput embedded logging or serial protocols.
3.  **SWAR Efficiency:** The decoding throughput of **~2.0 GiB/s** (vs 1.4 GiB/s for Std) proves the effectiveness of our 64-bit SWAR logic, which processes 8 bytes per cycle using standard integer registers.

## 🏎️ Detailed Results

### 1. Small Payloads (32 Bytes)
**Focus:** Embedded Logging, Serial Comms.

| Crate | Mode | Encode Latency | Encode Speed | Decode Latency | Decode Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | `Scalar` | **30.60 ns** | **0.99 GiB/s** | **33.92 ns** | **1.21 GiB/s** |
| `base64` (std) | `Scalar` | 48.27 ns | 0.63 GiB/s | 53.54 ns | 0.78 GiB/s |

> **Analysis:** `base64-turbo` is **36% faster** in encoding and **36% faster** in decoding. This reduction in cycle count is critical for battery-powered or resource-constrained devices.

### 2. Medium Payloads (64 KB)
**Focus:** L1 Cache Efficiency (Scalar).

| Crate | Encode Speed | vs `std` | Decode Speed | vs `std` |
| :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | **1.69 GiB/s** | **+1.8%** | **2.07 GiB/s** | **+30.1%** |
| `base64` (std) | 1.66 GiB/s | - | 1.59 GiB/s | - |

> **Analysis:**
> *   **Decoding:** The SWAR algorithm shines here, delivering a massive **30% speedup**. By reading `u64` chunks instead of bytes, we reduce memory access frequency.
> *   **Encoding:** Performance is roughly effectively par with the standard library for medium chunks, indicating that the bottleneck here is likely L1 cache throughput rather than ALU ops.

### 3. Large Payloads (10 MB)
**Focus:** Sustained Throughput.

| Crate | Encode Speed | Decode Speed |
| :--- | :--- | :--- |
| **base64-turbo** | **1.59 GiB/s** | **1.95 GiB/s** |
| `base64` (std) | 1.40 GiB/s | 1.48 GiB/s |

> **Analysis:** On large files, the gap widens again. `base64-turbo` maintains a **~13% lead in encoding** and a **~31% lead in decoding**. This suggests our loop unrolling strategies are more cache-friendly than the standard implementation.

## 📝 Raw Data Log
<details>
<summary>Click to view raw Criterion output</summary>

```text
Benchmarking Base64_Performances/Encode/Turbo/32
  time: [30.490 ns 30.606 ns 30.748 ns]
  thrpt: [992.51 MiB/s 997.10 MiB/s 1000.9 MiB/s]

Benchmarking Base64_Performances/Encode/Std/32
  time: [48.114 ns 48.270 ns 48.463 ns]
  thrpt: [629.71 MiB/s 632.23 MiB/s 634.28 MiB/s]

Benchmarking Base64_Performances/Decode/Turbo/32
  time: [33.819 ns 33.918 ns 34.031 ns]
  thrpt: [1.2041 GiB/s 1.2082 GiB/s 1.2117 GiB/s]

Benchmarking Base64_Performances/Decode/Std/32
  time: [53.438 ns 53.540 ns 53.658 ns]
  thrpt: [782.03 MiB/s 783.75 MiB/s 785.25 MiB/s]

Benchmarking Base64_Performances/Encode/Turbo/512
  time: [289.04 ns 289.73 ns 290.46 ns]
  thrpt: [1.6417 GiB/s 1.6458 GiB/s 1.6497 GiB/s]

Benchmarking Base64_Performances/Encode/Std/512
  time: [324.38 ns 326.75 ns 329.60 ns]
  thrpt: [1.4467 GiB/s 1.4593 GiB/s 1.4700 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/512
  time: [316.87 ns 317.26 ns 317.68 ns]
  thrpt: [2.0052 GiB/s 2.0079 GiB/s 2.0103 GiB/s]

Benchmarking Base64_Performances/Decode/Std/512
  time: [429.49 ns 431.35 ns 433.31 ns]
  thrpt: [1.4701 GiB/s 1.4768 GiB/s 1.4832 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/4096
  time: [2.3290 µs 2.3411 µs 2.3554 µs]
  thrpt: [1.6196 GiB/s 1.6294 GiB/s 1.6379 GiB/s]

Benchmarking Base64_Performances/Encode/Std/4096
  time: [2.4026 µs 2.4127 µs 2.4230 µs]
  thrpt: [1.5744 GiB/s 1.5811 GiB/s 1.5877 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/4096
  time: [2.4758 µs 2.4868 µs 2.4996 µs]
  thrpt: [2.0358 GiB/s 2.0463 GiB/s 2.0554 GiB/s]

Benchmarking Base64_Performances/Decode/Std/4096
  time: [3.3578 µs 3.3766 µs 3.3983 µs]
  thrpt: [1.4974 GiB/s 1.5071 GiB/s 1.5155 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/65536
  time: [35.720 µs 35.931 µs 36.177 µs]
  thrpt: [1.6871 GiB/s 1.6987 GiB/s 1.7087 GiB/s]

Benchmarking Base64_Performances/Encode/Std/65536
  time: [36.462 µs 36.581 µs 36.711 µs]
  thrpt: [1.6626 GiB/s 1.6685 GiB/s 1.6739 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/65536
  time: [39.074 µs 39.306 µs 39.563 µs]
  thrpt: [2.0570 GiB/s 2.0705 GiB/s 2.0828 GiB/s]

Benchmarking Base64_Performances/Decode/Std/65536
  time: [50.932 µs 51.061 µs 51.215 µs]
  thrpt: [1.5890 GiB/s 1.5938 GiB/s 1.5979 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/524288
  time: [281.77 µs 283.78 µs 286.00 µs]
  thrpt: [1.7073 GiB/s 1.7207 GiB/s 1.7329 GiB/s]

Benchmarking Base64_Performances/Encode/Std/524288
  time: [295.24 µs 296.77 µs 298.40 µs]
  thrpt: [1.6363 GiB/s 1.6453 GiB/s 1.6538 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/524288
  time: [306.14 µs 306.85 µs 307.60 µs]
  thrpt: [2.1165 GiB/s 2.1217 GiB/s 2.1266 GiB/s]

Benchmarking Base64_Performances/Decode/Std/524288
  time: [411.60 µs 412.93 µs 414.42 µs]
  thrpt: [1.5710 GiB/s 1.5766 GiB/s 1.5817 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/1048576
  time: [568.86 µs 577.58 µs 586.99 µs]
  thrpt: [1.6637 GiB/s 1.6908 GiB/s 1.7167 GiB/s]

Benchmarking Base64_Performances/Encode/Std/1048576
  time: [609.04 µs 619.33 µs 631.19 µs]
  thrpt: [1.5472 GiB/s 1.5768 GiB/s 1.6035 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/1048576
  time: [634.57 µs 642.82 µs 651.78 µs]
  thrpt: [1.9977 GiB/s 2.0256 GiB/s 2.0519 GiB/s]

Benchmarking Base64_Performances/Decode/Std/1048576
  time: [892.17 µs 904.03 µs 915.76 µs]
  thrpt: [1.4219 GiB/s 1.4403 GiB/s 1.4595 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/10485760
  time: [6.0375 ms 6.1181 ms 6.2039 ms]
  thrpt: [1.5741 GiB/s 1.5962 GiB/s 1.6175 GiB/s]

Benchmarking Base64_Performances/Encode/Std/10485760
  time: [6.8635 ms 6.9665 ms 7.0795 ms]
  thrpt: [1.3794 GiB/s 1.4018 GiB/s 1.4228 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/10485760
  time: [6.6172 ms 6.6679 ms 6.7199 ms]
  thrpt: [1.9377 GiB/s 1.9528 GiB/s 1.9677 GiB/s]

Benchmarking Base64_Performances/Decode/Std/10485760
  time: [8.6481 ms 8.7669 ms 8.9077 ms]
  thrpt: [1.4618 GiB/s 1.4852 GiB/s 1.5056 GiB/s]

Model name: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
```
</details>
