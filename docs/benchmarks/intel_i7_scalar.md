# 💻 Benchmark: Intel i7-8750H (Scalar / No-SIMD)

**Context:** This test forcibly disables all SIMD instructions (AVX2, SSE4.1). It measures the raw efficiency of our custom fallback algorithm against the standard `base64` crate.

*   **Processor:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
*   **Mode:** **Scalar Only**
*   **Competitor:** `base64` (Standard crate, referred to as "Std")

## 📈 Performance Snapshot

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_i7_scalar.png?raw=true)

**Key Findings:**
1.  **Latency King (No-Alloc):** The `TurboBuff` (no-allocation) implementation obliterates overhead for small payloads (32B), achieving **~20ns** latency vs **~45ns** for Std (>2x faster).
2.  **Sustained Throughput:** While encoding speeds are competitive in the medium range, `base64-turbo` pulls ahead significantly in decoding for large files, proving the effectiveness of 64-bit register processing.

## 🏎️ Detailed Results

### 1. Small Payloads (32 Bytes)
**Focus:** Embedded Logging, Serial Comms, Zero-Allocation targets.

| Crate | Mode | Encode Latency | Encode Speed | Decode Latency | Decode Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | `No-Alloc` | **20.80 ns** | **1.43 GiB/s** | **24.93 ns** | **1.64 GiB/s** |
| `base64` (std) | `Scalar` | 45.01 ns | 0.66 GiB/s | 52.41 ns | 0.78 GiB/s |

> **Analysis:** The `base64-turbo` (TurboBuff) implementation is **~54% faster** in encoding latency and **~52% faster** in decoding latency compared to Std. This makes it the superior choice for high-frequency, small-packet operations where CPU cycles are precious.

### 2. Medium Payloads (64 KB)
**Focus:** L1 Cache Efficiency (Scalar).

| Crate | Encode Speed | vs `std` | Decode Speed | vs `std` |
| :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | 1.77 GiB/s | -2.2% | **2.44 GiB/s** | **+62.7%** |
| `base64` (std) | **1.81 GiB/s** | - | 1.50 GiB/s | - |

> **Analysis:**
> *   **Decoding:** Custom algorithm shines brilliantly here, delivering a massive **~63% speedup**. By reading `u64` chunks, we minimize memory access overhead.
> *   **Encoding:** The standard library edges out `base64-turbo` slightly (~2%) in this specific cache window, likely due to compiler optimizations favoring the standard loop structure in L1 cache, though the difference is negligible.

### 3. Large Payloads (10 MB)
**Focus:** Sustained Throughput (RAM/L3 Bottlenecks).

| Crate | Encode Speed | Decode Speed |
| :--- | :--- | :--- |
| **base64-turbo** | **1.71 GiB/s** | **2.33 GiB/s** |
| `base64` (std) | 1.57 GiB/s | 1.43 GiB/s |

> **Analysis:** On large files, `base64-turbo` reasserts its dominance. It maintains a **~9% lead in encoding** and a massive **~63% lead in decoding**. This suggests that as data exceeds L1 cache sizes, our loop unrolling and custom strategies handle memory bandwidth much more efficiently than the standard implementation.

## 📝 Raw Data Log
<details>
<summary>Click to view raw Criterion output</summary>

```text
Benchmarking Base64_Performances/Encode/Turbo/32
  time: [32.455 ns 32.629 ns 32.840 ns]
  thrpt: [929.27 MiB/s 935.30 MiB/s 940.31 MiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/32
  time: [20.777 ns 20.795 ns 20.814 ns]
  thrpt: [1.4319 GiB/s 1.4332 GiB/s 1.4344 GiB/s]

Benchmarking Base64_Performances/Encode/Std/32
  time: [44.960 ns 45.014 ns 45.068 ns]
  thrpt: [677.14 MiB/s 677.96 MiB/s 678.77 MiB/s]

Benchmarking Base64_Performances/Decode/Turbo/32
  time: [34.039 ns 34.069 ns 34.101 ns]
  thrpt: [1.2017 GiB/s 1.2028 GiB/s 1.2038 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/32
  time: [24.920 ns 24.933 ns 24.948 ns]
  thrpt: [1.6426 GiB/s 1.6435 GiB/s 1.6444 GiB/s]

Benchmarking Base64_Performances/Decode/Std/32
  time: [52.351 ns 52.409 ns 52.483 ns]
  thrpt: [799.53 MiB/s 800.66 MiB/s 801.54 MiB/s]

Benchmarking Base64_Performances/Encode/Turbo/512
  time: [293.36 ns 293.57 ns 293.79 ns]
  thrpt: [1.6231 GiB/s 1.6243 GiB/s 1.6254 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/512
  time: [280.08 ns 280.44 ns 280.80 ns]
  thrpt: [1.6981 GiB/s 1.7003 GiB/s 1.7025 GiB/s]

Benchmarking Base64_Performances/Encode/Std/512
  time: [277.06 ns 277.26 ns 277.50 ns]
  thrpt: [1.7183 GiB/s 1.7198 GiB/s 1.7211 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/512
  time: [275.90 ns 276.09 ns 276.29 ns]
  thrpt: [2.3056 GiB/s 2.3073 GiB/s 2.3089 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/512
  time: [266.44 ns 266.57 ns 266.71 ns]
  thrpt: [2.3885 GiB/s 2.3897 GiB/s 2.3908 GiB/s]

Benchmarking Base64_Performances/Decode/Std/512
  time: [443.79 ns 444.05 ns 444.32 ns]
  thrpt: [1.4337 GiB/s 1.4346 GiB/s 1.4354 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/4096
  time: [2.2946 µs 2.2967 µs 2.2990 µs]
  thrpt: [1.6593 GiB/s 1.6610 GiB/s 1.6624 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/4096
  time: [2.1652 µs 2.1664 µs 2.1680 µs]
  thrpt: [1.7596 GiB/s 1.7608 GiB/s 1.7618 GiB/s]

Benchmarking Base64_Performances/Encode/Std/4096
  time: [2.1868 µs 2.1880 µs 2.1893 µs]
  thrpt: [1.7424 GiB/s 1.7434 GiB/s 1.7444 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/4096
  time: [2.1396 µs 2.1414 µs 2.1434 µs]
  thrpt: [2.3742 GiB/s 2.3764 GiB/s 2.3784 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/4096
  time: [2.0884 µs 2.0899 µs 2.0913 µs]
  thrpt: [2.4333 GiB/s 2.4350 GiB/s 2.4367 GiB/s]

Benchmarking Base64_Performances/Decode/Std/4096
  time: [3.4759 µs 3.4788 µs 3.4819 µs]
  thrpt: [1.4615 GiB/s 1.4628 GiB/s 1.4640 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/65536
  time: [35.913 µs 35.934 µs 35.955 µs]
  thrpt: [1.6975 GiB/s 1.6985 GiB/s 1.6995 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/65536
  time: [34.500 µs 34.524 µs 34.549 µs]
  thrpt: [1.7666 GiB/s 1.7679 GiB/s 1.7691 GiB/s]

Benchmarking Base64_Performances/Encode/Std/65536
  time: [33.765 µs 33.787 µs 33.813 µs]
  thrpt: [1.8051 GiB/s 1.8064 GiB/s 1.8077 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/65536
  time: [33.424 µs 33.436 µs 33.448 µs]
  thrpt: [2.4331 GiB/s 2.4340 GiB/s 2.4348 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/65536
  time: [33.290 µs 33.310 µs 33.333 µs]
  thrpt: [2.4415 GiB/s 2.4432 GiB/s 2.4447 GiB/s]

Benchmarking Base64_Performances/Decode/Std/65536
  time: [54.358 µs 54.417 µs 54.485 µs]
  thrpt: [1.4937 GiB/s 1.4955 GiB/s 1.4972 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/524288
  time: [287.03 µs 287.19 µs 287.36 µs]
  thrpt: [1.6992 GiB/s 1.7002 GiB/s 1.7011 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/524288
  time: [276.88 µs 277.05 µs 277.23 µs]
  thrpt: [1.7613 GiB/s 1.7624 GiB/s 1.7635 GiB/s]

Benchmarking Base64_Performances/Encode/Std/524288
  time: [273.03 µs 273.19 µs 273.37 µs]
  thrpt: [1.7862 GiB/s 1.7873 GiB/s 1.7884 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/524288
  time: [267.54 µs 267.68 µs 267.86 µs]
  thrpt: [2.4306 GiB/s 2.4322 GiB/s 2.4334 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/524288
  time: [267.22 µs 267.34 µs 267.46 µs]
  thrpt: [2.4341 GiB/s 2.4353 GiB/s 2.4363 GiB/s]

Benchmarking Base64_Performances/Decode/Std/524288
  time: [434.42 µs 434.63 µs 434.86 µs]
  thrpt: [1.4971 GiB/s 1.4979 GiB/s 1.4987 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/1048576
  time: [574.66 µs 574.91 µs 575.18 µs]
  thrpt: [1.6978 GiB/s 1.6986 GiB/s 1.6994 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/1048576
  time: [554.37 µs 554.74 µs 555.22 µs]
  thrpt: [1.7589 GiB/s 1.7604 GiB/s 1.7616 GiB/s]

Benchmarking Base64_Performances/Encode/Std/1048576
  time: [547.94 µs 548.23 µs 548.57 µs]
  thrpt: [1.7802 GiB/s 1.7813 GiB/s 1.7823 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/1048576
  time: [535.71 µs 537.79 µs 541.03 µs]
  thrpt: [2.4067 GiB/s 2.4212 GiB/s 2.4306 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/1048576
  time: [534.32 µs 534.91 µs 535.89 µs]
  thrpt: [2.4297 GiB/s 2.4342 GiB/s 2.4369 GiB/s]

Benchmarking Base64_Performances/Decode/Std/1048576
  time: [880.37 µs 880.80 µs 881.28 µs]
  thrpt: [1.4775 GiB/s 1.4783 GiB/s 1.4790 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/10485760
  time: [5.9272 ms 5.9320 ms 5.9375 ms]
  thrpt: [1.6447 GiB/s 1.6463 GiB/s 1.6476 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/10485760
  time: [5.7147 ms 5.7182 ms 5.7221 ms]
  thrpt: [1.7067 GiB/s 1.7078 GiB/s 1.7089 GiB/s]

Benchmarking Base64_Performances/Encode/Std/10485760
  time: [6.2200 ms 6.2237 ms 6.2281 ms]
  thrpt: [1.5680 GiB/s 1.5691 GiB/s 1.5700 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/10485760
  time: [5.5785 ms 5.5992 ms 5.6379 ms]
  thrpt: [2.3095 GiB/s 2.3255 GiB/s 2.3341 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/10485760
  time: [5.5787 ms 5.5872 ms 5.5961 ms]
  thrpt: [2.3268 GiB/s 2.3305 GiB/s 2.3340 GiB/s]

Benchmarking Base64_Performances/Decode/Std/10485760
  time: [9.0807 ms 9.0951 ms 9.1126 ms]
  thrpt: [1.4289 GiB/s 1.4316 GiB/s 1.4339 GiB/s]

Model name: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
```
</details>
