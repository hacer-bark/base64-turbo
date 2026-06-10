# 💻 Benchmark: Intel i7-8750H (AVX2)

**Context:** This benchmark runs on a standard consumer-grade laptop CPU (Intel Coffee Lake). It demonstrates how `base64-turbo` optimizes for the most common instruction set currently in use (AVX2).

*   **Processor:** Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
*   **Instruction Set:** **AVX2** (Enabled)
*   **Competitor:** `base64-simd` (The current high-performance standard)

## 📈 Performance Snapshot

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_i7_avx2.png?raw=true)

**Key Findings:**
1.  **Decoding Domination:** `base64-turbo` crushes the competition in decoding, reaching **~12.1 GiB/s** compared to `base64-simd`'s 8.0 GiB/s. That is a **+51% speedup**.
2.  **Latency Leader:** For small payloads (32B), the zero-allocation API is **~1.5x faster** (17ns vs 26ns), reducing overhead for micro-services.
3.  **Efficiency:** Even on older hardware, the library extracts significantly more throughput per clock cycle than competitors.

## 🏎️ Detailed Results

### 1. Small Payloads (32 Bytes)
**Focus:** API Overhead & Branch Prediction.

| Crate | Mode | Encode Latency | Encode Speed | Decode Latency | Decode Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | `TurboBuff` | **17.33 ns** | **1.72 GiB/s** | **18.05 ns** | **2.27 GiB/s** |
| **base64-turbo** | `Standard` | 24.96 ns | 1.19 GiB/s | 27.38 ns | 1.50 GiB/s |
| `base64-simd` | `Standard` | 26.78 ns | 1.11 GiB/s | 23.74 ns | 1.73 GiB/s |
| `base64` (std) | `Standard` | 49.62 ns | 0.62 GiB/s | 51.45 ns | 0.82 GiB/s |

> **Analysis:** `base64-turbo` (TurboBuff) is the clear winner here. By removing allocation overhead and utilizing efficient scalar fallback for small inputs, we achieve sub-20ns latency, which is critical for high-frequency messaging.

### 2. Medium Payloads (64 KB)
**Focus:** AVX2 Implementation & L1 Cache.

| Crate | Encode Speed | vs `base64-simd` | Decode Speed | vs `base64-simd` |
| :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | **8.93 GiB/s** | **+3.8%** | **12.14 GiB/s** | **+51.7%** |
| `base64-simd` | 8.60 GiB/s | - | 8.00 GiB/s | - |
| `base64` (std) | 1.66 GiB/s | -80% | 1.66 GiB/s | -79% |

> **Analysis:**
> *   **Decoding:** This is a massive gap. `base64-turbo`'s "Lane Stitching" technique allows it to consume data much faster than standard AVX2 implementations.
> *   **Encoding:** The gap is tighter (~4%), but `base64-turbo` consistently holds the lead, proving better pipeline saturation.

### 3. Large Payloads (10 MB)
**Focus:** Memory Bandwidth & Thermal Throttling.

| Crate | Encode Speed | Decode Speed |
| :--- | :--- | :--- |
| **base64-turbo** | **6.69 GiB/s** | **8.25 GiB/s** |
| `base64-simd` | 6.45 GiB/s | 6.48 GiB/s |
| `base64` (std) | 1.49 GiB/s | 1.59 GiB/s |

> **Analysis:** Even when limited by the laptop's RAM bandwidth and thermal limits, `base64-turbo` maintains a **27% lead** in decoding. This efficiency implies that for the same amount of data, `base64-turbo` keeps the CPU active for less time, potentially saving battery life on some devices.

## 📝 Raw Data Log
<details>
<summary>Click to view raw Criterion output</summary>

```text
Benchmarking Base64_Performances/Encode/Turbo/32
  time: [24.717 ns 24.958 ns 25.237 ns]
  thrpt: [1.1809 GiB/s 1.1941 GiB/s 1.2057 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/32
  time: [17.261 ns 17.331 ns 17.405 ns]
  thrpt: [1.7123 GiB/s 1.7196 GiB/s 1.7266 GiB/s]

Benchmarking Base64_Performances/Encode/Std/32
  time: [49.324 ns 49.615 ns 49.960 ns]
  thrpt: [610.84 MiB/s 615.08 MiB/s 618.71 MiB/s]

Benchmarking Base64_Performances/Encode/Simd/32
  time: [26.733 ns 26.779 ns 26.827 ns]
  thrpt: [1.1109 GiB/s 1.1129 GiB/s 1.1148 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/32
  time: [27.105 ns 27.383 ns 27.752 ns]
  thrpt: [1.4766 GiB/s 1.4965 GiB/s 1.5118 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/32
  time: [18.032 ns 18.051 ns 18.069 ns]
  thrpt: [2.2679 GiB/s 2.2702 GiB/s 2.2725 GiB/s]

Benchmarking Base64_Performances/Decode/Std/32
  time: [51.385 ns 51.451 ns 51.523 ns]
  thrpt: [814.42 MiB/s 815.57 MiB/s 816.61 MiB/s]

Benchmarking Base64_Performances/Decode/Simd/32
  time: [23.717 ns 23.744 ns 23.771 ns]
  thrpt: [1.7239 GiB/s 1.7258 GiB/s 1.7278 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/512
  time: [74.458 ns 74.595 ns 74.766 ns]
  thrpt: [6.3777 GiB/s 6.3923 GiB/s 6.4041 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/512
  time: [65.755 ns 65.825 ns 65.901 ns]
  thrpt: [7.2356 GiB/s 7.2440 GiB/s 7.2517 GiB/s]

Benchmarking Base64_Performances/Encode/Std/512
  time: [310.77 ns 311.20 ns 311.72 ns]
  thrpt: [1.5297 GiB/s 1.5322 GiB/s 1.5344 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/512
  time: [76.429 ns 76.522 ns 76.627 ns]
  thrpt: [6.2228 GiB/s 6.2313 GiB/s 6.2389 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/512
  time: [74.810 ns 74.899 ns 74.996 ns]
  thrpt: [8.4942 GiB/s 8.5051 GiB/s 8.5153 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/512
  time: [65.476 ns 65.765 ns 66.137 ns]
  thrpt: [9.6319 GiB/s 9.6864 GiB/s 9.7292 GiB/s]

Benchmarking Base64_Performances/Decode/Std/512
  time: [414.59 ns 416.22 ns 418.41 ns]
  thrpt: [1.5225 GiB/s 1.5305 GiB/s 1.5365 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/512
  time: [90.827 ns 91.007 ns 91.200 ns]
  thrpt: [6.9849 GiB/s 6.9998 GiB/s 7.0136 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/4096
  time: [490.31 ns 491.48 ns 492.72 ns]
  thrpt: [7.7421 GiB/s 7.7617 GiB/s 7.7801 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/4096
  time: [438.24 ns 439.15 ns 440.09 ns]
  thrpt: [8.6680 GiB/s 8.6866 GiB/s 8.7046 GiB/s]

Benchmarking Base64_Performances/Encode/Std/4096
  time: [2.3964 µs 2.4030 µs 2.4098 µs]
  thrpt: [1.5830 GiB/s 1.5875 GiB/s 1.5918 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/4096
  time: [502.63 ns 503.46 ns 504.36 ns]
  thrpt: [7.5635 GiB/s 7.5770 GiB/s 7.5895 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/4096
  time: [488.18 ns 488.76 ns 489.34 ns]
  thrpt: [10.399 GiB/s 10.412 GiB/s 10.424 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/4096
  time: [441.06 ns 442.03 ns 443.03 ns]
  thrpt: [11.486 GiB/s 11.512 GiB/s 11.537 GiB/s]

Benchmarking Base64_Performances/Decode/Std/4096
  time: [3.2332 µs 3.2488 µs 3.2663 µs]
  thrpt: [1.5579 GiB/s 1.5664 GiB/s 1.5739 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/4096
  time: [666.36 ns 667.38 ns 668.49 ns]
  thrpt: [7.6123 GiB/s 7.6250 GiB/s 7.6366 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/65536
  time: [6.8718 µs 6.8818 µs 6.8916 µs]
  thrpt: [8.8564 GiB/s 8.8690 GiB/s 8.8820 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/65536
  time: [6.8276 µs 6.8351 µs 6.8427 µs]
  thrpt: [8.9197 GiB/s 8.9296 GiB/s 8.9395 GiB/s]

Benchmarking Base64_Performances/Encode/Std/65536
  time: [36.702 µs 36.741 µs 36.779 µs]
  thrpt: [1.6595 GiB/s 1.6612 GiB/s 1.6630 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/65536
  time: [7.0851 µs 7.0980 µs 7.1112 µs]
  thrpt: [8.5830 GiB/s 8.5990 GiB/s 8.6146 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/65536
  time: [6.8517 µs 6.8584 µs 6.8654 µs]
  thrpt: [11.854 GiB/s 11.866 GiB/s 11.878 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/65536
  time: [6.6966 µs 6.7046 µs 6.7124 µs]
  thrpt: [12.124 GiB/s 12.138 GiB/s 12.153 GiB/s]

Benchmarking Base64_Performances/Decode/Std/65536
  time: [49.093 µs 49.149 µs 49.204 µs]
  thrpt: [1.6540 GiB/s 1.6558 GiB/s 1.6577 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/65536
  time: [10.164 µs 10.176 µs 10.189 µs]
  thrpt: [7.9874 GiB/s 7.9974 GiB/s 8.0071 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/524288
  time: [55.384 µs 55.459 µs 55.534 µs]
  thrpt: [8.7925 GiB/s 8.8044 GiB/s 8.8163 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/524288
  time: [55.024 µs 55.093 µs 55.164 µs]
  thrpt: [8.8515 GiB/s 8.8628 GiB/s 8.8740 GiB/s]

Benchmarking Base64_Performances/Encode/Std/524288
  time: [295.58 µs 297.34 µs 299.77 µs]
  thrpt: [1.6288 GiB/s 1.6422 GiB/s 1.6519 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/524288
  time: [58.199 µs 58.442 µs 58.709 µs]
  thrpt: [8.3170 GiB/s 8.3549 GiB/s 8.3898 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/524288
  time: [56.501 µs 56.556 µs 56.611 µs]
  thrpt: [11.500 GiB/s 11.512 GiB/s 11.523 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/524288
  time: [56.532 µs 56.783 µs 57.099 µs]
  thrpt: [11.402 GiB/s 11.465 GiB/s 11.516 GiB/s]

Benchmarking Base64_Performances/Decode/Std/524288
  time: [391.63 µs 391.99 µs 392.37 µs]
  thrpt: [1.6593 GiB/s 1.6609 GiB/s 1.6624 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/524288
  time: [82.073 µs 82.165 µs 82.259 µs]
  thrpt: [7.9145 GiB/s 7.9236 GiB/s 7.9325 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/1048576
  time: [110.24 µs 110.43 µs 110.63 µs]
  thrpt: [8.8276 GiB/s 8.8435 GiB/s 8.8585 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/1048576
  time: [110.13 µs 110.28 µs 110.45 µs]
  thrpt: [8.8417 GiB/s 8.8549 GiB/s 8.8670 GiB/s]

Benchmarking Base64_Performances/Encode/Std/1048576
  time: [585.31 µs 586.04 µs 586.79 µs]
  thrpt: [1.6643 GiB/s 1.6664 GiB/s 1.6685 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/1048576
  time: [112.77 µs 112.93 µs 113.10 µs]
  thrpt: [8.6345 GiB/s 8.6476 GiB/s 8.6600 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/1048576
  time: [110.88 µs 111.07 µs 111.28 µs]
  thrpt: [11.701 GiB/s 11.723 GiB/s 11.743 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/1048576
  time: [111.81 µs 112.29 µs 112.97 µs]
  thrpt: [11.526 GiB/s 11.596 GiB/s 11.645 GiB/s]

Benchmarking Base64_Performances/Decode/Std/1048576
  time: [801.54 µs 809.95 µs 821.54 µs]
  thrpt: [1.5849 GiB/s 1.6076 GiB/s 1.6245 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/1048576
  time: [163.62 µs 163.87 µs 164.16 µs]
  thrpt: [7.9320 GiB/s 7.9458 GiB/s 7.9581 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/10485760
  time: [1.4568 ms 1.4588 ms 1.4608 ms]
  thrpt: [6.6851 GiB/s 6.6943 GiB/s 6.7033 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/10485760
  time: [1.4572 ms 1.4591 ms 1.4611 ms]
  thrpt: [6.6836 GiB/s 6.6928 GiB/s 6.7015 GiB/s]

Benchmarking Base64_Performances/Encode/Std/10485760
  time: [6.5481 ms 6.5570 ms 6.5670 ms]
  thrpt: [1.4871 GiB/s 1.4893 GiB/s 1.4914 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/10485760
  time: [1.5120 ms 1.5137 ms 1.5155 ms]
  thrpt: [6.4437 GiB/s 6.4513 GiB/s 6.4587 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/10485760
  time: [1.5901 ms 1.6067 ms 1.6241 ms]
  thrpt: [8.0174 GiB/s 8.1041 GiB/s 8.1889 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/10485760
  time: [1.5770 ms 1.5789 ms 1.5812 ms]
  thrpt: [8.2349 GiB/s 8.2467 GiB/s 8.2567 GiB/s]

Benchmarking Base64_Performances/Decode/Std/10485760
  time: [8.1683 ms 8.1779 ms 8.1894 ms]
  thrpt: [1.5900 GiB/s 1.5922 GiB/s 1.5941 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/10485760
  time: [2.0058 ms 2.0079 ms 2.0099 ms]
  thrpt: [6.4783 GiB/s 6.4847 GiB/s 6.4916 GiB/s]

Model name: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
```
</details>
