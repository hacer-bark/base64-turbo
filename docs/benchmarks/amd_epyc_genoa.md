# Benchmark: AMD EPYC Genoa (Zen 4)

**Context:** Benchmarks executed on a high-performance AMD EPYC "Genoa" processor. This represents the current generation of AMD server capability, featuring full AVX512 support.

*   **Processor:** AMD EPYC-Genoa Processor
*   **Architecture:** Zen 4 (AVX512 supported)
*   **OS:** Debian 13

## Performance Snapshot

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_amd.png?raw=true)

**Key Findings:**
1.  **Decode throughput:** `base64-turbo` reaches **~15.4 GiB/s** decoding, 45% higher than `base64-simd`.
2.  **Small-payload latency:** The zero-allocation API (`TurboBuff`) achieves 10.5ns encoding latency, versus ~18.4ns for `base64-simd`.
3.  **Encoding throughput:** On this micro-architecture, `base64-simd` holds a small (~4%) lead in raw streaming encode speed, while `base64-turbo` remains ahead on small-input latency.

## Detailed Results

### 1. Small Payloads (32 Bytes)
**Focus:** Latency & Branch Prediction.
**Crucial for:** HFT Messaging, Authentication Headers.

| Crate | Mode | Encode Latency | Encode Speed | Decode Latency | Decode Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | `TurboBuff` | **10.52 ns** | **2.83 GiB/s** | **12.43 ns** | **3.30 GiB/s** |
| **base64-turbo** | `Standard` | 16.51 ns | 1.80 GiB/s | 19.45 ns | 2.11 GiB/s |
| `base64-simd` | `Standard` | 18.42 ns | 1.62 GiB/s | 15.14 ns | 2.71 GiB/s |
| `base64` (std) | `Standard` | 36.96 ns | 0.82 GiB/s | 31.90 ns | 1.28 GiB/s |

> **Analysis:** `base64-turbo` (TurboBuff) is 1.75x faster than `base64-simd` on encoding latency at this size, consistent with the scalar-fallback path being well-tuned for Zen 4's pipeline.

### 2. Medium Payloads (64 KB)
**Focus:** L1 Cache Saturation & AVX512 Implementation.

| Crate | Encode Speed | vs `base64-simd` | Decode Speed | vs `base64-simd` |
| :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | 11.08 GiB/s | -3.6% | **15.44 GiB/s** | **+45.8%** |
| `base64-simd` | **11.50 GiB/s** | - | 10.59 GiB/s | - |
| `base64` (std) | 2.02 GiB/s | -82% | 2.51 GiB/s | -76% |

> **Analysis:**
> *   **Decoding:** The register-heavy decode approach works well on Zen 4's AVX512 implementation, yielding a 45% lead over `base64-simd`.
> *   **Encoding:** `base64-simd` is marginally faster here (~4%), suggesting Zen 4 may favor its specific shuffle pattern slightly over our port-balancing strategy for encode workloads.

### 3. Large Payloads (10 MB)
**Focus:** RAM Bandwidth & Prefetching.

| Crate | Encode Speed | Decode Speed |
| :--- | :--- | :--- |
| **base64-turbo** | 10.92 GiB/s | **15.16 GiB/s** |
| `base64-simd` | **11.32 GiB/s** | 10.46 GiB/s |
| `base64` (std) | 2.00 GiB/s | 2.45 GiB/s |

> **Analysis:** Results scale linearly from 64KB, indicating no thermal throttling or cache-thrashing effects. `base64-turbo` holds a clear lead for read-heavy (decode) workloads, and is effectively tied with `base64-simd` for write-heavy (encode) workloads.

## Raw Data Log
<details>
<summary>Click to view raw Criterion output</summary>

```text
Benchmarking Base64_Performances/Encode/Turbo/32
  time: [16.442 ns 16.518 ns 16.611 ns]
  thrpt: [1.7942 GiB/s 1.8042 GiB/s 1.8125 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/32
  time: [10.505 ns 10.521 ns 10.538 ns]
  thrpt: [2.8279 GiB/s 2.8327 GiB/s 2.8369 GiB/s]

Benchmarking Base64_Performances/Encode/Std/32
  time: [36.892 ns 36.960 ns 37.040 ns]
  thrpt: [823.90 MiB/s 825.70 MiB/s 827.22 MiB/s]

Benchmarking Base64_Performances/Encode/Simd/32
  time: [18.377 ns 18.420 ns 18.465 ns]
  thrpt: [1.6140 GiB/s 1.6180 GiB/s 1.6217 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/32
  time: [19.402 ns 19.449 ns 19.500 ns]
  thrpt: [2.1014 GiB/s 2.1070 GiB/s 2.1121 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/32
  time: [12.413 ns 12.432 ns 12.450 ns]
  thrpt: [3.2913 GiB/s 3.2963 GiB/s 3.3011 GiB/s]

Benchmarking Base64_Performances/Decode/Std/32
  time: [31.805 ns 31.898 ns 32.018 ns]
  thrpt: [1.2798 GiB/s 1.2847 GiB/s 1.2884 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/32
  time: [15.104 ns 15.143 ns 15.190 ns]
  thrpt: [2.6977 GiB/s 2.7060 GiB/s 2.7130 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/512
  time: [67.993 ns 68.077 ns 68.175 ns]
  thrpt: [6.9943 GiB/s 7.0043 GiB/s 7.0131 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/512
  time: [62.199 ns 62.261 ns 62.326 ns]
  thrpt: [7.6507 GiB/s 7.6586 GiB/s 7.6663 GiB/s]

Benchmarking Base64_Performances/Encode/Std/512
  time: [248.43 ns 250.07 ns 251.89 ns]
  thrpt: [1.8930 GiB/s 1.9068 GiB/s 1.9194 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/512
  time: [59.898 ns 60.113 ns 60.353 ns]
  thrpt: [7.9008 GiB/s 7.9324 GiB/s 7.9608 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/512
  time: [68.340 ns 68.402 ns 68.470 ns]
  thrpt: [9.3038 GiB/s 9.3130 GiB/s 9.3213 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/512
  time: [62.089 ns 62.186 ns 62.291 ns]
  thrpt: [10.227 GiB/s 10.244 GiB/s 10.260 GiB/s]

Benchmarking Base64_Performances/Decode/Std/512
  time: [269.46 ns 270.75 ns 272.19 ns]
  thrpt: [2.3404 GiB/s 2.3528 GiB/s 2.3641 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/512
  time: [69.179 ns 69.467 ns 69.836 ns]
  thrpt: [9.1217 GiB/s 9.1702 GiB/s 9.2083 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/4096
  time: [380.73 ns 381.11 ns 381.52 ns]
  thrpt: [9.9986 GiB/s 10.009 GiB/s 10.020 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/4096
  time: [343.11 ns 343.50 ns 343.92 ns]
  thrpt: [11.092 GiB/s 11.106 GiB/s 11.118 GiB/s]

Benchmarking Base64_Performances/Encode/Std/4096
  time: [1.8963 µs 1.9023 µs 1.9098 µs]
  thrpt: [1.9974 GiB/s 2.0053 GiB/s 2.0116 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/4096
  time: [375.10 ns 375.65 ns 376.29 ns]
  thrpt: [10.138 GiB/s 10.155 GiB/s 10.170 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/4096
  time: [380.79 ns 382.72 ns 384.61 ns]
  thrpt: [13.231 GiB/s 13.296 GiB/s 13.364 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/4096
  time: [337.44 ns 337.76 ns 338.08 ns]
  thrpt: [15.052 GiB/s 15.066 GiB/s 15.080 GiB/s]

Benchmarking Base64_Performances/Decode/Std/4096
  time: [2.0792 µs 2.0847 µs 2.0915 µs]
  thrpt: [2.4331 GiB/s 2.4410 GiB/s 2.4474 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/4096
  time: [532.48 ns 534.57 ns 536.92 ns]
  thrpt: [9.4776 GiB/s 9.5193 GiB/s 9.5567 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/65536
  time: [5.5656 µs 5.5700 µs 5.5747 µs]
  thrpt: [10.949 GiB/s 10.958 GiB/s 10.967 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/65536
  time: [5.5042 µs 5.5070 µs 5.5100 µs]
  thrpt: [11.077 GiB/s 11.083 GiB/s 11.089 GiB/s]

Benchmarking Base64_Performances/Encode/Std/65536
  time: [30.070 µs 30.143 µs 30.228 µs]
  thrpt: [2.0192 GiB/s 2.0248 GiB/s 2.0298 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/65536
  time: [5.2970 µs 5.3060 µs 5.3166 µs]
  thrpt: [11.480 GiB/s 11.503 GiB/s 11.523 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/65536
  time: [5.3141 µs 5.3172 µs 5.3202 µs]
  thrpt: [15.297 GiB/s 15.306 GiB/s 15.314 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/65536
  time: [5.2656 µs 5.2687 µs 5.2721 µs]
  thrpt: [15.436 GiB/s 15.446 GiB/s 15.455 GiB/s]

Benchmarking Base64_Performances/Decode/Std/65536
  time: [32.306 µs 32.404 µs 32.528 µs]
  thrpt: [2.5019 GiB/s 2.5115 GiB/s 2.5191 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/65536
  time: [7.6700 µs 7.6866 µs 7.7060 µs]
  thrpt: [10.561 GiB/s 10.588 GiB/s 10.610 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/524288
  time: [44.727 µs 44.758 µs 44.791 µs]
  thrpt: [10.901 GiB/s 10.909 GiB/s 10.917 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/524288
  time: [44.645 µs 44.674 µs 44.706 µs]
  thrpt: [10.922 GiB/s 10.930 GiB/s 10.937 GiB/s]

Benchmarking Base64_Performances/Encode/Std/524288
  time: [241.80 µs 242.46 µs 243.22 µs]
  thrpt: [2.0076 GiB/s 2.0139 GiB/s 2.0194 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/524288
  time: [43.417 µs 43.539 µs 43.678 µs]
  thrpt: [11.179 GiB/s 11.215 GiB/s 11.246 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/524288
  time: [42.829 µs 42.857 µs 42.883 µs]
  thrpt: [15.182 GiB/s 15.191 GiB/s 15.201 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/524288
  time: [42.660 µs 42.684 µs 42.709 µs]
  thrpt: [15.244 GiB/s 15.253 GiB/s 15.261 GiB/s]

Benchmarking Base64_Performances/Decode/Std/524288
  time: [262.20 µs 262.93 µs 263.74 µs]
  thrpt: [2.4685 GiB/s 2.4761 GiB/s 2.4830 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/524288
  time: [62.420 µs 62.612 µs 62.804 µs]
  thrpt: [10.366 GiB/s 10.398 GiB/s 10.430 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/1048576
  time: [88.932 µs 88.994 µs 89.053 µs]
  thrpt: [10.966 GiB/s 10.973 GiB/s 10.981 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/1048576
  time: [88.856 µs 88.963 µs 89.068 µs]
  thrpt: [10.964 GiB/s 10.977 GiB/s 10.990 GiB/s]

Benchmarking Base64_Performances/Encode/Std/1048576
  time: [481.05 µs 482.27 µs 483.89 µs]
  thrpt: [2.0181 GiB/s 2.0249 GiB/s 2.0301 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/1048576
  time: [85.682 µs 85.879 µs 86.076 µs]
  thrpt: [11.345 GiB/s 11.371 GiB/s 11.398 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/1048576
  time: [85.683 µs 85.771 µs 85.846 µs]
  thrpt: [15.168 GiB/s 15.181 GiB/s 15.197 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/1048576
  time: [85.623 µs 86.077 µs 86.567 µs]
  thrpt: [15.041 GiB/s 15.127 GiB/s 15.207 GiB/s]

Benchmarking Base64_Performances/Decode/Std/1048576
  time: [523.52 µs 526.57 µs 530.25 µs]
  thrpt: [2.4556 GiB/s 2.4728 GiB/s 2.4872 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/1048576
  time: [124.54 µs 125.05 µs 125.65 µs]
  thrpt: [10.362 GiB/s 10.413 GiB/s 10.455 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/10485760
  time: [889.98 µs 891.24 µs 892.66 µs]
  thrpt: [10.940 GiB/s 10.957 GiB/s 10.973 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/10485760
  time: [892.20 µs 893.83 µs 895.68 µs]
  thrpt: [10.903 GiB/s 10.926 GiB/s 10.946 GiB/s]

Benchmarking Base64_Performances/Encode/Std/10485760
  time: [4.8412 ms 4.8607 ms 4.8815 ms]
  thrpt: [2.0005 GiB/s 2.0091 GiB/s 2.0172 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/10485760
  time: [859.86 µs 862.40 µs 865.87 µs]
  thrpt: [11.278 GiB/s 11.324 GiB/s 11.357 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/10485760
  time: [858.71 µs 859.93 µs 861.38 µs]
  thrpt: [15.116 GiB/s 15.142 GiB/s 15.163 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/10485760
  time: [857.84 µs 858.85 µs 859.93 µs]
  thrpt: [15.142 GiB/s 15.161 GiB/s 15.179 GiB/s]

Benchmarking Base64_Performances/Decode/Std/10485760
  time: [5.2860 ms 5.3130 ms 5.3451 ms]
  thrpt: [2.4360 GiB/s 2.4508 GiB/s 2.4633 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/10485760
  time: [1.2425 ms 1.2446 ms 1.2468 ms]
  thrpt: [10.443 GiB/s 10.462 GiB/s 10.480 GiB/s]

Model name: AMD EPYC-Genoa Processor
```
</details>
