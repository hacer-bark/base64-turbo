# Benchmark: Intel Xeon Platinum 8488C (AWS)

**Context:** This benchmark represents a current-generation cloud instance type. The tests were executed on an AWS `c7i.large` instance powered by Intel Sapphire Rapids.

*   **Processor:** Intel(R) Xeon(R) Platinum 8488C
*   **Instruction Set:** AVX512 (Enabled)
*   **Environment:** AWS Nitro Hypervisor
*   **OS:** Ubuntu latest (Minimal, `cargo` only)

## Performance Snapshot

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_intel.png?raw=true)

**Key Findings:**
1.  **Decode throughput:** `base64-turbo` reaches 21.04 GiB/s decoding, more than 2x the `base64` (std) baseline.
2.  **Low latency:** For small inputs (32B), the zero-allocation API (`TurboBuff`) offers ~10ns latency, relevant for HFT messaging.
3.  **Memory saturation:** On large payloads (10MB+), the library approaches the memory bandwidth ceiling of the test instance.

## Detailed Results

### 1. Small Payloads (32 Bytes)
**Focus:** Latency & Branch Prediction.
**Crucial for:** FIX Messages, API Keys, Headers.

| Crate | Mode | Encode Latency | Encode Throughput | Decode Latency | Decode Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | `TurboBuff` | **10.18 ns** | **2.93 GiB/s** | **13.49 ns** | **3.04 GiB/s** |
| **base64-turbo** | `Standard` | 16.61 ns | 1.79 GiB/s | 19.12 ns | 2.14 GiB/s |
| `base64-simd` | `Standard` | 18.12 ns | 1.64 GiB/s | 15.11 ns | 2.71 GiB/s |
| `base64` (std) | `Standard` | 56.51 ns | 0.54 GiB/s | 49.65 ns | 0.85 GiB/s |

> **Analysis:** `base64-turbo` (TurboBuff) is 1.8x faster than `base64-simd` on encoding latency at this size, reflecting the low overhead of the runtime feature detection and hot-path design.

### 2. Medium Payloads (64 KB)
**Focus:** L1/L2 Cache Saturation & AVX Efficiency.
**Crucial for:** JSON blobs, Binary responses, Images.

| Crate | Encode Speed | vs `base64-simd` | Decode Speed | vs `base64-simd` |
| :--- | :--- | :--- | :--- | :--- |
| **base64-turbo** | **12.48 GiB/s** | **+18.1%** | **21.04 GiB/s** | **+110.1%** |
| `base64-simd` | 10.57 GiB/s | - | 10.01 GiB/s | - |
| `base64` (std) | 2.42 GiB/s | -77% | 2.78 GiB/s | -72% |

> **Analysis:** This is where AVX512 provides the largest benefit.
> *   **Decoding:** The 2.1x speedup comes from utilizing the full width of AVX512 registers for the decode logic.
> *   **Encoding:** The ~18% gain is largely due to better port saturation from offloading shuffle operations onto other execution ports.

### 3. Large Payloads (10 MB)
**Focus:** RAM Bandwidth & Prefetching.
**Crucial for:** File uploads, Video streams, Backups.

| Crate | Encode Speed | Decode Speed |
| :--- | :--- | :--- |
| **base64-turbo** | **11.89 GiB/s** | **15.72 GiB/s** |
| `base64-simd` | 10.27 GiB/s | 9.98 GiB/s |
| `base64` (std) | 2.16 GiB/s | 2.60 GiB/s |

> **Analysis:** Even when bottlenecked by main system RAM, `base64-turbo` maintains a significant lead in decoding (+57%), consistent with the loop unrolling and prefetching strategies effectively hiding memory latency at this scale.

## Raw Data Log
<details>
<summary>Click to view raw Criterion output</summary>

```text
Benchmarking Base64_Performances/Encode/Turbo/32
  time: [16.556 ns 16.608 ns 16.659 ns]
  thrpt: [1.7889 GiB/s 1.7944 GiB/s 1.8001 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/32
  time: [10.139 ns 10.184 ns 10.227 ns]
  thrpt: [2.9140 GiB/s 2.9265 GiB/s 2.9393 GiB/s]

Benchmarking Base64_Performances/Encode/Std/32
  time: [56.361 ns 56.514 ns 56.664 ns]
  thrpt: [538.57 MiB/s 540.00 MiB/s 541.46 MiB/s]

Benchmarking Base64_Performances/Encode/Simd/32
  time: [18.080 ns 18.119 ns 18.160 ns]
  thrpt: [1.6411 GiB/s 1.6448 GiB/s 1.6484 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/32
  time: [19.068 ns 19.119 ns 19.172 ns]
  thrpt: [2.1374 GiB/s 2.1433 GiB/s 2.1491 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/32
  time: [13.451 ns 13.490 ns 13.528 ns]
  thrpt: [3.0291 GiB/s 3.0377 GiB/s 3.0464 GiB/s]

Benchmarking Base64_Performances/Decode/Std/32
  time: [49.491 ns 49.649 ns 49.803 ns]
  thrpt: [842.56 MiB/s 845.16 MiB/s 847.87 MiB/s]

Benchmarking Base64_Performances/Decode/Simd/32
  time: [15.073 ns 15.108 ns 15.144 ns]
  thrpt: [2.7059 GiB/s 2.7124 GiB/s 2.7187 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/512
  time: [58.685 ns 58.954 ns 59.241 ns]
  thrpt: [8.0491 GiB/s 8.0883 GiB/s 8.1254 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/512
  time: [53.162 ns 53.309 ns 53.458 ns]
  thrpt: [8.9198 GiB/s 8.9448 GiB/s 8.9696 GiB/s]

Benchmarking Base64_Performances/Encode/Std/512
  time: [239.67 ns 240.30 ns 240.96 ns]
  thrpt: [1.9789 GiB/s 1.9843 GiB/s 1.9896 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/512
  time: [56.475 ns 56.599 ns 56.730 ns]
  thrpt: [8.4054 GiB/s 8.4248 GiB/s 8.4433 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/512
  time: [53.100 ns 53.278 ns 53.469 ns]
  thrpt: [11.914 GiB/s 11.957 GiB/s 11.997 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/512
  time: [45.482 ns 45.651 ns 45.833 ns]
  thrpt: [13.899 GiB/s 13.954 GiB/s 14.006 GiB/s]

Benchmarking Base64_Performances/Decode/Std/512
  time: [259.82 ns 260.41 ns 261.03 ns]
  thrpt: [2.4405 GiB/s 2.4462 GiB/s 2.4518 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/512
  time: [71.091 ns 71.256 ns 71.425 ns]
  thrpt: [8.9188 GiB/s 8.9399 GiB/s 8.9607 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/4096
  time: [325.93 ns 327.28 ns 328.72 ns]
  thrpt: [11.605 GiB/s 11.656 GiB/s 11.704 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/4096
  time: [304.32 ns 306.57 ns 308.99 ns]
  thrpt: [12.346 GiB/s 12.443 GiB/s 12.535 GiB/s]

Benchmarking Base64_Performances/Encode/Std/4096
  time: [1.5540 µs 1.5577 µs 1.5616 µs]
  thrpt: [2.4428 GiB/s 2.4490 GiB/s 2.4548 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/4096
  time: [394.13 ns 394.90 ns 395.69 ns]
  thrpt: [9.6406 GiB/s 9.6600 GiB/s 9.6788 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/4096
  time: [280.57 ns 281.44 ns 282.36 ns]
  thrpt: [18.022 GiB/s 18.081 GiB/s 18.137 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/4096
  time: [247.08 ns 249.07 ns 251.16 ns]
  thrpt: [20.261 GiB/s 20.431 GiB/s 20.596 GiB/s]

Benchmarking Base64_Performances/Decode/Std/4096
  time: [1.8031 µs 1.8076 µs 1.8123 µs]
  thrpt: [2.8078 GiB/s 2.8151 GiB/s 2.8222 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/4096
  time: [534.95 ns 536.34 ns 537.74 ns]
  thrpt: [9.4632 GiB/s 9.4879 GiB/s 9.5126 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/65536
  time: [4.8585 µs 4.8892 µs 4.9224 µs]
  thrpt: [12.399 GiB/s 12.484 GiB/s 12.563 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/65536
  time: [4.6127 µs 4.6319 µs 4.6520 µs]
  thrpt: [13.120 GiB/s 13.177 GiB/s 13.232 GiB/s]

Benchmarking Base64_Performances/Encode/Std/65536
  time: [25.193 µs 25.260 µs 25.331 µs]
  thrpt: [2.4095 GiB/s 2.4163 GiB/s 2.4227 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/65536
  time: [5.7586 µs 5.7754 µs 5.7921 µs]
  thrpt: [10.538 GiB/s 10.568 GiB/s 10.599 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/65536
  time: [3.8486 µs 3.8689 µs 3.8908 µs]
  thrpt: [20.917 GiB/s 21.035 GiB/s 21.146 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/65536
  time: [3.8721 µs 3.9016 µs 3.9311 µs]
  thrpt: [20.703 GiB/s 20.859 GiB/s 21.018 GiB/s]

Benchmarking Base64_Performances/Decode/Std/65536
  time: [29.244 µs 29.315 µs 29.386 µs]
  thrpt: [2.7694 GiB/s 2.7762 GiB/s 2.7829 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/65536
  time: [8.0944 µs 8.1265 µs 8.1606 µs]
  thrpt: [9.9726 GiB/s 10.014 GiB/s 10.054 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/524288
  time: [36.667 µs 36.823 µs 36.992 µs]
  thrpt: [13.200 GiB/s 13.260 GiB/s 13.317 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/524288
  time: [37.962 µs 38.263 µs 38.581 µs]
  thrpt: [12.656 GiB/s 12.761 GiB/s 12.862 GiB/s]

Benchmarking Base64_Performances/Encode/Std/524288
  time: [205.36 µs 206.03 µs 206.73 µs]
  thrpt: [2.3619 GiB/s 2.3699 GiB/s 2.3777 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/524288
  time: [46.263 µs 46.398 µs 46.538 µs]
  thrpt: [10.492 GiB/s 10.524 GiB/s 10.555 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/524288
  time: [31.825 µs 32.037 µs 32.265 µs]
  thrpt: [20.178 GiB/s 20.322 GiB/s 20.457 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/524288
  time: [31.747 µs 31.959 µs 32.180 µs]
  thrpt: [20.232 GiB/s 20.371 GiB/s 20.507 GiB/s]

Benchmarking Base64_Performances/Decode/Std/524288
  time: [240.14 µs 240.67 µs 241.21 µs]
  thrpt: [2.6990 GiB/s 2.7051 GiB/s 2.7110 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/524288
  time: [63.604 µs 63.756 µs 63.916 µs]
  thrpt: [10.186 GiB/s 10.211 GiB/s 10.236 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/1048576
  time: [77.163 µs 77.937 µs 78.739 µs]
  thrpt: [12.403 GiB/s 12.530 GiB/s 12.656 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/1048576
  time: [75.110 µs 75.425 µs 75.743 µs]
  thrpt: [12.893 GiB/s 12.947 GiB/s 13.002 GiB/s]

Benchmarking Base64_Performances/Encode/Std/1048576
  time: [415.67 µs 416.69 µs 417.67 µs]
  thrpt: [2.3381 GiB/s 2.3436 GiB/s 2.3493 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/1048576
  time: [93.068 µs 93.258 µs 93.463 µs]
  thrpt: [10.449 GiB/s 10.472 GiB/s 10.493 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/1048576
  time: [67.230 µs 67.737 µs 68.169 µs]
  thrpt: [19.101 GiB/s 19.223 GiB/s 19.368 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/1048576
  time: [66.379 µs 66.725 µs 67.094 µs]
  thrpt: [19.407 GiB/s 19.514 GiB/s 19.616 GiB/s]

Benchmarking Base64_Performances/Decode/Std/1048576
  time: [476.81 µs 478.39 µs 480.27 µs]
  thrpt: [2.7111 GiB/s 2.7218 GiB/s 2.7308 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/1048576
  time: [128.75 µs 129.14 µs 129.52 µs]
  thrpt: [10.053 GiB/s 10.083 GiB/s 10.113 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/10485760
  time: [818.61 µs 821.57 µs 825.59 µs]
  thrpt: [11.829 GiB/s 11.887 GiB/s 11.930 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/10485760
  time: [834.83 µs 838.21 µs 841.07 µs]
  thrpt: [11.611 GiB/s 11.651 GiB/s 11.698 GiB/s]

Benchmarking Base64_Performances/Encode/Std/10485760
  time: [4.5108 ms 4.5299 ms 4.5560 ms]
  thrpt: [2.1435 GiB/s 2.1558 GiB/s 2.1650 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/10485760
  time: [940.42 µs 950.75 µs 960.55 µs]
  thrpt: [10.167 GiB/s 10.271 GiB/s 10.384 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/10485760
  time: [826.48 µs 828.31 µs 830.28 µs]
  thrpt: [15.682 GiB/s 15.720 GiB/s 15.755 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/10485760
  time: [827.66 µs 829.76 µs 832.17 µs]
  thrpt: [15.647 GiB/s 15.692 GiB/s 15.732 GiB/s]

Benchmarking Base64_Performances/Decode/Std/10485760
  time: [4.9892 ms 5.0094 ms 5.0337 ms]
  thrpt: [2.5868 GiB/s 2.5993 GiB/s 2.6098 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/10485760
  time: [1.2928 ms 1.3046 ms 1.3189 ms]
  thrpt: [9.8727 GiB/s 9.9810 GiB/s 10.072 GiB/s]

Model name: Intel(R) Xeon(R) Platinum 8488C
```
</details>
