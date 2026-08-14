<div align="center">
  <h1>Base64 Turbo</h1>
  <p><strong>A Rust Base64 codec that peaks past 100 GiB/s, with its <code>unsafe</code> SIMD checked by a model checker, not just by review.</strong></p>

  [![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg?style=for-the-badge&color=fc8d62)](https://crates.io/crates/base64-turbo)
  [![License](https://img.shields.io/crates/l/base64-turbo.svg?style=for-the-badge&color=8da0cb)](https://crates.io/crates/base64-turbo)
  [![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/verification.yml?label=Kani%20Verified&style=for-the-badge&color=e78ac3)](https://github.com/hacer-bark/base64-turbo/actions/workflows/verification.yml)
  [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified&style=for-the-badge&color=66c2a5)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)
</div>

<br/>

`base64-turbo` is built for high-throughput systems where CPU cycles are scarce and Undefined Behavior is unacceptable. It picks the **best kernel available at runtime**, without ever giving up on portability:

*   **x86_64:** AVX512 (incl. a VBMI fast path) or AVX2, via runtime CPU detection.
*   **ARM (aarch64):** NEON, via compile-time dispatch — no detection overhead.
*   **Other:** an optimized table-driven scalar kernel, in 100% safe Rust.

<img alt="Base64 throughput by payload size on AWS c8a.large (AMD EPYC 9R45) — base64-turbo peaks above 100 GiB/s for both encode and decode" src="benches/results/throughput.png">

<p align="center"><sub>AWS <code>c8a.large</code> (AMD EPYC 9R45). See <a href="#benchmarks">Benchmarks</a> for the same sweep on a smaller Intel box, and for how to reproduce either.</sub></p>

The 100+ GiB/s figures are the peak of the sweep charted above (4 KiB encode, 64 KiB decode), not a sustained number at every size — the chart shows the whole curve, including where it's lower, and [Benchmarks](#benchmarks) has the full sweep and how to reproduce it. We don't claim to be faster than unchecked C/assembly in general, but on at least one machine we've measured, we're no longer conceding that fight by default either — see [Ecosystem](#ecosystem). "Memory-safe" here is a specific, bounded statement — see [Safety & Verification](#safety--verification) for exactly what's proven and what still rests on human judgment.

## Quick Start

### Encoding

```rust
use base64_turbo::STANDARD;

fn main() {
    let data = b"Speed and Safety";

    // Returns String
    let encoded = STANDARD.encode(data);

    assert_eq!(encoded, "U3BlZWQgYW5kIFNhZmV0eQ==");
}
```

### Decoding

```rust
use base64_turbo::STANDARD;

fn main() {
    let encoded = "U3BlZWQgYW5kIFNhZmV0eQ==";

    // Returns Result<Vec<u8>, Error>
    let decoded = STANDARD.decode(encoded).unwrap();

    assert_eq!(decoded, b"Speed and Safety");
}
```

### Zero-Allocation (Stack / `no_std`)

For scenarios where heap allocation is too slow (e.g., hot paths), write directly to stack buffers — the `_into` APIs need no allocator. Size the buffers with the `encoded_len`/`estimate_decoded_len` helpers rather than guessing:

```rust
use base64_turbo::STANDARD;

fn main() {
    let input = b"Low Latency";

    let mut enc_buf = vec![0u8; STANDARD.encoded_len(input.len())];
    let enc_len = STANDARD.encode_into(input, &mut enc_buf).unwrap();

    let mut dec_buf = vec![0u8; STANDARD.estimate_decoded_len(enc_len)];
    let dec_len = STANDARD.decode_into(&enc_buf[..enc_len], &mut dec_buf).unwrap();

    assert_eq!(&dec_buf[..dec_len], input);
}
```

## Feature Flags

Each x86 SIMD kernel is its own knob, so you compile in only what your target CPUs are likely to support. Runtime detection still gates every call, so enabling a kernel the host lacks just falls back to scalar.

| Feature | Default | Description |
| :--- | :---: | :--- |
| `std` | **Yes** | `String`/`Vec` support. Disable for `no_std` (the `_into` APIs need no allocator). |
| `avx2` | **Yes** | AVX2 kernel + runtime detection on x86/x86_64. Implies `std`. |
| `avx512` | **Yes** | AVX-512F/BW kernel + runtime detection on x86/x86_64. Implies `std`. |
| `avx512-vbmi` | **Yes** | AVX-512 VBMI fast-path kernel on x86/x86_64. Implies `std`. |
| `simd` | **Yes** | Convenience meta-feature — turns on `avx2` + `avx512` + `avx512-vbmi` at once. |
| `neon` | **Yes** | NEON acceleration on aarch64. No `std` required. |
| `unstable` | **No** | Exposes the raw internal kernels (`encode_avx2`, `encode_neon`, …). The `*_scalar` accessors are **safe** (they may panic on a too-small buffer, but never invoke UB). |

**Scalar-only builds are `#![forbid(unsafe_code)]`.** Disable every SIMD kernel and the crate is pure scalar Rust — nothing to verify, nothing to audit. The allocating `encode`/`decode` swap their uninitialized-buffer fast path for a zero-filled, fully-checked one in this configuration.

## Compatibility & Stability

### Minimum Supported Rust Version (MSRV)
**This crate requires Rust 1.89.0 or newer.** We rely on recently stabilized AVX512 intrinsics in `core` and do not plan to lower this.

### Public API Stability
The public API is considered **Stable**. We adhere to Semantic Versioning; the current surface stays valid and backward-compatible throughout the `0.3.x` lifecycle.

Output conforms to RFC 4648 — `STANDARD` and `URL_SAFE` are drop-in compatible with the `base64` crate. `serde` support is not included, to keep the dependency tree empty.

## Performance & Architecture

The design goal is maximum throughput *within* Rust's safety guarantees: vectorized data movement instead of byte-at-a-time lookup tables. We batch 32–64 bytes per register and push padding/error detection to bitmasks *after* the vector op, so the hot loop stays branchless.

### Why Is It Fast?

What each kernel does with that:

*   **Scalar (wide tables).** 100% safe Rust, `#![forbid(unsafe_code)]`. Encode maps 12 input bits directly to the two characters they produce (4 lookups per 6-byte block instead of 8); decode folds each character's bit-shift into the table itself, so a 4-character group is four loads OR-ed together and validation falls out of the same OR.
*   **AVX2.** `vpshufb` shuffles contend for port 5, so AND/OR/shift work is interleaved onto ports 0/1/5 to keep the shuffle port from bottlenecking. 256-bit registers behave as two 128-bit lanes, which a sliding bit-stream must cross — bridged with an offset load plus a permute instead of dropping to scalar.
*   **AVX512.** `k`-mask registers let the 1–31 byte tail run as a single masked vector op instead of a scalar fallback, and 32 `zmm` registers (vs 16 `ymm`) keep every LUT resident while unrolling harder.
*   **AVX512-VBMI.** The fastest path we have. Encode is three ops for 48 bytes: a `vpermb` gather, one `vpmultishiftqb` that extracts all eight 6-bit fields at once, then a `vpermb` through the alphabet. Decode looks up characters with `vpermi2b` across a 128-byte reverse LUT and folds validity into a single `vpternlogd` OR tree.
*   **NEON.** 128-bit `q` registers, 12→16 bytes per encode step. `vqtbl1q_u8` gives the same shuffle primitive as `vpshufb`, with full cross-lane access, so no lane-stitching is needed. Mandatory on ARMv8-A, hence compile-time dispatch.
*   **Dispatch.** x86 picks AVX512 → AVX2 → scalar at runtime (guarding against `SIGILL`); aarch64 picks NEON → scalar at compile time.

### Benchmarks

The chart at the top of this README and the numbers below are straight `cargo bench` output — same numbers, no cherry-picking (`benches/encoding_bench.rs`). [criterion.rs](https://github.com/bheisler/criterion.rs), 5 s warm-up, 15 s measurement per group. Input sizes span 32 B → 10 MB to cross L1/L2/RAM boundaries. `std` is the `base64` crate on default features — since 0.23 that means its `simd-unsafe` path (AVX2/NEON with runtime detection) is active, not the old scalar-only engine, so this is the number most callers of that crate actually get.

**AWS `c8a.large` (AMD EPYC 9R45), the chart above:** at 64 KiB, `base64-turbo` hits 81.1 GiB/s encode / 107.5 GiB/s decode, vs 14.6 / 14.6 GiB/s for `base64-simd` (+467% / +637%) and 9.8 / 25.6 GiB/s for the `base64` crate (+744% / +321%). The sweep peaks at 105.6 GiB/s encode (4 KiB) and 107.5 GiB/s decode (64 KiB) — both comfortably past 100 GiB/s, single-threaded, from a Kani/MIRI/MSan-checked kernel rather than an unaudited one. Small-input latency (32 B, zero-alloc `_into`): ~7.4 ns encode, ~10.0 ns decode.

**AWS `c7i.large` (Intel Xeon Platinum 8488C), a smaller/cheaper box, run the same way:**

<img alt="Base64 throughput by payload size on AWS c7i.large (Intel Xeon Platinum 8488C) — a smaller instance run with the same methodology" src="benches/results/throughput-c7i.png">

At 64 KiB: 30.9 GiB/s encode / 46.7 GiB/s decode, vs 11.1 / 10.2 GiB/s for `base64-simd` (+179% / +358%) and 6.9 / 13.2 GiB/s for the `base64` crate (+348% / +254%). Small-input latency (32 B, zero-alloc `_into`): ~10 ns encode, ~13 ns decode. The two machines agree on the shape of the curve (same dip past L2/L3, same relative ordering of libraries) and disagree by roughly 2-3x on the absolute ceiling — which is itself the point: the 100+ GiB/s number is a real peak on real hardware, not a property of the algorithm that holds everywhere.

```bash
sudo apt update && sudo apt install -y build-essential git
curl --proto '=https' --tlsv1.3 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

git clone https://github.com/hacer-bark/base64-turbo
cd base64-turbo
BENCH_TARGET=all cargo bench 2>&1 | tee benches/results/raw.txt
python3 benches/scripts/plot_bench.py benches/results/raw.txt
```

Select comparison targets with `BENCH_TARGET` (comma-separated): `turbo` (default, allocating API), `turbo-buff` (zero-allocation API), `simd`, `std`, `all`.

<details>
<summary>Raw <code>cargo bench</code> output — AWS <code>c8a.large</code> (AMD EPYC 9R45), <code>BENCH_TARGET=all</code></summary>

```ignore
Benchmarking Base64_Performances/Encode/Turbo/32
  time:   [10.862 ns 10.865 ns 10.869 ns]
  thrpt:  [2.7419 GiB/s 2.7429 GiB/s 2.7438 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/32
  time:   [7.4477 ns 7.4489 ns 7.4502 ns]
  thrpt:  [4.0002 GiB/s 4.0009 GiB/s 4.0016 GiB/s]

Benchmarking Base64_Performances/Encode/Std/32
  time:   [24.492 ns 24.511 ns 24.530 ns]
  thrpt:  [1.2149 GiB/s 1.2159 GiB/s 1.2168 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/32
  time:   [16.332 ns 16.335 ns 16.339 ns]
  thrpt:  [1.8240 GiB/s 1.8244 GiB/s 1.8248 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/32
  time:   [16.935 ns 16.942 ns 16.950 ns]
  thrpt:  [2.4176 GiB/s 2.4187 GiB/s 2.4198 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/32
  time:   [10.040 ns 10.042 ns 10.044 ns]
  thrpt:  [4.0801 GiB/s 4.0807 GiB/s 4.0814 GiB/s]

Benchmarking Base64_Performances/Decode/Std/32
  time:   [27.043 ns 27.051 ns 27.061 ns]
  thrpt:  [1.5143 GiB/s 1.5148 GiB/s 1.5153 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/32
  time:   [8.9181 ns 8.9196 ns 8.9212 ns]
  thrpt:  [4.5933 GiB/s 4.5941 GiB/s 4.5949 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/512
  time:   [13.098 ns 13.104 ns 13.109 ns]
  thrpt:  [36.374 GiB/s 36.390 GiB/s 36.404 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/512
  time:   [9.3030 ns 9.3046 ns 9.3062 ns]
  thrpt:  [51.239 GiB/s 51.247 GiB/s 51.256 GiB/s]

Benchmarking Base64_Performances/Encode/Std/512
  time:   [72.002 ns 72.055 ns 72.114 ns]
  thrpt:  [6.6123 GiB/s 6.6177 GiB/s 6.6225 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/512
  time:   [41.065 ns 41.071 ns 41.077 ns]
  thrpt:  [11.608 GiB/s 11.610 GiB/s 11.612 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/512
  time:   [21.861 ns 21.874 ns 21.887 ns]
  thrpt:  [29.105 GiB/s 29.123 GiB/s 29.140 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/512
  time:   [13.652 ns 13.655 ns 13.659 ns]
  thrpt:  [46.638 GiB/s 46.650 GiB/s 46.662 GiB/s]

Benchmarking Base64_Performances/Decode/Std/512
  time:   [56.755 ns 56.765 ns 56.776 ns]
  thrpt:  [11.220 GiB/s 11.222 GiB/s 11.224 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/512
  time:   [46.519 ns 46.532 ns 46.545 ns]
  thrpt:  [13.686 GiB/s 13.690 GiB/s 13.694 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/4096
  time:   [60.342 ns 60.439 ns 60.533 ns]
  thrpt:  [63.018 GiB/s 63.116 GiB/s 63.218 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/4096
  time:   [36.092 ns 36.126 ns 36.162 ns]
  thrpt:  [105.49 GiB/s 105.60 GiB/s 105.69 GiB/s]

Benchmarking Base64_Performances/Encode/Std/4096
  time:   [446.15 ns 446.32 ns 446.50 ns]
  thrpt:  [8.5435 GiB/s 8.5470 GiB/s 8.5503 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/4096
  time:   [305.12 ns 305.19 ns 305.26 ns]
  thrpt:  [12.497 GiB/s 12.499 GiB/s 12.502 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/4096
  time:   [83.132 ns 83.155 ns 83.178 ns]
  thrpt:  [61.179 GiB/s 61.196 GiB/s 61.213 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/4096
  time:   [48.642 ns 48.663 ns 48.683 ns]
  thrpt:  [104.53 GiB/s 104.57 GiB/s 104.62 GiB/s]

Benchmarking Base64_Performances/Decode/Std/4096
  time:   [253.50 ns 253.57 ns 253.64 ns]
  thrpt:  [20.063 GiB/s 20.068 GiB/s 20.074 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/4096
  time:   [387.20 ns 387.27 ns 387.33 ns]
  thrpt:  [13.138 GiB/s 13.140 GiB/s 13.142 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/65536
  time:   [752.49 ns 752.70 ns 752.92 ns]
  thrpt:  [81.065 GiB/s 81.088 GiB/s 81.111 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/65536
  time:   [734.65 ns 734.79 ns 734.94 ns]
  thrpt:  [83.048 GiB/s 83.065 GiB/s 83.081 GiB/s]

Benchmarking Base64_Performances/Encode/Std/65536
  time:   [6.1991 µs 6.2005 µs 6.2020 µs]
  thrpt:  [9.8413 GiB/s 9.8436 GiB/s 9.8459 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/65536
  time:   [4.1696 µs 4.1703 µs 4.1711 µs]
  thrpt:  [14.633 GiB/s 14.636 GiB/s 14.638 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/65536
  time:   [756.63 ns 757.00 ns 757.40 ns]
  thrpt:  [107.45 GiB/s 107.51 GiB/s 107.56 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/65536
  time:   [758.09 ns 758.36 ns 758.62 ns]
  thrpt:  [107.28 GiB/s 107.31 GiB/s 107.35 GiB/s]

Benchmarking Base64_Performances/Decode/Std/65536
  time:   [3.1829 µs 3.1837 µs 3.1845 µs]
  thrpt:  [25.556 GiB/s 25.563 GiB/s 25.569 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/65536
  time:   [5.5782 µs 5.5792 µs 5.5802 µs]
  thrpt:  [14.584 GiB/s 14.587 GiB/s 14.589 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/524288
  time:   [8.3915 µs 8.3997 µs 8.4072 µs]
  thrpt:  [58.079 GiB/s 58.131 GiB/s 58.188 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/524288
  time:   [8.3551 µs 8.3640 µs 8.3727 µs]
  thrpt:  [58.318 GiB/s 58.379 GiB/s 58.441 GiB/s]

Benchmarking Base64_Performances/Encode/Std/524288
  time:   [49.169 µs 49.179 µs 49.190 µs]
  thrpt:  [9.9264 GiB/s 9.9286 GiB/s 9.9308 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/524288
  time:   [33.273 µs 33.279 µs 33.285 µs]
  thrpt:  [14.670 GiB/s 14.672 GiB/s 14.675 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/524288
  time:   [8.5997 µs 8.6068 µs 8.6136 µs]
  thrpt:  [75.583 GiB/s 75.643 GiB/s 75.706 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/524288
  time:   [8.5915 µs 8.5967 µs 8.6019 µs]
  thrpt:  [75.686 GiB/s 75.732 GiB/s 75.778 GiB/s]

Benchmarking Base64_Performances/Decode/Std/524288
  time:   [25.342 µs 25.346 µs 25.350 µs]
  thrpt:  [25.682 GiB/s 25.686 GiB/s 25.691 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/524288
  time:   [44.581 µs 44.589 µs 44.598 µs]
  thrpt:  [14.598 GiB/s 14.601 GiB/s 14.604 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/1048576
  time:   [20.303 µs 20.309 µs 20.314 µs]
  thrpt:  [48.074 GiB/s 48.086 GiB/s 48.099 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/1048576
  time:   [20.219 µs 20.228 µs 20.237 µs]
  thrpt:  [48.257 GiB/s 48.278 GiB/s 48.298 GiB/s]

Benchmarking Base64_Performances/Encode/Std/1048576
  time:   [102.98 µs 103.02 µs 103.06 µs]
  thrpt:  [9.4752 GiB/s 9.4793 GiB/s 9.4832 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/1048576
  time:   [66.594 µs 66.614 µs 66.635 µs]
  thrpt:  [14.655 GiB/s 14.660 GiB/s 14.664 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/1048576
  time:   [20.364 µs 20.373 µs 20.383 µs]
  thrpt:  [63.880 GiB/s 63.912 GiB/s 63.940 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/1048576
  time:   [20.547 µs 20.554 µs 20.560 µs]
  thrpt:  [63.331 GiB/s 63.351 GiB/s 63.371 GiB/s]

Benchmarking Base64_Performances/Decode/Std/1048576
  time:   [55.037 µs 55.056 µs 55.075 µs]
  thrpt:  [23.642 GiB/s 23.650 GiB/s 23.658 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/1048576
  time:   [90.016 µs 90.038 µs 90.062 µs]
  thrpt:  [14.458 GiB/s 14.462 GiB/s 14.465 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/10485760
  time:   [208.73 µs 209.09 µs 209.52 µs]
  thrpt:  [46.610 GiB/s 46.704 GiB/s 46.786 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/10485760
  time:   [210.23 µs 210.80 µs 211.47 µs]
  thrpt:  [46.180 GiB/s 46.326 GiB/s 46.452 GiB/s]

Benchmarking Base64_Performances/Encode/Std/10485760
  time:   [1.0407 ms 1.0414 ms 1.0420 ms]
  thrpt:  [9.3719 GiB/s 9.3773 GiB/s 9.3834 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/10485760
  time:   [675.11 µs 675.29 µs 675.45 µs]
  thrpt:  [14.458 GiB/s 14.461 GiB/s 14.465 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/10485760
  time:   [225.32 µs 225.96 µs 226.56 µs]
  thrpt:  [57.471 GiB/s 57.625 GiB/s 57.787 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/10485760
  time:   [224.19 µs 224.80 µs 225.40 µs]
  thrpt:  [57.767 GiB/s 57.921 GiB/s 58.080 GiB/s]

Benchmarking Base64_Performances/Decode/Std/10485760
  time:   [565.40 µs 565.89 µs 566.42 µs]
  thrpt:  [22.988 GiB/s 23.010 GiB/s 23.029 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/10485760
  time:   [905.91 µs 906.10 µs 906.30 µs]
  thrpt:  [14.367 GiB/s 14.370 GiB/s 14.373 GiB/s]
```

</details>

<details>
<summary>Raw <code>cargo bench</code> output — AWS <code>c7i.large</code> (Intel Xeon Platinum 8488C), <code>BENCH_TARGET=all</code></summary>

```ignore
Benchmarking Base64_Performances/Encode/Turbo/32
  time:   [14.668 ns 14.678 ns 14.686 ns]
  thrpt:  [2.0292 GiB/s 2.0305 GiB/s 2.0317 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/32
  time:   [10.047 ns 10.051 ns 10.055 ns]
  thrpt:  [2.9639 GiB/s 2.9651 GiB/s 2.9663 GiB/s]

Benchmarking Base64_Performances/Encode/Std/32
  time:   [31.882 ns 31.902 ns 31.923 ns]
  thrpt:  [955.98 MiB/s 956.61 MiB/s 957.20 MiB/s]

Benchmarking Base64_Performances/Encode/Simd/32
  time:   [22.205 ns 22.208 ns 22.213 ns]
  thrpt:  [1.3417 GiB/s 1.3419 GiB/s 1.3422 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/32
  time:   [21.370 ns 21.377 ns 21.384 ns]
  thrpt:  [1.9163 GiB/s 1.9169 GiB/s 1.9175 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/32
  time:   [13.355 ns 13.358 ns 13.362 ns]
  thrpt:  [3.0668 GiB/s 3.0676 GiB/s 3.0684 GiB/s]

Benchmarking Base64_Performances/Decode/Std/32
  time:   [33.923 ns 33.931 ns 33.938 ns]
  thrpt:  [1.2075 GiB/s 1.2077 GiB/s 1.2080 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/32
  time:   [13.256 ns 13.259 ns 13.263 ns]
  thrpt:  [3.0896 GiB/s 3.0906 GiB/s 3.0913 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/512
  time:   [17.432 ns 17.440 ns 17.449 ns]
  thrpt:  [27.328 GiB/s 27.341 GiB/s 27.354 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/512
  time:   [13.087 ns 13.094 ns 13.101 ns]
  thrpt:  [36.398 GiB/s 36.418 GiB/s 36.437 GiB/s]

Benchmarking Base64_Performances/Encode/Std/512
  time:   [96.101 ns 96.121 ns 96.139 ns]
  thrpt:  [4.9598 GiB/s 4.9608 GiB/s 4.9618 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/512
  time:   [54.673 ns 54.689 ns 54.712 ns]
  thrpt:  [8.7154 GiB/s 8.7191 GiB/s 8.7217 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/512
  time:   [27.364 ns 27.396 ns 27.431 ns]
  thrpt:  [23.222 GiB/s 23.252 GiB/s 23.280 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/512
  time:   [18.629 ns 18.633 ns 18.638 ns]
  thrpt:  [34.179 GiB/s 34.188 GiB/s 34.196 GiB/s]

Benchmarking Base64_Performances/Decode/Std/512
  time:   [85.862 ns 85.901 ns 85.936 ns]
  thrpt:  [7.4128 GiB/s 7.4158 GiB/s 7.4192 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/512
  time:   [67.789 ns 67.916 ns 68.069 ns]
  thrpt:  [9.3585 GiB/s 9.3796 GiB/s 9.3971 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/4096
  time:   [112.47 ns 112.85 ns 113.22 ns]
  thrpt:  [33.692 GiB/s 33.805 GiB/s 33.918 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/4096
  time:   [72.559 ns 72.680 ns 72.832 ns]
  thrpt:  [52.377 GiB/s 52.486 GiB/s 52.574 GiB/s]

Benchmarking Base64_Performances/Encode/Std/4096
  time:   [561.49 ns 561.64 ns 561.78 ns]
  thrpt:  [6.7904 GiB/s 6.7921 GiB/s 6.7939 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/4096
  time:   [390.33 ns 390.48 ns 390.65 ns]
  thrpt:  [9.7649 GiB/s 9.7692 GiB/s 9.7730 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/4096
  time:   [127.00 ns 127.50 ns 128.04 ns]
  thrpt:  [39.744 GiB/s 39.911 GiB/s 40.067 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/4096
  time:   [96.768 ns 96.829 ns 96.912 ns]
  thrpt:  [52.509 GiB/s 52.554 GiB/s 52.587 GiB/s]

Benchmarking Base64_Performances/Decode/Std/4096
  time:   [389.66 ns 389.91 ns 390.17 ns]
  thrpt:  [13.042 GiB/s 13.051 GiB/s 13.060 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/4096
  time:   [504.21 ns 504.27 ns 504.33 ns]
  thrpt:  [10.090 GiB/s 10.091 GiB/s 10.093 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/65536
  time:   [1.9721 µs 1.9728 µs 1.9735 µs]
  thrpt:  [30.927 GiB/s 30.939 GiB/s 30.949 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/65536
  time:   [1.9331 µs 1.9341 µs 1.9352 µs]
  thrpt:  [31.540 GiB/s 31.558 GiB/s 31.574 GiB/s]

Benchmarking Base64_Performances/Encode/Std/65536
  time:   [8.8401 µs 8.8437 µs 8.8471 µs]
  thrpt:  [6.8989 GiB/s 6.9016 GiB/s 6.9043 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/65536
  time:   [5.5112 µs 5.5121 µs 5.5131 µs]
  thrpt:  [11.071 GiB/s 11.073 GiB/s 11.075 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/65536
  time:   [1.7409 µs 1.7416 µs 1.7423 µs]
  thrpt:  [46.710 GiB/s 46.729 GiB/s 46.748 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/65536
  time:   [1.7835 µs 1.7921 µs 1.8017 µs]
  thrpt:  [45.171 GiB/s 45.411 GiB/s 45.632 GiB/s]

Benchmarking Base64_Performances/Decode/Std/65536
  time:   [6.1656 µs 6.1728 µs 6.1800 µs]
  thrpt:  [13.169 GiB/s 13.184 GiB/s 13.199 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/65536
  time:   [7.9646 µs 7.9757 µs 7.9861 µs]
  thrpt:  [10.191 GiB/s 10.204 GiB/s 10.218 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/524288
  time:   [14.698 µs 14.702 µs 14.707 µs]
  thrpt:  [33.200 GiB/s 33.211 GiB/s 33.222 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/524288
  time:   [15.037 µs 15.084 µs 15.139 µs]
  thrpt:  [32.253 GiB/s 32.371 GiB/s 32.472 GiB/s]

Benchmarking Base64_Performances/Encode/Std/524288
  time:   [74.420 µs 74.599 µs 74.856 µs]
  thrpt:  [6.5230 GiB/s 6.5454 GiB/s 6.5612 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/524288
  time:   [44.463 µs 44.478 µs 44.494 µs]
  thrpt:  [10.974 GiB/s 10.978 GiB/s 10.982 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/524288
  time:   [14.137 µs 14.145 µs 14.153 µs]
  thrpt:  [45.999 GiB/s 46.028 GiB/s 46.052 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/524288
  time:   [13.774 µs 13.780 µs 13.786 µs]
  thrpt:  [47.225 GiB/s 47.247 GiB/s 47.267 GiB/s]

Benchmarking Base64_Performances/Decode/Std/524288
  time:   [48.060 µs 48.068 µs 48.077 µs]
  thrpt:  [13.542 GiB/s 13.544 GiB/s 13.546 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/524288
  time:   [60.128 µs 60.160 µs 60.218 µs]
  thrpt:  [10.811 GiB/s 10.822 GiB/s 10.828 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/1048576
  time:   [55.547 µs 55.563 µs 55.577 µs]
  thrpt:  [17.571 GiB/s 17.576 GiB/s 17.581 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/1048576
  time:   [55.467 µs 55.480 µs 55.492 µs]
  thrpt:  [17.598 GiB/s 17.602 GiB/s 17.606 GiB/s]

Benchmarking Base64_Performances/Encode/Std/1048576
  time:   [149.53 µs 149.72 µs 149.89 µs]
  thrpt:  [6.5153 GiB/s 6.5226 GiB/s 6.5310 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/1048576
  time:   [87.903 µs 87.917 µs 87.932 µs]
  thrpt:  [11.106 GiB/s 11.108 GiB/s 11.110 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/1048576
  time:   [57.993 µs 58.015 µs 58.043 µs]
  thrpt:  [22.433 GiB/s 22.444 GiB/s 22.452 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/1048576
  time:   [57.837 µs 57.848 µs 57.860 µs]
  thrpt:  [22.504 GiB/s 22.509 GiB/s 22.513 GiB/s]

Benchmarking Base64_Performances/Decode/Std/1048576
  time:   [100.47 µs 100.48 µs 100.49 µs]
  thrpt:  [12.957 GiB/s 12.958 GiB/s 12.959 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/1048576
  time:   [120.71 µs 120.72 µs 120.74 µs]
  thrpt:  [10.784 GiB/s 10.786 GiB/s 10.787 GiB/s]

Benchmarking Base64_Performances/Encode/Turbo/10485760
  time:   [818.73 µs 818.88 µs 819.04 µs]
  thrpt:  [11.923 GiB/s 11.926 GiB/s 11.928 GiB/s]

Benchmarking Base64_Performances/Encode/TurboBuff/10485760
  time:   [842.43 µs 860.30 µs 875.16 µs]
  thrpt:  [11.159 GiB/s 11.351 GiB/s 11.592 GiB/s]

Benchmarking Base64_Performances/Encode/Std/10485760
  time:   [2.0863 ms 2.0878 ms 2.0893 ms]
  thrpt:  [4.6742 GiB/s 4.6774 GiB/s 4.6808 GiB/s]

Benchmarking Base64_Performances/Encode/Simd/10485760
  time:   [1.0348 ms 1.0392 ms 1.0444 ms]
  thrpt:  [9.3505 GiB/s 9.3969 GiB/s 9.4372 GiB/s]

Benchmarking Base64_Performances/Decode/Turbo/10485760
  time:   [916.59 µs 917.07 µs 917.55 µs]
  thrpt:  [14.191 GiB/s 14.198 GiB/s 14.206 GiB/s]

Benchmarking Base64_Performances/Decode/TurboBuff/10485760
  time:   [945.00 µs 969.61 µs 996.89 µs]
  thrpt:  [13.061 GiB/s 13.429 GiB/s 13.779 GiB/s]

Benchmarking Base64_Performances/Decode/Std/10485760
  time:   [1.8407 ms 1.9152 ms 1.9933 ms]
  thrpt:  [6.5323 GiB/s 6.7987 GiB/s 7.0739 GiB/s]

Benchmarking Base64_Performances/Decode/Simd/10485760
  time:   [1.6604 ms 1.6946 ms 1.7335 ms]
  thrpt:  [7.5114 GiB/s 7.6836 GiB/s 7.8421 GiB/s]
```

</details>

## Safety & Verification

**Philosophy:** `Safety > Performance > Convenience`. We use `unsafe` SIMD intrinsics and raw pointer arithmetic, so rather than rely on review alone we stack independent layers that cover each other's blind spots.

| Architecture | MIRI | MSan | Kani | Fuzzing |
| :--- | :---: | :---: | :---: | :---: |
| **AVX2** | ✅ | ✅ | ✅ (CI, unbounded) | ✅ |
| **AVX512** (`avx512f`+`avx512bw`) | ✅ | ✅ | ✅ (local only) | ✅ |
| **AVX512-VBMI** (`vpermb`/`vpermi2b`) | ✅ | ✅ | ❌ | ✅ |
| **NEON** | ✅ | ✅ | ❌ | ❌ |

*   **Kani** proves the kernels don't panic, don't read/write out of bounds, and round-trip exactly. For AVX2 the bounds result holds for *every* input length by a machine-checked induction over the loop's offset arithmetic — not just the lengths a harness happens to unwind. For the other paths it holds at the lengths the harnesses pin.
*   **MIRI** catches Undefined Behavior (provenance, alignment, OOB pointer arithmetic, data races) on every distinct code path — single-vector loop, wide unrolled loop, masked tail, scalar tail — for Scalar, AVX2, AVX512 and AVX512-VBMI. Branch coverage, not exhaustive input coverage.
*   **MSan** rebuilds the standard library with instrumentation (`-Z build-std -Z sanitizer=memory`) to confirm we never branch on or emit uninitialized memory, which matters given how much AVX512 masking we do.
*   **Fuzzing** — 2.5B+ `cargo-fuzz` iterations across all paths, no crashes to date.

**What still rests on human judgment:**

1.  The index proofs that make the AVX2 bound hold for every length mirror the loop's offset arithmetic; they don't execute it. If a stride changes without the model changing too, the proofs keep passing — treat those constants as part of the code.
2.  Kani can't execute SIMD, so each AVX2 intrinsic is a Rust transcription of the Intel Intrinsics Guide pseudocode. `avx2_stub_equivalence` (`cargo test`) runs every model against the real instruction on real hardware to catch transcription errors, but doesn't prove the models agree everywhere.
3.  AVX512, AVX512-VBMI and NEON haven't had the AVX2 treatment (symbolic index proofs covering every length) — for those, verification stops at the lengths MIRI's harnesses pin.

The AVX512 Kani harness isn't in CI (it exceeds GitHub Actions' time/memory budget); it's re-run locally before each release:

```sh
cargo kani --unstable stubbing --harness kani_verification_avx512
```

AVX512-VBMI has no Kani harness — `vpermb`/`vpermi2b` have no model yet. NEON has none either. Both rest on MIRI, MSan and fuzzing. Read the [CI logs](https://github.com/hacer-bark/base64-turbo/actions) and the `unsafe` blocks themselves — each documents the contract it relies on.

## Ecosystem

| Library | Lang | SIMD | Verified `unsafe` | Encode (64 KiB) | Decode (64 KiB) | Source |
| :--- | :---: | :---: | :---: | ---: | ---: | :--- |
| **base64-turbo** | Rust | ✅ | ✅ Kani + MIRI + MSan + Fuzz | 27.1 GiB/s | 34.6 GiB/s | our bench, same box † |
| [Turbo-Base64](https://github.com/powturbo/Turbo-Base64) | C | ✅ | ❌ | 18.4 GiB/s | 37.8 GiB/s | our bench, same box † |
| [base64](https://crates.io/crates/base64) (std) | Rust | ✅ (0.23+) | ✅ MIRI + Fuzz | 6.9 GiB/s | 13.2 GiB/s | our bench |
| [base64-simd](https://crates.io/crates/base64-simd) | Rust | ✅ | ❌ | 11.1 GiB/s | 10.2 GiB/s | our bench |
| [aklomp/base64](https://github.com/aklomp/base64) | C | ✅ | ❌ | 24.4 GiB/s | 21.0 GiB/s | vendor bench |
| [fastbase64](https://github.com/lemire/fastbase64) | C | ✅ | ❌ | 22.1 GiB/s | 19.8 GiB/s | vendor bench |

All Rust rows and the Turbo-Base64 row are ours, measured on the same AWS `c7i.large` in the same session — we cloned Turbo-Base64's real upstream C source, built it with its own official per-kernel flags (its `tb64v512vbmi` kernel auto-selects on this CPU, confirmed at runtime), verified our harness round-trips and rejects corrupt input the same as its own checked decode, and ran both back to back, pinned to one core. 

† — We wrote the C-side timing harness ourselves rather than using theirs, matched to our criterion methodology as closely as a hand-rolled harness reasonably can — solid, but it's one measurement session, not the statistical rigor criterion gives the Rust numbers, so treat the margins as directional rather than exact. On that comparison, we're ahead on both encode and decode — by a wide margin on encode, a smaller one on decode — which is close enough on the decode side that we no longer assume unchecked C is automatically ahead here, though we're not claiming a general win either. The aklomp/base64 and fastbase64 rows are still the vendors' own published numbers on an Intel i7-9700K from 2022 ([source](https://github.com/powturbo/Turbo-Base64#benchmark-incl-the-best-simd-base64-libs), decimal MB/s converted to GiB/s), unreproduced by us — treat those two as directional only.

`base64` (std) added a SIMD path in 0.23 (the default-on `simd-unsafe` feature, AVX2/NEON with runtime detection) — it's no longer the zero-`unsafe` scalar crate it used to be, and we bench it as most users get it, default features on. It publishes MIRI and fuzz coverage for that path but no Kani or MSan, which is the gap the two extra layers in [Safety & Verification](#safety--verification) close for us. `base64-simd` is a strong crate that raised the bar before us; we measure faster overall on this box and publish Kani/MIRI/MSan we couldn't find for it. The C libraries still get real advantages from unchecked pointer arithmetic and no published verification (`turbo-base64` is also GPLv3, against our MIT-or-Apache-2.0) — pick them if you need the absolute ceiling on unfamiliar hardware and will own the risk.

Also in the space: [vb64](https://crates.io/crates/vb64) (unmaintained), [base-d](https://crates.io/crates/base-d) (33+ alphabets, decode-only SIMD), [webbuf](https://crates.io/crates/webbuf) and [baste64](https://crates.io/crates/baste64) (WASM-oriented). None of these publish Kani or MIRI verification of their `unsafe` code, as far as we could find.

## Acknowledgements

The encode/decode kernels build on techniques published by others, all under permissive licenses:

*   **[Alfred Klomp](https://github.com/aklomp) — [`aklomp/base64`](https://github.com/aklomp/base64) (BSD-2-Clause).** Our decoder's nibble-lookup validation and our encoder's offset-load loop and single-LUT character mapping are direct ports from this library. The URL-safe tables aren't published anywhere we could find — we re-derived them and verified them exhaustively (`src/simd/avx2.rs`).
*   **[Daniel Lemire](https://github.com/lemire) and Wojciech Muła — [`lemire/fastbase64`](https://github.com/lemire/fastbase64) (BSD-2-Clause).** `fastavxbase64.c` independently documents the same nibble-lookup decode algorithm (originated by Muła, `+`/`/` disambiguation credited there to `@aqrit`), which we cross-referenced while implementing ours.
*   **[`base64-simd`](https://crates.io/crates/base64-simd) (MIT).** Its benchmarks and API design were a useful reference point throughout.

## License

Licensed under either of

- [Apache License, Version 2.0](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-APACHE)
- [MIT license](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this crate, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
