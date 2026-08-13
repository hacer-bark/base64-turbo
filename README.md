# Base64 Turbo

[![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![License](https://img.shields.io/crates/l/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
[![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/verification.yml?label=Kani%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/verification.yml)
[![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)

**A SIMD Base64 implementation whose `unsafe` paths are checked by a model checker, not just by review.**

`base64-turbo` is built for high-throughput systems where CPU cycles are scarce and Undefined Behavior is unacceptable. It picks the best kernel available at runtime:

*   **x86_64:** AVX512 (incl. a VBMI fast path) or AVX2, via runtime CPU detection.
*   **ARM (aarch64):** NEON, via compile-time dispatch — no detection overhead.
*   **Other:** An optimized SWAR scalar kernel.

### What we actually claim

We are **not** faster than unchecked C/assembly — we aren't and don't try to be (see [Ecosystem](#ecosystem)). The narrower claim: **within the set of crates combining SIMD-accelerated Base64 with Kani + MIRI verification, we are not aware of another that reaches AVX512 speeds.** If you know of one, open an issue.

"Memory-safe" here is a specific, bounded statement, not a blanket one. Read [Safety & Verification](#safety--verification) for exactly what each layer proves and what it doesn't — including the parts that rest on human argument rather than machine proof.

## Quick Start

```rust
use base64_turbo::STANDARD;

let encoded = STANDARD.encode(b"Speed and Safety");
assert_eq!(encoded, "U3BlZWQgYW5kIFNhZmV0eQ==");

let decoded = STANDARD.decode(&encoded).unwrap();
assert_eq!(decoded, b"Speed and Safety");
```

### Zero-allocation (stack / `no_std`)

```rust
use base64_turbo::STANDARD;

let input = b"Low Latency";
let mut output = [0u8; 64];

let len = STANDARD.encode_into(input, &mut output).unwrap();
assert_eq!(&output[..len], b"TG93IExhdGVuY3k=");
```

Size the buffers with the helpers rather than guessing:

```rust
use base64_turbo::STANDARD;

let input = b"Low Latency";
let mut enc_buf = vec![0u8; STANDARD.encoded_len(input.len())];
let enc_len = STANDARD.encode_into(input, &mut enc_buf).unwrap();

let mut dec_buf = vec![0u8; STANDARD.estimate_decoded_len(enc_len)];
let dec_len = STANDARD.decode_into(&enc_buf[..enc_len], &mut dec_buf).unwrap();
assert_eq!(&dec_buf[..dec_len], input);
```

## Feature Flags

| Feature | Default | Description |
| :--- | :---: | :--- |
| `std` | ✅ | `String`/`Vec` support. Disable for `no_std` (the `_into` APIs need no allocator). |
| `simd` | ✅ | Runtime detection for AVX512/AVX2 on x86/x86_64. |
| `neon` | ✅ | NEON acceleration on aarch64. No `std` required. |
| `unstable` | ❌ | Exposes the raw `unsafe` internals (`encode_avx2`, `encode_neon`, …). |

## Compatibility & Stability

**MSRV: Rust 1.89.0.** We rely on recently stabilized AVX512 intrinsics in `core`. We do not plan to lower this or to gate it behind feature flags.

The public API is **stable** and follows SemVer; it stays backward-compatible across the `0.2.x` line.

Output conforms to RFC 4648 — `STANDARD` and `URL_SAFE` are drop-in compatible with the `base64` crate. `serde` support is not included, to keep the dependency tree empty; wrap the API in your own serializer if you need it.

## Performance

![Benchmark Graph](https://github.com/hacer-bark/base64-turbo/blob/main/benches/img/base64_intel.png?raw=true)

Throughput at 64 KiB, our own `cargo bench` runs. `simd` = `base64-simd`, `std` = the `base64` crate.

| Machine | Encode | vs `simd` | Decode | vs `simd` | `std` (enc/dec) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Xeon Platinum 8488C, AVX512 (AWS `c7i.large`) | 12.48 GiB/s | +18% | **21.04 GiB/s** | +110% | 2.42 / 2.78 |
| EPYC Genoa (Zen 4), AVX512 (Vultr) | 11.08 GiB/s | **−3.6%** | 15.44 GiB/s | +46% | 2.02 / 2.51 |
| Core i7-8750H, AVX2 | 8.93 GiB/s | +3.8% | 12.14 GiB/s | +52% | 1.66 / 1.66 |
| Core i7-8750H, SIMD forced off (scalar) | 1.77 GiB/s | — | 2.44 GiB/s | — | 1.81 / 1.50 |

Small-input latency (32 B, zero-alloc `encode_into`): ~10 ns on Xeon and EPYC, ~17 ns on the i7, ~21 ns scalar — roughly 1.5–1.8x ahead of `base64-simd` and 2–4x ahead of `base64`.

Two honest notes: decode is where the real win is, and on Zen 4 `base64-simd` **beats us on streaming encode by ~4%**. Encode is compute-bound (3→4 byte expansion, complex bit-interleaving); decode is closer to memory-bound and saturates the test harness's bandwidth on the larger boxes. The scalar row is the same i7 with SIMD forcibly disabled, which is what an embedded or non-x86 target would see.

<details>
<summary><b>Benchmark methodology &amp; reproduction</b></summary>

[criterion.rs](https://github.com/bheisler/criterion.rs), 5 s warm-up, 15 s measurement per group, 0.05 noise threshold. 250 samples below 1 MB, 50 above. Input sizes span 32 B → 10 MB (32 B, 512 B, 4 KB, 64 KB, 512 KB, 1 MB, 10 MB) to cross L1/L2/RAM boundaries; plots use a log X-axis.

Select comparison targets with `BENCH_TARGET` (comma-separated): `turbo` (default, allocating API), `turbo-buff` (zero-allocation API), `simd`, `std`, `all`.

```bash
BENCH_TARGET=turbo,simd cargo bench   # head-to-head
BENCH_TARGET=all cargo bench          # everything (slow)
BENCH_TARGET=turbo-buff cargo bench   # zero-alloc only
```
</details>

## Safety & Verification

**Philosophy:** `Safety > Performance > Convenience`. We use `unsafe` SIMD intrinsics and raw pointer arithmetic, so rather than rely on review alone we stack independent layers that cover each other's blind spots.

| Architecture | MIRI | MSan | Kani | Fuzzing |
| :--- | :---: | :---: | :---: | :---: |
| **Scalar** | ✅ | ✅ | ✅ (CI) | ✅ |
| **AVX2** | ✅ | ✅ | ✅ (CI) | ✅ |
| **AVX512** (`avx512f`+`avx512bw`) | ✅ | ✅ | ✅ (local only) | ✅ |
| **AVX512-VBMI** (`vpermb`/`vpermi2b`) | ✅ | ✅ | ❌ | ✅ |
| **NEON** | ✅ | ✅ | ❌ | ❌ |

*   **Kani** — the model checker explores *every possible input byte value* at chosen lengths and proves the kernel does not panic, does not read out of bounds, and round-trips exactly. Scope and caveats below.
*   **MIRI** — checks for Undefined Behavior (strict provenance, alignment, OOB pointer arithmetic, data races) on the inputs it runs. Every distinct code path — single-vector loop, quad-vector loop, scalar tail — is exercised for Scalar, AVX2, AVX512 and AVX512-VBMI. This is branch coverage, not exhaustive input coverage.
*   **MSan** — the whole standard library is rebuilt with instrumentation (`-Z build-std -Z sanitizer=memory`) to confirm we never branch on or emit uninitialized memory, which matters given how much AVX512 masking we do.
*   **Fuzzing** — 2.5B+ `cargo-fuzz` iterations across all paths, no crashes to date.

### How the Kani proofs work, and what they don't cover

Base64 is a linear, block-structured operation, so we don't check every length. Each harness fixes a length chosen to exercise (a) the loop body **at least twice**, so the pointer state one iteration hands to the next is itself proven a valid entry state, (b) the SIMD→scalar handoff, and (c) a non-empty, non-aligned scalar tail. The input bytes are fully symbolic (`kani::any()`), so at `ENC_INDUCTION_LEN = 53` the encoder proof covers all 2^(53·8) possible inputs — not samples.

The constants come straight from the code's structure. `encode_slice_avx2` enters SIMD at `len >= 32` and works in 24-byte rounds, so `53 = 2×24 + 5`; `decode_slice_avx2` works in 32-byte passes, so `DEC_INDUCTION_LEN = 69 = 2×32 + 5`. `QUAD_ENC_INDUCTION_LEN = 125` is the smallest length hitting the 4×-unrolled quad tier, proven by its own harness since that tier has its own fixed-offset pointer arithmetic. (An earlier version used `29`, which was below the `len >= 32` guard and silently proved only the scalar fallback — the kind of mistake these constants are now derived, not guessed, to avoid.)

**Be precise about what that buys you.** Kani proves the base cases exhaustively. The generalization from "correct at 53 bytes" to "correct at any N" is a **documented human argument** — the loop stride is a fixed unconditional increment and the two-iteration proof shows the handoff state is stable — **not a machine-checked induction**. There is no symbolic `N` and no loop invariant in the harnesses. That argument is sound as far as we can tell, and it is the step you should scrutinize first if you're auditing this crate.

Three further limits, stated plainly:

1.  **Output buffers in the harnesses are oversized** (e.g. a 128-byte array for 72 bytes of real output). Kani therefore proves the kernels stay inside a *padded* buffer, not that they stay inside the exact output length. A write a few bytes past the legitimate tail would not be caught.
2.  **Only `padding: true` is proven on the SIMD paths.** `url_safe` is symbolic there; the no-padding tail is Kani-covered on the scalar path only.
3.  **The proofs run against stubs, not silicon.** ~25 intrinsics are replaced with hand-written Rust models, transcribed variable-for-variable from the [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) so they can be diffed against the reference. A wrong stub silently weakens the corresponding proof.

### Kani in CI vs. locally

`verification.yml` runs the Scalar and AVX2 harnesses. The AVX512 proof is **not** in CI — its state space exceeded GitHub Actions' time and memory budget. It is re-run locally before each release and passes. The harness is checked in, so you can reproduce it:

```sh
cargo kani --unstable stubbing --harness kani_verification_avx512
```

AVX512-VBMI has no proof at all: it depends on `vpermb` / `vpermi2b`, for which we haven't written stubs yet. NEON has no Kani harness either. Both rest on MIRI, MSan and fuzzing — a real bar, but a lower one than the proven paths.

### Verification FAQ

**The crate uses `unsafe`. How is that "safe"?** We separate "Safe Rust" (compiler-checked) from "memory-safe" (proven/checked by the layers above). For the Kani-proven paths, no input through the public API can trigger an overflow, segfault or panic, subject to the caveats above.

**Can I crash it with garbage input?** Not through the safe API. Invalid Base64, binary noise, or hostile payloads return `Err`. The decoder-robustness harnesses feed the decoder fully symbolic garbage and prove it never panics.

**What if I misuse the `unstable` internals?** Then the crash is yours. Those functions exist to skip bounds checks when every cycle matters; we verify that *our* safe API upholds their contracts, not that you will.

**Is it production-ready?** Scalar, AVX2 and plain AVX512 are Kani-proven plus MIRI/MSan/fuzz-clean. AVX512-VBMI and NEON have everything except Kani. We ship both; you should know which one your CPU picks.

**How do I know your stubs are right?** You read them against the Intel guide — they're written for exactly that. Nothing currently cross-checks them against real hardware, which is the honest answer.

**How can I trust any of this?** Don't. Read the [CI logs](https://github.com/hacer-bark/base64-turbo/actions), reproduce the proofs, and read the `unsafe` blocks — each documents the contract it relies on.

## Architecture

The design goal is maximum throughput *within* Rust's safety guarantees, by trading byte-at-a-time lookup tables (data-dependent, branch-heavy) for vectorized data movement: batch 32–64 bytes per register, and handle padding and error detection with bitmasks *after* the vector op so the hot loop stays branchless.

**Scalar (SWAR).** Casts to `u64` and builds indices with shifts and masks, moving 8 bytes per instruction instead of one. `unsafe` pointer casts, bounded by the Kani proofs.

**AVX2.** Tuned for execution-port balance: `vpshufb` shuffles contend for port 5, so we interleave AND/OR/shift work that can issue on ports 0/1/5 to keep the shuffle port from becoming the bottleneck. AVX2's 256-bit registers behave as two independent 128-bit lanes, which a sliding bit-stream like Base64 must cross — we bridge that with an offset load plus a permute ("lane stitching") instead of dropping to scalar.

**AVX512.** Two gains over AVX2: `k`-mask registers let the 1–31 byte tail be processed in a single masked vector op rather than a scalar fallback, and 32 `zmm` registers (vs 16 `ymm`) let us keep every LUT and constant resident while unrolling harder. The VBMI variant adds `vpermb`/`vpermi2b` and is dispatched only where supported.

**NEON.** 128-bit `q` registers, 12→16 bytes per encode step, 16→12 per decode. `vqtbl1q_u8` gives the same shuffle primitive as `vpshufb`, and NEON has full cross-lane access within the register, so no stitching is needed. Mandatory on ARMv8-A, hence compile-time dispatch and `no_std` compatibility.

**Dispatch.** x86 picks AVX512 → AVX2 → scalar at runtime (guarding against `SIGILL`); aarch64 picks NEON → scalar at compile time, since the ISA guarantees NEON.

## Ecosystem

| Library | Lang | SIMD | Verified `unsafe` | Encode (AVX2) | Source |
| :--- | :---: | :---: | :---: | ---: | :--- |
| **base64-turbo** | Rust | ✅ | ✅ Kani/MIRI/MSan | ~12.1 GiB/s | our bench |
| [base64-simd](https://crates.io/crates/base64-simd) | Rust | ✅ | ❌ | ~8.0 GiB/s | our bench |
| [base64](https://crates.io/crates/base64) (std) | Rust | ❌ | ✅ no `unsafe` | ~1.6 GiB/s | our bench |
| [Turbo-Base64](https://github.com/powturbo/Turbo-Base64) | C | ✅ | ❌ | ~29 GiB/s | vendor-reported |
| [fastbase64](https://github.com/lemire/fastbase64) | C | ✅ | ❌ | ~23 GiB/s | vendor-reported |
| [aklomp/base64](https://github.com/aklomp/base64) | C | ✅ | ❌ | ~25 GiB/s | vendor-reported |

The Rust rows are reproducible via `BENCH_TARGET=all cargo bench`. The C rows are **not** wired into our harness — those are the projects' own published figures, unverified by us.

**vs. the C libraries.** They are genuinely faster and get there through unchecked pointer arithmetic with no published verification. `turbo-base64` is also GPLv3/commercial, against our MIT-or-Apache-2.0. Pick them if you need the ceiling and will own the risk; pick us if you want to stay near C-level throughput with the `unsafe` paths checked.

**vs. `base64` (std).** Zero `unsafe`, rock-solid, scalar-only, ~4–8x slower. The right answer if `unsafe` is banned in your dependency tree.

**vs. `base64-simd`.** A strong crate that raised the bar before us. We measured faster overall (though it wins on Zen 4 streaming encode) and add Kani/MIRI verification, which we could not find published for it.

Also in the space: [vb64](https://crates.io/crates/vb64) (unmaintained, fails to build on modern nightly), [base-d](https://crates.io/crates/base-d) (33+ alphabets, decode-only SIMD — use it if you need exotic alphabets), [webbuf](https://crates.io/crates/webbuf) (WASM/convenience-oriented), [baste64](https://crates.io/crates/baste64) (WASM SIMD). With the exception of the `base64` crate, none of these publish Kani or MIRI verification of their `unsafe` code, as far as we could find.

## Acknowledgements

The encode/decode kernels build directly on techniques published by others, all under permissive licenses:

*   **[Alfred Klomp](https://github.com/aklomp) — [`aklomp/base64`](https://github.com/aklomp/base64) (BSD-2-Clause).** Our decoder's nibble-lookup validation (`lut_lo`/`lut_hi`/`lut_roll`) and our encoder's offset-load loop (avoiding a per-iteration lane permute) and single-LUT character mapping are direct ports of techniques from this library. The URL-safe variants of these tables aren't published anywhere we could find — we re-derived them following the same construction method and verified them exhaustively (see `src/simd/avx2.rs`).
*   **[Daniel Lemire](https://github.com/lemire) and Wojciech Muła — [`lemire/fastbase64`](https://github.com/lemire/fastbase64) (BSD-2-Clause).** Its `fastavxbase64.c` independently documents and credits the same nibble-lookup decode algorithm (originated by Muła, with the `+`/`/` disambiguation trick credited there to `@aqrit`), which we cross-referenced while implementing ours.
*   **[`base64-simd`](https://crates.io/crates/base64-simd) (MIT).** Its existence, benchmarks, and API design were a useful reference point throughout.

**We love open source. None of this would exist without people willing to share what they figured out.**

## License

Licensed under either the [MIT License](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-MIT) or the [Apache License, Version 2.0](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-APACHE) at your option.
