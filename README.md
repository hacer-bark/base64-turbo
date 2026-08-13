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
| **AVX2** | ✅ | ✅ | ✅ (CI, unbounded) | ✅ |
| **AVX512** (`avx512f`+`avx512bw`) | ✅ | ✅ | ✅ (local only) | ✅ |
| **AVX512-VBMI** (`vpermb`/`vpermi2b`) | ✅ | ✅ | ❌ | ✅ |
| **NEON** | ✅ | ✅ | ❌ | ❌ |

*   **Kani** — the model checker proves the kernels do not panic, do not read or write out of bounds, and round-trip exactly. For AVX2 the bounds result holds for *every* input length, by a machine-checked induction; for the other paths it holds at the lengths the harnesses pin. Details below.
*   **MIRI** — checks for Undefined Behavior (strict provenance, alignment, OOB pointer arithmetic, data races) on the inputs it runs. Every distinct code path — single-vector loop, quad-vector loop, scalar tail — is exercised for Scalar, AVX2, AVX512 and AVX512-VBMI. This is branch coverage, not exhaustive input coverage.
*   **MSan** — the whole standard library is rebuilt with instrumentation (`-Z build-std -Z sanitizer=memory`) to confirm we never branch on or emit uninitialized memory, which matters given how much AVX512 masking we do.
*   **Fuzzing** — 2.5B+ `cargo-fuzz` iterations across all paths, no crashes to date.

### How the Kani proofs work

Kani is a bounded model checker, so a proof that runs the real kernel over symbolic *bytes* has to pin the length — the loops get unwound, and the cost grows with them. Pin one length and you have proven one length. That is the trap most "formally verified" claims fall into, and for AVX2 we split the problem in two to get out of it.

**The index proofs** throw away the vectors entirely. Every load and store bound in the kernels is a statement about `rounds`, `remaining` and the `src`/`dst` offsets — plain `usize` arithmetic that no input byte influences. Those proofs run over a **fully symbolic `len`** and, critically, over an **arbitrary iteration index** rather than an unwound sequence of them:

```rust,ignore
let done: usize = kani::any();              // ANY iteration, not iteration 0..n
kani::assume(done >= 1 && done <= rounds);
kani::assume(rounds - done >= 4);           // the quad loop's guard still holds

let (src_off, dst_off) = enc_state(done);
assert!(src_off + 72 + 32 <= len);          // widest read in the body
assert!(dst_off + 96 + 32 <= cap);          // widest write in the body

assert_eq!((src_off + 96, dst_off + 128), enc_state(done + 4));  // step preserved
```

That last line is the inductive step, machine-checked: an arbitrary state satisfying the invariant produces a successor that satisfies it too. With a base case (`check_enc_first_block`) and an exit case (`check_enc_tail_handoff`) either side of it, the result covers **every length a Rust slice can have**, and it costs the solver almost nothing because no vector ever appears. The decoder gets the same treatment, including the fact that `pack_and_store!` touches a 28-byte span while advancing only 24 — a 4-byte overhang past every block that is now proven to stay inside the caller's buffer instead of being assumed to.

**The kernel proofs** then do what only symbolic bytes can: run the real code with `kani::any()` input to prove the character mapping, the validation LUTs and the absence of panics. Because the index proofs own the loop arithmetic, these no longer have to demonstrate any of it, so they only need to reach each distinct kernel once. That let us cut them down rather than grow them: the encoder harness went from 53 bytes to 37, the decoder from 69 to 37, and the 125-byte quad-tier roundtrip — by far the most expensive harness — was **deleted**, because the quad tier runs the same kernel as the single tier and everything that distinguishes it is offset arithmetic that is now proven for every iteration rather than one. Total solver time went down while the claim got stronger.

Destination buffers in those harnesses are sized to **exactly** what the public API guarantees (`encoded_len` for encode, `estimate_decoded_len` for decode), so a kernel that overruns its real output by even one byte fails the proof. Alphabets are split into separate harnesses rather than taken symbolically: two lean solver runs beat one that needs a bigger machine.

### What still rests on human judgment

Two things, and they should be the first things an auditor attacks:

1.  **The index proofs mirror the loop arithmetic; they do not execute it.** Roughly a dozen lines of model sit beside the real loops, each constant annotated with the operation it mirrors. If someone edits a stride without editing the model, the proofs keep passing. Treat those constants as part of the code.
2.  **The proofs run against models of the AVX2 instructions, not the instructions.** Kani cannot execute SIMD, so each intrinsic is a Rust transcription of the Intel Intrinsics Guide pseudocode. That is now checked rather than trusted: `avx2_stub_equivalence` runs every model against the real instruction on real hardware under plain `cargo test`, over saturation and sign boundaries, shuffle-index patterns and deterministic noise. It cannot prove the models agree everywhere, but it catches transcription errors, which is the realistic failure mode.

The same split has not yet been applied to AVX512, AVX512-VBMI or NEON — for those, the older caveats stand.

### Kani in CI vs. locally

`verification.yml` runs the Scalar and AVX2 harnesses: the index proofs together in one job, each kernel proof on its own runner. The AVX512 proof is **not** in CI — its state space exceeded GitHub Actions' time and memory budget. It is re-run locally before each release and passes. The harness is checked in, so you can reproduce it:

```sh
cargo kani --unstable stubbing --harness kani_verification_avx512
```

AVX512-VBMI has no proof at all: it depends on `vpermb` / `vpermi2b`, for which we haven't written models yet. NEON has no Kani harness either. Both rest on MIRI, MSan and fuzzing — a real bar, but a lower one than the proven paths.

### Verification FAQ

**The crate uses `unsafe`. How is that "safe"?** We separate "Safe Rust" (compiler-checked) from "memory-safe" (proven/checked by the layers above). For the Kani-proven paths, no input through the public API can trigger an overflow, segfault or panic, subject to the caveats above.

**Can I crash it with garbage input?** Not through the safe API. Invalid Base64, binary noise, or hostile payloads return `Err`. The decoder-robustness harnesses feed the decoder fully symbolic garbage and prove it never panics.

**What if I misuse the `unstable` internals?** Then the crash is yours. Those functions exist to skip bounds checks when every cycle matters; we verify that *our* safe API upholds their contracts, not that you will.

**Is it production-ready?** Scalar, AVX2 and plain AVX512 are Kani-proven plus MIRI/MSan/fuzz-clean. AVX512-VBMI and NEON have everything except Kani. We ship both; you should know which one your CPU picks.

**How do I know your intrinsic models are right?** For AVX2, `cargo test` runs every model against the real instruction on real hardware and fails if they disagree. For the other paths, you read them against the Intel guide — they're written for exactly that, but nothing cross-checks them yet.

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

Licensed under either of

- [Apache License, Version 2.0](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-APACHE)
- [MIT license](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE-MIT)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this crate, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
