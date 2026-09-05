<div align="center">
  <h1>Base64 Turbo</h1>
  <p><strong>A Rust Base64 codec that peaks past 100 GiB/s, with its <code>unsafe</code> SIMD checked by a model checker, not just by review.</strong></p>

  [![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg?style=for-the-badge&color=fc8d62)](https://crates.io/crates/base64-turbo)
  [![License](https://img.shields.io/badge/license-0BSD-8da0cb.svg?style=for-the-badge)](#license)
  [![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/verification.yml?label=Kani%20Verified&style=for-the-badge&color=e78ac3)](https://github.com/hacer-bark/base64-turbo/actions/workflows/verification.yml)
  [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified&style=for-the-badge&color=66c2a5)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)
</div>

<br/>

`base64-turbo` targets high-throughput systems where CPU cycles are scarce and Undefined
Behavior is unacceptable. It picks the best kernel available at runtime:

* **x86_64:** AVX-512 VBMI or AVX2, via runtime CPU detection.
* **ARM (aarch64):** NEON, via compile-time dispatch — no detection overhead.
* **Other:** an optimized table-driven scalar kernel, in 100% safe Rust.

<img alt="Base64 throughput by payload size on AWS c8a.large (AMD EPYC 9R45) — base64-turbo peaks above 100 GiB/s for both encode and decode" src="benches/results/throughput.png">

<p align="center"><sub>AWS <code>c8a.large</code> (AMD EPYC 9R45). See <a href="#benchmarks">Benchmarks</a>.</sub></p>

The 100+ GiB/s figures are the peak of the sweep above (4 KiB encode, 64 KiB decode), not
a sustained number at every size — [Benchmarks](#benchmarks) has the full curve and how to
reproduce it, and [Safety & Verification](#safety--verification) says exactly what's
proven and what still rests on human judgment.

If you need WASM SIMD, stable NEON, or a dozen encodings in one crate, this isn't that
crate — see the [FAQ](#faq).

## Contents

- [Quick Start](#quick-start)
- [Zero-Allocation API](#zero-allocation-stack--no_std)
- [Custom Alphabets](#custom-alphabets)
- [Feature Flags](#feature-flags)
- [Compatibility & Stability](#compatibility--stability)
- [Performance & Architecture](#performance--architecture)
- [Benchmarks](#benchmarks)
- [Safety & Verification](#safety--verification)
- [Ecosystem](#ecosystem)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Quick Start

```rust
use base64_turbo::STANDARD;

let data = b"Speed and Safety";
let encoded = STANDARD.encode(data); // String
assert_eq!(encoded, "U3BlZWQgYW5kIFNhZmV0eQ==");

let decoded = STANDARD.decode(&encoded).unwrap(); // Vec<u8>
assert_eq!(decoded, data);
```

`URL_SAFE` is the same engine with the RFC 4648 §5 alphabet.

### Zero-Allocation (Stack / `no_std`)

For hot paths where heap allocation is too slow, write directly to stack buffers — the
slice APIs need no allocator. Size the buffers with `Engine::encoded_len`/
`Engine::decoded_len_estimate` rather than guessing:

```rust
use base64_turbo::STANDARD;

let input = b"Low Latency";

let mut enc_buf = vec![0u8; STANDARD.encoded_len(input.len()).unwrap()];
let enc_len = STANDARD.encode_slice(input, &mut enc_buf).unwrap();

let mut dec_buf = vec![0u8; STANDARD.decoded_len_estimate(enc_len)];
let dec_len = STANDARD.decode_slice(&enc_buf[..enc_len], &mut dec_buf).unwrap();

assert_eq!(&dec_buf[..dec_len], input);
```

### Custom Alphabets

Any 64-character set works, not just the two RFC 4648 ones. `Alphabet::new` is a `const
fn`, so the ~12 KiB of lookup tables it derives are built at compile time into a `static`
and cost nothing at run time:

```rust
use base64_turbo::{Alphabet, Engine};

// bcrypt / crypt(3): `.` and `/` first, digits after the letters.
static BCRYPT: Alphabet = match Alphabet::new(
    b"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
) {
    Some(a) => a,
    None => unreachable!(),
};
static ENGINE: Engine = Engine::custom(&BCRYPT, false); // false = no `=` padding

assert_eq!(ENGINE.encode(b"hello"), "YETqZE6");
assert_eq!(ENGINE.decode("YETqZE6").unwrap(), b"hello");
```

`Alphabet::new` returns `None` unless all 64 characters are distinct, printable ASCII
(`0x21`–`0x7E`) and none of them is `=`, which stays the padding character for every
alphabet. The ASCII bound is not cosmetic: it is what lets the AVX-512 VBMI decoder
validate a whole 64-byte vector with one `vpermi2b` against a 128-entry table.

Which kernels a custom alphabet gets:

| Kernel | Standard / URL-safe | Custom alphabet |
| :--- | :---: | :---: |
| Scalar | ✅ | ✅ |
| AVX512-VBMI | ✅ | ✅ |
| AVX2 | ✅ | ❌ |
| NEON | ✅ | ❌ |

Scalar and AVX-512 VBMI are pure table lookups and the tables come out of the `Alphabet`,
so a custom alphabet runs there at exactly built-in speed. AVX2 and NEON compute
characters *arithmetically* from the RFC 4648 layout, so they can't serve an arbitrary
one. Passing the standard or URL-safe characters to `Alphabet::new` is recognized as such,
so those keep every kernel and match `STANDARD` / `URL_SAFE` byte for byte.

## Feature Flags

Each x86 SIMD kernel is its own knob, so you compile in only what your target CPUs are
likely to support. Runtime detection still gates every call — enabling a kernel the host
lacks just falls back to scalar.

| Feature | Default | Description |
| :--- | :---: | :--- |
| `std` | **Yes** | `String`/`Vec` support. Disable for `no_std` (the slice APIs need no allocator). |
| `avx2` | **Yes** | AVX2 kernel + runtime detection on x86/x86_64. Implies `std`. |
| `avx512-vbmi` | **Yes** | AVX-512 VBMI fast-path kernel on x86/x86_64. Implies `std`. |
| `simd` | **Yes** | Convenience meta-feature — turns on `avx2` + `avx512-vbmi` at once. |
| `neon` | **Yes** | NEON acceleration on aarch64. No `std` required. |
| `unstable` | **No** | Exposes the raw internal kernels (`encode_avx2`, `encode_avx512_vbmi`, `encode_neon`, …). The `*_scalar` accessors are **safe** (they may panic on a too-small buffer, but never invoke UB). |

Scalar-only builds are `#![forbid(unsafe_code)]`. Disable every SIMD kernel and the crate
is pure scalar Rust — nothing to verify, nothing to audit. The allocating `encode`/`decode`
swap their uninitialized-buffer fast path for a zero-filled, fully-checked one in this
configuration.

## Compatibility & Stability

**MSRV:** Rust 1.89.0. We rely on recently stabilized AVX-512 intrinsics in `core` and do
not plan to lower this.

**API stability:** The public API is **Stable** and follows Semantic Versioning. It stays
valid and backward-compatible throughout the `0.3.x` lifecycle.

Output conforms to RFC 4648 — `STANDARD` and `URL_SAFE` are drop-in compatible with the
`base64` crate, and [custom alphabets](#custom-alphabets) match its `GeneralPurpose`
engine built over the same characters.

## Performance & Architecture

<details>
<summary>Why is it fast — per-kernel breakdown</summary>

The design goal is maximum throughput *within* Rust's safety guarantees: vectorized data
movement instead of byte-at-a-time lookup tables. We batch 32–64 bytes per register and
push padding/error detection to bitmasks *after* the vector op, so the hot loop stays
branchless.

* **Scalar (wide tables).** 100% safe Rust, `#![forbid(unsafe_code)]`. Encode maps 12
  input bits directly to the two characters they produce (4 lookups per 6-byte block
  instead of 8); decode folds each character's bit-shift into the table itself, so a
  4-character group is four loads OR-ed together and validation falls out of the same OR.
  All four tables hang off the engine's `Alphabet`, so selecting one is a pointer load
  rather than a branch — and an arbitrary alphabet costs nothing extra.
* **AVX2.** `vpshufb` shuffles contend for port 5, so AND/OR/shift work is interleaved
  onto ports 0/1/5 to keep the shuffle port from bottlenecking. 256-bit registers behave
  as two 128-bit lanes, which a sliding bit-stream must cross — bridged with an offset
  load plus a permute instead of dropping to scalar.
* **AVX512-VBMI.** The fastest path we have. `k`-mask registers let the 1–31 byte tail
  run as a single masked vector op instead of a scalar fallback, and 32 `zmm` registers
  (vs 16 `ymm`) keep every LUT resident while unrolling harder. Encode is three ops for
  48 bytes: a `vpermb` gather, one `vpmultishiftqb` that extracts all eight 6-bit fields
  at once, then a `vpermb` through the alphabet. Decode looks up characters with
  `vpermi2b` across a 128-byte reverse LUT and folds validity into a single `vpternlogd`
  OR tree. Both permutes read their control vectors straight out of the `Alphabet`, which
  is why [custom alphabets](#custom-alphabets) run here at full speed.
* **NEON.** 128-bit `q` registers, 12→16 bytes per encode step. `vqtbl1q_u8` gives the
  same shuffle primitive as `vpshufb`, with full cross-lane access, so no lane-stitching
  is needed. Mandatory on ARMv8-A, hence compile-time dispatch.
* **Dispatch.** x86 picks AVX-512 VBMI → AVX2 → scalar at runtime (guarding against
  `SIGILL`); aarch64 picks NEON → scalar at compile time. A custom alphabet skips the
  AVX2 and NEON arms, since those two derive characters from the RFC 4648 layout.

</details>

## Benchmarks

Straight `cargo bench` output (`benches/encoding_bench.rs`) — same numbers charted at the
top of this README, no cherry-picking.
[criterion.rs](https://github.com/bheisler/criterion.rs), 5 s warm-up, 15 s measurement
per group. Input sizes span 32 B → 10 MB to cross L1/L2/RAM boundaries. `std` is the
`base64` crate on default features, which since 0.23 means its `simd-unsafe` path
(AVX2/NEON with runtime detection) is active — the number most callers of that crate
actually get.

At 64 KiB, both boxes, encode / decode:

| Machine | base64-turbo | `base64-simd` | `base64` (std) |
| :--- | ---: | ---: | ---: |
| `c8a.large` (AMD EPYC 9R45) | 81.1 / 107.5 GiB/s | 14.6 / 14.6 GiB/s | 9.8 / 25.6 GiB/s |
| `c7i.large` (Xeon Platinum 8488C) | 30.9 / 46.7 GiB/s | 11.1 / 10.2 GiB/s | 6.9 / 13.2 GiB/s |

**AWS `c8a.large`, the chart above:** the sweep peaks at 105.6 GiB/s encode (4 KiB) and
107.5 GiB/s decode (64 KiB) — both past 100 GiB/s, single-threaded, from a
Kani/MIRI/MSan-checked kernel rather than an unaudited one. Small-input latency (32 B,
zero-alloc `_into`): ~7.4 ns encode, ~10.0 ns decode.

The two machines agree on the shape of the curve (same dip past L2/L3, same ordering of
libraries) and disagree by roughly 2-3x on the absolute ceiling — which is itself the
point: the 100+ GiB/s number is a real peak on real hardware, not a property of the
algorithm that holds everywhere.

<details>
<summary>AWS <code>c7i.large</code> chart — smaller/cheaper box, same methodology</summary>

<img alt="Base64 throughput by payload size on AWS c7i.large (Intel Xeon Platinum 8488C) — a smaller instance run with the same methodology" src="benches/results/throughput-c7i.png">

Small-input latency (32 B, zero-alloc `_into`): ~10 ns encode, ~13 ns decode.

</details>

Reproduce it:

```bash
sudo apt update && sudo apt install -y build-essential git
curl --proto '=https' --tlsv1.3 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

git clone https://github.com/hacer-bark/base64-turbo
cd base64-turbo
BENCH_TARGET=all cargo bench 2>&1 | tee benches/results/raw.txt
python3 benches/scripts/plot_bench.py benches/results/raw.txt
```

Select comparison targets with `BENCH_TARGET` (comma-separated): `turbo` (default,
allocating API), `turbo-buff` (zero-allocation API), `simd`, `std`, `all`.

Raw `cargo bench` output — 32 B through 10 MB, every target — is checked in:
[`c8a-large-latest.txt`](benches/results/c8a-large-latest.txt) and
[`c7i-large-latest.txt`](benches/results/c7i-large-latest.txt).

## Safety & Verification

**Philosophy:** `Safety > Performance > Convenience`. We use `unsafe` SIMD intrinsics and
raw pointer arithmetic, so rather than rely on review alone we stack independent layers
that cover each other's blind spots.

| Architecture | MIRI | MSan | Kani | Fuzzing |
| :--- | :---: | :---: | :---: | :---: |
| **AVX2** | ✅ | ✅ | ✅ | ✅ |
| **AVX512-VBMI** | ✅ | ✅ | ✅ | ✅ |
| **NEON** | ✅ | ✅ | ❌ | ❌ |

* **Kani** proves the kernels don't panic, don't read/write out of bounds, and agree —
  bytes decoded and inputs rejected alike — with a naive safe-Rust transcription of
  RFC 4648 (`src/simd/refcodec.rs`), which shares no code with either kernel and is itself
  pinned to the scalar kernel by a test. For AVX2 and AVX512-VBMI the bounds result holds
  for *every* input length, by a machine-checked induction over the loop's offset
  arithmetic.
* **MIRI** catches Undefined Behavior (provenance, alignment, OOB pointer arithmetic,
  data races) on every distinct code path — single-vector loop, wide unrolled loop,
  masked tail, scalar tail, non-temporal tier — for Scalar, AVX2 and AVX512-VBMI. Branch
  coverage, not exhaustive input coverage.
* **MSan** rebuilds the standard library with instrumentation
  (`-Z build-std -Z sanitizer=memory`) to confirm we never branch on or emit
  uninitialized memory, which matters given how much AVX512-VBMI masking we do.
* **Fuzzing** — 250M+ `cargo-fuzz` iterations across all paths, no crashes to date.

<details>
<summary>Where each layer stops — exclusions worth naming</summary>

**Kani.** Three paths are proved by arithmetic but never *executed* by a proof: AVX2's
non-temporal store path (it needs a 4 MiB input, far past what a model checker can
unwind, so its 16-byte alignment precondition rests on a hardware test instead),
AVX512-VBMI's 4×-unrolled quad tiers (256 symbolic characters through four `vpermi2b`
lookups is out of CBMC's reach), and AVX512-VBMI's non-temporal tier — whose alignment
peel, 64-byte store alignment invariant and "the peel cannot starve the loop" property
are all proved symbolically (`check_vbmi_enc_stream_peel`, `check_vbmi_dec_stream_peel`,
`check_vbmi_*_stream_step`), but at a 512 KiB gate no harness can execute one. In each
case the offsets are proved for every length; it is the *contents* no harness checks.

**MIRI, on AVX512-VBMI specifically.** The three byte permutes (`vpermb`, `vpermi2b`,
`vpmultishiftqb`) are swapped for the Intel-pseudocode models under `cfg(miri)`, so the
Miri leg checks the kernel's *addressing* against real semantics and its *arithmetic*
against those models — it is not an independent check of the models. And the streaming
stores become ordinary stores under Miri, so Miri cannot fault on a misaligned
`vmovntdq`; the thresholds are lowered under `cfg(miri)` so the tier and its peel are
still reached and their addressing checked, with the alignment itself carried by the Kani
proofs plus a runtime re-test on the loop guard.

</details>

<details>
<summary>What still rests on human judgment</summary>

1. The index proofs that make the bounds hold for every length mirror the loops' offset
   arithmetic; they don't execute it. Every stride is imported from the kernel module
   rather than restated in the proof, so a stride can't change under a proof without
   changing it too — but the *shape* of the model is still hand-written, and a
   restructured loop needs a restructured proof.
2. Kani can't execute SIMD, so each intrinsic it meets is a line-by-line Rust
   transcription of the Intel Intrinsics Guide pseudocode. `avx2_stub_equivalence` and
   `avx512_vbmi_stub_equivalence` (`cargo test`) run every model against the real
   instruction on real hardware, each skipping if the host lacks the subset. They catch
   transcription errors; they don't prove the models agree everywhere.
3. Those two suites are the only thing that ever executes a real instruction, and
   GitHub's hosted runners do not reliably have AVX-512 VBMI — so on many CI runs
   **nothing executes a real VBMI instruction at all**. The `simd-avx512-vbmi` job
   reports which case a given run was, and fails if a VBMI runner somehow skips anyway,
   but a green tick is not by itself evidence that the models were checked against
   silicon that run.
4. Kani harnesses run only if `verification.yml` names them, and that list is
   hand-maintained. The round-trip harnesses (`check_vbmi_roundtrip_standard`,
   `check_avx2_roundtrip_*`) are deliberately not in it — they carry the encoder's
   symbolic output through the decoder, far more state than starting from free bytes,
   and the `matches_ref` harnesses that replaced them in CI are the stronger property
   anyway. Run them by hand when either kernel changes shape.
5. NEON has no Kani harness at all, and rests on MIRI, MSan and fuzzing.

Read the [CI logs](https://github.com/hacer-bark/base64-turbo/actions) and the `unsafe`
blocks themselves — each documents the contract it relies on.

</details>

<details>
<summary>Raw <code>cargo fuzz</code> output — AWS <code>c8a.large</code>, 1h run, <code>-fork=2</code></summary>

See [`fuzz/logs/2026-08-16-c8a-large.log`](fuzz/logs/2026-08-16-c8a-large.log) for the
full unedited `libFuzzer` output: 328,814,323 executions in 3606s, `oom/timeout/crash: 0/0/0`
throughout. The corpus this run accumulated is checked in at
[`fuzz/corpus/`](fuzz/corpus/) so the coverage is reproducible and future runs start
from it instead of from scratch.

</details>

## Ecosystem

| Library | Lang | SIMD | Verified `unsafe` | Encode (64 KiB) | Decode (64 KiB) | Source |
| :--- | :---: | :---: | :---: | ---: | ---: | :--- |
| **base64-turbo** | Rust | ✅ | ✅ Kani + MIRI + MSan + Fuzz | 27.1 GiB/s | 34.6 GiB/s | our bench, same box † |
| [Turbo-Base64](https://github.com/powturbo/Turbo-Base64) | C | ✅ | ❌ | 18.4 GiB/s | 37.8 GiB/s | our bench, same box † |
| [base64](https://crates.io/crates/base64) (std) | Rust | ✅ (0.23+) | ✅ MIRI + Fuzz | 6.9 GiB/s | 13.2 GiB/s | our bench |
| [base64-simd](https://crates.io/crates/base64-simd) | Rust | ✅ | ❌ | 11.1 GiB/s | 10.2 GiB/s | our bench |
| [base64-ng](https://crates.io/crates/base64-ng) | Rust | ✅ | ❌ | — | — | not yet benched |
| [aklomp/base64](https://github.com/aklomp/base64) | C | ✅ | ❌ | 24.4 GiB/s | 21.0 GiB/s | vendor bench |
| [fastbase64](https://github.com/lemire/fastbase64) | C | ✅ | ❌ | 22.1 GiB/s | 19.8 GiB/s | vendor bench |

All Rust rows (except `base64-ng`) and the Turbo-Base64 row are ours, measured on the same
AWS `c7i.large` in the same session, pinned to one core. The aklomp/base64 and fastbase64
rows are the vendors' own published numbers on an Intel i7-9700K from 2022
([source](https://github.com/powturbo/Turbo-Base64#benchmark-incl-the-best-simd-base64-libs),
decimal MB/s converted to GiB/s), unreproduced by us — treat those two as directional only.

† — For Turbo-Base64 we cloned its real upstream C source, built it with its own official
per-kernel flags (its `tb64v512vbmi` kernel auto-selects on this CPU, confirmed at
runtime), verified our harness round-trips and rejects corrupt input the same as its own
checked decode, and ran both back to back. But the C-side timing harness is ours, not
theirs — one measurement session, not the statistical rigor criterion gives the Rust
numbers, so treat the margins as directional. On that comparison we're ahead on both
directions — wide on encode, narrow on decode — which is close enough that we no longer
assume unchecked C is automatically ahead, without claiming a general win either.

How the Rust alternatives compare:

* **`base64` (std)** added a SIMD path in 0.23 (default-on `simd-unsafe`, AVX2/NEON with
  runtime detection), so it's no longer the zero-`unsafe` scalar crate it used to be. It
  publishes MIRI and fuzz coverage for that path, but no Kani or MSan.
* **`base64-simd`** is a strong crate that raised the bar before us; we measure faster
  overall on this box and publish Kani/MIRI/MSan we couldn't find for it.
* **`base64-ng`** is a newer entrant we haven't benchmarked — no speed claim either way.
* **The C libraries** still get real advantages from unchecked pointer arithmetic and no
  published verification (`turbo-base64` is also GPLv3, against our 0BSD) — pick them if
  you need the absolute ceiling on unfamiliar hardware and will own the risk.

Also in the space: [vb64](https://crates.io/crates/vb64) (unmaintained),
[base-d](https://crates.io/crates/base-d) (33+ alphabets, decode-only SIMD),
[webbuf](https://crates.io/crates/webbuf) and [baste64](https://crates.io/crates/baste64)
(WASM-oriented). None of these publish Kani or MIRI verification of their `unsafe` code,
as far as we could find.

## FAQ

**Why no SSE, WASM, or other SIMD backends?**
We optimize for one target class — x86 with AVX2 or AVX-512 VBMI — rather than spreading
across every instruction set a CPU might expose. Every extra backend is another kernel to
prove safe and another surface for a transcription bug to hide in. If you need SIMD
everywhere, look elsewhere.

**Is NEON production-ready?**
No. It compiles and passes MIRI/MSan/tests, but it has none of the symbolic Kani proofs
that cover AVX2 and AVX512-VBMI, and CI doesn't run it on real ARM hardware yet. Treat it
as best-effort; it may be deprecated in a future release.

**Does this replace the `base64` crate?**
For most callers, yes — `STANDARD` and `URL_SAFE` are drop-in RFC 4648 compatible, and
[custom alphabets](#custom-alphabets) are supported too. The difference is throughput and
verification depth (see [Ecosystem](#ecosystem)), not API surface. If you need streaming
`Read`/`Write` adapters or don't care about the last 20-80 GiB/s, the `base64` crate is a
perfectly reasonable, smaller dependency.

**Why is `unsafe` acceptable here at all?**
Because vectorized Base64 can't be written in safe Rust at these throughputs — the SIMD
intrinsics themselves require `unsafe`. Our answer is to prove it correct with
independent tools (Kani, MIRI, MSan, fuzzing) instead of asking you to trust code review.
Scalar-only builds drop `unsafe` entirely (`#![forbid(unsafe_code)]`).

**What happens on a CPU without AVX2 or AVX-512 VBMI?**
Runtime detection falls back to the scalar kernel automatically — no crash, no feature
gating at the call site. You lose throughput, not correctness or safety.

## Acknowledgements

The encode/decode kernels build on techniques published by others, all under permissive
licenses:

* **[Alfred Klomp](https://github.com/aklomp) — [`aklomp/base64`](https://github.com/aklomp/base64) (BSD-2-Clause).**
  Our decoder's nibble-lookup validation and our encoder's offset-load loop and
  single-LUT character mapping are direct ports from this library. The URL-safe tables
  aren't published anywhere we could find — we re-derived them and verified them
  exhaustively (`src/simd/avx2/mod.rs`).
* **[Daniel Lemire](https://github.com/lemire) and Wojciech Muła — [`lemire/fastbase64`](https://github.com/lemire/fastbase64) (BSD-2-Clause).**
  `fastavxbase64.c` independently documents the same nibble-lookup decode algorithm
  (originated by Muła, `+`/`/` disambiguation credited there to `@aqrit`), which we
  cross-referenced while implementing ours.
* **[`base64-simd`](https://crates.io/crates/base64-simd) (MIT).** Its benchmarks and API
  design were a useful reference point throughout.

## License

Licensed under the [0BSD license](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this crate shall be licensed as above, without any additional terms or conditions.
