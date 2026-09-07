# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, a minor bump (`0.x`) is where breaking changes land; patch releases stay
compatible.

## 0.4.0

### Changed — breaking

* **License is now 0BSD**, replacing `MIT OR Apache-2.0`. 0BSD drops the
  attribution requirement entirely; anyone already using the crate under the old
  terms is unaffected.
* **Slice APIs renamed** for consistency with the rest of the ecosystem:
  * `Engine::encode_into` → `Engine::encode_slice`
  * `Engine::decode_into` → `Engine::decode_slice`
  * `Engine::estimate_decoded_len` → `Engine::decoded_len_estimate`

### Added

* **Custom alphabets.** `Alphabet::new` accepts any 64 distinct printable-ASCII
  characters (no `=`) and is a `const fn`, so its ~12 KiB of lookup tables are
  built at compile time into a `static`; `Engine::custom` builds an engine over
  one. The scalar and AVX-512 VBMI kernels are pure table lookups driven by the
  alphabet, so a custom set runs there at built-in speed; AVX2 and NEON derive
  characters arithmetically from the RFC 4648 layout and fall back to scalar.
* **Padding-indifferent engines.** `STANDARD_PAD_INDIFFERENT` and
  `URL_SAFE_PAD_INDIFFERENT` pad when encoding but accept padded or unpadded
  input when decoding. Their decode path is scalar only — the vector kernels'
  tail owns one fixed padding rule.
* **Convenience APIs:** free `encode`/`decode` functions over `STANDARD`,
  `Engine::encode_string`, `Engine::decode_vec`, `Engine::alphabet`, and
  `Engine::encode_padding`.
* **AVX-512 VBMI raw accessors** under the `unstable` feature:
  `Engine::encode_avx512_vbmi` and `Engine::decode_avx512_vbmi`.

### Performance

* **AVX-512 VBMI kernel reworked** around `vpermb` / `vpermi2b` /
  `vpmultishiftqb`: encode is three ops per 48-byte vector, decode folds
  validation into one `vpternlogd` OR tree. The remainder now runs through masked
  vector passes, and the final group is decided inline rather than by entering
  the scalar kernel — worth +21–27% on small decodes, which were paying a flat
  ~20 cycles per call for that handoff.
* **Non-temporal store gate is now derived from the last-level cache** (5/8 of
  it, via `CPUID`) rather than a flat length, in *both* x86 kernels. A fixed
  threshold streams into a destination that would have stayed resident on any
  machine with a larger cache than the one it was tuned on; measured on Zen 5,
  that cost −39% encode / −36% decode at 1 MiB. Each kernel keeps its old
  constant as a floor, so the gate can only move up.
* **AVX2 and scalar kernels retuned.** Scalar encode gained a two-block main
  loop (+6% at 384 B, +10% from 4 KiB up); AVX2 register pressure and dispatch
  thresholds were re-measured on Coffee Lake.
* **Dispatch thresholds re-measured through the public API** rather than by
  racing kernels in a loop, which flatters the vector path by keeping its lookup
  vectors and branch history hot. VBMI now enters at 16 bytes (encode) and 28
  characters (decode).

### Fixed

* `Engine::encode_neon`'s `# Safety` section documented the *decoder's* buffer
  contract — an under-estimate a caller could have acted on. It now states the
  encoded-length requirement.
* `Engine::decode_avx512_vbmi` demanded `(input.len() / 4 + 1) * 3` bytes of
  destination capacity while also saying to size it with
  `decoded_len_estimate`, which is smaller. The stricter bound was a leftover
  from an earlier store layout: a quad step never writes past the 192 bytes it
  produces, so the estimate is sufficient. Same leftover removed from
  `Engine::decode_neon`.
* `Error::InvalidCharacter` claimed to cover misplaced `=`. Which of
  `InvalidLength` and `InvalidCharacter` a malformed pad produces actually
  depends on which kernel met it; both variants now say so, and that it is
  deliberately unspecified.
* `Engine::decode_slice` now documents that `output` must be at least
  `decoded_len_estimate` — the conservative bound — not the exact decoded length.
* `Engine::encoded_len` saturates at `usize::MAX` instead of wrapping, and says
  so. The bound is unreachable for a real slice — one is at most `isize::MAX`
  bytes and 4/3 of that still fits in a `usize` — so only a fabricated
  `input_len` can reach it, and the slice APIs turn that into
  `Error::BufferTooSmall`. The return type stays `usize`.
* The `Alphabet` doc example described the bcrypt alphabet as having "digits
  before letters"; it has them after.

### Verification

* **Kani proofs extended to the AVX-512 VBMI kernel**, including symbolic index
  and induction proofs for the streaming tier's alignment peel.
* Kani harnesses now check both vector kernels against `src/simd/refcodec.rs`, a
  naive safe-Rust transcription of RFC 4648 that shares no code with either and
  is itself pinned to the scalar kernel by a test.
* 328M `cargo-fuzz` executions on `c8a.large` with no crashes, timeouts or OOMs;
  the accumulated corpus is checked in under `fuzz/corpus/`.

### Internal

* Turbo-Base64 (C) is now measured by the same harness in the same session,
  rather than cited from its own published numbers.
* The published package no longer ships `benches/`. The bench harness needs the
  `tb64-sys` path dev-dependency, which cargo strips on publish, so the shipped
  bench could not have compiled for anyone who downloaded the crate.
* CI gained an MSRV job (pinned to the `rust-version` in `Cargo.toml`, which is
  itself pinned by the rustc that `cargo kani` ships) and a `rustdoc` job, so the
  denied `rustdoc` lints are actually enforced before a release rather than on
  docs.rs after one.

## 0.3.0

Released from commit `4e6caf0`, untagged. Highlights, reconstructed from the
history: the scalar kernel was ported to fully safe Rust and the crate now
carries `#![forbid(unsafe_code)]` when no SIMD kernel is enabled; verification
logic was split out of the kernels into per-backend `verify` modules; the AVX2
kernel was refactored and its Kani stubs fixed; the raw AVX-512 accessors were
removed pending the rework that landed in 0.4.0.

## Earlier releases

`0.1.0` through `0.2.0` predate this file. No tags exist for them — see the
commit history for what changed.
