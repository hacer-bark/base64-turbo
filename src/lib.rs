//! # Base64 Turbo
//!
//! [![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
//! [![License](https://img.shields.io/crates/l/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
//! [![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/verification.yml?label=Kani%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/verification.yml)
//! [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)
//!
//! **A Rust Base64 codec that peaks past 100 GiB/s, with its `unsafe` SIMD checked by a
//! model checker, not just by review.**
//!
//! `base64-turbo` targets high-throughput systems where CPU cycles are scarce and
//! Undefined Behavior is unacceptable. "Memory-safe" here is a specific, bounded claim:
//! the `unsafe` SIMD paths are checked by the [Kani](https://github.com/model-checking/kani)
//! model checker and [MIRI](https://github.com/rust-lang/miri) (a strict UB interpreter),
//! on top of `MemorySanitizer` audits and continuous fuzzing — see the "Safety &
//! Verification" section below for what each layer does and does not cover per
//! architecture. This crate is **not** faster than unchecked C/assembly implementations
//! and does not claim to be; within the narrower set of crates combining SIMD-accelerated
//! Base64 with Kani + MIRI verification, we are not aware of another one that reaches
//! AVX-512 VBMI speeds.
//!
//! It picks the best kernel available at runtime: **AVX-512 VBMI** or **AVX2** on
//! `x86_64` via runtime CPU detection, **NEON** on `aarch64` via compile-time dispatch,
//! and an optimized table-driven scalar kernel elsewhere, in 100% safe Rust. `no_std`
//! environments are supported.
//!
//! ### Basic API (Allocating)
//!
//! Standard usage for general applications. Requires the `std` feature (enabled by default).
//!
//! ```rust
//! # #[cfg(feature = "std")]
//! # {
//! use base64_turbo::STANDARD;
//!
//! let data = b"Hello world";
//!
//! // Encode to String
//! let encoded = STANDARD.encode(data);
//! assert_eq!(encoded, "SGVsbG8gd29ybGQ=");
//!
//! // Decode to Vec<u8>
//! let decoded = STANDARD.decode(&encoded).unwrap();
//! assert_eq!(decoded, data);
//! # }
//! ```
//!
//! ### Zero-Allocation API (Slice-based)
//!
//! For low-latency scenarios or `no_std` environments where heap allocation is undesirable.
//! These methods write directly into a user-provided mutable slice.
//!
//! ```rust
//! use base64_turbo::STANDARD;
//!
//! let input = b"Raw bytes";
//! let mut output = [0u8; 64]; // Pre-allocated stack buffer
//!
//! // Returns Result<usize, Error> indicating bytes written
//! let len = STANDARD.encode_slice(input, &mut output).unwrap();
//!
//! assert_eq!(&output[..len], b"UmF3IGJ5dGVz");
//! ```
//!
//! ### Custom Alphabets
//!
//! Any 64-character set works, not only the two RFC 4648 ones. [`Alphabet::new`] is a
//! `const fn`, so its lookup tables are built at compile time into a `static`:
//!
//! ```rust
//! use base64_turbo::{Alphabet, Engine};
//!
//! // bcrypt / crypt(3): `.` and `/` first, digits after the letters.
//! static BCRYPT: Alphabet = match Alphabet::new(
//!     b"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
//! ) {
//!     Some(a) => a,
//!     None => unreachable!(),
//! };
//! static ENGINE: Engine = Engine::custom(&BCRYPT, false);
//!
//! # #[cfg(feature = "std")] {
//! assert_eq!(ENGINE.encode(b"hello"), "YETqZE6");
//! # }
//! ```
//!
//! The scalar and AVX-512 VBMI kernels are table lookups driven by the alphabet, so a
//! custom one runs on both at full speed. The AVX2 and NEON kernels map characters
//! arithmetically from the RFC 4648 layout and cannot serve one, so those targets fall
//! back to the scalar kernel — see [`Engine::custom`].
//!
//! ### Padding
//!
//! [`STANDARD`] and [`URL_SAFE`] pad with `=` and require it when decoding;
//! [`STANDARD_NO_PAD`] and [`URL_SAFE_NO_PAD`] do neither. For input whose
//! padding is out of your control, [`STANDARD_PAD_INDIFFERENT`] and
//! [`URL_SAFE_PAD_INDIFFERENT`] pad on encode but accept either shape on decode
//! — at the cost of a scalar-only decode path, since the vector kernels' tail
//! owns one fixed padding rule. Encoding keeps every kernel.
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | **`std`** | **Yes** | `String`/`Vec` support. Disable for `no_std`; the slice APIs need no allocator, and every SIMD kernel works without it. |
//! | **`unstable`** | **No** | Exposes the raw internal kernels (`encode_avx2`, `encode_avx512_vbmi`, `encode_neon`, …). The `*_scalar` accessors are safe. |
//!
//! Which vector kernels are compiled in is **not** a feature: it follows from
//! the target. Every kernel the target can run is built, and runtime CPU
//! detection picks between them per call, so a kernel the host lacks simply
//! falls back to scalar.
//!
//! ## Selecting a Backend
//!
//! To narrow that — for a smaller binary, or for a build with no `unsafe` in it
//! at all — pass one `--cfg` in `RUSTFLAGS`:
//!
//! ```text
//! RUSTFLAGS='--cfg base64_turbo_backend="avx2"' cargo build
//! ```
//!
//! | Value | Effect |
//! |-------|--------|
//! | *(unset)* | Every kernel the target can run, chosen at run time. The default. |
//! | **`soft`** | No vector kernel. The crate is pure safe Rust and carries `#![forbid(unsafe_code)]` — memory safety holds by construction, with no `unsafe` anywhere to audit. |
//! | **`avx2`** | The AVX2 kernel only, dropping the larger AVX-512 VBMI one. |
//! | **`avx512`** | The AVX-512 VBMI kernel only. A CPU without VBMI then falls back to *scalar* rather than to AVX2, so pick this only for a fleet known to have it. |
//! | **`neon`** | The NEON kernel only; on `aarch64` that is what the default already selects. |
//!
//! A value naming a kernel the target cannot run leaves the build scalar rather
//! than failing, so one flag can cover a mixed-architecture workspace.
//!
//! Unlike a Cargo feature, a `RUSTFLAGS` `--cfg` applies to the whole dependency
//! graph and is not additive under feature unification — so this is a knob for
//! the final binary's build, not something a library should set on its
//! dependents' behalf.
//!
//! ## Safety & Verification
//!
//! We use `unsafe` SIMD intrinsics and raw pointer arithmetic, so rather than rely on
//! review alone we stack independent verification layers that cover each other's blind
//! spots:
//!
//! *   **Model checking (Kani):** For the Scalar and AVX2 kernels, Kani explores
//!     *every possible input byte value* at lengths chosen to exercise each loop tier and the
//!     scalar-tail handoff, proving the kernel does not panic, does not read or write out of
//!     bounds, and round-trips exactly. On AVX2 a second layer of proofs takes the loop
//!     arithmetic on its own, over an unbounded symbolic length and an arbitrary iteration, so
//!     the in-bounds result there is a machine-checked induction covering every length rather
//!     than the ones a harness happens to pin. The README spells out what that does and does not
//!     buy you, along with the AVX512-VBMI and NEON gaps.
//! *   **MIRI:** All SIMD paths (AVX512-VBMI, AVX2, NEON) and the scalar fallback run under
//!     **MIRI** (an Undefined Behavior interpreter) in CI, covering every distinct code path at
//!     least once.
//! *   **`MemorySanitizer`:** The standard library is rebuilt with instrumentation to confirm we
//!     never branch on or emit uninitialized memory.
//! *   **Fuzzing:** 250M+ `cargo-fuzz` iterations across all paths, no crashes to date.
//!
//! **[Learn More](https://github.com/hacer-bark/base64-turbo#safety--verification)**: exactly what is proven, and what isn't.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc(issue_tracker_base_url = "https://github.com/hacer-bark/base64-turbo/issues/")]
#![cfg_attr(not(unsafe_simd), forbid(unsafe_code))]
#![forbid(elided_lifetimes_in_paths)]
// This crate casts pointers to wider SIMD vector types (`__m128i`, `__m256i`, `__m512i`)
// purely to call `_mm*_loadu_*`/`_mm*_storeu_*` intrinsics, which are explicitly
// documented to work on any alignment ("u" = unaligned).
#![allow(clippy::cast_ptr_alignment)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

#[cfg(all(doctest, feature = "std"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod alphabet;
mod engine;
mod error;
mod scalar;

/// Runtime CPU capability detection, needed only where a kernel is chosen at
/// run time; the NEON build dispatches at compile time and the scalar build has
/// nothing to choose.
#[cfg(x86_simd)]
mod cpu;

#[cfg(unsafe_simd)]
mod simd;

pub use alphabet::Alphabet;
pub use engine::{
    Engine, STANDARD, STANDARD_NO_PAD, STANDARD_PAD_INDIFFERENT, URL_SAFE, URL_SAFE_NO_PAD,
    URL_SAFE_PAD_INDIFFERENT,
};
pub use error::Error;

pub(crate) use engine::Config;

/// Encodes `input` with the standard RFC 4648 alphabet and padding.
#[cfg(feature = "std")]
#[inline]
pub fn encode<T: AsRef<[u8]>>(input: T) -> String {
    STANDARD.encode(input)
}

/// Decodes `input` with the standard RFC 4648 alphabet and padding.
///
/// # Errors
///
/// Returns [`Error::InvalidLength`] / [`Error::InvalidCharacter`] if `input` is not
/// valid Base64.
#[cfg(feature = "std")]
#[inline]
pub fn decode<T: AsRef<[u8]>>(input: T) -> Result<Vec<u8>, Error> {
    STANDARD.decode(input)
}
