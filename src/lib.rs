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
//! Each x86 SIMD kernel is an independent knob, so a target can compile in only
//! what its CPUs are likely to support. Runtime detection still gates every call,
//! so a kernel the host lacks simply falls back to scalar.
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | **`std`** | **Yes** | `String`/`Vec` support. Disable for `no_std` (the slice APIs need no allocator). |
//! | **`avx2`** | **Yes** | AVX2 kernel + runtime detection on `x86`/`x86_64`. Implies `std`. |
//! | **`avx512-vbmi`** | **Yes** | AVX-512 VBMI fast-path kernel on `x86`/`x86_64`. Implies `std`. |
//! | **`simd`** | **Yes** | Convenience meta-feature — turns on `avx2` + `avx512-vbmi` at once. |
//! | **`neon`** | **Yes** | NEON acceleration on aarch64. No `std` required — compile-time dispatch. |
//! | **`unstable`** | **No** | Exposes the raw internal kernels (`encode_avx2`, `encode_avx512_vbmi`, `encode_neon`, …). The `*_scalar` accessors are safe. |
//!
//! If **no** SIMD kernel is enabled (no `avx2`/`avx512-vbmi` on x86, no
//! `neon` on aarch64), the build is pure scalar Rust and the crate carries
//! `#![forbid(unsafe_code)]` — memory safety then holds by construction, with no
//! `unsafe` anywhere to audit.
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
// Scalar implementation
mod scalar;
// SIMD implementations, compiled when any vectorized kernel is enabled.
#[cfg(unsafe_simd)]
mod simd;

pub use alphabet::Alphabet;

/// Runtime CPU capability detection for the x86 kernels, resolved once and cached.
///
/// `std::is_x86_feature_detected!` already caches its answer internally, but this
/// collapses the whole *tier* decision — which of the compiled kernels to run —
/// into a single load after the first call, instead of re-checking each feature
/// bit on every encode/decode.
#[cfg(x86_simd)]
mod cpu {
    use std::sync::OnceLock;

    // Tier levels, ordered least- to most-capable so callers compare with `>=`.
    // Each level exists only when its kernel was compiled in, which keeps the
    // detection and dispatch arms in lockstep with the feature set.
    #[cfg(feature = "avx2")]
    pub(crate) const AVX2: u8 = 1;
    #[cfg(feature = "avx512-vbmi")]
    pub(crate) const AVX512_VBMI: u8 = 2;

    fn detect() -> u8 {
        #[cfg(feature = "avx512-vbmi")]
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
        {
            return AVX512_VBMI;
        }
        #[cfg(feature = "avx2")]
        if std::is_x86_feature_detected!("avx2") {
            return AVX2;
        }
        0 // scalar
    }

    /// The best compiled-in kernel tier the current CPU supports. Detected on the
    /// first call and cached for the lifetime of the process.
    #[inline]
    pub(crate) fn tier() -> u8 {
        static CACHE: OnceLock<u8> = OnceLock::new();
        *CACHE.get_or_init(detect)
    }

    /// Size in bytes of the largest data cache `CPUID` reports, or `None` when
    /// it reports none — an emulator, or a hypervisor that masks the leaves.
    ///
    /// Intel's leaf 4 and AMD's leaf `0x8000_001D` encode a cache's geometry in
    /// exactly the same EAX/EBX/ECX layout, so one walk serves both vendors and
    /// only the leaf number differs. Sub-leaves are enumerated until one reports
    /// type 0; instruction caches are skipped, since the thing being sized is a
    /// data working set.
    ///
    /// Absent under Miri and Kani. Neither executes `cpuid` — Miri interprets,
    /// Kani is symbolic — and neither needs to: the only caller, the AVX-512
    /// streaming gate, takes its floor when no cache size is available, which
    /// is exactly the bound the stream-peel proofs are stated against.
    ///
    /// `__cpuid_count` became a safe function in 1.94; at this crate's 1.93 MSRV
    /// it is still `unsafe`. The blocks are what 1.93 needs, and the `allow` is
    /// what stops 1.94-and-later failing the crate's `unused` deny over blocks
    /// it considers redundant. Both halves are load-bearing until the MSRV
    /// reaches 1.94.
    #[cfg(not(any(miri, kani)))]
    #[allow(unused_unsafe)]
    fn detect_llc() -> Option<usize> {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::__cpuid_count;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::__cpuid_count;

        // Both leaves have to be checked for existence first: reading an
        // unimplemented leaf returns another leaf's contents, not zeros.
        let leaf = if unsafe { __cpuid_count(0x8000_0000, 0) }.eax >= 0x8000_001D {
            0x8000_001D
        } else if unsafe { __cpuid_count(0, 0) }.eax >= 4 {
            4
        } else {
            return None;
        };

        let mut best: u64 = 0;
        for sub in 0..16 {
            let r = unsafe { __cpuid_count(leaf, sub) };
            match r.eax & 0x1f {
                0 => break,    // no cache at this sub-leaf, and none after it
                2 => continue, // instruction cache
                _ => {}
            }
            let ways = u64::from((r.ebx >> 22) & 0x3ff) + 1;
            let partitions = u64::from((r.ebx >> 12) & 0x3ff) + 1;
            let line = u64::from(r.ebx & 0xfff) + 1;
            let sets = u64::from(r.ecx) + 1;
            best = best.max(ways * partitions * line * sets);
        }
        (best > 0).then(|| usize::try_from(best).unwrap_or(usize::MAX))
    }

    /// Size of the last-level data cache, detected once and cached. `None` if
    /// `CPUID` does not report one; callers pick their own fallback.
    #[cfg(not(any(miri, kani)))]
    #[inline]
    pub(crate) fn llc_bytes() -> Option<usize> {
        static CACHE: OnceLock<Option<usize>> = OnceLock::new();
        *CACHE.get_or_init(detect_llc)
    }
}

// ======================================================================
// Error Definition
// ======================================================================

/// Errors that can occur during Base64 encoding or decoding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The input length is invalid for Base64 decoding.
    ///
    /// Base64 encoded data (with padding) must strictly have a length divisible by 4.
    /// If the input string is truncated or has incorrect padding length, this error is returned.
    InvalidLength,

    /// An invalid character was encountered during decoding.
    ///
    /// This occurs if the input contains bytes that do not belong to the
    /// selected Base64 alphabet — symbols outside the chosen character set, or
    /// a `=` in a position the config does not allow.
    ///
    /// # Which variant a misplaced `=` produces
    ///
    /// Malformed padding is rejected by every configuration, but *which* of
    /// [`Error::InvalidLength`] and [`Error::InvalidCharacter`] comes back is
    /// deliberately unspecified and may change between releases. It depends on
    /// which kernel met the character: the vector tiers map `=` to the same
    /// invalid-symbol sentinel as any other foreign byte, while the scalar tail
    /// that owns the padding rules reads a stray `=` as a length error. Match on
    /// `is_err()`, not on the variant, when validating untrusted input.
    InvalidCharacter,

    /// The provided output buffer is too small to hold the result.
    ///
    /// This error is returned by the slice APIs (e.g., `encode_slice`, `decode_slice`)
    /// when the destination slice passed by the user does not have enough capacity
    /// to store the encoded or decoded data.
    BufferTooSmall,
}

// Standard Display implementation for better error messages
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLength => {
                write!(f, "Invalid Base64 input length (must be divisible by 4)")
            }
            Self::InvalidCharacter => write!(f, "Invalid character found in Base64 input"),
            Self::BufferTooSmall => write!(f, "Destination buffer is too small"),
        }
    }
}

// Enable std::error::Error trait when the 'std' feature is active
#[cfg(feature = "std")]
impl std::error::Error for Error {}

// ======================================================================
// Configuration & Types
// ======================================================================

/// Internal configuration for the Base64 engine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Config {
    /// The alphabet and every lookup table the kernels drive from it.
    pub alphabet: &'static Alphabet,
    /// If true, writes `=` padding characters to the output.
    pub padding: bool,
}

/// A high-performance, stateless Base64 encoder/decoder.
///
/// This struct holds the configuration for encoding/decoding (alphabet choice and padding).
/// It is designed to be immutable and thread-safe.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "std")]
/// # {
/// use base64_turbo::STANDARD;
///
/// let data = b"Hello world";
///
/// // Encode to String
/// let encoded = STANDARD.encode(data);
/// assert_eq!(encoded, "SGVsbG8gd29ybGQ=");
///
/// // Decode to Result<Vec<u8>, Error>
/// let decoded = STANDARD.decode(&encoded).unwrap();
/// assert_eq!(decoded, data);
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Engine {
    pub(crate) config: Config,
    decode_padding_indifferent: bool,
}

// ======================================================================
// Pre-defined Engines
// ======================================================================

/// Standard Base64 (RFC 4648) with padding (`=`).
///
/// Uses the `+` and `/` characters. This is the most common configuration.
pub const STANDARD: Engine = Engine {
    config: Config {
        alphabet: &alphabet::STANDARD_TABLE,
        padding: true,
    },
    decode_padding_indifferent: false,
};

/// Standard Base64 (RFC 4648) **without** padding.
///
/// Uses the `+` and `/` characters, but omits trailing `=` signs.
/// Useful for raw data streams or specific protocol requirements.
pub const STANDARD_NO_PAD: Engine = Engine {
    config: Config {
        alphabet: &alphabet::STANDARD_TABLE,
        padding: false,
    },
    decode_padding_indifferent: false,
};

/// URL-Safe Base64 with padding.
///
/// Uses `-` and `_` instead of `+` and `/`. Safe for use in filenames and URLs.
pub const URL_SAFE: Engine = Engine {
    config: Config {
        alphabet: &alphabet::URL_SAFE_TABLE,
        padding: true,
    },
    decode_padding_indifferent: false,
};

/// URL-Safe Base64 **without** padding.
///
/// Uses `-` and `_`. Commonly used in JWTs (JSON Web Tokens) and other web standards.
pub const URL_SAFE_NO_PAD: Engine = Engine {
    config: Config {
        alphabet: &alphabet::URL_SAFE_TABLE,
        padding: false,
    },
    decode_padding_indifferent: false,
};

/// Standard Base64 with padding when encoding, accepting padded or unpadded input when decoding.
///
/// # Performance
///
/// Encoding runs on every kernel, exactly as [`STANDARD`] does. **Decoding is
/// scalar only**: the vector kernels hand their final group to a tail that owns
/// the padding and length rules, and that tail assumes one fixed rule, so a
/// decoder that accepts either shape cannot use them. Reach for this when the
/// input's padding is genuinely out of your control; prefer [`STANDARD`] or
/// [`STANDARD_NO_PAD`] on a hot decode path where it is not.
pub const STANDARD_PAD_INDIFFERENT: Engine = Engine {
    config: Config {
        alphabet: &alphabet::STANDARD_TABLE,
        padding: true,
    },
    decode_padding_indifferent: true,
};

/// URL-safe Base64 with padding when encoding, accepting padded or unpadded input when decoding.
///
/// # Performance
///
/// Decoding is scalar only, for the reason given on
/// [`STANDARD_PAD_INDIFFERENT`]. Encoding is unaffected.
pub const URL_SAFE_PAD_INDIFFERENT: Engine = Engine {
    config: Config {
        alphabet: &alphabet::URL_SAFE_TABLE,
        padding: true,
    },
    decode_padding_indifferent: true,
};

// ======================================================================
// Allocating-API helpers (std only)
//
// These isolate the one place the SIMD and scalar-only builds genuinely differ:
// the SIMD build already contains `unsafe`, so it skips zeroing and validation;
// the scalar-only build forbids `unsafe`, so it pays a linear pass for the same
// result. `encode`/`decode` themselves stay identical across both.
// ======================================================================

/// A `len`-byte buffer for a dispatcher to fill: uninitialized on SIMD builds.
#[cfg(all(feature = "std", unsafe_simd))]
#[inline]
fn spare(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    // SAFETY: the caller passes `out` straight to a dispatcher, which writes the
    // whole `len`-byte encode output / the decoded prefix; `encode` reads all of
    // it and `decode` truncates to the written prefix, so no uninitialized byte
    // is ever observed.
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(len);
    }
    out
}

/// A `len`-byte buffer for a dispatcher to fill: zeroed on the safe scalar build.
#[cfg(all(feature = "std", not(unsafe_simd)))]
#[inline]
fn spare(len: usize) -> Vec<u8> {
    vec![0u8; len]
}

/// Wraps encoder output (guaranteed ASCII) as a `String` without re-validating.
#[cfg(all(feature = "std", unsafe_simd))]
#[inline]
fn into_ascii_string(bytes: Vec<u8>) -> String {
    // SAFETY: the Base64 alphabet is strictly ASCII, hence valid UTF-8.
    unsafe { String::from_utf8_unchecked(bytes) }
}

/// Safe-build counterpart: validate on the way out. The bytes are always ASCII,
/// so the happy path reuses the buffer's allocation and the `Err` arm is dead.
#[cfg(all(feature = "std", not(unsafe_simd)))]
#[inline]
fn into_ascii_string(bytes: Vec<u8>) -> String {
    match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
    }
}

impl Engine {
    // ======================================================================
    // Construction
    // ======================================================================

    /// Builds an engine over a custom [`Alphabet`].
    ///
    /// `padding` selects whether encoding appends `=` and whether decoding
    /// requires it, exactly as it does for the built-in engines; the padding
    /// character is `=` for every alphabet.
    ///
    /// The alphabet must outlive the program because the engines are `Copy` and
    /// carry no lifetime. [`Alphabet::new`] is a `const fn`, so the usual way
    /// there is a `static`; an alphabet only known at run time can be given the
    /// same lifetime with `Box::leak`.
    ///
    /// # Performance
    ///
    /// A custom alphabet runs on the scalar and AVX-512-VBMI kernels at their
    /// full speed — both are table lookups, and the tables come from the
    /// alphabet. The AVX2 and NEON kernels compute characters arithmetically
    /// from the RFC 4648 layout, so they cannot serve one; those targets fall
    /// back to the scalar kernel. An alphabet built from the standard or
    /// URL-safe characters is recognized as such and keeps every kernel.
    ///
    /// # Examples
    ///
    /// ```
    /// use base64_turbo::{Alphabet, Engine};
    ///
    /// static ORDERED: Alphabet = match Alphabet::new(
    ///     b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz",
    /// ) {
    ///     Some(a) => a,
    ///     None => unreachable!(),
    /// };
    /// static ENGINE: Engine = Engine::custom(&ORDERED, true);
    ///
    /// # #[cfg(feature = "std")] {
    /// let encoded = ENGINE.encode(b"Hello world");
    /// assert_eq!(ENGINE.decode(&encoded).unwrap(), b"Hello world");
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn custom(alphabet: &'static Alphabet, padding: bool) -> Self {
        Self {
            config: Config { alphabet, padding },
            decode_padding_indifferent: false,
        }
    }

    /// The alphabet this engine encodes and decodes with.
    #[inline]
    #[must_use]
    pub const fn alphabet(&self) -> &'static Alphabet {
        self.config.alphabet
    }

    // ======================================================================
    // Length calculations
    // ======================================================================

    /// Calculates the buffer size required to encode `input_len` bytes with this engine.
    ///
    /// Saturates at `usize::MAX` rather than wrapping, so the result is never a
    /// too-small buffer size. That bound is unreachable for any input you
    /// actually hold: a slice is at most `isize::MAX` bytes, and 4/3 of that
    /// still fits in a `usize`. Only a fabricated `input_len` can saturate, and
    /// a buffer that size cannot be allocated anyway — the slice APIs turn it
    /// into [`Error::BufferTooSmall`].
    #[inline]
    #[must_use]
    pub const fn encoded_len(&self, input_len: usize) -> usize {
        let complete_len = (input_len / 3).saturating_mul(4);

        match (input_len % 3, self.config.padding) {
            (0, _) => complete_len,
            (_, true) => complete_len.saturating_add(4),
            (1, false) => complete_len.saturating_add(2),
            (_, false) => complete_len.saturating_add(3),
        }
    }

    /// Returns a conservative decoded-length estimate for `encoded_len` Base64 symbols.
    ///
    /// The returned size is safe for a decode buffer and can exceed the actual decoded
    /// length by up to two bytes. This estimate does not depend on the engine's
    /// configuration, but is a method for API symmetry with [`Engine::encoded_len`].
    #[inline]
    #[must_use]
    pub const fn decoded_len_estimate(&self, encoded_len: usize) -> usize {
        encoded_len.saturating_add(3) / 4 * 3
    }

    // ======================================================================
    // Slice APIs
    // ======================================================================

    /// Encodes `input` into the provided `output` buffer.
    ///
    /// This is a "Zero-Allocation" API designed for hot paths. It writes directly
    /// into the destination slice without creating intermediate `Vec`.
    ///
    /// # Arguments
    ///
    /// * `input`: The binary data to encode.
    /// * `output`: A mutable slice to write the Base64 string into.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)`: The actual number of bytes written to `output`.
    /// * `Err(Error::BufferTooSmall)`: If `output.len()` is less than [`Engine::encoded_len`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] if `output` is not large enough to hold the
    /// encoded data (see [`Engine::encoded_len`]).
    #[inline]
    pub fn encode_slice<T: AsRef<[u8]>>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }

        let req_len = self.encoded_len(len);
        if output.len() < req_len {
            return Err(Error::BufferTooSmall);
        }

        // --- Normal Path ---
        // We checked output.len() >= req_len above.
        Self::encode_dispatch(self, input, &mut output[..req_len]);

        Ok(req_len)
    }

    /// Decodes `input` into the provided `output` buffer.
    ///
    /// `output` must be at least [`Engine::decoded_len_estimate`] bytes — the
    /// conservative upper bound, not the exact decoded length. A padded input
    /// decodes to up to two bytes fewer than the estimate, and a buffer sized to
    /// that exact result is rejected with [`Error::BufferTooSmall`]; the return
    /// value reports how much of `output` was actually written.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)`: The actual number of bytes written to `output`.
    /// * `Err(Error)`: If the input is invalid or the buffer is too small.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] if `output` is smaller than
    /// [`Engine::decoded_len_estimate`], or
    /// [`Error::InvalidLength`] / [`Error::InvalidCharacter`] if `input` is not
    /// valid Base64.
    #[inline]
    pub fn decode_slice<T: AsRef<[u8]>>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }

        let req_len = self.decoded_len_estimate(len);
        if output.len() < req_len {
            return Err(Error::BufferTooSmall);
        }

        // --- Normal Path ---
        let real_len = Self::decode_dispatch(self, input, &mut output[..req_len])?;

        Ok(real_len)
    }

    /// Returns whether this engine emits padding when encoding.
    #[inline]
    #[must_use]
    pub const fn encode_padding(&self) -> bool {
        self.config.padding
    }

    // ========================================================================
    // Allocating APIs (std)
    // ========================================================================

    /// Allocates a new `String` and encodes the input data into it.
    ///
    /// This is the most convenient method for general usage.
    ///
    /// # Examples
    ///
    /// ```
    /// use base64_turbo::STANDARD;
    /// let b64 = STANDARD.encode(b"hello");
    /// assert_eq!(b64, "aGVsbG8=");
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn encode<T: AsRef<[u8]>>(&self, input: T) -> String {
        let input = input.as_ref();
        let output_len = self.encoded_len(input.len());
        let mut out = spare(output_len);
        Self::encode_dispatch(self, input, &mut out);
        into_ascii_string(out)
    }

    /// Allocates a new `Vec<u8>` and decodes the input data into it.
    ///
    /// # Errors
    /// Returns `Error` if the input contains invalid characters or has an invalid length.
    ///
    /// # Examples
    ///
    /// ```
    /// use base64_turbo::STANDARD;
    /// let bytes = STANDARD.decode("aGVsbG8=").unwrap();
    /// assert_eq!(bytes, b"hello");
    /// ```
    #[inline]
    #[cfg(feature = "std")]
    pub fn decode<T: AsRef<[u8]>>(&self, input: T) -> Result<Vec<u8>, Error> {
        let input = input.as_ref();

        // `spare` gives us the upper-bound-sized buffer; `decode_slice` writes the
        // decoded prefix and reports its exact length, then `truncate` drops the
        // unwritten tail. `truncate` is safe on both buffer flavors — for `u8`
        // there is nothing to run, it just shortens the live length — and on error
        // the whole buffer is dropped without exposing an unwritten byte.
        let mut out = spare(self.decoded_len_estimate(input.len()));
        let written = Self::decode_slice(self, input, &mut out)?;
        out.truncate(written);
        Ok(out)
    }

    /// Encodes `input` and appends it to `output`.
    #[inline]
    #[cfg(feature = "std")]
    pub fn encode_string<T: AsRef<[u8]>>(&self, input: T, output: &mut String) {
        output.push_str(&Self::encode(self, input));
    }

    /// Decodes `input` and appends the result to `output`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] / [`Error::InvalidCharacter`] if `input` is not
    /// valid Base64.
    #[inline]
    #[cfg(feature = "std")]
    pub fn decode_vec<T: AsRef<[u8]>>(&self, input: T, output: &mut Vec<u8>) -> Result<(), Error> {
        let input = input.as_ref();
        let start = output.len();
        let estimate = self.decoded_len_estimate(input.len());
        output.resize(start.saturating_add(estimate), 0);
        let written = Self::decode_slice(self, input, &mut output[start..])?;
        output.truncate(start + written);
        Ok(())
    }

    // ========================================================================
    // Internal Dispatchers
    // ========================================================================

    // `&self` (a small Copy `Engine`) is kept by-ref for consistency with the
    // rest of the `Engine` methods, not because the reference is required.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    #[inline]
    fn encode_dispatch(&self, input: &[u8], dst: &mut [u8]) {
        #[cfg(x86_simd)]
        {
            let len = input.len();
            let tier = cpu::tier();

            // Smart degrade by length: a kernel is only worth entering once it
            // beats the scalar kernel, which is not the same thing as its vector
            // width. AVX2 encode covers one 24-byte round and hands the rest to
            // the scalar tail, so below ~44 bytes it is doing a vector round
            // *and* most of the scalar work; racing the two kernels in one
            // binary on Coffee Lake, scalar wins by 16% at 32 bytes and 17% at
            // 40, and AVX2 wins by 15% at 48. Decode has no such gap -- its
            // vector tier already covers 32 of the 44 characters a 32-byte
            // input encodes to -- and keeps its own threshold.
            //
            // VBMI starts far earlier than that. Its masked tiers and inline
            // final group mean a short input costs little more than the table
            // setup, and racing the two kernels directly on Zen 5 puts the
            // encode crossover between 8 and 12 bytes: at 12 bytes VBMI is
            // already 20% ahead, at 16 bytes 28%, at 24 bytes 45%. 16 keeps
            // margin over the measured crossover, since this constant is shared
            // with Intel parts that were not measured.
            #[cfg(feature = "avx512-vbmi")]
            if len >= 16 && tier == cpu::AVX512_VBMI {
                // VBMI fast-path: vpermb replaces the 8-instruction char mapping.
                // SAFETY: tier() confirmed AVX-512F/BW/VBMI on this CPU.
                unsafe { simd::encode_slice_avx512_vbmi(&self.config, input, dst) };
                return;
            }
            #[cfg(feature = "avx2")]
            if len >= 48 && tier >= cpu::AVX2 && self.config.alphabet.has_arithmetic_kernels() {
                // SAFETY: tier() confirmed AVX2 on this CPU.
                unsafe { simd::encode_slice_avx2(&self.config, input, dst) };
                return;
            }
        }

        // NEON path (aarch64): compile-time dispatch, no runtime detection.
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if input.len() >= 16 && self.config.alphabet.has_arithmetic_kernels() {
            // SAFETY: NEON is baseline on aarch64.
            unsafe { simd::encode_slice_neon(&self.config, input, dst) };
            return;
        }

        // Fallback: Scalar / non-SIMD target / short inputs.
        scalar::encode_slice(&self.config, input, dst);
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    #[inline]
    fn decode_dispatch(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        if self.decode_padding_indifferent {
            return scalar::decode_slice_forgiving(&self.config, input, dst);
        }

        #[cfg(x86_simd)]
        {
            let len = input.len();
            let tier = cpu::tier();

            // As in `encode_dispatch`, the masked tails let VBMI start earlier
            // than AVX2. `len` is characters here.
            //
            // This one has to be measured through the allocating API, not by
            // racing the two kernels in a loop: back to back, VBMI already wins
            // at 24 characters, but interleaved with an allocation the way a
            // real caller runs it, 24 characters is 23% *slower* and only 28 and
            // up come out ahead (+7%). A tight loop keeps the lookup vectors and
            // the branch history hot in a way that one short call does not.
            #[cfg(feature = "avx512-vbmi")]
            if len >= 28 && tier == cpu::AVX512_VBMI {
                // VBMI fast-path: vpermi2b collapses decode+validate to ~4 instructions.
                // SAFETY: tier() confirmed AVX-512F/BW/VBMI on this CPU.
                return unsafe { simd::decode_slice_avx512_vbmi(&self.config, input, dst) };
            }
            #[cfg(feature = "avx2")]
            if len >= 36 && tier >= cpu::AVX2 && self.config.alphabet.has_arithmetic_kernels() {
                // SAFETY: tier() confirmed AVX2 on this CPU.
                return unsafe { simd::decode_slice_avx2(&self.config, input, dst) };
            }
        }

        // NEON path (aarch64): compile-time dispatch, no runtime detection.
        // Its single tier is a 16-in/12-out block plus a 4-byte read-ahead
        // margin (see `neon::decode_slice_neon`), so it needs 20 bytes to run.
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if input.len() >= 20 && self.config.alphabet.has_arithmetic_kernels() {
            // SAFETY: NEON is baseline on aarch64.
            return unsafe { simd::decode_slice_neon(&self.config, input, dst) };
        }

        // Fallback: Scalar / non-SIMD target / short inputs.
        scalar::decode_slice(&self.config, input, dst)
    }

    // ========================================================================
    // Raw unsafe access (unstable feature)
    // ========================================================================

    /// Raw access to the direct AVX2 encoding logic.
    ///
    /// # Safety
    ///
    /// - `dst` must point to a mutable region with sufficient capacity. The required size
    ///   depends on `config.padding`:
    ///   - With padding: `input.len().div_ceil(3) * 4`
    ///   - Without padding: `(input.len() * 4).div_ceil(3)`
    ///   - Prefer [`Engine::encoded_len`] to compute it.
    /// - The caller must ensure the target CPU supports AVX2 at runtime. Running this on a CPU
    ///   without AVX2 causes an illegal instruction crash.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::encode`]) unless you need this bypass.
    #[cfg(all(x86_simd, feature = "avx2", feature = "unstable"))]
    pub unsafe fn encode_avx2(&self, input: &[u8], dst: &mut [u8]) {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::encode_slice_avx2(&self.config, input, dst) }
    }

    /// Raw access to the direct AVX2 decoding logic.
    ///
    /// # Safety
    ///
    /// - `dst` must point to a mutable region with sufficient capacity. Prefer
    ///   [`Engine::decoded_len_estimate`] to compute it.
    /// - The caller must ensure the target CPU supports AVX2 at runtime. Running this on a CPU
    ///   without AVX2 causes an illegal instruction crash.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::decode`]) unless you need this bypass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] or [`Error::InvalidCharacter`] if `input` is not
    /// valid Base64.
    #[cfg(all(x86_simd, feature = "avx2", feature = "unstable"))]
    pub unsafe fn decode_avx2(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::decode_slice_avx2(&self.config, input, dst) }
    }

    /// Raw access to the direct AVX-512-VBMI encoding logic, the fastest kernel in the crate.
    ///
    /// # Safety
    ///
    /// - `dst` must point to a mutable region with sufficient capacity. The required size
    ///   depends on `config.padding`:
    ///   - With padding: `input.len().div_ceil(3) * 4`
    ///   - Without padding: `(input.len() * 4).div_ceil(3)`
    ///   - Prefer [`Engine::encoded_len`] to compute it.
    /// - The caller must ensure the target CPU supports the `avx512f`, `avx512bw` and
    ///   `avx512vbmi` subsets at runtime. Running this without all three causes an illegal
    ///   instruction crash.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::encode`]) unless you need this bypass.
    #[cfg(all(x86_simd, feature = "avx512-vbmi", feature = "unstable"))]
    pub unsafe fn encode_avx512_vbmi(&self, input: &[u8], dst: &mut [u8]) {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::encode_slice_avx512_vbmi(&self.config, input, dst) }
    }

    /// Raw access to the direct AVX-512-VBMI decoding logic.
    ///
    /// # Safety
    ///
    /// - `dst` must point to a mutable region of at least [`Engine::decoded_len_estimate`]
    ///   bytes. The quad tier's first three stores are unmasked, each overhanging the 48
    ///   bytes it produces by 16, but the next store in that same iteration rewrites the
    ///   overhang and the last one is masked, so a step never writes past the 192 bytes it
    ///   produces. No capacity beyond the estimate is needed.
    /// - The caller must ensure the target CPU supports the `avx512f`, `avx512bw` and
    ///   `avx512vbmi` subsets at runtime. Running this without all three causes an illegal
    ///   instruction crash.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::decode`]) unless you need this bypass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] or [`Error::InvalidCharacter`] if `input` is not
    /// valid Base64.
    #[cfg(all(x86_simd, feature = "avx512-vbmi", feature = "unstable"))]
    pub unsafe fn decode_avx512_vbmi(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::decode_slice_avx512_vbmi(&self.config, input, dst) }
    }

    /// Raw access to the direct scalar encoding logic.
    ///
    /// Unlike the SIMD accessors, this is a **safe** function: the scalar kernel uses no
    /// `unsafe`, so every write is bounds-checked.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is smaller than the encoded length (a bounds check, not memory
    /// corruption). Size it with [`Engine::encoded_len`]:
    /// - With padding: `input.len().div_ceil(3) * 4`
    /// - Without padding: `(input.len() * 4).div_ceil(3)`
    #[cfg(feature = "unstable")]
    pub fn encode_scalar(&self, input: &[u8], dst: &mut [u8]) {
        scalar::encode_slice(&self.config, input, dst);
    }

    /// Raw access to the direct scalar decoding logic.
    ///
    /// Like [`Engine::encode_scalar`], this is a **safe** function — the scalar kernel
    /// contains no `unsafe`, so a too-small `dst` panics on a bounds check rather than
    /// corrupting memory. Size `dst` with [`Engine::decoded_len_estimate`].
    ///
    /// # Panics
    ///
    /// Panics if `dst` is too small to hold the decoded output.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] or [`Error::InvalidCharacter`] if `input` is not
    /// valid Base64.
    #[cfg(feature = "unstable")]
    pub fn decode_scalar(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        scalar::decode_slice(&self.config, input, dst)
    }

    /// Raw access to the direct NEON encoding logic.
    ///
    /// # Safety
    ///
    /// - `dst` must point to a mutable region with sufficient capacity. The required size
    ///   depends on `config.padding`:
    ///   - With padding: `input.len().div_ceil(3) * 4`
    ///   - Without padding: `(input.len() * 4).div_ceil(3)`
    ///   - Prefer [`Engine::encoded_len`] to compute it.
    /// - NEON is baseline on `aarch64`, so there is no runtime feature to check.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::encode`]) unless you need this bypass.
    #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "unstable"))]
    pub unsafe fn encode_neon(&self, input: &[u8], dst: &mut [u8]) {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::encode_slice_neon(&self.config, input, dst) }
    }

    /// Raw access to the direct NEON decoding logic.
    ///
    /// # Safety
    ///
    /// - `dst` must point to a mutable region of at least [`Engine::decoded_len_estimate`]
    ///   bytes. Each pack stores a full 16-byte vector for 12 bytes of output, but no
    ///   vector pass starts unless enough characters remain for the tail to absorb that
    ///   4-byte overhang, so the estimate is sufficient.
    /// - NEON is baseline on `aarch64`, so there is no runtime feature to check.
    ///
    /// Prefer the safe higher-level APIs (e.g. [`Engine::decode`]) unless you need this bypass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLength`] or [`Error::InvalidCharacter`] if `input` is not
    /// valid Base64.
    #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "unstable"))]
    pub unsafe fn decode_neon(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::decode_slice_neon(&self.config, input, dst) }
    }
}

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
