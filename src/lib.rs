//! # Base64 Turbo
//!
//! [![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
//! [![License](https://img.shields.io/crates/l/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
//! [![Kani Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/verification.yml?label=Kani%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/verification.yml)
//! [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)
//!
//! **A SIMD Base64 implementation whose `unsafe` paths are checked by a model checker, not just by review.**
//!
//! `base64-turbo` is a production-grade library engineered for high-throughput systems where CPU cycles are scarce and Undefined Behavior (UB) is unacceptable.
//!
//! "Memory-safe" here is a specific, bounded claim: the `unsafe` SIMD paths are checked by the
//! [Kani](https://github.com/model-checking/kani) model checker and [MIRI](https://github.com/rust-lang/miri)
//! (a strict UB interpreter), on top of `MemorySanitizer` audits and continuous fuzzing — see the
//! "Safety & Verification" section below for what each layer does and does not cover per
//! architecture. This crate is **not** faster than unchecked C/assembly implementations and does
//! not claim to be; within the narrower set of crates combining SIMD-accelerated Base64 with
//! Kani + MIRI verification, we are not aware of another one that reaches AVX512 speeds.
//!
//! This crate provides runtime CPU detection to utilize **AVX512** or **AVX2** intrinsics on `x86_64`,
//! and compile-time **NEON** acceleration on `aarch64`.
//! It includes a highly optimized scalar fallback for non-SIMD targets and supports `no_std` environments.
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
//! let len = STANDARD.encode_into(input, &mut output).unwrap();
//!
//! assert_eq!(&output[..len], b"UmF3IGJ5dGVz");
//! ```
//!
//! ## Feature Flags
//!
//! Each x86 SIMD kernel is an independent knob, so a target can compile in only
//! what its CPUs are likely to support. Runtime detection still gates every call,
//! so a kernel the host lacks simply falls back to scalar.
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | **`std`** | **Yes** | Enables `String` and `Vec` support. Disable this for `no_std` environments. |
//! | **`avx2`** | **Yes** | AVX2 kernel + runtime detection on `x86`/`x86_64`. Implies `std`. |
//! | **`avx512`** | **Yes** | AVX-512F/BW kernel + runtime detection on `x86`/`x86_64`. Implies `std`. |
//! | **`avx512-vbmi`** | **Yes** | AVX-512 VBMI fast-path kernel on `x86`/`x86_64`. Implies `std`. |
//! | **`simd`** | **Yes** | Convenience meta-feature: enables `avx2` + `avx512` + `avx512-vbmi` at once. |
//! | **`neon`** | **Yes** | **NEON** acceleration on aarch64 (ARM64). No `std` required — compile-time dispatch. |
//! | **`unstable`** | **No** | Exposes the raw internal kernels (e.g. `encode_avx2`; the `*_scalar` accessors are safe). |
//!
//! If **no** SIMD kernel is enabled (no `avx2`/`avx512`/`avx512-vbmi` on x86, no
//! `neon` on aarch64), the build is pure scalar Rust and the crate carries
//! `#![forbid(unsafe_code)]` — memory safety then holds by construction, with no
//! `unsafe` anywhere to audit.
//!
//! ## Safety & Verification
//!
//! This crate utilizes `unsafe` code for SIMD intrinsics and pointer arithmetic to achieve maximum performance.
//! To ensure safety, we employ a "Swiss Cheese" model of verification layers:
//!
//! *   **Model checking (Kani):** For the Scalar, AVX2 and plain AVX512 kernels, Kani explores
//!     *every possible input byte value* at lengths chosen to exercise each loop tier and the
//!     scalar-tail handoff, proving the kernel does not panic, does not read or write out of
//!     bounds, and round-trips exactly. On AVX2 a second layer of proofs takes the loop
//!     arithmetic on its own, over an unbounded symbolic length and an arbitrary iteration, so
//!     the in-bounds result there is a machine-checked induction covering every length rather
//!     than the ones a harness happens to pin. The README spells out what that does and does not
//!     buy you, along with the AVX512-VBMI and NEON gaps.
//! *   **MIRI Audited:** All SIMD paths (AVX512, AVX2, NEON) and Scalar fallbacks are run under
//!     **MIRI** (Undefined Behavior checker) in CI, covering every distinct code path at least once.
//! *   **`MemorySanitizer`:** The codebase is audited with `MSan` to prevent logic errors derived from reading uninitialized memory.
//! *   **Fuzzing:** The codebase is fuzz-tested via `cargo-fuzz` (2.5B+ iterations).
//!
//! **[Learn More](https://github.com/hacer-bark/base64-turbo#safety--verification)**: exactly what is proven, and what isn't.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc(issue_tracker_base_url = "https://github.com/hacer-bark/base64-turbo/issues/")]
// When no SIMD kernel is compiled in (`unsafe_simd` off, set by build.rs), the
// crate is pure scalar Rust with no `unsafe` anywhere — so we forbid it
// crate-wide and memory safety stops resting on review at all.
#![cfg_attr(not(unsafe_simd), forbid(unsafe_code))]
#![forbid(elided_lifetimes_in_paths)]
// This crate casts pointers to wider SIMD vector types (`__m128i`, `__m256i`, `__m512i`)
// purely to call `_mm*_loadu_*`/`_mm*_storeu_*` intrinsics, which are explicitly
// documented to work on any alignment ("u" = unaligned).
#![allow(clippy::cast_ptr_alignment)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(doctest, feature = "std"))]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

// Scalar implementation
mod scalar;
// SIMD implementations, compiled when any vectorized kernel is enabled.
#[cfg(unsafe_simd)]
mod simd;

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
    #[cfg(feature = "avx512")]
    pub(crate) const AVX512: u8 = 2;
    #[cfg(feature = "avx512-vbmi")]
    pub(crate) const AVX512_VBMI: u8 = 3;

    fn detect() -> u8 {
        #[cfg(feature = "avx512-vbmi")]
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi")
        {
            return AVX512_VBMI;
        }
        #[cfg(feature = "avx512")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return AVX512;
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
}

// ======================================================================
// ERROR DEFINITION
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
    /// selected Base64 alphabet (e.g., symbols not in the standard set) or
    /// if padding characters (`=`) appear in invalid positions.
    InvalidCharacter,

    /// The provided output buffer is too small to hold the result.
    ///
    /// This error is returned by the zero-allocation APIs (e.g., `encode_into`, `decode_into`)
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
// Internal Lookup Tables
// ======================================================================

/// The Standard RFC 4648 Base64 Alphabet.
/// Used for `STANDARD` and `STANDARD_NO_PAD`.
const STANDARD_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Computed compile-time reverse lookup table for the Standard alphabet.
/// Maps ASCII bytes back to 6-bit indices. 0xFF indicates an invalid character.
#[allow(clippy::cast_possible_truncation)] // `i` is always < 64, fits in u8
const STANDARD_DECODE_TABLE: [u8; 256] = {
    let mut table = [0xFF; 256];
    let mut i = 0;
    while i < 64 {
        table[STANDARD_ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

/// The URL-Safe Base64 Alphabet.
/// Replaces `+` with `-` and `/` with `_`. Used for `URL_SAFE` and `URL_SAFE_NO_PAD`.
const URL_SAFE_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Computed compile-time reverse lookup table for the URL-Safe alphabet.
/// Maps ASCII bytes back to 6-bit indices. 0xFF indicates an invalid character.
#[allow(clippy::cast_possible_truncation)] // `i` is always < 64, fits in u8
const URL_SAFE_DECODE_TABLE: [u8; 256] = {
    let mut table = [0xFF; 256];
    let mut i = 0;
    while i < 64 {
        table[URL_SAFE_ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table
};

// ======================================================================
// Configuration & Types
// ======================================================================

/// Internal configuration for the Base64 engine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Config {
    /// If true, uses `-` and `_` instead of `+` and `/`.
    pub url_safe: bool,
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
}

// ======================================================================
// Pre-defined Engines
// ======================================================================

/// Standard Base64 (RFC 4648) with padding (`=`).
///
/// Uses the `+` and `/` characters. This is the most common configuration.
pub const STANDARD: Engine = Engine {
    config: Config {
        url_safe: false,
        padding: true,
    },
};

/// Standard Base64 (RFC 4648) **without** padding.
///
/// Uses the `+` and `/` characters, but omits trailing `=` signs.
/// Useful for raw data streams or specific protocol requirements.
pub const STANDARD_NO_PAD: Engine = Engine {
    config: Config {
        url_safe: false,
        padding: false,
    },
};

/// URL-Safe Base64 with padding.
///
/// Uses `-` and `_` instead of `+` and `/`. Safe for use in filenames and URLs.
pub const URL_SAFE: Engine = Engine {
    config: Config {
        url_safe: true,
        padding: true,
    },
};

/// URL-Safe Base64 **without** padding.
///
/// Uses `-` and `_`. Commonly used in JWTs (JSON Web Tokens) and other web standards.
pub const URL_SAFE_NO_PAD: Engine = Engine {
    config: Config {
        url_safe: true,
        padding: false,
    },
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
    // Length Calculators
    // ======================================================================

    /// Calculates the exact buffer size required to encode `input_len` bytes.
    ///
    /// This method computes the size based on the current configuration (padding vs. no padding).
    ///
    /// # Examples
    ///
    /// ```
    /// use base64_turbo::STANDARD;
    ///
    /// assert_eq!(STANDARD.encoded_len(3), 4);
    /// assert_eq!(STANDARD.encoded_len(1), 4); // With padding
    /// ```
    #[inline]
    #[must_use]
    pub const fn encoded_len(&self, input_len: usize) -> usize {
        if self.config.padding {
            // (n + 2) / 3 * 4
            input_len.div_ceil(3) * 4
        } else {
            // (n * 4 + 2) / 3
            (input_len * 4).div_ceil(3)
        }
    }

    /// Calculates the **maximum** buffer size required to decode `input_len` bytes.
    ///
    /// # Note
    /// This is an upper-bound estimate. The actual number of bytes written during
    /// decoding will likely be smaller.
    ///
    /// You should rely on the `usize` returned by [`decode_into`](Self::decode_into)
    /// to determine the actual valid slice of the output buffer.
    #[inline]
    #[must_use]
    pub const fn estimate_decoded_len(&self, input_len: usize) -> usize {
        // Conservative estimate: 3 bytes for every 4 chars, plus a safety margin
        // for unpadded/chunked logic.
        (input_len / 4 + 1) * 3
    }

    // ======================================================================
    // Zero-Allocation APIs
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
    /// * `Err(Error::BufferTooSmall)`: If `output.len()` is less than [`encoded_len`](Self::encoded_len).
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] if `output` is not large enough to hold the
    /// encoded data (see [`encoded_len`](Self::encoded_len)).
    #[inline]
    pub fn encode_into<T: AsRef<[u8]>>(&self, input: T, output: &mut [u8]) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }

        let req_len = Self::encoded_len(self, len);
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
    /// # Returns
    ///
    /// * `Ok(usize)`: The actual number of bytes written to `output`.
    /// * `Err(Error)`: If the input is invalid or the buffer is too small.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BufferTooSmall`] if `output` is not large enough, or
    /// [`Error::InvalidLength`] / [`Error::InvalidCharacter`] if `input` is not
    /// valid Base64.
    #[inline]
    pub fn decode_into<T: AsRef<[u8]>>(&self, input: T, output: &mut [u8]) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }

        let req_len = Self::estimate_decoded_len(self, len);
        if output.len() < req_len {
            return Err(Error::BufferTooSmall);
        }

        // --- Normal Path ---
        let real_len = Self::decode_dispatch(self, input, &mut output[..req_len])?;

        Ok(real_len)
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

        // Base64 encoding is deterministic, so this is the EXACT output size.
        // `spare` hands the dispatcher a full-length buffer (uninitialized on
        // SIMD builds, zeroed on the scalar-only safe build); the dispatcher then
        // overwrites every byte, and the output is pure ASCII.
        let mut out = spare(Self::encoded_len(self, input.len()));
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

        // `spare` gives us the upper-bound-sized buffer; `decode_into` writes the
        // decoded prefix and reports its exact length, then `truncate` drops the
        // unwritten tail. `truncate` is safe on both buffer flavors — for `u8`
        // there is nothing to run, it just shortens the live length — and on error
        // the whole buffer is dropped without exposing an unwritten byte.
        let mut out = spare(Self::estimate_decoded_len(self, input.len()));
        let written = Self::decode_into(self, input, &mut out)?;
        out.truncate(written);
        Ok(out)
    }

    // ========================================================================
    // Internal Dispatchers
    // ========================================================================

    // TODO: Recalculate lengths for SIMDs paths.

    // `&self` (a 2-byte Copy `Engine`) is kept by-ref for consistency with the
    // rest of the `Engine` methods, not because the reference is required.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    #[inline]
    fn encode_dispatch(&self, input: &[u8], dst: &mut [u8]) {
        #[cfg(x86_simd)]
        {
            let len = input.len();
            let tier = cpu::tier();

            // Smart degrade by length: a kernel is only worth entering once the
            // input fills its vector width (64 for AVX512, 32 for AVX2).
            #[cfg(feature = "avx512-vbmi")]
            if len >= 64 && tier == cpu::AVX512_VBMI {
                // VBMI fast-path: vpermb replaces the 8-instruction char mapping.
                // SAFETY: tier() confirmed AVX-512F/BW/VBMI on this CPU.
                unsafe { simd::encode_slice_avx512_vbmi(&self.config, input, dst) };
                return;
            }
            #[cfg(feature = "avx512")]
            if len >= 64 && tier >= cpu::AVX512 {
                // SAFETY: tier() confirmed AVX-512F/BW on this CPU.
                unsafe { simd::encode_slice_avx512(&self.config, input, dst) };
                return;
            }
            #[cfg(feature = "avx2")]
            if len >= 32 && tier >= cpu::AVX2 {
                // SAFETY: tier() confirmed AVX2 on this CPU.
                unsafe { simd::encode_slice_avx2(&self.config, input, dst) };
                return;
            }
        }

        // NEON path (aarch64): compile-time dispatch, no runtime detection.
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if input.len() >= 16 {
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
        #[cfg(x86_simd)]
        {
            let len = input.len();
            let tier = cpu::tier();

            #[cfg(feature = "avx512-vbmi")]
            if len >= 64 && tier == cpu::AVX512_VBMI {
                // VBMI fast-path: vpermi2b collapses decode+validate to ~4 instructions.
                // SAFETY: tier() confirmed AVX-512F/BW/VBMI on this CPU.
                return unsafe { simd::decode_slice_avx512_vbmi(&self.config, input, dst) };
            }
            #[cfg(feature = "avx512")]
            if len >= 64 && tier >= cpu::AVX512 {
                // SAFETY: tier() confirmed AVX-512F/BW on this CPU.
                return unsafe { simd::decode_slice_avx512(&self.config, input, dst) };
            }
            #[cfg(feature = "avx2")]
            if len >= 32 && tier >= cpu::AVX2 {
                // SAFETY: tier() confirmed AVX2 on this CPU.
                return unsafe { simd::decode_slice_avx2(&self.config, input, dst) };
            }
        }

        // NEON path (aarch64): compile-time dispatch, no runtime detection.
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        if input.len() >= 16 {
            // SAFETY: NEON is baseline on aarch64.
            return unsafe { simd::decode_slice_neon(&self.config, input, dst) };
        }

        // Fallback: Scalar / non-SIMD target / short inputs.
        scalar::decode_slice(&self.config, input, dst)
    }

    // ========================================================================
    // Raw unsafe access (unstable feature)
    // ========================================================================

    /// Encodes a byte slice into Base64 using a highly optimized AVX2 SIMD implementation.
    ///
    /// This provides raw access to the direct AVX2 encoding logic.
    ///
    /// # Safety
    ///
    /// This function is **unsafe** and requires the caller to uphold strict memory contracts.
    /// Failure to do so will result in **undefined behavior** (e.g., buffer overflow).
    ///
    /// - The destination pointer `dst` must be valid and point to a mutable memory region with
    ///   sufficient capacity. The required size depends on `config.padding`:
    ///   - With padding: `input.len().div_ceil(3) * 4`
    ///   - Without padding: `(input.len() * 4).div_ceil(3)`
    ///   - Highly recommended: use `Engine::encoded_len` to compute length.
    ///
    /// - The caller **must** ensure the target CPU supports AVX2 instructions at runtime.
    ///   Executing this function on a CPU without AVX2 support will cause crashes or incorrect
    ///   behavior.
    ///
    /// # Warning
    ///
    /// This is a low-level, unsafe primitive. Misuse can lead to undefined behavior regardless
    /// of other crate guarantees. For better memory safety, use the safe higher-level APIs
    /// (e.g., `Engine::encode`).
    #[cfg(all(x86_simd, feature = "avx2", feature = "unstable"))]
    pub unsafe fn encode_avx2(&self, input: &[u8], dst: &mut [u8]) {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::encode_slice_avx2(&self.config, input, dst) }
    }

    /// Encodes a byte slice into Base64 using a highly optimized AVX2 SIMD implementation.
    ///
    /// This provides raw access to the direct AVX2 encoding logic.
    ///
    /// # Safety
    ///
    /// This function is **unsafe** and requires the caller to uphold strict memory contracts.
    /// Failure to do so will result in **undefined behavior** (e.g., buffer overflow).
    ///
    /// - The destination pointer `dst` must be valid and point to a mutable memory region with
    ///   sufficient capacity. The required size depends on `config.padding`:
    ///   - With padding: `input.len().div_ceil(3) * 4`
    ///   - Without padding: `(input.len() * 4).div_ceil(3)`
    ///
    /// - Highly recommended: use `Engine::estimate_decoded_len` to compute length.
    ///
    /// - The caller **must** ensure the target CPU supports AVX2 instructions at runtime.
    ///   Executing this function on a CPU without AVX2 support will cause an illegal instruction
    ///   crash.
    ///
    /// # Warning
    ///
    /// This is a low-level, unsafe primitive. Misuse can lead to undefined behavior regardless
    /// of other crate guarantees. For better memory safety, use the safe higher-level APIs
    /// (e.g., `Engine::encode`).
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

    /// Encodes a byte slice into Base64 using the optimized scalar (non-SIMD) algorithm.
    ///
    /// This provides raw access to the direct scalar encoding logic. Unlike the SIMD
    /// accessors, this is a **safe** function: the scalar kernel uses no `unsafe`,
    /// so every write is bounds-checked.
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

    /// Decodes a Base64 byte slice using the optimized scalar (non-SIMD) algorithm.
    ///
    /// This provides raw access to the direct scalar decoding logic. Like
    /// [`Engine::encode_scalar`], it is a **safe** function — the scalar kernel
    /// contains no `unsafe`, so a too-small `dst` panics on a bounds check rather
    /// than corrupting memory.
    ///
    /// Size `dst` with [`Engine::estimate_decoded_len`].
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

    /// Encodes a byte slice into Base64 using the NEON SIMD implementation.
    ///
    /// # Safety
    ///
    /// This function is **unsafe** and requires the caller to uphold strict memory contracts.
    /// Failure to do so will result in **undefined behavior** (e.g., buffer overflow).
    ///
    /// - The destination pointer `dst` must be valid and point to a mutable memory region with
    ///   at least `(input.len() / 4 + 1) * 3` bytes of capacity. The extra space is required due
    ///   to the implementation performing overlapping writes.
    ///  - Highly recommended: use `Engine::estimate_decoded_len` to compute length.
    ///
    /// # Warning
    ///
    /// This is a low-level, unsafe primitive. Misuse can lead to undefined behavior regardless
    /// of other crate guarantees. For better memory safety, use the safe higher-level APIs
    /// (e.g., `Engine::decode`).
    #[cfg(all(target_arch = "aarch64", feature = "neon", feature = "unstable"))]
    pub unsafe fn encode_neon(&self, input: &[u8], dst: &mut [u8]) {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::encode_slice_neon(&self.config, input, dst) }
    }

    /// Decodes a Base64 byte slice using the NEON SIMD implementation.
    ///
    /// # Safety
    ///
    /// This function is **unsafe** and requires the caller to uphold strict memory contracts.
    /// Failure to do so will result in **undefined behavior** (e.g., buffer overflow).
    ///
    /// - The destination pointer `dst` must be valid and point to a mutable memory region with
    ///   at least `(input.len() / 4 + 1) * 3` bytes of capacity. The extra space is required due
    ///   to the implementation performing overlapping writes.
    ///  - Highly recommended: use `Engine::estimate_decoded_len` to compute length.
    ///
    /// # Warning
    ///
    /// This is a low-level, unsafe primitive. Misuse can lead to undefined behavior regardless
    /// of other crate guarantees. For better memory safety, use the safe higher-level APIs
    /// (e.g., `Engine::decode`).
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
