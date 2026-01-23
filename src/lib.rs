//! # Base64 Turbo
//!
//! [![Crates.io](https://img.shields.io/crates/v/base64-turbo.svg)](https://crates.io/crates/base64-turbo)
//! [![Documentation](https://docs.rs/base64-turbo/badge.svg)](https://docs.rs/base64-turbo)
//! [![License](https://img.shields.io/github/license/hacer-bark/base64-turbo)](https://github.com/hacer-bark/base64-turbo/blob/main/LICENSE)
//! [![Formal Verification](https://img.shields.io/badge/Formal%20Verification-Kani%20Verified-success)](https://github.com/model-checking/kani)
//! [![MIRI Verified](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/miri.yml?label=MIRI%20Verified)](https://github.com/hacer-bark/base64-turbo/actions/workflows/miri.yml)
//! [![Logic Tests](https://img.shields.io/github/actions/workflow/status/hacer-bark/base64-turbo/tests.yml?label=Logic%20Tests)](https://github.com/hacer-bark/base64-turbo/actions/workflows/tests.yml)
//!
//! A SIMD-accelerated Base64 encoder/decoder for Rust, optimized for high-throughput systems.
//!
//! This crate provides runtime CPU detection to utilize AVX2, SSSE3, or AVX512 (via feature flag) intrinsics.
//! It includes a highly optimized scalar fallback for non-SIMD targets and supports `no_std` environments.
//!
//! ## Usage
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! base64-turbo = "0.1"
//! ```
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
//! For low-latency/HFT scenarios or `no_std` environments where heap allocation is undesirable.
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
//! This crate is highly configurable via Cargo features:
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | **`std`** | **Yes** | Enables `String` and `Vec` support. Disable this for `no_std` environments. |
//! | **`simd`** | **Yes** | Enables runtime detection for AVX2 and SSSE3 intrinsics. If disabled or unsupported by hardware, the crate falls back to scalar logic automatic. |
//! | **`parallel`** | **No** | Enables [Rayon](https://crates.io/crates/rayon) support. Automatically parallelizes processing for payloads larger than 512KB. Recommended only for massive data ingestion tasks. |
//! | **`avx512`** | **No** | Enables AVX512 intrinsics. |
//!
//! ## Safety & Verification
//!
//! This crate utilizes `unsafe` code for SIMD intrinsics and pointer arithmetic to achieve maximum performance.
//!
//! *   **Formal Verification (Kani):** Scalar (Done), SSSE3 (In Progress), AVX2 (Done), AVX512 (In Progress) code mathematic proven to be UB free and panic free.
//! *   **MIRI Tests:** Core SIMD logic and scalar fallbacks are verified with **MIRI** (Undefined Behavior checker) in CI.
//! *   **Fuzzing:** The codebase is fuzz-tested via `cargo-fuzz`.
//! *   **Fallback:** Invalid or unsupported hardware instruction sets are detected at runtime, ensuring safe fallback to scalar code.

// TODO: Add docs for SIMD
// TODO: Update SSSE3 and AVX512 logic for new algo
// TODO: Investigate low speed at small payloads

#![cfg_attr(not(any(feature = "std", test)), no_std)]

#![doc(issue_tracker_base_url = "https://github.com/hacer-bark/base64-turbo/issues/")]

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![warn(unused_qualifications)]

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "simd")]
mod simd;
mod scalar;

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
            Error::InvalidLength => write!(f, "Invalid Base64 input length (must be divisible by 4)"),
            Error::InvalidCharacter => write!(f, "Invalid character found in Base64 input"),
            Error::BufferTooSmall => write!(f, "Destination buffer is too small"),
        }
    }
}

// Enable std::error::Error trait when the 'std' feature is active
#[cfg(feature = "std")]
impl std::error::Error for Error {}

// ======================================================================
// Internal
// Tuning Constants (Parallelism)
// ======================================================================

/// Input chunk size for parallel processing (24 KB).
///
/// This size is chosen to fit comfortably within the L1/L2 cache of most modern
/// CPUs, ensuring that hot loops inside the encoder stay cache-resident.
#[cfg(feature = "parallel")]
const ENCODE_CHUNK_SIZE: usize = 24 * 1024;

/// Output chunk size corresponding to `ENCODE_CHUNK_SIZE`.
///
/// Base64 encoding expands data by 4/3. For a 24KB input, the output is 32KB.
#[cfg(feature = "parallel")]
const DECODE_CHUNK_SIZE: usize = (ENCODE_CHUNK_SIZE / 3) * 4;

/// Threshold to enable Rayon parallelism (512 KB).
///
/// For payloads smaller than this, the overhead of context switching and
/// thread synchronization outweighs the throughput gains of multi-threading.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 512 * 1024;

// ======================================================================
// Internal Lookup Tables
// ======================================================================

/// The Standard RFC 4648 Base64 Alphabet.
/// Used for `STANDARD` and `STANDARD_NO_PAD`.
const STANDARD_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Computed compile-time reverse lookup table for the Standard alphabet.
/// Maps ASCII bytes back to 6-bit indices. 0xFF indicates an invalid character.
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
const URL_SAFE_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Computed compile-time reverse lookup table for the URL-Safe alphabet.
/// Maps ASCII bytes back to 6-bit indices. 0xFF indicates an invalid character.
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
///
/// This struct uses `repr(C)` to ensure predictable memory layout.
#[repr(C)]
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
/// // Decode to Vec<u8>
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
    /// # Parallelism
    /// If the `parallel` feature is enabled and the input size exceeds the
    /// internal threshold (default: 512KB), this method automatically uses
    /// Rayon to process chunks in parallel, saturating memory bandwidth.
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
    #[inline]
    pub fn encode_into<T: AsRef<[u8]> + Sync>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }

        let req_len = Self::encoded_len(self, len);
        if output.len() < req_len {
            return Err(Error::BufferTooSmall);
        }

        // --- Parallel Path ---
        #[cfg(feature = "parallel")]
        {
            if len >= PARALLEL_THRESHOLD {
                // Split input and output into corresponding chunks
                let out_slice = &mut output[..req_len];

                // Base64 expands 3 bytes -> 4 chars. 
                // We chunk based on ENCODE_CHUNK_SIZE (24KB) to stay cache-friendly.
                out_slice
                    .par_chunks_mut((ENCODE_CHUNK_SIZE / 3) * 4)
                    .zip(input.par_chunks(ENCODE_CHUNK_SIZE))
                    .for_each(|(out_chunk, in_chunk)| {
                        // Safe: We know the chunk sizes match the expansion ratio logic
                        Self::encode_dispatch(self, in_chunk, out_chunk.as_mut_ptr());
                    });
                
                return Ok(req_len);
            }
        }

        // --- Serial Path ---
        // Pass the raw pointer to the dispatcher. 
        // SAFETY: We checked output.len() >= req_len above.
        Self::encode_dispatch(self, input, output[..req_len].as_mut_ptr());

        Ok(req_len)
    }

    /// Decodes `input` into the provided `output` buffer.
    ///
    /// # Performance
    /// Like encoding, this method supports automatic parallelization for large payloads.
    /// It verifies the validity of the Base64 input while decoding.
    ///
    /// # Returns
    ///
    /// * `Ok(usize)`: The actual number of bytes written to `output`.
    /// * `Err(Error)`: If the input is invalid or the buffer is too small.
    #[inline]
    pub fn decode_into<T: AsRef<[u8]> + Sync>(
        &self,
        input: T,
        output: &mut [u8],
    ) -> Result<usize, Error> {
        let input = input.as_ref();
        let len = input.len();

        if len == 0 {
            return Ok(0);
        }

        let req_len = Self::estimate_decoded_len(self, len);
        if output.len() < req_len {
            return Err(Error::BufferTooSmall);
        }

        // --- Parallel Path ---
        #[cfg(feature = "parallel")]
        {
            if len >= PARALLEL_THRESHOLD {
                let out_slice = &mut output[..req_len];

                // Parallel Reduce:
                // 1. Split input/output into chunks.
                // 2. Decode chunks independently.
                // 3. Sum the number of bytes written by each chunk.
                // 4. Return error if any chunk fails.
                let real_len = out_slice
                    .par_chunks_mut((DECODE_CHUNK_SIZE / 4) * 3)
                    .zip(input.par_chunks(DECODE_CHUNK_SIZE))
                    .try_fold(
                        || 0usize,
                        |acc, (out_chunk, in_chunk)| {
                            let written = Self::decode_dispatch(self, in_chunk, out_chunk.as_mut_ptr())?;
                            Ok(acc + written)
                        },
                    )
                    .try_reduce(
                        || 0usize,
                        |a, b| Ok(a + b),
                    )?;

                return Ok(real_len);
            }
        }

        // --- Serial Path ---
        let real_len = Self::decode_dispatch(self, input, output[..req_len].as_mut_ptr())?;

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
    pub fn encode<T: AsRef<[u8]> + Sync>(&self, input: T) -> String {
        let input = input.as_ref();

        // 1. Calculate EXACT required size. Base64 encoding is deterministic.
        let len = Self::encoded_len(self, input.len());

        // 2. Allocate uninitialized buffer
        let mut out = Vec::with_capacity(len);

        // 3. Set length immediately
        // SAFETY: We are about to overwrite the entire buffer in `encode_into`.
        // We require a valid `&mut [u8]` slice for the internal logic (especially Rayon) to work.
        // Since `encode_into` guarantees it writes exactly `len` bytes or fails (and we panic on fail),
        // we won't expose uninitialized memory.
        #[allow(clippy::uninit_vec)]
        unsafe { out.set_len(len); }

        // 4. Encode
        // We trust our `encoded_len` math completely.
        Self::encode_into(self, input, &mut out).expect("Base64 logic error: buffer size mismatch");

        // 5. Convert to String
        // SAFETY: The Base64 alphabet consists strictly of ASCII characters,
        // which are valid UTF-8.
        unsafe { String::from_utf8_unchecked(out) }
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
    pub fn decode<T: AsRef<[u8]> + Sync>(&self, input: T) -> Result<Vec<u8>, Error> {
        let input = input.as_ref();

        // 1. Calculate MAXIMUM required size (upper bound)
        let max_len = Self::estimate_decoded_len(self, input.len());

        // 2. Allocate buffer
        let mut out = Vec::with_capacity(max_len);

        // 3. Set length to MAX
        // SAFETY: We temporarily expose uninitialized memory to the `decode_into` function
        // so it can write into the slice. We strictly sanitize the length in step 5.
        #[allow(clippy::uninit_vec)]
        unsafe { out.set_len(max_len); }

        // 4. Decode
        // `decode_into` handles parallel/serial dispatch and returns the `actual_len`.
        match Self::decode_into(self, input, &mut out) {
            Ok(actual_len) => {
                // 5. Shrink to fit the real data
                // SAFETY: `decode_into` reported it successfully wrote `actual_len` valid bytes.
                // We truncate the Vec to this length, discarding any trailing garbage/uninitialized memory.
                unsafe { out.set_len(actual_len); }
                Ok(out)
            }
            Err(e) => {
                // SAFETY: If an error occurred, we force the length to 0.
                // This prevents the caller from accidentally inspecting uninitialized memory
                // if they were to (incorrectly) reuse the Vec from a partial result.
                unsafe { out.set_len(0); }
                Err(e)
            }
        }
    }

    // ========================================================================
    // Internal Dispatchers
    // ========================================================================

    #[inline(always)]
    fn encode_dispatch(&self, input: &[u8], dst: *mut u8) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "simd")]
        {
            let len = input.len();

            #[cfg(feature = "avx512")]
            // Smart degrade: If len < 64, don't bother checking AVX512 features or setting up ZMM register
            if len >= 64 
                && std::is_x86_feature_detected!("avx512f") 
                && std::is_x86_feature_detected!("avx512bw") 
            {
                unsafe { simd::encode_slice_avx512(&self.config, input, dst); }
                return;
            }

            // Smart degrade: If len < 32, skip AVX2.
            if len >= 32 && std::is_x86_feature_detected!("avx2") {
                unsafe { simd::encode_slice_avx2(&self.config, input, dst); }
                return;
            }

            // Smart degrade: If len < 16, skip SSSE3 and go straight to scalar.
            if len >= 16 && std::is_x86_feature_detected!("ssse3") {
                unsafe { simd::encode_slice_simd(&self.config, input, dst); }
                return;
            }
        }

        // Fallback: Scalar / Non-x86 / Short inputs
        // Safety: Pointers verified by caller
        unsafe { scalar::encode_slice_unsafe(&self.config, input, dst); }
    }

    #[inline(always)]
    fn decode_dispatch(&self, input: &[u8], dst: *mut u8) -> Result<usize, Error> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[cfg(feature = "simd")]
        {
            let len = input.len();

            #[cfg(feature = "avx512")]
            // Smart degrade: Don't enter AVX512 path if we don't have a full vector of input.
            if len >= 64 
                && std::is_x86_feature_detected!("avx512f") 
                && std::is_x86_feature_detected!("avx512bw") 
            {
                return unsafe { simd::decode_slice_avx512(&self.config, input, dst) };
            }

            // Smart degrade: Fallback to AVX2 if len is between 32 and 64, or if AVX512 is missing.
            if len >= 32 && std::is_x86_feature_detected!("avx2") {
                return unsafe { simd::decode_slice_avx2(&self.config, input, dst) };
            }

            // Smart degrade: Fallback to SSSE3 if len is between 16 and 32.
            if len >= 16 && std::is_x86_feature_detected!("ssse3") {
                return unsafe { simd::decode_slice_simd(&self.config, input, dst) };
            }
        }

        // Fallback: Scalar / Non-x86 / Short inputs
        // Safety: Pointers verified by caller
        unsafe { scalar::decode_slice_unsafe(&self.config, input, dst) }
    }
}
