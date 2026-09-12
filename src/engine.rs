//! The [`Engine`] type, the ready-made engines, and the kernel dispatchers.
//!
//! An engine is just an alphabet plus two padding rules, so it is `Copy` and
//! carries no state; all the machinery here is in the two dispatchers, which
//! pick a kernel per call from the runtime CPU tier and the input length.

use crate::{Alphabet, Error, alphabet, scalar};

#[cfg(x86_simd)]
use crate::cpu;
#[cfg(unsafe_simd)]
use crate::simd;

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

/// A `len`-byte buffer for a dispatcher to fill: uninitialized on SIMD builds.
///
/// This and [`into_ascii_string`] isolate the one place the SIMD and
/// scalar-only builds genuinely differ. The SIMD build already contains
/// `unsafe`, so it skips the zeroing and the UTF-8 validation; the scalar-only
/// build forbids `unsafe`, so it pays a linear pass for the same result.
/// [`Engine::encode`] and [`Engine::decode`] stay identical across both.
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

        // `output` was just checked to be at least `req_len` bytes.
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

        let real_len = Self::decode_dispatch(self, input, &mut output[..req_len])?;

        Ok(real_len)
    }

    /// Returns whether this engine emits padding when encoding.
    #[inline]
    #[must_use]
    pub const fn encode_padding(&self) -> bool {
        self.config.padding
    }

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
            #[cfg(b64_avx512)]
            if len >= 16 && tier == cpu::AVX512_VBMI {
                // VBMI fast-path: vpermb replaces the 8-instruction char mapping.
                // SAFETY: tier() confirmed AVX-512F/BW/VBMI on this CPU.
                unsafe { simd::encode_slice_avx512_vbmi(&self.config, input, dst) };
                return;
            }
            #[cfg(b64_avx2)]
            if len >= 48 && tier >= cpu::AVX2 && self.config.alphabet.has_arithmetic_kernels() {
                // SAFETY: tier() confirmed AVX2 on this CPU.
                unsafe { simd::encode_slice_avx2(&self.config, input, dst) };
                return;
            }
        }

        // NEON path (aarch64): compile-time dispatch, no runtime detection.
        #[cfg(b64_neon)]
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
            #[cfg(b64_avx512)]
            if len >= 28 && tier == cpu::AVX512_VBMI {
                // VBMI fast-path: vpermi2b collapses decode+validate to ~4 instructions.
                // SAFETY: tier() confirmed AVX-512F/BW/VBMI on this CPU.
                return unsafe { simd::decode_slice_avx512_vbmi(&self.config, input, dst) };
            }
            #[cfg(b64_avx2)]
            if len >= 36 && tier >= cpu::AVX2 && self.config.alphabet.has_arithmetic_kernels() {
                // SAFETY: tier() confirmed AVX2 on this CPU.
                return unsafe { simd::decode_slice_avx2(&self.config, input, dst) };
            }
        }

        // NEON path (aarch64): compile-time dispatch, no runtime detection.
        // Its single tier is a 16-in/12-out block plus a 4-byte read-ahead
        // margin (see `neon::decode_slice_neon`), so it needs 20 bytes to run.
        #[cfg(b64_neon)]
        if input.len() >= 20 && self.config.alphabet.has_arithmetic_kernels() {
            // SAFETY: NEON is baseline on aarch64.
            return unsafe { simd::decode_slice_neon(&self.config, input, dst) };
        }

        // Fallback: Scalar / non-SIMD target / short inputs.
        scalar::decode_slice(&self.config, input, dst)
    }

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
    #[cfg(all(b64_avx2, feature = "unstable"))]
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
    #[cfg(all(b64_avx2, feature = "unstable"))]
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
    #[cfg(all(b64_avx512, feature = "unstable"))]
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
    #[cfg(all(b64_avx512, feature = "unstable"))]
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
    #[cfg(all(b64_neon, feature = "unstable"))]
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
    #[cfg(all(b64_neon, feature = "unstable"))]
    pub unsafe fn decode_neon(&self, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
        // SAFETY: Caller must uphold the contracts documented on this function.
        unsafe { simd::decode_slice_neon(&self.config, input, dst) }
    }
}
