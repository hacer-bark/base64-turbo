//! Base64 alphabets: the two RFC 4648 built-ins and user-supplied custom ones.
//!
//! An [`Alphabet`] owns every lookup table the kernels drive from it, so
//! picking an alphabet is a pointer load rather than a branch:
//!
//! * `chars` — 6-bit index -> character. Also the AVX-512-VBMI `vpermb` encode
//!   control, loaded straight out of this field.
//! * `decode` — character -> 6-bit index, `0xFF` for invalid. Its first 128
//!   entries are the VBMI `vpermi2b` decode control, for the same reason.
//! * `pairs` / `shifted` — the widened scalar tables described in
//!   [`crate::scalar`].
//!
//! That is ~12 KiB per alphabet, so [`Alphabet::new`] is meant to build a
//! `static` once, not a value per call.

use core::fmt;

/// The Standard RFC 4648 alphabet.
const STANDARD_CHARS: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The URL-safe RFC 4648 alphabet: `+` and `/` swapped for `-` and `_`.
const URL_SAFE_CHARS: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Which kernels an alphabet may run on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Standard,
    UrlSafe,
    /// Anything else. The AVX2 and NEON kernels derive characters arithmetically
    /// from the RFC 4648 layout, so they cannot serve these; the scalar and
    /// AVX-512-VBMI kernels are pure table lookups and can.
    Custom,
}

/// Maps a 12-bit value to the two characters it encodes, packed little-endian
/// so the first character lands in the low byte.
const fn encode_pair_table(chars: &[u8; 64]) -> [u16; 4096] {
    let mut table = [0u16; 4096];
    let mut i = 0;
    while i < 4096 {
        table[i] = (chars[i >> 6] as u16) | ((chars[i & 0x3F] as u16) << 8);
        i += 1;
    }
    table
}

/// Reverse lookup with the 6-bit index pre-shifted into its position within a
/// 24-bit group. Invalid characters map to `u32::MAX`, so OR-ing a whole group
/// together pushes the result above `0x00FF_FFFF` if any character was bad.
const fn decode_shift_table(chars: &[u8; 64], shift: u32) -> [u32; 256] {
    let mut table = [u32::MAX; 256];
    let mut i: u32 = 0;
    while i < 64 {
        table[chars[i as usize] as usize] = i << shift;
        i += 1;
    }
    table
}

const fn same_chars(a: &[u8; 64], b: &[u8; 64]) -> bool {
    let mut i = 0;
    while i < 64 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// A Base64 alphabet and the lookup tables derived from it.
///
/// Two are built in — the RFC 4648 standard and URL-safe sets, behind the
/// [`STANDARD`](crate::STANDARD) and [`URL_SAFE`](crate::URL_SAFE) engines. Any
/// other 64-character set goes through [`Alphabet::new`] and
/// [`Engine::custom`](crate::Engine::custom).
///
/// # Examples
///
/// ```
/// use base64_turbo::{Alphabet, Engine};
///
/// // The bcrypt/crypt(3) alphabet: `.` and `/` first, digits before letters.
/// static BCRYPT: Alphabet = match Alphabet::new(
///     b"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
/// ) {
///     Some(a) => a,
///     None => unreachable!(),
/// };
/// static ENGINE: Engine = Engine::custom(&BCRYPT, false);
///
/// # #[cfg(feature = "std")] {
/// assert_eq!(ENGINE.encode(b"hello"), "YETqZE6");
/// assert_eq!(ENGINE.decode("YETqZE6").unwrap(), b"hello");
/// # }
/// ```
pub struct Alphabet {
    kind: Kind,
    /// 6-bit index -> character.
    chars: [u8; 64],
    /// Character -> 6-bit index, `0xFF` for invalid.
    decode: [u8; 256],
    /// 12 bits of input -> the two characters they encode.
    pairs: [u16; 4096],
    /// Character -> index, pre-shifted into each of the four group positions.
    shifted: [[u32; 256]; 4],
}

impl Alphabet {
    /// Builds an alphabet and its tables. No validation: callers must have
    /// established the [`Alphabet::new`] invariants.
    #[allow(clippy::cast_possible_truncation)] // `i` is always < 64, fits in u8
    const fn build(chars: &[u8; 64]) -> Self {
        let mut decode = [0xFFu8; 256];
        let mut i = 0;
        while i < 64 {
            decode[chars[i] as usize] = i as u8;
            i += 1;
        }

        let kind = if same_chars(chars, STANDARD_CHARS) {
            Kind::Standard
        } else if same_chars(chars, URL_SAFE_CHARS) {
            Kind::UrlSafe
        } else {
            Kind::Custom
        };

        Self {
            kind,
            chars: *chars,
            decode,
            pairs: encode_pair_table(chars),
            shifted: [
                decode_shift_table(chars, 18),
                decode_shift_table(chars, 12),
                decode_shift_table(chars, 6),
                decode_shift_table(chars, 0),
            ],
        }
    }

    /// Builds a custom alphabet from 64 characters, index 0 first.
    ///
    /// Returns `None` unless every character is
    ///
    /// * printable ASCII (`!`, `0x21`, through `~`, `0x7E`) — control bytes,
    ///   whitespace and non-ASCII are rejected, and the ASCII bound is what lets
    ///   the AVX-512-VBMI kernel validate a whole vector with one lookup;
    /// * not `=`, which is reserved for padding regardless of alphabet; and
    /// * distinct from the other 63.
    ///
    /// The tables are ~12 KiB, so build this once into a `static` rather than
    /// per call. It is a `const fn`, so that costs nothing at run time:
    ///
    /// ```
    /// use base64_turbo::Alphabet;
    ///
    /// static ORDERED: Alphabet = match Alphabet::new(
    ///     b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz",
    /// ) {
    ///     Some(a) => a,
    ///     None => unreachable!(),
    /// };
    /// # let _ = &ORDERED;
    /// ```
    ///
    /// Passing the standard or URL-safe characters here yields an alphabet
    /// indistinguishable from the built-in one, SIMD coverage included.
    #[must_use]
    pub const fn new(chars: &[u8; 64]) -> Option<Self> {
        let mut seen = [false; 256];
        let mut i = 0;
        while i < 64 {
            let c = chars[i];
            if c < b'!' || c > b'~' || c == b'=' || seen[c as usize] {
                return None;
            }
            seen[c as usize] = true;
            i += 1;
        }
        Some(Self::build(chars))
    }

    /// The 64 characters, index 0 first.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 64] {
        &self.chars
    }

    /// Character -> 6-bit index, `0xFF` for invalid. The VBMI decoder loads the
    /// first 128 entries as its `vpermi2b` control, which is why the `new`
    /// invariants cap characters at `0x7E`.
    #[inline]
    pub(crate) const fn decode_table(&self) -> &[u8; 256] {
        &self.decode
    }

    /// The widened scalar encode table.
    #[inline]
    pub(crate) const fn pairs(&self) -> &[u16; 4096] {
        &self.pairs
    }

    /// The four position-shifted scalar decode tables.
    #[inline]
    pub(crate) const fn shifted(&self) -> &[[u32; 256]; 4] {
        &self.shifted
    }

    /// Whether the AVX2 and NEON kernels can serve this alphabet.
    #[cfg(arithmetic_simd)]
    #[inline]
    pub(crate) const fn has_arithmetic_kernels(&self) -> bool {
        !matches!(self.kind, Kind::Custom)
    }

    /// Selects the `+/` or `-_` constants baked into those kernels.
    #[cfg(arithmetic_simd)]
    #[inline]
    pub(crate) const fn is_url_safe(&self) -> bool {
        matches!(self.kind, Kind::UrlSafe)
    }
}

/// Prints the alphabet itself, not the 12 KiB of tables behind it.
impl fmt::Debug for Alphabet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Alphabet")
            .field("kind", &self.kind)
            .field("chars", &core::str::from_utf8(&self.chars))
            .finish_non_exhaustive()
    }
}

/// Tables for the standard RFC 4648 alphabet.
pub(crate) static STANDARD_TABLE: Alphabet = Alphabet::build(STANDARD_CHARS);

/// Tables for the URL-safe RFC 4648 alphabet.
pub(crate) static URL_SAFE_TABLE: Alphabet = Alphabet::build(URL_SAFE_CHARS);

/// Picks a built-in by the flag the older `Config` carried. Used by the
/// verification harnesses, which enumerate both built-ins by boolean. Those
/// live under `test`/`kani` on x86 and under Miri only on aarch64.
#[cfg(any(
    all(x86_simd, any(test, kani)),
    all(target_arch = "aarch64", feature = "neon", test, miri)
))]
pub(crate) const fn builtin(url_safe: bool) -> &'static Alphabet {
    if url_safe {
        &URL_SAFE_TABLE
    } else {
        &STANDARD_TABLE
    }
}
