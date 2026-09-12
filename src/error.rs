//! The error type shared by every encode and decode entry point.

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

#[cfg(feature = "std")]
impl std::error::Error for Error {}
