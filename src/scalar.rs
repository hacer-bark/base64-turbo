//! Scalar (non-SIMD) Base64 encode/decode primitives.
//!
//! This module is **100% safe Rust**: `unsafe` is forbidden crate-wide for this
//! file (see the inner attribute below). Both primitives take a `&mut [u8]`
//! destination, so every write is bounds-checked at compile time / runtime
//! rather than relying on a caller-upheld pointer contract. Index math on the
//! alphabet and decode tables is masked into range (`& 0x3F`, or a `u8` index
//! into a 256-entry table), so those lookups compile to bounds-check-free code.

#![forbid(unsafe_code)]
// The rest of the crate threads `&Config` everywhere (dispatch + SIMD); keep the
// scalar primitives consistent rather than special-casing a by-value `Config`.
#![allow(clippy::trivially_copy_pass_by_ref)]

use crate::{
    Config, Error, STANDARD_ALPHABET, STANDARD_DECODE_TABLE, URL_SAFE_ALPHABET,
    URL_SAFE_DECODE_TABLE,
};

/// Encodes `input` into Base64, writing the result into `dst`.
///
/// `dst` must be at least the encoded length for `input`:
/// * padded:   `input.len().div_ceil(3) * 4`
/// * unpadded: `(input.len() * 4).div_ceil(3)`
///
/// A `dst` that is too small will panic (bounds check) rather than corrupt
/// memory. Callers should prefer the safe, higher-level APIs (e.g.
/// `Engine::encode`), which size the buffer automatically.
#[inline]
pub(crate) fn encode_slice(config: &Config, input: &[u8], dst: &mut [u8]) {
    // Select the alphabet based on configuration. This branch predicts
    // perfectly since config doesn't change during the loop.
    let alphabet = if config.url_safe {
        URL_SAFE_ALPHABET
    } else {
        STANDARD_ALPHABET
    };

    let len = input.len();
    let blocks = len / 6; // full 6-byte input blocks

    // Split input/output into the fast-loop region and the tail. Using
    // `chunks_exact` over the split halves keeps the hot loop free of bounds
    // checks (the chunk lengths are statically known: 6 in, 8 out).
    let (in_main, in_tail) = input.split_at(blocks * 6);
    let (out_main, out_tail) = dst.split_at_mut(blocks * 8);

    // --- MAIN LOOP ---
    // Process 6 input bytes -> 8 output bytes per iteration.
    for (chunk, out) in in_main.chunks_exact(6).zip(out_main.chunks_exact_mut(8)) {
        // Read two overlapping big-endian u32s to avoid complex shifting logic.
        let reg_a = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let reg_b = u32::from_be_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]);

        let n1 = (reg_a >> 8) as usize; // Bytes 0, 1, 2
        let n2 = (reg_b & 0x00_FF_FF_FF) as usize; // Bytes 3, 4, 5

        // Map indices to Base64 characters and pack into a single u64.
        let pack = u64::from(alphabet[(n1 >> 18) & 0x3F])
            | (u64::from(alphabet[(n1 >> 12) & 0x3F]) << 8)
            | (u64::from(alphabet[(n1 >> 6) & 0x3F]) << 16)
            | (u64::from(alphabet[n1 & 0x3F]) << 24)
            | (u64::from(alphabet[(n2 >> 18) & 0x3F]) << 32)
            | (u64::from(alphabet[(n2 >> 12) & 0x3F]) << 40)
            | (u64::from(alphabet[(n2 >> 6) & 0x3F]) << 48)
            | (u64::from(alphabet[n2 & 0x3F]) << 56);

        out.copy_from_slice(&pack.to_le_bytes());
    }

    // --- TAIL HANDLING ---
    // `in_tail` is 0..=5 bytes. Track offsets into the tail halves.
    let mut ti = 0; // offset into in_tail
    let mut oi = 0; // offset into out_tail

    // Handle a remaining full 3-byte chunk (4 output chars).
    if in_tail.len() - ti >= 3 {
        let n = (usize::from(in_tail[ti]) << 16)
            | (usize::from(in_tail[ti + 1]) << 8)
            | usize::from(in_tail[ti + 2]);

        let packed = u32::from(alphabet[(n >> 18) & 0x3F])
            | (u32::from(alphabet[(n >> 12) & 0x3F]) << 8)
            | (u32::from(alphabet[(n >> 6) & 0x3F]) << 16)
            | (u32::from(alphabet[n & 0x3F]) << 24);

        out_tail[oi..oi + 4].copy_from_slice(&packed.to_le_bytes());
        ti += 3;
        oi += 4;
    }

    // Handle the final 1 or 2 bytes with padding logic.
    let rem = in_tail.len() - ti;
    if rem > 0 {
        let b0 = usize::from(in_tail[ti]);
        let b1 = if rem == 2 {
            usize::from(in_tail[ti + 1])
        } else {
            0
        };
        let n = (b0 << 16) | (b1 << 8);

        // The first 2 characters are always present.
        out_tail[oi] = alphabet[(n >> 18) & 0x3F];
        out_tail[oi + 1] = alphabet[(n >> 12) & 0x3F];

        // Handle the 3rd and 4th characters (data vs padding).
        if rem == 2 {
            out_tail[oi + 2] = alphabet[(n >> 6) & 0x3F];
            if config.padding {
                out_tail[oi + 3] = b'=';
            }
        } else if config.padding {
            out_tail[oi + 2] = b'=';
            out_tail[oi + 3] = b'=';
        }
    }
}

/// Decodes a Base64 `input` into `dst`, returning the number of bytes written.
///
/// Unlike the SIMD paths, this writes exactly the decoded bytes (no overlapping
/// over-writes), so `dst` only needs to be as large as the true decoded length.
/// A `dst` that is too small will panic (bounds check) rather than corrupt
/// memory. Callers should prefer the safe, higher-level APIs (e.g.
/// `Engine::decode`).
///
/// # Errors
/// Returns [`Error::InvalidCharacter`] or [`Error::InvalidLength`] if `input` is
/// not valid Base64 for `config`.
#[inline]
pub(crate) fn decode_slice(config: &Config, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
    let len = input.len();
    if len == 0 {
        return Ok(0);
    }

    // The table maps valid characters to 0..=63 and invalid characters to 0xFF.
    let table = if config.url_safe {
        &URL_SAFE_DECODE_TABLE
    } else {
        &STANDARD_DECODE_TABLE
    };

    // Fast loop bounds: process 8 input bytes -> 6 output bytes per iteration,
    // reserving the last 4 input bytes so the tail can handle padding carefully.
    let len_safe = len.saturating_sub(4);
    let len_fast = len_safe - (len_safe % 8);

    let mut i = 0; // input offset
    let mut o = 0; // output offset

    // --- FAST LOOP (Middle Chunks) ---
    while i < len_fast {
        // Load 8 bytes and look them up in the table.
        let d0 = table[usize::from(input[i])];
        let d1 = table[usize::from(input[i + 1])];
        let d2 = table[usize::from(input[i + 2])];
        let d3 = table[usize::from(input[i + 3])];
        let d4 = table[usize::from(input[i + 4])];
        let d5 = table[usize::from(input[i + 5])];
        let d6 = table[usize::from(input[i + 6])];
        let d7 = table[usize::from(input[i + 7])];

        // Valid characters map to 0..=63 (00xxxxxx); invalid map to 0xFF.
        // OR-ing accumulates the high bits, so any invalid char sets 0xC0.
        if (d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7) & 0xC0 != 0 {
            return Err(Error::InvalidCharacter);
        }

        // Pack each group of 4x 6-bit indices into a 24-bit value.
        let n1 =
            (u32::from(d0) << 18) | (u32::from(d1) << 12) | (u32::from(d2) << 6) | u32::from(d3);
        let n2 =
            (u32::from(d4) << 18) | (u32::from(d5) << 12) | (u32::from(d6) << 6) | u32::from(d7);

        // Write exactly 6 bytes (3 per group), in one bounds-checked copy.
        dst[o..o + 6].copy_from_slice(&[
            ((n1 >> 16) & 0xFF) as u8,
            ((n1 >> 8) & 0xFF) as u8,
            (n1 & 0xFF) as u8,
            ((n2 >> 16) & 0xFF) as u8,
            ((n2 >> 8) & 0xFF) as u8,
            (n2 & 0xFF) as u8,
        ]);

        i += 8;
        o += 6;
    }

    decode_tail(config, table, input, i, dst, o)
}

/// Decodes the final input bytes (from offset `i`) of a scalar decode pass,
/// including any padding logic. Split out of [`decode_slice`] purely to keep
/// that function under the `clippy::too_many_lines` threshold.
#[inline]
fn decode_tail(
    config: &Config,
    table: &[u8; 256],
    input: &[u8],
    mut i: usize,
    dst: &mut [u8],
    mut o: usize,
) -> Result<usize, Error> {
    let len = input.len();

    while i < len {
        let remaining = len - i;

        // Case A: Full 4-byte block (possibly containing padding at the end).
        if remaining >= 4 {
            let b0 = input[i];
            let b1 = input[i + 1];
            let b2 = input[i + 2];
            let b3 = input[i + 3];

            let d0 = table[usize::from(b0)];
            let d1 = table[usize::from(b1)];

            // Check for padding ('=').
            if b3 == b'=' {
                if b2 == b'=' {
                    // "XX==" -> 1 byte output
                    if (d0 | d1) & 0xC0 != 0 {
                        return Err(Error::InvalidCharacter);
                    }
                    let n = (u32::from(d0) << 18) | (u32::from(d1) << 12);
                    dst[o] = ((n >> 16) & 0xFF) as u8;
                    o += 1;
                } else {
                    // "XXX=" -> 2 bytes output
                    let d2 = table[usize::from(b2)];
                    if (d0 | d1 | d2) & 0xC0 != 0 {
                        return Err(Error::InvalidCharacter);
                    }
                    let n = (u32::from(d0) << 18) | (u32::from(d1) << 12) | (u32::from(d2) << 6);
                    dst[o] = ((n >> 16) & 0xFF) as u8;
                    dst[o + 1] = ((n >> 8) & 0xFF) as u8;
                    o += 2;
                }
                // Padding signals the end of the stream.
                return Ok(o);
            }

            // No padding: "XXXX" -> 3 bytes output
            let d2 = table[usize::from(b2)];
            let d3 = table[usize::from(b3)];

            if (d0 | d1 | d2 | d3) & 0xC0 != 0 {
                return Err(Error::InvalidCharacter);
            }

            let n = (u32::from(d0) << 18)
                | (u32::from(d1) << 12)
                | (u32::from(d2) << 6)
                | u32::from(d3);
            dst[o..o + 3].copy_from_slice(&[
                ((n >> 16) & 0xFF) as u8,
                ((n >> 8) & 0xFF) as u8,
                (n & 0xFF) as u8,
            ]);

            i += 4;
            o += 3;
        } else {
            // Case B: Partial block (1-3 bytes left).
            // If padding is strictly required, this is an error (len % 4 != 0).
            if config.padding {
                return Err(Error::InvalidLength);
            }

            let d0 = table[usize::from(input[i])];

            if remaining == 1 {
                // A single byte is invalid in Base64 (cannot form a full byte).
                return Err(Error::InvalidLength);
            }

            let d1 = table[usize::from(input[i + 1])];
            if (d0 | d1) & 0xC0 != 0 {
                return Err(Error::InvalidCharacter);
            }

            let mut n = (u32::from(d0) << 18) | (u32::from(d1) << 12);

            if remaining == 2 {
                // "XY" -> 1 byte output
                dst[o] = ((n >> 16) & 0xFF) as u8;
                o += 1;
            } else {
                // "XYZ" -> 2 bytes output
                let d2 = table[usize::from(input[i + 2])];
                if d2 & 0xC0 != 0 {
                    return Err(Error::InvalidCharacter);
                }

                n |= u32::from(d2) << 6;
                dst[o] = ((n >> 16) & 0xFF) as u8;
                dst[o + 1] = ((n >> 8) & 0xFF) as u8;
                o += 2;
            }

            break;
        }
    }

    Ok(o)
}
