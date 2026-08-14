//! Scalar (non-SIMD) Base64 encode/decode primitives.
//!
//! This module is **100% safe Rust**: `unsafe` is forbidden crate-wide for this
//! file (see the inner attribute below). Both primitives take a `&mut [u8]`
//! destination, so every write is bounds-checked at compile time / runtime
//! rather than relying on a caller-upheld pointer contract. Every table index is
//! masked into range (`& 0x3F`, `& 0xFFF`, or a `u8` index into a 256-entry
//! table), so those lookups compile to bounds-check-free code.
//!
//! Both kernels are table-driven and limited by retired *loads* rather than by
//! arithmetic, so both tables are widened to cut the number of lookups per byte:
//!
//! * encode: one 8 KiB table per alphabet maps 12 input bits straight to the two
//!   output characters they encode, halving the lookups (8 -> 4 per 6-byte
//!   block).
//! * decode: four 1 KiB tables per alphabet fold the `<< 18 / << 12 / << 6`
//!   position shifts into the lookup itself, so decoding a 4-character group is
//!   four loads OR-ed together, with validation falling out of the same OR.
//!
//! That costs 24 KiB of `.rodata` and roughly doubles both kernels. The narrow
//! `*_ALPHABET` / `*_DECODE_TABLE` tables are still used by the decode tail,
//! where a handful of bytes cannot amortize a wide table's cache footprint.

#![forbid(unsafe_code)]
// The rest of the crate threads `&Config` everywhere (dispatch + SIMD); keep the
// scalar primitives consistent rather than special-casing a by-value `Config`.
#![allow(clippy::trivially_copy_pass_by_ref)]

use crate::{
    Config, Error, STANDARD_ALPHABET, STANDARD_DECODE_TABLE, URL_SAFE_ALPHABET,
    URL_SAFE_DECODE_TABLE,
};

/// Maps a 12-bit value to the two Base64 characters it encodes, packed
/// little-endian so the first character lands in the low byte.
const fn encode_pair_table(alphabet: &[u8; 64]) -> [u16; 4096] {
    let mut table = [0u16; 4096];
    let mut i = 0;
    while i < 4096 {
        table[i] = (alphabet[i >> 6] as u16) | ((alphabet[i & 0x3F] as u16) << 8);
        i += 1;
    }
    table
}

static STANDARD_ENCODE_PAIRS: [u16; 4096] = encode_pair_table(STANDARD_ALPHABET);
static URL_SAFE_ENCODE_PAIRS: [u16; 4096] = encode_pair_table(URL_SAFE_ALPHABET);

/// Reverse lookup with the 6-bit index pre-shifted into its position within a
/// 24-bit group. Invalid characters map to `u32::MAX`, so OR-ing a whole group
/// together pushes the result above `0x00FF_FFFF` if any character was bad.
const fn decode_shift_table(alphabet: &[u8; 64], shift: u32) -> [u32; 256] {
    let mut table = [u32::MAX; 256];
    let mut i: u32 = 0;
    while i < 64 {
        table[alphabet[i as usize] as usize] = i << shift;
        i += 1;
    }
    table
}

/// The four position tables as one array, indexed by a character's position
/// within its 4-character group. Keeping them contiguous matters: as four
/// separate statics, selecting the alphabet costs four `cmov`s that LLVM hoists
/// into the function entry even when the fast loop never runs, which is pure
/// overhead for inputs of a few characters. As one array it is a single `cmov`
/// plus constant offsets.
const fn decode_shift_tables(alphabet: &[u8; 64]) -> [[u32; 256]; 4] {
    [
        decode_shift_table(alphabet, 18),
        decode_shift_table(alphabet, 12),
        decode_shift_table(alphabet, 6),
        decode_shift_table(alphabet, 0),
    ]
}

static STANDARD_DECODE_SHIFTED: [[u32; 256]; 4] = decode_shift_tables(STANDARD_ALPHABET);
static URL_SAFE_DECODE_SHIFTED: [[u32; 256]; 4] = decode_shift_tables(URL_SAFE_ALPHABET);

/// Largest value a valid 4-character group can OR to (24 significant bits).
const GROUP_MAX: u32 = 0x00FF_FFFF;

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
    // Select the table based on configuration. This branch predicts perfectly
    // since config doesn't change during the loop. The tail below reads its
    // characters out of this same table so that this stays the *only* selection
    // in the function; a second one is hoisted into the entry block by LLVM and
    // measurably slows down one- and two-byte inputs, which do no other work.
    let pairs: &[u16; 4096] = if config.url_safe {
        &URL_SAFE_ENCODE_PAIRS
    } else {
        &STANDARD_ENCODE_PAIRS
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
        // `first_chunk`/`last_chunk` (bytes 0..4 and 2..6 of a 6-byte chunk) are
        // what let LLVM emit two 32-bit loads here; indexing the chunk
        // byte-by-byte instead compiles to a `movzbl`-and-shift pile. Neither
        // can be `None`, and both fold away once the chunk length is known.
        let reg_a = u32::from_be_bytes(chunk.first_chunk::<4>().copied().unwrap_or_default());
        let reg_b = u32::from_be_bytes(chunk.last_chunk::<4>().copied().unwrap_or_default());

        let n1 = (reg_a >> 8) as usize; // Bytes 0, 1, 2
        let n2 = (reg_b & 0x00_FF_FF_FF) as usize; // Bytes 3, 4, 5

        // Two 12-bit halves per group, two characters per lookup. Emitting two
        // 32-bit stores rather than assembling one u64 keeps the OR chain short.
        let lo = u32::from(pairs[n1 >> 12]) | (u32::from(pairs[n1 & 0xFFF]) << 16);
        let hi = u32::from(pairs[n2 >> 12]) | (u32::from(pairs[n2 & 0xFFF]) << 16);

        out[0..4].copy_from_slice(&lo.to_le_bytes());
        out[4..8].copy_from_slice(&hi.to_le_bytes());
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

        let packed = u32::from(pairs[n >> 12]) | (u32::from(pairs[n & 0xFFF]) << 16);

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

        // The first 2 characters are always present, and are exactly the pair
        // that the top 12 bits of `n` encode.
        let first_two = pairs[n >> 12].to_le_bytes();
        out_tail[oi] = first_two[0];
        out_tail[oi + 1] = first_two[1];

        // Handle the 3rd and 4th characters (data vs padding).
        if rem == 2 {
            // The character for the 6-bit index `(n >> 6) & 0x3F`. A pair index
            // of `index << 6` places that index in the pair's *first* slot,
            // so the low byte of the entry is the character wanted here.
            out_tail[oi + 2] = pairs[n & 0xFC0].to_le_bytes()[0];
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
    // It is only needed by the tail; the fast loop uses the pre-shifted tables.
    let table = if config.url_safe {
        &URL_SAFE_DECODE_TABLE
    } else {
        &STANDARD_DECODE_TABLE
    };
    // Fast loop bounds: process 8 input bytes -> 6 output bytes per iteration,
    // reserving the last 4 input bytes so the tail can handle padding carefully.
    let len_safe = len.saturating_sub(4);
    let len_fast = len_safe - (len_safe % 8);
    // `len_fast <= len - 4`, so this is always within `estimate_decoded_len`.
    let out_fast = len_fast / 8 * 6;

    let shifted: &[[u32; 256]; 4] = if config.url_safe {
        &URL_SAFE_DECODE_SHIFTED
    } else {
        &STANDARD_DECODE_SHIFTED
    };

    // --- FAST LOOP (Middle Chunks) ---
    // Slicing both sides up front and pairing them with `chunks_exact` hoists
    // every bounds check out of the loop; indexing `input[i + n]` and
    // `dst[o..o + 6]` per iteration leaves two compares and two branches behind
    // instead.
    for (chars, out) in input[..len_fast]
        .chunks_exact(8)
        .zip(dst[..out_fast].chunks_exact_mut(6))
    {
        // Each lookup already carries its position shift, so a group is just
        // four loads OR-ed together. Invalid characters contribute `u32::MAX`,
        // lifting the result above the 24 bits a valid group can occupy.
        let n1 = shifted[0][usize::from(chars[0])]
            | shifted[1][usize::from(chars[1])]
            | shifted[2][usize::from(chars[2])]
            | shifted[3][usize::from(chars[3])];
        let n2 = shifted[0][usize::from(chars[4])]
            | shifted[1][usize::from(chars[5])]
            | shifted[2][usize::from(chars[6])]
            | shifted[3][usize::from(chars[7])];

        if (n1 | n2) > GROUP_MAX {
            return Err(Error::InvalidCharacter);
        }

        // Both groups land in the top 48 bits, so one byte-swap emits all 6
        // output bytes in order.
        let packed = ((u64::from(n1) << 40) | (u64::from(n2) << 16)).to_be_bytes();
        out.copy_from_slice(&packed[..6]);
    }

    decode_tail(config, table, input, len_fast, dst, out_fast)
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
