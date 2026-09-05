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
//! That costs ~12 KiB of `.rodata` per alphabet and roughly doubles both
//! kernels. The narrow 256-entry decode table is still used by the decode tail,
//! where a handful of bytes cannot amortize a wide table's cache footprint.
//! All four tables live in the [`Alphabet`](crate::Alphabet) the `Config`
//! points at, so a custom alphabet runs this kernel at the same speed as the
//! built-ins.

#![forbid(unsafe_code)]
#![allow(clippy::trivially_copy_pass_by_ref)]

use crate::{Config, Error};

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
    // One load, no selection: the alphabet's tables hang off the `Config`, so
    // there is no per-call branch to hoist and the tail below reads its
    // characters out of this same table.
    let pairs: &[u16; 4096] = config.alphabet.pairs();

    let len = input.len();
    let blocks = len / 6; // full 6-byte input blocks

    // Split input/output into the fast-loop region and the tail. Using
    // `as_chunks` over the split halves keeps the hot loop free of bounds
    // checks (the chunk lengths are statically known: 6 in, 8 out).
    let (in_main, in_tail) = input.split_at(blocks * 6);
    let (out_main, out_tail) = dst.split_at_mut(blocks * 8);

    // --- MAIN LOOP ---
    // Two blocks per iteration once the input is long enough to pay for the
    // wider setup, and the original one-block loop below that.
    //
    // The wide form lives out of line on purpose. Inlining it here grows
    // `encode_slice` enough to push `Engine::encode` past LLVM's inlining
    // threshold, turning the whole allocating API into an out-of-line call --
    // worth -8% on an 8- or 12-byte encode, which never reaches the wide loop at
    // all. Keeping it behind a call leaves the short path exactly as small as it
    // was, and a call amortised over 8+ blocks costs the long path nothing.
    if blocks >= 8 {
        encode_main_wide(pairs, in_main, out_main);
    } else {
        for (chunk, out) in in_main
            .as_chunks::<6>()
            .0
            .iter()
            .zip(out_main.as_chunks_mut::<8>().0.iter_mut())
        {
            let reg_a = u32::from_be_bytes(chunk.first_chunk::<4>().copied().unwrap_or_default());
            let reg_b = u32::from_be_bytes(chunk.last_chunk::<4>().copied().unwrap_or_default());
            let n1 = (reg_a >> 8) as usize;
            let n2 = (reg_b & 0x00_FF_FF_FF) as usize;
            let lo = u32::from(pairs[n1 >> 12]) | (u32::from(pairs[n1 & 0xFFF]) << 16);
            let hi = u32::from(pairs[n2 >> 12]) | (u32::from(pairs[n2 & 0xFFF]) << 16);
            out[0..4].copy_from_slice(&lo.to_le_bytes());
            out[4..8].copy_from_slice(&hi.to_le_bytes());
        }
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

/// The encoder's two-blocks-per-iteration main loop, for inputs with at least
/// eight whole blocks in them.
///
/// `in_main` is a whole number of 6-byte blocks and `out_main` the matching
/// 8-byte outputs; both are consumed entirely. Out of line so that
/// [`encode_slice`] stays small enough for `Engine::encode` to keep inlining it
/// -- see the comment at the call site. Measured on Zen 5 at +6% for 384 bytes
/// rising to +10% from 4 KiB up, against the one-block loop.
#[inline(never)]
fn encode_main_wide(pairs: &[u16; 4096], in_main: &[u8], out_main: &mut [u8]) {
    let pairs_of_blocks = in_main.len() / 12;
    let (in_wide, in_rest) = in_main.split_at(pairs_of_blocks * 12);
    let (out_wide, out_rest) = out_main.split_at_mut(pairs_of_blocks * 16);

    for (chunk, out) in in_wide
        .as_chunks::<12>()
        .0
        .iter()
        .zip(out_wide.as_chunks_mut::<16>().0.iter_mut())
    {
        // Two overlapping big-endian `u32`s per block, as in the single-block
        // form: `first_chunk`/`last_chunk` are what let LLVM emit 32-bit loads
        // instead of a `movzbl`-and-shift pile.
        let a0 = u32::from_be_bytes(chunk.first_chunk::<4>().copied().unwrap_or_default());
        let b0 = u32::from_be_bytes(chunk[2..6].first_chunk::<4>().copied().unwrap_or_default());
        let a1 = u32::from_be_bytes(chunk[6..10].first_chunk::<4>().copied().unwrap_or_default());
        let b1 = u32::from_be_bytes(chunk[8..12].first_chunk::<4>().copied().unwrap_or_default());

        let n1 = (a0 >> 8) as usize;
        let n2 = (b0 & 0x00_FF_FF_FF) as usize;
        let n3 = (a1 >> 8) as usize;
        let n4 = (b1 & 0x00_FF_FF_FF) as usize;

        let w0 = u32::from(pairs[n1 >> 12]) | (u32::from(pairs[n1 & 0xFFF]) << 16);
        let w1 = u32::from(pairs[n2 >> 12]) | (u32::from(pairs[n2 & 0xFFF]) << 16);
        let w2 = u32::from(pairs[n3 >> 12]) | (u32::from(pairs[n3 & 0xFFF]) << 16);
        let w3 = u32::from(pairs[n4 >> 12]) | (u32::from(pairs[n4 & 0xFFF]) << 16);

        out[0..4].copy_from_slice(&w0.to_le_bytes());
        out[4..8].copy_from_slice(&w1.to_le_bytes());
        out[8..12].copy_from_slice(&w2.to_le_bytes());
        out[12..16].copy_from_slice(&w3.to_le_bytes());
    }

    // At most one 6-byte block is left over.
    if let (Some(chunk), Some(out)) = (in_rest.first_chunk::<6>(), out_rest.first_chunk_mut::<8>())
    {
        let reg_a = u32::from_be_bytes(chunk.first_chunk::<4>().copied().unwrap_or_default());
        let reg_b = u32::from_be_bytes(chunk.last_chunk::<4>().copied().unwrap_or_default());
        let n1 = (reg_a >> 8) as usize;
        let n2 = (reg_b & 0x00_FF_FF_FF) as usize;
        let lo = u32::from(pairs[n1 >> 12]) | (u32::from(pairs[n1 & 0xFFF]) << 16);
        let hi = u32::from(pairs[n2 >> 12]) | (u32::from(pairs[n2 & 0xFFF]) << 16);
        out[0..4].copy_from_slice(&lo.to_le_bytes());
        out[4..8].copy_from_slice(&hi.to_le_bytes());
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
    decode_slice_impl(config, input, dst, false)
}

/// Decodes input while accepting both padded and unpadded final groups.
#[inline]
pub(crate) fn decode_slice_forgiving(
    config: &Config,
    input: &[u8],
    dst: &mut [u8],
) -> Result<usize, Error> {
    decode_slice_impl(config, input, dst, true)
}

#[inline]
fn decode_slice_impl(
    config: &Config,
    input: &[u8],
    dst: &mut [u8],
    padding_indifferent: bool,
) -> Result<usize, Error> {
    let len = input.len();
    if len == 0 {
        return Ok(0);
    }

    // The table maps valid characters to 0..=63 and invalid characters to 0xFF.
    // It is only needed by the tail; the fast loop uses the pre-shifted tables.
    let table = config.alphabet.decode_table();
    // Fast loop bounds: process 8 input bytes -> 6 output bytes per iteration,
    // reserving the last 4 input bytes so the tail can handle padding carefully.
    let len_safe = len.saturating_sub(4);
    let len_fast = len_safe - (len_safe % 8);
    // `len_fast <= len - 4`, so this is always within `decoded_len_estimate`.
    let out_fast = len_fast / 8 * 6;

    let shifted: &[[u32; 256]; 4] = config.alphabet.shifted();

    // --- FAST LOOP (Middle Chunks) ---
    // Slicing both sides up front and pairing them with `as_chunks` hoists
    // every bounds check out of the loop; indexing `input[i + n]` and
    // `dst[o..o + 6]` per iteration leaves two compares and two branches behind
    // instead.
    for (chars, out) in input[..len_fast]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(dst[..out_fast].as_chunks_mut::<6>().0.iter_mut())
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

    decode_tail(
        config,
        table,
        input,
        len_fast,
        dst,
        out_fast,
        padding_indifferent,
    )
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
    padding_indifferent: bool,
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
                if (!config.padding && !padding_indifferent) || i + 4 != len {
                    return Err(Error::InvalidLength);
                }

                if b2 == b'=' {
                    // "XX==" -> 1 byte output
                    if (d0 | d1) & 0xC0 != 0 || d1 & 0x0F != 0 {
                        return Err(Error::InvalidCharacter);
                    }
                    let n = (u32::from(d0) << 18) | (u32::from(d1) << 12);
                    dst[o] = ((n >> 16) & 0xFF) as u8;
                    o += 1;
                } else {
                    // "XXX=" -> 2 bytes output
                    let d2 = table[usize::from(b2)];
                    if (d0 | d1 | d2) & 0xC0 != 0 || d2 & 0x03 != 0 {
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
            if config.padding && !padding_indifferent {
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
                if d1 & 0x0F != 0 {
                    return Err(Error::InvalidCharacter);
                }
                dst[o] = ((n >> 16) & 0xFF) as u8;
                o += 1;
            } else {
                // "XYZ" -> 2 bytes output
                let d2 = table[usize::from(input[i + 2])];
                if d2 & 0xC0 != 0 || d2 & 0x03 != 0 {
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
