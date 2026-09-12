//! AVX2 backend.
//!
//! Both kernels map characters arithmetically from the RFC 4648 layout rather
//! than by table lookup, so neither can serve a custom [`Alphabet`](crate::Alphabet);
//! the dispatcher sends those to the scalar kernel instead. What the arithmetic
//! buys is a character map with no memory operand in it at all, which is why
//! this path stays ahead of a shuffle-based one on every part measured.
//!
//! Encode runs a 24-byte window per round, decode a 32-character vector, and
//! each has a wide tier that unrolls the steady state; the constants below
//! record what each unroll factor was measured at and why it is the one chosen.

use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};
use crate::{Config, Error};
use core::hint::black_box;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, __m256i, _mm_sfence, _mm_storeu_si128, _mm_stream_si128, _mm256_add_epi8,
    _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_mullo_epi16, _mm256_or_si256, _mm256_permutevar8x32_epi32,
    _mm256_set_epi8, _mm256_set1_epi8, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256,
    _mm256_sub_epi8, _mm256_subs_epu8, _mm256_testz_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_sfence, _mm_storeu_si128, _mm_stream_si128, _mm256_add_epi8,
    _mm256_and_si256, _mm256_andnot_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_mullo_epi16, _mm256_or_si256, _mm256_permutevar8x32_epi32,
    _mm256_set_epi8, _mm256_set1_epi8, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256,
    _mm256_sub_epi8, _mm256_subs_epu8, _mm256_testz_si256,
};

// Verification: Kani proofs, intrinsic models, model/hardware equivalence,
// and the Miri + hardware coverage suites.
#[cfg(any(kani, test))]
mod verify;

/// Floor on the input length from which the encoder switches to non-temporal
/// stores.
///
/// Above it the input plus its 4/3-sized output no longer fit in a typical
/// last-level cache, so the ordinary stores spend a third of the memory
/// bandwidth on read-for-ownership traffic for lines that are then overwritten
/// whole. Below it the output usually *is* reused from cache and bypassing it
/// costs more than the RFO traffic saves; measured on a 9 MiB-L3 Coffee Lake,
/// the crossover sits between 2 and 4 MiB.
///
/// That 4 MiB happens to land near 5/8 of the L3 it was tuned against, which is
/// where [`super::NONTEMPORAL_LLC_RATIO`] independently puts the crossover — so
/// it is right on that box and arbitrary on any other. It stays here as the
/// *floor* rather than the whole threshold: the hardware coverage test in
/// `verify` enters the tier at exactly this length, and the README's note that
/// the path needs a 4 MiB input to execute is stated against it. Scaling can
/// only raise the gate above it.
const NT_STORE_MIN_LEN: usize = 4 << 20;

/// Input length from which the encoder switches to non-temporal stores: 5/8 of
/// the last-level cache, never below [`NT_STORE_MIN_LEN`].
///
/// Shared with the AVX-512 kernel, which met the same hazard first — a flat
/// threshold streams into a destination that would have stayed resident on any
/// machine whose cache is larger than the one it was tuned on. See
/// [`super::nontemporal_min`].
#[inline]
fn nt_store_min() -> usize {
    super::nontemporal_min(NT_STORE_MIN_LEN)
}

/// Rounds per iteration of the encoder's wide tier.
const ENC_UNROLL: usize = 8;
/// Steady-state rounds from which the out-of-line wide tier is worth calling.
///
/// Two unrolled iterations. One is not enough to amortize the call and the
/// spill of the constants the callee borrows: entering at a single iteration
/// costs ~9% at 256 B against leaving those rounds to the single-round tier.
const ENC_WIDE_MIN_ROUNDS: usize = 2 * ENC_UNROLL;
/// Vectors per iteration of the decoder's wide tier.
///
/// Four, not eight. A decoded vector plus the eight live constants is nine
/// registers; eight of them in flight at once needs seventeen, and the loop
/// spilled the lookup tables to the stack and reloaded them for every vector.
/// Racing 1/2/3/4/6/8 on Coffee Lake, the curve is flat from three upwards, so
/// this takes the smallest unroll that reaches the plateau.
const DEC_UNROLL: usize = 4;

// Stride constants. The Kani index proofs in `verify` reason over this same
// arithmetic symbolically, and import these rather than restating them, so a
// stride that changes here changes the proofs too instead of silently drifting
// out from under them.

/// Logical input bytes a steady-state encode round consumes.
const ENC_ROUND_IN: usize = 24;
/// Characters an encode round writes.
const ENC_ROUND_OUT: usize = 32;
/// Bytes each encode load reads: a full vector, of which only the middle
/// [`ENC_ROUND_IN`] (offset by [`ENC_LEAD`]) are consumed.
const ENC_VEC: usize = 32;
/// Input bytes sitting to the left of a steady-state round's 24-byte window.
const ENC_LEAD: usize = 4;
/// `src` advance after the permuted first round, which manufactures its own
/// lead instead of reading one.
const ENC_FIRST_ADVANCE: usize = ENC_ROUND_IN - ENC_LEAD;

/// Input characters a single-vector decode pass consumes, which is also exactly
/// what each of its loads reads.
const DEC_BLOCK_IN: usize = 32;
/// Bytes a single-vector decode pass advances `dst` by.
const DEC_BLOCK_OUT: usize = 24;
/// Trailing margin: no single-vector pass may start unless at least this many
/// characters remain after it. `pack_and_store!` overhangs `dst` by 4 bytes past
/// [`DEC_BLOCK_OUT`] (its second lane lands at [`DEC_PACK_LANE_OFF`] + 16), so the
/// smallest possible tail — exactly 4 leftover characters — would let that overhang
/// write past the destination's estimated capacity. Requiring one more byte of
/// margin than that keeps a real tail always wide enough to absorb it.
const DEC_LEAD: usize = 5;
/// Offset of `pack_and_store!`'s second 16-byte lane, which is what makes its
/// written span wider than the 24 bytes it advances.
const DEC_PACK_LANE_OFF: usize = 12;

/// Precomputed AVX2 encode constants, factored out of [`encode_slice_avx2`] so
/// they are materialized once per call rather than once per round.
///
/// Credit: the reshuffle bit-extraction and single-LUT character mapping are
/// Alfred Klomp's (`aklomp/base64`, BSD); see the README. The URL-safe
/// `translate` LUT (only the `+`/`/` vs `-`/`_` deltas differ) was re-derived
/// for this crate and checked against all 64 indices (see the length sweep).
struct EncodeConstantsAvx2 {
    reshuffle: __m256i,
    align_mul: __m256i,
    field_mask: __m256i,
    field_mul: __m256i,
    translate: __m256i,
    c51: __m256i,
    c25: __m256i,
}

#[target_feature(enable = "avx2")]
fn encode_constants_avx2<const URL: bool>() -> EncodeConstantsAvx2 {
    let translate = if URL {
        _mm256_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -17, 32, 0, 0, 65, 71, -4, -4, -4, -4,
            -4, -4, -4, -4, -4, -4, -17, 32, 0, 0,
        )
    } else {
        _mm256_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0, 65, 71, -4, -4, -4, -4,
            -4, -4, -4, -4, -4, -4, -19, -16, 0, 0,
        )
    };

    EncodeConstantsAvx2 {
        reshuffle: _mm256_set_epi8(
            10, 11, 9, 10, 7, 8, 6, 7, 4, 5, 3, 4, 1, 2, 0, 1, 14, 15, 13, 14, 11, 12, 10, 11, 8,
            9, 7, 8, 5, 6, 4, 5,
        ),
        // Both multipliers are per-16-bit-lane powers of two, which LLVM will
        // happily strength-reduce back into shift-and-blend sequences that cost
        // two to six extra uops apiece and land on the already-saturated shuffle
        // port. `black_box` keeps them opaque so a single `vpmullw` survives; it
        // runs once per call, outside the loop. Racing the two forms on Coffee
        // Lake, dropping the `black_box` costs 23% at 512 B and 43% at 64 KiB.
        align_mul: black_box(_mm256_set1_epi32(0x0010_0001)),
        field_mask: _mm256_set1_epi32(0x003F_03F0),
        field_mul: black_box(_mm256_set1_epi32(0x0100_0010)),
        translate,
        c51: _mm256_set1_epi8(51),
        c25: _mm256_set1_epi8(25),
    }
}

/// Encodes 32 raw input bytes (only the middle 24, byte-shifted by 4, are
/// logically consumed) into 32 Base64 characters.
///
/// The two multiplies split the four 6-bit fields of each 3-byte group into
/// their own bytes. `align_mul` scales the odd 16-bit halfword by 16 so that a
/// single `>> 10` lands both of that dword's "high" fields at bit 0 of their
/// byte; `field_mul` shifts the two "low" fields up by 4 and 8 into bits 8..13.
/// The results occupy disjoint bits, so one `or` merges them.
#[target_feature(enable = "avx2")]
fn encode_vec_avx2(input: __m256i, k: &EncodeConstantsAvx2) -> __m256i {
    let shuffled = _mm256_shuffle_epi8(input, k.reshuffle);
    let aligned = _mm256_srli_epi16(_mm256_mullo_epi16(shuffled, k.align_mul), 10);
    let fields = _mm256_mullo_epi16(_mm256_and_si256(shuffled, k.field_mask), k.field_mul);
    let indices = _mm256_or_si256(aligned, fields);

    let lut_idx = _mm256_sub_epi8(
        _mm256_subs_epu8(indices, k.c51),
        _mm256_cmpgt_epi8(indices, k.c25),
    );
    _mm256_add_epi8(indices, _mm256_shuffle_epi8(k.translate, lut_idx))
}

/// # Safety
/// `dst` must be valid for a 32-byte write, and 16-byte aligned when `NT`.
#[target_feature(enable = "avx2")]
unsafe fn store_chars_avx2<const NT: bool>(dst: *mut u8, chars: __m256i) {
    if NT {
        let half = dst.cast::<__m128i>();
        unsafe {
            _mm_stream_si128(half, _mm256_castsi256_si128(chars));
            _mm_stream_si128(half.add(1), _mm256_extracti128_si256(chars, 1));
        }
    } else {
        unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), chars) };
    }
}

/// The wide tier: `rounds` steady-state encode rounds, `rounds` a nonzero
/// multiple of [`ENC_UNROLL`]. Each round reads the 32 bytes at `src`, consumes
/// the middle 24 (`src[4..28]`), and writes 32 characters.
///
/// Deliberately out of line, and it rebuilds the constants rather than taking
/// them by reference. Inlined into [`encode_slice_avx2`] the register allocator
/// ran out of ymm registers: it sank the eight loads down among the arithmetic
/// instead of issuing them up front, and re-materialized three constants from
/// `.rodata` inside the loop. That cost 15% at 64 KiB and 17% at 4 KiB against
/// this same loop compiled on its own. Out of line the schedule comes back --
/// seven constants hoisted, eight loads up front, no stack traffic -- and the
/// call is paid only by inputs with at least [`ENC_UNROLL`] steady-state rounds
/// to amortize it over.
///
/// # Safety
/// For every `i < rounds`, `src.add(24 * i)` must be valid for a 32-byte read
/// and `dst.add(32 * i)` for a 32-byte write; when `NT`, `dst` must also be
/// 16-byte aligned.
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn encode_wide_avx2<const NT: bool, const URL: bool>(
    src: *const u8,
    dst: *mut u8,
    rounds: usize,
    k: &EncodeConstantsAvx2,
) {
    let mut src = src;
    let mut dst = dst;
    let mut remaining = rounds;

    while remaining >= ENC_UNROLL {
        // Loads first, stores second: the eight independent chains keep the
        // multiply latency covered without the scheduler having to reorder
        // across a store.
        let mut chunk = [_mm256_setzero_si256(); ENC_UNROLL];
        for (i, slot) in chunk.iter_mut().enumerate() {
            *slot = unsafe { _mm256_loadu_si256(src.add(ENC_ROUND_IN * i).cast::<__m256i>()) };
        }
        for (i, raw) in chunk.into_iter().enumerate() {
            let chars = encode_vec_avx2(raw, k);
            unsafe { store_chars_avx2::<NT>(dst.add(ENC_ROUND_OUT * i), chars) };
        }

        src = unsafe { src.add(ENC_ROUND_IN * ENC_UNROLL) };
        dst = unsafe { dst.add(ENC_ROUND_OUT * ENC_UNROLL) };
        remaining -= ENC_UNROLL;
    }

    if NT {
        // Non-temporal stores are not ordered against the caller's later loads.
        _mm_sfence();
    }
}

#[target_feature(enable = "avx2")]
unsafe fn encode_impl_avx2<const URL: bool>(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let k = encode_constants_avx2::<URL>();

    if len >= ENC_VEC {
        let rounds = (len - ENC_LEAD) / ENC_ROUND_IN;

        // First round: the steady-state rounds read `src[4..28]`, so the very
        // first one has no four bytes to its left. Permuting the load down by
        // one dword manufactures them, at the cost of advancing `src` by only
        // 20; the trailing `src.add(4)` below repays that.
        let first = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let first = _mm256_permutevar8x32_epi32(first, _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6));
        let out0 = encode_vec_avx2(first, &k);
        unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), out0) };
        src = unsafe { src.add(ENC_FIRST_ADVANCE) };
        dst = unsafe { dst.add(ENC_ROUND_OUT) };

        let mut remaining = rounds - 1;

        // Wide tier, out of line. The call has to be amortized over enough
        // iterations to pay for itself, so this needs ENC_WIDE_MIN_ROUNDS
        // rounds, not merely one iteration's worth.
        let wide = remaining - (remaining % ENC_UNROLL);
        if remaining >= ENC_WIDE_MIN_ROUNDS {
            // Every store sits at `dst_start + 32 * n`, so one alignment test up
            // front covers the whole loop.
            if len >= nt_store_min() && dst_start.align_offset(16) == 0 {
                unsafe { encode_wide_avx2::<true, URL>(src, dst, wide, &k) };
            } else {
                unsafe { encode_wide_avx2::<false, URL>(src, dst, wide, &k) };
            }
            src = unsafe { src.add(ENC_ROUND_IN * wide) };
            dst = unsafe { dst.add(ENC_ROUND_OUT * wide) };
            remaining -= wide;
        }

        // Single-round tier: fewer than ENC_UNROLL rounds left, inline.
        while remaining > 0 {
            let raw = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
            let chars = encode_vec_avx2(raw, &k);
            unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), chars) };

            src = unsafe { src.add(ENC_ROUND_IN) };
            dst = unsafe { dst.add(ENC_ROUND_OUT) };
            remaining -= 1;
        }

        // Undo the first round's 20-vs-24 pointer-advancement deficit.
        src = unsafe { src.add(ENC_LEAD) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::encode(config, input, src, dst_slice, dst_off) };
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode_slice_avx2(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    if config.alphabet.is_url_safe() {
        unsafe { encode_impl_avx2::<true>(config, input, dst_slice) };
    } else {
        unsafe { encode_impl_avx2::<false>(config, input, dst_slice) };
    }
}

/// Precomputed AVX2 decode constants, factored out of [`decode_slice_avx2`]
/// only to keep its body under clippy's line-count threshold.
///
/// The nibble-lookup validation/decode is Wojciech Muła's (with `@aqrit`'s
/// `/`-vs-`+` trick), as in `aklomp/base64` and `lemire/fastbase64` (BSD); see
/// the README. That algorithm covers only the standard alphabet; the URL-safe
/// `lut_lo`/`lut_hi`/`lut_roll` were re-derived here and verified against all
/// 256 byte values (see `avx2_lut_url_safe_matches_scalar`).
struct DecodeConstantsAvx2 {
    lut_lo: __m256i,
    lut_hi: __m256i,
    lut_roll: __m256i,
    eq_char: __m256i,
    eq_shift: __m256i,
    pack_l1: __m256i,
    pack_l2: __m256i,
    pack_shuffle: __m256i,
    mask_nibble: __m256i,
}

/// Stores a low-nibble validation row complemented, and broadcast to both lanes.
///
/// The invariant is unchanged from the uncomplemented form: a byte is invalid
/// iff `lut_lo[byte & 0xF] & lut_hi[byte >> 4] != 0`. Storing `lut_lo` negated
/// turns that `and` into `andnot(lo, hi)`, which computes the same bits.
///
/// The decoder combines this row with `vpandn`, which is what lets the same
/// lookup be indexed by the raw byte instead of by its masked low nibble: a byte
/// `>= 0x80` shuffles to zero, and `andnot(0, hi)` is `hi`, so every guard bit
/// its high nibble sets survives and the byte is rejected. Indexing by the raw
/// byte is what removes the per-vector mask.
const fn lut_lo_complement(row: [u8; 16]) -> [i8; 32] {
    let mut out = [0i8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = (!row[i % 16]).cast_signed();
        i += 1;
    }
    out
}

/// Guard bits per high nibble: 2=`+`/`/`(0x01), 3=digits(0x02),
/// 4/6=`A`-`O`/`a`-`o`(0x04), 5/7=`P`-`Z`/`p`-`z`(0x08).
const DEC_LUT_LO_STD: [i8; 32] = lut_lo_complement([
    0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A,
]);

/// Guard bits per high nibble: 2=`-`(0x01), 3=digits(0x02),
/// 4/6=`A`-`O`/`a`-`o`(0x04), 5=`P`-`Z`+`_`(0x08), 7=`p`-`z`(0x20).
/// Row 5 breaks symmetry with row 7 (the `_`), so both need own bits.
const DEC_LUT_LO_URL: [i8; 32] = lut_lo_complement([
    0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x3B, 0x3B, 0x3A, 0x3B, 0x33,
]);

#[target_feature(enable = "avx2")]
unsafe fn decode_constants_avx2<const URL: bool>() -> DecodeConstantsAvx2 {
    // Bit 0x10 is a catch-all in every `lut_lo`, paired with `lut_hi = 0x10` on
    // rows with no valid chars (0, 1, 8..=15). Rows 2..=7 each get a guard bit
    // that `lut_lo` clears only for that row's valid low nibbles.
    let (lut_lo, lut_hi, lut_roll, eq_char, eq_shift) = if URL {
        let lut_hi = _mm256_setr_epi8(
            0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x20, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
            0x10, 0x10, 0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x20, 0x10, 0x10, 0x10, 0x10,
            0x10, 0x10, 0x10, 0x10,
        );
        // Delta from ASCII to 6-bit value. Row 5 is ambiguous (`P`..`Z` need
        // -65, `_` needs -32), so `_` is pushed to slot 5+8=13 for its delta.
        let lut_roll = _mm256_setr_epi8(
            0, 0, 17, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, -32, 0, 0, 0, 0, 17, 4, -65, -65, -71,
            -71, 0, 0, 0, 0, 0, -32, 0, 0,
        );
        (DEC_LUT_LO_URL, lut_hi, lut_roll, b'_', 8i8)
    } else {
        let lut_hi = _mm256_setr_epi8(
            0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
            0x10, 0x10, 0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08, 0x10, 0x10, 0x10, 0x10,
            0x10, 0x10, 0x10, 0x10,
        );
        // Row 2 is ambiguous (`+` needs +19, `/` needs +16), so `/` is pulled
        // to slot 1 for its delta.
        let lut_roll = _mm256_setr_epi8(
            0, 16, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 19, 4, -65, -65, -71,
            -71, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        (DEC_LUT_LO_STD, lut_hi, lut_roll, b'/', -1i8)
    };

    let eq_char = _mm256_set1_epi8(eq_char.cast_signed());
    let eq_shift = _mm256_set1_epi8(eq_shift);

    DecodeConstantsAvx2 {
        lut_lo: unsafe { _mm256_loadu_si256(lut_lo.as_ptr().cast::<__m256i>()) },
        lut_hi,
        lut_roll,
        eq_char,
        eq_shift,
        // Packing constants.
        pack_l1: unsafe { _mm256_loadu_si256(PACK_L1.as_ptr().cast::<__m256i>()) },
        pack_l2: unsafe { _mm256_loadu_si256(PACK_L2.as_ptr().cast::<__m256i>()) },
        pack_shuffle: unsafe { _mm256_loadu_si256(PACK_SHUFFLE.as_ptr().cast::<__m256i>()) },
        // Mask for high-nibble extraction.
        mask_nibble: _mm256_set1_epi8(0x0F),
    }
}

/// Decodes `input` with the alphabet fixed at compile time.
///
/// `URL` is a const parameter rather than a `Config` read because the standard
/// alphabet's `eq_shift` is `-1`, and `_mm256_cmpeq_epi8` already yields `0` or
/// `-1`: with the shift known, masking the comparison against it is provably a
/// no-op and disappears, taking a `vpand` out of every vector. The URL-safe
/// alphabet shifts by 8 and still needs it.
#[target_feature(enable = "avx2")]
unsafe fn decode_impl_avx2<const URL: bool>(
    config: &Config,
    input: &[u8],
    dst_slice: &mut [u8],
) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let DecodeConstantsAvx2 {
        lut_lo,
        lut_hi,
        lut_roll,
        eq_char,
        eq_shift,
        pack_l1,
        pack_l2,
        pack_shuffle,
        mask_nibble,
    } = unsafe { decode_constants_avx2::<URL>() };

    // Validate + decode one vector (nibble lookup, roll-based; see the struct
    // doc above for credit).
    macro_rules! decode_vec {
        ($input:expr) => {{
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16($input, 4), mask_nibble);

            // `lut_lo` is complemented and indexed by the raw byte; see
            // `lut_lo_complement` for why that is sound and what it saves.
            let lo = _mm256_shuffle_epi8(lut_lo, $input);
            let hi = _mm256_shuffle_epi8(lut_hi, hi_nibbles);
            let err = _mm256_andnot_si256(lo, hi);

            let eq = _mm256_cmpeq_epi8($input, eq_char);
            let shift = if URL {
                _mm256_and_si256(eq, eq_shift)
            } else {
                eq
            };
            let roll_idx = _mm256_add_epi8(hi_nibbles, shift);
            let roll = _mm256_shuffle_epi8(lut_roll, roll_idx);
            let indices = _mm256_add_epi8($input, roll);

            (indices, err)
        }};
    }

    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            let m = _mm256_maddubs_epi16($indices, pack_l1);
            let p = _mm256_madd_epi16(m, pack_l2);
            let out = _mm256_shuffle_epi8(p, pack_shuffle);

            let lane_0 = _mm256_castsi256_si128(out);
            unsafe { _mm_storeu_si128($dst_ptr.cast::<__m128i>(), lane_0) };
            let lane_1 = _mm256_extracti128_si256(out, 1);
            unsafe { _mm_storeu_si128($dst_ptr.add(DEC_PACK_LANE_OFF).cast::<__m128i>(), lane_1) };
        }};
    }

    // Every load reads a full 32-byte vector per 32 bytes consumed, so no pass
    // may start within 4 bytes of the end; each tier rounds `safe_len` down to
    // its own block size.
    let safe_len = len.saturating_sub(DEC_LEAD);
    let block_wide = DEC_BLOCK_IN * DEC_UNROLL;
    let aligned_len_wide = safe_len - (safe_len % block_wide);
    let aligned_len_32 = safe_len - (safe_len % DEC_BLOCK_IN);
    let src_end_wide = unsafe { src.add(aligned_len_wide) };
    let src_end_32 = unsafe { src.add(aligned_len_32) };

    // Invalid characters are folded into one accumulator and reported after the
    // loops rather than per block. Bailing out mid-loop would force every
    // vector's inputs to stay live across a branch, which costs more registers
    // than this machine has; the caller sees the same `Err` either way, and the
    // bytes written before it are already unspecified on the error path.
    let mut err_acc = _mm256_setzero_si256();

    // Wide tier: 128 input bytes -> 96 output. Each vector is carried all the
    // way to its store before the next one is loaded: holding all `DEC_UNROLL`
    // of them live at once costs more registers than the machine has, and the
    // out-of-order window overlaps the independent chains anyway.
    while src < src_end_wide {
        for i in 0..DEC_UNROLL {
            let raw = unsafe { _mm256_loadu_si256(src.add(DEC_BLOCK_IN * i).cast::<__m256i>()) };
            let (indices, err) = decode_vec!(raw);
            err_acc = _mm256_or_si256(err_acc, err);
            let out = unsafe { dst.add(DEC_BLOCK_OUT * i) };
            pack_and_store!(indices, out);
        }

        src = unsafe { src.add(DEC_BLOCK_IN * DEC_UNROLL) };
        dst = unsafe { dst.add(DEC_BLOCK_OUT * DEC_UNROLL) };
    }

    // Single tier: 32 input bytes -> 24 output.
    while src < src_end_32 {
        let raw = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let (indices, err) = decode_vec!(raw);
        err_acc = _mm256_or_si256(err_acc, err);

        pack_and_store!(indices, dst);

        src = unsafe { src.add(DEC_BLOCK_IN) };
        dst = unsafe { dst.add(DEC_BLOCK_OUT) };
    }

    if _mm256_testz_si256(err_acc, err_acc) != 1 {
        return Err(Error::InvalidCharacter);
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn decode_slice_avx2(
    config: &Config,
    input: &[u8],
    dst_slice: &mut [u8],
) -> Result<usize, Error> {
    if config.alphabet.is_url_safe() {
        unsafe { decode_impl_avx2::<true>(config, input, dst_slice) }
    } else {
        unsafe { decode_impl_avx2::<false>(config, input, dst_slice) }
    }
}
