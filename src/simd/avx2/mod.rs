use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};
use crate::{Config, Error};
use core::hint::black_box;

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m256i, _mm_sfence, _mm_storeu_si128, _mm_stream_si128, _mm256_add_epi8,
    _mm256_and_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_mullo_epi16, _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set_epi8,
    _mm256_set1_epi8, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_subs_epu8,
    _mm256_testz_si256,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm_sfence, _mm_storeu_si128, _mm_stream_si128, _mm256_add_epi8,
    _mm256_and_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8,
    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_mullo_epi16, _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set_epi8,
    _mm256_set1_epi8, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_subs_epu8,
    _mm256_testz_si256,
};

/// Input length from which the encoder switches to non-temporal stores.
///
/// Above it the input plus its 4/3-sized output no longer fit in a typical
/// last-level cache, so the ordinary stores spend a third of the memory
/// bandwidth on read-for-ownership traffic for lines that are then overwritten
/// whole. Below it the output usually *is* reused from cache and bypassing it
/// costs more than the RFO traffic saves; measured on a 9 MiB-L3 Coffee Lake,
/// the crossover sits between 2 and 4 MiB.
const NT_STORE_MIN_LEN: usize = 4 << 20;

/// Rounds per iteration of the encoder's wide tier.
const ENC_UNROLL: usize = 8;
/// Vectors per iteration of the decoder's wide tier.
const DEC_UNROLL: usize = 8;

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
fn encode_constants_avx2(config: Config) -> EncodeConstantsAvx2 {
    let translate = if config.url_safe {
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
        // runs once per call, outside the loop.
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

/// Runs `rounds` steady-state encode rounds: each reads the 32 bytes at `src`,
/// consumes the middle 24 (`src[4..28]`), and writes 32 characters.
///
/// # Safety
/// For every `i < rounds`, `src.add(24 * i)` must be valid for a 32-byte read
/// and `dst.add(32 * i)` for a 32-byte write; when `NT`, `dst` must also be
/// 16-byte aligned.
#[target_feature(enable = "avx2")]
unsafe fn encode_rounds_avx2<const NT: bool>(
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
            *slot = unsafe { _mm256_loadu_si256(src.add(24 * i).cast::<__m256i>()) };
        }
        for (i, raw) in chunk.into_iter().enumerate() {
            let chars = encode_vec_avx2(raw, k);
            unsafe { store_chars_avx2::<NT>(dst.add(32 * i), chars) };
        }

        src = unsafe { src.add(24 * ENC_UNROLL) };
        dst = unsafe { dst.add(32 * ENC_UNROLL) };
        remaining -= ENC_UNROLL;
    }

    while remaining > 0 {
        let raw = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let chars = encode_vec_avx2(raw, k);
        unsafe { store_chars_avx2::<NT>(dst, chars) };

        src = unsafe { src.add(24) };
        dst = unsafe { dst.add(32) };
        remaining -= 1;
    }

    if NT {
        // Non-temporal stores are not ordered against the caller's later loads.
        _mm_sfence();
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode_slice_avx2(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let k = encode_constants_avx2(*config);

    if len >= 32 {
        let rounds = (len - 4) / 24;

        // First round: the steady-state rounds read `src[4..28]`, so the very
        // first one has no four bytes to its left. Permuting the load down by
        // one dword manufactures them, at the cost of advancing `src` by only
        // 20; the trailing `src.add(4)` below repays that.
        let first = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let first = _mm256_permutevar8x32_epi32(first, _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6));
        let out0 = encode_vec_avx2(first, &k);
        unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), out0) };
        src = unsafe { src.add(20) };
        dst = unsafe { dst.add(32) };

        let remaining = rounds - 1;

        // Every store sits at `dst_start + 32 * n`, so one alignment test up
        // front covers the whole loop.
        if len >= NT_STORE_MIN_LEN && dst_start.align_offset(16) == 0 {
            unsafe { encode_rounds_avx2::<true>(src, dst, remaining, &k) };
        } else {
            unsafe { encode_rounds_avx2::<false>(src, dst, remaining, &k) };
        }

        // Undo the first round's 20-vs-24 pointer-advancement deficit.
        src = unsafe { src.add(24 * remaining + 4) };
        dst = unsafe { dst.add(32 * remaining) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::encode(config, input, src, dst_slice, dst_off) };
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

#[target_feature(enable = "avx2")]
unsafe fn decode_constants_avx2(config: &Config) -> DecodeConstantsAvx2 {
    // Nibble bitmask LUTs: a byte is invalid iff
    // `lut_lo[byte & 0xF] & lut_hi[byte >> 4] != 0`. Bit 0x10 is a catch-all in
    // every `lut_lo`, paired with `lut_hi = 0x10` on rows with no valid chars
    // (0, 1, 8..=15). Rows 2..=7 each get a guard bit that `lut_lo` clears only
    // for that row's valid low nibbles.
    let (lut_lo, lut_hi, lut_roll, eq_char, eq_shift) = if config.url_safe {
        // Guard bits per high nibble: 2=`-`(0x01), 3=digits(0x02),
        // 4/6=`A`-`O`/`a`-`o`(0x04), 5=`P`-`Z`+`_`(0x08), 7=`p`-`z`(0x20).
        // Row 5 breaks symmetry with row 7 (the `_`), so both need own bits.
        let lut_lo = _mm256_setr_epi8(
            0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x3B, 0x3B, 0x3A,
            0x3B, 0x33, 0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x3B,
            0x3B, 0x3A, 0x3B, 0x33,
        );
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
        (lut_lo, lut_hi, lut_roll, b'_', 8i8)
    } else {
        // Guard bits per high nibble: 2=`+`/`/`(0x01), 3=digits(0x02),
        // 4/6=`A`-`O`/`a`-`o`(0x04), 5/7=`P`-`Z`/`p`-`z`(0x08).
        let lut_lo = _mm256_setr_epi8(
            0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B,
            0x1B, 0x1A, 0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x1A,
            0x1B, 0x1B, 0x1B, 0x1A,
        );
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
        (lut_lo, lut_hi, lut_roll, b'/', -1i8)
    };

    let eq_char = _mm256_set1_epi8(eq_char.cast_signed());
    let eq_shift = _mm256_set1_epi8(eq_shift);

    // Packing Constants
    let pack_l1 = unsafe { _mm256_loadu_si256(PACK_L1.as_ptr().cast::<__m256i>()) };
    let pack_l2 = unsafe { _mm256_loadu_si256(PACK_L2.as_ptr().cast::<__m256i>()) };
    let pack_shuffle = unsafe { _mm256_loadu_si256(PACK_SHUFFLE.as_ptr().cast::<__m256i>()) };

    // Mask for nibble extraction (both low and high nibbles).
    let mask_nibble = _mm256_set1_epi8(0x0F);

    DecodeConstantsAvx2 {
        lut_lo,
        lut_hi,
        lut_roll,
        eq_char,
        eq_shift,
        pack_l1,
        pack_l2,
        pack_shuffle,
        mask_nibble,
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn decode_slice_avx2(
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
    } = unsafe { decode_constants_avx2(config) };

    // Validate + decode one vector (nibble lookup, roll-based; see the struct
    // doc above for credit).
    macro_rules! decode_vec {
        ($input:expr) => {{
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16($input, 4), mask_nibble);
            let lo_nibbles = _mm256_and_si256($input, mask_nibble);

            let lo = _mm256_shuffle_epi8(lut_lo, lo_nibbles);
            let hi = _mm256_shuffle_epi8(lut_hi, hi_nibbles);
            let err = _mm256_and_si256(lo, hi);

            let eq = _mm256_cmpeq_epi8($input, eq_char);
            let roll_idx = _mm256_add_epi8(hi_nibbles, _mm256_and_si256(eq, eq_shift));
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
            unsafe { _mm_storeu_si128($dst_ptr.add(12).cast::<__m128i>(), lane_1) };
        }};
    }

    // Every load reads a full 32-byte vector per 32 bytes consumed, so no pass
    // may start within 4 bytes of the end; each tier rounds `safe_len` down to
    // its own block size.
    let safe_len = len.saturating_sub(4);
    let block_wide = 32 * DEC_UNROLL;
    let aligned_len_wide = safe_len - (safe_len % block_wide);
    let aligned_len_32 = safe_len - (safe_len % 32);
    let src_end_wide = unsafe { src.add(aligned_len_wide) };
    let src_end_32 = unsafe { src.add(aligned_len_32) };

    // Invalid characters are folded into one accumulator and reported after the
    // loops rather than per block. Bailing out mid-loop would force every
    // vector's inputs to stay live across a branch, which costs more registers
    // than this machine has; the caller sees the same `Err` either way, and the
    // bytes written before it are already unspecified on the error path.
    let mut err_acc = _mm256_setzero_si256();

    // Wide tier: 256 input bytes -> 192 output.
    while src < src_end_wide {
        let mut decoded = [_mm256_setzero_si256(); DEC_UNROLL];
        for (i, slot) in decoded.iter_mut().enumerate() {
            let raw = unsafe { _mm256_loadu_si256(src.add(32 * i).cast::<__m256i>()) };
            let (indices, err) = decode_vec!(raw);
            *slot = indices;
            err_acc = _mm256_or_si256(err_acc, err);
        }
        for (i, indices) in decoded.into_iter().enumerate() {
            let out = unsafe { dst.add(24 * i) };
            pack_and_store!(indices, out);
        }

        src = unsafe { src.add(32 * DEC_UNROLL) };
        dst = unsafe { dst.add(24 * DEC_UNROLL) };
    }

    // Single tier: 32 input bytes -> 24 output.
    while src < src_end_32 {
        let raw = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let (indices, err) = decode_vec!(raw);
        err_acc = _mm256_or_si256(err_acc, err);

        pack_and_store!(indices, dst);

        src = unsafe { src.add(32) };
        dst = unsafe { dst.add(24) };
    }

    if _mm256_testz_si256(err_acc, err_acc) != 1 {
        return Err(Error::InvalidCharacter);
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

// Verification: Kani proofs, intrinsic models, model/hardware equivalence,
// and the Miri + hardware coverage suites.
#[cfg(any(kani, test))]
mod verify;
