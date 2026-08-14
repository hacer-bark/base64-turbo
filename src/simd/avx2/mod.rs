use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};
use crate::{Config, Error};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m256i, _mm_storeu_si128, _mm256_add_epi8, _mm256_and_si256, _mm256_castsi256_si128,
    _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mulhi_epu16, _mm256_mullo_epi16,
    _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set_epi8, _mm256_set1_epi8,
    _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_sub_epi8, _mm256_subs_epu8, _mm256_testz_si256,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m256i, _mm_storeu_si128, _mm256_add_epi8, _mm256_and_si256, _mm256_castsi256_si128,
    _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_extracti128_si256, _mm256_loadu_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mulhi_epu16, _mm256_mullo_epi16,
    _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set_epi8, _mm256_set1_epi8,
    _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_sub_epi8, _mm256_subs_epu8, _mm256_testz_si256,
};

/// Encodes 32 raw input bytes (only the low 24, byte-shifted by 4, are
/// logically consumed) into 32 Base64 characters.
///
/// Credit: the reshuffle bit-extraction and single-LUT character mapping are
/// Alfred Klomp's (`aklomp/base64`, BSD); see the README. The URL-safe
/// `translate_lut` (only the `+`/`/` vs `-`/`_` deltas differ) was re-derived
/// for this crate and checked against all 64 indices (see the length sweep).
#[target_feature(enable = "avx2")]
unsafe fn encode_vec_avx2(input: __m256i, translate_lut: __m256i) -> __m256i {
    // Reshuffle so each 32-bit lane holds one 3-byte group, then pull the four
    // 6-bit indices per lane via two masked multiplies (mulhi/mullo) rather
    // than per-group shifts.
    let reshuffle = _mm256_set_epi8(
        10, 11, 9, 10, 7, 8, 6, 7, 4, 5, 3, 4, 1, 2, 0, 1, 14, 15, 13, 14, 11, 12, 10, 11, 8, 9, 7,
        8, 5, 6, 4, 5,
    );
    let shuffled = _mm256_shuffle_epi8(input, reshuffle);
    let t0 = _mm256_and_si256(shuffled, _mm256_set1_epi32(0x0FC0_FC00));
    let t1 = _mm256_mulhi_epu16(t0, _mm256_set1_epi32(0x0400_0040));
    let t2 = _mm256_and_si256(shuffled, _mm256_set1_epi32(0x003F_03F0));
    let t3 = _mm256_mullo_epi16(t2, _mm256_set1_epi32(0x0100_0010));
    let indices = _mm256_or_si256(t1, t3);

    // Map each 6-bit index to its character with one LUT lookup: `subs_epu8`
    // by 51 offsets into the last 3 ranges (digits, `+`/`-`, `/`/`_`), and
    // `cmpgt_epi8(idx, 25)` bumps it past the uppercase range.
    let set_51 = _mm256_set1_epi8(51);
    let set_25 = _mm256_set1_epi8(25);
    let lut_idx = _mm256_sub_epi8(
        _mm256_subs_epu8(indices, set_51),
        _mm256_cmpgt_epi8(indices, set_25),
    );
    _mm256_add_epi8(indices, _mm256_shuffle_epi8(translate_lut, lut_idx))
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode_slice_avx2(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let translate_lut = if config.url_safe {
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

    // Each round consumes 24 input bytes, writes 32, but loads a 32-byte vector
    // shifted 4 ahead of what it consumes — the shift lets the reshuffle skip a
    // per-iteration cross-lane permute. Round 1 can't read before `input`, so it
    // loads at offset 0 and applies a one-time permute; later rounds get the
    // shift free by advancing `src` only 20 (the trailing `+= 4` repays it).
    // `rounds = (len - 4) / 24` keeps every load in bounds.
    if len >= 32 {
        let rounds = (len - 4) / 24;

        let first = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let first = _mm256_permutevar8x32_epi32(first, _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6));
        let out0 = unsafe { encode_vec_avx2(first, translate_lut) };
        unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), out0) };
        src = unsafe { src.add(20) };
        dst = unsafe { dst.add(32) };

        let mut remaining = rounds - 1;

        while remaining >= 4 {
            let v0 = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
            let v1 = unsafe { _mm256_loadu_si256(src.add(24).cast::<__m256i>()) };
            let v2 = unsafe { _mm256_loadu_si256(src.add(48).cast::<__m256i>()) };
            let v3 = unsafe { _mm256_loadu_si256(src.add(72).cast::<__m256i>()) };

            let o0 = unsafe { encode_vec_avx2(v0, translate_lut) };
            let o1 = unsafe { encode_vec_avx2(v1, translate_lut) };
            let o2 = unsafe { encode_vec_avx2(v2, translate_lut) };
            let o3 = unsafe { encode_vec_avx2(v3, translate_lut) };

            unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), o0) };
            unsafe { _mm256_storeu_si256(dst.add(32).cast::<__m256i>(), o1) };
            unsafe { _mm256_storeu_si256(dst.add(64).cast::<__m256i>(), o2) };
            unsafe { _mm256_storeu_si256(dst.add(96).cast::<__m256i>(), o3) };

            src = unsafe { src.add(96) };
            dst = unsafe { dst.add(128) };
            remaining -= 4;
        }

        while remaining > 0 {
            let v = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
            let out = unsafe { encode_vec_avx2(v, translate_lut) };
            unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), out) };

            src = unsafe { src.add(24) };
            dst = unsafe { dst.add(32) };
            remaining -= 1;
        }

        // Undo the first round's 20-vs-24 pointer-advancement deficit.
        src = unsafe { src.add(4) };
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
    let aligned_len_128 = safe_len - (safe_len % 128);
    let aligned_len_32 = safe_len - (safe_len % 32);
    let src_end_128 = unsafe { src.add(aligned_len_128) };
    let src_end_32 = unsafe { src.add(aligned_len_32) };

    // Quad tier: 128 input bytes -> 96 output.
    while src < src_end_128 {
        let v0 = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let v1 = unsafe { _mm256_loadu_si256(src.add(32).cast::<__m256i>()) };
        let v2 = unsafe { _mm256_loadu_si256(src.add(64).cast::<__m256i>()) };
        let v3 = unsafe { _mm256_loadu_si256(src.add(96).cast::<__m256i>()) };

        let (i0, e0) = decode_vec!(v0);
        let (i1, e1) = decode_vec!(v1);
        let (i2, e2) = decode_vec!(v2);
        let (i3, e3) = decode_vec!(v3);

        let err_any = _mm256_or_si256(_mm256_or_si256(e0, e1), _mm256_or_si256(e2, e3));
        if _mm256_testz_si256(err_any, err_any) != 1 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(i0, dst);
        pack_and_store!(i1, dst.add(24));
        pack_and_store!(i2, dst.add(48));
        pack_and_store!(i3, dst.add(72));

        src = unsafe { src.add(128) };
        dst = unsafe { dst.add(96) };
    }

    // Single tier: 32 input bytes -> 24 output.
    while src < src_end_32 {
        let v = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let (idx, err) = decode_vec!(v);

        if _mm256_testz_si256(err, err) != 1 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(idx, dst);

        src = unsafe { src.add(32) };
        dst = unsafe { dst.add(24) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

// Verification: Kani proofs, intrinsic models, model/hardware equivalence,
// and the Miri + hardware coverage suites.
#[cfg(any(kani, test))]
mod verify;
