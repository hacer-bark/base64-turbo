use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};
use crate::{Config, Error, scalar};

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

/// Encodes 32 raw input bytes (of which only the low 24 bytes, byte-shifted
/// by 4, are logically consumed) into 32 Base64 characters.
///
/// Credit: the reshuffle bit-extraction (`shuffle_epi8` + `and`/`mulhi`/`and`/
/// `mullo`/`or`) and the single-LUT character-mapping technique below are
/// Alfred Klomp's (`aklomp/base64`, BSD-licensed); see the README for full
/// credit. The URL-safe `translate_lut` variant was re-derived for this crate
/// (only the `+`/`/` vs `-`/`_` delta entries differ) and cross-checked
/// exhaustively against all 64 alphabet indices in a standalone Python
/// script; see `avx2_encode_url_safe_all_lengths_0_to_400` below.
#[target_feature(enable = "avx2")]
unsafe fn encode_vec_avx2(input: __m256i, translate_lut: __m256i) -> __m256i {
    // Reshuffle the (4-byte-shifted) input so each 32-bit lane holds one
    // 3-byte Base64 group, then extract the four 6-bit indices per lane via
    // two masked multiplies (mulhi for the high half, mullo for the low
    // half of each 16-bit sub-lane) instead of per-group shifts.
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

    // Map each 6-bit index (0..=63) to its Base64 character via one LUT
    // lookup: `subs_epu8(idx, 51)` gives a 0-based offset into the last 3
    // ranges (digits, `+`/`-`, `/`/`_`), and `cmpgt_epi8(idx, 25)` bumps that
    // offset by 1 for indices past the uppercase-letter range so it also
    // reaches the digit/lowercase-letter table slots correctly.
    let set_51 = _mm256_set1_epi8(51);
    let set_25 = _mm256_set1_epi8(25);
    let lut_idx = _mm256_sub_epi8(
        _mm256_subs_epu8(indices, set_51),
        _mm256_cmpgt_epi8(indices, set_25),
    );
    _mm256_add_epi8(indices, _mm256_shuffle_epi8(translate_lut, lut_idx))
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn encode_slice_avx2(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

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

    // Each round consumes 24 logical input bytes and produces 32 output
    // bytes, but every round's *load* reads a full 32-byte vector that is
    // shifted 4 bytes "into the future" relative to what it logically
    // consumes (this is what lets `encode_vec_avx2`'s reshuffle avoid a
    // cross-lane permute on every iteration). The very first round can't
    // read 4 bytes before the start of `input`, so it instead reads at
    // offset 0 and reproduces the same shift with a one-time lane permute;
    // every subsequent round gets the shift for free because the previous
    // round only advanced the read pointer by 20 bytes (not 24), so the
    // next 32-byte load naturally starts 4 bytes "early". The final `+= 4`
    // below undoes that bookkeeping deficit once the loop is done.
    //
    // Safety margin: round `k` (1-indexed) reads `[24*(k-1) - 4, 24*(k-1) + 28)`
    // relative to `input`'s start (round 1 reads `[0, 32)` instead, via the
    // permute). Requiring `rounds <= (len - 4) / 24` (i.e. `rounds` computed
    // via truncating integer division) keeps every round's read within
    // `input`, including the last one.
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

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

/// Precomputed AVX2 vector constants shared by every lane processed in
/// [`decode_slice_avx2`]. Factored out purely to keep that function's body
/// under clippy's line-count threshold; the values themselves are unchanged.
///
/// The validation/decode strategy is the nibble-lookup algorithm originated by
/// Wojciech Muła (with the `/`-vs-`+` disambiguation trick credited to `@aqrit`),
/// as implemented in Alfred Klomp's `aklomp/base64` and Daniel Lemire's
/// `lemire/fastbase64`. Both are BSD/permissive-licensed; see the README for
/// full credit. That published algorithm only covers the standard alphabet
/// (`+`/`/`); the `lut_lo`/`lut_hi`/`lut_roll` values for the URL-safe alphabet
/// (`-`/`_`) below were re-derived from scratch for this crate, following the
/// same construction technique, and are verified exhaustively against all 256
/// byte values (see the Kani proof and the `avx2_lut_url_safe_matches_scalar`
/// test below).
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
    // Nibble-indexed bitmask LUTs: a byte is invalid iff
    // `lut_lo[byte & 0xF] & lut_hi[byte >> 4] != 0`. Bit 4 (0x10) is a
    // catch-all set in every `lut_lo` entry, paired with `lut_hi = 0x10` for
    // every high-nibble row that contains no valid Base64 characters at all
    // (rows 0, 1, 8..=15). Each high-nibble row that *does* contain valid
    // characters (rows 2..=7) gets its own guard bit in `lut_hi`; `lut_lo`
    // clears that guard bit only for the low-nibble values valid in that row.
    let (lut_lo, lut_hi, lut_roll, eq_char, eq_shift) = if config.url_safe {
        // Row 2 (0x2_): only `-` (0x2D, lo=13) is valid  -> guard bit 0x01.
        // Row 3 (0x3_): digits `0`-`9` (lo=0..=9), same as standard -> 0x02.
        // Row 4 (0x4_): `A`-`O` (lo=1..=15) -> 0x04, shared with row 6.
        // Row 5 (0x5_): `P`-`Z` (lo=0..=10) *and* `_` (0x5F, lo=15) -> 0x08.
        //   Unlike the standard alphabet, row 5 is no longer symmetric with
        //   row 7 (only `_` breaks the pattern), so it needs its own bit.
        // Row 6 (0x6_): `a`-`o` (lo=1..=15) -> 0x04, shared with row 4.
        // Row 7 (0x7_): `p`-`z` (lo=0..=10) -> 0x20 (its own bit, see row 5).
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
        // Delta to add to the raw ASCII byte to get its 6-bit value. Row 5 is
        // ambiguous (`P`..`Z` need -65, but `_` needs -32), so when the byte
        // equals `_` exactly, the lookup index is pushed from 5 to 5+8=13 (an
        // otherwise-dead slot) to select the alternate delta.
        let lut_roll = _mm256_setr_epi8(
            0, 0, 17, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, -32, 0, 0, 0, 0, 17, 4, -65, -65, -71,
            -71, 0, 0, 0, 0, 0, -32, 0, 0,
        );
        (lut_lo, lut_hi, lut_roll, b'_', 8i8)
    } else {
        // Row 2 (0x2_): `+` (0x2B, lo=11) and `/` (0x2F, lo=15) -> 0x01.
        // Row 3 (0x3_): digits `0`-`9` (lo=0..=9) -> 0x02.
        // Row 4 (0x4_): `A`-`O` (lo=1..=15) -> 0x04, shared with row 6.
        // Row 5 (0x5_): `P`-`Z` (lo=0..=10) -> 0x08, shared with row 7.
        // Row 6 (0x6_): `a`-`o` (lo=1..=15) -> 0x04, shared with row 4.
        // Row 7 (0x7_): `p`-`z` (lo=0..=10) -> 0x08, shared with row 5.
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
        // Row 2 is ambiguous (`+` needs +19, `/` needs +16), so when the byte
        // equals `/` exactly, the lookup index is pulled from 2 down to 1 to
        // select the alternate delta.
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
    mut dst: *mut u8,
) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst;

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

    // Decode & Validate Single Vector.
    //
    // Credit: nibble-lookup validation + roll-based decode, originated by
    // Wojciech Muła / `@aqrit`, as implemented in `aklomp/base64` and
    // `lemire/fastbase64` (both BSD-licensed) — see the struct doc above.
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

    // Process 128 bytes (4 chunks) at a time
    let safe_len_128 = len.saturating_sub(4);
    let aligned_len_128 = safe_len_128 - (safe_len_128 % 128);
    let src_end_128 = unsafe { src.add(aligned_len_128) };

    while src < src_end_128 {
        // Load 4 vectors
        let v0 = unsafe { _mm256_loadu_si256(src.cast::<__m256i>()) };
        let v1 = unsafe { _mm256_loadu_si256(src.add(32).cast::<__m256i>()) };
        let v2 = unsafe { _mm256_loadu_si256(src.add(64).cast::<__m256i>()) };
        let v3 = unsafe { _mm256_loadu_si256(src.add(96).cast::<__m256i>()) };

        // Process
        let (i0, e0) = decode_vec!(v0);
        let (i1, e1) = decode_vec!(v1);
        let (i2, e2) = decode_vec!(v2);
        let (i3, e3) = decode_vec!(v3);

        // Check Errors
        let err_any = _mm256_or_si256(_mm256_or_si256(e0, e1), _mm256_or_si256(e2, e3));

        if _mm256_testz_si256(err_any, err_any) != 1 {
            return Err(Error::InvalidCharacter);
        }

        // Store 4 chunks
        pack_and_store!(i0, dst);
        pack_and_store!(i1, dst.add(24));
        pack_and_store!(i2, dst.add(48));
        pack_and_store!(i3, dst.add(72));

        src = unsafe { src.add(128) };
        dst = unsafe { dst.add(96) };
    }

    // Process remaining 32-byte chunks
    let safe_len_32 = len.saturating_sub(4);
    let aligned_len_32 = safe_len_32 - (safe_len_32 % 32);
    let src_end_32 = unsafe { input.as_ptr().add(aligned_len_32) };

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

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < len {
        dst = unsafe {
            dst.add(scalar::decode_slice_unsafe(
                config,
                &input[processed_len..],
                dst,
            )?)
        };
    }

    Ok(unsafe { dst.offset_from(dst_start) }.cast_unsigned())
}

#[cfg(kani)]
mod kani_verification_avx2 {
    use super::*;
    use crate::{Config, STANDARD as TURBO_STANDARD, STANDARD_NO_PAD as TURBO_STANDARD_NO_PAD};
    use std::mem::transmute;

    // --- CONSTANTS ---

    // Encoder Induction Size: 28 (1 AVX2 Loop) + 1 (Scalar Transition)
    const ENC_INDUCTION_LEN: usize = 29;

    // Decoder Induction Size: 36 (1 AVX2 Loop) + 1 (Scalar Transition)
    const DEC_INDUCTION_LEN: usize = 37;

    // --- HELPERS ---

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding {
            TURBO_STANDARD.encoded_len(len)
        } else {
            TURBO_STANDARD_NO_PAD.encoded_len(len)
        }
    }

    // --- STUBS ---

    // STUB: _mm256_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_epi8
    #[allow(dead_code)]
    unsafe fn _mm256_shuffle_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [u8; 32] = unsafe { transmute(b) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*8
            // (In Rust we access bytes 'j' so '*8' offset is not needed)
            let i = j;

            // IF b[i+7] == 1
            if (b[i] & 0x80) != 0 {
                // dst[i+7:i] := 0
                dst[i] = 0;
            } else {
                // index[3:0] := b[i+3:i]
                let index = b[i] & 0x0F;
                // dst[i+7:i] := a[index*8+7:index*8]
                dst[i] = a[index as usize];
            }
            // FI

            // IF b[128+i+7] == 1
            if (b[16 + i] & 0x80) != 0 {
                // dst[128+i+7:128+i] := 0
                dst[16 + i] = 0;
            } else {
                // index[3:0] := b[128+i+3:128+i]
                let index = b[16 + i] & 0x0F;
                // dst[128+i+7:128+i] := a[128+index*8+7:128+index*8]
                dst[16 + i] = a[(16 + index) as usize];
            }
            // FI
        }
        // ENDFOR

        // dst[MAX:256] := 0
        // (__m256i is exactly 256 bits. There are no bits beyond 256 to zero out)

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_subs_epu8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_subs_epu8
    #[allow(dead_code)]
    unsafe fn _mm256_subs_epu8_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [u8; 32] = unsafe { transmute(b) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := j*8
            let i = j;

            // dst[i+7:i] := SaturateU8(a[i+7:i] - b[i+7:i])
            dst[i] = a[i].saturating_sub(b[i]);
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_testz_si256
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_testz_si256
    // Note: in this logic added complexity as Rust do not support 256 bits values.
    #[allow(dead_code)]
    unsafe fn _mm256_testz_si256_stub(a: __m256i, b: __m256i) -> i32 {
        let a: [u64; 4] = unsafe { transmute(a) };
        let b: [u64; 4] = unsafe { transmute(b) };
        let zf: i32;
        let _cf: i32;

        // Perform 256 bit AND
        let res_and = [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]];

        // IF ((a[255:0] AND b[255:0]) == 0)
        if res_and[0] == 0 && res_and[1] == 0 && res_and[2] == 0 && res_and[3] == 0 {
            // ZF := 1
            zf = 1;
        } else {
            // ZF := 0
            zf = 0;
        }
        // FI

        // Perform 256 bit (NOT a) AND b
        let res_not_and = [
            (!a[0]) & b[0],
            (!a[1]) & b[1],
            (!a[2]) & b[2],
            (!a[3]) & b[3],
        ];

        // IF (((NOT a[255:0]) AND b[255:0]) == 0)
        if res_not_and[0] == 0 && res_not_and[1] == 0 && res_not_and[2] == 0 && res_not_and[3] == 0
        {
            // CF := 1
            _cf = 1;
        } else {
            // CF := 0
            _cf = 0;
        }
        // FI

        // RETURN ZF
        return zf;
    }

    // STUB: _mm256_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maddubs_epi16
    #[allow(dead_code)]
    unsafe fn _mm256_maddubs_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [i8; 32] = unsafe { transmute(b) };
        let mut dst = [0i16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j * 2;

            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i + 1] as i16) * (b[i + 1] as i16))
                .saturating_add((a[i] as i16) * (b[i] as i16));
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_madd_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_madd_epi16
    #[allow(dead_code)]
    unsafe fn _mm256_madd_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [i16; 16] = unsafe { transmute(a) };
        let b: [i16; 16] = unsafe { transmute(b) };
        let mut dst = [0i32; 8];

        // FOR j := 0 to 7
        for j in 0..8 {
            // i := j*32
            let i = j * 2;

            // dst[i+31:i] := SignExtend32(a[i+31:i+16]*b[i+31:i+16]) + SignExtend32(a[i+15:i]*b[i+15:i])
            dst[j] = (a[i + 1] as i32)
                .wrapping_mul(b[i + 1] as i32)
                .wrapping_add((a[i] as i32).wrapping_mul(b[i] as i32));
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // --- PROOFS ---

    /// **Proof 1: Roundtrip Correctness (The Logic Check)**
    ///
    /// Verifies that `Decode(Encode(X)) == X`.
    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, _mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, _mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, _mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, _mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, _mm256_madd_epi16_stub)]
    fn check_avx2_roundtrip_correctness() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; ENC_INDUCTION_LEN] = kani::any();

        // Buffers
        let mut enc_buf = [0u8; 128];
        let mut dec_buf = [0u8; 128];

        unsafe {
            // 1. Encode
            encode_slice_avx2(&config, &input, enc_buf.as_mut_ptr());

            // Calculate actual encoded length for slicing
            let enc_len = encoded_size(ENC_INDUCTION_LEN, config.padding);
            let encoded_slice = &enc_buf[..enc_len];

            // 2. Decode
            // This MUST succeed for valid encoded output
            let dec_len = decode_slice_avx2(&config, encoded_slice, dec_buf.as_mut_ptr())
                .expect("Valid encoding failed to decode");

            // 3. Verify
            assert_eq!(dec_len, ENC_INDUCTION_LEN);
            assert_eq!(&dec_buf[..dec_len], &input, "Roundtrip mismatch");
        }
    }

    /// **Proof 2: Decoder Robustness & Induction**
    ///
    /// Verifies that `decode_slice_avx2`:
    /// 1. Accepts ANY 33 bytes of garbage input.
    /// 2. Never Segfaults, Panics, or causes UB.
    /// 3. Safely handles the SIMD->Scalar pointer transition.
    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, _mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, _mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, _mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, _mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, _mm256_madd_epi16_stub)]
    fn check_avx2_decode_robustness() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };

        // Input: 33 bytes of unrestricted symbolic data (garbage)
        let input: [u8; DEC_INDUCTION_LEN] = kani::any();

        // Output Buffer: Max estimated size
        let mut output = [0u8; 128];

        unsafe {
            // We ignore the Result. We only care that this function call
            // returns safely (Ok or Err) and does not crash.
            let _ = decode_slice_avx2(&config, &input, output.as_mut_ptr());
        }
    }
}

#[cfg(all(test, miri))]
mod miri_avx2_coverage {
    use super::*;
    use base64::{
        Engine,
        engine::general_purpose::{STANDARD, URL_SAFE},
    };
    use rand::{RngExt, rng};

    // --- Mock Infrastructure ---
    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    /// Helper to verify AVX2 encoding against the 'base64' crate oracle
    fn verify_encode_avx2(config: &Config, oracle: &impl Engine, input_len: usize) {
        let input = random_bytes(input_len);
        let expected = oracle.encode(&input);

        // Allocate buffer (Base64 is ~4/3 larger)
        let mut dst = vec![0u8; expected.len() * 2]; // Safety margin

        unsafe {
            encode_slice_avx2(config, &input, dst.as_mut_ptr());
        }

        // Verify prefix matches expected
        let result = &dst[..expected.len()];
        assert_eq!(
            std::str::from_utf8(result).unwrap(),
            expected,
            "Encode len {}",
            input_len
        );
    }

    /// Helper to verify AVX2 decoding against the 'base64' crate oracle
    fn verify_decode_avx2(config: &Config, oracle: &impl Engine, original_len: usize) {
        // 1. Generate valid Base64 via oracle
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();

        // 2. Run AVX2 Decoder
        let mut dst = vec![0u8; original_len + 64]; // Safety margin

        let len = unsafe {
            decode_slice_avx2(config, encoded_bytes, dst.as_mut_ptr())
                .expect("Valid input failed to decode")
        };

        // 3. Verify
        assert_eq!(&dst[..len], &input_bytes, "Decode len {}", original_len);
    }

    // ----------------------------------------------------------------------
    // 1. Encoder Coverage Tests
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx2_encode_scalar_fallback() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Test < 24 bytes (Hits scalar fallback immediately)
        verify_encode_avx2(&config, &STANDARD, 1);
        verify_encode_avx2(&config, &STANDARD, 23);
    }

    #[test]
    fn miri_avx2_encode_single_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Your code uses 24-byte chunks (32-byte registers reading 24 bytes).
        // Test exactly 24 (1 loop)
        verify_encode_avx2(&config, &STANDARD, 24);
        // Test 48 (2 loops - proves src.add(24) works)
        verify_encode_avx2(&config, &STANDARD, 48);
        // Test 25 (1 loop + 1 byte scalar fallback)
        verify_encode_avx2(&config, &STANDARD, 25);
    }

    #[test]
    fn miri_encode_quad_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Your code uses 96-byte blocks (4 * 24).
        // Test exactly 96 (1 quad loop)
        verify_encode_avx2(&config, &STANDARD, 96);
        // Test 192 (2 quad loops - proves src.add(96) works)
        verify_encode_avx2(&config, &STANDARD, 192);
        // Test 97 (1 quad loop + 0 single + 1 scalar)
        verify_encode_avx2(&config, &STANDARD, 97);
        // Test 120 (1 quad loop + 1 single loop)
        verify_encode_avx2(&config, &STANDARD, 120);
    }

    #[test]
    fn miri_avx2_encode_url_safe() {
        // Verify the lookup table switching logic
        let config = Config {
            url_safe: true,
            padding: true,
        };
        verify_encode_avx2(&config, &URL_SAFE, 50);
    }

    // ----------------------------------------------------------------------
    // 2. Decoder Coverage Tests
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx2_decode_scalar_fallback() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Your code falls back for < 32 bytes
        // Note: Base64 expands 3 bytes -> 4 chars.
        // Input length 4 chars -> 3 bytes output.
        verify_decode_avx2(&config, &STANDARD, 3); // 4 chars
        verify_decode_avx2(&config, &STANDARD, 21); // 28 chars (< 32)
    }

    #[test]
    fn miri_avx2_decode_single_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Your code processes 32-byte chunks.
        // 32 bytes of Base64 = 24 bytes of decoded data.
        verify_decode_avx2(&config, &STANDARD, 24); // Exactly 32 bytes input
        verify_decode_avx2(&config, &STANDARD, 48); // Exactly 64 bytes input (2 loops)
        verify_decode_avx2(&config, &STANDARD, 25); // 32 bytes + scalar remainder
    }

    #[test]
    fn miri_avx2_decode_quad_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Your code processes 128-byte chunks (4 * 32).
        // 128 bytes input = 96 bytes decoded.
        verify_decode_avx2(&config, &STANDARD, 96); // Exactly 128 bytes input
        verify_decode_avx2(&config, &STANDARD, 192); // Exactly 256 bytes input (2 loops)
        verify_decode_avx2(&config, &STANDARD, 97); // 1 quad + remainder
    }

    #[test]
    fn miri_avx2_decode_url_safe() {
        // Verify '-' and '_' handling in the SIMD path
        let config = Config {
            url_safe: true,
            padding: false,
        };

        // Construct specific input with URL safe chars
        // 0x3F (?) is usually '/', in URL safe it is '_'
        // 0x3E (>) is usually '+', in URL safe it is '-'
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_"; // 32 bytes
        let mut dst = [0u8; 32];

        unsafe {
            decode_slice_avx2(&config, input, dst.as_mut_ptr()).unwrap();
        }
    }

    // ----------------------------------------------------------------------
    // 3. Error Logic Coverage
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx2_decode_error_detection() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        let mut dst = [0u8; 256];

        // Case 1: Error in the Quad loop (byte 127)
        let mut bad_input_128 = vec![b'A'; 128];
        bad_input_128[127] = b'$'; // Invalid char
        let res = unsafe { decode_slice_avx2(&config, &bad_input_128, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Quad Loop lane 4");

        // Case 2: Error in the Single loop (byte 31)
        let mut bad_input_32 = vec![b'A'; 32];
        bad_input_32[31] = b'?'; // Invalid char
        let res = unsafe { decode_slice_avx2(&config, &bad_input_32, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Single Loop");

        // Case 3: Error in Quad Loop (first vector, first byte)
        let mut bad_input_128_first = vec![b'A'; 128];
        bad_input_128_first[0] = b'$';
        let res = unsafe { decode_slice_avx2(&config, &bad_input_128_first, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Quad Loop lane 1");

        // Case 4: Error in Scalar Fallback (after SIMD processing)
        let mut bad_input_33 = vec![b'A'; 33];
        bad_input_33[32] = b'?'; // Invalid in scalar region
        let res = unsafe { decode_slice_avx2(&config, &bad_input_33, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Scalar Fallback");
    }

    // ----------------------------------------------------------------------
    // 4. Roundtrip & Config Coverage
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx2_roundtrip_standard() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for &len in &[24, 48, 96, 97, 120, 192] {
            let input = random_bytes(len);
            let expected = STANDARD.encode(&input);
            let mut enc = vec![0u8; expected.len() * 2];
            unsafe {
                encode_slice_avx2(&config, &input, enc.as_mut_ptr());
            }
            let encoded = &enc[..expected.len()];
            assert_eq!(std::str::from_utf8(encoded).unwrap(), expected);

            let mut dec = vec![0u8; len + 64];
            let dec_len = unsafe { decode_slice_avx2(&config, encoded, dec.as_mut_ptr()).unwrap() };
            assert_eq!(&dec[..dec_len], &input, "Roundtrip len {}", len);
        }
    }

    #[test]
    fn miri_avx2_encode_no_padding() {
        use base64::engine::general_purpose::STANDARD_NO_PAD;
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[1, 24, 25, 48, 96, 97] {
            verify_encode_avx2(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx2_decode_no_padding() {
        use base64::engine::general_purpose::STANDARD_NO_PAD;
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[3, 24, 25, 48, 96, 97] {
            let input_bytes = random_bytes(len);
            let encoded = STANDARD_NO_PAD.encode(&input_bytes);
            let mut dst = vec![0u8; len + 64];
            let dec_len = unsafe {
                decode_slice_avx2(&config, encoded.as_bytes(), dst.as_mut_ptr()).unwrap()
            };
            assert_eq!(&dst[..dec_len], &input_bytes, "No-pad decode len {}", len);
        }
    }

    #[test]
    fn miri_avx2_decode_url_safe_padded() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        verify_decode_avx2(&config, &URL_SAFE, 50);
    }
}

/// Exhaustive regression test for the nibble-lookup `lut_lo`/`lut_hi`/`lut_roll`
/// tables in [`decode_constants_avx2`]. Runs on real AVX2 hardware (not a Kani
/// stub or a Python model), unlike the Kani harnesses this doesn't need the
/// `kani` toolchain, so it always runs under plain `cargo test`.
///
/// The table values were derived by hand (see the doc comment on
/// [`DecodeConstantsAvx2`]) and cross-checked exhaustively against this
/// crate's own scalar decode tables in a standalone Python script before
/// being transcribed into Rust; this test guards against transcription
/// mistakes (e.g. a mistyped hex literal) that a one-off script can't catch.
#[cfg(test)]
mod avx2_decode_lut_exhaustive {
    use super::*;

    /// For every possible byte value (0..=255), build a 36-byte input whose
    /// first 32 bytes are all that byte value (the one AVX2 vector processed
    /// by the fast path) followed by 4 valid filler bytes (pushed to the
    /// scalar tail), and check that `decode_slice_avx2` agrees with the
    /// scalar decoder on both validity and the decoded value.
    fn check_all_byte_values(config: &Config) {
        for candidate in 0u8..=255 {
            let mut input = [candidate; 36];
            input[32..].copy_from_slice(b"AAAA");

            let mut avx2_out = [0u8; 64];
            let avx2_result = unsafe { decode_slice_avx2(config, &input, avx2_out.as_mut_ptr()) };

            // Oracle: the first 32 bytes decoded on their own via the scalar
            // path (already covered by its own dedicated tests/Kani proofs).
            let mut scalar_out = [0u8; 64];
            let scalar_result = unsafe {
                crate::scalar::decode_slice_unsafe(config, &input[..32], scalar_out.as_mut_ptr())
            };

            match scalar_result {
                Ok(scalar_len) => {
                    assert!(
                        avx2_result.is_ok(),
                        "byte {candidate:#04x} ({candidate}): scalar accepted it (decoded \
                         {scalar_len} bytes) but avx2 rejected it with {avx2_result:?}"
                    );
                    let avx2_len = avx2_result.expect("checked above");
                    // avx2_len covers all 36 input bytes (32 vectorized + 4
                    // scalar-tail bytes "AAAA" -> 3 more decoded bytes).
                    assert_eq!(
                        avx2_len,
                        scalar_len + 3,
                        "byte {candidate:#04x}: length mismatch"
                    );
                    assert_eq!(
                        &avx2_out[..scalar_len],
                        &scalar_out[..scalar_len],
                        "byte {candidate:#04x} ({candidate}): decoded value mismatch"
                    );
                }
                Err(scalar_err) => {
                    assert_eq!(
                        avx2_result,
                        Err(scalar_err),
                        "byte {candidate:#04x} ({candidate}): avx2/scalar disagree on validity"
                    );
                }
            }
        }
    }

    #[test]
    fn avx2_lut_standard_matches_scalar() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        check_all_byte_values(&config);
    }

    #[test]
    fn avx2_lut_url_safe_matches_scalar() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        check_all_byte_values(&config);
    }
}

/// Exhaustive length-boundary regression test for the offset-load
/// `encode_slice_avx2` rewrite. Runs on real AVX2 hardware under plain
/// `cargo test` (no Kani needed) and compares against the independent
/// `base64` crate oracle at every length from 0 to 400 bytes — densely
/// covering the `rounds = (len - 4) / 24` arithmetic and the 4-round batch
/// boundary — plus a handful of large random lengths.
#[cfg(test)]
mod avx2_encode_length_sweep {
    use super::*;
    use base64::engine::general_purpose::{STANDARD as REF_STANDARD, URL_SAFE as REF_URL_SAFE};

    fn check(config: &Config, oracle: &impl base64::Engine, len: usize) {
        let input: Vec<u8> = (0..len)
            .map(|i| u8::try_from((i * 37 + 11) % 256).expect("value masked to fit in u8"))
            .collect();
        let expected = oracle.encode(&input);

        // `encoded_len` for padded output is always >= input.len() * 4 / 3;
        // add a little extra slack since this is a raw byte buffer, not the
        // exact `Engine::encoded_len` (padding doesn't matter here, we only
        // compare the encoded prefix).
        let mut dst = vec![0u8; len * 2 + 64];
        unsafe { encode_slice_avx2(config, &input, dst.as_mut_ptr()) };

        assert_eq!(
            &dst[..expected.len()],
            expected.as_bytes(),
            "encode mismatch at len={len}"
        );
    }

    #[test]
    fn avx2_encode_standard_all_lengths_0_to_400() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for len in 0..=400 {
            check(&config, &REF_STANDARD, len);
        }
    }

    #[test]
    fn avx2_encode_url_safe_all_lengths_0_to_400() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        for len in 0..=400 {
            check(&config, &REF_URL_SAFE, len);
        }
    }

    #[test]
    fn avx2_encode_large_lengths() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for len in [1_000, 10_000, 100_000, 1_000_003] {
            check(&config, &REF_STANDARD, len);
        }
    }
}
