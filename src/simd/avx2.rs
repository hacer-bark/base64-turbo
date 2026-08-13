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

    // Both loops read a full 32-byte vector per 32 bytes they consume, so
    // neither may start within 4 bytes of the end: `safe_len` is how much
    // input the SIMD path is allowed to look at, and each tier rounds it
    // down to its own block size.
    let safe_len = len.saturating_sub(4);
    let aligned_len_128 = safe_len - (safe_len % 128);
    let aligned_len_32 = safe_len - (safe_len % 32);
    let src_end_128 = unsafe { src.add(aligned_len_128) };
    let src_end_32 = unsafe { src.add(aligned_len_32) };

    // Process 128 bytes (4 chunks) at a time

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

    use super::intrinsic_models as m;

    // Layer 1 — index proofs. Every load/store bound is a statement about
    // `rounds`, `remaining` and the `src`/`dst` offsets, not about the input
    // bytes, so these drop the vectors and reason over a symbolic `len` and
    // an *arbitrary* iteration index. That makes them an induction — base
    // case, step, exit — covering all N at near-zero solver cost. They
    // mirror the offset arithmetic of the kernels; each constant names the
    // operation it tracks, and nothing but this file keeps them in sync, so
    // treat them as code. The README's "How the Kani proofs work" has the
    // full argument.

    /// Largest `len` the index proofs consider: `Engine::encoded_len`'s
    /// unpadded branch does `len * 4`, which overflows past `usize::MAX / 4`
    /// (~4 EB), so above it the public API cannot size a buffer anyway.
    const MAX_LEN: usize = usize::MAX / 4;

    // Encoder model, mirroring `encode_slice_avx2`.
    const ENC_ROUND_IN: usize = 24; // logical input bytes per round
    const ENC_ROUND_OUT: usize = 32; // output bytes per round
    const ENC_LOAD: usize = 32; // bytes each `_mm256_loadu_si256` reads
    const ENC_FIRST_ADVANCE: usize = 20; // `src.add(20)` after the first round
    const ENC_UNROLL: usize = 4; // rounds per 4x-unrolled iteration

    fn enc_cap(len: usize, padding: bool) -> usize {
        if padding {
            TURBO_STANDARD.encoded_len(len)
        } else {
            TURBO_STANDARD_NO_PAD.encoded_len(len)
        }
    }

    /// A symbolic `rounds` pinned to `(len - 4) / 24` by the two inequalities
    /// that define it. Constraining the quotient is far cheaper for CBMC than
    /// the division; `check_enc_rounds_model` proves the two agree.
    fn any_enc_rounds(len: usize) -> usize {
        let rounds: usize = kani::any();
        kani::assume(rounds <= MAX_LEN / ENC_ROUND_IN);
        kani::assume(ENC_ROUND_IN * rounds <= len - 4);
        kani::assume(len - 4 < ENC_ROUND_IN * (rounds + 1));
        rounds
    }

    /// `(src_off, dst_off)` after `done >= 1` rounds. The first round is the
    /// permuted block advancing `src` by only 20; later rounds advance 24,
    /// giving the uniform `24 * done - 4` (the deficit repaid by the trailing
    /// `src.add(4)`).
    fn enc_state(done: usize) -> (usize, usize) {
        (ENC_ROUND_IN * done - 4, ENC_ROUND_OUT * done)
    }

    /// Isolated so the suite's one non-power-of-two division owns its run.
    #[kani::proof]
    fn check_enc_rounds_model() {
        let len: usize = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        assert_eq!(rounds, (len - 4) / ENC_ROUND_IN);
        // `remaining = rounds - 1` must not underflow — the reason for the
        // `len >= 32` guard rather than `len >= 28`.
        assert!(rounds >= 1);
    }

    /// Base case: the permuted first round is in bounds and leaves the state
    /// the loop invariant assumes.
    #[kani::proof]
    fn check_enc_first_block() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let cap = enc_cap(len, padding);

        assert!(ENC_LOAD <= len); // reads [0, 32)
        assert!(ENC_ROUND_OUT <= cap); // writes [0, 32)

        let (src_off, dst_off) = enc_state(1);
        assert_eq!(src_off, ENC_FIRST_ADVANCE);
        assert_eq!(dst_off, ENC_ROUND_OUT);
        assert!(rounds >= 1);
    }

    /// Inductive step for the 4x-unrolled tier, over an arbitrary iteration.
    #[kani::proof]
    fn check_enc_quad_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let cap = enc_cap(len, padding);

        let done: usize = kani::any();
        kani::assume(done >= 1 && done <= rounds);
        let remaining = rounds - done;
        kani::assume(remaining >= ENC_UNROLL); // guard `while remaining >= 4`

        let (src_off, dst_off) = enc_state(done);

        // Widest body accesses: `src.add(72)` load, `dst.add(96)` store.
        assert!(src_off + 72 + ENC_LOAD <= len, "quad load leaves input");
        assert!(
            dst_off + 96 + ENC_ROUND_OUT <= cap,
            "quad store leaves output"
        );

        // Update `src.add(96)`, `dst.add(128)`, `remaining -= 4` lands
        // exactly on the invariant's next state — the machine-checked step.
        let done_next = done + ENC_UNROLL;
        assert_eq!((src_off + 96, dst_off + 128), enc_state(done_next));
        assert!(done_next <= rounds);
        assert_eq!(remaining - ENC_UNROLL, rounds - done_next);
    }

    /// Inductive step for the single-round tier.
    #[kani::proof]
    fn check_enc_single_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let cap = enc_cap(len, padding);

        let done: usize = kani::any();
        kani::assume(done >= 1 && done <= rounds);
        let remaining = rounds - done;
        kani::assume(remaining >= 1);

        let (src_off, dst_off) = enc_state(done);
        assert!(src_off + ENC_LOAD <= len, "single load leaves input");
        assert!(dst_off + ENC_ROUND_OUT <= cap, "single store leaves output");

        // Update `src.add(24)`, `dst.add(32)`, `remaining -= 1`.
        let done_next = done + 1;
        assert_eq!(
            (src_off + ENC_ROUND_IN, dst_off + ENC_ROUND_OUT),
            enc_state(done_next)
        );
        assert!(done_next <= rounds);
        assert_eq!(remaining - 1, rounds - done_next);
    }

    /// Exit case: after both loops drain, the trailing `src.add(4)` and the
    /// scalar handoff account for exactly what is left.
    #[kani::proof]
    fn check_enc_tail_handoff() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let (src_off, dst_off) = enc_state(rounds);

        let processed = src_off + 4; // repays the first round's deficit
        assert_eq!(processed, ENC_ROUND_IN * rounds);

        // `&input[processed..]` never panics and is never empty: `rounds`
        // caps at `(len - 4) / 24`, leaving at least 4 bytes for the tail.
        assert!(processed < len);
        let tail = len - processed;
        assert!(tail >= 4);

        // SIMD prefix plus scalar tail is exactly the encoded length — no
        // overrun, no short write (`processed` is a multiple of 3).
        assert_eq!(
            dst_off + enc_cap(tail, padding),
            enc_cap(len, padding),
            "prefix + tail must equal encoded length"
        );
    }

    // Decoder model, mirroring `decode_slice_avx2`.
    const DEC_LOAD: usize = 32; // bytes each `_mm256_loadu_si256` reads
    const DEC_BLOCK_IN: usize = 32; // input bytes per single-vector pass
    const DEC_BLOCK_OUT: usize = 24; // dst advance per single-vector pass
    /// Bytes `pack_and_store!` touches: 16 at `dst` + 16 at `dst.add(12)`,
    /// 4 wider than the 24 it advances — every store overhangs its block.
    const DEC_STORE_SPAN: usize = 28;
    const DEC_QUAD_IN: usize = 128; // input bytes per quad-tier iteration
    const DEC_QUAD_OUT: usize = 96; // dst advance per quad-tier iteration

    fn dec_cap(len: usize) -> usize {
        TURBO_STANDARD.estimate_decoded_len(len)
    }

    /// The `aligned_len_128` / `aligned_len_32` loop windows, both from the
    /// `len.saturating_sub(4)` margin that keeps a 32-byte load in bounds.
    fn dec_windows(len: usize) -> (usize, usize) {
        let safe = len.saturating_sub(4);
        (safe - safe % DEC_QUAD_IN, safe - safe % DEC_BLOCK_IN)
    }

    /// Inductive step for the decoder's quad tier, over an arbitrary iteration.
    #[kani::proof]
    fn check_dec_quad_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (aligned_quad, _) = dec_windows(len);
        let cap = dec_cap(len);

        let i: usize = kani::any();
        kani::assume(i <= MAX_LEN / DEC_QUAD_IN);
        let (src_off, dst_off) = (DEC_QUAD_IN * i, DEC_QUAD_OUT * i);
        kani::assume(src_off < aligned_quad); // guard `src < src_end_128`

        // Widest: `src.add(96)` load, `pack_and_store!(_, dst.add(72))`.
        assert!(src_off + 96 + DEC_LOAD <= len, "quad load leaves input");
        assert!(
            dst_off + 72 + DEC_STORE_SPAN <= cap,
            "quad store leaves output"
        );

        // Update `src.add(128)`, `dst.add(96)`.
        assert_eq!(
            (src_off + DEC_QUAD_IN, dst_off + DEC_QUAD_OUT),
            (DEC_QUAD_IN * (i + 1), DEC_QUAD_OUT * (i + 1))
        );
    }

    /// Inductive step for the decoder's single-vector tier, entered from
    /// wherever the quad tier stopped.
    #[kani::proof]
    fn check_dec_single_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (aligned_quad, aligned_block) = dec_windows(len);
        let cap = dec_cap(len);
        let quads = aligned_quad / DEC_QUAD_IN;

        let j: usize = kani::any();
        kani::assume(j <= MAX_LEN / DEC_BLOCK_IN);
        let src_off = aligned_quad + DEC_BLOCK_IN * j;
        let dst_off = DEC_QUAD_OUT * quads + DEC_BLOCK_OUT * j;
        kani::assume(src_off < aligned_block); // guard `src < src_end_32`

        assert!(src_off + DEC_LOAD <= len, "single load leaves input");
        assert!(
            dst_off + DEC_STORE_SPAN <= cap,
            "single store leaves output"
        );

        // Update `src.add(32)`, `dst.add(24)`.
        assert_eq!(
            (src_off + DEC_BLOCK_IN, dst_off + DEC_BLOCK_OUT),
            (
                aligned_quad + DEC_BLOCK_IN * (j + 1),
                DEC_QUAD_OUT * quads + DEC_BLOCK_OUT * (j + 1)
            )
        );
    }

    /// Exit case: whatever the loops leave fits the space the caller
    /// guaranteed, so the scalar decoder cannot overrun it.
    #[kani::proof]
    fn check_dec_tail_handoff() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (_, aligned_block) = dec_windows(len);
        let cap = dec_cap(len);

        // Both tiers advance dst 3 bytes per 4 consumed, so the handover
        // offset depends only on the window, not on which tier got there.
        let dst_off = DEC_BLOCK_OUT * (aligned_block / DEC_BLOCK_IN);
        assert!(aligned_block <= len);
        let tail = len - aligned_block;
        assert!(
            dst_off + dec_cap(tail) <= cap,
            "scalar tail can overrun output"
        );
    }

    // Layer 2 — kernel proofs. These run the real code over fully symbolic
    // bytes, covering the character mapping, validation LUTs and panic
    // freedom. Layer 1 owns the loop arithmetic, so each need only reach its
    // kernel once: one round, not two, and no quad-tier roundtrip (the quad
    // tier runs the same kernel, and its offsets are proven above). Buffers
    // are the exact public-API capacities, so any real overrun fails.

    /// One first round + 13-byte scalar tail.
    const ENC_KERNEL_LEN: usize = 37;
    /// One single-vector pass + 5-byte scalar tail.
    const DEC_KERNEL_LEN: usize = 37;

    // A length below its tier's guard would pass while proving nothing (an
    // earlier revision used 29, under the encoder's `len >= 32` guard, and
    // silently verified only the scalar fallback). Fail the build, not late.
    const _: () = assert!(
        ENC_KERNEL_LEN >= 32
            && (ENC_KERNEL_LEN - 4) / ENC_ROUND_IN == 1
            && ENC_KERNEL_LEN % ENC_ROUND_IN != 0,
        "ENC_KERNEL_LEN must run one AVX2 round and leave an unaligned tail"
    );
    const _: () = assert!(
        (DEC_KERNEL_LEN - 4) / DEC_BLOCK_IN == 1,
        "DEC_KERNEL_LEN must run one single-vector decode pass"
    );

    const ENC_KERNEL_CAP: usize = TURBO_STANDARD.encoded_len(ENC_KERNEL_LEN);
    const ENC_KERNEL_DEC_CAP: usize = TURBO_STANDARD.estimate_decoded_len(ENC_KERNEL_CAP);
    const DEC_KERNEL_CAP: usize = TURBO_STANDARD.estimate_decoded_len(DEC_KERNEL_LEN);

    /// `Decode(Encode(x)) == x` over every 37-byte input. The alphabet is a
    /// parameter, not symbolic: it only selects constant LUTs, so two lean
    /// runs cover it without doubling one run's state.
    fn roundtrip_kernel(url_safe: bool) {
        let config = Config {
            url_safe,
            padding: true,
        };
        let input: [u8; ENC_KERNEL_LEN] = kani::any();

        let mut enc_buf = [0u8; ENC_KERNEL_CAP];
        let mut dec_buf = [0u8; ENC_KERNEL_DEC_CAP];

        unsafe {
            encode_slice_avx2(&config, &input, enc_buf.as_mut_ptr());
            let dec_len = decode_slice_avx2(&config, &enc_buf, dec_buf.as_mut_ptr())
                .expect("valid encoding failed to decode");
            assert_eq!(dec_len, ENC_KERNEL_LEN);
            assert_eq!(&dec_buf[..dec_len], &input, "roundtrip mismatch");
        }
    }

    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, m::_mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, m::_mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, m::_mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, m::_mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, m::_mm256_madd_epi16_stub)]
    #[kani::stub(_mm256_mulhi_epu16, m::_mm256_mulhi_epu16_stub)]
    #[kani::stub(_mm256_permutevar8x32_epi32, m::_mm256_permutevar8x32_epi32_stub)]
    fn check_avx2_roundtrip_standard() {
        roundtrip_kernel(false);
    }

    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, m::_mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, m::_mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, m::_mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, m::_mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, m::_mm256_madd_epi16_stub)]
    #[kani::stub(_mm256_mulhi_epu16, m::_mm256_mulhi_epu16_stub)]
    #[kani::stub(_mm256_permutevar8x32_epi32, m::_mm256_permutevar8x32_epi32_stub)]
    fn check_avx2_roundtrip_url_safe() {
        roundtrip_kernel(true);
    }

    /// Every 37-byte garbage input either decodes or returns `Err`, never
    /// panicking or overrunning — covering the validation LUTs over all 256
    /// byte values in every lane.
    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, m::_mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, m::_mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, m::_mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, m::_mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, m::_mm256_madd_epi16_stub)]
    fn check_avx2_decode_robustness() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; DEC_KERNEL_LEN] = kani::any();
        let mut output = [0u8; DEC_KERNEL_CAP];
        unsafe {
            let _ = decode_slice_avx2(&config, &input, output.as_mut_ptr());
        }
    }
}

/// Rust models of every AVX2 intrinsic the kernels use, for the Kani proofs.
///
/// Each model is transcribed line for line from the Intel Intrinsics Guide
/// pseudocode — comments and all — so it can be diffed against the reference.
/// The Layer 2 proofs are proven about *these*, so a wrong model weakens
/// every proof that stubs it; `avx2_stub_equivalence` re-checks them against
/// real hardware, which is why this compiles under `test` as well as `kani`.
/// Do not "clean up" a model: fidelity to the pseudocode is the point, which
/// is also why the transcription lints below are silenced rather than fixed.
#[cfg(any(kani, test))]
#[allow(non_snake_case)]
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::missing_transmute_annotations,
    clippy::needless_late_init,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::used_underscore_items
)]
pub(super) mod intrinsic_models {
    use super::*;
    use std::mem::transmute;

    // STUB: _mm256_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_epi8
    pub(super) unsafe fn _mm256_shuffle_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
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
    pub(super) unsafe fn _mm256_subs_epu8_stub(a: __m256i, b: __m256i) -> __m256i {
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
    // Split into four u64 lanes since Rust has no native 256-bit integer type.
    pub(super) unsafe fn _mm256_testz_si256_stub(a: __m256i, b: __m256i) -> i32 {
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
    pub(super) unsafe fn _mm256_maddubs_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
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
    pub(super) unsafe fn _mm256_madd_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
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

    // STUB: _mm256_mulhi_epu16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mulhi_epu16
    pub(super) unsafe fn _mm256_mulhi_epu16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u16; 16] = unsafe { transmute(a) };
        let b: [u16; 16] = unsafe { transmute(b) };
        let mut dst = [0u16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j;
            // tmp[31:0] := ZeroExtend32(a[i+15:i]) * ZeroExtend32(b[i+15:i])
            let tmp: u32 = (a[i] as u32) * (b[i] as u32);
            // dst[i+15:i] := tmp[31:16]
            dst[i] = (tmp >> 16) as u16;
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_permutevar8x32_epi32
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permutevar8x32_epi32
    pub(super) unsafe fn _mm256_permutevar8x32_epi32_stub(a: __m256i, idx: __m256i) -> __m256i {
        let a: [u32; 8] = unsafe { transmute(a) };
        let idx: [u32; 8] = unsafe { transmute(idx) };
        let mut dst = [0u32; 8];

        // FOR j := 0 to 7
        for j in 0..8 {
            // id := idx[j*32+2:j*32]
            let id = (idx[j] & 0x7) as usize;
            // dst[j*32+31:j*32] := a[id*32+31:id*32]
            dst[j] = a[id];
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }
}

/// Checks every model in [`intrinsic_models`] against the real instruction
/// on AVX2 hardware, under plain `cargo test`. The Kani proofs trust the
/// models, so a model that disagrees with the silicon is the one assumption
/// underneath every Layer 2 result that Kani cannot itself check.
#[cfg(test)]
#[cfg(not(miri))]
#[allow(clippy::used_underscore_items)] // calling the models is the point
mod avx2_stub_equivalence {
    use super::intrinsic_models as model;
    use super::*;
    use std::mem::transmute;

    /// Saturation and sign boundaries, the high bit that zeroes a shuffle
    /// lane, index-shaped bytes, and deterministic noise.
    fn probes() -> Vec<[u8; 32]> {
        let byte = |i: usize| u8::try_from(i).expect("index below the 32-byte vector width");

        let mut out = vec![[0x00; 32], [0xFF; 32], [0x80; 32], [0x7F; 32], [0x01; 32]];
        out.push(core::array::from_fn(byte));
        out.push(core::array::from_fn(|i| byte(i) | 0x80));
        out.push(core::array::from_fn(|i| byte(i % 16)));
        out.push(core::array::from_fn(|i| 0xFF - byte(i)));

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..12 {
            out.push(core::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u8::try_from(state >> 56).expect("shifted down to 8 bits")
            }));
        }
        out
    }

    #[target_feature(enable = "avx2")]
    unsafe fn compare_all() {
        let probes = probes();
        // SAFETY: `__m256i` has no invalid bit patterns, so it and `[u8; 32]`
        // are freely transmutable both ways.
        let bytes = |v: __m256i| -> [u8; 32] { unsafe { transmute::<__m256i, [u8; 32]>(v) } };

        // Each arm: `real(a, b)` must equal `model(a, b)` for every probe pair.
        macro_rules! same {
            ($real:ident, $model:ident, $wrap:expr) => {
                for x in &probes {
                    for y in &probes {
                        let (a, b) = unsafe {
                            (
                                transmute::<[u8; 32], __m256i>(*x),
                                transmute::<[u8; 32], __m256i>(*y),
                            )
                        };
                        assert_eq!(
                            $wrap($real(a, b)),
                            $wrap(unsafe { model::$model(a, b) }),
                            "{}: a={x:02x?} b={y:02x?}",
                            stringify!($real)
                        );
                    }
                }
            };
        }

        same!(_mm256_shuffle_epi8, _mm256_shuffle_epi8_stub, bytes);
        same!(_mm256_subs_epu8, _mm256_subs_epu8_stub, bytes);
        same!(_mm256_maddubs_epi16, _mm256_maddubs_epi16_stub, bytes);
        same!(_mm256_madd_epi16, _mm256_madd_epi16_stub, bytes);
        same!(_mm256_mulhi_epu16, _mm256_mulhi_epu16_stub, bytes);
        same!(
            _mm256_permutevar8x32_epi32,
            _mm256_permutevar8x32_epi32_stub,
            bytes
        );
        same!(
            _mm256_testz_si256,
            _mm256_testz_si256_stub,
            core::convert::identity
        );
    }

    #[test]
    fn avx2_models_match_hardware() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no AVX2 on this machine");
            return;
        }
        unsafe { compare_all() };
    }
}

#[cfg(all(test, miri))]
mod miri_avx2_coverage {
    use super::*;
    use base64::{
        Engine,
        engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE},
    };

    /// Seeded xorshift, not `rand`, so a MIRI failure reproduces exactly.
    fn bytes(len: usize) -> Vec<u8> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u8::try_from(state >> 56).expect("shifted down to 8 bits")
            })
            .collect()
    }

    /// Encode against the oracle and decode back, with buffers sized exactly
    /// as the public API sizes them so MIRI sees the real caller's provenance.
    fn check(config: &Config, oracle: &impl Engine, len: usize) {
        let input = bytes(len);
        let expected = oracle.encode(&input);

        let mut enc = vec![0u8; expected.len()];
        unsafe { encode_slice_avx2(config, &input, enc.as_mut_ptr()) };
        assert_eq!(
            core::str::from_utf8(&enc).unwrap(),
            expected,
            "encode mismatch at len {len}"
        );

        let mut dec = vec![0u8; (enc.len() / 4 + 1) * 3];
        let dec_len = unsafe {
            decode_slice_avx2(config, &enc, dec.as_mut_ptr()).expect("valid input failed to decode")
        };
        assert_eq!(&dec[..dec_len], &input[..], "decode mismatch at len {len}");
    }

    /// One raw length per distinct code path; the label names the path.
    const TIER_LENGTHS: &[(usize, &str)] = &[
        (0, "empty"),
        (1, "scalar only"),
        (23, "scalar only, longest sub-round"),
        (31, "scalar only, just under the SIMD guard"),
        (32, "encode: first block, no loop"),
        (37, "encode: first block + unaligned scalar tail"),
        (53, "encode: first block + one single-tier round"),
        (96, "decode: quad window not yet reached"),
        (97, "decode: exactly one quad pass"),
        (124, "encode: exactly one quad pass"),
        (192, "both: quad pass then single-tier rounds"),
    ];

    #[test]
    fn miri_avx2_standard() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for &(len, tier) in TIER_LENGTHS {
            println!("standard: len {len} ({tier})");
            check(&config, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx2_url_safe() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        for &(len, tier) in TIER_LENGTHS {
            println!("url-safe: len {len} ({tier})");
            check(&config, &URL_SAFE, len);
        }
    }

    #[test]
    fn miri_avx2_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &(len, tier) in TIER_LENGTHS {
            println!("no-pad: len {len} ({tier})");
            check(&config, &STANDARD_NO_PAD, len);
        }
    }

    /// Invalid bytes must be caught in every tier, including the last lane
    /// of a quad pass, where an early-out would otherwise have already
    /// stored three sub-blocks.
    #[test]
    fn miri_avx2_decode_rejects_invalid() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        let mut dst = [0u8; 256];

        for &(len, bad_at, where_) in &[
            (32, 31, "single tier"),
            (33, 32, "scalar tail"),
            (132, 0, "quad tier, first lane"),
            (132, 127, "quad tier, last lane"),
        ] {
            let mut input = vec![b'A'; len];
            input[bad_at] = b'$';
            let res = unsafe { decode_slice_avx2(&config, &input, dst.as_mut_ptr()) };
            assert!(res.is_err(), "missed invalid byte in {where_}");
        }
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
#[cfg(not(miri))]
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
#[cfg(not(miri))]
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
