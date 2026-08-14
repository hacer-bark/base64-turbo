use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};
use crate::{Config, Error};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m512i, _knot_mask64, _kor_mask64, _mm_loadu_si128, _mm_setr_epi8, _mm_storeu_si128,
    _mm512_add_epi8, _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_castsi512_si128,
    _mm512_cmpeq_epi8_mask, _mm512_cmpgt_epi8_mask, _mm512_cmple_epu8_mask,
    _mm512_extracti32x4_epi32, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16,
    _mm512_mask_add_epi8, _mm512_permutexvar_epi32, _mm512_set1_epi8, _mm512_set1_epi32,
    _mm512_setr_epi32, _mm512_shuffle_epi8, _mm512_sllv_epi16, _mm512_srli_epi16,
    _mm512_srlv_epi16, _mm512_storeu_si512, _mm512_sub_epi8, _mm512_subs_epu8,
    _mm512_ternarylogic_epi32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m512i, _knot_mask64, _kor_mask64, _mm_loadu_si128, _mm_setr_epi8, _mm_storeu_si128,
    _mm512_add_epi8, _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_castsi512_si128,
    _mm512_cmpeq_epi8_mask, _mm512_cmpgt_epi8_mask, _mm512_cmple_epu8_mask,
    _mm512_extracti32x4_epi32, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16,
    _mm512_mask_add_epi8, _mm512_permutexvar_epi32, _mm512_set1_epi8, _mm512_set1_epi32,
    _mm512_setr_epi32, _mm512_shuffle_epi8, _mm512_sllv_epi16, _mm512_srli_epi16,
    _mm512_srlv_epi16, _mm512_storeu_si512, _mm512_sub_epi8, _mm512_subs_epu8,
    _mm512_ternarylogic_epi32,
};

// --- Plain AVX-512F/BW encoder ---

#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn encode_slice_avx512(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    // Shuffle bytes for mul
    let shuffle = _mm512_broadcast_i32x4(_mm_setr_epi8(
        1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
    ));

    let set_25 = _mm512_set1_epi8(25);
    let set_51 = _mm512_set1_epi8(51);
    let one = _mm512_set1_epi8(1);
    let translate_lut = if config.url_safe {
        _mm512_broadcast_i32x4(_mm_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -17, 32, 0, 0,
        ))
    } else {
        _mm512_broadcast_i32x4(_mm_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0,
        ))
    };

    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // 3 bytes -> 4 six-bit indices via AVX-512F/BW variable shifts
            // (no VBMI, no mullo/mulhi trick).
            let v = _mm512_shuffle_epi8($in_vec, shuffle);

            let t0 = _mm512_and_si512(v, _mm512_set1_epi32(0x0fc0_fc00));
            let t1 = _mm512_srlv_epi16(t0, _mm512_set1_epi32(0x0006_000a));
            let t2 = _mm512_sllv_epi16(v, _mm512_set1_epi32(0x0008_0004));
            let indices = _mm512_ternarylogic_epi32::<0xca>(_mm512_set1_epi32(0x3f00_3f00), t2, t1);

            let sub_base = _mm512_subs_epu8(indices, set_51);
            let m_gt25 = _mm512_cmpgt_epi8_mask(indices, set_25);
            let lut_idx = _mm512_mask_add_epi8(sub_base, m_gt25, sub_base, one);

            _mm512_add_epi8(indices, _mm512_shuffle_epi8(translate_lut, lut_idx))
        }};
    }

    // Permutation index for 48-byte distribution into 128-bit lanes
    let permute_idx = _mm512_setr_epi32(
        0, 1, 2, 3, // Lane 0 gets elements 0, 1, 2, and 3 (bytes 12-15 as garbage)
        3, 4, 5, 6, // Lane 1 gets elements 3, 4, 5, and 6 (bytes 24-27 as garbage)
        6, 7, 8, 9, // Lane 2 gets elements 6, 7, 8, and 9
        9, 10, 11, 12, // Lane 3 gets elements 9, 10, 11, and 12
    );

    macro_rules! load_48_bytes {
        ($ptr:expr) => {{
            let v = unsafe { _mm512_loadu_si512($ptr.cast()) };
            _mm512_permutexvar_epi32(permute_idx, v)
        }};
    }

    // Quad tier: 192 input bytes -> 256 output.
    let safe_len_192 = len.saturating_sub(16);
    let aligned_len_192 = safe_len_192 - (safe_len_192 % 192);
    let src_end_192 = unsafe { src.add(aligned_len_192) };

    while src < src_end_192 {
        let v0 = load_48_bytes!(src);
        let v1 = load_48_bytes!(src.add(48));
        let v2 = load_48_bytes!(src.add(96));
        let v3 = load_48_bytes!(src.add(144));

        let i0 = encode_vec!(v0);
        let i1 = encode_vec!(v1);
        let i2 = encode_vec!(v2);
        let i3 = encode_vec!(v3);

        unsafe { _mm512_storeu_si512(dst.cast(), i0) };
        unsafe { _mm512_storeu_si512(dst.add(64).cast(), i1) };
        unsafe { _mm512_storeu_si512(dst.add(128).cast(), i2) };
        unsafe { _mm512_storeu_si512(dst.add(192).cast(), i3) };

        src = unsafe { src.add(192) };
        dst = unsafe { dst.add(256) };
    }

    // Single tier: 48 input bytes -> 64 output.
    let safe_len_single = len.saturating_sub(16);
    let aligned_len_single = safe_len_single - (safe_len_single % 48);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let v = load_48_bytes!(src);
        let res = encode_vec!(v);
        unsafe { _mm512_storeu_si512(dst.cast(), res) };

        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::encode(config, input, src, dst_slice, dst_off) };
}

/// Precomputed AVX-512 decode constants, factored out of [`decode_slice_avx512`]
/// only to keep its body under clippy's line-count threshold.
struct DecodeConstantsAvx512 {
    lut_hi_nibble: __m512i,
    sym_62: __m512i,
    sym_63: __m512i,
    delta_62: __m512i,
    delta_63: __m512i,
    range_0: __m512i,
    digit_span: __m512i,
    range_a: __m512i,
    upper_span: __m512i,
    range_a_low: __m512i,
    range_z_low_len: __m512i,
    pack_l1: __m512i,
    pack_l2: __m512i,
    pack_shuffle: __m512i,
    mask_hi_nibble: __m512i,
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn decode_constants_avx512(config: &Config) -> DecodeConstantsAvx512 {
    // LUT for offsets based on high nibble (bits 4-7).
    let lut_hi_nibble = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 0, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0,
    ));

    // Range and offsets of special chars
    let (char_62, char_63) = if config.url_safe {
        (b'-', b'_')
    } else {
        (b'+', b'/')
    };
    let sym_62 = _mm512_set1_epi8(char_62.cast_signed());
    let sym_63 = _mm512_set1_epi8(char_63.cast_signed());

    let (fix_62, fix_63) = if config.url_safe { (-2, 33) } else { (0, -3) };
    let delta_62 = _mm512_set1_epi8(fix_62);
    let delta_63 = _mm512_set1_epi8(fix_63);

    // Range Validation Constants
    let range_0 = _mm512_set1_epi8(b'0'.cast_signed());
    let digit_span = _mm512_set1_epi8(9);

    let range_a = _mm512_set1_epi8(b'A'.cast_signed());
    let upper_span = _mm512_set1_epi8(25);

    let range_a_low = _mm512_set1_epi8(b'a'.cast_signed());
    let range_z_low_len = _mm512_set1_epi8(25);

    // Packing Constants
    let pack_l1 =
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_L1.as_ptr().cast::<__m128i>())) };
    let pack_l2 =
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_L2.as_ptr().cast::<__m128i>())) };
    let pack_shuffle =
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_SHUFFLE.as_ptr().cast::<__m128i>())) };

    // Masks for nibble extraction
    let mask_hi_nibble = _mm512_set1_epi8(0x0F);

    DecodeConstantsAvx512 {
        lut_hi_nibble,
        sym_62,
        sym_63,
        delta_62,
        delta_63,
        range_0,
        digit_span,
        range_a,
        upper_span,
        range_a_low,
        range_z_low_len,
        pack_l1,
        pack_l2,
        pack_shuffle,
        mask_hi_nibble,
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn decode_slice_avx512(
    config: &Config,
    input: &[u8],
    dst_slice: &mut [u8],
) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let DecodeConstantsAvx512 {
        lut_hi_nibble,
        sym_62,
        sym_63,
        delta_62,
        delta_63,
        range_0,
        digit_span,
        range_a,
        upper_span,
        range_a_low,
        range_z_low_len,
        pack_l1,
        pack_l2,
        pack_shuffle,
        mask_hi_nibble,
    } = unsafe { decode_constants_avx512(config) };

    // Validate + decode one vector, using mask ops for zero-blend fixups.
    macro_rules! decode_vec {
        ($input:expr) => {{
            let hi = _mm512_and_si512(_mm512_srli_epi16($input, 4), mask_hi_nibble);
            let offset = _mm512_shuffle_epi8(lut_hi_nibble, hi);
            let mut indices = _mm512_add_epi8($input, offset);

            let mask_62 = _mm512_cmpeq_epi8_mask($input, sym_62);
            let mask_63 = _mm512_cmpeq_epi8_mask($input, sym_63);

            indices = _mm512_mask_add_epi8(indices, mask_62, indices, delta_62);
            indices = _mm512_mask_add_epi8(indices, mask_63, indices, delta_63);

            let is_sym = _kor_mask64(mask_62, mask_63);

            let sub_0 = _mm512_sub_epi8($input, range_0);
            let is_num = _mm512_cmple_epu8_mask(sub_0, digit_span);

            let sub_a = _mm512_sub_epi8($input, range_a);
            let is_upper = _mm512_cmple_epu8_mask(sub_a, upper_span);

            let sub_a_low = _mm512_sub_epi8($input, range_a_low);
            let is_lower = _mm512_cmple_epu8_mask(sub_a_low, range_z_low_len);

            let is_char = _kor_mask64(is_num, _kor_mask64(is_upper, is_lower));
            let is_valid = _kor_mask64(is_char, is_sym);
            let err_mask = _knot_mask64(is_valid);

            (indices, err_mask)
        }};
    }

    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            let m = _mm512_maddubs_epi16($indices, pack_l1);
            let p = _mm512_madd_epi16(m, pack_l2);
            let out = _mm512_shuffle_epi8(p, pack_shuffle);

            let lane0 = _mm512_castsi512_si128(out);
            unsafe { _mm_storeu_si128($dst_ptr.cast::<__m128i>(), lane0) };
            let lane1 = _mm512_extracti32x4_epi32(out, 1);
            unsafe { _mm_storeu_si128($dst_ptr.add(12).cast::<__m128i>(), lane1) };
            let lane2 = _mm512_extracti32x4_epi32(out, 2);
            unsafe { _mm_storeu_si128($dst_ptr.add(24).cast::<__m128i>(), lane2) };
            let lane3 = _mm512_extracti32x4_epi32(out, 3);
            unsafe { _mm_storeu_si128($dst_ptr.add(36).cast::<__m128i>(), lane3) };
        }};
    }

    // Quad tier: 256 input bytes -> 192 output.
    let safe_len_256 = len.saturating_sub(4);
    let aligned_len_256 = safe_len_256 - (safe_len_256 % 256);
    let src_end_256 = unsafe { src.add(aligned_len_256) };

    while src < src_end_256 {
        let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(64).cast::<__m512i>()) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(128).cast::<__m512i>()) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(192).cast::<__m512i>()) };

        let (i0, e0) = decode_vec!(v0);
        let (i1, e1) = decode_vec!(v1);
        let (i2, e2) = decode_vec!(v2);
        let (i3, e3) = decode_vec!(v3);

        if (e0 | e1 | e2 | e3) != 0 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(i0, dst);
        pack_and_store!(i1, dst.add(48));
        pack_and_store!(i2, dst.add(96));
        pack_and_store!(i3, dst.add(144));

        src = unsafe { src.add(256) };
        dst = unsafe { dst.add(192) };
    }

    // Single tier: 64 input bytes -> 48 output.
    let safe_len_64 = len.saturating_sub(4);
    let aligned_len_64 = safe_len_64 - (safe_len_64 % 64);
    let src_end_64 = unsafe { input.as_ptr().add(aligned_len_64) };

    while src < src_end_64 {
        let v = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let (idx, err_mask) = decode_vec!(v);

        if err_mask != 0 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(idx, dst);

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

// Verification: Kani proofs and the Miri coverage suite.
#[cfg(any(kani, test))]
mod verify;
