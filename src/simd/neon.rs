use crate::{Config, Error};

use core::arch::aarch64::{
    int8x16_t, int16x8_t, int32x4_t, uint8x16_t, uint16x8_t, vaddq_s8, vandq_s8, vandq_u8,
    vandq_u16, vceqq_u8, vcgeq_u8, vcgtq_s8, vcleq_u8, vcombine_u16, vdupq_n_s8, vdupq_n_u8,
    vdupq_n_u16, vget_low_s8, vget_low_s16, vget_low_u8, vget_low_u16, vld1q_s8, vld1q_s16,
    vld1q_u8, vld1q_u16, vmaxvq_u8, vmull_high_s8, vmull_high_s16, vmull_high_u16, vmull_s8,
    vmull_s16, vmull_u16, vmulq_u16, vmvnq_u8, vorrq_s8, vorrq_u8, vorrq_u16, vpaddq_s16,
    vpaddq_s32, vqsubq_u8, vqtbl1q_s8, vqtbl1q_u8, vreinterpret_s8_u8, vreinterpretq_s8_u8,
    vreinterpretq_u8_s8, vreinterpretq_u8_s32, vreinterpretq_u8_u16, vreinterpretq_u16_u8,
    vshrn_n_u32, vshrq_n_u8, vst1q_u8,
};

/// Unsigned multiply-high for u16x8. NEON has no `mulhi_u16`, so emulate it
/// with a widening multiply and narrowing shift.
#[inline]
unsafe fn vmulhq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    unsafe {
        let lo = vshrn_n_u32(vmull_u16(vget_low_u16(a), vget_low_u16(b)), 16);
        let hi = vshrn_n_u32(vmull_high_u16(a, b), 16);
        vcombine_u16(lo, hi)
    }
}

/// Equivalent of `_mm_maddubs_epi16`: widen u8*s8 to s16, then pairwise-add
/// adjacent lanes to `[a0*b0+a1*b1, ...]`.
#[inline]
unsafe fn vmaddubs_s16(a: uint8x16_t, b: int8x16_t) -> int16x8_t {
    unsafe {
        let prod_lo = vmull_s8(vreinterpret_s8_u8(vget_low_u8(a)), vget_low_s8(b));
        let prod_hi = vmull_high_s8(vreinterpretq_s8_u8(a), b);
        vpaddq_s16(prod_lo, prod_hi)
    }
}

/// Equivalent of `_mm_madd_epi16`: pairwise s16*s16 multiply-add into s32.
#[inline]
unsafe fn vmadd_s32(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    unsafe {
        let prod_lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        let prod_hi = vmull_high_s16(a, b);
        vpaddq_s32(prod_lo, prod_hi)
    }
}

// --- NEON encoder ---

#[target_feature(enable = "neon")]
pub(crate) unsafe fn encode_slice_neon(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    // Shuffle: rearrange 12 input bytes into positions for 6-bit extraction
    let shuffle = unsafe {
        let s: [u8; 16] = [1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10];
        vld1q_u8(s.as_ptr())
    };

    // Multipliers for shift-via-multiply (same constants as AVX2)
    let mul_right_shift: uint16x8_t = unsafe {
        let m: [u16; 8] = [
            0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400,
        ];
        vld1q_u16(m.as_ptr())
    };
    let mul_left_shift: uint16x8_t = unsafe {
        let m: [u16; 8] = [
            0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100,
        ];
        vld1q_u16(m.as_ptr())
    };

    let mask_lo_6bits = vdupq_n_u16(0x003F);
    let mask_hi_6bits = vdupq_n_u16(0x3F00);

    // Character mapping constants
    let offset_base = vdupq_n_s8(65); // 'A'
    let set_25 = vdupq_n_s8(25);
    let delta_lower = vdupq_n_s8(6);
    let set_51 = vdupq_n_u8(51);

    let (sym_plus, sym_slash): (i8, i8) = if config.url_safe {
        (-88, -39)
    } else {
        (-90, -87)
    };

    let lut_offsets = unsafe {
        let l: [i8; 16] = [
            0, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, sym_plus, sym_slash, 0, 0, 0,
        ];
        vld1q_s8(l.as_ptr())
    };

    // Encode one 128-bit vector: 12 input bytes -> 16 output bytes.
    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Shuffle, then extract the 6-bit indices via multiply-shift.
            let v = vqtbl1q_u8($in_vec, shuffle);
            let v_u16 = vreinterpretq_u16_u8(v);
            let lo = vmulq_u16(v_u16, mul_left_shift);
            let hi = unsafe { vmulhq_u16(v_u16, mul_right_shift) };
            let indices_u8 = vreinterpretq_u8_u16(vorrq_u16(
                vandq_u16(lo, mask_hi_6bits),
                vandq_u16(hi, mask_lo_6bits),
            ));

            // Map indices -> characters branchlessly, then fix digits/+//.
            let indices_s8 = vreinterpretq_s8_u8(indices_u8);
            let mut char_val = vaddq_s8(indices_s8, offset_base);
            let gt25 = vcgtq_s8(indices_s8, set_25);
            char_val = vaddq_s8(char_val, vandq_s8(vreinterpretq_s8_u8(gt25), delta_lower));

            let offset_special = vqtbl1q_s8(lut_offsets, vqsubq_u8(indices_u8, set_51));
            vreinterpretq_u8_s8(vaddq_s8(char_val, offset_special))
        }};
    }

    // Quad tier: 48 input bytes -> 64 output.
    let safe_len_48 = len.saturating_sub(4);
    let aligned_len_48 = safe_len_48 - (safe_len_48 % 48);
    let src_end_48 = unsafe { src.add(aligned_len_48) };

    while src < src_end_48 {
        let v0 = encode_vec!(unsafe { vld1q_u8(src) });
        let v1 = encode_vec!(unsafe { vld1q_u8(src.add(12)) });
        let v2 = encode_vec!(unsafe { vld1q_u8(src.add(24)) });
        let v3 = encode_vec!(unsafe { vld1q_u8(src.add(36)) });

        unsafe { vst1q_u8(dst, v0) };
        unsafe { vst1q_u8(dst.add(16), v1) };
        unsafe { vst1q_u8(dst.add(32), v2) };
        unsafe { vst1q_u8(dst.add(48), v3) };

        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
    }

    // Single tier: 12 input bytes -> 16 output.
    let safe_len_12 = len.saturating_sub(4);
    let aligned_len_12 = safe_len_12 - (safe_len_12 % 12);
    let src_end_12 = unsafe { input.as_ptr().add(aligned_len_12) };

    while src < src_end_12 {
        let v = encode_vec!(unsafe { vld1q_u8(src) });
        unsafe { vst1q_u8(dst, v) };

        src = unsafe { src.add(12) };
        dst = unsafe { dst.add(16) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::encode(config, input, src, dst_slice, dst_off) };
}

// --- NEON decoder ---

/// Precomputed NEON decode constants, factored out of [`decode_slice_neon`]
/// only to keep its body under clippy's line-count threshold.
struct DecodeConstantsNeon {
    lut_hi_nibble: int8x16_t,
    sym_62: uint8x16_t,
    sym_63: uint8x16_t,
    delta_62: int8x16_t,
    delta_63: int8x16_t,
    range_0: uint8x16_t,
    range_9_end: uint8x16_t,
    range_a: uint8x16_t,
    range_z: uint8x16_t,
    range_lower_start: uint8x16_t,
    range_lower_end: uint8x16_t,
    pack_l1: int8x16_t,
    pack_l2: int16x8_t,
    pack_shuffle: uint8x16_t,
    mask_hi_nibble: uint8x16_t,
}

#[target_feature(enable = "neon")]
unsafe fn decode_constants_neon(config: &Config) -> DecodeConstantsNeon {
    // High-nibble LUT for character → index offset
    let lut_hi_nibble = unsafe {
        let l: [i8; 16] = [0, 0, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0];
        vld1q_s8(l.as_ptr())
    };

    // Special character handling
    let (char_62, char_63) = if config.url_safe {
        (b'-', b'_')
    } else {
        (b'+', b'/')
    };
    let sym_62 = vdupq_n_u8(char_62);
    let sym_63 = vdupq_n_u8(char_63);

    let (fix_62, fix_63): (i8, i8) = if config.url_safe { (-2, 33) } else { (0, -3) };
    let delta_62 = vdupq_n_s8(fix_62);
    let delta_63 = vdupq_n_s8(fix_63);

    // Range validation constants
    let range_0 = vdupq_n_u8(b'0');
    let range_9_end = vdupq_n_u8(b'9');
    let range_a = vdupq_n_u8(b'A');
    let range_z = vdupq_n_u8(b'Z');
    let range_lower_start = vdupq_n_u8(b'a');
    let range_lower_end = vdupq_n_u8(b'z');

    // Packing constants (same as x86 PACK_L1/L2/SHUFFLE but 128-bit)
    let pack_l1 = unsafe {
        let p: [i8; 16] = [
            0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01,
            0x40, 0x01,
        ];
        vld1q_s8(p.as_ptr())
    };
    let pack_l2 = unsafe {
        let p: [i16; 8] = [
            0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001,
        ];
        vld1q_s16(p.as_ptr())
    };
    let pack_shuffle = unsafe {
        let p: [u8; 16] = [
            2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0xFF, 0xFF, 0xFF, 0xFF,
        ];
        vld1q_u8(p.as_ptr())
    };

    let mask_hi_nibble = vdupq_n_u8(0x0F);

    DecodeConstantsNeon {
        lut_hi_nibble,
        sym_62,
        sym_63,
        delta_62,
        delta_63,
        range_0,
        range_9_end,
        range_a,
        range_z,
        range_lower_start,
        range_lower_end,
        pack_l1,
        pack_l2,
        pack_shuffle,
        mask_hi_nibble,
    }
}

#[target_feature(enable = "neon")]
pub(crate) unsafe fn decode_slice_neon(
    config: &Config,
    input: &[u8],
    dst_slice: &mut [u8],
) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    let DecodeConstantsNeon {
        lut_hi_nibble,
        sym_62,
        sym_63,
        delta_62,
        delta_63,
        range_0,
        range_9_end,
        range_a,
        range_z,
        range_lower_start,
        range_lower_end,
        pack_l1,
        pack_l2,
        pack_shuffle,
        mask_hi_nibble,
    } = unsafe { decode_constants_neon(config) };

    // Validate + decode one 128-bit vector.
    macro_rules! decode_vec {
        ($input_vec:expr) => {{
            // High nibble picks the index offset from the LUT.
            let hi = vandq_u8(vshrq_n_u8($input_vec, 4), mask_hi_nibble);
            let offset = vqtbl1q_s8(lut_hi_nibble, hi);
            let mut indices = vaddq_s8(vreinterpretq_s8_u8($input_vec), offset);

            // Fixups for the two special characters.
            let mask_62 = vceqq_u8($input_vec, sym_62);
            let mask_63 = vceqq_u8($input_vec, sym_63);
            let fix = vorrq_s8(
                vandq_s8(vreinterpretq_s8_u8(mask_62), delta_62),
                vandq_s8(vreinterpretq_s8_u8(mask_63), delta_63),
            );
            indices = vaddq_s8(indices, fix);

            // Valid iff the byte is a symbol, digit, upper, or lower letter.
            let is_sym = vorrq_u8(mask_62, mask_63);
            let is_num = vandq_u8(
                vcgeq_u8($input_vec, range_0),
                vcleq_u8($input_vec, range_9_end),
            );
            let is_upper = vandq_u8(vcgeq_u8($input_vec, range_a), vcleq_u8($input_vec, range_z));
            let is_lower = vandq_u8(
                vcgeq_u8($input_vec, range_lower_start),
                vcleq_u8($input_vec, range_lower_end),
            );
            let is_valid = vorrq_u8(is_sym, vorrq_u8(is_num, vorrq_u8(is_upper, is_lower)));

            // Reduce the per-byte "not valid" mask; nonzero means an error.
            let err_any = vmaxvq_u8(vmvnq_u8(is_valid));

            (vreinterpretq_u8_s8(indices), err_any)
        }};
    }

    // Pack 6-bit indices to bytes: maddubs, madd, then shuffle out 3 bytes per
    // 4-byte lane. Writes 16 bytes; the high 4 are overwritten next iteration.
    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            let m = unsafe { vmaddubs_s16($indices, pack_l1) };
            let p = unsafe { vmadd_s32(m, pack_l2) };
            let out = vqtbl1q_u8(vreinterpretq_u8_s32(p), pack_shuffle);
            unsafe { vst1q_u8($dst_ptr, out) };
        }};
    }

    // Quad tier: 64 input bytes -> 48 output.
    let safe_len_64 = len.saturating_sub(4);
    let aligned_len_64 = safe_len_64 - (safe_len_64 % 64);
    let src_end_64 = unsafe { src.add(aligned_len_64) };

    while src < src_end_64 {
        let v0 = unsafe { vld1q_u8(src) };
        let v1 = unsafe { vld1q_u8(src.add(16)) };
        let v2 = unsafe { vld1q_u8(src.add(32)) };
        let v3 = unsafe { vld1q_u8(src.add(48)) };

        let (i0, e0) = decode_vec!(v0);
        let (i1, e1) = decode_vec!(v1);
        let (i2, e2) = decode_vec!(v2);
        let (i3, e3) = decode_vec!(v3);

        if (e0 | e1 | e2 | e3) != 0 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(i0, dst);
        pack_and_store!(i1, dst.add(12));
        pack_and_store!(i2, dst.add(24));
        pack_and_store!(i3, dst.add(36));

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
    }

    // Single tier: 16 input bytes -> 12 output.
    let safe_len_16 = len.saturating_sub(4);
    let aligned_len_16 = safe_len_16 - (safe_len_16 % 16);
    let src_end_16 = unsafe { input.as_ptr().add(aligned_len_16) };

    while src < src_end_16 {
        let v = unsafe { vld1q_u8(src) };
        let (idx, err) = decode_vec!(v);

        if err != 0 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(idx, dst);

        src = unsafe { src.add(16) };
        dst = unsafe { dst.add(12) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

#[cfg(all(test, miri))]
mod miri_neon_coverage {
    use super::*;
    use crate::simd::testutil::{check_decode, check_encode};
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE};

    fn enc(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_encode(config, oracle, encode_slice_neon, len);
    }
    fn dec(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_decode(config, oracle, decode_slice_neon, len);
    }

    const STD: Config = Config {
        url_safe: false,
        padding: true,
    };

    // Encoder tiers: single-vector is 12 bytes, quad is 48.
    #[test]
    fn miri_neon_encode_scalar_fallback() {
        enc(&STD, &STANDARD, 1); // < 12 -> pure scalar
        enc(&STD, &STANDARD, 11);
    }

    #[test]
    fn miri_neon_encode_single_vector_loop() {
        enc(&STD, &STANDARD, 12); // 1 loop
        enc(&STD, &STANDARD, 24); // 2 loops
        enc(&STD, &STANDARD, 13); // 1 loop + scalar
    }

    #[test]
    fn miri_neon_encode_quad_vector_loop() {
        enc(&STD, &STANDARD, 48); // 1 quad
        enc(&STD, &STANDARD, 96); // 2 quads
        enc(&STD, &STANDARD, 49); // 1 quad + scalar
        enc(&STD, &STANDARD, 60); // 1 quad + 1 single
    }

    #[test]
    fn miri_neon_encode_url_safe() {
        enc(
            &Config {
                url_safe: true,
                padding: true,
            },
            &URL_SAFE,
            50,
        );
    }

    // Decoder tiers: single-vector is 16 bytes, quad is 64.
    #[test]
    fn miri_neon_decode_scalar_fallback() {
        dec(&STD, &STANDARD, 3); // 4 chars
        dec(&STD, &STANDARD, 9); // 12 chars, < 16
    }

    #[test]
    fn miri_neon_decode_single_vector_loop() {
        dec(&STD, &STANDARD, 12); // 1 loop
        dec(&STD, &STANDARD, 24); // 2 loops
        dec(&STD, &STANDARD, 13); // 1 loop + scalar
    }

    #[test]
    fn miri_neon_decode_quad_vector_loop() {
        dec(&STD, &STANDARD, 48); // 1 quad
        dec(&STD, &STANDARD, 96); // 2 quads
        dec(&STD, &STANDARD, 49); // 1 quad + remainder
    }

    #[test]
    fn miri_neon_decode_url_safe() {
        let config = Config {
            url_safe: true,
            padding: false,
        };
        let input = b"-_-_-_-_-_-_-_-_"; // 16 bytes
        let mut dst = [0u8; 16];
        unsafe {
            decode_slice_neon(&config, input, &mut dst).unwrap();
        }
    }

    /// An invalid byte must be caught in every tier and the scalar tail.
    #[test]
    fn miri_neon_decode_error_detection() {
        let mut dst = [0u8; 128];
        for &(len, bad_at, where_) in &[
            (64, 63, "quad tier, last lane"),
            (16, 15, "single tier"),
            (64, 0, "quad tier, first byte"),
            (17, 16, "scalar tail"),
        ] {
            let mut input = vec![b'A'; len];
            input[bad_at] = b'$';
            let res = unsafe { decode_slice_neon(&STD, &input, &mut dst) };
            assert!(res.is_err(), "missed invalid byte in {where_}");
        }
    }

    #[test]
    fn miri_neon_roundtrip_standard() {
        for &len in &[12, 24, 48, 49, 60, 96] {
            enc(&STD, &STANDARD, len);
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_neon_encode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[1, 12, 13, 24, 48, 49] {
            enc(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_neon_decode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[3, 12, 13, 24, 48, 49] {
            dec(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_neon_decode_url_safe_padded() {
        dec(
            &Config {
                url_safe: true,
                padding: true,
            },
            &URL_SAFE,
            50,
        );
    }
}
