use crate::{Config, Error, scalar};

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

// ======================================================================
// Helper: unsigned multiply-high for u16x8
// NEON lacks a direct mulhi_u16; we emulate via widening multiply.
// ======================================================================

#[inline]
unsafe fn vmulhq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    unsafe {
        let lo = vshrn_n_u32(vmull_u16(vget_low_u16(a), vget_low_u16(b)), 16);
        let hi = vshrn_n_u32(vmull_high_u16(a, b), 16);
        vcombine_u16(lo, hi)
    }
}

// ======================================================================
// Helper: maddubs equivalent (unsigned × signed bytes, pairwise add → i16)
// Equivalent to _mm_maddubs_epi16(a, b)
// result[k] = saturate_i16(a[2k]*b[2k] + a[2k+1]*b[2k+1])
// ======================================================================

#[inline]
unsafe fn vmaddubs_s16(a: uint8x16_t, b: int8x16_t) -> int16x8_t {
    unsafe {
        // Widening multiply: u8 * s8 → s16 (low and high halves)
        let prod_lo = vmull_s8(vreinterpret_s8_u8(vget_low_u8(a)), vget_low_s8(b));
        let prod_hi = vmull_high_s8(vreinterpretq_s8_u8(a), b);

        // Pairwise add adjacent s16 within each vector → s32, then narrow back
        // prod_lo = [a0*b0, a1*b1, a2*b2, a3*b3, a4*b4, a5*b5, a6*b6, a7*b7]
        // We need [a0*b0+a1*b1, a2*b2+a3*b3, a4*b4+a5*b5, a6*b6+a7*b7]
        let sum_lo = vpaddq_s16(prod_lo, prod_hi);
        // vpaddq gives [lo_pairs..., hi_pairs...] in lane order
        // prod_lo pairs: [p0+p1, p2+p3, p4+p5, p6+p7] in low 4 lanes
        // prod_hi pairs: [p8+p9, p10+p11, p12+p13, p14+p15] in high 4 lanes
        sum_lo
    }
}

// ======================================================================
// Helper: madd equivalent (signed i16 pairwise multiply-add → i32)
// Equivalent to _mm_madd_epi16(a, b)
// result[k] = a[2k]*b[2k] + a[2k+1]*b[2k+1]
// ======================================================================

#[inline]
unsafe fn vmadd_s32(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    unsafe {
        let prod_lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        let prod_hi = vmull_high_s16(a, b);
        vpaddq_s32(prod_lo, prod_hi)
    }
}

// ======================================================================
// NEON Encoder
// ======================================================================

#[target_feature(enable = "neon")]
pub(crate) unsafe fn encode_slice_neon(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

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

    // Encode one 128-bit vector (12 input bytes → 16 output bytes)
    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Byte shuffle: rearrange for 6-bit extraction
            let v = vqtbl1q_u8($in_vec, shuffle);
            let v_u16 = vreinterpretq_u16_u8(v);

            // Multiply-shift to extract 6-bit indices
            let lo = vmulq_u16(v_u16, mul_left_shift);
            let hi = unsafe { vmulhq_u16(v_u16, mul_right_shift) };
            let indices_u8 = vreinterpretq_u8_u16(vorrq_u16(
                vandq_u16(lo, mask_hi_6bits),
                vandq_u16(hi, mask_lo_6bits),
            ));

            // Map indices → Base64 characters (branchless)
            let indices_s8 = vreinterpretq_s8_u8(indices_u8);
            let mut char_val = vaddq_s8(indices_s8, offset_base);
            let gt25 = vcgtq_s8(indices_s8, set_25);
            char_val = vaddq_s8(char_val, vandq_s8(vreinterpretq_s8_u8(gt25), delta_lower));

            // Special chars (digits, +, /)
            let offset_special = vqtbl1q_s8(lut_offsets, vqsubq_u8(indices_u8, set_51));
            vreinterpretq_u8_s8(vaddq_s8(char_val, offset_special))
        }};
    }

    // --- Quad-unrolled loop: 48 input → 64 output ---
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

    // --- Single-vector loop: 12 input → 16 output ---
    let safe_len_12 = len.saturating_sub(4);
    let aligned_len_12 = safe_len_12 - (safe_len_12 % 12);
    let src_end_12 = unsafe { input.as_ptr().add(aligned_len_12) };

    while src < src_end_12 {
        let v = encode_vec!(unsafe { vld1q_u8(src) });
        unsafe { vst1q_u8(dst, v) };

        src = unsafe { src.add(12) };
        dst = unsafe { dst.add(16) };
    }

    // --- Scalar fallback for tail ---
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

// ======================================================================
// NEON Decoder
// ======================================================================

/// Precomputed NEON vector constants shared by every lane processed in
/// [`decode_slice_neon`]. Factored out purely to keep that function's body
/// under clippy's line-count threshold; the values themselves are unchanged.
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
    mut dst: *mut u8,
) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst;

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

    // Decode & validate one 128-bit vector
    macro_rules! decode_vec {
        ($input_vec:expr) => {{
            // High nibble → LUT offset
            let hi = vandq_u8(vshrq_n_u8($input_vec, 4), mask_hi_nibble);
            let offset = vqtbl1q_s8(lut_hi_nibble, hi);
            let mut indices = vaddq_s8(vreinterpretq_s8_u8($input_vec), offset);

            // Fix special characters
            let mask_62 = vceqq_u8($input_vec, sym_62);
            let mask_63 = vceqq_u8($input_vec, sym_63);
            let fix = vorrq_s8(
                vandq_s8(vreinterpretq_s8_u8(mask_62), delta_62),
                vandq_s8(vreinterpretq_s8_u8(mask_63), delta_63),
            );
            indices = vaddq_s8(indices, fix);

            // Validate: check that every byte is in a valid range
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

            // err_any != 0 means there are invalid bytes
            let err = vmvnq_u8(is_valid); // NOT valid = error
            let err_any = vmaxvq_u8(err);

            (vreinterpretq_u8_s8(indices), err_any)
        }};
    }

    // Pack 6-bit indices → bytes and store 12 bytes
    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            // Step 1: maddubs — pair adjacent 6-bit values
            let m = unsafe { vmaddubs_s16($indices, pack_l1) };
            // Step 2: madd — pair adjacent 12-bit values → 24-bit in i32
            let p = unsafe { vmadd_s32(m, pack_l2) };
            // Step 3: shuffle to extract 3 bytes from each 4-byte lane
            let out = vqtbl1q_u8(vreinterpretq_u8_s32(p), pack_shuffle);
            // Store 12 bytes (write 16, last 4 are garbage overwritten next iter)
            unsafe { vst1q_u8($dst_ptr, out) };
        }};
    }

    // --- Quad-unrolled loop: 64 input → 48 output ---
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

    // --- Single-vector loop: 16 input → 12 output ---
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

    unsafe { decode_scalar_tail(config, input, src, dst, dst_start) }
}

/// Decodes any bytes left over after the vectorized main/tail loops via the
/// scalar fallback, then returns the total number of bytes written.
///
/// # Safety
/// `src` must point within `input`, and `dst`/`dst_start` must satisfy the
/// same contract as [`scalar::decode_slice_unsafe`].
unsafe fn decode_scalar_tail(
    config: &Config,
    input: &[u8],
    src: *const u8,
    mut dst: *mut u8,
    dst_start: *mut u8,
) -> Result<usize, Error> {
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < input.len() {
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

#[cfg(all(test, miri))]
mod miri_neon_coverage {
    use super::*;
    use base64::{
        Engine,
        engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE},
    };
    use rand::{RngExt, rng};

    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    fn verify_encode_neon(config: &Config, oracle: &impl Engine, input_len: usize) {
        let input = random_bytes(input_len);
        let expected = oracle.encode(&input);
        let mut dst = vec![0u8; expected.len() * 2];

        unsafe {
            encode_slice_neon(config, &input, dst.as_mut_ptr());
        }

        let result = &dst[..expected.len()];
        assert_eq!(
            std::str::from_utf8(result).unwrap(),
            expected,
            "Encode len {input_len}"
        );
    }

    fn verify_decode_neon(config: &Config, oracle: &impl Engine, original_len: usize) {
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();
        let mut dst = vec![0u8; original_len + 64];

        let len = unsafe {
            decode_slice_neon(config, encoded_bytes, dst.as_mut_ptr())
                .expect("Valid input failed to decode")
        };

        assert_eq!(&dst[..len], &input_bytes, "Decode len {original_len}");
    }

    // ----------------------------------------------------------------------
    // 1. Encoder Coverage Tests
    // ----------------------------------------------------------------------

    #[test]
    fn miri_neon_encode_scalar_fallback() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // < 12 bytes → pure scalar
        verify_encode_neon(&config, &STANDARD, 1);
        verify_encode_neon(&config, &STANDARD, 11);
    }

    #[test]
    fn miri_neon_encode_single_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        verify_encode_neon(&config, &STANDARD, 12); // Exactly 1 loop
        verify_encode_neon(&config, &STANDARD, 24); // Exactly 2 loops
        verify_encode_neon(&config, &STANDARD, 13); // 1 loop + 1 byte scalar
    }

    #[test]
    fn miri_neon_encode_quad_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        verify_encode_neon(&config, &STANDARD, 48); // Exactly 1 quad loop
        verify_encode_neon(&config, &STANDARD, 96); // Exactly 2 quad loops
        verify_encode_neon(&config, &STANDARD, 49); // 1 quad + scalar
        verify_encode_neon(&config, &STANDARD, 60); // 1 quad + 1 single
    }

    #[test]
    fn miri_neon_encode_url_safe() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        verify_encode_neon(&config, &URL_SAFE, 50);
    }

    // ----------------------------------------------------------------------
    // 2. Decoder Coverage Tests
    // ----------------------------------------------------------------------

    #[test]
    fn miri_neon_decode_scalar_fallback() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        verify_decode_neon(&config, &STANDARD, 3); // 4 chars
        verify_decode_neon(&config, &STANDARD, 9); // 12 chars (< 16)
    }

    #[test]
    fn miri_neon_decode_single_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        verify_decode_neon(&config, &STANDARD, 12); // 16 chars → 1 loop
        verify_decode_neon(&config, &STANDARD, 24); // 32 chars → 2 loops
        verify_decode_neon(&config, &STANDARD, 13); // 16 chars + scalar
    }

    #[test]
    fn miri_neon_decode_quad_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        verify_decode_neon(&config, &STANDARD, 48); // 64 chars → 1 quad loop
        verify_decode_neon(&config, &STANDARD, 96); // 128 chars → 2 quad loops
        verify_decode_neon(&config, &STANDARD, 49); // 1 quad + remainder
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
            decode_slice_neon(&config, input, dst.as_mut_ptr()).unwrap();
        }
    }

    // ----------------------------------------------------------------------
    // 3. Error Logic Coverage
    // ----------------------------------------------------------------------

    #[test]
    fn miri_neon_decode_error_detection() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        let mut dst = [0u8; 128];

        // Error in Quad Loop
        let mut bad_input_64 = vec![b'A'; 64];
        bad_input_64[63] = b'$';
        let res = unsafe { decode_slice_neon(&config, &bad_input_64, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Quad Loop");

        // Error in Single Loop
        let mut bad_input_16 = vec![b'A'; 16];
        bad_input_16[15] = b'?';
        let res = unsafe { decode_slice_neon(&config, &bad_input_16, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Single Loop");

        // Error in Quad Loop (first byte)
        let mut bad_input_64_first = vec![b'A'; 64];
        bad_input_64_first[0] = b'$';
        let res = unsafe { decode_slice_neon(&config, &bad_input_64_first, dst.as_mut_ptr()) };
        assert!(
            res.is_err(),
            "Failed to catch error in Quad Loop first byte"
        );

        // Error in Scalar Fallback
        let mut bad_input_17 = vec![b'A'; 17];
        bad_input_17[16] = b'?';
        let res = unsafe { decode_slice_neon(&config, &bad_input_17, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Scalar Fallback");
    }

    // ----------------------------------------------------------------------
    // 4. Roundtrip & Config Coverage
    // ----------------------------------------------------------------------

    #[test]
    fn miri_neon_roundtrip_standard() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for &len in &[12, 24, 48, 49, 60, 96] {
            let input = random_bytes(len);
            let expected = STANDARD.encode(&input);
            let mut enc = vec![0u8; expected.len() * 2];
            unsafe {
                encode_slice_neon(&config, &input, enc.as_mut_ptr());
            }
            let encoded = &enc[..expected.len()];
            assert_eq!(std::str::from_utf8(encoded).unwrap(), expected);

            let mut dec = vec![0u8; len + 64];
            let dec_len = unsafe { decode_slice_neon(&config, encoded, dec.as_mut_ptr()).unwrap() };
            assert_eq!(&dec[..dec_len], &input, "Roundtrip len {len}");
        }
    }

    #[test]
    fn miri_neon_encode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[1, 12, 13, 24, 48, 49] {
            verify_encode_neon(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_neon_decode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[3, 12, 13, 24, 48, 49] {
            let input_bytes = random_bytes(len);
            let encoded = STANDARD_NO_PAD.encode(&input_bytes);
            let mut dst = vec![0u8; len + 64];
            let dec_len = unsafe {
                decode_slice_neon(&config, encoded.as_bytes(), dst.as_mut_ptr()).unwrap()
            };
            assert_eq!(&dst[..dec_len], &input_bytes, "No-pad decode len {len}");
        }
    }

    #[test]
    fn miri_neon_decode_url_safe_padded() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        verify_decode_neon(&config, &URL_SAFE, 50);
    }
}
