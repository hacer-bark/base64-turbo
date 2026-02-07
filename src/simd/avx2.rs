use crate::{Error, Config, scalar};
use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};

// TODO: Rethink encoding and decoding logic. Could squeeze more performance.
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-accelerated implementation of Base64 encoding.
///
/// # Safety
/// This function is **unsafe** and requires the caller to uphold strict memory contracts. 
/// Failure to do so will result in **Undefined Behavior** (buffer overflow).
///
/// * **Output Capacity**: The memory region pointed to by `dst` must have sufficient capacity 
///   to store the encoded output. The required minimum size depends on `config.padding`:
///     * If `padding` is **true**: `input.len().div_ceil(3) * 4`
///     * If `padding` is **false**: `(input.len() * 4).div_ceil(3)`
/// * **Pointer Validity**: `dst` must point to a valid, mutable memory region.
///
/// # Internal Use Only
/// This is a low-level primitive intended for internal use. Callers should prefer the 
/// safe, higher-level APIs (e.g., `Engine::encode`) which automatically handle 
/// buffer allocation and configuration logic.
#[target_feature(enable = "avx2")]
pub unsafe fn encode_slice_avx2(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

    // Shuffle bytes for mul
    let shuffle = _mm256_setr_epi8(
        1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
        1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
    );

    // Masks for bit extraction
    let mask_lo_6bits = _mm256_set1_epi16(0x003F);
    let mask_hi_6bits = _mm256_set1_epi16(0x3F00);

    // Multiplier for shift of bytes.
    let mul_right_shift = _mm256_setr_epi16(
        0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400,
        0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400,
    );
    let mul_left_shift = _mm256_setr_epi16(
        0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100,
        0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100,
    );

    // Mapping logic for letters
    let offset_base = _mm256_set1_epi8(65);
    let set_25 = _mm256_set1_epi8(25);
    let delta_lower = _mm256_set1_epi8(6);
    let set_51 = _mm256_set1_epi8(51);

    // LUT Table for numbers and special chars
    let (sym_plus, sym_slash) = if config.url_safe { (-88, -39) } else { (-90, -87) };
    let lut_offsets = _mm256_setr_epi8(
        0, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, sym_plus, sym_slash, 0, 0, 0,
        0, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, sym_plus, sym_slash, 0, 0, 0
    );

    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Compute 3 bytes => 4 letters
            let v = _mm256_shuffle_epi8($in_vec, shuffle);

            let lo = _mm256_mullo_epi16(v, mul_left_shift);
            let hi = _mm256_mulhi_epu16(v, mul_right_shift);
            let indices = _mm256_or_si256(
                _mm256_and_si256(lo, mask_hi_6bits),
                _mm256_and_si256(hi, mask_lo_6bits),
            );

            // Found char values offsets
            let mut char_val = _mm256_add_epi8(indices, offset_base);
            let offset_lower = _mm256_and_si256(_mm256_cmpgt_epi8(indices, set_25), delta_lower);
            char_val = _mm256_add_epi8(char_val, offset_lower);

            // Found numbers and special symbols offsets
            let offset_special = _mm256_shuffle_epi8(lut_offsets, _mm256_subs_epu8(indices, set_51));

            // Final sum
            _mm256_add_epi8(char_val, offset_special)
        }};
    }

    macro_rules! load_24_bytes {
        ($ptr:expr) => {{
            let c_lo = unsafe { _mm_loadu_si128($ptr as *const __m128i) };
            let c_hi = unsafe { _mm_loadu_si128($ptr.add(12) as *const __m128i) };
            _mm256_inserti128_si256(_mm256_castsi128_si256(c_lo), c_hi, 1)
        }};
    }

    // Process 96 bytes (4 chunks) at a time
    let safe_len_96 = len.saturating_sub(4);
    let aligned_len_96 = safe_len_96 - (safe_len_96 % 96);
    let src_end_96 = unsafe { src.add(aligned_len_96) };

    while src < src_end_96 {
        // Load 4 vectors
        let v0 = load_24_bytes!(src);
        let v1 = load_24_bytes!(src.add(24));
        let v2 = load_24_bytes!(src.add(48));
        let v3 = load_24_bytes!(src.add(72));

        // Process
        let i0 = encode_vec!(v0);
        let i1 = encode_vec!(v1);
        let i2 = encode_vec!(v2);
        let i3 = encode_vec!(v3);

        // Store 4 chunks
        unsafe { _mm256_storeu_si256(dst as *mut __m256i, i0) };
        unsafe { _mm256_storeu_si256(dst.add(32) as *mut __m256i, i1) };
        unsafe { _mm256_storeu_si256(dst.add(64) as *mut __m256i, i2) };
        unsafe { _mm256_storeu_si256(dst.add(96) as *mut __m256i, i3) };

        src = unsafe { src.add(96) };
        dst = unsafe { dst.add(128) };
    }

    // Process remaining 24-byte chunks
    let safe_len_single = len.saturating_sub(4);
    let aligned_len_single = safe_len_single - (safe_len_single % 24);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let v = load_24_bytes!(src);
        let res = encode_vec!(v);
        unsafe { _mm256_storeu_si256(dst as *mut __m256i, res) };

        src = unsafe { src.add(24) };
        dst = unsafe { dst.add(32) };
    }

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) } as usize;
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

/// AVX2-accelerated implementation of Base64 decoding.
///
/// # Safety
/// This function is **unsafe** and requires the caller to uphold strict memory contracts. 
/// Failure to do so will result in **Undefined Behavior** (buffer overflow).
///
/// * **Output Capacity**: The memory region pointed to by `dst` must have sufficient capacity 
///   to store the decoded output. Due to SIMD optimizations performing overlapping writes, 
///   the destination buffer **must** be at least `(input.len() / 4 + 1) * 3` bytes.
/// * **Pointer Validity**: `dst` must point to a valid, mutable memory region.
///
/// # Internal Use Only
/// This is a low-level primitive intended for internal use by the `Engine`.
/// Callers should prefer the safe, higher-level APIs (e.g., `Engine::decode`), which
/// automatically handle buffer sizing via `Engine::estimate_decoded_len`.
#[target_feature(enable = "avx2")]
pub unsafe fn decode_slice_avx2(config: &Config, input: &[u8], mut dst: *mut u8) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst;

    // LUT for offsets based on high nibble (bits 4-7).
    // 0x2_: '+'(43) -> 62 (diff +19).
    // 0x3_: '0'(48) -> 52 (diff +4).
    // 0x4_: 'A'(65) -> 0  (diff -65).
    // 0x5_: 'P'(80) -> 15 (diff -65).
    // 0x6_: 'a'(97) -> 26 (diff -71).
    // 0x7_: 'p'(112)-> 41 (diff -71).
    let lut_hi_nibble = _mm256_setr_epi8(
        0, 0, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0,
    );

    // Range and offsets of special chars
    let (char_62, char_63) = if config.url_safe { (b'-', b'_') } else { (b'+', b'/') };
    let sym_62 = _mm256_set1_epi8(char_62 as i8);
    let sym_63 = _mm256_set1_epi8(char_63 as i8);

    let (fix_62, fix_63) = if config.url_safe { (-2, 33) } else { (0, -3) };
    let delta_62 = _mm256_set1_epi8(fix_62);
    let delta_63 = _mm256_set1_epi8(fix_63);

    // Range Validation Constants
    let range_0 = _mm256_set1_epi8(b'0' as i8);
    let range_9_len = _mm256_set1_epi8(9);

    let range_a = _mm256_set1_epi8(b'A' as i8);
    let range_z_len = _mm256_set1_epi8(25);

    let range_a_low = _mm256_set1_epi8(b'a' as i8);
    let range_z_low_len = _mm256_set1_epi8(25);

    // Packing Constants
    let pack_l1 = unsafe { _mm256_loadu_si256(PACK_L1.as_ptr() as *const __m256i) };
    let pack_l2 = unsafe { _mm256_loadu_si256(PACK_L2.as_ptr() as *const __m256i) };
    let pack_shuffle = unsafe { _mm256_loadu_si256(PACK_SHUFFLE.as_ptr() as *const __m256i) };

    // Masks for nibble extraction
    let mask_hi_nibble = _mm256_set1_epi8(0x0F);

    // Decode & Validate Single Vector
    macro_rules! decode_vec {
        ($input:expr) => {{
            let hi = _mm256_and_si256(_mm256_srli_epi16($input, 4), mask_hi_nibble);
            let offset = _mm256_shuffle_epi8(lut_hi_nibble, hi);
            let mut indices = _mm256_add_epi8($input, offset);

            let mask_62 = _mm256_cmpeq_epi8($input, sym_62);
            let mask_63 = _mm256_cmpeq_epi8($input, sym_63);

            let fix = _mm256_or_si256(
                _mm256_and_si256(mask_62, delta_62),
                _mm256_and_si256(mask_63, delta_63),
            );
            indices = _mm256_add_epi8(indices, fix);

            let is_sym = _mm256_or_si256(mask_62, mask_63);

            let sub_0 = _mm256_subs_epu8(_mm256_sub_epi8($input, range_0), range_9_len);
            let sub_a = _mm256_subs_epu8(_mm256_sub_epi8($input, range_a), range_z_len);
            let sub_a_low = _mm256_subs_epu8(_mm256_sub_epi8($input, range_a_low), range_z_low_len);

            let err = _mm256_andnot_si256(
                is_sym,
                _mm256_and_si256(sub_0, _mm256_and_si256(sub_a, sub_a_low)),
            );

            (indices, err)
        }};
    }

    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            let m = _mm256_maddubs_epi16($indices, pack_l1);
            let p = _mm256_madd_epi16(m, pack_l2);
            let out = _mm256_shuffle_epi8(p, pack_shuffle);

            let lane_0 = _mm256_castsi256_si128(out);
            unsafe { _mm_storeu_si128($dst_ptr as *mut __m128i, lane_0) };
            let lane_1 = _mm256_extracti128_si256(out, 1);
            unsafe { _mm_storeu_si128($dst_ptr.add(12) as *mut __m128i, lane_1) };
        }};
    }

    // Process 128 bytes (4 chunks) at a time
    let safe_len_128 = len.saturating_sub(4);
    let aligned_len_128 = safe_len_128 - (safe_len_128 % 128);
    let src_end_128 = unsafe { src.add(aligned_len_128) };

    while src < src_end_128 {
        // Load 4 vectors
        let v0 = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let v1 = unsafe { _mm256_loadu_si256(src.add(32) as *const __m256i) };
        let v2 = unsafe { _mm256_loadu_si256(src.add(64) as *const __m256i) };
        let v3 = unsafe { _mm256_loadu_si256(src.add(96) as *const __m256i) };

        // Process
        let (i0, e0) = decode_vec!(v0);
        let (i1, e1) = decode_vec!(v1);
        let (i2, e2) = decode_vec!(v2);
        let (i3, e3) = decode_vec!(v3);

        // Check Errors
        let err_any = _mm256_or_si256(
            _mm256_or_si256(e0, e1), 
            _mm256_or_si256(e2, e3)
        );

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
        let v = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let (idx, err) = decode_vec!(v);

        if _mm256_testz_si256(err, err) != 1 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(idx, dst);

        src = unsafe { src.add(32) };
        dst = unsafe { dst.add(24) };
    }

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) } as usize;
    if processed_len < len {
        dst = unsafe { dst.add(scalar::decode_slice_unsafe(config, &input[processed_len..], dst)?) };
    }

    Ok(unsafe { dst.offset_from(dst_start) } as usize)
}

#[cfg(kani)]
mod kani_verification_avx2 {
    use super::*;
    use crate::{Config, STANDARD as TURBO_STANDARD, STANDARD_NO_PAD as TURBO_STANDARD_NO_PAD};
    use core::mem::transmute;

    // Magic number: 36
    // It handles one small loop unroll and Scalar tail fallback.
    // Cuz we're using same macros in Small loop and Big loop we prove math for both of them.
    // For edge cages which might happen at big loop unrolling refer to Miri implementation.
    // 
    // Note: if we can prove what one Small loop for any inputs, it automatically proves math
    // for inputs of any size. From zero to infinity, it won't UB or Panic.
    const INPUT_LEN: usize = 36;

    // --- HELPERS AND STUBS ---

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding { TURBO_STANDARD.encoded_len(len) } else { TURBO_STANDARD_NO_PAD.encoded_len(len) }
    }

    // STUB: _mm256_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_epi8
    #[allow(dead_code)]
    unsafe fn mm256_shuffle_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
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

    // STUB: _mm256_mulhi_epu16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mulhi_epu16
    #[allow(dead_code)]
    unsafe fn mm256_mulhi_epu16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u16; 16] = unsafe { transmute(a) };
        let b: [u16; 16] = unsafe { transmute(b) };
        let mut dst = [0u16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j;

            // tmp[31:0] := a[i+15:i] * b[i+15:i]
            let op1 = a[i] as u32;
            let op2 = b[i] as u32;
            let tmp = op1 * op2;

            // dst[i+15:i] := tmp[31:16]
            dst[i] = (tmp >> 16) as u16;
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_mullo_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mullo_epi16
    #[allow(dead_code)]
    unsafe fn mm256_mullo_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [i16; 16] = unsafe { transmute(a) };
        let b: [i16; 16] = unsafe { transmute(b) };
        let mut dst = [0i16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j;

            // tmp[31:0] := SignExtend32(a[i+15:i]) * SignExtend32(b[i+15:i])
            let op1 = a[i] as i32;
            let op2 = b[i] as i32;
            let tmp: i32 = op1.wrapping_mul(op2);

            // dst[i+15:i] := tmp[15:0]
            dst[j] = tmp as i16;
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_add_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_add_epi8
    #[allow(dead_code)]
    unsafe fn mm256_add_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [u8; 32] = unsafe { transmute(b) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := j*8
            let i = j;

	        // dst[i+7:i] := a[i+7:i] + b[i+7:i]
            dst[i] = a[i].wrapping_add(b[i]);
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_subs_epu8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_subs_epu8
    #[allow(dead_code)]
    unsafe fn mm256_subs_epu8_stub(a: __m256i, b: __m256i) -> __m256i {
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
    unsafe fn mm256_testz_si256_stub(a: __m256i, b: __m256i) -> i32 {
        let a: [u64; 4] = unsafe { transmute(a) };
        let b: [u64; 4] = unsafe { transmute(b) };
        let zf: i32;
        let _cf: i32;

        // Perform 256 bit AND
        let res_and = [
            a[0] & b[0],
            a[1] & b[1],
            a[2] & b[2],
            a[3] & b[3],
        ];

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
        if res_not_and[0] == 0 && res_not_and[1] == 0 && res_not_and[2] == 0 && res_not_and[3] == 0 {
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
    unsafe fn mm256_maddubs_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [i8; 32] = unsafe { transmute(b) };
        let mut dst = [0i16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j * 2;

            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i+1] as i16) * (b[i+1] as i16)).saturating_add((a[i] as i16) * (b[i] as i16));
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_madd_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_madd_epi16
    #[allow(dead_code)]
    unsafe fn mm256_madd_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [i16; 16] = unsafe { transmute(a) };
        let b: [i16; 16] = unsafe { transmute(b) };
        let mut dst = [0i32; 8];

        // FOR j := 0 to 7
        for j in 0..8 {
            // i := j*32
            let i = j * 2;

            // dst[i+31:i] := SignExtend32(a[i+31:i+16]*b[i+31:i+16]) + SignExtend32(a[i+15:i]*b[i+15:i])
            dst[j] = (a[i+1] as i32).wrapping_mul(b[i+1] as i32).wrapping_add((a[i] as i32).wrapping_mul(b[i] as i32));
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_sub_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_sub_epi8
    #[allow(dead_code)]
    unsafe fn mm256_sub_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [u8; 32] = unsafe { transmute(b) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := j*8
            let i = j;

            // dst[i+7:i] := a[i+7:i] - b[i+7:i]
            dst[i] = a[i].wrapping_sub(b[i]);
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // -- REAL LOGIC --- 

    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_mulhi_epu16, mm256_mulhi_epu16_stub)]
    #[kani::stub(_mm256_mullo_epi16, mm256_mullo_epi16_stub)]
    #[kani::stub(_mm256_add_epi8, mm256_add_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, mm256_madd_epi16_stub)]
    fn check_roundtrip_safety() {
        // Symbolic Config
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };

        // Symbolic Input
        let input: [u8; INPUT_LEN] = kani::any();

        // Setup Buffers
        let enc_len = encoded_size(INPUT_LEN, config.padding);
        let mut enc_buf = [0u8; 512];
        let mut dec_buf = [0u8; 512];

        unsafe {
            // Encode
            encode_slice_avx2(&config, &input, enc_buf.as_mut_ptr());

            // Decode
            let src_slice = &enc_buf[..enc_len];
            let written = decode_slice_avx2(&config, src_slice, dec_buf.as_mut_ptr()).expect("Decoder failed");

            // Verification
            assert_eq!(&dec_buf[..written], &input, "AVX2 Roundtrip Failed");
        }
    }

    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_mulhi_epu16, mm256_mulhi_epu16_stub)]
    #[kani::stub(_mm256_mullo_epi16, mm256_mullo_epi16_stub)]
    #[kani::stub(_mm256_add_epi8, mm256_add_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, mm256_madd_epi16_stub)]
    #[kani::stub(_mm256_sub_epi8, mm256_sub_epi8_stub)]
    fn check_decoder_robustness() {
        // Symbolic Config
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };

        // Symbolic Input (Random Garbage)
        let input: [u8; INPUT_LEN] = kani::any();

        // Setup Buffer
        let mut dec_buf = [0u8; 64];

        unsafe {
            // We verify what function NEVER panics/crashes
            let _ = decode_slice_avx2(&config, &input, dec_buf.as_mut_ptr());
        }
    }
}

#[cfg(all(test, miri))]
mod miri_avx2_coverage {
    use super::*;
    use rand::{Rng, rng};
    use base64::{engine::general_purpose::{STANDARD, URL_SAFE}, Engine};

    // --- Mock Infrastructure for Miri ---
    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    /// Helper to verify AVX2 encoding against the 'base64' crate oracle
    fn verify_encode_avx2(config: &Config, oracle: &impl Engine, input_len: usize) {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip on machines without AVX2 support (or Miri without flags)
        }

        let input = random_bytes(input_len);
        let expected = oracle.encode(&input);

        // Allocate buffer (Base64 is ~4/3 larger)
        let mut dst = vec![0u8; expected.len() * 2]; // Safety margin

        unsafe { encode_slice_avx2(config, &input, dst.as_mut_ptr()); }

        // Verify prefix matches expected
        let result = &dst[..expected.len()];
        assert_eq!(std::str::from_utf8(result).unwrap(), expected, "Encode len {}", input_len);
    }

    /// Helper to verify AVX2 decoding against the 'base64' crate oracle
    fn verify_decode_avx2(config: &Config, oracle: &impl Engine, original_len: usize) {
        if !is_x86_feature_detected!("avx2") { return; }

        // 1. Generate valid Base64 via oracle
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();

        // 2. Run AVX2 Decoder
        let mut dst = vec![0u8; original_len + 64]; // Safety margin

        let len = unsafe {
            decode_slice_avx2(config, encoded_bytes, dst.as_mut_ptr()).expect("Valid input failed to decode")
        };

        // 3. Verify
        assert_eq!(&dst[..len], &input_bytes, "Decode len {}", original_len);
    }

    // ----------------------------------------------------------------------
    // 1. Encoder Coverage Tests
    // ----------------------------------------------------------------------

    #[test]
    fn miri_encode_scalar_fallback() {
        let config = Config { url_safe: false, padding: true };
        // Test < 24 bytes (Hits scalar fallback immediately)
        verify_encode_avx2(&config, &STANDARD, 1);
        verify_encode_avx2(&config, &STANDARD, 23);
    }

    #[test]
    fn miri_encode_single_vector_loop() {
        let config = Config { url_safe: false, padding: true };
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
        let config = Config { url_safe: false, padding: true };
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
    fn miri_encode_url_safe() {
        // Verify the lookup table switching logic
        let config = Config { url_safe: true, padding: true };
        verify_encode_avx2(&config, &URL_SAFE, 50);
    }

    // ----------------------------------------------------------------------
    // 2. Decoder Coverage Tests
    // ----------------------------------------------------------------------

    #[test]
    fn miri_decode_scalar_fallback() {
        let config = Config { url_safe: false, padding: true };
        // Your code falls back for < 32 bytes
        // Note: Base64 expands 3 bytes -> 4 chars.
        // Input length 4 chars -> 3 bytes output.
        verify_decode_avx2(&config, &STANDARD, 3); // 4 chars
        verify_decode_avx2(&config, &STANDARD, 21); // 28 chars (< 32)
    }

    #[test]
    fn miri_decode_single_vector_loop() {
        let config = Config { url_safe: false, padding: true };
        // Your code processes 32-byte chunks.
        // 32 bytes of Base64 = 24 bytes of decoded data.
        verify_decode_avx2(&config, &STANDARD, 24); // Exactly 32 bytes input
        verify_decode_avx2(&config, &STANDARD, 48); // Exactly 64 bytes input (2 loops)
        verify_decode_avx2(&config, &STANDARD, 25); // 32 bytes + scalar remainder
    }

    #[test]
    fn miri_decode_quad_vector_loop() {
        let config = Config { url_safe: false, padding: true };
        // Your code processes 128-byte chunks (4 * 32).
        // 128 bytes input = 96 bytes decoded.
        verify_decode_avx2(&config, &STANDARD, 96); // Exactly 128 bytes input
        verify_decode_avx2(&config, &STANDARD, 192); // Exactly 256 bytes input (2 loops)
        verify_decode_avx2(&config, &STANDARD, 97); // 1 quad + remainder
    }

    #[test]
    fn miri_decode_url_safe() {
        // Verify '-' and '_' handling in the SIMD path
        let config = Config { url_safe: true, padding: false };
        
        // Construct specific input with URL safe chars
        // 0x3F (?) is usually '/', in URL safe it is '_'
        // 0x3E (>) is usually '+', in URL safe it is '-'
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_"; // 32 bytes
        let mut dst = [0u8; 32];
        
        unsafe { decode_slice_avx2(&config, input, dst.as_mut_ptr()).unwrap(); }
    }

    // ----------------------------------------------------------------------
    // 3. Error Logic Coverage
    // ----------------------------------------------------------------------

    #[test]
    fn miri_decode_error_detection() {
        if !is_x86_feature_detected!("avx2") { return; }
        
        let config = Config { url_safe: false, padding: true };
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
    }
}
