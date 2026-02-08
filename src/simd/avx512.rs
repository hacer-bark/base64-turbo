use crate::{Error, Config, scalar};
use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};

// TODO: Consider add special AVX-512VBMI support
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn encode_slice_avx512(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

    // Shuffle bytes for mul
    let shuffle = _mm512_broadcast_i32x4( _mm_setr_epi8(1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10));

    // Masks and multiplier
    let mask_lo_6bits = _mm512_set1_epi16(0x003F);
    let mask_hi_6bits = _mm512_set1_epi16(0x3F00);
    let mul_right_shift = _mm512_set1_epi32(0x04000040);
    let mul_left_shift  = _mm512_set1_epi32(0x01000010);

    // Character mapping
    let offset_base = _mm512_set1_epi8(65);
    let set_25 = _mm512_set1_epi8(25);
    let delta_lower = _mm512_set1_epi8(6); 
    let set_51 = _mm512_set1_epi8(51);

    // LUT Table for numbers and special chars
    let (sym_plus, sym_slash) = if config.url_safe { (-88, -39) } else { (-90, -87) };
    let lut_offsets = _mm512_broadcast_i32x4(_mm_setr_epi8(0, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, sym_plus, sym_slash, 0, 0, 0));

    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Compute 3 bytes => 4 letters
            let v = _mm512_shuffle_epi8($in_vec, shuffle);

            let lo = _mm512_mullo_epi16(v, mul_left_shift);
            let hi = _mm512_mulhi_epu16(v, mul_right_shift);
            let indices = _mm512_or_si512(
                _mm512_and_si512(hi, mask_lo_6bits),
                _mm512_and_si512(lo, mask_hi_6bits),
            );

            // Compute letters offsets
            let mut char_val = _mm512_add_epi8(indices, offset_base);
            let m_gt25 = _mm512_cmpgt_epi8_mask(indices, set_25);
            char_val = _mm512_mask_add_epi8(char_val, m_gt25, char_val, delta_lower);

            // Compute special chars offset
            let offset_special = _mm512_shuffle_epi8(lut_offsets, _mm512_subs_epu8(indices, set_51));
            
            _mm512_add_epi8(char_val, offset_special)
        }};
    }

    macro_rules! load_48_bytes {
        ($ptr:expr) => {{
            let p = $ptr;
            let v0 = unsafe { _mm_loadu_si128(p as *const _) }; 
            let v1 = unsafe {_mm_loadu_si128(p.add(12) as *const _) };
            let v2 = unsafe { _mm_loadu_si128(p.add(24) as *const _) };
            let v3 = unsafe { _mm_loadu_si128(p.add(36) as *const _) };

            // Combine into ZMM
            let z = _mm512_castsi128_si512(v0);
            let z = _mm512_inserti32x4(z, v1, 1);
            let z = _mm512_inserti32x4(z, v2, 2);
            _mm512_inserti32x4(z, v3, 3)
        }};
    }

    // Process 192 bytes (4 chunks) at a time
    let safe_len_192 = len.saturating_sub(4);
    let aligned_len_192 = safe_len_192 - (safe_len_192 % 192);
    let src_end_192 = unsafe { src.add(aligned_len_192) };

    while src < src_end_192 {
        // Load 4 vectors
        let v0 = load_48_bytes!(src);
        let v1 = load_48_bytes!(unsafe { src.add(48) });
        let v2 = load_48_bytes!(unsafe { src.add(96) });
        let v3 = load_48_bytes!(unsafe { src.add(144) });

        // Process
        let i0 = encode_vec!(v0);
        let i1 = encode_vec!(v1);
        let i2 = encode_vec!(v2);
        let i3 = encode_vec!(v3);

        // Store results
        unsafe { _mm512_storeu_si512(dst as *mut _, i0) };
        unsafe { _mm512_storeu_si512(dst.add(64) as *mut _, i1) };
        unsafe { _mm512_storeu_si512(dst.add(128) as *mut _, i2) };
        unsafe { _mm512_storeu_si512(dst.add(192) as *mut _, i3) };

        src = unsafe { src.add(192) };
        dst = unsafe { dst.add(256) };
    }

    // Process remaining 48-byte chunks
    let safe_len_single = len.saturating_sub(4);
    let aligned_len_single = safe_len_single - (safe_len_single % 48);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let v = load_48_bytes!(src);
        let res = encode_vec!(v);
        unsafe { _mm512_storeu_si512(dst as *mut _, res) };

        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
    }

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) } as usize;
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn decode_slice_avx512(config: &Config, input: &[u8], mut dst: *mut u8) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst;

    // LUT for offsets based on high nibble (bits 4-7). 
    let lut_hi_nibble = _mm512_broadcast_i32x4(_mm_setr_epi8(0, 0, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0));

    // Range and offsets of special chars
    let (char_62, char_63) = if config.url_safe { (b'-', b'_') } else { (b'+', b'/') };
    let sym_62 = _mm512_set1_epi8(char_62 as i8);
    let sym_63 = _mm512_set1_epi8(char_63 as i8);

    let (fix_62, fix_63) = if config.url_safe { (-2, 33) } else { (0, -3) };
    let delta_62 = _mm512_set1_epi8(fix_62);
    let delta_63 = _mm512_set1_epi8(fix_63);

    // Range Validation Constants
    let range_0 = _mm512_set1_epi8(b'0' as i8);
    let range_9_len = _mm512_set1_epi8(9);

    let range_a = _mm512_set1_epi8(b'A' as i8);
    let range_z_len = _mm512_set1_epi8(25);

    let range_a_low = _mm512_set1_epi8(b'a' as i8);
    let range_z_low_len = _mm512_set1_epi8(25);

    // Packing Constants
    let pack_l1 = unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_L1.as_ptr() as *const __m128i)) };
    let pack_l2 = unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_L2.as_ptr() as *const __m128i)) };
    let pack_shuffle = unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_SHUFFLE.as_ptr() as *const __m128i)) };

    // Masks for nibble extraction and zeros
    let mask_hi_nibble = _mm512_set1_epi8(0x0F);
    let zeros = _mm512_setzero_si512();

    // Decode & Validate Single Vector
    // TODO: Think how we can compute `err` as mask
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

            let sub_0 = _mm512_subs_epu8(_mm512_sub_epi8($input, range_0), range_9_len);
            let sub_a = _mm512_subs_epu8(_mm512_sub_epi8($input, range_a), range_z_len);
            let sub_a_low = _mm512_subs_epu8(_mm512_sub_epi8($input, range_a_low), range_z_low_len);

            let is_char = _mm512_and_si512(sub_0, _mm512_and_si512(sub_a, sub_a_low));

            let err = _mm512_mask_blend_epi8(is_sym, is_char, zeros);

            (indices, err)
        }};
    }

    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            let m = _mm512_maddubs_epi16($indices, pack_l1);
            let p = _mm512_madd_epi16(m, pack_l2);
            let out = _mm512_shuffle_epi8(p, pack_shuffle);

            let lane0 = _mm512_castsi512_si128(out);
            unsafe { _mm_storeu_si128($dst_ptr as *mut __m128i, lane0) };
            let lane1 = _mm512_extracti32x4_epi32(out, 1);
            unsafe { _mm_storeu_si128($dst_ptr.add(12) as *mut __m128i, lane1) };
            let lane2 = _mm512_extracti32x4_epi32(out, 2);
            unsafe { _mm_storeu_si128($dst_ptr.add(24) as *mut __m128i, lane2) };
            let lane3 = _mm512_extracti32x4_epi32(out, 3);
            unsafe { _mm_storeu_si128($dst_ptr.add(36) as *mut __m128i, lane3) };
        }};
    }

    // Process 128 bytes (4 chunks) at a time
    let safe_len_256 = len.saturating_sub(4);
    let aligned_len_256 = safe_len_256 - (safe_len_256 % 256);
    let src_end_256 = unsafe { src.add(aligned_len_256) };

    while src < src_end_256 {
        // Load 4 vectors
        let v0 = unsafe { _mm512_loadu_si512(src as *const __m512i) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(64) as *const __m512i) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(128) as *const __m512i) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(192) as *const __m512i) };

        // Process
        let (i0, e0) = decode_vec!(v0);
        let (i1, e1) = decode_vec!(v1);
        let (i2, e2) = decode_vec!(v2);
        let (i3, e3) = decode_vec!(v3);

        // Check errors
        let m0 = _mm512_test_epi8_mask(e0, e0);
        let m1 = _mm512_test_epi8_mask(e1, e1);
        let m2 = _mm512_test_epi8_mask(e2, e2);
        let m3 = _mm512_test_epi8_mask(e3, e3);

        if (m0 | m1 | m2 | m3) != 0 {
            return Err(Error::InvalidCharacter);
        }

        // Store 4 chunks
        pack_and_store!(i0, dst);
        pack_and_store!(i1, dst.add(48));
        pack_and_store!(i2, dst.add(96));
        pack_and_store!(i3, dst.add(144));

        src = unsafe { src.add(256) };
        dst = unsafe { dst.add(192) };
    }

    // Process remaining 32-byte chunks
    let safe_len_64 = len.saturating_sub(4);
    let aligned_len_64 = safe_len_64 - (safe_len_64 % 64);
    let src_end_64 = unsafe { input.as_ptr().add(aligned_len_64) };

    while src < src_end_64 {
        let v = unsafe { _mm512_loadu_si512(src as *const __m512i) };
        let (idx, err) = decode_vec!(v);

        if _mm512_test_epi8_mask(err, err) != 0 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(idx, dst);

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
    }

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) } as usize;
    if processed_len < len {
        dst = unsafe { dst.add(scalar::decode_slice_unsafe(config, &input[processed_len..], dst)?) };
    }

    Ok(unsafe { dst.offset_from(dst_start) } as usize)
}

#[cfg(all(test, miri))]
mod miri_avx512_coverage {
    use super::*;
    use base64::{engine::general_purpose::{STANDARD, URL_SAFE}, Engine};
    use rand::{Rng, rng};

    // --- Mock Infrastructure ---
    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    /// Helper to verify AVX512 encoding against the 'base64' crate oracle
    fn verify_encode_avx512(config: &Config, oracle: &impl Engine, input_len: usize) {
        let input = random_bytes(input_len);
        let expected = oracle.encode(&input);
        let mut dst = vec![0u8; expected.len() * 2]; // Safety margin

        unsafe { encode_slice_avx512(config, &input, dst.as_mut_ptr()); }

        let result = &dst[..expected.len()];
        assert_eq!(std::str::from_utf8(result).unwrap(), expected, "Encode len {}", input_len);
    }

    /// Helper to verify AVX512 decoding against the 'base64' crate oracle
    fn verify_decode_avx512(config: &Config, oracle: &impl Engine, original_len: usize) {
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();
        let mut dst = vec![0u8; original_len + 64];

        let len = unsafe {
            decode_slice_avx512(config, encoded_bytes, dst.as_mut_ptr()).expect("Valid input failed to decode")
        };

        assert_eq!(&dst[..len], &input_bytes, "Decode len {}", original_len);
    }

    // ----------------------------------------------------------------------
    // 1. Encoder Coverage Tests (AVX512)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_encode_scalar_fallback() {
        let config = Config { url_safe: false, padding: true };
        // AVX512 Single Loop threshold is 48 bytes.
        // Test < 48 bytes -> Pure Scalar
        verify_encode_avx512(&config, &STANDARD, 1);
        verify_encode_avx512(&config, &STANDARD, 47);
    }

    #[test]
    fn miri_avx512_encode_single_vector_loop() {
        let config = Config { url_safe: false, padding: true };
        // AVX512 Single Vector processes 48 input bytes.
        // Exactly 1 loop
        verify_encode_avx512(&config, &STANDARD, 48);
        // Exactly 2 loops (Proves pointer math)
        verify_encode_avx512(&config, &STANDARD, 96);
        // 1 loop + 1 byte scalar fallback
        verify_encode_avx512(&config, &STANDARD, 49);
    }

    #[test]
    fn miri_avx512_encode_quad_vector_loop() {
        let config = Config { url_safe: false, padding: true };
        // AVX512 Batch Loop processes 192 input bytes (4 * 48).
        // Exactly 1 Quad Loop
        verify_encode_avx512(&config, &STANDARD, 192);
        // Exactly 2 Quad Loops
        verify_encode_avx512(&config, &STANDARD, 384);
        // 1 Quad Loop + 0 Single + 1 byte Scalar
        verify_encode_avx512(&config, &STANDARD, 193);
        // 1 Quad Loop + 1 Single Loop + 0 Scalar (192 + 48)
        verify_encode_avx512(&config, &STANDARD, 240);
    }

    #[test]
    fn miri_avx512_encode_url_safe() {
        let config = Config { url_safe: true, padding: true };
        verify_encode_avx512(&config, &URL_SAFE, 100);
    }

    // ----------------------------------------------------------------------
    // 2. Decoder Coverage Tests (AVX512)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_decode_scalar_fallback() {
        let config = Config { url_safe: false, padding: true };
        // AVX512 decode loop threshold is 64 bytes.
        // < 64 bytes input -> Pure Scalar
        verify_decode_avx512(&config, &STANDARD, 3);  // 4 encoded chars
        verify_decode_avx512(&config, &STANDARD, 45); // 60 encoded chars
    }

    #[test]
    fn miri_avx512_decode_single_vector_loop() {
        let config = Config { url_safe: false, padding: true };
        // AVX512 Single Vector processes 64 input bytes.
        // Exactly 1 Single Loop
        verify_decode_avx512(&config, &STANDARD, 48); // 64 bytes encoded
        // Exactly 2 Single Loops
        verify_decode_avx512(&config, &STANDARD, 96); // 128 bytes encoded
        // 1 Single Loop + Scalar Remainder
        verify_decode_avx512(&config, &STANDARD, 49); // 64 bytes + extra
    }

    #[test]
    fn miri_avx512_decode_quad_vector_loop() {
        let config = Config { url_safe: false, padding: true };
        // AVX512 Batch Loop processes 256 input bytes (4 * 64).
        // Exactly 1 Quad Loop
        verify_decode_avx512(&config, &STANDARD, 192); // 256 bytes encoded
        // Exactly 2 Quad Loops
        verify_decode_avx512(&config, &STANDARD, 384); // 512 bytes encoded
        // 1 Quad Loop + Scalar Remainder
        verify_decode_avx512(&config, &STANDARD, 193); // 256 bytes + extra
    }

    #[test]
    fn miri_avx512_decode_url_safe() {
        let config = Config { url_safe: true, padding: false };
        // 64 bytes input to trigger one AVX512 vector
        // Repeated pattern of - and _
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_";
        let mut dst = [0u8; 64];
        unsafe { decode_slice_avx512(&config, input, dst.as_mut_ptr()).unwrap(); }
    }

    // ----------------------------------------------------------------------
    // 3. Error Logic Coverage (AVX512)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_decode_error_detection() {
        let config = Config { url_safe: false, padding: true };
        let mut dst = [0u8; 512];

        // Case 1: Error in Quad Loop (last vector, last lane)
        // Batch size is 256 bytes.
        let mut bad_input_256 = vec![b'A'; 256];
        bad_input_256[255] = b'$'; // Invalid char
        let res = unsafe { decode_slice_avx512(&config, &bad_input_256, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Quad Loop");

        // Case 2: Error in Single Loop
        // Vector size is 64 bytes.
        let mut bad_input_64 = vec![b'A'; 64];
        bad_input_64[63] = b'?'; // Invalid char
        let res = unsafe { decode_slice_avx512(&config, &bad_input_64, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Single Loop");
    }
}
