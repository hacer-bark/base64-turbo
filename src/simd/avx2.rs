use crate::{Error, Config, scalar};
use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};

use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn encode_slice_avx2(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();
    let mut i = 0;

    let shuffle = _mm256_setr_epi8(
        1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
        1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
    );
    let mask_6bit = _mm256_set1_epi16(0x003F);
    let mask_hi_bits = _mm256_set1_epi16(0x3F00);
    let mul_right_shift = _mm256_setr_epi16(
        0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400,
        0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400
    );
    let mul_left_shift = _mm256_setr_epi16(
        0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100,
        0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100
    );

    let set_25 = _mm256_set1_epi8(25);
    let set_51 = _mm256_set1_epi8(51);
    let set_61 = _mm256_set1_epi8(61);
    let set_63 = _mm256_set1_epi8(63);

    let offset_base = _mm256_set1_epi8(65);
    let delta_25 = _mm256_set1_epi8(6);
    let delta_51 = _mm256_set1_epi8(-75);

    let (val_61, val_63) = if config.url_safe { (-13, 49) } else { (-15, 3) };
    let delta_61 = _mm256_set1_epi8(val_61);
    let delta_63 = _mm256_set1_epi8(val_63);

    macro_rules! process_indices {
        ($indices:expr) => {{
            let gt_25 = _mm256_cmpgt_epi8($indices, set_25);
            let gt_51 = _mm256_cmpgt_epi8($indices, set_51);
            let gt_61 = _mm256_cmpgt_epi8($indices, set_61);
            let eq_63 = _mm256_cmpeq_epi8($indices, set_63);

            let d25 = _mm256_and_si256(gt_25, delta_25);
            let d51 = _mm256_and_si256(gt_51, delta_51);
            let d61 = _mm256_and_si256(gt_61, delta_61);
            let d63 = _mm256_and_si256(eq_63, delta_63);

            let sum_a = _mm256_add_epi8(d25, d51);
            let sum_b = _mm256_add_epi8(d61, d63);
            let sum_c = _mm256_add_epi8(sum_a, sum_b);
            
            let offset = _mm256_add_epi8(sum_c, offset_base);
            _mm256_add_epi8($indices, offset)
        }};
    }

    let safe_len = len.saturating_sub(48);
    let aligned_len = safe_len - (safe_len % 48);
    let src_end_aligned = unsafe { src.add(aligned_len) };

    while src < src_end_aligned {
        // Chunk 1
        let c1_0 = unsafe { _mm_loadu_si128(src as *const __m128i) };
        let c1_1 = unsafe { _mm_loadu_si128(src.add(12) as *const __m128i) };
        let v1_in = _mm256_inserti128_si256(_mm256_castsi128_si256(c1_0), c1_1, 1);
        // let v1_in = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let v1 = _mm256_shuffle_epi8(v1_in, shuffle);

        let hi1 = _mm256_mulhi_epu16(v1, mul_right_shift);
        let lo1 = _mm256_mullo_epi16(v1, mul_left_shift);
        let idx1 = _mm256_or_si256(
            _mm256_and_si256(hi1, mask_6bit),
            _mm256_and_si256(lo1, mask_hi_bits)
        );

        // Chunk 2
        let c2_0 = unsafe { _mm_loadu_si128(src.add(24) as *const __m128i) };
        let c2_1 = unsafe { _mm_loadu_si128(src.add(36) as *const __m128i) };
        let v2_in = _mm256_inserti128_si256(_mm256_castsi128_si256(c2_0), c2_1, 1);
        let v2 = _mm256_shuffle_epi8(v2_in, shuffle);

        let hi2 = _mm256_mulhi_epu16(v2, mul_right_shift);
        let lo2 = _mm256_mullo_epi16(v2, mul_left_shift);
        let idx2 = _mm256_or_si256(
            _mm256_and_si256(hi2, mask_6bit),
            _mm256_and_si256(lo2, mask_hi_bits)
        );

        // Process
        let res1 = process_indices!(idx1);
        let res2 = process_indices!(idx2);

        // Store
        unsafe { _mm256_storeu_si256(dst as *mut __m256i, res1) };
        unsafe { _mm256_storeu_si256(dst.add(32) as *mut __m256i, res2) };

        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
        i += 48;
    }

    let safe_len_single = len.saturating_sub(8);
    let aligned_len_single = safe_len_single - (safe_len_single % 24);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let chunk0 = unsafe { _mm_loadu_si128(src as *const __m128i) };
        let chunk1 = unsafe { _mm_loadu_si128(src.add(12) as *const __m128i) };
        let v_in = _mm256_inserti128_si256(_mm256_castsi128_si256(chunk0), chunk1, 1);
        let v = _mm256_shuffle_epi8(v_in, shuffle);
        
        let hi = _mm256_mulhi_epu16(v, mul_right_shift);
        let lo = _mm256_mullo_epi16(v, mul_left_shift);
        let indices = _mm256_or_si256(
            _mm256_and_si256(hi, mask_6bit),
            _mm256_and_si256(lo, mask_hi_bits),
        );

        let res = process_indices!(indices);
        unsafe { _mm256_storeu_si256(dst as *mut __m256i, res) };

        src = unsafe { src.add(24) };
        dst = unsafe { dst.add(32) };
        i += 24;
    }

    if i < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[i..], dst) };
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn decode_slice_avx2(config: &Config, input: &[u8], mut dst: *mut u8) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let mut i = 0;
    let dst_start = dst;

    // --- CONSTANTS ---
    let range_a = _mm256_set1_epi8(b'A' as i8 - 1);
    let range_z = _mm256_set1_epi8(b'Z' as i8 + 1);
    let range_lower_a = _mm256_set1_epi8(b'a' as i8 - 1);
    let range_lower_z = _mm256_set1_epi8(b'z' as i8 + 1);
    let range_0 = _mm256_set1_epi8(b'0' as i8 - 1);
    let range_9 = _mm256_set1_epi8(b'9' as i8 + 1);

    // Deltas to map ASCII to Base64 Index
    // 'A' (65) -> 0.  Delta = -65.
    // 'a' (97) -> 26. Delta = -71.
    // '0' (48) -> 52. Delta = 4.
    let delta_upper = _mm256_set1_epi8(-65);
    let delta_lower = _mm256_set1_epi8(-71);
    let delta_digit = _mm256_set1_epi8(4);

    let (s62, s63) = if config.url_safe { (b'-', b'_') } else { (b'+', b'/') };
    let sym_62 = _mm256_set1_epi8(s62 as i8);
    let sym_63 = _mm256_set1_epi8(s63 as i8);

    // For the symbols, we blend in the final index directly
    let val_62 = _mm256_set1_epi8(62);
    let val_63 = _mm256_set1_epi8(63);

    // Packing constants
    let pack_l1 = unsafe { _mm256_loadu_si256(PACK_L1.as_ptr() as *const __m256i) };
    let pack_l2 = unsafe { _mm256_loadu_si256(PACK_L2.as_ptr() as *const __m256i) };
    let pack_shuffle = unsafe { _mm256_loadu_si256(PACK_SHUFFLE.as_ptr() as *const __m256i) };

    // --- MACRO ---
    macro_rules! process_chunk {
        ($v:expr) => {{
            // 1. Generate Masks
            // Upper Case ('A'..'Z')
            let mask_upper = _mm256_and_si256(
                _mm256_cmpgt_epi8($v, range_a),
                _mm256_cmpgt_epi8(range_z, $v)
            );

            // Lower Case ('a'..'z')
            let mask_lower = _mm256_and_si256(
                _mm256_cmpgt_epi8($v, range_lower_a),
                _mm256_cmpgt_epi8(range_lower_z, $v)
            );

            // Digits ('0'..'9')
            let mask_digit = _mm256_and_si256(
                _mm256_cmpgt_epi8($v, range_0),
                _mm256_cmpgt_epi8(range_9, $v)
            );

            // Symbols
            let mask_62 = _mm256_cmpeq_epi8($v, sym_62);
            let mask_63 = _mm256_cmpeq_epi8($v, sym_63);

            // 2. Error Check
            let any_valid = _mm256_or_si256(
                _mm256_or_si256(mask_upper, mask_lower),
                _mm256_or_si256(mask_digit, _mm256_or_si256(mask_62, mask_63))
            );

            if _mm256_movemask_epi8(any_valid) as u32 != 0xFFFFFFFF {
                return Err(Error::InvalidCharacter);
            }

            // 3. Calculate Delta/Index using Blends
            // If mask_62 is set, result is 62. Else 0.
            let mut acc = _mm256_and_si256(mask_62, val_62);
            // If mask_63 is set, result is 63. Else keep prev.
            acc = _mm256_or_si256(acc, _mm256_and_si256(mask_63, val_63));

            // We use the "overwrite" strategy:
            // indices = (v + delta_range) OR (val_symbol)

            let mut delta = _mm256_setzero_si256();
            delta = _mm256_blendv_epi8(delta, delta_upper, mask_upper);
            delta = _mm256_blendv_epi8(delta, delta_lower, mask_lower);
            delta = _mm256_blendv_epi8(delta, delta_digit, mask_digit);
            
            let range_indices = _mm256_add_epi8($v, delta);

            // Combine: The symbol masks (62/63) have 0 in range_indices (because delta was 0, and v is symbol code, result is garbage but ignored).
            // Actually, blending is safer.
            let res = _mm256_or_si256(
                _mm256_andnot_si256(_mm256_or_si256(mask_62, mask_63), range_indices),
                acc
            );
            res
        }};
    }

    // --- MAIN LOOP (64 bytes in -> 48 bytes out) ---
    // Safe buffer for 64-byte reads
    let safe_len = len.saturating_sub(64);
    let aligned_len = safe_len - (safe_len % 64);
    let src_end_aligned = unsafe { src.add(aligned_len) };

    while src < src_end_aligned {
        let v_1 = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let v_2 = unsafe { _mm256_loadu_si256(src.add(32) as *const __m256i) };

        let idx_1 = process_chunk!(v_1);
        let idx_2 = process_chunk!(v_2);

        // Pack 32 bytes -> 24 bytes
        let m1 = _mm256_maddubs_epi16(idx_1, pack_l1);
        let p1 = _mm256_madd_epi16(m1, pack_l2);
        let out_1 = _mm256_shuffle_epi8(p1, pack_shuffle);

        let m2 = _mm256_maddubs_epi16(idx_2, pack_l1);
        let p2 = _mm256_madd_epi16(m2, pack_l2);
        let out_2 = _mm256_shuffle_epi8(p2, pack_shuffle);

        // Store Logic
        // Chunk 1 stores
        let lane1_lo = _mm256_castsi256_si128(out_1);
        let lane1_hi = _mm256_extracti128_si256(out_1, 1);
        unsafe { _mm_storeu_si128(dst as *mut __m128i, lane1_lo) };
        unsafe { _mm_storeu_si128(dst.add(12) as *mut __m128i, lane1_hi) };

        // Chunk 2 stores
        let lane2_lo = _mm256_castsi256_si128(out_2);
        let lane2_hi = _mm256_extracti128_si256(out_2, 1);
        unsafe { _mm_storeu_si128(dst.add(24) as *mut __m128i, lane2_lo) };
        unsafe { _mm_storeu_si128(dst.add(36) as *mut __m128i, lane2_hi) };

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
        i += 64;
    }

    // --- REMAINDER LOOP (32 bytes in -> 24 bytes out) ---
    let safe_len_single = len.saturating_sub(32);
    let aligned_len_single = safe_len_single - (safe_len_single % 32);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let v = unsafe { _mm256_loadu_si256(src as *const __m256i) };
        let idx = process_chunk!(v);

        let m = _mm256_maddubs_epi16(idx, pack_l1);
        let p = _mm256_madd_epi16(m, pack_l2);
        let out = _mm256_shuffle_epi8(p, pack_shuffle);

        let lane_lo = _mm256_castsi256_si128(out);
        let lane_hi = _mm256_extracti128_si256(out, 1);
        unsafe { _mm_storeu_si128(dst as *mut __m128i, lane_lo) };
        unsafe { _mm_storeu_si128(dst.add(12) as *mut __m128i, lane_hi) };

        src = unsafe { src.add(32) };
        dst = unsafe { dst.add(24) };
        i += 32;
    }

    // --- SCALAR FALLBACK ---
    if i < len {
        dst = unsafe { dst.add(scalar::decode_slice_unsafe(config, &input[i..], dst)?) };
    }

    Ok(unsafe { dst.offset_from(dst_start) } as usize)
}

#[cfg(all(test, miri))]
mod avx2_miri_tests {
    use super::{encode_slice_avx2, decode_slice_avx2};
    use crate::Config;
    use base64::{engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD}};
    use rand::{Rng, rng};

    // --- Helpers ---

    fn encoded_size(len: usize) -> usize { (len + 2) / 3 * 4 }
    fn encoded_size_unpadded(len: usize) -> usize { (len * 4 + 2) / 3 }
    fn estimated_decoded_length(len: usize) -> usize { (len / 4 + 1) * 3 }

    /// Miri Runner:
    /// 1. Runs deterministic boundary tests (0..64 bytes) to hit every loop edge.
    /// 2. Runs a small set of random fuzz tests (50 iterations) to catch weird patterns.
    fn run_miri_cycle<E: base64::Engine>(config: Config, reference_engine: &E) {
        // PART 1: Deterministic Boundary Testing
        for len in 0..=64 {
            let mut rng = rng();
            let mut input = vec![0u8; len];
            rng.fill(&mut input[..]);

            verify_roundtrip(&config, &input, reference_engine);
        }

        // PART 2: Small Fuzzing (Random Lengths)
        let mut rng = rng();
        for _ in 0..100 {
            let len = rng.random_range(65..512);
            let mut input = vec![0u8; len];
            rng.fill(&mut input[..]);

            verify_roundtrip(&config, &input, reference_engine);
        }
    }

    fn verify_roundtrip<E: base64::Engine>(config: &Config, input: &[u8], reference_engine: &E) {
        let len = input.len();

        // --- Encoding ---
        let expected_string = reference_engine.encode(input);

        let enc_len = if config.padding { encoded_size(len) } else { encoded_size_unpadded(len) };
        let mut enc_buf = vec![0u8; enc_len];

        unsafe { 
            encode_slice_avx2(config, input, enc_buf.as_mut_ptr()); 
        }

        assert_eq!(
            &enc_buf, 
            expected_string.as_bytes(), 
            "Miri Encoding Mismatch! Len: {}", len
        );

        // --- Decoding ---
        let dec_max_len = estimated_decoded_length(enc_len);
        let mut dec_buf = vec![0u8; dec_max_len];

        unsafe {
            let written = decode_slice_avx2(config, &enc_buf, dec_buf.as_mut_ptr())
                .expect("Decoder returned error on valid input");

            let my_decoded = &dec_buf[..written];

            assert_eq!(
                my_decoded, 
                input, 
                "Miri Decoding Mismatch! Len: {}", len
            );
        }
    }

    // --- Tests ---

    #[test]
    fn miri_avx2_url_safe_roundtrip() {
        run_miri_cycle(
            Config { url_safe: true, padding: true }, 
            &URL_SAFE
        );
    }

    #[test]
    fn miri_avx2_url_safe_no_pad_roundtrip() {
        run_miri_cycle(
            Config { url_safe: true, padding: false }, 
            &URL_SAFE_NO_PAD
        );
    }

    #[test]
    fn miri_avx2_standard_roundtrip() {
        run_miri_cycle(
            Config { url_safe: false, padding: true }, 
            &STANDARD
        );
    }

    #[test]
    fn miri_avx2_standard_no_pad_roundtrip() {
        run_miri_cycle(
            Config { url_safe: false, padding: false }, 
            &STANDARD_NO_PAD
        );
    }

    // --- Error Checks ---

    #[test]
    fn miri_avx2_invalid_input() {
        let config = Config { url_safe: true, padding: false };
        let mut out = vec![0u8; 10];

        // Pointer math check: Ensure reading invalid chars doesn't cause OOB reads
        let bad_chars = b"heap+"; 
        unsafe {
            let res = decode_slice_avx2(&config, bad_chars, out.as_mut_ptr());
            assert!(res.is_err());
        }
    }
}
