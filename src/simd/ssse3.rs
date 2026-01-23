use crate::{Error, Config, scalar};
use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};
use core::arch::x86_64::*;

#[target_feature(enable = "ssse3")]
pub unsafe fn encode_slice_simd(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();
    let mut i = 0;

    let safe_limit = len.saturating_sub(4);
    let aligned_len = safe_limit - (safe_limit % 12);
    let src_end_aligned = unsafe { src.add(aligned_len) };

    let shuffle = _mm_setr_epi8(1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10);
    let mask_6bit = _mm_set1_epi16(0x003F);
    let mask_hi_bits = _mm_set1_epi16(0x3F00);
    let mul_right_shift = _mm_setr_epi16(0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400);
    let mul_left_shift = _mm_setr_epi16(0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100);
    let set_25 = _mm_set1_epi8(25);
    let set_51 = _mm_set1_epi8(51);
    let set_61 = _mm_set1_epi8(61);
    let set_63 = _mm_set1_epi8(63);
    let offset_base = _mm_set1_epi8(65);
    let delta_25 = _mm_set1_epi8(6);
    let delta_51 = _mm_set1_epi8(-75);

    let (val_61, val_63) = if config.url_safe { (-13, 49) } else { (-15, 3) };
    let delta_61 = _mm_set1_epi8(val_61);
    let delta_63 = _mm_set1_epi8(val_63);

    while src < src_end_aligned {
        let v_in = unsafe { _mm_loadu_si128(src as *const __m128i) };
        let v = _mm_shuffle_epi8(v_in, shuffle);

        let res_low = _mm_and_si128(_mm_mulhi_epu16(v, mul_right_shift), mask_6bit);
        let res_high = _mm_and_si128(_mm_mullo_epi16(v, mul_left_shift), mask_hi_bits);
        let indices = _mm_or_si128(res_low, res_high);

        let gt_25 = _mm_cmpgt_epi8(indices, set_25);
        let gt_51 = _mm_cmpgt_epi8(indices, set_51);
        let gt_61 = _mm_cmpgt_epi8(indices, set_61);
        let eq_63 = _mm_cmpeq_epi8(indices, set_63);

        let d25 = _mm_and_si128(gt_25, delta_25);
        let d51 = _mm_and_si128(gt_51, delta_51);
        let d61 = _mm_and_si128(gt_61, delta_61);
        let d63 = _mm_and_si128(eq_63, delta_63);

        let sum_a = _mm_add_epi8(d25, d51);
        let sum_b = _mm_add_epi8(d61, d63);
        let sum_c = _mm_add_epi8(sum_a, sum_b);
        let total_offset = _mm_add_epi8(sum_c, offset_base);

        let result = _mm_add_epi8(indices, total_offset);

        unsafe { _mm_storeu_si128(dst as *mut __m128i, result) };

        src = unsafe { src.add(12) };
        dst = unsafe { dst.add(16) };
        i += 12;
    }

    if i < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[i..], dst) };
    }
}

#[target_feature(enable = "ssse3")]
pub unsafe fn decode_slice_simd(config: &Config, input: &[u8], mut dst: *mut u8) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let mut i = 0;
    let dst_start = dst;

    let safe_len = len.saturating_sub(16);
    let aligned_len = safe_len - (safe_len % 16);
    let src_end_aligned = unsafe { src.add(aligned_len) };

    let range_a = _mm_set1_epi8(b'a' as i8);
    let range_z = _mm_set1_epi8(b'z' as i8);
    let range_capital_a = _mm_set1_epi8(b'A' as i8);
    let range_capital_z = _mm_set1_epi8(b'Z' as i8);
    let range_0 = _mm_set1_epi8(b'0' as i8);
    let range_9 = _mm_set1_epi8(b'9' as i8);

    let (sym_62, sym_63) = if config.url_safe { (b'-', b'_') } else { (b'+', b'/') };
    let char_62 = _mm_set1_epi8(sym_62 as i8);
    let char_63 = _mm_set1_epi8(sym_63 as i8);

    let delta_lower = _mm_set1_epi8(-71);
    let delta_upper = _mm_set1_epi8(-65);
    let delta_digit = _mm_set1_epi8(4);
    let val_62 = _mm_set1_epi8(62);
    let val_63 = _mm_set1_epi8(63);

    let pack_l1 = unsafe { _mm_loadu_si128(PACK_L1.as_ptr() as *const __m128i) };
    let pack_l2 = unsafe { _mm_loadu_si128(PACK_L2.as_ptr() as *const __m128i) };
    let pack_shuffle = unsafe { _mm_loadu_si128(PACK_SHUFFLE.as_ptr() as *const __m128i) };

    while src < src_end_aligned {
        let v = unsafe { _mm_loadu_si128(src as *const __m128i) };

        let ge_a = _mm_cmpgt_epi8(v, _mm_sub_epi8(range_a, _mm_set1_epi8(1)));
        let le_z = _mm_cmpgt_epi8(_mm_add_epi8(range_z, _mm_set1_epi8(1)), v);
        let mk_lower = _mm_and_si128(ge_a, le_z);

        let ge_capital_a = _mm_cmpgt_epi8(v, _mm_sub_epi8(range_capital_a, _mm_set1_epi8(1)));
        let le_capital_z = _mm_cmpgt_epi8(_mm_add_epi8(range_capital_z, _mm_set1_epi8(1)), v);
        let mk_upper = _mm_and_si128(ge_capital_a, le_capital_z);

        let ge_0 = _mm_cmpgt_epi8(v, _mm_sub_epi8(range_0, _mm_set1_epi8(1)));
        let le_9 = _mm_cmpgt_epi8(_mm_add_epi8(range_9, _mm_set1_epi8(1)), v);
        let mk_digit = _mm_and_si128(ge_0, le_9);

        let mk_62 = _mm_cmpeq_epi8(v, char_62);
        let mk_63 = _mm_cmpeq_epi8(v, char_63);

        let valid_mask = _mm_or_si128(
            _mm_or_si128(mk_lower, mk_upper),
            _mm_or_si128(mk_digit, _mm_or_si128(mk_62, mk_63))
        );

        if _mm_movemask_epi8(valid_mask) != 0xFFFF {
            return Err(Error::InvalidCharacter);
        }

        let mut indices = _mm_and_si128(mk_lower, _mm_add_epi8(v, delta_lower));
        indices = _mm_or_si128(indices, _mm_and_si128(mk_upper, _mm_add_epi8(v, delta_upper)));
        indices = _mm_or_si128(indices, _mm_and_si128(mk_digit, _mm_add_epi8(v, delta_digit)));
        indices = _mm_or_si128(indices, _mm_and_si128(mk_62, val_62));
        indices = _mm_or_si128(indices, _mm_and_si128(mk_63, val_63));

        let merged = _mm_maddubs_epi16(indices, pack_l1);
        let packed_u32 = _mm_madd_epi16(merged, pack_l2);
        let final_bytes = _mm_shuffle_epi8(packed_u32, pack_shuffle);

        unsafe { _mm_storeu_si128(dst as *mut __m128i, final_bytes) };

        src = unsafe { src.add(16) };
        dst = unsafe { dst.add(12) };
        i += 16;
    }

    if i < len {
        dst = unsafe { dst.add(scalar::decode_slice_unsafe(config, &input[i..], dst)?) };
    }
    Ok(unsafe { dst.offset_from(dst_start) } as usize)
}

#[cfg(kani)]
mod kani_verification_ssse3 {
    use super::*;

    // 120 bytes input.
    const TEST_LIMIT: usize = 120;
    const MAX_ENCODED_SIZE: usize = 160;

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding { (len + 2) / 3 * 4 } else { (len * 4 + 2) / 3 }
    }

    #[kani::proof]
    #[kani::unwind(121)]
    fn check_round_trip() {
        // Symbolic Config
        let config = Config {
            url_safe: kani::any(),
            padding: kani::any(),
        };

        // Symbolic Length
        let len: usize = kani::any();
        kani::assume(len <= TEST_LIMIT);

        // Symbolic Input Data
        let input_arr: [u8; TEST_LIMIT] = kani::any();
        let input = &input_arr[..len];

        // Setup Encoding Buffer 
        let enc_len = encoded_size(len, config.padding);

        // Sanity check for the verification harness itself
        assert!(enc_len <= MAX_ENCODED_SIZE);

        let mut enc_buf = [0u8; MAX_ENCODED_SIZE];
        unsafe { encode_slice_simd(&config, input, enc_buf.as_mut_ptr()); }

        // Decoding
        let mut dec_buf = [0u8; TEST_LIMIT];

        unsafe {
            let src_slice = &enc_buf[..enc_len];

            let written = decode_slice_simd(&config, src_slice, dec_buf.as_mut_ptr())
                .expect("Decoder returned error on valid input");

            let my_decoded = &dec_buf[..written];

            assert_eq!(my_decoded, input, "Kani Decoding Mismatch!");
        }
    }

    #[kani::proof]
    #[kani::unwind(121)]
    fn check_decoder_robustness() {
        // Symbolic Config
        let config = Config {
            url_safe: kani::any(),
            padding: kani::any(),
        };

        // Symbolic Input (Random Garbage)
        let len: usize = kani::any();
        kani::assume(len <= MAX_ENCODED_SIZE);
        
        let input_arr: [u8; MAX_ENCODED_SIZE] = kani::any();
        let input = &input_arr[..len];

        // Decoding Buffer
        let mut dec_buf = [0u8; MAX_ENCODED_SIZE];

        unsafe {
            // We verify what function NEVER panics/crashes
            let _ = decode_slice_simd(&config, input, dec_buf.as_mut_ptr());
        }
    }
}

#[cfg(all(test, miri))]
mod ssse3_miri_tests {
    use super::{encode_slice_simd, decode_slice_simd};
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
            encode_slice_simd(config, input, enc_buf.as_mut_ptr()); 
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
            let written = decode_slice_simd(config, &enc_buf, dec_buf.as_mut_ptr())
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
    fn miri_ssse3_url_safe_roundtrip() {
        run_miri_cycle(
            Config { url_safe: true, padding: true }, 
            &URL_SAFE
        );
    }

    #[test]
    fn miri_ssse3_url_safe_no_pad_roundtrip() {
        run_miri_cycle(
            Config { url_safe: true, padding: false }, 
            &URL_SAFE_NO_PAD
        );
    }

    #[test]
    fn miri_ssse3_standard_roundtrip() {
        run_miri_cycle(
            Config { url_safe: false, padding: true }, 
            &STANDARD
        );
    }

    #[test]
    fn miri_ssse3_standard_no_pad_roundtrip() {
        run_miri_cycle(
            Config { url_safe: false, padding: false }, 
            &STANDARD_NO_PAD
        );
    }

    // --- Error Checks ---

    #[test]
    fn miri_ssse3_invalid_input() {
        let config = Config { url_safe: true, padding: false };
        let mut out = vec![0u8; 10];

        // Pointer math check: Ensure reading invalid chars doesn't cause OOB reads
        let bad_chars = b"heap+"; 
        unsafe {
            let res = decode_slice_simd(&config, bad_chars, out.as_mut_ptr());
            assert!(res.is_err());
        }
    }
}
