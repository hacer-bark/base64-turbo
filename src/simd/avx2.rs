use crate::{Error, Config, scalar};
use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};

use core::arch::x86_64::*;

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
