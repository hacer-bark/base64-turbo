use crate::{Error, Config, scalar};
use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn encode_slice_simd(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

    // Shuffle bytes for mul
    let shuffle = _mm_setr_epi8(1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10);

    // Masks for bit extraction
    let mask_lo_6bits = _mm_set1_epi16(0x003F);
    let mask_hi_6bits = _mm_set1_epi16(0x3F00);

    // Multiplier for shift of bytes.
    let mul_right_shift = _mm_setr_epi16(0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400, 0x0040, 0x0400);
    let mul_left_shift = _mm_setr_epi16(0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100, 0x0010, 0x0100);

    // Mapping logic for letters
    let offset_base = _mm_set1_epi8(65);
    let set_25 = _mm_set1_epi8(25);
    let delta_lower = _mm_set1_epi8(6);
    let set_51 = _mm_set1_epi8(51);

    // LUT Table for numbers and special chars
    let (sym_plus, sym_slash) = if config.url_safe { (-88, -39) } else { (-90, -87) };
    let lut_offsets = _mm_setr_epi8(0, -75, -75, -75, -75, -75, -75, -75, -75, -75, -75, sym_plus, sym_slash, 0, 0, 0);

    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Compute 3 bytes => 4 letters
            let v = _mm_shuffle_epi8($in_vec, shuffle);

            let lo = _mm_mullo_epi16(v, mul_left_shift);
            let hi = _mm_mulhi_epu16(v, mul_right_shift);
            let indices = _mm_or_si128(
                _mm_and_si128(lo, mask_hi_6bits),
                _mm_and_si128(hi, mask_lo_6bits),
            );

            // Found char values offsets
            let mut char_val = _mm_add_epi8(indices, offset_base);
            let offset_lower = _mm_and_si128(_mm_cmpgt_epi8(indices, set_25), delta_lower);
            char_val = _mm_add_epi8(char_val, offset_lower);

            // Found numbers and special symbols offsets
            let offset_special = _mm_shuffle_epi8(lut_offsets, _mm_subs_epu8(indices, set_51));

            // Final sum
            _mm_add_epi8(char_val, offset_special)
        }};
    }

    // Process 48 bytes (4 chunks) at a time
    let safe_len_batch = len.saturating_sub(4);
    let aligned_len_batch = safe_len_batch - (safe_len_batch % 48);
    let src_end_batch = unsafe { src.add(aligned_len_batch) };

    while src < src_end_batch {
        // Load 4 vectors
        let v0 = unsafe { _mm_loadu_si128(src as *const __m128i) };
        let v1 = unsafe { _mm_loadu_si128(src.add(12) as *const __m128i) };
        let v2 = unsafe { _mm_loadu_si128(src.add(24) as *const __m128i) };
        let v3 = unsafe { _mm_loadu_si128(src.add(36) as *const __m128i) };

        // Process
        let r0 = encode_vec!(v0);
        let r1 = encode_vec!(v1);
        let r2 = encode_vec!(v2);
        let r3 = encode_vec!(v3);

        // Store 4 chunks
        unsafe { _mm_storeu_si128(dst as *mut __m128i, r0) };
        unsafe { _mm_storeu_si128(dst.add(16) as *mut __m128i, r1) };
        unsafe { _mm_storeu_si128(dst.add(32) as *mut __m128i, r2) };
        unsafe { _mm_storeu_si128(dst.add(48) as *mut __m128i, r3) };

        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
    }

    // Process remaining 12-byte chunks
    let safe_len_single = len.saturating_sub(4);
    let aligned_len_single = safe_len_single - (safe_len_single % 12);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let v = unsafe { _mm_loadu_si128(src as *const __m128i) };
        let res = encode_vec!(v);
        unsafe { _mm_storeu_si128(dst as *mut __m128i, res) };

        src = unsafe { src.add(12) };
        dst = unsafe { dst.add(16) };
    }

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) } as usize;
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

#[target_feature(enable = "ssse3,sse4.1")]
pub unsafe fn decode_slice_simd(config: &Config, input: &[u8], mut dst: *mut u8) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst;

    // LUT for offsets based on high nibble (bits 4-7).
    let lut_hi_nibble = _mm_setr_epi8(0, 0, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0);

    // Range and offsets of special chars
    let (char_62, char_63) = if config.url_safe { (b'-', b'_') } else { (b'+', b'/') };
    let sym_62 = _mm_set1_epi8(char_62 as i8);
    let sym_63 = _mm_set1_epi8(char_63 as i8);

    let (fix_62, fix_63) = if config.url_safe { (-2, 33) } else { (0, -3) };
    let delta_62 = _mm_set1_epi8(fix_62);
    let delta_63 = _mm_set1_epi8(fix_63);

    // Range Validation Constants
    let range_0 = _mm_set1_epi8(b'0' as i8);
    let range_9_len = _mm_set1_epi8(9);

    let range_a = _mm_set1_epi8(b'A' as i8);
    let range_z_len = _mm_set1_epi8(25);

    let range_a_low = _mm_set1_epi8(b'a' as i8);
    let range_z_low_len = _mm_set1_epi8(25);

    // Packing Constants
    let pack_l1 = unsafe { _mm_loadu_si128(PACK_L1.as_ptr() as *const __m128i) };
    let pack_l2 = unsafe { _mm_loadu_si128(PACK_L2.as_ptr() as *const __m128i) };
    let pack_shuffle = unsafe { _mm_loadu_si128(PACK_SHUFFLE.as_ptr() as *const __m128i) };

    // Masks for nibble extraction
    let mask_hi_nibble = _mm_set1_epi8(0x0F);

    // Decode & Validate Single Vector
    macro_rules! decode_vec {
        ($input:expr) => {{
            let hi = _mm_and_si128(_mm_srli_epi16($input, 4), mask_hi_nibble);
            let offset = _mm_shuffle_epi8(lut_hi_nibble, hi);
            let mut indices = _mm_add_epi8($input, offset);

            let mask_62 = _mm_cmpeq_epi8($input, sym_62);
            let mask_63 = _mm_cmpeq_epi8($input, sym_63);

            let fix = _mm_or_si128(
                _mm_and_si128(mask_62, delta_62),
                _mm_and_si128(mask_63, delta_63),
            );
            indices = _mm_add_epi8(indices, fix);

            let is_sym = _mm_or_si128(mask_62, mask_63);

            let sub_0 = _mm_subs_epu8(_mm_sub_epi8($input, range_0), range_9_len);
            let sub_a = _mm_subs_epu8(_mm_sub_epi8($input, range_a), range_z_len);
            let sub_a_low = _mm_subs_epu8(_mm_sub_epi8($input, range_a_low), range_z_low_len);

            let err = _mm_andnot_si128(
                is_sym,
                _mm_and_si128(sub_0, _mm_and_si128(sub_a, sub_a_low)),
            );

            (indices, err)
        }};
    }

    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            let m = _mm_maddubs_epi16($indices, pack_l1);
            let p = _mm_madd_epi16(m, pack_l2);
            let out = _mm_shuffle_epi8(p, pack_shuffle);

            unsafe { _mm_storeu_si128($dst_ptr as *mut __m128i, out) };
        }};
    }

    // Process 64 bytes (4 chunks) at a time
    let safe_len_batch = len.saturating_sub(4);
    let aligned_len_batch = safe_len_batch - (safe_len_batch % 64);
    let src_end_batch = unsafe { src.add(aligned_len_batch) };

    while src < src_end_batch {
        // Load 4 vectors
        let v0 = unsafe { _mm_loadu_si128(src as *const __m128i) };
        let v1 = unsafe { _mm_loadu_si128(src.add(16) as *const __m128i) };
        let v2 = unsafe { _mm_loadu_si128(src.add(32) as *const __m128i) };
        let v3 = unsafe { _mm_loadu_si128(src.add(48) as *const __m128i) };

        // Process
        let (r0, e0) = decode_vec!(v0);
        let (r1, e1) = decode_vec!(v1);
        let (r2, e2) = decode_vec!(v2);
        let (r3, e3) = decode_vec!(v3);

        // Check Errors
        let err_any = _mm_or_si128(
            _mm_or_si128(e0, e1), 
            _mm_or_si128(e2, e3)
        );

        if _mm_testz_si128(err_any, err_any) != 1 {
            return Err(Error::InvalidCharacter);
        }

        // Store 4 chunks
        pack_and_store!(r0, dst);
        pack_and_store!(r1, dst.add(12));
        pack_and_store!(r2, dst.add(24));
        pack_and_store!(r3, dst.add(36));

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
    }

    // Process remaining 16-byte chunks
    let safe_len_single = len.saturating_sub(4);
    let aligned_len_single = safe_len_single - (safe_len_single % 16);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let v = unsafe { _mm_loadu_si128(src as *const __m128i) };
        let (idx, err) = decode_vec!(v);

        if _mm_testz_si128(err, err) != 1 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(idx, dst);

        src = unsafe { src.add(16) };
        dst = unsafe { dst.add(12) };
    }

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) } as usize;
    if processed_len < len {
        dst = unsafe { dst.add(scalar::decode_slice_unsafe(config, &input[processed_len..], dst)?) };
    }

    Ok(unsafe { dst.offset_from(dst_start) } as usize)
}

// #[cfg(kani)]
// mod kani_verification_ssse3 {
//     use super::*;

//     // TODO: Need to write harness for all logic. As well as rewrite it.

//     // 120 bytes input.
//     const TEST_LIMIT: usize = 120;
//     const MAX_ENCODED_SIZE: usize = 160;

//     fn encoded_size(len: usize, padding: bool) -> usize {
//         if padding { (len + 2) / 3 * 4 } else { (len * 4 + 2) / 3 }
//     }

//     #[kani::proof]
//     #[kani::unwind(121)]
//     fn check_round_trip() {
//         // Symbolic Config
//         let config = Config {
//             url_safe: kani::any(),
//             padding: kani::any(),
//         };

//         // Symbolic Length
//         let len: usize = kani::any();
//         kani::assume(len <= TEST_LIMIT);

//         // Symbolic Input Data
//         let input_arr: [u8; TEST_LIMIT] = kani::any();
//         let input = &input_arr[..len];

//         // Setup Encoding Buffer 
//         let enc_len = encoded_size(len, config.padding);

//         // Sanity check for the verification harness itself
//         assert!(enc_len <= MAX_ENCODED_SIZE);

//         let mut enc_buf = [0u8; MAX_ENCODED_SIZE];
//         unsafe { encode_slice_simd(&config, input, enc_buf.as_mut_ptr()); }

//         // Decoding
//         let mut dec_buf = [0u8; TEST_LIMIT];

//         unsafe {
//             let src_slice = &enc_buf[..enc_len];

//             let written = decode_slice_simd(&config, src_slice, dec_buf.as_mut_ptr())
//                 .expect("Decoder returned error on valid input");

//             let my_decoded = &dec_buf[..written];

//             assert_eq!(my_decoded, input, "Kani Decoding Mismatch!");
//         }
//     }

//     #[kani::proof]
//     #[kani::unwind(121)]
//     fn check_decoder_robustness() {
//         // Symbolic Config
//         let config = Config {
//             url_safe: kani::any(),
//             padding: kani::any(),
//         };

//         // Symbolic Input (Random Garbage)
//         let len: usize = kani::any();
//         kani::assume(len <= MAX_ENCODED_SIZE);
        
//         let input_arr: [u8; MAX_ENCODED_SIZE] = kani::any();
//         let input = &input_arr[..len];

//         // Decoding Buffer
//         let mut dec_buf = [0u8; MAX_ENCODED_SIZE];

//         unsafe {
//             // We verify what function NEVER panics/crashes
//             let _ = decode_slice_simd(&config, input, dec_buf.as_mut_ptr());
//         }
//     }
// }

#[cfg(all(test, miri))]
mod sse4_miri_tests {
    use super::{encode_slice_simd, decode_slice_simd};
    use crate::Config;
    use base64::{engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD}};
    use rand::{Rng, rng};

    // --- Helpers ---

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding { (len + 2) / 3 * 4 } else { (len * 4 + 2) / 3 }
    }
    fn estimated_decoded_length(len: usize) -> usize { (len / 4 + 1) * 3 }

    /// Miri Runner:
    /// 1. Runs deterministic boundary tests (0..64 bytes) to hit every loop edge.
    /// 2. Runs a small set of random fuzz tests (50 iterations) to catch weird patterns.
    fn run_miri_cycle<E: base64::Engine>(config: Config, reference_engine: &E) {
        // Deterministic Boundary Testing
        for len in 0..=64 {
            let mut rng = rng();
            let mut input = vec![0u8; len];
            rng.fill(&mut input[..]);

            verify_roundtrip(&config, &input, reference_engine);
        }

        // Small Fuzzing (Random Lengths)
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

        let enc_len = encoded_size(len, config.padding);
        let mut enc_buf = vec![0u8; enc_len];

        unsafe { encode_slice_simd(config, input, enc_buf.as_mut_ptr()); }

        assert_eq!(&enc_buf, expected_string.as_bytes(), "Miri Encoding Mismatch!");

        // --- Decoding ---
        let dec_max_len = estimated_decoded_length(enc_len);
        let mut dec_buf = vec![0u8; dec_max_len];

        unsafe {
            let written = decode_slice_simd(config, &enc_buf, dec_buf.as_mut_ptr())
                .expect("Decoder returned error on valid input");

            let my_decoded = &dec_buf[..written];

            assert_eq!(my_decoded, input, "Miri Decoding Mismatch!");
        }
    }

    // --- Tests ---

    #[test]
    fn miri_sse4_url_safe_roundtrip() {
        run_miri_cycle(
            Config { url_safe: true, padding: true }, 
            &URL_SAFE
        );
    }

    #[test]
    fn miri_sse4_url_safe_no_pad_roundtrip() {
        run_miri_cycle(
            Config { url_safe: true, padding: false }, 
            &URL_SAFE_NO_PAD
        );
    }

    #[test]
    fn miri_sse4_standard_roundtrip() {
        run_miri_cycle(
            Config { url_safe: false, padding: true }, 
            &STANDARD
        );
    }

    #[test]
    fn miri_sse4_standard_no_pad_roundtrip() {
        run_miri_cycle(
            Config { url_safe: false, padding: false }, 
            &STANDARD_NO_PAD
        );
    }

    // --- Error Checks ---

    #[test]
    fn miri_sse4_invalid_input() {
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
