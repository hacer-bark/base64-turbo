use crate::{Error, Config, scalar};
use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};

// TODO: Consider add special AVX-512VBMI support
use core::arch::x86_64::*;

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

#[cfg(kani)]
mod kani_verification_avx512 {
    use super::*;

    // 240 bytes input.
    const TEST_LIMIT: usize = 240;
    const MAX_ENCODED_SIZE: usize = 320;

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding { (len + 2) / 3 * 4 } else { (len * 4 + 2) / 3 }
    }

    #[kani::proof]
    #[kani::unwind(241)]
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
        unsafe { encode_slice_avx512(&config, input, enc_buf.as_mut_ptr()); }

        // Decoding
        let mut dec_buf = [0u8; TEST_LIMIT];

        unsafe {
            let src_slice = &enc_buf[..enc_len];

            let written = decode_slice_avx512(&config, src_slice, dec_buf.as_mut_ptr())
                .expect("Decoder returned error on valid input");

            let my_decoded = &dec_buf[..written];

            assert_eq!(my_decoded, input, "Kani Decoding Mismatch!");
        }
    }

    #[kani::proof]
    #[kani::unwind(241)]
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
            let _ = decode_slice_avx512(&config, input, dec_buf.as_mut_ptr());
        }
    }
}

// Hoping for the support of AVX512 in Miri...

// Tests itself outdated. If and when support for AVX512 would released, tests must be fixed for newer syntax.
// #[cfg(all(test, miri))]
// mod avx512_miri_tests {
//     use super::{encode_slice_avx512, decode_slice_avx512};
//     use crate::Config;
//     use base64::{engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD}};
//     use rand::{Rng, rng};

//     // --- Helpers ---

//     fn encoded_size(len: usize) -> usize { (len + 2) / 3 * 4 }
//     fn encoded_size_unpadded(len: usize) -> usize { (len * 4 + 2) / 3 }
//     fn estimated_decoded_length(len: usize) -> usize { (len / 4 + 1) * 3 }

//     /// Miri Runner:
//     /// 1. Runs deterministic boundary tests (0..64 bytes) to hit every loop edge.
//     /// 2. Runs a small set of random fuzz tests (50 iterations) to catch weird patterns.
//     fn run_miri_cycle<E: base64::Engine>(config: Config, reference_engine: &E) {
//         // PART 1: Deterministic Boundary Testing
//         for len in 0..=64 {
//             let mut rng = rng();
//             let mut input = vec![0u8; len];
//             rng.fill(&mut input[..]);

//             verify_roundtrip(&config, &input, reference_engine);
//         }

//         // PART 2: Small Fuzzing (Random Lengths)
//         let mut rng = rng();
//         for _ in 0..100 {
//             let len = rng.random_range(65..512);
//             let mut input = vec![0u8; len];
//             rng.fill(&mut input[..]);

//             verify_roundtrip(&config, &input, reference_engine);
//         }
//     }

//     fn verify_roundtrip<E: base64::Engine>(config: &Config, input: &[u8], reference_engine: &E) {
//         let len = input.len();

//         // --- Encoding ---
//         let expected_string = reference_engine.encode(input);

//         let enc_len = if config.padding { encoded_size(len) } else { encoded_size_unpadded(len) };
//         let mut enc_buf = vec![0u8; enc_len];

//         unsafe { 
//             encode_slice_avx512(config, input, &mut enc_buf); 
//         }

//         assert_eq!(
//             &enc_buf, 
//             expected_string.as_bytes(), 
//             "Miri Encoding Mismatch! Len: {}", len
//         );

//         // --- Decoding ---
//         let dec_max_len = estimated_decoded_length(enc_len);
//         let mut dec_buf = vec![0u8; dec_max_len];

//         unsafe {
//             let written = decode_slice_avx512(config, &enc_buf, &mut dec_buf)
//                 .expect("Decoder returned error on valid input");

//             let my_decoded = &dec_buf[..written];

//             assert_eq!(
//                 my_decoded, 
//                 input, 
//                 "Miri Decoding Mismatch! Len: {}", len
//             );
//         }
//     }

//     // --- Tests ---

//     #[test]
//     fn miri_avx512_url_safe_roundtrip() {
//         run_miri_cycle(
//             Config { url_safe: true, padding: true }, 
//             &URL_SAFE
//         );
//     }

//     #[test]
//     fn miri_avx512_url_safe_no_pad_roundtrip() {
//         run_miri_cycle(
//             Config { url_safe: true, padding: false }, 
//             &URL_SAFE_NO_PAD
//         );
//     }

//     #[test]
//     fn miri_avx512_standard_roundtrip() {
//         run_miri_cycle(
//             Config { url_safe: false, padding: true }, 
//             &STANDARD
//         );
//     }

//     #[test]
//     fn miri_avx512_standard_no_pad_roundtrip() {
//         run_miri_cycle(
//             Config { url_safe: false, padding: false }, 
//             &STANDARD_NO_PAD
//         );
//     }

//     // --- Error Checks ---

//     #[test]
//     fn miri_avx512_invalid_input() {
//         let config = Config { url_safe: true, padding: false };
//         let mut out = vec![0u8; 10];

//         // Pointer math check: Ensure reading invalid chars doesn't cause OOB reads
//         let bad_chars = b"heap+"; 
//         unsafe {
//             let res = decode_slice_avx512(&config, bad_chars, &mut out);
//             assert!(res.is_err());
//         }
//     }
// }
