use crate::{Error, Config, scalar};
use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};

use core::arch::x86_64::*;

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn encode_slice_avx512(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();
    let mut i = 0;

    let safe_len = len.saturating_sub(16);
    let aligned_len = safe_len - (safe_len % 48);
    let src_end_aligned = unsafe { src.add(aligned_len) };

    let shuffle_128 = _mm_setr_epi8(
        1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
    );
    let shuffle = _mm512_broadcast_i32x4(shuffle_128);
    let permute_indices = _mm512_setr_epi32(
        0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 0,
    );

    let mask_6bit = _mm512_set1_epi16(0x003F);
    let mask_hi_bits = _mm512_set1_epi16(0x3F00);

    let mul_right_shift = _mm512_set1_epi32(0x04000040); 
    let mul_left_shift = _mm512_set1_epi32(0x01000010);

    let set_25 = _mm512_set1_epi8(25);
    let set_51 = _mm512_set1_epi8(51);
    let set_61 = _mm512_set1_epi8(61);
    let set_63 = _mm512_set1_epi8(63);

    let offset_base = _mm512_set1_epi8(65);
    let delta_25 = _mm512_set1_epi8(6);
    let delta_51 = _mm512_set1_epi8(-75);

    let (val_61, val_63) = if config.url_safe { (-13, 49) } else { (-15, 3) };
    let delta_61 = _mm512_set1_epi8(val_61);
    let delta_63 = _mm512_set1_epi8(val_63);

    while src < src_end_aligned {
        let raw_bytes = unsafe { _mm512_maskz_loadu_epi8(0x0000_FFFF_FFFF_FFFF, src as *const _) };

        let v_aligned = _mm512_permutexvar_epi32(permute_indices, raw_bytes);

        let v = _mm512_shuffle_epi8(v_aligned, shuffle);

        let res_low = _mm512_and_si512(_mm512_mulhi_epu16(v, mul_right_shift), mask_6bit);
        let res_high = _mm512_and_si512(_mm512_mullo_epi16(v, mul_left_shift), mask_hi_bits);
        
        let indices = _mm512_or_si512(res_low, res_high);

        let gt_25 = _mm512_cmpgt_epi8_mask(indices, set_25);
        let gt_51 = _mm512_cmpgt_epi8_mask(indices, set_51);
        let gt_61 = _mm512_cmpgt_epi8_mask(indices, set_61);
        let eq_63 = _mm512_cmpeq_epi8_mask(indices, set_63);

        let mut total_offset = offset_base;
        total_offset = _mm512_mask_add_epi8(total_offset, gt_25, total_offset, delta_25);
        total_offset = _mm512_mask_add_epi8(total_offset, gt_51, total_offset, delta_51);
        total_offset = _mm512_mask_add_epi8(total_offset, gt_61, total_offset, delta_61);
        total_offset = _mm512_mask_add_epi8(total_offset, eq_63, total_offset, delta_63);

        let result = _mm512_add_epi8(indices, total_offset);

        unsafe { _mm512_storeu_si512(dst as *mut _, result) };

        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
        i += 48;
    }

    if i < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[i..], dst) };
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn decode_slice_avx512(config: &Config, input: &[u8], mut dst: *mut u8) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let mut i = 0;
    let dst_start = dst;

    let safe_len = len.saturating_sub(64);
    let aligned_len = safe_len - (safe_len % 64);
    let src_end_aligned = unsafe { src.add(aligned_len) };

    let range_a = _mm512_set1_epi8(b'a' as i8);
    let range_z = _mm512_set1_epi8(b'z' as i8);
    let range_capital_a = _mm512_set1_epi8(b'A' as i8);
    let range_capital_z = _mm512_set1_epi8(b'Z' as i8);
    let range_0 = _mm512_set1_epi8(b'0' as i8);
    let range_9 = _mm512_set1_epi8(b'9' as i8);

    let (sym_62, sym_63) = if config.url_safe { (b'-', b'_') } else { (b'+', b'/') };
    let char_62 = _mm512_set1_epi8(sym_62 as i8);
    let char_63 = _mm512_set1_epi8(sym_63 as i8);

    let delta_lower = _mm512_set1_epi8(-71);
    let delta_upper = _mm512_set1_epi8(-65);
    let delta_digit = _mm512_set1_epi8(4);
    let val_dash = _mm512_set1_epi8(62);
    let val_under = _mm512_set1_epi8(63);

    let pack_l1_128 = unsafe { _mm_loadu_si128(PACK_L1.as_ptr() as *const __m128i) };
    let pack_l1 = _mm512_broadcast_i32x4(pack_l1_128);

    let pack_l2_128 = unsafe { _mm_loadu_si128(PACK_L2.as_ptr() as *const __m128i) };
    let pack_l2 = _mm512_broadcast_i32x4(pack_l2_128);

    let pack_shuffle_128 = unsafe { _mm_loadu_si128(PACK_SHUFFLE.as_ptr() as *const __m128i) };
    let pack_shuffle = _mm512_broadcast_i32x4(pack_shuffle_128);

    while src < src_end_aligned {
        let v = unsafe { _mm512_loadu_si512(src as *const _) };

        let ge_a = _mm512_cmpgt_epi8_mask(v, _mm512_sub_epi8(range_a, _mm512_set1_epi8(1)));
        let le_z = _mm512_cmpgt_epi8_mask(_mm512_add_epi8(range_z, _mm512_set1_epi8(1)), v);
        let mk_lower = ge_a & le_z;

        let ge_capital_a = _mm512_cmpgt_epi8_mask(v, _mm512_sub_epi8(range_capital_a, _mm512_set1_epi8(1)));
        let le_capital_z = _mm512_cmpgt_epi8_mask(_mm512_add_epi8(range_capital_z, _mm512_set1_epi8(1)), v);
        let mk_upper = ge_capital_a & le_capital_z;

        let ge_0 = _mm512_cmpgt_epi8_mask(v, _mm512_sub_epi8(range_0, _mm512_set1_epi8(1)));
        let le_9 = _mm512_cmpgt_epi8_mask(_mm512_add_epi8(range_9, _mm512_set1_epi8(1)), v);
        let mk_digit = ge_0 & le_9;

        let mk_dash = _mm512_cmpeq_epi8_mask(v, char_62);
        let mk_under = _mm512_cmpeq_epi8_mask(v, char_63);

        let valid_mask = mk_lower | mk_upper | mk_digit | mk_dash | mk_under;

        if valid_mask != !0u64 {
            return Err(Error::InvalidCharacter);
        }

        let mut indices = _mm512_setzero_si512();
        
        indices = _mm512_mask_add_epi8(indices, mk_lower, v, delta_lower);
        indices = _mm512_mask_add_epi8(indices, mk_upper, v, delta_upper);
        indices = _mm512_mask_add_epi8(indices, mk_digit, v, delta_digit);
 
        indices = _mm512_mask_mov_epi8(indices, mk_dash, val_dash);
        indices = _mm512_mask_mov_epi8(indices, mk_under, val_under);

        let merged = _mm512_maddubs_epi16(indices, pack_l1);
        let packed_u32 = _mm512_madd_epi16(merged, pack_l2);
        let final_bytes = _mm512_shuffle_epi8(packed_u32, pack_shuffle);

        let lane0 = _mm512_castsi512_si128(final_bytes);
        unsafe { _mm_storeu_si128(dst as *mut __m128i, lane0) };

        let lane1 = _mm512_extracti32x4_epi32(final_bytes, 1);
        unsafe { _mm_storeu_si128(dst.add(12) as *mut __m128i, lane1) };

        let lane2 = _mm512_extracti32x4_epi32(final_bytes, 2);
        unsafe { _mm_storeu_si128(dst.add(24) as *mut __m128i, lane2) };


        let lane3 = _mm512_extracti32x4_epi32(final_bytes, 3);
        unsafe { _mm_storeu_si128(dst.add(36) as *mut __m128i, lane3) };

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
        i += 64;
    }

    if i < len {
        dst = unsafe { dst.add(scalar::decode_slice_unsafe(config, &input[i..], dst)?) };
    }
    Ok(unsafe { dst.offset_from(dst_start) } as usize)
}

// Hoping for the support of AVX512...

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
