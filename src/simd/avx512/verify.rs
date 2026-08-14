//! Plain AVX-512F/BW verification: Kani proofs (with intrinsic stubs) and the
//! Miri coverage suite. Split out of the production module to keep it lean.

// Re-export the production module to the cfg-gated child suites below; unused
// in a plain (non-Miri, non-Kani) test build where none of them compile.
#[allow(unused_imports)]
use super::*;

#[cfg(kani)]
mod kani_verification_avx512 {
    use super::*;
    use crate::{Config, STANDARD as TURBO_STANDARD, STANDARD_NO_PAD as TURBO_STANDARD_NO_PAD};
    #[cfg(target_arch = "x86")]
    use std::arch::x86::__mmask64;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::__mmask64;
    use std::mem::transmute;

    // --- CONSTANTS ---

    // 2x 48-byte single-vector passes (96) + a 17-byte scalar remainder (the
    // +17 covers the loop's own 16-byte margin).
    const ENC_INDUCTION_LEN: usize = 113;

    // 2x 64-byte single-vector passes (128) + 5-byte scalar tail (4-byte margin).
    const DEC_INDUCTION_LEN: usize = 133;

    // Smallest length triggering exactly 1 quad-loop pass (0 single passes) plus
    // a scalar remainder; used only by `check_avx512_quad_tier_roundtrip`.
    const QUAD_ENC_INDUCTION_LEN: usize = 209;

    // --- HELPERS ---

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding {
            TURBO_STANDARD.encoded_len(len)
        } else {
            TURBO_STANDARD_NO_PAD.encoded_len(len)
        }
    }

    // --- STUBS ---

    // STUB: _mm512_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shuffle_epi8
    unsafe fn _mm512_shuffle_epi8_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            // (In Rust we access bytes 'j' so '*8' offset is not needed)
            let i = j;

            // IF b[i+7] == 1
            if (b[i] & 0x80) != 0 {
                // dst[i+7:i] := 0
                dst[i] = 0;
            // ELSE
            } else {
                // index[5:0] := b[i+3:i] + (j & 0x30)
                let index: u8 = (b[i] & 0x0F) + (j as u8 & 0x30);
                // dst[i+7:i] := a[index*8+7:index*8]
                dst[i] = a[index as usize];
                // FI
            }
            // ENDFOR
        }
        // dst[MAX:512] := 0
        // (No extra bits beyond 512 in __m512i)

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_mask_add_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_add_epi8
    unsafe fn _mm512_mask_add_epi8_stub(
        src: __m512i,
        k: __mmask64,
        a: __m512i,
        b: __m512i,
    ) -> __m512i {
        let src_bytes: [u8; 64] = unsafe { transmute(src) };
        let a_bytes: [u8; 64] = unsafe { transmute(a) };
        let b_bytes: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            let i = j;

            // IF k[j]
            if (k & (1 << j)) != 0 {
                // dst[i+7:i] := a[i+7:i] + b[i+7:i]
                dst[i] = a_bytes[i].wrapping_add(b_bytes[i]);
            // ELSE
            } else {
                // dst[i+7:i] := src[i+7:i]
                dst[i] = src_bytes[i];
                // FI
            }
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maddubs_epi16
    unsafe fn _mm512_maddubs_epi16_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [i8; 64] = unsafe { transmute(b) };
        let mut dst = [0i16; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := j*16
            let i = j * 2;
            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i + 1] as i16) * (b[i + 1] as i16))
                .saturating_add((a[i] as i16) * (b[i] as i16));
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_madd_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_madd_epi16
    unsafe fn _mm512_madd_epi16_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [i16; 32] = unsafe { transmute(a) };
        let b: [i16; 32] = unsafe { transmute(b) };
        let mut dst = [0i32; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*32
            let i = j * 2;

            // dst[i+31:i] := SignExtend32(a[i+31:i+16]*b[i+31:i+16]) + SignExtend32(a[i+15:i]*b[i+15:i])
            dst[j] = (a[i + 1] as i32)
                .wrapping_mul(b[i + 1] as i32)
                .wrapping_add((a[i] as i32).wrapping_mul(b[i] as i32));
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_permutexvar_epi32
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutexvar_epi32
    unsafe fn _mm512_permutexvar_epi32_stub(idx: __m512i, a: __m512i) -> __m512i {
        let idx: [u32; 16] = unsafe { transmute(idx) };
        let a: [u32; 16] = unsafe { transmute(a) };
        let mut dst = [0u32; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // id := idx[j*32+3:j*32]
            let id = (idx[j] & 0xF) as usize;
            // dst[j*32+31:j*32] := a[id*32+31:id*32]
            dst[j] = a[id];
        }
        // ENDFOR
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_sub_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_sub_epi8
    unsafe fn _mm512_sub_epi8_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            let i = j;
            // dst[i+7:i] := a[i+7:i] - b[i+7:i]
            dst[i] = a[i].wrapping_sub(b[i]);
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // --- PROOFS ---

    /// **Proof 1: Roundtrip Correctness (The Logic Check)**
    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, _mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_mask_add_epi8, _mm512_mask_add_epi8_stub)]
    #[kani::stub(_mm512_maddubs_epi16, _mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_madd_epi16, _mm512_madd_epi16_stub)]
    #[kani::stub(_mm512_sub_epi8, _mm512_sub_epi8_stub)]
    #[kani::stub(_mm512_permutexvar_epi32, _mm512_permutexvar_epi32_stub)]
    fn check_avx512_roundtrip_correctness() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; ENC_INDUCTION_LEN] = kani::any();

        let mut enc_buf = [0u8; 256];
        let mut dec_buf = [0u8; 256];

        unsafe {
            encode_slice_avx512(&config, &input, &mut enc_buf);

            let enc_len = encoded_size(ENC_INDUCTION_LEN, config.padding);
            let encoded_slice = &enc_buf[..enc_len];

            let dec_len = decode_slice_avx512(&config, encoded_slice, &mut dec_buf)
                .expect("Valid encoding failed to decode");

            assert_eq!(dec_len, ENC_INDUCTION_LEN);
            assert_eq!(&dec_buf[..dec_len], &input, "Roundtrip mismatch");
        }
    }

    /// **Proof 2: Decoder Robustness & Induction**
    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, _mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_mask_add_epi8, _mm512_mask_add_epi8_stub)]
    #[kani::stub(_mm512_maddubs_epi16, _mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_madd_epi16, _mm512_madd_epi16_stub)]
    #[kani::stub(_mm512_sub_epi8, _mm512_sub_epi8_stub)]
    fn check_avx512_decode_robustness() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };

        let input: [u8; DEC_INDUCTION_LEN] = kani::any();
        let mut output = [0u8; 256];

        // Ignore the Result — only that the call returns safely, no crash/UB.
        unsafe {
            let _ = decode_slice_avx512(&config, &input, &mut output);
        }
    }

    /// **Proof 3: Quad-Tier Loop Coverage**
    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, _mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_mask_add_epi8, _mm512_mask_add_epi8_stub)]
    #[kani::stub(_mm512_maddubs_epi16, _mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_madd_epi16, _mm512_madd_epi16_stub)]
    #[kani::stub(_mm512_sub_epi8, _mm512_sub_epi8_stub)]
    #[kani::stub(_mm512_permutexvar_epi32, _mm512_permutexvar_epi32_stub)]
    fn check_avx512_quad_tier_roundtrip() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; QUAD_ENC_INDUCTION_LEN] = kani::any();

        // encoded_size(209, true) = 280; buffers sized with margin.
        let mut enc_buf = [0u8; 320];
        let mut dec_buf = [0u8; 320];

        unsafe {
            // Both the encode and decode of this length land in the quad tier.
            encode_slice_avx512(&config, &input, &mut enc_buf);

            let enc_len = encoded_size(QUAD_ENC_INDUCTION_LEN, config.padding);
            let encoded_slice = &enc_buf[..enc_len];

            let dec_len = decode_slice_avx512(&config, encoded_slice, &mut dec_buf)
                .expect("Valid encoding failed to decode");

            assert_eq!(dec_len, QUAD_ENC_INDUCTION_LEN);
            assert_eq!(&dec_buf[..dec_len], &input, "Quad-tier roundtrip mismatch");
        }
    }
}

#[cfg(all(test, miri))]
mod miri_avx512_coverage {
    use super::*;
    use crate::simd::testutil::{check_decode, check_encode};
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};

    fn enc(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_encode(config, oracle, encode_slice_avx512, len);
    }
    fn dec(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_decode(config, oracle, decode_slice_avx512, len);
    }

    const STD: Config = Config {
        url_safe: false,
        padding: true,
    };

    // Encoder: single-vector tier is 48 bytes; quad tier needs len >= 208 (the
    // 16-byte margin means 192/193 still resolve to 0 quad iterations).
    #[test]
    fn miri_avx512_encode_scalar_fallback() {
        enc(&STD, &STANDARD, 1); // < 48 -> pure scalar
        enc(&STD, &STANDARD, 47);
    }

    #[test]
    fn miri_avx512_encode_single_vector_loop() {
        enc(&STD, &STANDARD, 48); // 1 loop
        enc(&STD, &STANDARD, 96); // 2 loops (pointer math)
        enc(&STD, &STANDARD, 49); // 1 loop + scalar tail
    }

    #[test]
    fn miri_avx512_encode_quad_vector_loop() {
        enc(&STD, &STANDARD, 192); // 0 quad iters (single-vector only)
        enc(&STD, &STANDARD, 193);
        enc(&STD, &STANDARD, 208); // 1 quad iter, 16-byte scalar remainder
        enc(&STD, &STANDARD, 384); // 2 quad iters
        enc(&STD, &STANDARD, 240); // 1 quad + 1 single, no remainder
    }

    #[test]
    fn miri_avx512_encode_url_safe() {
        enc(
            &Config {
                url_safe: true,
                padding: true,
            },
            &URL_SAFE,
            100,
        );
    }

    // Decoder: single-vector tier is 64 bytes; quad tier needs >= 260 encoded
    // bytes (raw=192 -> 256B misses it; raw=193 -> 260B hits it once).
    #[test]
    fn miri_avx512_decode_scalar_fallback() {
        dec(&STD, &STANDARD, 3); // 4 encoded chars
        dec(&STD, &STANDARD, 45); // 60 encoded chars, < 64
    }

    #[test]
    fn miri_avx512_decode_single_vector_loop() {
        dec(&STD, &STANDARD, 48); // 1 loop
        dec(&STD, &STANDARD, 96); // 2 loops
        dec(&STD, &STANDARD, 49); // 1 loop + scalar tail
    }

    #[test]
    fn miri_avx512_decode_quad_vector_loop() {
        dec(&STD, &STANDARD, 192); // 256B: single-vector tier only
        dec(&STD, &STANDARD, 193); // 260B: 1 quad iter
        dec(&STD, &STANDARD, 384); // 512B: 1 quad + single-tier handoff
    }

    #[test]
    fn miri_avx512_decode_url_safe() {
        let config = Config {
            url_safe: true,
            padding: false,
        };
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_";
        let mut dst = [0u8; 64];
        unsafe {
            decode_slice_avx512(&config, input, &mut dst).unwrap();
        }
    }

    /// An invalid byte must be caught in every tier and the scalar tail.
    #[test]
    fn miri_avx512_decode_error_detection() {
        let mut dst = [0u8; 512];
        for &(len, bad_at, where_) in &[
            (256, 255, "quad tier, last lane"),
            (64, 63, "single tier"),
            (256, 0, "quad tier, first byte"),
            (65, 64, "scalar tail"),
        ] {
            let mut input = vec![b'A'; len];
            input[bad_at] = b'$';
            let res = unsafe { decode_slice_avx512(&STD, &input, &mut dst) };
            assert!(res.is_err(), "missed invalid byte in {where_}");
        }
    }

    #[test]
    fn miri_avx512_roundtrip_standard() {
        for &len in &[48, 96, 192, 193, 240, 384] {
            enc(&STD, &STANDARD, len);
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_encode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[1, 48, 49, 96, 192, 193] {
            enc(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_decode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[3, 48, 49, 96, 192, 193] {
            dec(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_encode_url_safe_no_pad() {
        let config = Config {
            url_safe: true,
            padding: false,
        };
        for &len in &[48, 96, 192] {
            enc(&config, &URL_SAFE_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_decode_url_safe_roundtrip() {
        dec(
            &Config {
                url_safe: true,
                padding: true,
            },
            &URL_SAFE,
            100,
        );
    }
}
