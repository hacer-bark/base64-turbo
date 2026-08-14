//! AVX-512-VBMI verification: the Miri and real-hardware coverage suites.
//! Split out of the production module to keep it lean.

use super::*;

#[cfg(all(test, miri))]
mod miri_avx512_vbmi_coverage {
    use super::*;
    use crate::simd::testutil::{check_decode, check_decode_exact, check_encode};
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};

    // Same tier thresholds as the plain AVX-512 path (see
    // `super::avx512::miri_avx512_coverage`); only the LUT lookup differs.
    fn enc(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_encode(config, oracle, encode_slice_avx512_vbmi, len);
    }
    fn dec(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_decode(config, oracle, decode_slice_avx512_vbmi, len);
    }
    fn exact(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_decode_exact(config, oracle, decode_slice_avx512_vbmi, len);
    }

    const STD: Config = Config {
        url_safe: false,
        padding: true,
    };
    const URL: Config = Config {
        url_safe: true,
        padding: true,
    };
    const NO_PAD: Config = Config {
        url_safe: false,
        padding: false,
    };
    const NO_PAD_URL: Config = Config {
        url_safe: true,
        padding: false,
    };

    #[test]
    fn miri_avx512_vbmi_encode_scalar_fallback() {
        enc(&STD, &STANDARD, 1);
        enc(&STD, &STANDARD, 47);
    }

    #[test]
    fn miri_avx512_vbmi_encode_single_vector_loop() {
        enc(&STD, &STANDARD, 48);
        enc(&STD, &STANDARD, 96);
        enc(&STD, &STANDARD, 49);
    }

    #[test]
    fn miri_avx512_vbmi_encode_quad_vector_loop() {
        enc(&STD, &STANDARD, 192); // 0 quad iters
        enc(&STD, &STANDARD, 193);
        enc(&STD, &STANDARD, 208); // 1 quad iter
        enc(&STD, &STANDARD, 384); // 2 quad iters
        enc(&STD, &STANDARD, 240); // 1 quad + 1 single
    }

    #[test]
    fn miri_avx512_vbmi_encode_url_safe() {
        enc(&URL, &URL_SAFE, 100);
    }

    #[test]
    fn miri_avx512_vbmi_decode_scalar_fallback() {
        dec(&STD, &STANDARD, 3);
        dec(&STD, &STANDARD, 45);
    }

    #[test]
    fn miri_avx512_vbmi_decode_single_vector_loop() {
        dec(&STD, &STANDARD, 48);
        dec(&STD, &STANDARD, 96);
        dec(&STD, &STANDARD, 49);
    }

    #[test]
    fn miri_avx512_vbmi_decode_quad_vector_loop() {
        dec(&STD, &STANDARD, 192); // 256B: misses quad loop
        dec(&STD, &STANDARD, 193); // 260B: 1 quad iter
        dec(&STD, &STANDARD, 384);
    }

    #[test]
    fn miri_avx512_vbmi_decode_url_safe() {
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_";
        let mut dst = [0u8; 64];
        unsafe {
            decode_slice_avx512_vbmi(&NO_PAD_URL, input, &mut dst).unwrap();
        }
    }

    /// Invalid bytes must be caught in every tier and the scalar tail. Unlike
    /// the other paths, VBMI has a second rejection route: bytes >= 128 alias
    /// into the LUT via bit 6, so `is_high_bit` must reject them.
    #[test]
    fn miri_avx512_vbmi_decode_error_detection() {
        let mut dst = [0u8; 512];
        for &(len, bad_at, byte, where_) in &[
            (256, 255, b'$', "sentinel, quad tier"),
            (64, 63, b'?', "sentinel, single tier"),
            (256, 0, 0x80u8, "high bit, quad tier"),
            (64, 0, 0xFF, "high bit, single tier"),
            (65, 64, b'?', "scalar tail"),
        ] {
            let mut input = vec![b'A'; len];
            input[bad_at] = byte;
            let res = unsafe { decode_slice_avx512_vbmi(&STD, &input, &mut dst) };
            assert!(res.is_err(), "missed invalid byte in {where_}");
        }
    }

    #[test]
    fn miri_avx512_vbmi_roundtrip_standard() {
        for &len in &[48, 96, 192, 193, 240, 384] {
            enc(&STD, &STANDARD, len);
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_encode_no_padding() {
        for &len in &[1, 48, 49, 96, 192, 193] {
            enc(&NO_PAD, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_no_padding() {
        for &len in &[3, 48, 49, 96, 192, 193] {
            dec(&NO_PAD, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_encode_url_safe_no_pad() {
        for &len in &[48, 96, 192] {
            enc(&NO_PAD_URL, &URL_SAFE_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_url_safe_roundtrip() {
        dec(&URL, &URL_SAFE, 100);
    }

    /// Masked-store regression: every chunk-boundary length must decode into an
    /// exactly-sized buffer without overrunning, for both alphabets and
    /// padded/unpadded input.
    #[test]
    fn miri_avx512_vbmi_decode_exact_buffer_boundaries() {
        for &len in &[3, 45, 48, 96, 192, 193, 240, 384, 1000, 1001] {
            exact(&STD, &STANDARD, len);
            exact(&URL, &URL_SAFE, len);
            exact(&NO_PAD, &STANDARD_NO_PAD, len);
        }
    }
}

#[cfg(all(test, not(miri)))]
mod avx512_vbmi_hardware_coverage {
    use super::*;
    use crate::simd::testutil::check_decode_exact;
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE};

    /// The same exact-buffer masked-store regression the Miri suite runs, but on
    /// real AVX-512-VBMI silicon (skipped when the host CPU lacks it).
    #[test]
    fn hw_avx512_vbmi_decode_exact_buffer_boundaries() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi"))
        {
            eprintln!("skipping: host CPU lacks AVX-512-VBMI");
            return;
        }

        let standard = Config {
            url_safe: false,
            padding: true,
        };
        let url_safe = Config {
            url_safe: true,
            padding: true,
        };
        let no_pad = Config {
            url_safe: false,
            padding: false,
        };

        for &len in &[3, 45, 48, 96, 192, 193, 240, 384, 1000, 1001] {
            check_decode_exact(&standard, &STANDARD, decode_slice_avx512_vbmi, len);
            check_decode_exact(&url_safe, &URL_SAFE, decode_slice_avx512_vbmi, len);
            check_decode_exact(&no_pad, &STANDARD_NO_PAD, decode_slice_avx512_vbmi, len);
        }
    }
}
