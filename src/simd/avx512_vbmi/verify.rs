//! AVX-512-VBMI verification: the Miri and real-hardware coverage suites.
//! Split out of the production module to keep it lean.

use super::*;

#[cfg(all(test, miri))]
mod miri_avx512_vbmi_coverage {
    use super::*;
    use crate::simd::testutil::{check_decode, check_decode_exact, check_encode};
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};

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

    /// Tier boundaries, encode. The vector path now runs down to 3 bytes, so
    /// scalar only ever sees a final 1-2 byte group: quad at >= 256, single at
    /// >= 64 (a plain load reads 64 to consume 48), masked below that.
    #[test]
    fn miri_avx512_vbmi_encode_tier_boundaries() {
        for &len in &[0, 1, 2, 3, 4, 5, 47, 48, 49, 63, 64, 65, 66, 95, 96] {
            enc(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_encode_quad_tier_boundaries() {
        for &len in &[190, 192, 255, 256, 257, 259, 448] {
            enc(&STD, &STANDARD, len);
        }
    }

    /// The masked tier runs twice whenever the remainder exceeds 48 bytes,
    /// which is every length whose `len % 48` lands in 49..63 after the single
    /// tier stops.
    #[test]
    fn miri_avx512_vbmi_encode_masked_tier_two_passes() {
        for &len in &[51, 54, 60, 62, 63] {
            enc(&STD, &STANDARD, len);
            enc(&NO_PAD, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_encode_url_safe() {
        enc(&URL, &URL_SAFE, 100);
        for &len in &[3, 47, 63, 100, 259] {
            enc(&NO_PAD_URL, &URL_SAFE_NO_PAD, len);
        }
    }

    /// Tier boundaries, decode, in *decoded* byte lengths. The character
    /// thresholds are 260 / 68 / 8; every tier stops 4 characters short so the
    /// only group that may carry '=' is always the scalar tail's.
    #[test]
    fn miri_avx512_vbmi_decode_tier_boundaries() {
        for &len in &[0, 1, 2, 3, 4, 5, 6, 45, 48, 49, 50, 51, 52, 66, 96] {
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_quad_tier_boundaries() {
        for &len in &[144, 192, 193, 194, 195, 196, 255, 300] {
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_url_safe() {
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_";
        let mut dst = [0u8; 64];
        unsafe {
            decode_slice_avx512_vbmi(&NO_PAD_URL, input, &mut dst).unwrap();
        }
        dec(&URL, &URL_SAFE, 100);
    }

    /// Invalid bytes must be caught in every tier and the scalar tail. Unlike
    /// the other paths, VBMI has a second rejection route: bytes >= 128 alias
    /// into the LUT via bit 6, so the accumulator must catch them too. The
    /// accumulator spans the whole call, so a byte anywhere in the vector
    /// region still fails the single test after the loops.
    #[test]
    fn miri_avx512_vbmi_decode_error_detection() {
        let mut dst = [0u8; 512];
        for &(len, bad_at, byte, where_) in &[
            (260, 0, b'$', "sentinel, quad tier, first byte"),
            (260, 255, b'$', "sentinel, quad tier, last byte"),
            (68, 63, b'?', "sentinel, single tier"),
            (12, 5, b'?', "sentinel, masked tier"),
            (260, 0, 0x80u8, "high bit, quad tier"),
            (68, 0, 0xFF, "high bit, single tier"),
            (12, 0, 0x80u8, "high bit, masked tier"),
            (68, 65, b'?', "scalar tail"),
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
    fn miri_avx512_vbmi_no_padding() {
        for &len in &[1, 3, 48, 49, 63, 96, 192, 193] {
            enc(&NO_PAD, &STANDARD_NO_PAD, len);
        }
        for &len in &[3, 6, 48, 49, 51, 96, 192, 193] {
            dec(&NO_PAD, &STANDARD_NO_PAD, len);
        }
    }

    /// Masked-store regression: every chunk-boundary length must decode into an
    /// exactly-sized buffer without overrunning, for both alphabets and
    /// padded/unpadded input.
    #[test]
    fn miri_avx512_vbmi_decode_exact_buffer_boundaries() {
        for &len in &[3, 6, 45, 48, 51, 96, 192, 193, 195, 240, 384, 1000, 1001] {
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

        for &len in &[3, 6, 45, 48, 51, 96, 192, 193, 195, 240, 384, 1000, 1001] {
            check_decode_exact(&standard, &STANDARD, decode_slice_avx512_vbmi, len);
            check_decode_exact(&url_safe, &URL_SAFE, decode_slice_avx512_vbmi, len);
            check_decode_exact(&no_pad, &STANDARD_NO_PAD, decode_slice_avx512_vbmi, len);
        }
    }
}
