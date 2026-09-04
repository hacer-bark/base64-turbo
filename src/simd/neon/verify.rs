//! NEON verification: the Miri coverage suite. Split out of the production
//! module purely to keep it lean.

use super::*;

#[cfg(all(test, miri))]
mod miri_neon_coverage {
    use super::*;
    use crate::simd::testutil::{check_decode, check_encode};
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE};

    fn enc(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_encode(config, oracle, encode_slice_neon, len);
    }
    fn dec(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_decode(config, oracle, decode_slice_neon, len);
    }

    const STD: Config = Config {
        alphabet: crate::alphabet::builtin(false),
        padding: true,
    };

    // Encoder tiers: single-vector is 12 bytes, quad is 48.
    #[test]
    fn miri_neon_encode_scalar_fallback() {
        enc(&STD, &STANDARD, 1); // < 12 -> pure scalar
        enc(&STD, &STANDARD, 11);
    }

    #[test]
    fn miri_neon_encode_single_vector_loop() {
        enc(&STD, &STANDARD, 12); // 1 loop
        enc(&STD, &STANDARD, 24); // 2 loops
        enc(&STD, &STANDARD, 13); // 1 loop + scalar
    }

    #[test]
    fn miri_neon_encode_quad_vector_loop() {
        enc(&STD, &STANDARD, 48); // 1 quad
        enc(&STD, &STANDARD, 96); // 2 quads
        enc(&STD, &STANDARD, 49); // 1 quad + scalar
        enc(&STD, &STANDARD, 60); // 1 quad + 1 single
    }

    #[test]
    fn miri_neon_encode_url_safe() {
        enc(
            &Config {
                alphabet: crate::alphabet::builtin(true),
                padding: true,
            },
            &URL_SAFE,
            50,
        );
    }

    // Decoder tiers: single-vector is 16 bytes, quad is 64.
    #[test]
    fn miri_neon_decode_scalar_fallback() {
        dec(&STD, &STANDARD, 3); // 4 chars
        dec(&STD, &STANDARD, 9); // 12 chars, < 16
    }

    #[test]
    fn miri_neon_decode_single_vector_loop() {
        dec(&STD, &STANDARD, 12); // 1 loop
        dec(&STD, &STANDARD, 24); // 2 loops
        dec(&STD, &STANDARD, 13); // 1 loop + scalar
    }

    #[test]
    fn miri_neon_decode_quad_vector_loop() {
        dec(&STD, &STANDARD, 48); // 1 quad
        dec(&STD, &STANDARD, 96); // 2 quads
        dec(&STD, &STANDARD, 49); // 1 quad + remainder
    }

    #[test]
    fn miri_neon_decode_url_safe() {
        let config = Config {
            alphabet: crate::alphabet::builtin(true),
            padding: false,
        };
        let input = b"-_-_-_-_-_-_-_-_"; // 16 bytes
        let mut dst = [0u8; 16];
        unsafe {
            decode_slice_neon(&config, input, &mut dst).unwrap();
        }
    }

    /// An invalid byte must be caught in every tier and the scalar tail.
    #[test]
    fn miri_neon_decode_error_detection() {
        let mut dst = [0u8; 128];
        for &(len, bad_at, where_) in &[
            (64, 63, "quad tier, last lane"),
            (16, 15, "single tier"),
            (64, 0, "quad tier, first byte"),
            (17, 16, "scalar tail"),
        ] {
            let mut input = vec![b'A'; len];
            input[bad_at] = b'$';
            let res = unsafe { decode_slice_neon(&STD, &input, &mut dst) };
            assert!(res.is_err(), "missed invalid byte in {where_}");
        }
    }

    #[test]
    fn miri_neon_roundtrip_standard() {
        for &len in &[12, 24, 48, 49, 60, 96] {
            enc(&STD, &STANDARD, len);
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_neon_encode_no_padding() {
        let config = Config {
            alphabet: crate::alphabet::builtin(false),
            padding: false,
        };
        for &len in &[1, 12, 13, 24, 48, 49] {
            enc(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_neon_decode_no_padding() {
        let config = Config {
            alphabet: crate::alphabet::builtin(false),
            padding: false,
        };
        for &len in &[3, 12, 13, 24, 48, 49] {
            dec(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_neon_decode_url_safe_padded() {
        dec(
            &Config {
                alphabet: crate::alphabet::builtin(true),
                padding: true,
            },
            &URL_SAFE,
            50,
        );
    }
}
