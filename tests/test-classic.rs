#[cfg(all(test, not(miri)))]
mod exhaustive_tests {
    // --- Imports ---
    use base64_turbo::{Engine, STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};

    // Reference Crate Aliases
    use base64::{
        Engine as _,
        engine::general_purpose::{
            STANDARD as REF_STANDARD,
            STANDARD_NO_PAD as REF_STANDARD_NO_PAD,
            URL_SAFE as REF_URL_SAFE,
            URL_SAFE_NO_PAD as REF_URL_SAFE_NO_PAD,
        }
    };
    use rand::{Rng, rng};

    // --- Oracle Helpers ---

    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    /// The "Oracle" Test.
    #[track_caller]
    fn assert_oracle_match(
        input: &[u8],
        turbo_engine: &Engine,
        ref_engine: &base64::engine::GeneralPurpose,
    ) {
        // 1. Truth (Reference crate)
        let expected_encoded = ref_engine.encode(input);

        // 2. Turbo Encode (Manual Buffer Management)
        let mut enc_buf = vec![0u8; turbo_engine.encoded_len(input.len())];

        turbo_engine.encode_into(input, &mut enc_buf).expect("Turbo encode_into failed");

        // Verify content match
        assert_eq!(enc_buf, expected_encoded.as_bytes(), "Encode content mismatch. Input Len: {}", input.len());

        // 3. Turbo Decode (Manual Buffer Management)
        let mut dec_buf = vec![0u8; turbo_engine.estimate_decoded_len(expected_encoded.len())];

        let written_dec = turbo_engine.decode_into(expected_encoded.as_bytes(), &mut dec_buf)
            .expect("Turbo decode_into failed on valid reference output");

        // Verify Round Trip
        let turbo_decoded_slice = &dec_buf[..written_dec];
        assert_eq!(turbo_decoded_slice, input, "Decode content mismatch / Round trip failed");
    }

    // --- 1. Basic Correctness ---

    #[test]
    fn test_oracle_standard_exhaustive_small() {
        // Test 0 to 1024 bytes inclusive.
        for i in 0..=1024 {
            let data = random_bytes(i);
            assert_oracle_match(&data, &STANDARD, &REF_STANDARD);
        }
    }

    #[test]
    fn test_oracle_standard_fuzz_medium() {
        let mut rng = rng();
        // 1,000 iterations of random sizes up to 64KB
        for _ in 0..1_000 {
            let len = rng.random_range(1025..65536);
            let data = random_bytes(len);
            assert_oracle_match(&data, &STANDARD, &REF_STANDARD);
        }
    }

    // --- 2. Configuration Variants ---

    #[test]
    fn test_oracle_no_pad_logic() {
        // Validates bit-buffering logic when padding is disabled
        for i in 0..200 {
            let data = random_bytes(i);
            assert_oracle_match(&data, &STANDARD_NO_PAD, &REF_STANDARD_NO_PAD);
        }
    }

    #[test]
    fn test_oracle_url_safe_alphabet() {
        // Edge case: produce bytes that result in '+' (0xFB) or '/' (0xF0 range)
        let tricky_bytes = vec![
            0xFB, 0xFF, 0xBF, // High bits often trigger +
            0x3E, 0x3F,       // Boundaries
            0x00, 0x00, 0x00,
            0xFF, 0xFF, 0xFF
        ];
        assert_oracle_match(&tricky_bytes, &URL_SAFE, &REF_URL_SAFE);

        let mut rng = rng();
        for _ in 0..500 {
            let len = rng.random_range(1..2048);
            let data = random_bytes(len);
            assert_oracle_match(&data, &URL_SAFE, &REF_URL_SAFE);
        }
    }

    // --- 3. Architecture/SIMD Boundary Tests ---

    #[test]
    fn test_simd_scalar_transition_boundaries() {
        // We explicitly test lengths that sit on the boundaries of SIMD chunks.
        let boundaries = [
            7, 8, 9,           // Scalar Unroll (8 bytes)
            15, 16, 17,        // SSSE3 (16 bytes)
            23, 24, 25,        // Base64 Block alignment (3 bytes in -> 4 chars out)
            31, 32, 33,        // AVX2 (32 bytes)
            63, 64, 65,        // Cache Line / AVX512 (64 bytes)
            255, 256, 257,     // u8 Index overflow edge case
        ];

        for &len in &boundaries {
            let data = random_bytes(len);
            assert_oracle_match(&data, &STANDARD, &REF_STANDARD);
            assert_oracle_match(&data, &URL_SAFE_NO_PAD, &REF_URL_SAFE_NO_PAD);
        }
    }

    // --- 4. Negative / Security Tests ---

    #[test]
    fn test_reject_invalid_chars() {
        let inputs = vec![
            "Abc-",  // Invalid char '-' for Standard
            "Abc_",  // Invalid char '_' for Standard
            "Abc ",  // Space
            "Abc\n", // Newline
            "Abc\0", // Null byte
            "Abc!",  // Garbage
        ];

        // Reuse a buffer for decoding attempts
        let mut buf = [0u8; 100];

        for inp in inputs {
            assert!(
                STANDARD.decode_into(inp.as_bytes(), &mut buf).is_err(),
                "Standard Engine failed to reject invalid input: {:?}", inp
            );
        }
    }

    #[test]
    fn test_reject_invalid_padding() {
        let bad_inputs = vec![
            "A", "AA", "AAA",   // Wrong length
            "AAAA=",            // Length 5
            "AA=A",             // Padding in middle
            "A===",             // Too much padding
            "====",             // Only padding
        ];

        let mut buf = [0u8; 100];

        for bad in bad_inputs {
            assert!(
                STANDARD.decode_into(bad.as_bytes(), &mut buf).is_err(),
                "Failed to reject bad padding/length: '{}'", bad
            );
        }
    }

    #[test]
    fn test_buffer_overflow_protection_in_safe_api() {
        // Verify that decode doesn't panic if we feed it garbage that hypothetically
        // looks like it writes a lot.
        let data = vec![b'A'; 1024];
        let mut buf = vec![0u8; 1024];

        // This should return Ok (if valid base64) or Err, but NOT Panic.
        // We aren't checking the result, just that it doesn't crash the test runner.
        let _ = STANDARD.decode_into(&data, &mut buf);
    }
}
