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
    /// 1. Encodes `input` using Reference crate.
    /// 2. Encodes `input` using Turbo.
    /// 3. Asserts Equality.
    /// 4. Decodes Reference output using Turbo.
    /// 5. Asserts Round Trip.
    #[track_caller] // Shows line number of the failure in the loop
    fn assert_oracle_match(
        input: &[u8], 
        turbo_engine: &Engine, 
        ref_engine: &base64::engine::GeneralPurpose,
    ) {
        // 1. Truth
        let expected_encoded = ref_engine.encode(input);

        // 2. Turbo Encode (Allocating)
        let turbo_encoded = turbo_engine.encode(input);

        assert_eq!(turbo_encoded, expected_encoded, "Encode mismatch. Len: {}", input.len());

        // 3. Turbo Decode (Allocating)
        let turbo_decoded = turbo_engine.decode(&expected_encoded)
            .expect("Turbo failed to decode valid reference output");

        assert_eq!(turbo_decoded, input, "Decode mismatch. Len: {}", input.len());

        // 4. Turbo Zero-Allocation API Check (HFT Path)
        // We verify that the slice-based API produces the exact same result as the allocating one

        // Encode_into
        let required_len = (input.len() + 2) / 3 * 4; // Base64 expansion
        // Add extra buffer to ensure we don't write past reported length
        let mut enc_buf = vec![0u8; required_len + 10]; 

        let written_enc = turbo_engine.encode_into(input, &mut enc_buf).unwrap();
        assert_eq!(written_enc, expected_encoded.len(), "Zero-alloc encode length mismatch");
        assert_eq!(&enc_buf[..written_enc], expected_encoded.as_bytes(), "Zero-alloc encode content mismatch");

        // Decode_into
        let mut dec_buf = vec![0u8; input.len() + 10];
        let written_dec =turbo_engine.decode_into(expected_encoded.as_bytes(), &mut dec_buf).expect("Zero-alloc decode failed");

        assert_eq!(written_dec, input.len(), "Zero-alloc decode length mismatch");
        assert_eq!(&dec_buf[..written_dec], input, "Zero-alloc decode content mismatch");
    }

    // --- 1. Basic Correctness ---

    #[test]
    fn test_oracle_standard_exhaustive_small() {
        // Test 0 to 1024 bytes inclusive.
        // This covers Scalar unroll loops (8 bytes), SSSE3 (16 bytes), and AVX2 (32 bytes)
        // boundaries multiple times.
        for i in 0..=1024 {
            let data = random_bytes(i);
            assert_oracle_match(&data, &STANDARD, &REF_STANDARD);
        }
    }

    #[test]
    fn test_oracle_standard_fuzz_medium() {
        let mut rng = rng();
        // 10,000 iterations of random sizes up to 64KB
        for _ in 0..10_000 {
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
        // Even if SIMD is disabled via Cargo features, these tests ensure the Scalar
        // fallback logic handles chunking correctly.
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

    // --- 4. Parallelism Tests (Conditional) ---

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_correctness_large() {
        // 10MB input ensures Rayon kicks in (threshold is usually ~512KB)
        let size = 10 * 1024 * 1024; 
        let data = random_bytes(size);

        // We assume REF_STANDARD is correct. We verify STANDARD (Turbo) matches.
        // Note: Reference crate is single-threaded, so this might take a moment.
        let ref_encoded = REF_STANDARD.encode(&data);
        let turbo_encoded = STANDARD.encode(&data);

        assert_eq!(turbo_encoded, ref_encoded, "Parallel Encoding Mismatch");

        // Decode back
        let turbo_decoded = STANDARD.decode(&turbo_encoded).expect("Parallel Decoding Failed");

        assert_eq!(turbo_decoded, data, "Parallel Round-trip Mismatch");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_threshold_boundary() {
        // Test right around the threshold where it might switch from Scalar/SIMD to Rayon
        let threshold = 512 * 1024; // 512 KB

        for size in [threshold - 1, threshold, threshold + 1] {
            let data = random_bytes(size);
            assert_oracle_match(&data, &STANDARD, &REF_STANDARD);
        }
    }

    // --- 5. Negative / Security Tests ---

    #[test]
    fn test_reject_invalid_chars() {
        let inputs = vec![
            "Abc-",      // Invalid char '-' for Standard
            "Abc_",      // Invalid char '_' for Standard
            "Abc ",      // Space
            "Abc\n",     // Newline
            "Abc\0",     // Null byte
            "Abc!",      // Garbage
        ];

        for inp in inputs {
            assert!(
                STANDARD.decode(inp).is_err(), 
                "Standard Engine failed to reject invalid input: {:?}", inp
            );
        }
    }

    #[test]
    fn test_reject_invalid_padding() {
        // Base64 length must be % 4 == 0.
        // If it has padding, '=' must only be at the end (max 2).
        let bad_inputs = vec![
            "A", "AA", "AAA",      // Wrong length
            "AAAA=",               // Length 5
            "AA=A",                // Padding in middle
            "A===",                // Too much padding
            "====",                // Only padding
        ];

        for bad in bad_inputs {
            assert!(
                STANDARD.decode(bad).is_err(),
                "Failed to reject bad padding/length: '{}'", bad
            );
        }
    }

    #[test]
    fn test_buffer_overflow_protection_in_safe_api() {
        // Verify that decode doesn't panic if we feed it garbage that hypothetically
        // looks like it writes a lot.
        let data = vec![b'A'; 1024];
        let _ = STANDARD.decode(&data); // Should just work or err, not panic/segfault
    }
}
