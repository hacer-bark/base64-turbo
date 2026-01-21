#[cfg(all(test, feature = "std", not(miri)))]
mod std_alloc_tests {
    use base64_turbo::{STANDARD, URL_SAFE};
    use base64::{
        Engine as _, 
        engine::general_purpose::{STANDARD as REF_STANDARD, URL_SAFE as REF_URL_SAFE}
    };
    use rand::{Rng, rng};

    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    #[test]
    fn test_alloc_encode_correctness() {
        // Verifies Engine::encode(input) -> String
        let mut rng = rng();
        for _ in 0..1000 {
            let len = rng.random_range(0..1024);
            let data = random_bytes(len);

            let turbo_str = STANDARD.encode(&data);
            let ref_str = REF_STANDARD.encode(&data);

            assert_eq!(turbo_str, ref_str, "Allocating encode mismatch at len {}", len);
        }
    }

    #[test]
    fn test_alloc_decode_correctness() {
        // Verifies Engine::decode(input) -> Result<Vec<u8>, _>
        let mut rng = rng();
        for _ in 0..1000 {
            let len = rng.random_range(0..1024);
            let data = random_bytes(len);
            
            // Get valid base64 first
            let encoded = REF_STANDARD.encode(&data);

            let turbo_vec = STANDARD.decode(&encoded)
                .expect("Allocating decode failed on valid input");

            assert_eq!(turbo_vec, data, "Allocating decode mismatch at len {}", len);
        }
    }

    #[test]
    fn test_alloc_url_safe_roundtrip() {
        // Specific test for URL Safe alphabet allocation
        let data = vec![0xFB, 0xFF, 0xBF, 0x3E, 0x3F]; // Triggers + and / in standard, - and _ in URL

        let encoded = URL_SAFE.encode(&data);
        let ref_encoded = REF_URL_SAFE.encode(&data);

        assert_eq!(encoded, ref_encoded, "URL Safe encode mismatch");

        let decoded = URL_SAFE.decode(&encoded).unwrap();
        assert_eq!(decoded, data, "URL Safe decode roundtrip failed");
    }

    #[test]
    fn test_alloc_decode_errors() {
        // Ensure the allocating decode returns errors properly instead of panicking
        let invalid_inputs = vec![
            "A", "AA", "AAA",   // Invalid Length
            "@@@@",             // Invalid Char
            "A===",             // Invalid Padding
        ];

        for input in invalid_inputs {
            let res = STANDARD.decode(input);
            assert!(res.is_err(), "Allocating decode should have failed for input: '{}'", input);
        }
    }
}
