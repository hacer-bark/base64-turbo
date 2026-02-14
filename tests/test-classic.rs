use base64_turbo::{Engine, Error, STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};

// Reference Crate for Oracle Verification
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

// ======================================================================
// Helpers
// ======================================================================

fn random_bytes(len: usize) -> Vec<u8> {
    let mut rng = rng();
    (0..len).map(|_| rng.random()).collect()
}

/// The "Oracle" Test.
/// Verifies that base64-turbo output exactly matches the 'base64' crate.
#[track_caller]
fn assert_oracle_match(
    input: &[u8],
    turbo_engine: &Engine,
    ref_engine: &base64::engine::GeneralPurpose,
) {
    // 1. Reference Truth
    let expected_encoded = ref_engine.encode(input);

    // 2. Test Zero-Allocation API (Slice)
    let mut enc_buf = vec![0u8; turbo_engine.encoded_len(input.len())];
    let enc_len = turbo_engine.encode_into(input, &mut enc_buf).expect("encode_into failed");
    assert_eq!(&enc_buf[..enc_len], expected_encoded.as_bytes(), "Slice Encode mismatch");

    // 3. Test Allocating API (String) [If std enabled]
    #[cfg(feature = "std")]
    {
        let alloc_str = turbo_engine.encode(input);
        assert_eq!(alloc_str, expected_encoded, "Allocating Encode mismatch");
    }

    // 4. Test Zero-Allocation Decode (Slice)
    // Note: We allocate based on estimate, but verify exact write length
    let mut dec_buf = vec![0u8; turbo_engine.estimate_decoded_len(expected_encoded.len())];
    let dec_len = turbo_engine.decode_into(expected_encoded.as_bytes(), &mut dec_buf)
        .expect("decode_into failed");
    assert_eq!(&dec_buf[..dec_len], input, "Slice Decode mismatch");

    // 5. Test Allocating Decode (Vec) [If std enabled]
    #[cfg(feature = "std")]
    {
        let alloc_vec = turbo_engine.decode(&expected_encoded).expect("decode failed");
        assert_eq!(alloc_vec, input, "Allocating Decode mismatch");
    }
}

// ======================================================================
// 1. Coverage: Basic Logic & Oracle Matching
// ======================================================================

#[test]
fn test_oracle_standard_exhaustive_small() {
    // Covers 0..92 to hit all SIMD mask boundaries, alignment issues, and scalar fallbacks.
    // This implicitly covers `encoded_len` and `estimate_decoded_len` correctness via helpers.
    for i in 0..=92 {
        let data = random_bytes(i);
        assert_oracle_match(&data, &STANDARD, &REF_STANDARD);
    }
}

#[test]
#[cfg(not(miri))]
fn test_oracle_fuzz_large() {
    // Random sizes up to 64KB to trigger AVX2/AVX512 loops multiple times.
    let mut rng = rng();
    for _ in 0..100 {
        let len = rng.random_range(1024..65536);
        let data = random_bytes(len);
        assert_oracle_match(&data, &STANDARD, &REF_STANDARD);
    }
}

#[test]
fn test_oracle_configs() {
    // Verify URL_SAFE and NO_PAD variants logic
    let mut rng = rng();
    for _ in 0..25 {
        let len = rng.random_range(1..512);
        let data = random_bytes(len);
        assert_oracle_match(&data, &STANDARD_NO_PAD, &REF_STANDARD_NO_PAD);
        assert_oracle_match(&data, &URL_SAFE, &REF_URL_SAFE);
        assert_oracle_match(&data, &URL_SAFE_NO_PAD, &REF_URL_SAFE_NO_PAD);
    }
}

// ======================================================================
// 2. Coverage: API Constraints & Error Handling
// ======================================================================

#[test]
fn test_reject_invalid_chars() {
    let bad_inputs = ["Abc!", "Ab c", "Abc\0", "Abc-", "Abc_"];
    let mut buf = [0u8; 100];
    for bad in bad_inputs {
        // STANDARD engine should reject '-' and '_'
        assert_eq!(
            STANDARD.decode_into(bad, &mut buf), 
            Err(Error::InvalidCharacter),
            "Failed to reject: {:?}", bad,
        );
    }
}

#[test]
fn test_reject_invalid_length_padding() {
    let inputs = ["A", "AA", "AAA", "AAAA=", "A===", "===="];
    let mut buf = [0u8; 100];
    for inp in inputs {
        let res = STANDARD.decode_into(inp, &mut buf);
        // Can be InvalidLength or InvalidCharacter depending on implementation specifics
        assert!(res.is_err(), "Should fail on invalid padding/length: {}", inp);
    }
}

// ======================================================================
// 3. Coverage: Unstable API (Feature Gated)
// ======================================================================

#[test]
#[cfg(feature = "unstable")]
#[cfg(not(miri))]
fn test_unstable_apis() {
    let input = random_bytes(1024);
    let expected = REF_STANDARD.encode(&input);

    // --- Scalar (Always Available) ---
    unsafe {
        let mut dst = vec![0u8; STANDARD.encoded_len(input.len())];
        STANDARD.encode_scalar(&input, &mut dst);
        assert_eq!(&dst, expected.as_bytes(), "Scalar Unsafe Encode");

        let mut dec = vec![0u8; STANDARD.estimate_decoded_len(dst.len())];
        let len = STANDARD.decode_scalar(&dst, &mut dec).unwrap();
        assert_eq!(&dec[..len], &input, "Scalar Unsafe Decode");
    }

    // --- SSE4.1 ---
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("sse4.1") {
        unsafe {
            let mut dst = vec![0u8; STANDARD.encoded_len(input.len())];
            STANDARD.encode_sse4(&input, &mut dst);
            assert_eq!(&dst, expected.as_bytes(), "SSE4.1 Unsafe Encode");

            let mut dec = vec![0u8; STANDARD.estimate_decoded_len(dst.len())];
            let len = STANDARD.decode_sse4(&dst, &mut dec).unwrap();
            assert_eq!(&dec[..len], &input, "SSE4.1 Unsafe Decode");
        }
    } else {
        println!("Skipping SSE4.1 Unstable test (hardware unsupported)");
    }

    // --- AVX2 ---
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            let mut dst = vec![0u8; STANDARD.encoded_len(input.len())];
            STANDARD.encode_avx2(&input, &mut dst);
            assert_eq!(&dst, expected.as_bytes(), "AVX2 Unsafe Encode");

            let mut dec = vec![0u8; STANDARD.estimate_decoded_len(dst.len())];
            let len = STANDARD.decode_avx2(&dst, &mut dec).unwrap();
            assert_eq!(&dec[..len], &input, "AVX2 Unsafe Decode");
        }
    } else {
        println!("Skipping AVX2 Unstable test (hardware unsupported)");
    }
}
