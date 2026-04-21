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
use rand::{RngExt, rng};

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
// 2. Coverage: Empty Input (len == 0 early return)
// ======================================================================

#[test]
fn test_empty_input() {
    let empty: &[u8] = b"";
    let mut enc_buf = [0u8; 16];
    let mut dec_buf = [0u8; 16];

    // Encode empty -> 0 bytes written
    let enc_len = STANDARD.encode_into(empty, &mut enc_buf).unwrap();
    assert_eq!(enc_len, 0);

    // Decode empty -> 0 bytes written
    let dec_len = STANDARD.decode_into(empty, &mut dec_buf).unwrap();
    assert_eq!(dec_len, 0);

    // All configs
    for engine in &[STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD] {
        assert_eq!(engine.encode_into(empty, &mut enc_buf).unwrap(), 0);
        assert_eq!(engine.decode_into(empty, &mut dec_buf).unwrap(), 0);
    }

    // Allocating APIs
    #[cfg(feature = "std")]
    {
        assert_eq!(STANDARD.encode(b""), "");
        assert_eq!(STANDARD.decode("").unwrap(), Vec::<u8>::new());
    }
}

// ======================================================================
// 3. Coverage: encoded_len & estimate_decoded_len correctness
// ======================================================================

#[test]
fn test_encoded_len_correctness() {
    // Padded encoding: ceil(n/3) * 4
    assert_eq!(STANDARD.encoded_len(0), 0);
    assert_eq!(STANDARD.encoded_len(1), 4);
    assert_eq!(STANDARD.encoded_len(2), 4);
    assert_eq!(STANDARD.encoded_len(3), 4);
    assert_eq!(STANDARD.encoded_len(4), 8);
    assert_eq!(STANDARD.encoded_len(5), 8);
    assert_eq!(STANDARD.encoded_len(6), 8);
    assert_eq!(STANDARD.encoded_len(7), 12);

    // No-pad encoding: ceil(n*4/3)
    assert_eq!(STANDARD_NO_PAD.encoded_len(0), 0);
    assert_eq!(STANDARD_NO_PAD.encoded_len(1), 2);
    assert_eq!(STANDARD_NO_PAD.encoded_len(2), 3);
    assert_eq!(STANDARD_NO_PAD.encoded_len(3), 4);
    assert_eq!(STANDARD_NO_PAD.encoded_len(4), 6);
    assert_eq!(STANDARD_NO_PAD.encoded_len(5), 7);
    assert_eq!(STANDARD_NO_PAD.encoded_len(6), 8);
    assert_eq!(STANDARD_NO_PAD.encoded_len(7), 10);

    // URL_SAFE uses same math as STANDARD (just different alphabet)
    assert_eq!(URL_SAFE.encoded_len(10), STANDARD.encoded_len(10));
    assert_eq!(URL_SAFE_NO_PAD.encoded_len(10), STANDARD_NO_PAD.encoded_len(10));
}

#[test]
fn test_estimate_decoded_len() {
    // (input_len / 4 + 1) * 3
    assert_eq!(STANDARD.estimate_decoded_len(0), 3);
    assert_eq!(STANDARD.estimate_decoded_len(4), 6);
    assert_eq!(STANDARD.estimate_decoded_len(8), 9);
    assert_eq!(STANDARD.estimate_decoded_len(12), 12);

    // Sanity: estimate should always be >= actual
    for n in 0..=50 {
        let data = random_bytes(n);
        let encoded = REF_STANDARD.encode(&data);
        assert!(STANDARD.estimate_decoded_len(encoded.len()) >= n,
            "Estimate too small for n={}", n);
    }
}

// ======================================================================
// 4. Coverage: BufferTooSmall Error
// ======================================================================

#[test]
fn test_buffer_too_small_encode() {
    let input = b"Hello world";
    let required = STANDARD.encoded_len(input.len());

    // Buffer exactly 1 byte too small
    let mut small_buf = vec![0u8; required - 1];
    assert_eq!(
        STANDARD.encode_into(input, &mut small_buf),
        Err(Error::BufferTooSmall),
    );

    // Zero-size buffer
    let mut zero_buf: [u8; 0] = [];
    assert_eq!(
        STANDARD.encode_into(input, &mut zero_buf),
        Err(Error::BufferTooSmall),
    );
}

#[test]
fn test_buffer_too_small_decode() {
    let encoded = "SGVsbG8gd29ybGQ="; // "Hello world"
    let required = STANDARD.estimate_decoded_len(encoded.len());

    // Buffer exactly 1 byte too small
    let mut small_buf = vec![0u8; required - 1];
    assert_eq!(
        STANDARD.decode_into(encoded, &mut small_buf),
        Err(Error::BufferTooSmall),
    );
}

// ======================================================================
// 5. Coverage: Error Handling (Invalid Characters & Lengths)
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

#[test]
fn test_reject_url_safe_chars_in_standard() {
    // '-' and '_' are valid in URL_SAFE but invalid in STANDARD
    let mut buf = [0u8; 100];
    assert!(STANDARD.decode_into("-___", &mut buf).is_err());
    assert!(STANDARD.decode_into("A-B_", &mut buf).is_err());
}

#[test]
fn test_reject_standard_chars_in_url_safe() {
    // '+' and '/' are valid in STANDARD but invalid in URL_SAFE
    let mut buf = [0u8; 100];
    assert!(URL_SAFE.decode_into("+///", &mut buf).is_err());
    assert!(URL_SAFE.decode_into("A+B/", &mut buf).is_err());
}

#[test]
fn test_decode_errors_via_allocating_api() {
    // Test that the allocating decode API properly returns errors
    #[cfg(feature = "std")]
    {
        // Invalid character
        assert_eq!(STANDARD.decode("!!!$"), Err(Error::InvalidCharacter));

        // Invalid length (single byte, no-pad config)
        assert_eq!(STANDARD_NO_PAD.decode("A"), Err(Error::InvalidLength));

        // Invalid length (requires padding but missing)
        assert_eq!(STANDARD.decode("AA"), Err(Error::InvalidLength));
    }
}

// ======================================================================
// 6. Coverage: Display & Error Trait Implementations
// ======================================================================

#[test]
fn test_error_display() {
    // Verify Display output for all Error variants
    let msg = format!("{}", Error::InvalidLength);
    assert!(msg.contains("length"), "InvalidLength message: {}", msg);

    let msg = format!("{}", Error::InvalidCharacter);
    assert!(msg.contains("character") || msg.contains("Character"), "InvalidCharacter message: {}", msg);

    let msg = format!("{}", Error::BufferTooSmall);
    assert!(msg.contains("buffer") || msg.contains("Buffer"), "BufferTooSmall message: {}", msg);
}

#[test]
fn test_error_traits() {
    // Verify Debug, Clone, Copy, PartialEq, Eq
    let e1 = Error::InvalidCharacter;
    let e2 = e1; // Copy
    let e3 = e1.clone(); // Clone
    assert_eq!(e1, e2); // PartialEq + Eq
    assert_eq!(e2, e3);
    assert_ne!(Error::InvalidLength, Error::InvalidCharacter);
    assert_ne!(Error::BufferTooSmall, Error::InvalidLength);

    // Debug
    let debug_str = format!("{:?}", Error::InvalidLength);
    assert!(debug_str.contains("InvalidLength"));

    // std::error::Error trait
    #[cfg(feature = "std")]
    {
        fn _assert_error<E: std::error::Error>() {}
        _assert_error::<Error>();
    }
}

// ======================================================================
// 7. Coverage: Known-Value Tests (Deterministic)
// ======================================================================

#[test]
fn test_known_values_standard() {
    let mut buf = [0u8; 64];
    let mut dec = [0u8; 64];

    // RFC 4648 test vectors
    let cases: &[(&[u8], &str)] = &[
        (b"", ""),
        (b"f", "Zg=="),
        (b"fo", "Zm8="),
        (b"foo", "Zm9v"),
        (b"foob", "Zm9vYg=="),
        (b"fooba", "Zm9vYmE="),
        (b"foobar", "Zm9vYmFy"),
    ];

    for (input, expected) in cases {
        if input.is_empty() { continue; }
        let len = STANDARD.encode_into(*input, &mut buf).unwrap();
        assert_eq!(&buf[..len], expected.as_bytes(), "Encode {:?}", input);

        let dec_len = STANDARD.decode_into(expected.as_bytes(), &mut dec).unwrap();
        assert_eq!(&dec[..dec_len], *input, "Decode {:?}", expected);
    }
}

#[test]
fn test_known_values_url_safe() {
    // Input that produces + and / in standard -> - and _ in URL-safe
    let input = &[0xFB, 0xFF, 0xFE];
    let mut buf = [0u8; 8];

    let expected = REF_URL_SAFE_NO_PAD.encode(input);
    let len = URL_SAFE_NO_PAD.encode_into(input, &mut buf).unwrap();
    assert_eq!(&buf[..len], expected.as_bytes());

    // Verify decode roundtrip
    let mut dec = [0u8; 8];
    let dec_len = URL_SAFE_NO_PAD.decode_into(&buf[..len], &mut dec).unwrap();
    assert_eq!(&dec[..dec_len], input);
}

// ======================================================================
// 8. Coverage: All 256 Byte Values (Full Alphabet Coverage)
// ======================================================================

#[test]
fn test_all_byte_values() {
    // Generate a 256-byte input containing every possible byte value
    let input: Vec<u8> = (0..=255).collect();

    for (turbo, reference) in [
        (&STANDARD, &REF_STANDARD),
        (&STANDARD_NO_PAD, &REF_STANDARD_NO_PAD),
        (&URL_SAFE, &REF_URL_SAFE),
        (&URL_SAFE_NO_PAD, &REF_URL_SAFE_NO_PAD),
    ] {
        assert_oracle_match(&input, turbo, reference);
    }
}

// ======================================================================
// 9. Coverage: Boundary-Triggering Sizes (SIMD Thresholds)
// ======================================================================

#[test]
#[cfg(not(miri))]
fn test_simd_threshold_boundaries() {
    // These sizes are chosen to hit exact SIMD loop boundaries
    let boundary_sizes = [
        // SIMD boundaries (alignment, chunks)
        11, 12, 13, 15, 16, 17, 23, 24, 25,
        // AVX2 boundaries (32-byte vectors, 24-byte chunks)
        47, 48, 49, 71, 72, 73, 95, 96, 97,
        // AVX512 boundaries (64-byte vectors, 48-byte chunks)
        47, 48, 49, 95, 96, 97, 143, 144, 145, 191, 192, 193,
        // Multi-loop boundaries
        239, 240, 241, 383, 384, 385, 767, 768, 769,
    ];

    for &size in &boundary_sizes {
        let data = random_bytes(size);
        assert_oracle_match(&data, &STANDARD, &REF_STANDARD);
        assert_oracle_match(&data, &URL_SAFE_NO_PAD, &REF_URL_SAFE_NO_PAD);
    }
}

// ======================================================================
// 10. Coverage: Unstable API (Feature Gated)
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
