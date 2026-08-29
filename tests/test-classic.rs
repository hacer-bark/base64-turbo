//! Integration tests verifying `base64-turbo`'s output against the reference `base64` crate.
#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use base64_turbo::{
    Engine, Error, STANDARD, STANDARD_NO_PAD, STANDARD_PAD_INDIFFERENT, URL_SAFE, URL_SAFE_NO_PAD,
};
#[cfg(feature = "std")]
use base64_turbo::{URL_SAFE_PAD_INDIFFERENT, decode, encode};

// Reference crate for oracle verification.
use base64::{
    Engine as _,
    engine::general_purpose::{
        STANDARD as REF_STANDARD, STANDARD_NO_PAD as REF_STANDARD_NO_PAD, URL_SAFE as REF_URL_SAFE,
        URL_SAFE_NO_PAD as REF_URL_SAFE_NO_PAD,
    },
};
use rand::RngExt;

// ======================================================================
// Helpers
// ======================================================================

fn random_bytes(len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    rand::rng().fill(&mut bytes);
    bytes
}

const fn engines() -> [(Engine, &'static base64::engine::GeneralPurpose); 4] {
    [
        (STANDARD, &REF_STANDARD),
        (STANDARD_NO_PAD, &REF_STANDARD_NO_PAD),
        (URL_SAFE, &REF_URL_SAFE),
        (URL_SAFE_NO_PAD, &REF_URL_SAFE_NO_PAD),
    ]
}

fn assert_oracle_matches(
    input: &[u8],
    engine_pairs: &[(Engine, &'static base64::engine::GeneralPurpose)],
) {
    for &(turbo, reference) in engine_pairs {
        assert_oracle_match(input, turbo, reference);
    }
}

fn assert_all_oracle_matches(input: &[u8]) {
    assert_oracle_matches(input, &engines());
}

fn assert_rejects(engine: Engine, inputs: &[&str]) {
    let mut buffer = [0u8; 100];
    for &input in inputs {
        assert!(
            engine.decode_slice(input, &mut buffer).is_err(),
            "accepted {input:?}",
        );
    }
}

const fn assert_clone<T: Clone>() {}

fn expected_encoded_len(input_len: usize, padding: bool) -> usize {
    let complete_groups = input_len / 3;
    match (input_len % 3, padding) {
        (0, _) => complete_groups * 4,
        (_, true) => (complete_groups + 1) * 4,
        (1, false) => complete_groups * 4 + 2,
        (2, false) => complete_groups * 4 + 3,
        _ => unreachable!(),
    }
}

fn assert_encoded_lengths(engine: Engine, padding: bool) {
    for input_len in 0..=7 {
        assert_eq!(
            engine.encoded_len(input_len).unwrap(),
            expected_encoded_len(input_len, padding),
            "encoded length for {input_len} bytes",
        );
    }
}

#[cfg(not(miri))]
fn large_test_lengths() -> impl Iterator<Item = usize> {
    // Varied sizes that execute every large-input dispatch loop.
    [1024, 4095, 8192, 16383, 32768, 65535].into_iter()
}

fn config_test_lengths() -> impl Iterator<Item = usize> {
    [1, 2, 3, 11, 12, 13, 47, 48, 49, 255, 511].into_iter()
}

#[cfg(not(miri))]
fn boundary_test_lengths() -> impl Iterator<Item = usize> {
    [
        11, 12, 13, 15, 16, 17, 23, 24, 25, 47, 48, 49, 59, 60, 61, 71, 72, 73, 95, 96, 97, 143,
        144, 145, 191, 192, 193, 239, 240, 241, 383, 384, 385, 767, 768, 769,
    ]
    .into_iter()
}

fn all_byte_values() -> Vec<u8> {
    (0..=u8::MAX).collect()
}

/// Verifies that `base64-turbo` output exactly matches the `base64` crate, across
/// both the zero-allocation slice API and (with `std`) the allocating API.
#[track_caller]
fn assert_oracle_match(
    input: &[u8],
    turbo_engine: Engine,
    ref_engine: &base64::engine::GeneralPurpose,
) {
    let expected_encoded = ref_engine.encode(input);

    let mut enc_buf = vec![0u8; turbo_engine.encoded_len(input.len()).unwrap()];
    let enc_len = turbo_engine
        .encode_slice(input, &mut enc_buf)
        .expect("encode_into failed");
    assert_eq!(
        &enc_buf[..enc_len],
        expected_encoded.as_bytes(),
        "slice encode mismatch"
    );

    #[cfg(feature = "std")]
    {
        let alloc_str = turbo_engine.encode(input);
        assert_eq!(alloc_str, expected_encoded, "allocating encode mismatch");
    }

    // Allocated based on the estimate, but the write length is checked exactly.
    let mut dec_buf = vec![0u8; turbo_engine.decoded_len_estimate(expected_encoded.len())];
    let dec_len = turbo_engine
        .decode_slice(expected_encoded.as_bytes(), &mut dec_buf)
        .expect("decode_into failed");
    assert_eq!(&dec_buf[..dec_len], input, "slice decode mismatch");

    #[cfg(feature = "std")]
    {
        let alloc_vec = turbo_engine
            .decode(&expected_encoded)
            .expect("decode failed");
        assert_eq!(alloc_vec, input, "allocating decode mismatch");
    }
}

// ======================================================================
// 1. Coverage: Basic Logic & Oracle Matching
// ======================================================================

#[test]
fn test_oracle_standard_exhaustive_small() {
    // Covers 0..92 to hit all SIMD mask boundaries, alignment issues, and scalar fallbacks.
    // This implicitly covers `encoded_len` and `decoded_len_estimate` correctness via helpers.
    for i in 0..=92 {
        let data = random_bytes(i);
        assert_oracle_match(&data, STANDARD, &REF_STANDARD);
    }
}

#[test]
#[cfg(not(miri))]
fn test_oracle_large_inputs() {
    // Random sizes up to 64KB to trigger AVX2/AVX512-VBMI loops multiple times.
    for len in large_test_lengths() {
        let data = random_bytes(len);
        assert_oracle_match(&data, STANDARD, &REF_STANDARD);
    }
}

#[test]
fn test_oracle_configs() {
    // Verify URL_SAFE and NO_PAD variants logic
    for len in config_test_lengths() {
        let data = random_bytes(len);
        assert_oracle_matches(&data, &engines()[1..]);
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
    let enc_len = STANDARD.encode_slice(empty, &mut enc_buf).unwrap();
    assert_eq!(enc_len, 0);

    // Decode empty -> 0 bytes written
    let dec_len = STANDARD.decode_slice(empty, &mut dec_buf).unwrap();
    assert_eq!(dec_len, 0);

    for engine in &[STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD] {
        assert_eq!(engine.encode_slice(empty, &mut enc_buf).unwrap(), 0);
        assert_eq!(engine.decode_slice(empty, &mut dec_buf).unwrap(), 0);
    }

    #[cfg(feature = "std")]
    {
        assert_eq!(STANDARD.encode(b""), "");
        assert_eq!(STANDARD.decode("").unwrap(), Vec::<u8>::new());
    }
}

// ======================================================================
// 3. Coverage: encoded_len & decoded_len_estimate correctness
// ======================================================================

#[test]
fn test_encoded_len_correctness() {
    // Padded encoding: ceil(n/3) * 4
    assert_encoded_lengths(STANDARD, true);

    // No-pad encoding: ceil(n*4/3)
    assert_encoded_lengths(STANDARD_NO_PAD, false);

    // URL_SAFE uses same math as STANDARD (just different alphabet)
    assert_eq!(
        STANDARD.encoded_len(10).unwrap(),
        URL_SAFE.encoded_len(10).unwrap()
    );
    assert_eq!(
        STANDARD_NO_PAD.encoded_len(10).unwrap(),
        URL_SAFE_NO_PAD.encoded_len(10).unwrap()
    );

    assert_eq!(STANDARD.encoded_len(usize::MAX), None);
    assert_eq!(STANDARD_NO_PAD.encoded_len(usize::MAX), None);
}

#[test]
fn test_decoded_len_estimate() {
    assert_eq!(STANDARD.decoded_len_estimate(0), 0);
    assert_eq!(STANDARD.decoded_len_estimate(4), 3);
    assert_eq!(STANDARD.decoded_len_estimate(8), 6);
    assert_eq!(STANDARD.decoded_len_estimate(12), 9);

    // Sanity: estimate should always be >= actual
    for n in 0..=50 {
        let data = random_bytes(n);
        let encoded = REF_STANDARD.encode(&data);
        assert!(
            STANDARD.decoded_len_estimate(encoded.len()) >= n,
            "estimate too small for n={n}"
        );
    }
}

#[test]
fn test_forgiving_padding() {
    #[cfg(feature = "std")]
    {
        assert_eq!(STANDARD_PAD_INDIFFERENT.decode("Zg").unwrap(), b"f");
        assert_eq!(STANDARD_PAD_INDIFFERENT.decode("Zg==").unwrap(), b"f");
        assert_eq!(
            URL_SAFE_PAD_INDIFFERENT.decode("-_8").unwrap(),
            [0xFB, 0xFF]
        );
        assert_eq!(
            URL_SAFE_PAD_INDIFFERENT.decode("-_8=").unwrap(),
            [0xFB, 0xFF]
        );
    }

    let mut output = [0u8; 3];
    assert_eq!(
        STANDARD_PAD_INDIFFERENT.decode_slice("Zm8", &mut output),
        Ok(2)
    );
    assert_eq!(&output[..2], b"fo");
}

#[cfg(feature = "std")]
#[test]
fn test_convenience_and_append_apis() {
    assert_eq!(encode(b"hello"), "aGVsbG8=");
    assert_eq!(decode("aGVsbG8=").unwrap(), b"hello");

    let mut text = String::from("prefix:");
    STANDARD.encode_string(b"hi", &mut text);
    assert_eq!(text, "prefix:aGk=");

    let mut bytes = b"prefix:".to_vec();
    STANDARD.decode_vec("aGk=", &mut bytes).unwrap();
    assert_eq!(bytes, b"prefix:hi");
}

// ======================================================================
// 4. Coverage: BufferTooSmall Error
// ======================================================================

#[test]
fn test_buffer_too_small_encode() {
    let input = b"Hello world";
    let required = STANDARD.encoded_len(input.len()).unwrap();

    // Buffer exactly 1 byte too small
    let mut small_buf = vec![0u8; required - 1];
    assert_eq!(
        STANDARD.encode_slice(input, &mut small_buf),
        Err(Error::BufferTooSmall),
    );

    // Zero-size buffer
    let mut zero_buf: [u8; 0] = [];
    assert_eq!(
        STANDARD.encode_slice(input, &mut zero_buf),
        Err(Error::BufferTooSmall),
    );
}

#[test]
fn test_buffer_too_small_decode() {
    let encoded = "SGVsbG8gd29ybGQ="; // "Hello world"
    let required = STANDARD.decoded_len_estimate(encoded.len());

    // Buffer exactly 1 byte too small
    let mut small_buf = vec![0u8; required - 1];
    assert_eq!(
        STANDARD.decode_slice(encoded, &mut small_buf),
        Err(Error::BufferTooSmall),
    );
}

// ======================================================================
// 5. Coverage: Error Handling (Invalid Characters & Lengths)
// ======================================================================

#[test]
fn test_reject_invalid_chars() {
    let bad_inputs = ["Abc!", "Ab c", "Abc\0", "Abc-", "Abc_"];
    for bad in bad_inputs {
        let mut buf = [0u8; 100];
        assert_eq!(
            STANDARD.decode_slice(bad, &mut buf),
            Err(Error::InvalidCharacter),
            "failed to reject: {bad:?}",
        );
    }
}

#[test]
fn test_reject_invalid_length_padding() {
    let inputs = ["A", "AA", "AAA", "AAAA=", "A===", "===="];
    let mut buf = [0u8; 100];
    for inp in inputs {
        let res = STANDARD.decode_slice(inp, &mut buf);
        // Either InvalidLength or InvalidCharacter is acceptable here.
        assert!(res.is_err(), "should fail on invalid padding/length: {inp}");
    }
}

#[test]
fn test_reject_non_terminal_or_non_canonical_padding() {
    assert_rejects(STANDARD, &["TQ==AAAA", "TQ==!", "TR==", "TWF="]);
    assert_rejects(STANDARD_NO_PAD, &["TQ==", "TR", "TWF"]);
}

#[test]
fn test_reject_url_safe_chars_in_standard() {
    // '-' and '_' are valid in URL_SAFE but invalid in STANDARD
    assert_rejects(STANDARD, &["-___", "A-B_"]);
}

#[test]
fn test_reject_standard_chars_in_url_safe() {
    // '+' and '/' are valid in STANDARD but invalid in URL_SAFE
    assert_rejects(URL_SAFE, &["+///", "A+B/"]);
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
    assert!(msg.contains("length"), "InvalidLength message: {msg}");

    let msg = format!("{}", Error::InvalidCharacter);
    assert!(
        msg.contains("character") || msg.contains("Character"),
        "InvalidCharacter message: {msg}"
    );

    let msg = format!("{}", Error::BufferTooSmall);
    assert!(
        msg.contains("buffer") || msg.contains("Buffer"),
        "BufferTooSmall message: {msg}"
    );
}

#[test]
fn test_error_traits() {
    // Verify Debug, Clone, Copy, PartialEq, Eq
    let e1 = Error::InvalidCharacter;
    let e2 = e1; // Copy
    let e3 = e1;
    assert_eq!(e1, e2); // PartialEq + Eq
    assert_eq!(e2, e3);
    assert_ne!(Error::InvalidLength, Error::InvalidCharacter);
    assert_ne!(Error::BufferTooSmall, Error::InvalidLength);

    assert_clone::<Error>();

    // Debug
    let debug_str = format!("{:?}", Error::InvalidLength);
    assert!(debug_str.contains("InvalidLength"));

    // std::error::Error trait
    #[cfg(feature = "std")]
    {
        fn assert_error<E: std::error::Error>() {}
        assert_error::<Error>();
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
        let len = STANDARD.encode_slice(*input, &mut buf).unwrap();
        assert_eq!(&buf[..len], expected.as_bytes(), "encode {input:?}");

        let dec_len = STANDARD
            .decode_slice(expected.as_bytes(), &mut dec)
            .unwrap();
        assert_eq!(&dec[..dec_len], *input, "decode {expected:?}");
    }
}

#[test]
fn test_known_values_url_safe() {
    // Input that produces + and / in standard -> - and _ in URL-safe
    let input = &[0xFB, 0xFF, 0xFE];
    let mut buf = [0u8; 8];

    let expected = REF_URL_SAFE_NO_PAD.encode(input);
    let len = URL_SAFE_NO_PAD.encode_slice(input, &mut buf).unwrap();
    assert_eq!(&buf[..len], expected.as_bytes());

    // Verify decode roundtrip
    let mut dec = [0u8; 8];
    let dec_len = URL_SAFE_NO_PAD.decode_slice(&buf[..len], &mut dec).unwrap();
    assert_eq!(&dec[..dec_len], input);
}

// ======================================================================
// 8. Coverage: All 256 Byte Values (Full Alphabet Coverage)
// ======================================================================

#[test]
fn test_all_byte_values() {
    // Generate a 256-byte input containing every possible byte value
    let input = all_byte_values();
    assert_all_oracle_matches(&input);
}

// ======================================================================
// 9. Coverage: Boundary-Triggering Sizes (SIMD Thresholds)
// ======================================================================

#[test]
#[cfg(not(miri))]
fn test_simd_threshold_boundaries() {
    // These sizes are chosen to hit exact SIMD loop boundaries
    for size in boundary_test_lengths() {
        let data = random_bytes(size);
        assert_oracle_match(&data, STANDARD, &REF_STANDARD);
        assert_oracle_match(&data, URL_SAFE_NO_PAD, &REF_URL_SAFE_NO_PAD);
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

    // --- Scalar (always available, and a safe API) ---
    {
        let mut dst = vec![0u8; STANDARD.encoded_len(input.len()).unwrap()];
        STANDARD.encode_scalar(&input, &mut dst);
        assert_eq!(&dst, expected.as_bytes(), "scalar: encode mismatch");

        let mut dec = vec![0u8; STANDARD.decoded_len_estimate(dst.len())];
        let len = STANDARD.decode_scalar(&dst, &mut dec).unwrap();
        assert_eq!(&dec[..len], &input, "scalar: decode mismatch");
    }

    // --- AVX2 ---
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe {
            let mut dst = vec![0u8; STANDARD.encoded_len(input.len()).unwrap()];
            STANDARD.encode_avx2(&input, &mut dst);
            assert_eq!(&dst, expected.as_bytes(), "avx2: encode mismatch");

            let mut dec = vec![0u8; STANDARD.decoded_len_estimate(dst.len())];
            let len = STANDARD.decode_avx2(&dst, &mut dec).unwrap();
            assert_eq!(&dec[..len], &input, "avx2: decode mismatch");
        }
    } else {
        println!("skipping AVX2 unstable test (hardware unsupported)");
    }

    // --- AVX-512-VBMI ---
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "avx512-vbmi"
    ))]
    if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
    {
        unsafe {
            let mut dst = vec![0u8; STANDARD.encoded_len(input.len()).unwrap()];
            STANDARD.encode_avx512_vbmi(&input, &mut dst);
            assert_eq!(&dst, expected.as_bytes(), "avx512-vbmi: encode mismatch");

            let mut dec = vec![0u8; STANDARD.decoded_len_estimate(dst.len())];
            let len = STANDARD.decode_avx512_vbmi(&dst, &mut dec).unwrap();
            assert_eq!(&dec[..len], &input, "avx512-vbmi: decode mismatch");
        }
    } else {
        println!("skipping AVX512-VBMI unstable test (hardware unsupported)");
    }

    // --- NEON ---
    #[cfg(target_arch = "aarch64")]
    #[cfg(feature = "neon")]
    unsafe {
        let mut dst = vec![0u8; STANDARD.encoded_len(input.len()).unwrap()];
        STANDARD.encode_neon(&input, &mut dst);
        assert_eq!(&dst, expected.as_bytes(), "neon: encode mismatch");

        let mut dec = vec![0u8; STANDARD.decoded_len_estimate(dst.len())];
        let len = STANDARD.decode_neon(&dst, &mut dec).unwrap();
        assert_eq!(&dec[..len], &input, "neon: decode mismatch");
    }
}
