#![no_main]
use libfuzzer_sys::fuzz_target;

use base64::engine::general_purpose::{
    STANDARD as REF_STD, STANDARD_NO_PAD as REF_STD_NP,
    URL_SAFE as REF_URL, URL_SAFE_NO_PAD as REF_URL_NP,
};
use base64::Engine as _;

use base64_turbo::{
    Engine, Error,
    STANDARD as TURBO_STD, STANDARD_NO_PAD as TURBO_STD_NP,
    URL_SAFE as TURBO_URL, URL_SAFE_NO_PAD as TURBO_URL_NP,
};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Select one of the four configurations based on the first byte
    let config_idx = (data[0] % 4) as usize;
    let turbo_engines: [&Engine; 4] = [&TURBO_STD, &TURBO_STD_NP, &TURBO_URL, &TURBO_URL_NP];
    let ref_engines: [&base64::engine::GeneralPurpose; 4] = [&REF_STD, &REF_STD_NP, &REF_URL, &REF_URL_NP];

    let engine = turbo_engines[config_idx];
    let ref_engine = ref_engines[config_idx];

    let payload = &data[1..];

    // ----------------------------------------------------------------------
    // 1. Safe allocating APIs (.encode / .decode)
    // ----------------------------------------------------------------------
    let encoded_ref = ref_engine.encode(payload);
    let encoded_turbo = engine.encode(payload);
    assert_eq!(encoded_ref, encoded_turbo);

    // Round-trip valid encoded data (tests allocating .decode on valid input)
    let decoded = engine.decode(&encoded_turbo).unwrap();
    assert_eq!(decoded.as_slice(), payload);

    // ----------------------------------------------------------------------
    // 2. Zero-allocation APIs (.encode_into / .decode_into)
    // ----------------------------------------------------------------------
    let enc_len = engine.encoded_len(payload.len());
    let mut enc_buf = vec![0u8; enc_len.max(1)]; // at least 1 to avoid zero-length issues

    let written_enc = engine.encode_into(payload, &mut enc_buf[..enc_len]).unwrap();
    assert_eq!(written_enc, enc_len);
    assert_eq!(&enc_buf[..written_enc], encoded_turbo.as_bytes());

    // Insufficient buffer for encoding (must return error, no panic/UB)
    if enc_len > 0 {
        let mut small_enc = vec![0u8; enc_len - 1];
        assert!(matches!(
            engine.encode_into(payload, &mut small_enc),
            Err(Error::BufferTooSmall)
        ));
    }

    // Use the encoded length for decode estimate (not payload len)
    let dec_est = engine.estimate_decoded_len(written_enc);
    let mut dec_buf = vec![0u8; dec_est.max(payload.len() + 16)]; // generously sized for robustness

    // Decode valid data
    let written_dec = engine.decode_into(&enc_buf[..written_enc], &mut dec_buf).unwrap();
    assert_eq!(&dec_buf[..written_dec], payload);

    // Decode arbitrary/invalid data (robustness, must not panic/UB)
    // Note: We use decode_into with large buffer to test low-level robustness without allocation wrapper
    let _ = engine.decode_into(payload, &mut dec_buf);

    // Insufficient buffer for decoding arbitrary input (must return error, no panic/UB)
    if !payload.is_empty() {
        let mut small_dec = vec![0u8; 1];
        let res = engine.decode_into(payload, &mut small_dec);
        assert!(matches!(res, Err(Error::BufferTooSmall) | Err(Error::InvalidCharacter) | Err(Error::InvalidLength)));
    }

    // ----------------------------------------------------------------------
    // 3. Raw unsafe paths (unstable feature)
    //    - Only executed when buffer sizes are sufficient
    //    - SIMD paths guarded by runtime feature detection to avoid illegal instructions
    //    - Decode unsafe only on *valid* input (to avoid potential UB as low-level paths assume validity)
    // ----------------------------------------------------------------------

    let valid_encoded = &enc_buf[..written_enc];

    // ----- Scalar (always available) -----
    if enc_len > 0 {
        let mut out_enc = vec![0u8; enc_len];
        unsafe { engine.encode_scalar(payload, out_enc.as_mut_ptr()) };
        assert_eq!(&out_enc[..enc_len], valid_encoded);
    }

    if valid_encoded.len() > 0 {
        let mut out_dec = vec![0u8; dec_est + 3]; // slight overallocation for safety
        let res = unsafe { engine.decode_scalar(valid_encoded, out_dec.as_mut_ptr()) };
        let written = res.unwrap();
        assert_eq!(written, payload.len());
        assert_eq!(&out_dec[..written], payload);
    }

    // ----- SSE4.1 (x86/x86_64 only) -----
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("sse4.1") {
        if enc_len > 0 {
            let mut out_enc = vec![0u8; enc_len];
            unsafe { engine.encode_sse4(payload, out_enc.as_mut_ptr()) };
            assert_eq!(&out_enc[..enc_len], valid_encoded);
        }

        if valid_encoded.len() > 0 {
            let mut out_dec = vec![0u8; dec_est + 3]; // slight overallocation for safety
            let res = unsafe { engine.decode_sse4(valid_encoded, out_dec.as_mut_ptr()) };
            let written = res.unwrap();
            assert_eq!(written, payload.len());
            assert_eq!(&out_dec[..written], payload);
        }
    }

    // ----- AVX2 (x86/x86_64 only) -----
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        if enc_len > 0 {
            let mut out_enc = vec![0u8; enc_len];
            unsafe { engine.encode_avx2(payload, out_enc.as_mut_ptr()) };
            assert_eq!(&out_enc[..enc_len], valid_encoded);
        }

        if valid_encoded.len() > 0 {
            let mut out_dec = vec![0u8; dec_est + 3]; // slight overallocation for safety
            let res = unsafe { engine.decode_avx2(valid_encoded, out_dec.as_mut_ptr()) };
            let written = res.unwrap();
            assert_eq!(written, payload.len());
            assert_eq!(&out_dec[..written], payload);
        }
    }

    // Note: Dispatch logic (AVX512/AVX2/SSE4/scalar selection)
    // TODO: In feature will add explicit support for AVX512 instructions.
});
