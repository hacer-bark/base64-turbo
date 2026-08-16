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

/// True when the host implements every subset the VBMI kernel issues. All three
/// are required: `vpermb`/`vpermi2b`/`vpmultishiftqb` are VBMI, the masked
/// `vmovdqu8` tiers are BW, and the 512-bit registers are F.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn has_avx512_vbmi() -> bool {
    std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vbmi")
}

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
    // 3. Raw unsafe kernels (unstable feature)
    //
    //    Every buffer below is sized to *exactly* the capacity the kernel's
    //    safety contract asks for -- `encoded_len` to encode,
    //    `estimate_decoded_len` to decode -- and not a byte more. Slack here
    //    would hide the one bug class this section exists to find: a kernel
    //    whose overlapping or masked stores reach past the bound it documents.
    //    With ASan on, an overrun of these allocations is a hard failure.
    //
    //    Both valid and arbitrary input go through the decoders. The kernels
    //    fold validation into an accumulator they only test after their loops,
    //    so they may write garbage for invalid input -- but that garbage must
    //    still land inside `estimate_decoded_len`, and the call must report
    //    `Err` rather than panic.
    // ----------------------------------------------------------------------

    let valid_encoded = &enc_buf[..written_enc];
    let arbitrary_dec_est = engine.estimate_decoded_len(payload.len());

    // Runs one kernel pair over: encode(payload), decode(valid), decode(arbitrary).
    macro_rules! exercise_kernel {
        ($name:literal, $encode:ident, $decode:ident) => {{
            if enc_len > 0 {
                let mut out_enc = vec![0u8; enc_len];
                unsafe { engine.$encode(payload, &mut out_enc) };
                assert_eq!(&out_enc[..], valid_encoded, concat!($name, ": encode mismatch"));
            }

            if !valid_encoded.is_empty() {
                let mut out_dec = vec![0u8; dec_est];
                let written = unsafe { engine.$decode(valid_encoded, &mut out_dec) }
                    .expect(concat!($name, ": valid input failed to decode"));
                assert_eq!(written, payload.len(), concat!($name, ": decoded length mismatch"));
                assert_eq!(&out_dec[..written], payload, concat!($name, ": decode mismatch"));
            }

            if !payload.is_empty() {
                let mut out_dec = vec![0u8; arbitrary_dec_est];
                // Arbitrary bytes: any result is acceptable, a panic or an
                // out-of-bounds write is not.
                let _ = unsafe { engine.$decode(payload, &mut out_dec) };
            }
        }};
    }

    // ----- Scalar (always available) -----
    // Safe, not unsafe -- the scalar kernel forbids `unsafe` -- but exercised
    // through the same shape so the three kernels stay comparable.
    if enc_len > 0 {
        let mut out_enc = vec![0u8; enc_len];
        engine.encode_scalar(payload, &mut out_enc);
        assert_eq!(&out_enc[..], valid_encoded, "scalar: encode mismatch");
    }

    if !valid_encoded.is_empty() {
        let mut out_dec = vec![0u8; dec_est];
        let written = engine.decode_scalar(valid_encoded, &mut out_dec)
            .expect("scalar: valid input failed to decode");
        assert_eq!(written, payload.len(), "scalar: decoded length mismatch");
        assert_eq!(&out_dec[..written], payload, "scalar: decode mismatch");
    }

    if !payload.is_empty() {
        let mut out_dec = vec![0u8; arbitrary_dec_est];
        let _ = engine.decode_scalar(payload, &mut out_dec);
    }

    // ----- AVX2 (x86/x86_64 only) -----
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        exercise_kernel!("avx2", encode_avx2, decode_avx2);
    }

    // ----- AVX-512-VBMI (x86/x86_64 only) -----
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if has_avx512_vbmi() {
        exercise_kernel!("avx512-vbmi", encode_avx512_vbmi, decode_avx512_vbmi);
    }

    // ----- NEON (aarch64 only) -----
    #[cfg(target_arch = "aarch64")]
    {
        exercise_kernel!("neon", encode_neon, decode_neon);
    }
});
