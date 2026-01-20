#![no_main]
use libfuzzer_sys::fuzz_target;
use base64::engine::general_purpose::{
    STANDARD as REF_STD, STANDARD_NO_PAD as REF_STD_NP,
    URL_SAFE as REF_URL, URL_SAFE_NO_PAD as REF_URL_NP
};
use base64_turbo::{
    STANDARD as TURBO_STD, STANDARD_NO_PAD as TURBO_STD_NP,
    URL_SAFE as TURBO_URL, URL_SAFE_NO_PAD as TURBO_URL_NP
};

fuzz_target!(|data: &[u8]| {
    // We expect large data here, but handle small just in case
    if data.len() < 1 { return; }

    // Use the first byte to determine the config
    let mode = data[0] % 4;
    let payload = &data[1..];

    match mode {
        0 => compare(payload, &TURBO_STD, &REF_STD),
        1 => compare(payload, &TURBO_STD_NP, &REF_STD_NP),
        2 => compare(payload, &TURBO_URL, &REF_URL),
        3 => compare(payload, &TURBO_URL_NP, &REF_URL_NP),
        _ => unreachable!(),
    }
});

fn compare(
    input: &[u8], 
    turbo: &base64_turbo::Engine, 
    reference: &impl base64::Engine,
) {
    // 1. Encoding check (Oracle)
    let ref_enc = reference.encode(input);
    let tur_enc = turbo.encode(input);
    
    // We only assert equality. Printing 1MB strings on failure is useless and crashes terminals.
    if ref_enc != tur_enc {
        panic!("Encoding mismatch! Input len: {}", input.len());
    }

    // 2. Roundtrip check
    // If encode worked, we trust the output is valid base64, so unwrap is safe-ish.
    let decoded = turbo.decode(&tur_enc).expect("Failed to decode valid output");
    
    if input != decoded {
        panic!("Roundtrip mismatch! Input len: {}", input.len());
    }
    
    // 3. Robustness check (random input)
    // Testing decode on raw random 1MB garbage is a heavy stress test for error handling logic
    let _ = turbo.decode(input);
}
