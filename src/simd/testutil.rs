//! Shared helpers for the per-ISA SIMD coverage tests. Each backend's test
//! module passes in its own `encode`/`decode` function pointer; the input
//! generation and oracle comparison are identical across backends and live
//! here so the AVX2, AVX-512 and NEON suites stay the same logic, differing
//! only in the function under test and the tier lengths they probe.
//!
//! Which helpers are live depends on the target arch and whether Miri is
//! running, so unused ones are expected under some build configs.
#![allow(dead_code)]
// `&Config` mirrors the production `encode`/`decode` signatures the fn
// pointers point at, so keep it by-reference here too.
#![allow(clippy::trivially_copy_pass_by_ref)]

use crate::{Config, Error};
use base64::Engine;

type EncodeFn = unsafe fn(&Config, &[u8], &mut [u8]);
type DecodeFn = unsafe fn(&Config, &[u8], &mut [u8]) -> Result<usize, Error>;

/// Seeded xorshift, not `rand`, so a failure reproduces exactly.
pub(crate) fn bytes(len: usize) -> Vec<u8> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            u8::try_from(state >> 56).expect("shifted down to 8 bits")
        })
        .collect()
}

/// Encode `len` bytes and assert the SIMD output matches the oracle. The
/// buffer is the exact encoded length, matching the real caller, so Miri
/// catches any store overrun.
pub(crate) fn check_encode(config: &Config, oracle: &impl Engine, encode: EncodeFn, len: usize) {
    let input = bytes(len);
    let expected = oracle.encode(&input);
    let mut dst = vec![0u8; expected.len()];
    unsafe { encode(config, &input, &mut dst) };
    assert_eq!(
        core::str::from_utf8(&dst).unwrap(),
        expected,
        "encode mismatch at len {len}"
    );
}

/// Round-trip `len` bytes: oracle-encode, then SIMD-decode back. The decode
/// buffer carries a margin because every non-VBMI packing store overhangs its
/// block by 4 bytes; see [`check_decode_exact`] for the tight-buffer variant.
pub(crate) fn check_decode(config: &Config, oracle: &impl Engine, decode: DecodeFn, len: usize) {
    let input = bytes(len);
    let encoded = oracle.encode(&input);
    let mut dst = vec![0u8; len + 64];
    let n = unsafe {
        decode(config, encoded.as_bytes(), &mut dst).expect("valid input failed to decode")
    };
    assert_eq!(&dst[..n], &input[..], "decode mismatch at len {len}");
}

/// Like [`check_decode`] but the buffer is sized to the exact decoded length,
/// so Miri (and hardware) catch a masked-store overrun by even one byte. Used
/// by the AVX-512-VBMI masked-store path.
pub(crate) fn check_decode_exact(
    config: &Config,
    oracle: &impl Engine,
    decode: DecodeFn,
    len: usize,
) {
    let input = bytes(len);
    let encoded = oracle.encode(&input);
    let mut dst = vec![0u8; len];
    let n = unsafe {
        decode(config, encoded.as_bytes(), &mut dst).expect("valid input failed to decode")
    };
    assert_eq!(n, len, "exact-buffer decode len {len}");
    assert_eq!(
        &dst[..n],
        &input[..],
        "exact-buffer decode mismatch at len {len}"
    );
}
