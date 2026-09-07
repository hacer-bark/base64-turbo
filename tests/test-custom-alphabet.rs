//! Custom-alphabet integration tests, checked against the `base64` crate's
//! `GeneralPurpose` engine built over the same alphabet.
#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use base64::Engine as _;
use base64_turbo::{Alphabet, Engine, Error, STANDARD};
use rand::RngExt;

/// bcrypt/crypt(3): `.` and `/` at indices 0 and 1, digits after the letters.
/// Shares no character *position* with RFC 4648, so a kernel that fell back to
/// the built-in tables would fail on the first byte.
const BCRYPT_CHARS: &[u8; 64] = b"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

static BCRYPT: Alphabet = match Alphabet::new(BCRYPT_CHARS) {
    Some(a) => a,
    None => unreachable!(),
};

static BCRYPT_PAD: Engine = Engine::custom(&BCRYPT, true);
static BCRYPT_NO_PAD: Engine = Engine::custom(&BCRYPT, false);

fn oracle(padding: bool) -> base64::engine::GeneralPurpose {
    let alphabet =
        base64::alphabet::Alphabet::new(core::str::from_utf8(BCRYPT_CHARS).unwrap()).unwrap();
    let config = base64::engine::GeneralPurposeConfig::new()
        .with_encode_padding(padding)
        .with_decode_padding_mode(if padding {
            base64::engine::DecodePaddingMode::RequireCanonical
        } else {
            base64::engine::DecodePaddingMode::RequireNone
        });
    base64::engine::GeneralPurpose::new(&alphabet, config)
}

/// The slice APIs, so this file runs unchanged in a `no_std` build where the
/// allocating `encode`/`decode` are compiled out.
fn encode(engine: &Engine, input: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; engine.encoded_len(input.len())];
    let written = engine.encode_slice(input, &mut out).unwrap();
    out.truncate(written);
    out
}

fn decode(engine: &Engine, input: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out = vec![0u8; engine.decoded_len_estimate(input.len())];
    let written = engine.decode_slice(input, &mut out)?;
    out.truncate(written);
    Ok(out)
}

fn random_bytes(len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    rand::rng().fill(&mut bytes);
    bytes
}

/// Lengths spanning every dispatch tier: the scalar-only shorts and the
/// AVX-512 masked/single/quad boundaries, plus — off Miri, where they would
/// take hours — a size past the non-temporal store threshold.
fn lengths() -> Vec<usize> {
    let mut lengths = vec![0, 1, 2, 3, 4, 5, 47, 48, 49, 63, 64, 65, 191, 192, 193, 259];
    if !cfg!(miri) {
        lengths.extend([1001, 600_000]);
    }
    lengths
}

#[test]
fn custom_alphabet_matches_oracle() {
    for &(engine, padding) in &[(BCRYPT_PAD, true), (BCRYPT_NO_PAD, false)] {
        let reference = oracle(padding);
        for len in lengths() {
            let input = random_bytes(len);
            let encoded = encode(&engine, &input);
            assert_eq!(
                encoded,
                reference.encode(&input).into_bytes(),
                "encode at len {len}"
            );
            assert_eq!(
                decode(&engine, &encoded).unwrap(),
                input,
                "decode at len {len}"
            );
        }
    }
}

/// The standard characters routed through the custom API must be
/// indistinguishable from the built-in engine — that equivalence is what lets
/// `Engine::custom` keep the AVX2 and NEON kernels for them.
#[test]
fn standard_chars_via_custom_api_match_builtin() {
    static STD: Alphabet =
        match Alphabet::new(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/") {
            Some(a) => a,
            None => unreachable!(),
        };
    static ENGINE: Engine = Engine::custom(&STD, true);

    for len in lengths() {
        let input = random_bytes(len);
        assert_eq!(
            encode(&ENGINE, &input),
            encode(&STANDARD, &input),
            "len {len}"
        );
    }
}

#[test]
fn foreign_characters_are_rejected() {
    // `+` and `/` are absent from the bcrypt alphabet, and `.` from the
    // standard one, so each engine must reject the other's output whenever the
    // distinguishing characters appear.
    let input = b"\xfb\xff\xbe\x00\x3e\xf0";
    assert!(decode(&BCRYPT_PAD, &encode(&STANDARD, input)).is_err());
    assert!(decode(&STANDARD, &encode(&BCRYPT_PAD, input)).is_err());
}

#[test]
fn invalid_alphabets_are_rejected() {
    let cases: [(&[u8; 64], &str); 5] = [
        (
            b"AACDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
            "duplicate character",
        ),
        (
            b"=BCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
            "the padding character",
        ),
        (
            b"\xffBCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
            "non-ASCII",
        ),
        (
            b" BCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
            "whitespace",
        ),
        (
            b"\x00BCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
            "a control byte",
        ),
    ];
    for (chars, why) in cases {
        assert!(Alphabet::new(chars).is_none(), "accepted {why}");
    }
}
