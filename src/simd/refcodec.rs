//! A deliberately naive Base64 codec, in safe Rust, used as the oracle for the
//! Kani kernel proofs.
//!
//! It exists because `crate::scalar` is no longer a cheap oracle for a model
//! checker: the encoder reaches its answer through a 4096-entry pair table and a
//! 12-bytes-per-iteration unrolled loop, and the decoder through four
//! pre-shifted 256-entry `u32` tables. Over symbolic input those become dozens
//! of very wide symbolic selects, and the two encode harnesses ran past CI's
//! 45-minute per-harness budget.
//!
//! This transcribes RFC 4648 §§3.5, 4 instead: three bytes to four characters
//! and back, one 64-wide scan of the alphabet per character, no tables. That is
//! both cheap enough for CBMC and a stronger oracle — it shares no code with the
//! implementations, so it cannot be fooled by a bug the scalar and vector
//! kernels have in common. It is only ever an oracle: nothing outside the proofs
//! calls it, and it is not compiled outside `cfg(kani)`.
//!
//! Both functions model a **padded** config (`Config::padding == true`), which
//! is what every kernel harness uses.

use crate::Error;
use crate::alphabet::Alphabet;

/// Position of `b` in the alphabet, or `None` if it is not a Base64 character.
/// `Alphabet::new` rejects duplicate characters, so the match is unique.
fn sextet(chars: &[u8; 64], b: u8) -> Option<u32> {
    (0..64u32).find(|&i| chars[i as usize] == b)
}

/// RFC 4648 §4: every three input bytes become four characters, and a partial
/// final group is padded out with `=`. Writes `4 * ceil(input.len() / 3)` bytes.
pub(crate) fn encode(alphabet: &Alphabet, input: &[u8], dst: &mut [u8]) {
    let chars = alphabet.as_bytes();
    let mut out = 0;

    for group in input.chunks(3) {
        let mut acc = 0u32;
        for (i, &b) in group.iter().enumerate() {
            acc |= u32::from(b) << (16 - 8 * i);
        }

        // A whole group emits 4 characters; a partial one emits a character per
        // 6 bits it actually covers, then pads out to 4.
        let emitted = group.len() + 1;
        for i in 0..emitted {
            dst[out + i] = chars[(acc >> (18 - 6 * i)) as usize & 63];
        }
        for i in emitted..4 {
            dst[out + i] = b'=';
        }
        out += 4;
    }
}

/// The inverse, for a padded config: length must be a multiple of 4, `=` may
/// only occupy the last one or two positions of the final group, and the bits a
/// dropped character would have carried must be zero (RFC 4648 §3.5).
///
/// Only the accept/reject split and the decoded bytes are meaningful. Which
/// [`Error`] a rejection carries is not: the kernels decide length and character
/// validity in a different order, so the harnesses compare `is_err()`.
pub(crate) fn decode(alphabet: &Alphabet, input: &[u8], dst: &mut [u8]) -> Result<usize, Error> {
    let chars = alphabet.as_bytes();

    if input.is_empty() {
        return Ok(0);
    }
    if !input.len().is_multiple_of(4) {
        return Err(Error::InvalidLength);
    }

    let groups = input.len() / 4;
    let mut out = 0;

    for (g, group) in input.chunks(4).enumerate() {
        // `=` is not in any alphabet, so a `=` anywhere else falls out below as
        // an invalid character. Only the final group's tail is padding.
        let pad = match (group[2], group[3]) {
            (b'=', b'=') => 2,
            (_, b'=') => 1,
            _ => 0,
        };
        if pad > 0 && g + 1 != groups {
            return Err(Error::InvalidLength);
        }

        let mut acc = 0u32;
        for (i, &c) in group.iter().take(4 - pad).enumerate() {
            match sextet(chars, c) {
                Some(s) => acc |= s << (18 - 6 * i),
                None => return Err(Error::InvalidCharacter),
            }
        }

        // The group carries 24 bits and emits `3 - pad` bytes off the top; every
        // bit below them belongs to a character that isn't there and must be 0.
        if acc & ((1 << (8 * pad)) - 1) != 0 {
            return Err(Error::InvalidCharacter);
        }

        // `acc` holds the group's 24 bits in its low three bytes, so the output
        // is just those bytes in order.
        let bytes = acc.to_be_bytes();
        dst[out..out + 3 - pad].copy_from_slice(&bytes[1..4 - pad]);
        out += 3 - pad;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;
    use crate::alphabet::builtin;
    use rand::RngExt;

    /// The oracle and the scalar kernel agree, on both directions and on which
    /// inputs are rejected. Kani checks the vector kernels against this file, so
    /// a bug here would silently weaken every kernel proof — and nothing else
    /// exercises it, since `cfg(kani)` code is invisible to a normal build.
    #[test]
    fn agrees_with_scalar_kernel() {
        let mut rng = rand::rng();

        for url_safe in [false, true] {
            let alphabet = builtin(url_safe);
            let config = Config {
                alphabet,
                padding: true,
            };

            for len in 0..512usize {
                let input: Vec<u8> = (0..len).map(|_| rng.random()).collect();

                let mut want = vec![0u8; len.div_ceil(3) * 4];
                let mut got = vec![0u8; want.len()];
                crate::scalar::encode_slice(&config, &input, &mut want);
                encode(alphabet, &input, &mut got);
                assert_eq!(got, want, "encode disagrees at len {len}");

                // Decode both the encoder's own output and near-miss inputs:
                // random bytes over the alphabet, with the odd `=` and a stray
                // byte mixed in, so rejection is compared as well as value.
                let mut chars = got.clone();
                for _ in 0..8 {
                    if !chars.is_empty() {
                        let i = rng.random_range(0..chars.len());
                        chars[i] = match rng.random_range(0..3) {
                            0 => b'=',
                            1 => rng.random(),
                            _ => alphabet.as_bytes()[rng.random_range(0..64)],
                        };
                    }

                    let cap = chars.len() / 4 * 3;
                    let mut want = vec![0u8; cap];
                    let mut got = vec![0u8; cap];
                    let want_res = crate::scalar::decode_slice(&config, &chars, &mut want);
                    let got_res = decode(alphabet, &chars, &mut got);

                    match want_res {
                        Ok(n) => {
                            assert_eq!(got_res, Ok(n), "oracle rejected {chars:?}");
                            assert_eq!(got[..n], want[..n], "decode disagrees on {chars:?}");
                        }
                        Err(_) => assert!(got_res.is_err(), "oracle accepted {chars:?}"),
                    }
                }
            }
        }
    }
}
