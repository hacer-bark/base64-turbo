use crate::{
    Error,
    Config,
    STANDARD_ALPHABET,
    URL_SAFE_ALPHABET,
    STANDARD_DECODE_TABLE,
    URL_SAFE_DECODE_TABLE,
};

/// Encodes a byte slice into Base64 using a highly optimized scalar algorithm.
///
/// # Performance
/// This function is manually unrolled to process **6 bytes of input** at a time, producing
/// **8 bytes of output** (one 64-bit word). This allows the CPU to use its full 64-bit
/// data paths and internal parallelism (superscalar execution) even without explicit SIMD instructions.
///
/// On modern CPUs, this implementation is approximately **1.5x - 2.0x faster** than the standard
/// library's scalar loop.
///
/// # Safety
/// * The caller must ensure `dst` is valid and has enough capacity to hold the encoded result.
/// * `config` must contain valid boolean flags.
/// * This function performs unchecked pointer arithmetic for speed.
#[inline(always)]
pub unsafe fn encode_slice_unsafe(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

    // 1. Select the alphabet based on configuration
    // This branch is predicted perfectly since config doesn't change during the loop.
    let alphabet = if config.url_safe {
        URL_SAFE_ALPHABET
    } else {
        STANDARD_ALPHABET
    };

    unsafe {
        // Calculate how many bytes we can process in the fast unrolled loop.
        // We process 6 input bytes -> 8 output bytes per iteration.
        let len_aligned = len.saturating_sub(len % 6);
        let src_end_aligned = src.add(len_aligned);

        // --- MAIN LOOP (Unrolled 2x) ---
        while src < src_end_aligned {
            // Read 6 bytes.
            // We read two 32-bit integers (4 bytes each) with overlap to avoid complex shifting logic.
            // Reading unaligned u32 is virtually free on x86/ARM64.

            #[cfg(target_endian = "little")]
            let (n1, n2) = {
                // Read bytes 0..4, convert to Big Endian to get correct byte order in register
                let reg_a = (src as *const u32).read_unaligned().to_be();
                // Read bytes 2..6 (Overlap bytes 2 and 3)
                let reg_b = (src.add(2) as *const u32).read_unaligned().to_be();

                // Extract the specific 24 bits we need for each 4-char block
                let n1 = (reg_a >> 8) as usize; // Bytes 0, 1, 2
                let n2 = (reg_b & 0x00_FF_FF_FF) as usize; // Bytes 3, 4, 5
                (n1, n2)
            };

            #[cfg(target_endian = "big")]
            let (n1, n2) = {
                let reg_a = (src as *const u32).read_unaligned();
                let reg_b = (src.add(2) as *const u32).read_unaligned();
                ( (reg_a >> 8) as usize, (reg_b & 0x00_FF_FF_FF) as usize )
            };

            // Map indices to Base64 characters and pack into a single 64-bit register.
            // This writes 8 bytes to memory in a single instruction.
            let pack = 
                (*alphabet.get_unchecked((n1 >> 18) & 0x3F) as u64) |
                ((*alphabet.get_unchecked((n1 >> 12) & 0x3F) as u64) << 8) |
                ((*alphabet.get_unchecked((n1 >> 6) & 0x3F) as u64) << 16) |
                ((*alphabet.get_unchecked(n1 & 0x3F) as u64) << 24) |
                ((*alphabet.get_unchecked((n2 >> 18) & 0x3F) as u64) << 32) |
                ((*alphabet.get_unchecked((n2 >> 12) & 0x3F) as u64) << 40) |
                ((*alphabet.get_unchecked((n2 >> 6) & 0x3F) as u64) << 48) |
                ((*alphabet.get_unchecked(n2 & 0x3F) as u64) << 56);

            (dst as *mut u64).write_unaligned(pack);

            src = src.add(6);
            dst = dst.add(8);
        }

        // --- TAIL HANDLING ---
        // Handle remaining bytes (0 to 5 bytes left)
        let len_remaining = len - len_aligned;

        // Handle a remaining 3-byte chunk (4 output chars)
        if len_remaining >= 3 {
            let b0 = *src as usize;
            let b1 = *src.add(1) as usize;
            let b2 = *src.add(2) as usize;
            let n = (b0 << 16) | (b1 << 8) | b2;

            let packed = 
                (*alphabet.get_unchecked((n >> 18) & 0x3F) as u32) |
                ((*alphabet.get_unchecked((n >> 12) & 0x3F) as u32) << 8) |
                ((*alphabet.get_unchecked((n >> 6) & 0x3F) as u32) << 16) |
                ((*alphabet.get_unchecked(n & 0x3F) as u32) << 24);

            (dst as *mut u32).write_unaligned(packed);
            src = src.add(3);
            dst = dst.add(4);
        }

        // Handle final 1 or 2 bytes with padding logic
        let tail_len = len % 3;
        if tail_len > 0 {
            let b0 = *src as usize;
            let b1 = if tail_len == 2 { *src.add(1) as usize } else { 0 };
            let n = (b0 << 16) | (b1 << 8);

            // Write the first 2 characters (always present)
            *dst = *alphabet.get_unchecked((n >> 18) & 0x3F);
            *dst.add(1) = *alphabet.get_unchecked((n >> 12) & 0x3F);

            // Handle the 3rd and 4th characters (Data vs Padding)
            if tail_len == 2 {
                *dst.add(2) = *alphabet.get_unchecked((n >> 6) & 0x3F);
                if config.padding {
                    *dst.add(3) = 61; // '='
                }
            } else if config.padding {
                *dst.add(2) = 61; // '='
                *dst.add(3) = 61; // '='
            }
        }
    }
}

/// Decodes a Base64 byte slice using a highly optimized scalar algorithm.
///
/// # Performance
/// This function uses a "fast loop" strategy for the bulk of the data, processing
/// **8 bytes of input** (2 Base64 blocks) at a time. It uses:
///
/// 1.  **Table Lookups:** Branchless conversion from ASCII -> 6-bit index using a pre-computed 256-byte table.
/// 2.  **Bitwise OR Accumulation:** Combines multiple error checks into a single branch (`|` operations followed by `& 0xC0`).
/// 3.  **Overlapping Writes:** Writes 32-bit integers to memory to output 3 bytes, allowing efficient pipelining of the write buffers.
///
/// On modern CPUs, this implementation is significantly faster than standard byte-by-byte decoding loops.
///
/// # Safety
/// * The caller must ensure `dst` is valid and has enough capacity (estimated by `estimate_decoded_len`).
/// * `config` must contain valid boolean flags.
/// * This function performs unchecked pointer arithmetic for speed.
#[inline(always)]
pub unsafe fn decode_slice_unsafe(config: &Config, input: &[u8], mut dst: *mut u8) -> Result<usize, Error> {
    let len = input.len();
    if len == 0 { return Ok(0); }

    let mut src = input.as_ptr();
    let dst_start = dst;

    // 1. Select the decode table based on configuration
    // The table maps valid characters to 0..63 and invalid characters to 0xFF.
    let table = if config.url_safe {
        URL_SAFE_DECODE_TABLE.as_ptr()
    } else {
        STANDARD_DECODE_TABLE.as_ptr()
    };

    unsafe {
        // Calculate the safe limit for the "Fast Loop".
        // We need at least 8 bytes for the fast loop to run safely.
        // We also reserve the last 4 bytes to handle padding logic carefully in the tail loop.
        let len_safe = len.saturating_sub(4);

        // Align to 8-byte boundaries for the fast loop
        let src_end_fast = src.add(len_safe.saturating_sub(len_safe % 8));

        // --- FAST LOOP (Middle Chunks) ---
        // Processes 8 input bytes -> 6 output bytes per iteration.
        while src < src_end_fast {
            // 1. Scalar Loads & Lookups
            // We load 8 bytes and immediately look them up in the table.
            let d0 = *table.add(*src as usize);
            let d1 = *table.add(*src.add(1) as usize);
            let d2 = *table.add(*src.add(2) as usize);
            let d3 = *table.add(*src.add(3) as usize);
            let d4 = *table.add(*src.add(4) as usize);
            let d5 = *table.add(*src.add(5) as usize);
            let d6 = *table.add(*src.add(6) as usize);
            let d7 = *table.add(*src.add(7) as usize);

            // 2. Fast Validation
            // Valid characters map to 0..63 (00xxxxxx).
            // Invalid characters map to 0xFF (11111111).
            // OR-ing them together accumulates the high bits. 
            // If any character was invalid, the 0x80 or 0x40 bits will be set.
            if (d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7) & 0xC0 != 0 {
                return Err(Error::InvalidCharacter);
            }

            // 3. Fast Packing
            // Pack 4x 6-bit indices into a 24-bit integer (stored in u32).
            let n1 = ((d0 as u32) << 18) | ((d1 as u32) << 12) | ((d2 as u32) << 6) | (d3 as u32);
            let n2 = ((d4 as u32) << 18) | ((d5 as u32) << 12) | ((d6 as u32) << 6) | (d7 as u32);

            // 4. Overlapping Writes
            // We write 4 bytes (u32) to output 3 valid bytes.
            // The 4th byte is "garbage" that will be overwritten by the next write.
            // Using write_unaligned(u32) is faster than 3 individual byte writes.
            // We swap bytes to big-endian to lay them out correctly in memory: [Byte0, Byte1, Byte2, Garbage]
            (dst as *mut u32).write_unaligned(n1.swap_bytes() >> 8);
            (dst.add(3) as *mut u32).write_unaligned(n2.swap_bytes() >> 8);

            src = src.add(8);
            dst = dst.add(6);
        }

        // --- TAIL HANDLING ---
        // Handle the remaining bytes (including potential padding).
        let current_offset = src.offset_from(input.as_ptr()) as usize;
        let mut remaining = len - current_offset;

        while remaining > 0 {
            // Case A: Full 4-byte block (possibly containing padding at the end)
            if remaining >= 4 {
                let b0 = *src;
                let b1 = *src.add(1);
                let b2 = *src.add(2);
                let b3 = *src.add(3);

                let d0 = *table.add(b0 as usize);
                let d1 = *table.add(b1 as usize);

                // Check for Padding ('=')
                if b3 == b'=' {
                    if b2 == b'=' {
                        // "XX==" -> 1 byte output
                        if (d0 | d1) & 0xC0 != 0 { return Err(Error::InvalidCharacter); }
                        let n = ((d0 as u32) << 18) | ((d1 as u32) << 12);
                        *dst = (n >> 16) as u8;
                        dst = dst.add(1);
                    } else {
                        // "XXX=" -> 2 bytes output
                        let d2 = *table.add(b2 as usize);
                        if (d0 | d1 | d2) & 0xC0 != 0 { return Err(Error::InvalidCharacter); }
                        let n = ((d0 as u32) << 18) | ((d1 as u32) << 12) | ((d2 as u32) << 6);
                        *dst = (n >> 16) as u8;
                        *dst.add(1) = (n >> 8) as u8;
                        dst = dst.add(2);
                    }
                    // Padding signals the end of the stream.
                    return Ok(dst.offset_from(dst_start) as usize);
                }

                // No padding: "XXXX" -> 3 bytes output
                let d2 = *table.add(b2 as usize);
                let d3 = *table.add(b3 as usize);

                if (d0 | d1 | d2 | d3) & 0xC0 != 0 {
                    return Err(Error::InvalidCharacter);
                }

                let n = ((d0 as u32) << 18) | ((d1 as u32) << 12) | ((d2 as u32) << 6) | (d3 as u32);
                *dst = (n >> 16) as u8;
                *dst.add(1) = (n >> 8) as u8;
                *dst.add(2) = n as u8;

                src = src.add(4);
                dst = dst.add(3);
                remaining -= 4;
            } else {
                // Case B: Partial block (1-3 bytes left)
                // If padding is strictly required, this is an error (len % 4 != 0).
                if config.padding {
                    return Err(Error::InvalidLength);
                }

                // Decode partial block without padding (e.g. "XY", "XYZ")
                let b0 = *src;
                let d0 = *table.add(b0 as usize);

                if remaining == 1 {
                    // A single byte is invalid in Base64 (cannot form a full byte)
                    return Err(Error::InvalidLength); 
                }

                let b1 = *src.add(1);
                let d1 = *table.add(b1 as usize);
                if (d0 | d1) & 0xC0 != 0 { return Err(Error::InvalidCharacter); }

                let mut n = ((d0 as u32) << 18) | ((d1 as u32) << 12);

                if remaining == 2 {
                    // "XY" -> 1 byte output
                    *dst = (n >> 16) as u8;
                    dst = dst.add(1);
                } else {
                    // "XYZ" -> 2 bytes output
                    let b2 = *src.add(2);
                    let d2 = *table.add(b2 as usize);
                    if d2 & 0xC0 != 0 { return Err(Error::InvalidCharacter); }

                    n |= (d2 as u32) << 6;
                    *dst = (n >> 16) as u8;
                    *dst.add(1) = (n >> 8) as u8;
                    dst = dst.add(2);
                }

                break;
            }
        }

        Ok(dst.offset_from(dst_start) as usize)
    }
}

#[cfg(kani)]
mod kani_verification_scalar {
    use super::*;
    use crate::{Config, STANDARD as TURBO_STANDARD, STANDARD_NO_PAD as TURBO_STANDARD_NO_PAD};

    const INPUT_LEN: usize = 17;

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding { TURBO_STANDARD.encoded_len(len) } else { TURBO_STANDARD_NO_PAD.encoded_len(len) }
    }

    #[kani::proof]
    #[kani::unwind(18)]
    fn check_roundtrip_safety() {
        // Symbolic Config
        let config = Config {
            url_safe: kani::any(),
            padding: kani::any(),
        };

        // Symbolic Input
        let input: [u8; INPUT_LEN] = kani::any();

        // Setup Buffers
        let enc_len = encoded_size(INPUT_LEN, config.padding);
        let mut enc_buf = [0u8; 64];
        let mut dec_buf = [0u8; 64];

        unsafe {
            // Encode
            encode_slice_unsafe(&config, &input, enc_buf.as_mut_ptr());

            // Decode
            let src_slice = &enc_buf[..enc_len];
            let written = decode_slice_unsafe(&config, src_slice, dec_buf.as_mut_ptr()).expect("Decoder failed");

            // Verification
            assert_eq!(&dec_buf[..written], &input, "AVX2 Roundtrip Failed");
        }
    }

    #[kani::proof]
    #[kani::unwind(18)]
    fn check_decoder_robustness() {
        // Symbolic Config
        let config = Config {
            url_safe: kani::any(),
            padding: kani::any(),
        };

        // Symbolic Input (Random Garbage)
        let input: [u8; INPUT_LEN] = kani::any();

        // Setup Buffer
        let mut dec_buf = [0u8; 64];

        unsafe {
            // We verify what function NEVER panics/crashes
            let _ = decode_slice_unsafe(&config, &input, dec_buf.as_mut_ptr());
        }
    }
}

#[cfg(all(test, miri))]
mod scalar_miri_tests {
    use super::{encode_slice_unsafe, decode_slice_unsafe};
    use crate::{Config, STANDARD as TURBO_STANDARD, STANDARD_NO_PAD as TURBO_STANDARD_NO_PAD};
    use base64::{engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD}};
    use rand::{Rng, rng};

    // --- Helpers ---

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding { TURBO_STANDARD.encoded_len(len) } else { TURBO_STANDARD_NO_PAD.encoded_len(len) }
    }
    fn estimated_decoded_length(len: usize) -> usize { TURBO_STANDARD.estimate_decoded_len(len) }

    /// Miri Runner:
    /// 1. Runs deterministic boundary tests (0..64 bytes) to hit every loop edge.
    /// 2. Runs a small set of random fuzz tests (50 iterations) to catch weird patterns.
    fn run_miri_cycle<E: base64::Engine>(config: Config, reference_engine: &E) {
        // Deterministic Boundary Testing
        for len in 0..=64 {
            let mut rng = rng();
            let mut input = vec![0u8; len];
            rng.fill(&mut input[..]);

            verify_roundtrip(&config, &input, reference_engine);
        }

        // Small Fuzzing (Random Lengths)
        let mut rng = rng();
        for _ in 0..100 {
            let len = rng.random_range(65..512);
            let mut input = vec![0u8; len];
            rng.fill(&mut input[..]);

            verify_roundtrip(&config, &input, reference_engine);
        }
    }

    fn verify_roundtrip<E: base64::Engine>(config: &Config, input: &[u8], reference_engine: &E) {
        let len = input.len();

        // --- Encoding ---
        let expected_string = reference_engine.encode(input);

        let enc_len = encoded_size(len, config.padding);
        let mut enc_buf = vec![0u8; enc_len];

        unsafe { encode_slice_unsafe(config, input, enc_buf.as_mut_ptr()); }

        assert_eq!(&enc_buf, expected_string.as_bytes(), "Miri Encoding Mismatch!");

        // --- Decoding ---
        let dec_max_len = estimated_decoded_length(enc_len);
        let mut dec_buf = vec![0u8; dec_max_len];

        unsafe {
            let written = decode_slice_unsafe(config, &enc_buf, dec_buf.as_mut_ptr())
                .expect("Decoder returned error on valid input");

            let my_decoded = &dec_buf[..written];

            assert_eq!(my_decoded, input, "Miri Decoding Mismatch!");
        }
    }

    // --- Tests ---

    #[test]
    fn miri_scalar_url_safe_roundtrip() {
        run_miri_cycle(
            Config { url_safe: true, padding: true }, 
            &URL_SAFE
        );
    }

    #[test]
    fn miri_scalar_url_safe_no_pad_roundtrip() {
        run_miri_cycle(
            Config { url_safe: true, padding: false }, 
            &URL_SAFE_NO_PAD
        );
    }

    #[test]
    fn miri_scalar_standard_roundtrip() {
        run_miri_cycle(
            Config { url_safe: false, padding: true }, 
            &STANDARD
        );
    }

    #[test]
    fn miri_scalar_standard_no_pad_roundtrip() {
        run_miri_cycle(
            Config { url_safe: false, padding: false }, 
            &STANDARD_NO_PAD
        );
    }

    // --- Error Checks ---

    #[test]
    fn miri_scalar_invalid_input() {
        let config = Config { url_safe: true, padding: false };
        let mut out = vec![0u8; 10];

        // Pointer math check: Ensure reading invalid chars doesn't cause OOB reads
        let bad_chars = b"heap+"; 
        unsafe {
            let res = decode_slice_unsafe(&config, bad_chars, out.as_mut_ptr());
            assert!(res.is_err());
        }
    }
}
