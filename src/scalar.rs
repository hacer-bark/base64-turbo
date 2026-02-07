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
/// # Safety
/// This function is **unsafe** and requires the caller to uphold strict memory contracts. 
/// Failure to do so will result in **Undefined Behavior** (buffer overflow).
///
/// * **Output Capacity**: The memory region pointed to by `dst` must have sufficient capacity 
///   to store the encoded output. The required minimum size depends on `config.padding`:
///     * If `padding` is **true**: `input.len().div_ceil(3) * 4`
///     * If `padding` is **false**: `(input.len() * 4).div_ceil(3)`
/// * **Pointer Validity**: `dst` must point to a valid, mutable memory region.
///
/// # Internal Use Only
/// This is a low-level primitive intended for internal use. Callers should prefer the 
/// safe, higher-level APIs (e.g., `Engine::encode`) which automatically handle 
/// buffer allocation and configuration logic.
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
/// # Safety
/// This function is **unsafe** and requires the caller to uphold strict memory contracts. 
/// Failure to do so will result in **Undefined Behavior** (buffer overflow).
///
/// * **Output Capacity**: The memory region pointed to by `dst` must have sufficient capacity 
///   to store the decoded output. Due to SIMD optimizations performing overlapping writes, 
///   the destination buffer **must** be at least `(input.len() / 4 + 1) * 3` bytes.
/// * **Pointer Validity**: `dst` must point to a valid, mutable memory region.
///
/// # Internal Use Only
/// This is a low-level primitive intended for internal use by the `Engine`.
/// Callers should prefer the safe, higher-level APIs (e.g., `Engine::decode`), which
/// automatically handle buffer sizing via `Engine::estimate_decoded_len`.
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

    // Magic number
    // It handles 2 loops unroll + tail.
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
mod scalar_miri_coverage {
    use super::*;
    use base64::{engine::general_purpose::{STANDARD, STANDARD_NO_PAD}, Engine};
    use rand::{Rng, rng};

    // --- Mock Infrastructure ---
    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    /// Helper to verify Scalar encoding against the 'base64' crate oracle
    fn verify_encode(config: &Config, oracle: &impl Engine, input_len: usize) {
        let input = random_bytes(input_len);
        let expected = oracle.encode(&input);

        // Calculate exact required size
        let len_required = if config.padding { input_len.div_ceil(3) * 4 } else { (input_len * 4).div_ceil(3) };
        let mut dst = vec![0u8; len_required];

        unsafe { encode_slice_unsafe(config, &input, dst.as_mut_ptr()); }

        assert_eq!(std::str::from_utf8(&dst).unwrap(), expected, "Encode len {}", input_len);
    }

    /// Helper to verify Scalar decoding against the 'base64' crate oracle
    fn verify_decode(config: &Config, oracle: &impl Engine, original_len: usize) {
        // 1. Generate valid Base64 via oracle
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();

        // 2. Prepare destination (Exact size required by contract)
        // Contract: (input.len() / 4 + 1) * 3
        let cap = (encoded_bytes.len() / 4 + 1) * 3;
        let mut dst = vec![0u8; cap];

        // 3. Run
        let len = unsafe {
            decode_slice_unsafe(config, encoded_bytes, dst.as_mut_ptr()).expect("Valid input failed to decode")
        };

        // 4. Verify
        assert_eq!(&dst[..len], &input_bytes, "Decode len {}", original_len);
    }

    // ----------------------------------------------------------------------
    // 1. Encoder Logic Coverage
    // ----------------------------------------------------------------------

    #[test]
    fn miri_scalar_encode_fast_loop() {
        let config = Config { url_safe: false, padding: true };
        // The loop processes 6 bytes at a time.
        verify_encode(&config, &STANDARD, 6); // Exactly 1 loop
        verify_encode(&config, &STANDARD, 12); // Exactly 2 loops
    }

    #[test]
    fn miri_scalar_encode_tail_logic() {
        let config = Config { url_safe: false, padding: true };
        // 3 bytes -> handled by `if len_remaining >= 3`
        verify_encode(&config, &STANDARD, 3);
        // 4 bytes -> 3 byte block + 1 byte tail (Padding: "==")
        verify_encode(&config, &STANDARD, 4);
        // 5 bytes -> 3 byte block + 2 byte tail (Padding: "=")
        verify_encode(&config, &STANDARD, 5);
        // 1 byte -> Direct tail (Padding: "==")
        verify_encode(&config, &STANDARD, 1);
        // 2 bytes -> Direct tail (Padding: "=")
        verify_encode(&config, &STANDARD, 2);
    }

    #[test]
    fn miri_scalar_encode_no_padding() {
        let config = Config { url_safe: false, padding: false };
        // Verify `if config.padding` checks in tail logic
        verify_encode(&config, &STANDARD_NO_PAD, 1);
        verify_encode(&config, &STANDARD_NO_PAD, 2);
        verify_encode(&config, &STANDARD_NO_PAD, 4);
        verify_encode(&config, &STANDARD_NO_PAD, 5);
    }

    #[test]
    fn miri_scalar_encode_url_safe() {
        // Verify alphabet switch
        let config = Config { url_safe: true, padding: false };
        let input = vec![0xFB, 0xFF, 0xFF]; 
        let mut dst = vec![0u8; 4];
        unsafe { encode_slice_unsafe(&config, &input, dst.as_mut_ptr()) };
        assert_eq!(&dst, b"-___");
    }

    // ----------------------------------------------------------------------
    // 2. Decoder Logic Coverage
    // ----------------------------------------------------------------------

    #[test]
    fn miri_scalar_decode_fast_loop() {
        let config = Config { url_safe: false, padding: true };
        // Fast loop processes 8 bytes (2 blocks).
        verify_decode(&config, &STANDARD, 6);  // 8 chars encoded
        verify_decode(&config, &STANDARD, 12); // 16 chars encoded (2 loops)
    }

    #[test]
    fn miri_scalar_decode_tail_padded() {
        let config = Config { url_safe: false, padding: true };
        // Logic: if b3 == '=' and if b2 == '='

        // "XX==" case (1 output byte)
        let mut dst = [0u8; 3];
        let len = unsafe { decode_slice_unsafe(&config, b"QQ==", dst.as_mut_ptr()).unwrap() };
        assert_eq!(len, 1);

        // "XXX=" case (2 output bytes)
        let len = unsafe { decode_slice_unsafe(&config, b"QUE=", dst.as_mut_ptr()).unwrap() };
        assert_eq!(len, 2);
    }

    #[test]
    fn miri_scalar_decode_tail_no_pad() {
        let config = Config { url_safe: false, padding: false };
        // Logic: `Case B: Partial block`

        // "XY" (2 chars -> 1 byte)
        let mut dst = [0u8; 3];
        let len = unsafe { decode_slice_unsafe(&config, b"QQ", dst.as_mut_ptr()).unwrap() };
        assert_eq!(len, 1);

        // "XYZ" (3 chars -> 2 bytes)
        let len = unsafe { decode_slice_unsafe(&config, b"QUE", dst.as_mut_ptr()).unwrap() };
        assert_eq!(len, 2);
    }

    #[test]
    fn miri_scalar_decode_url_safe() {
        let config = Config { url_safe: true, padding: false };
        // Verify table selection
        let mut dst = [0u8; 3];
        // '-' (62) and '_' (63)
        let len = unsafe { decode_slice_unsafe(&config, b"-_", dst.as_mut_ptr()).unwrap() };
        assert_eq!(len, 1);
    }

    // ----------------------------------------------------------------------
    // 3. Error Logic Coverage
    // ----------------------------------------------------------------------

    #[test]
    fn miri_scalar_errors() {
        let config = Config { url_safe: false, padding: true };
        let mut dst = [0u8; 10];

        // 1. Invalid Char in Fast Loop
        // "A" maps to 0, "!" maps to 0xFF. 0xFF | 0x... has high bit set.
        let bad_fast = b"AAAAAA!A"; 
        unsafe {
            assert_eq!(decode_slice_unsafe(&config, bad_fast, dst.as_mut_ptr()), Err(Error::InvalidCharacter));
        }

        // 2. Invalid Char in Tail (Full Block)
        // "AA!A"
        let bad_tail = b"AA!A";
        unsafe {
            assert_eq!(decode_slice_unsafe(&config, bad_tail, dst.as_mut_ptr()), Err(Error::InvalidCharacter));
        }

        // 3. Invalid Padding Position
        // "A==A" -> b2 is =, b3 is A.
        let bad_pad = b"A==A";
        unsafe {
            // This falls through to standard decode of '=', which is 0xFF (invalid)
            assert_eq!(decode_slice_unsafe(&config, bad_pad, dst.as_mut_ptr()), Err(Error::InvalidCharacter));
        }

        // 4. Missing Padding when Config Requires It
        let short_input = b"QQ"; // valid chars, but length 2
        unsafe {
            assert_eq!(decode_slice_unsafe(&config, short_input, dst.as_mut_ptr()), Err(Error::InvalidLength));
        }

        // 5. Single Byte (Impossible in Base64)
        let single = b"A";
        // Check with padding=false to hit the specific branch in `Case B`
        let config_no_pad = Config { url_safe: false, padding: false };
        unsafe {
            assert_eq!(decode_slice_unsafe(&config_no_pad, single, dst.as_mut_ptr()), Err(Error::InvalidLength));
        }
    }
}
