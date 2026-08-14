//! AVX2 verification: Kani proofs, Intel-pseudocode intrinsic model stubs,
//! the model/hardware equivalence check, and the Miri + hardware coverage
//! suites. Split out of the production module purely to keep it lean.

use super::*;

#[cfg(kani)]
mod kani_verification_avx2 {
    use super::*;
    use crate::{Config, STANDARD as TURBO_STANDARD, STANDARD_NO_PAD as TURBO_STANDARD_NO_PAD};

    // Only used inside `#[kani::stub(...)]` paths, which don't count as a use.
    #[allow(unused_imports)]
    use super::intrinsic_models as m;

    // Layer 1 — index proofs: reason over a symbolic `len` and an arbitrary
    // iteration index (no vectors), giving an induction (base/step/exit) that
    // covers all N cheaply. The consts mirror the kernels' offset arithmetic;
    // keep them in sync by hand. See the README's "How the Kani proofs work".

    /// Largest `len` considered: above `usize::MAX / 4` the unpadded
    /// `encoded_len`'s `len * 4` overflows, so the API can't size a buffer.
    const MAX_LEN: usize = usize::MAX / 4;

    // Encoder model, mirroring `encode_slice_avx2`.
    const ENC_ROUND_IN: usize = 24; // logical input bytes per round
    const ENC_ROUND_OUT: usize = 32; // output bytes per round
    const ENC_LOAD: usize = 32; // bytes each `_mm256_loadu_si256` reads
    const ENC_FIRST_ADVANCE: usize = 20; // `src.add(20)` after the first round
    const ENC_UNROLL: usize = 4; // rounds per 4x-unrolled iteration

    fn enc_cap(len: usize, padding: bool) -> usize {
        if padding {
            TURBO_STANDARD.encoded_len(len)
        } else {
            TURBO_STANDARD_NO_PAD.encoded_len(len)
        }
    }

    /// Symbolic `rounds` pinned to `(len - 4) / 24` via inequalities (cheaper
    /// for CBMC than division; `check_enc_rounds_model` proves they agree).
    fn any_enc_rounds(len: usize) -> usize {
        let rounds: usize = kani::any();
        kani::assume(rounds <= MAX_LEN / ENC_ROUND_IN);
        kani::assume(ENC_ROUND_IN * rounds <= len - 4);
        kani::assume(len - 4 < ENC_ROUND_IN * (rounds + 1));
        rounds
    }

    /// `(src_off, dst_off)` after `done >= 1` rounds: the first round advances
    /// `src` by only 20, giving the uniform `24 * done - 4`.
    fn enc_state(done: usize) -> (usize, usize) {
        (ENC_ROUND_IN * done - 4, ENC_ROUND_OUT * done)
    }

    /// Isolated so the suite's one non-power-of-two division owns its run.
    #[kani::proof]
    fn check_enc_rounds_model() {
        let len: usize = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        assert_eq!(rounds, (len - 4) / ENC_ROUND_IN);
        // `remaining = rounds - 1` must not underflow (why the guard is >= 32).
        assert!(rounds >= 1);
    }

    /// Base case: the permuted first round is in bounds.
    #[kani::proof]
    fn check_enc_first_block() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let cap = enc_cap(len, padding);

        assert!(ENC_LOAD <= len); // reads [0, 32)
        assert!(ENC_ROUND_OUT <= cap); // writes [0, 32)

        let (src_off, dst_off) = enc_state(1);
        assert_eq!(src_off, ENC_FIRST_ADVANCE);
        assert_eq!(dst_off, ENC_ROUND_OUT);
        assert!(rounds >= 1);
    }

    /// Inductive step for the 4x-unrolled tier, over an arbitrary iteration.
    #[kani::proof]
    fn check_enc_quad_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let cap = enc_cap(len, padding);

        let done: usize = kani::any();
        kani::assume(done >= 1 && done <= rounds);
        let remaining = rounds - done;
        kani::assume(remaining >= ENC_UNROLL); // guard `while remaining >= 4`

        let (src_off, dst_off) = enc_state(done);

        // Widest body accesses: `src.add(72)` load, `dst.add(96)` store.
        assert!(src_off + 72 + ENC_LOAD <= len, "quad load leaves input");
        assert!(
            dst_off + 96 + ENC_ROUND_OUT <= cap,
            "quad store leaves output"
        );

        // `src += 96`, `dst += 128`, `remaining -= 4` lands on the next state.
        let done_next = done + ENC_UNROLL;
        assert_eq!((src_off + 96, dst_off + 128), enc_state(done_next));
        assert!(done_next <= rounds);
        assert_eq!(remaining - ENC_UNROLL, rounds - done_next);
    }

    /// Inductive step for the single-round tier.
    #[kani::proof]
    fn check_enc_single_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let cap = enc_cap(len, padding);

        let done: usize = kani::any();
        kani::assume(done >= 1 && done <= rounds);
        let remaining = rounds - done;
        kani::assume(remaining >= 1);

        let (src_off, dst_off) = enc_state(done);
        assert!(src_off + ENC_LOAD <= len, "single load leaves input");
        assert!(dst_off + ENC_ROUND_OUT <= cap, "single store leaves output");

        // Update `src.add(24)`, `dst.add(32)`, `remaining -= 1`.
        let done_next = done + 1;
        assert_eq!(
            (src_off + ENC_ROUND_IN, dst_off + ENC_ROUND_OUT),
            enc_state(done_next)
        );
        assert!(done_next <= rounds);
        assert_eq!(remaining - 1, rounds - done_next);
    }

    /// Exit case: the trailing `src.add(4)` + scalar handoff account for the rest.
    #[kani::proof]
    fn check_enc_tail_handoff() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((32..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let (src_off, dst_off) = enc_state(rounds);

        let processed = src_off + 4; // repays the first round's deficit
        assert_eq!(processed, ENC_ROUND_IN * rounds);

        // `rounds` caps at `(len - 4) / 24`, so the tail is non-empty (>= 4).
        assert!(processed < len);
        let tail = len - processed;
        assert!(tail >= 4);

        // Prefix + scalar tail is exactly the encoded length (no over/short write).
        assert_eq!(
            dst_off + enc_cap(tail, padding),
            enc_cap(len, padding),
            "prefix + tail must equal encoded length"
        );
    }

    // Decoder model, mirroring `decode_slice_avx2`.
    const DEC_LOAD: usize = 32; // bytes each `_mm256_loadu_si256` reads
    const DEC_BLOCK_IN: usize = 32; // input bytes per single-vector pass
    const DEC_BLOCK_OUT: usize = 24; // dst advance per single-vector pass
    /// Bytes `pack_and_store!` touches (16 at `dst` + 16 at `dst.add(12)`),
    /// 4 wider than the 24 it advances.
    const DEC_STORE_SPAN: usize = 28;
    const DEC_QUAD_IN: usize = 128; // input bytes per quad-tier iteration
    const DEC_QUAD_OUT: usize = 96; // dst advance per quad-tier iteration

    fn dec_cap(len: usize) -> usize {
        TURBO_STANDARD.estimate_decoded_len(len)
    }

    /// The `aligned_len_128` / `aligned_len_32` loop windows (from the
    /// `saturating_sub(4)` margin that keeps a 32-byte load in bounds).
    fn dec_windows(len: usize) -> (usize, usize) {
        let safe = len.saturating_sub(4);
        (safe - safe % DEC_QUAD_IN, safe - safe % DEC_BLOCK_IN)
    }

    /// Inductive step for the decoder's quad tier, over an arbitrary iteration.
    #[kani::proof]
    fn check_dec_quad_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (aligned_quad, _) = dec_windows(len);
        let cap = dec_cap(len);

        let i: usize = kani::any();
        kani::assume(i <= MAX_LEN / DEC_QUAD_IN);
        let (src_off, dst_off) = (DEC_QUAD_IN * i, DEC_QUAD_OUT * i);
        kani::assume(src_off < aligned_quad); // guard `src < src_end_128`

        // Widest: `src.add(96)` load, `pack_and_store!(_, dst.add(72))`.
        assert!(src_off + 96 + DEC_LOAD <= len, "quad load leaves input");
        assert!(
            dst_off + 72 + DEC_STORE_SPAN <= cap,
            "quad store leaves output"
        );

        // Update `src.add(128)`, `dst.add(96)`.
        assert_eq!(
            (src_off + DEC_QUAD_IN, dst_off + DEC_QUAD_OUT),
            (DEC_QUAD_IN * (i + 1), DEC_QUAD_OUT * (i + 1))
        );
    }

    /// Inductive step for the decoder's single-vector tier, entered from
    /// wherever the quad tier stopped.
    #[kani::proof]
    fn check_dec_single_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (aligned_quad, aligned_block) = dec_windows(len);
        let cap = dec_cap(len);
        let quads = aligned_quad / DEC_QUAD_IN;

        let j: usize = kani::any();
        kani::assume(j <= MAX_LEN / DEC_BLOCK_IN);
        let src_off = aligned_quad + DEC_BLOCK_IN * j;
        let dst_off = DEC_QUAD_OUT * quads + DEC_BLOCK_OUT * j;
        kani::assume(src_off < aligned_block); // guard `src < src_end_32`

        assert!(src_off + DEC_LOAD <= len, "single load leaves input");
        assert!(
            dst_off + DEC_STORE_SPAN <= cap,
            "single store leaves output"
        );

        // Update `src.add(32)`, `dst.add(24)`.
        assert_eq!(
            (src_off + DEC_BLOCK_IN, dst_off + DEC_BLOCK_OUT),
            (
                aligned_quad + DEC_BLOCK_IN * (j + 1),
                DEC_QUAD_OUT * quads + DEC_BLOCK_OUT * (j + 1)
            )
        );
    }

    /// Exit case: whatever the loops leave fits the space the caller
    /// guaranteed, so the scalar decoder cannot overrun it.
    #[kani::proof]
    fn check_dec_tail_handoff() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (_, aligned_block) = dec_windows(len);
        let cap = dec_cap(len);

        // Both tiers advance dst 3 bytes per 4 consumed, so the handover
        // offset depends only on the window.
        let dst_off = DEC_BLOCK_OUT * (aligned_block / DEC_BLOCK_IN);
        assert!(aligned_block <= len);
        let tail = len - aligned_block;
        assert!(
            dst_off + dec_cap(tail) <= cap,
            "scalar tail can overrun output"
        );
    }

    // Layer 2 — kernel proofs: run the real code over symbolic bytes (character
    // mapping, validation LUTs, panic freedom). Layer 1 owns the loop
    // arithmetic, so each reaches its kernel once. Buffers are the exact
    // public-API capacities, so any real overrun fails.

    /// One first round + 13-byte scalar tail.
    const ENC_KERNEL_LEN: usize = 37;
    /// One single-vector pass + 5-byte scalar tail.
    const DEC_KERNEL_LEN: usize = 37;

    // Guard: a length below its tier's threshold would prove nothing (a past
    // revision silently verified only the scalar fallback). Fail the build.
    const _: () = assert!(
        ENC_KERNEL_LEN >= 32
            && (ENC_KERNEL_LEN - 4) / ENC_ROUND_IN == 1
            && ENC_KERNEL_LEN % ENC_ROUND_IN != 0,
        "ENC_KERNEL_LEN must run one AVX2 round and leave an unaligned tail"
    );
    const _: () = assert!(
        (DEC_KERNEL_LEN - 4) / DEC_BLOCK_IN == 1,
        "DEC_KERNEL_LEN must run one single-vector decode pass"
    );

    const ENC_KERNEL_CAP: usize = TURBO_STANDARD.encoded_len(ENC_KERNEL_LEN);
    const ENC_KERNEL_DEC_CAP: usize = TURBO_STANDARD.estimate_decoded_len(ENC_KERNEL_CAP);
    const DEC_KERNEL_CAP: usize = TURBO_STANDARD.estimate_decoded_len(DEC_KERNEL_LEN);

    /// `Decode(Encode(x)) == x` over every 37-byte input. `url_safe` is a
    /// parameter (not symbolic) since it only selects constant LUTs.
    fn roundtrip_kernel(url_safe: bool) {
        let config = Config {
            url_safe,
            padding: true,
        };
        let input: [u8; ENC_KERNEL_LEN] = kani::any();

        let mut enc_buf = [0u8; ENC_KERNEL_CAP];
        let mut dec_buf = [0u8; ENC_KERNEL_DEC_CAP];

        unsafe {
            encode_slice_avx2(&config, &input, &mut enc_buf);
            let dec_len = decode_slice_avx2(&config, &enc_buf, &mut dec_buf)
                .expect("valid encoding failed to decode");
            assert_eq!(dec_len, ENC_KERNEL_LEN);
            assert_eq!(&dec_buf[..dec_len], &input, "roundtrip mismatch");
        }
    }

    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, m::_mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, m::_mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, m::_mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, m::_mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, m::_mm256_madd_epi16_stub)]
    #[kani::stub(_mm256_mulhi_epu16, m::_mm256_mulhi_epu16_stub)]
    #[kani::stub(_mm256_permutevar8x32_epi32, m::_mm256_permutevar8x32_epi32_stub)]
    fn check_avx2_roundtrip_standard() {
        roundtrip_kernel(false);
    }

    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, m::_mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, m::_mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, m::_mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, m::_mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, m::_mm256_madd_epi16_stub)]
    #[kani::stub(_mm256_mulhi_epu16, m::_mm256_mulhi_epu16_stub)]
    #[kani::stub(_mm256_permutevar8x32_epi32, m::_mm256_permutevar8x32_epi32_stub)]
    fn check_avx2_roundtrip_url_safe() {
        roundtrip_kernel(true);
    }

    /// Every 37-byte garbage input decodes or returns `Err`, never panicking or
    /// overrunning — covers the validation LUTs over all 256 byte values.
    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, m::_mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, m::_mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, m::_mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, m::_mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, m::_mm256_madd_epi16_stub)]
    fn check_avx2_decode_robustness() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; DEC_KERNEL_LEN] = kani::any();
        let mut output = [0u8; DEC_KERNEL_CAP];
        unsafe {
            let _ = decode_slice_avx2(&config, &input, &mut output);
        }
    }
}

/// Rust models of every AVX2 intrinsic the kernels use, for the Kani proofs.
#[cfg(any(kani, test))]
#[allow(non_snake_case)]
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::missing_transmute_annotations,
    clippy::needless_late_init,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::used_underscore_items
)]
pub(super) mod intrinsic_models {
    use super::*;
    use std::mem::transmute;

    // STUB: _mm256_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_epi8
    pub(super) unsafe fn _mm256_shuffle_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [u8; 32] = unsafe { transmute(b) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*8
            // (In Rust we access bytes 'j' so '*8' offset is not needed)
            let i = j;

            // IF b[i+7] == 1
            if (b[i] & 0x80) != 0 {
                // dst[i+7:i] := 0
                dst[i] = 0;
            } else {
                // index[3:0] := b[i+3:i]
                let index = b[i] & 0x0F;
                // dst[i+7:i] := a[index*8+7:index*8]
                dst[i] = a[index as usize];
            }
            // FI

            // IF b[128+i+7] == 1
            if (b[16 + i] & 0x80) != 0 {
                // dst[128+i+7:128+i] := 0
                dst[16 + i] = 0;
            } else {
                // index[3:0] := b[128+i+3:128+i]
                let index = b[16 + i] & 0x0F;
                // dst[128+i+7:128+i] := a[128+index*8+7:128+index*8]
                dst[16 + i] = a[(16 + index) as usize];
            }
            // FI
        }
        // ENDFOR

        // dst[MAX:256] := 0
        // (__m256i is exactly 256 bits. There are no bits beyond 256 to zero out)

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_subs_epu8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_subs_epu8
    pub(super) unsafe fn _mm256_subs_epu8_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [u8; 32] = unsafe { transmute(b) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := j*8
            let i = j;

            // dst[i+7:i] := SaturateU8(a[i+7:i] - b[i+7:i])
            dst[i] = a[i].saturating_sub(b[i]);
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_testz_si256
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_testz_si256
    // Split into four u64 lanes since Rust has no native 256-bit integer type.
    pub(super) unsafe fn _mm256_testz_si256_stub(a: __m256i, b: __m256i) -> i32 {
        let a: [u64; 4] = unsafe { transmute(a) };
        let b: [u64; 4] = unsafe { transmute(b) };
        let zf: i32;
        let _cf: i32;

        // Perform 256 bit AND
        let res_and = [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]];

        // IF ((a[255:0] AND b[255:0]) == 0)
        if res_and[0] == 0 && res_and[1] == 0 && res_and[2] == 0 && res_and[3] == 0 {
            // ZF := 1
            zf = 1;
        } else {
            // ZF := 0
            zf = 0;
        }
        // FI

        // Perform 256 bit (NOT a) AND b
        let res_not_and = [
            (!a[0]) & b[0],
            (!a[1]) & b[1],
            (!a[2]) & b[2],
            (!a[3]) & b[3],
        ];

        // IF (((NOT a[255:0]) AND b[255:0]) == 0)
        if res_not_and[0] == 0 && res_not_and[1] == 0 && res_not_and[2] == 0 && res_not_and[3] == 0
        {
            // CF := 1
            _cf = 1;
        } else {
            // CF := 0
            _cf = 0;
        }
        // FI

        // RETURN ZF
        return zf;
    }

    // STUB: _mm256_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maddubs_epi16
    pub(super) unsafe fn _mm256_maddubs_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [i8; 32] = unsafe { transmute(b) };
        let mut dst = [0i16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j * 2;

            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i + 1] as i16) * (b[i + 1] as i16))
                .saturating_add((a[i] as i16) * (b[i] as i16));
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_madd_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_madd_epi16
    pub(super) unsafe fn _mm256_madd_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [i16; 16] = unsafe { transmute(a) };
        let b: [i16; 16] = unsafe { transmute(b) };
        let mut dst = [0i32; 8];

        // FOR j := 0 to 7
        for j in 0..8 {
            // i := j*32
            let i = j * 2;

            // dst[i+31:i] := SignExtend32(a[i+31:i+16]*b[i+31:i+16]) + SignExtend32(a[i+15:i]*b[i+15:i])
            dst[j] = (a[i + 1] as i32)
                .wrapping_mul(b[i + 1] as i32)
                .wrapping_add((a[i] as i32).wrapping_mul(b[i] as i32));
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_mulhi_epu16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mulhi_epu16
    pub(super) unsafe fn _mm256_mulhi_epu16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u16; 16] = unsafe { transmute(a) };
        let b: [u16; 16] = unsafe { transmute(b) };
        let mut dst = [0u16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*16
            let i = j;
            // tmp[31:0] := ZeroExtend32(a[i+15:i]) * ZeroExtend32(b[i+15:i])
            let tmp: u32 = (a[i] as u32) * (b[i] as u32);
            // dst[i+15:i] := tmp[31:16]
            dst[i] = (tmp >> 16) as u16;
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_permutevar8x32_epi32
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permutevar8x32_epi32
    pub(super) unsafe fn _mm256_permutevar8x32_epi32_stub(a: __m256i, idx: __m256i) -> __m256i {
        let a: [u32; 8] = unsafe { transmute(a) };
        let idx: [u32; 8] = unsafe { transmute(idx) };
        let mut dst = [0u32; 8];

        // FOR j := 0 to 7
        for j in 0..8 {
            // id := idx[j*32+2:j*32]
            let id = (idx[j] & 0x7) as usize;
            // dst[j*32+31:j*32] := a[id*32+31:id*32]
            dst[j] = a[id];
        }
        // ENDFOR

        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }
}

/// Checks every model in [`intrinsic_models`] against the real instruction
/// on AVX2 hardware, under plain `cargo test`. The Kani proofs trust the
/// models, so a model that disagrees with the silicon is the one assumption
/// underneath every Layer 2 result that Kani cannot itself check.
#[cfg(test)]
#[cfg(not(miri))]
#[allow(clippy::used_underscore_items)] // calling the models is the point
mod avx2_stub_equivalence {
    use super::intrinsic_models as model;
    use super::*;
    use std::mem::transmute;

    /// Saturation and sign boundaries, the high bit that zeroes a shuffle
    /// lane, index-shaped bytes, and deterministic noise.
    fn probes() -> Vec<[u8; 32]> {
        let byte = |i: usize| u8::try_from(i).expect("index below the 32-byte vector width");

        let mut out = vec![[0x00; 32], [0xFF; 32], [0x80; 32], [0x7F; 32], [0x01; 32]];
        out.push(core::array::from_fn(byte));
        out.push(core::array::from_fn(|i| byte(i) | 0x80));
        out.push(core::array::from_fn(|i| byte(i % 16)));
        out.push(core::array::from_fn(|i| 0xFF - byte(i)));

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..12 {
            out.push(core::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u8::try_from(state >> 56).expect("shifted down to 8 bits")
            }));
        }
        out
    }

    #[target_feature(enable = "avx2")]
    unsafe fn compare_all() {
        let probes = probes();
        // SAFETY: `__m256i` has no invalid bit patterns, so it and `[u8; 32]`
        // are freely transmutable both ways.
        let bytes = |v: __m256i| -> [u8; 32] { unsafe { transmute::<__m256i, [u8; 32]>(v) } };

        // Each arm: `real(a, b)` must equal `model(a, b)` for every probe pair.
        macro_rules! same {
            ($real:ident, $model:ident, $wrap:expr) => {
                for x in &probes {
                    for y in &probes {
                        let (a, b) = unsafe {
                            (
                                transmute::<[u8; 32], __m256i>(*x),
                                transmute::<[u8; 32], __m256i>(*y),
                            )
                        };
                        assert_eq!(
                            $wrap($real(a, b)),
                            $wrap(unsafe { model::$model(a, b) }),
                            "{}: a={x:02x?} b={y:02x?}",
                            stringify!($real)
                        );
                    }
                }
            };
        }

        same!(_mm256_shuffle_epi8, _mm256_shuffle_epi8_stub, bytes);
        same!(_mm256_subs_epu8, _mm256_subs_epu8_stub, bytes);
        same!(_mm256_maddubs_epi16, _mm256_maddubs_epi16_stub, bytes);
        same!(_mm256_madd_epi16, _mm256_madd_epi16_stub, bytes);
        same!(_mm256_mulhi_epu16, _mm256_mulhi_epu16_stub, bytes);
        same!(
            _mm256_permutevar8x32_epi32,
            _mm256_permutevar8x32_epi32_stub,
            bytes
        );
        same!(
            _mm256_testz_si256,
            _mm256_testz_si256_stub,
            core::convert::identity
        );
    }

    #[test]
    fn avx2_models_match_hardware() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: no AVX2 on this machine");
            return;
        }
        unsafe { compare_all() };
    }
}

#[cfg(all(test, miri))]
mod miri_avx2_coverage {
    use super::*;
    use crate::simd::testutil::{check_decode, check_encode};
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE};

    /// Encode against the oracle and decode back, buffers sized as the public
    /// API sizes them so MIRI sees the real caller's provenance.
    fn check(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_encode(config, oracle, encode_slice_avx2, len);
        check_decode(config, oracle, decode_slice_avx2, len);
    }

    /// One raw length per distinct code path; the label names the path.
    const TIER_LENGTHS: &[(usize, &str)] = &[
        (0, "empty"),
        (1, "scalar only"),
        (23, "scalar only, longest sub-round"),
        (31, "scalar only, just under the SIMD guard"),
        (32, "encode: first block, no loop"),
        (37, "encode: first block + unaligned scalar tail"),
        (53, "encode: first block + one single-tier round"),
        (96, "decode: quad window not yet reached"),
        (97, "decode: exactly one quad pass"),
        (124, "encode: exactly one quad pass"),
        (192, "both: quad pass then single-tier rounds"),
    ];

    #[test]
    fn miri_avx2_standard() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for &(len, tier) in TIER_LENGTHS {
            println!("standard: len {len} ({tier})");
            check(&config, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx2_url_safe() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        for &(len, tier) in TIER_LENGTHS {
            println!("url-safe: len {len} ({tier})");
            check(&config, &URL_SAFE, len);
        }
    }

    #[test]
    fn miri_avx2_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &(len, tier) in TIER_LENGTHS {
            println!("no-pad: len {len} ({tier})");
            check(&config, &STANDARD_NO_PAD, len);
        }
    }

    /// Invalid bytes must be caught in every tier, including the last lane
    /// of a quad pass, where an early-out would otherwise have already
    /// stored three sub-blocks.
    #[test]
    fn miri_avx2_decode_rejects_invalid() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        let mut dst = [0u8; 256];

        for &(len, bad_at, where_) in &[
            (32, 31, "single tier"),
            (33, 32, "scalar tail"),
            (132, 0, "quad tier, first lane"),
            (132, 127, "quad tier, last lane"),
        ] {
            let mut input = vec![b'A'; len];
            input[bad_at] = b'$';
            let res = unsafe { decode_slice_avx2(&config, &input, &mut dst) };
            assert!(res.is_err(), "missed invalid byte in {where_}");
        }
    }
}

/// Exhaustive regression test for the nibble-lookup `lut_lo`/`lut_hi`/`lut_roll`
/// tables in [`decode_constants_avx2`], guarding against transcription typos.
/// Runs on real AVX2 hardware under plain `cargo test` (no Kani toolchain);
/// the tables were hand-derived (see [`DecodeConstantsAvx2`]).
#[cfg(test)]
#[cfg(not(miri))]
mod avx2_decode_lut_exhaustive {
    use super::*;

    /// For every byte value, decode a 36-byte input of that byte (32 through
    /// the AVX2 fast path, 4 valid filler bytes into the scalar tail) and check
    /// `decode_slice_avx2` agrees with the scalar decoder on validity and value.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn check_all_byte_values(config: &Config) {
        for candidate in 0u8..=255 {
            let mut input = [candidate; 36];
            input[32..].copy_from_slice(b"AAAA");

            let mut avx2_out = [0u8; 64];
            let avx2_result = unsafe { decode_slice_avx2(config, &input, &mut avx2_out) };

            // Oracle: the first 32 bytes via the (separately tested) scalar path.
            let mut scalar_out = [0u8; 64];
            let scalar_result = crate::scalar::decode_slice(config, &input[..32], &mut scalar_out);

            match scalar_result {
                Ok(scalar_len) => {
                    assert!(
                        avx2_result.is_ok(),
                        "byte {candidate:#04x} ({candidate}): scalar accepted it (decoded \
                         {scalar_len} bytes) but avx2 rejected it with {avx2_result:?}"
                    );
                    let avx2_len = avx2_result.expect("checked above");
                    // 32 vectorized bytes + "AAAA" tail -> scalar_len + 3.
                    assert_eq!(
                        avx2_len,
                        scalar_len + 3,
                        "byte {candidate:#04x}: length mismatch"
                    );
                    assert_eq!(
                        &avx2_out[..scalar_len],
                        &scalar_out[..scalar_len],
                        "byte {candidate:#04x} ({candidate}): decoded value mismatch"
                    );
                }
                Err(scalar_err) => {
                    assert_eq!(
                        avx2_result,
                        Err(scalar_err),
                        "byte {candidate:#04x} ({candidate}): avx2/scalar disagree on validity"
                    );
                }
            }
        }
    }

    #[test]
    fn avx2_lut_standard_matches_scalar() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        check_all_byte_values(&config);
    }

    #[test]
    fn avx2_lut_url_safe_matches_scalar() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        check_all_byte_values(&config);
    }
}

/// Exhaustive length-boundary regression for the offset-load `encode_slice_avx2`
/// rewrite: compares against the `base64` oracle at every length 0..=400,
/// densely covering the `rounds = (len - 4) / 24` arithmetic and the 4-round
/// batch boundary, plus a few large lengths.
#[cfg(test)]
#[cfg(not(miri))]
mod avx2_encode_length_sweep {
    use super::*;
    use crate::simd::testutil::check_encode;
    use base64::engine::general_purpose::{STANDARD as REF_STANDARD, URL_SAFE as REF_URL_SAFE};

    #[test]
    fn avx2_encode_standard_all_lengths_0_to_400() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for len in 0..=400 {
            check_encode(&config, &REF_STANDARD, encode_slice_avx2, len);
        }
    }

    #[test]
    fn avx2_encode_url_safe_all_lengths_0_to_400() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        for len in 0..=400 {
            check_encode(&config, &REF_URL_SAFE, encode_slice_avx2, len);
        }
    }

    #[test]
    fn avx2_encode_large_lengths() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for len in [1_000, 10_000, 100_000, 1_000_003] {
            check_encode(&config, &REF_STANDARD, encode_slice_avx2, len);
        }
    }
}
