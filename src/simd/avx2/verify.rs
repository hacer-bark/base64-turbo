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
    // covers all N cheaply. Every stride is imported from the kernel module
    // rather than restated, so a stride that changes there changes these proofs
    // too. See the README's "Safety & Verification".

    /// Largest `len` considered: above `usize::MAX / 4` the unpadded
    /// `encoded_len`'s `len * 4` overflows, so the API can't size a buffer.
    const MAX_LEN: usize = usize::MAX / 4;

    // Encoder model, mirroring `encode_slice_avx2`.
    use super::super::{
        ENC_FIRST_ADVANCE, ENC_LEAD, ENC_ROUND_IN, ENC_ROUND_OUT, ENC_UNROLL, ENC_VEC as ENC_LOAD,
    };

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
        kani::assume(ENC_ROUND_IN * rounds <= len - ENC_LEAD);
        kani::assume(len - ENC_LEAD < ENC_ROUND_IN * (rounds + 1));
        rounds
    }

    /// `(src_off, dst_off)` after `done >= 1` rounds: the first round advances
    /// `src` by only 20, giving the uniform `24 * done - 4`.
    fn enc_state(done: usize) -> (usize, usize) {
        (ENC_ROUND_IN * done - ENC_LEAD, ENC_ROUND_OUT * done)
    }

    /// Isolated so the suite's one non-power-of-two division owns its run.
    #[kani::proof]
    fn check_enc_rounds_model() {
        let len: usize = kani::any();
        kani::assume((ENC_LOAD..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        assert_eq!(rounds, (len - ENC_LEAD) / ENC_ROUND_IN);
        // `remaining = rounds - 1` must not underflow (why the guard is >= 32).
        assert!(rounds >= 1);
    }

    /// Base case: the permuted first round is in bounds.
    #[kani::proof]
    fn check_enc_first_block() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((ENC_LOAD..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let cap = enc_cap(len, padding);

        assert!(ENC_LOAD <= len); // reads [0, 32)
        assert!(ENC_ROUND_OUT <= cap); // writes [0, 32)

        let (src_off, dst_off) = enc_state(1);
        assert_eq!(src_off, ENC_FIRST_ADVANCE);
        assert_eq!(dst_off, ENC_ROUND_OUT);
        assert!(rounds >= 1);
    }

    /// Inductive step for the wide (8x-unrolled) tier, over an arbitrary
    /// iteration.
    #[kani::proof]
    fn check_enc_wide_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((ENC_LOAD..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let cap = enc_cap(len, padding);

        let done: usize = kani::any();
        kani::assume(done >= 1 && done <= rounds);
        let remaining = rounds - done;
        kani::assume(remaining >= ENC_UNROLL); // guard `while remaining >= 8`

        let (src_off, dst_off) = enc_state(done);

        // Widest body accesses: the `i = ENC_UNROLL - 1` load and store.
        let last_src = ENC_ROUND_IN * (ENC_UNROLL - 1);
        let last_dst = ENC_ROUND_OUT * (ENC_UNROLL - 1);
        assert!(
            src_off + last_src + ENC_LOAD <= len,
            "wide load leaves input"
        );
        assert!(
            dst_off + last_dst + ENC_ROUND_OUT <= cap,
            "wide store leaves output"
        );

        // `src += 24*8`, `dst += 32*8`, `remaining -= 8` lands on the next state.
        let done_next = done + ENC_UNROLL;
        assert_eq!(
            (
                src_off + ENC_ROUND_IN * ENC_UNROLL,
                dst_off + ENC_ROUND_OUT * ENC_UNROLL
            ),
            enc_state(done_next)
        );
        assert!(done_next <= rounds);
        assert_eq!(remaining - ENC_UNROLL, rounds - done_next);
    }

    /// Inductive step for the single-round tier.
    #[kani::proof]
    fn check_enc_single_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume((ENC_LOAD..=MAX_LEN).contains(&len));

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
        kani::assume((ENC_LOAD..=MAX_LEN).contains(&len));

        let rounds = any_enc_rounds(len);
        let (src_off, dst_off) = enc_state(rounds);

        let processed = src_off + ENC_LEAD; // repays the first round's deficit
        assert_eq!(processed, ENC_ROUND_IN * rounds);

        // `rounds` caps at `(len - 4) / 24`, so the tail is non-empty (>= 4).
        assert!(processed < len);
        let tail = len - processed;
        assert!(tail >= ENC_LEAD);

        // Prefix + scalar tail is exactly the encoded length (no over/short write).
        assert_eq!(
            dst_off + enc_cap(tail, padding),
            enc_cap(len, padding),
            "prefix + tail must equal encoded length"
        );
    }

    // Decoder model, mirroring `decode_slice_avx2`.
    use super::super::{
        DEC_BLOCK_IN, DEC_BLOCK_IN as DEC_LOAD, DEC_BLOCK_OUT, DEC_LEAD, DEC_PACK_LANE_OFF,
        DEC_UNROLL,
    };

    /// Bytes `pack_and_store!` touches (16 at `dst` + 16 at `dst.add(12)`),
    /// 4 wider than the 24 it advances.
    const DEC_STORE_SPAN: usize = DEC_PACK_LANE_OFF + 16;
    const DEC_WIDE_IN: usize = DEC_BLOCK_IN * DEC_UNROLL; // input bytes per wide-tier iteration
    const DEC_WIDE_OUT: usize = DEC_BLOCK_OUT * DEC_UNROLL; // dst advance per wide-tier iteration

    fn dec_cap(len: usize) -> usize {
        TURBO_STANDARD.estimate_decoded_len(len)
    }

    /// The `aligned_len_128` / `aligned_len_32` loop windows (from the
    /// `saturating_sub(4)` margin that keeps a 32-byte load in bounds).
    fn dec_windows(len: usize) -> (usize, usize) {
        let safe = len.saturating_sub(DEC_LEAD);
        (safe - safe % DEC_WIDE_IN, safe - safe % DEC_BLOCK_IN)
    }

    /// Inductive step for the decoder's wide tier, over an arbitrary iteration.
    #[kani::proof]
    fn check_dec_wide_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (aligned_wide, _) = dec_windows(len);
        let cap = dec_cap(len);

        let i: usize = kani::any();
        kani::assume(i <= MAX_LEN / DEC_WIDE_IN);
        let (src_off, dst_off) = (DEC_WIDE_IN * i, DEC_WIDE_OUT * i);
        kani::assume(src_off < aligned_wide); // guard `src < src_end_wide`

        // Widest: the `i = DEC_UNROLL - 1` load and `pack_and_store!`.
        let last_src = DEC_BLOCK_IN * (DEC_UNROLL - 1);
        let last_dst = DEC_BLOCK_OUT * (DEC_UNROLL - 1);
        assert!(
            src_off + last_src + DEC_LOAD <= len,
            "wide load leaves input"
        );
        assert!(
            dst_off + last_dst + DEC_STORE_SPAN <= cap,
            "wide store leaves output"
        );

        // Update `src.add(256)`, `dst.add(192)`.
        assert_eq!(
            (src_off + DEC_WIDE_IN, dst_off + DEC_WIDE_OUT),
            (DEC_WIDE_IN * (i + 1), DEC_WIDE_OUT * (i + 1))
        );
    }

    /// Inductive step for the decoder's single-vector tier, entered from
    /// wherever the wide tier stopped.
    #[kani::proof]
    fn check_dec_single_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (aligned_wide, aligned_block) = dec_windows(len);
        let cap = dec_cap(len);
        let wides = aligned_wide / DEC_WIDE_IN;

        let j: usize = kani::any();
        kani::assume(j <= MAX_LEN / DEC_BLOCK_IN);
        let src_off = aligned_wide + DEC_BLOCK_IN * j;
        let dst_off = DEC_WIDE_OUT * wides + DEC_BLOCK_OUT * j;
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
                aligned_wide + DEC_BLOCK_IN * (j + 1),
                DEC_WIDE_OUT * wides + DEC_BLOCK_OUT * (j + 1)
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

    /// The permuted first round, *one steady-state round*, and a 13-byte scalar
    /// tail. The second round is the point: at 37 bytes `rounds` is 1, so
    /// `encode_rounds_avx2` never ran and nothing executed the steady-state
    /// `src[24n-4 .. 24n+20]` window — a wrong per-round stride would have
    /// passed. Layer 1 asserts that arithmetic; this makes a round live with it.
    const ENC_KERNEL_LEN: usize = 61;
    /// One single-vector pass + a 4-character scalar tail.
    const DEC_KERNEL_LEN: usize = 36;

    // Guard: a length below its tier's threshold would prove nothing (a past
    // revision silently verified only the scalar fallback). Fail the build.
    const _: () = assert!(
        ENC_KERNEL_LEN >= ENC_LOAD
            && (ENC_KERNEL_LEN - ENC_LEAD) / ENC_ROUND_IN >= 2
            && ENC_KERNEL_LEN % ENC_ROUND_IN != 0,
        "ENC_KERNEL_LEN must run a first round plus a steady-state round, and \
         leave an unaligned tail"
    );
    const _: () = assert!(
        (DEC_KERNEL_LEN - DEC_LEAD) / DEC_BLOCK_IN == 1,
        "DEC_KERNEL_LEN must run one single-vector decode pass"
    );
    // A length that is not a multiple of 4 can only ever decode to `Err` under a
    // padded config, which would make the `Ok` half of the equivalence proof
    // below vacuous.
    const _: () = assert!(
        DEC_KERNEL_LEN % 4 == 0,
        "DEC_KERNEL_LEN must be able to decode successfully"
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
    #[kani::stub(_mm256_mullo_epi16, m::_mm256_mullo_epi16_stub)]
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
    #[kani::stub(_mm256_mullo_epi16, m::_mm256_mullo_epi16_stub)]
    #[kani::stub(_mm256_permutevar8x32_epi32, m::_mm256_permutevar8x32_epi32_stub)]
    fn check_avx2_roundtrip_url_safe() {
        roundtrip_kernel(true);
    }

    /// The vectorized decoder agrees with the scalar one on every 36-character
    /// input, which is strictly stronger than the panic-freedom this harness
    /// used to prove: it pins *rejection* as well, over all 256 values in every
    /// lane at once. `crate::scalar` is `#![forbid(unsafe_code)]` and separately
    /// tested, so it is the natural oracle.
    ///
    /// Error *kinds* are deliberately not compared. The two decoders reach a bad
    /// input at different points — the vector path ORs every lane's verdict into
    /// one accumulator and reports it before the scalar tail ever runs, so an
    /// input that is both mis-sized and mis-charactered can legitimately be
    /// `InvalidLength` for one and `InvalidCharacter` for the other. Rejecting
    /// it at all is the contract.
    #[kani::proof]
    #[kani::stub(_mm256_shuffle_epi8, m::_mm256_shuffle_epi8_stub)]
    #[kani::stub(_mm256_subs_epu8, m::_mm256_subs_epu8_stub)]
    #[kani::stub(_mm256_testz_si256, m::_mm256_testz_si256_stub)]
    #[kani::stub(_mm256_maddubs_epi16, m::_mm256_maddubs_epi16_stub)]
    #[kani::stub(_mm256_madd_epi16, m::_mm256_madd_epi16_stub)]
    fn check_avx2_decode_matches_scalar() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; DEC_KERNEL_LEN] = kani::any();

        // Both sized as the public API sizes them, so a real overrun still fails.
        let mut simd_out = [0u8; DEC_KERNEL_CAP];
        let mut scalar_out = [0u8; DEC_KERNEL_CAP];

        let simd = unsafe { decode_slice_avx2(&config, &input, &mut simd_out) };
        let scalar = crate::scalar::decode_slice(&config, &input, &mut scalar_out);

        match scalar {
            Ok(n) => {
                assert_eq!(simd, Ok(n), "scalar accepted an input the kernel rejected");
                assert_eq!(
                    &simd_out[..n],
                    &scalar_out[..n],
                    "kernel and scalar decoded to different bytes"
                );
            }
            Err(_) => assert!(simd.is_err(), "kernel accepted an input scalar rejected"),
        }
    }
}

/// Rust models of every AVX2 intrinsic the kernels use, for the Kani proofs.
///
/// Each is a transcription of the `<operation>` pseudocode published in the Intel
/// Intrinsics Guide (data version 3.6.9), quoted line for line in the comments
/// with the Rust statement it became directly beneath it. Nothing is
/// paraphrased, condensed or "improved": if Intel writes a branch where a
/// ternary would do, so does the model, because the comments are the
/// specification these proofs are checked against and a reader has to be able to
/// diff them against the guide symbol by symbol.
///
/// One systematic departure, and only one. Intel addresses vectors by **bit**
/// offset — `i := j*8`, then `dst[i+7:i]` for the byte at that offset. Rust
/// indexes bytes, so every transcription keeps Intel's bit-offset variables
/// verbatim and divides by 8 at the point of access (`dst[i / 8]`). Where Intel
/// addresses an individual bit, [`bit`] does it. Lines carrying no Intel text
/// are marked `NOTE:`.
#[cfg(any(kani, test))]
// Every consumer of this module is invisible to rustc's dead-code pass: the
// Kani proofs reach the models through `#[kani::stub(...)]` attribute
// arguments, and a Miri build reaches the real instructions instead of these.
// So which models look "used" depends on which harness is being compiled, and
// the answer is never the whole set.
#[allow(dead_code)]
#[allow(non_snake_case)]
// These are all "the transcription is more literal than idiomatic Rust would
// be" lints: Intel writes a late-initialized `IF/ELSE/FI` where Rust would use
// an `if` expression, and a plain `RETURN` where Rust would use a tail
// expression. Following the pseudocode is the point, so they are turned off
// rather than the transcriptions being reshaped to satisfy them.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_const_for_fn,
    clippy::missing_transmute_annotations,
    clippy::needless_late_init,
    clippy::needless_range_loop,
    clippy::needless_return,
    clippy::used_underscore_items
)]
pub(super) mod intrinsic_models {
    use super::*;
    use std::mem::transmute;

    // NOTE: scaffolding, not from Intel. Reads bit `n` of a little-endian byte
    // vector, for the places the pseudocode indexes a single bit.
    fn bit(v: &[u8; 32], n: usize) -> u8 {
        (v[n / 8] >> (n % 8)) & 1
    }

    // NOTE: scaffolding, not from Intel. The `SaturateU8` and `Saturate16`
    // helpers the pseudocode calls by name.
    fn SaturateU8(x: i16) -> u8 {
        x.clamp(0, 255) as u8
    }
    fn Saturate16(x: i32) -> i16 {
        x.clamp(-32768, 32767) as i16
    }

    // STUB: _mm256_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_epi8
    pub(super) unsafe fn _mm256_shuffle_epi8_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [u8; 32] = unsafe { transmute(b) };
        let mut dst = [0u8; 32];

        // FOR j := 0 to 15
        for j in 0..16 {
            // 	i := j*8
            let i = j * 8;
            // 	IF b[i+7] == 1
            if bit(&b, i + 7) == 1 {
                // 		dst[i+7:i] := 0
                dst[i / 8] = 0;
            // 	ELSE
            } else {
                // 		index[3:0] := b[i+3:i]
                let index = usize::from(b[i / 8] & 0x0F);
                // 		dst[i+7:i] := a[index*8+7:index*8]
                dst[i / 8] = a[(index * 8) / 8];
            }
            // 	FI
            // 	IF b[128+i+7] == 1
            if bit(&b, 128 + i + 7) == 1 {
                // 		dst[128+i+7:128+i] := 0
                dst[(128 + i) / 8] = 0;
            // 	ELSE
            } else {
                // 		index[3:0] := b[128+i+3:128+i]
                let index = usize::from(b[(128 + i) / 8] & 0x0F);
                // 		dst[128+i+7:128+i] := a[128+index*8+7:128+index*8]
                dst[(128 + i) / 8] = a[(128 + index * 8) / 8];
            }
            // 	FI
        }
        // ENDFOR
        // dst[MAX:256] := 0
        // NOTE: `__m256i` is exactly 256 bits; there is nothing above to zero.

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
            // 	i := j*8
            let i = j * 8;
            // 	dst[i+7:i] := SaturateU8(a[i+7:i] - b[i+7:i])
            dst[i / 8] = SaturateU8(i16::from(a[i / 8]) - i16::from(b[i / 8]));
        }
        // ENDFOR
        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_testz_si256
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_testz_si256
    pub(super) unsafe fn _mm256_testz_si256_stub(a: __m256i, b: __m256i) -> i32 {
        // NOTE: Rust has no 256-bit integer, so `a[255:0]` and `b[255:0]` are
        // held as four u64 limbs and every whole-vector operation below is
        // applied limb-wise.
        let a: [u64; 4] = unsafe { transmute(a) };
        let b: [u64; 4] = unsafe { transmute(b) };
        let zf: i32;
        let _cf: i32;

        let a_and_b = [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]];
        // IF ((a[255:0] AND b[255:0]) == 0)
        if a_and_b == [0, 0, 0, 0] {
            // 	ZF := 1
            zf = 1;
        // ELSE
        } else {
            // 	ZF := 0
            zf = 0;
        }
        // FI

        let not_a_and_b = [
            (!a[0]) & b[0],
            (!a[1]) & b[1],
            (!a[2]) & b[2],
            (!a[3]) & b[3],
        ];
        // IF (((NOT a[255:0]) AND b[255:0]) == 0)
        if not_a_and_b == [0, 0, 0, 0] {
            // 	CF := 1
            _cf = 1;
        // ELSE
        } else {
            // 	CF := 0
            _cf = 0;
        }
        // FI

        // RETURN ZF
        return zf;
    }

    // STUB: _mm256_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maddubs_epi16
    pub(super) unsafe fn _mm256_maddubs_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        // NOTE: `a` holds unsigned bytes, `b` signed ones.
        let a: [u8; 32] = unsafe { transmute(a) };
        let b: [i8; 32] = unsafe { transmute(b) };
        let mut dst = [0i16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // 	i := j*16
            let i = j * 16;
            // 	dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[i / 16] = Saturate16(
                i32::from(a[(i + 8) / 8]) * i32::from(b[(i + 8) / 8])
                    + i32::from(a[i / 8]) * i32::from(b[i / 8]),
            );
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
            // 	i := j*32
            let i = j * 32;
            // 	dst[i+31:i] := SignExtend32(a[i+31:i+16]*b[i+31:i+16]) + SignExtend32(a[i+15:i]*b[i+15:i])
            // NOTE: an i16*i16 product *is* its own sign-extended 32-bit value;
            // the sum is `wrapping` because it lands in a 32-bit destination,
            // which the two extreme products can overflow.
            dst[i / 32] = (i32::from(a[(i + 16) / 16]) * i32::from(b[(i + 16) / 16]))
                .wrapping_add(i32::from(a[i / 16]) * i32::from(b[i / 16]));
        }
        // ENDFOR
        // dst[MAX:256] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm256_mullo_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mullo_epi16
    pub(super) unsafe fn _mm256_mullo_epi16_stub(a: __m256i, b: __m256i) -> __m256i {
        let a: [i16; 16] = unsafe { transmute(a) };
        let b: [i16; 16] = unsafe { transmute(b) };
        let mut dst = [0i16; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // 	i := j*16
            let i = j * 16;
            // 	tmp[31:0] := SignExtend32(a[i+15:i]) * SignExtend32(b[i+15:i])
            let tmp: i32 = i32::from(a[i / 16]) * i32::from(b[i / 16]);
            // 	dst[i+15:i] := tmp[15:0]
            dst[i / 16] = tmp as i16;
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
            // 	i := j*32
            let i = j * 32;
            // 	id := idx[i+2:i]*32
            let id = ((idx[i / 32] & 0x7) * 32) as usize;
            // 	dst[i+31:i] := a[id+31:id]
            dst[i / 32] = a[id / 32];
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
        same!(_mm256_mullo_epi16, _mm256_mullo_epi16_stub, bytes);
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
        (124, "encode: single-tier rounds only"),
        (192, "decode: single-tier passes only"),
        (244, "encode: one wide pass, then a single-tier round"),
        (260, "decode: wide window not yet reached"),
        (292, "decode: one wide pass, then a single-tier pass"),
        (700, "both: several wide passes plus single-tier rounds"),
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

    /// Invalid bytes must be caught in every tier, wherever they sit in a
    /// pass. The kernel folds all validation into one accumulator checked after
    /// the loops, so a byte in the very last lane must still fail the call.
    #[test]
    fn miri_avx2_decode_rejects_invalid() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        let mut dst = [0u8; 512];

        for &(len, bad_at, where_) in &[
            (36, 31, "single tier"),
            (37, 36, "scalar tail"),
            (260, 0, "wide tier, first lane"),
            (260, 255, "wide tier, last lane"),
            (292, 288, "wide tier then single tier, last lane"),
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

/// Covers the encoder's non-temporal store path, which needs an input at least
/// [`NT_STORE_MIN_LEN`] long and so is out of reach for Miri (and for the length
/// sweep below). The hazard it guards is `_mm_stream_si128`'s 16-byte alignment
/// requirement, so the destinations here straddle the alignment gate: only the
/// 0- and 16-shifted ones can take the path, and all must agree with the oracle.
#[cfg(test)]
#[cfg(not(miri))]
mod avx2_encode_non_temporal {
    use super::*;
    use base64::Engine as _;
    use base64::engine::general_purpose::{STANDARD as REF_STANDARD, URL_SAFE as REF_URL_SAFE};

    #[test]
    fn avx2_encode_above_non_temporal_threshold() {
        // One length exactly at the threshold and one comfortably past it with a
        // remainder that leaves both a wide-tier and a single-tier round plus a
        // scalar tail.
        for len in [NT_STORE_MIN_LEN, NT_STORE_MIN_LEN + 4099] {
            let input = crate::simd::testutil::bytes(len);

            for (config, oracle) in [
                (
                    Config {
                        url_safe: false,
                        padding: true,
                    },
                    &REF_STANDARD,
                ),
                (
                    Config {
                        url_safe: true,
                        padding: true,
                    },
                    &REF_URL_SAFE,
                ),
            ] {
                let expected = oracle.encode(&input);
                for shift in [0usize, 1, 8, 16] {
                    let mut dst = vec![0u8; expected.len() + shift];
                    unsafe { encode_slice_avx2(&config, &input, &mut dst[shift..]) };
                    assert_eq!(
                        core::str::from_utf8(&dst[shift..]).unwrap(),
                        expected,
                        "len {len}, url_safe {}, dst shift {shift}",
                        config.url_safe
                    );
                }
            }
        }
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
