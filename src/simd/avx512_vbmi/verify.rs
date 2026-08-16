//! AVX-512-VBMI verification: Kani proofs, Intel-pseudocode intrinsic models,
//! and the Miri + hardware coverage suites. Split out of the production module
//! purely to keep it lean.

use super::*;

#[cfg(kani)]
mod kani_verification_avx512_vbmi {
    use super::*;
    use crate::{Config, STANDARD as TURBO_STANDARD, STANDARD_NO_PAD as TURBO_STANDARD_NO_PAD};

    // Only used inside `#[kani::stub(...)]` paths, which don't count as a use.
    #[allow(unused_imports)]
    use super::intrinsic_models as m;

    // Layer 1 — index proofs: reason over a symbolic `len` and an arbitrary
    // iteration (no vectors), giving an induction (step/exit) that covers all N
    // cheaply. Every stride is imported from the kernel module rather than
    // restated, so a stride that changes there changes these proofs too. See
    // the README's "Safety & Verification".
    //
    // The VBMI kernels make this cleaner than the AVX2 ones: there is no
    // permuted first round to repay, and every tier consumes a whole number of
    // Base64 groups. So instead of tracking bytes, both models track *groups*
    // consumed — which sidesteps the division-by-3 and division-by-4 that would
    // otherwise sit in every proof, and makes the loop invariant definitional:
    // after `g` groups the encoder has written `4 * g` characters and the
    // decoder `3 * g` bytes, and each step proof shows the tier's stride lands
    // exactly on the next multiple.

    use super::super::{
        DEC_GROUP, DEC_LEAD, DEC_MASKED_MIN, DEC_QUAD_IN, DEC_QUAD_MIN, DEC_QUAD_OUT,
        DEC_SINGLE_MIN, DEC_VEC_IN, DEC_VEC_OUT, ENC_GROUP, ENC_QUAD_IN, ENC_QUAD_MIN,
        ENC_QUAD_OUT, ENC_SINGLE_MIN, ENC_VEC, ENC_VEC_IN, ENC_VEC_OUT,
    };

    /// Largest `len` considered: above `usize::MAX / 4` the unpadded
    /// `encoded_len`'s `len * 4` overflows, so the API can't size a buffer.
    const MAX_LEN: usize = usize::MAX / 4;

    /// A full-width store writes exactly as many bytes as a full-width load
    /// reads; the decoder's quad tier is the only place the distinction shows,
    /// where three of its four stores are unmasked and so overhang their 48.
    const DEC_STORE_WIDE: usize = DEC_VEC_IN;

    /// Groups consumed per iteration of each tier.
    const ENC_QUAD_GROUPS: usize = ENC_QUAD_IN / ENC_GROUP;
    const ENC_SINGLE_GROUPS: usize = ENC_VEC_IN / ENC_GROUP;
    const DEC_QUAD_GROUPS: usize = DEC_QUAD_IN / DEC_GROUP;
    const DEC_SINGLE_GROUPS: usize = DEC_VEC_IN / DEC_GROUP;

    // The group model above is only faithful if every tier really does consume
    // whole groups at the documented ratio. Fail the build if a stride is
    // edited into something that doesn't.
    const _: () = assert!(
        ENC_QUAD_IN % ENC_GROUP == 0 && ENC_VEC_IN % ENC_GROUP == 0,
        "every encode tier must consume whole 3-byte groups"
    );
    const _: () = assert!(
        ENC_QUAD_GROUPS * 4 == ENC_QUAD_OUT && ENC_SINGLE_GROUPS * 4 == ENC_VEC_OUT,
        "every encode tier must emit 4 characters per group"
    );
    const _: () = assert!(
        DEC_QUAD_IN % DEC_GROUP == 0 && DEC_VEC_IN % DEC_GROUP == 0,
        "every decode tier must consume whole 4-character groups"
    );
    const _: () = assert!(
        DEC_QUAD_GROUPS * 3 == DEC_QUAD_OUT && DEC_SINGLE_GROUPS * 3 == DEC_VEC_OUT,
        "every decode tier must emit 3 bytes per group"
    );

    fn enc_cap(len: usize, padding: bool) -> usize {
        if padding {
            TURBO_STANDARD.encoded_len(len)
        } else {
            TURBO_STANDARD_NO_PAD.encoded_len(len)
        }
    }

    fn dec_cap(len: usize) -> usize {
        TURBO_STANDARD.estimate_decoded_len(len)
    }

    /// An arbitrary reachable encoder state: `g` whole groups consumed, with
    /// `rem` input bytes still to go. Returns `(done, dst_off, rem)`.
    fn any_enc_state(len: usize) -> (usize, usize, usize) {
        let g: usize = kani::any();
        kani::assume(g <= MAX_LEN / ENC_GROUP);
        let done = ENC_GROUP * g;
        kani::assume(done <= len);
        (done, 4 * g, len - done)
    }

    /// As [`any_enc_state`], for the decoder: 4 characters in, 3 bytes out.
    fn any_dec_state(len: usize) -> (usize, usize, usize) {
        let g: usize = kani::any();
        kani::assume(g <= MAX_LEN / DEC_GROUP);
        let done = DEC_GROUP * g;
        kani::assume(done <= len);
        (done, 3 * g, len - done)
    }

    // --- Encoder ---

    /// Inductive step for the encoder's quad tier.
    #[kani::proof]
    fn check_vbmi_enc_quad_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume(len <= MAX_LEN);

        let (done, dst_off, rem) = any_enc_state(len);
        kani::assume(rem >= ENC_QUAD_MIN); // guard `while rem >= 256`
        let cap = enc_cap(len, padding);

        // Widest accesses: the fourth load starts 3 vectors in and reads a full
        // 64, and the fourth store writes a full 64 at 3 vectors out.
        assert!(
            done + 3 * ENC_VEC_IN + ENC_VEC <= len,
            "quad load leaves input"
        );
        assert!(
            dst_off + 3 * ENC_VEC_OUT + ENC_VEC_OUT <= cap,
            "quad store leaves output"
        );

        // `src += 192`, `dst += 256`, `rem -= 192` lands on the next state.
        assert_eq!(
            dst_off + ENC_QUAD_OUT,
            4 * (done / ENC_GROUP + ENC_QUAD_GROUPS)
        );
        assert!(done + ENC_QUAD_IN <= len);
        assert_eq!(rem - ENC_QUAD_IN, len - (done + ENC_QUAD_IN));
    }

    /// Inductive step for the encoder's single tier.
    #[kani::proof]
    fn check_vbmi_enc_single_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume(len <= MAX_LEN);

        let (done, dst_off, rem) = any_enc_state(len);
        kani::assume(rem >= ENC_SINGLE_MIN); // guard `while rem >= 64`
        let cap = enc_cap(len, padding);

        // A plain load reads a whole vector to consume 48 of it.
        assert!(done + ENC_VEC <= len, "single load leaves input");
        assert!(dst_off + ENC_VEC_OUT <= cap, "single store leaves output");

        assert_eq!(
            dst_off + ENC_VEC_OUT,
            4 * (done / ENC_GROUP + ENC_SINGLE_GROUPS)
        );
        assert!(done + ENC_VEC_IN <= len);
        assert_eq!(rem - ENC_VEC_IN, len - (done + ENC_VEC_IN));
    }

    /// Inductive step for the encoder's masked tier. The masked load and store
    /// touch only their masked lanes, so the obligation is `take` bytes in and
    /// `out` bytes out — not a full vector of either.
    #[kani::proof]
    fn check_vbmi_enc_masked_step() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume(len <= MAX_LEN);

        let (done, dst_off, rem) = any_enc_state(len);
        // Guard `while rem >= 3`, entered only once the single tier has stopped.
        kani::assume(rem >= ENC_GROUP && rem < ENC_SINGLE_MIN);
        let cap = enc_cap(len, padding);

        let take = (rem - rem % ENC_GROUP).min(ENC_VEC_IN);
        let out = take / ENC_GROUP * 4;

        // `u64::MAX >> (64 - take)` is only defined, and only non-empty, in this
        // range; likewise the store's `64 - out`.
        assert!(
            (ENC_GROUP..=ENC_VEC_IN).contains(&take),
            "load mask shift out of range"
        );
        assert!(
            (4..=ENC_VEC_OUT).contains(&out),
            "store mask shift out of range"
        );

        assert!(done + take <= len, "masked load leaves input");
        assert!(dst_off + out <= cap, "masked store leaves output");

        // Whole groups only, so the invariant survives the step.
        assert_eq!(take % ENC_GROUP, 0);
        assert_eq!(dst_off + out, 4 * ((done + take) / ENC_GROUP));
    }

    /// The masked tier's loop terminates, in at most the two iterations its
    /// comment claims: `take` is always a whole group (so progress is real),
    /// and two rounds of it always land below the guard.
    #[kani::proof]
    fn check_vbmi_enc_masked_terminates() {
        let rem: usize = kani::any();
        kani::assume(rem >= ENC_GROUP && rem < ENC_SINGLE_MIN);

        let take1 = (rem - rem % ENC_GROUP).min(ENC_VEC_IN);
        assert!(take1 >= ENC_GROUP, "first pass makes no progress");
        let rem1 = rem - take1;

        if rem1 >= ENC_GROUP {
            let take2 = (rem1 - rem1 % ENC_GROUP).min(ENC_VEC_IN);
            assert!(take2 >= ENC_GROUP, "second pass makes no progress");
            assert!(rem1 - take2 < ENC_GROUP, "a third pass would be needed");
        }
    }

    /// Exit case: the scalar encoder sees at most a final partial group, and
    /// the vector prefix plus that group is exactly the encoded length.
    #[kani::proof]
    fn check_vbmi_enc_tail_handoff() {
        let len: usize = kani::any();
        let padding: bool = kani::any();
        kani::assume(len <= MAX_LEN);

        let (done, dst_off, rem) = any_enc_state(len);
        kani::assume(rem < ENC_GROUP); // every loop has exited

        // Prefix + scalar tail is exactly the encoded length (no over/short
        // write), which is also what makes padding purely the scalar kernel's
        // business: the vector tiers never emit a group that could carry '='.
        assert_eq!(
            dst_off + enc_cap(rem, padding),
            enc_cap(len, padding),
            "prefix + tail must equal encoded length"
        );
        assert_eq!(done + rem, len);
    }

    // --- Decoder ---

    /// Inductive step for the decoder's quad tier.
    #[kani::proof]
    fn check_vbmi_dec_quad_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (done, dst_off, rem) = any_dec_state(len);
        kani::assume(rem >= DEC_QUAD_MIN); // guard `while rem >= 260`
        let cap = dec_cap(len);

        assert!(
            done + 3 * DEC_VEC_IN + DEC_VEC_IN <= len,
            "quad load leaves input"
        );

        // The first three stores are unmasked, so each overhangs its 48 bytes
        // by 16; the third reaches furthest of them. Only the fourth is masked,
        // and it closes the iteration exactly at 192.
        assert!(
            dst_off + 2 * DEC_VEC_OUT + DEC_STORE_WIDE <= cap,
            "quad unmasked store overhang leaves output"
        );
        assert!(
            dst_off + DEC_QUAD_OUT <= cap,
            "quad masked store leaves output"
        );

        assert_eq!(
            dst_off + DEC_QUAD_OUT,
            3 * (done / DEC_GROUP + DEC_QUAD_GROUPS)
        );
        assert!(done + DEC_QUAD_IN <= len);
        assert_eq!(rem - DEC_QUAD_IN, len - (done + DEC_QUAD_IN));
    }

    /// Inductive step for the decoder's single tier.
    #[kani::proof]
    fn check_vbmi_dec_single_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (done, dst_off, rem) = any_dec_state(len);
        kani::assume(rem >= DEC_SINGLE_MIN); // guard `while rem >= 68`
        let cap = dec_cap(len);

        assert!(done + DEC_VEC_IN <= len, "single load leaves input");
        assert!(dst_off + DEC_VEC_OUT <= cap, "single store leaves output");

        assert_eq!(
            dst_off + DEC_VEC_OUT,
            3 * (done / DEC_GROUP + DEC_SINGLE_GROUPS)
        );
        assert!(done + DEC_VEC_IN <= len);
        assert_eq!(rem - DEC_VEC_IN, len - (done + DEC_VEC_IN));
    }

    /// The decoder's masked tier: mask shifts in range, accesses in bounds, and
    /// the invariant preserved.
    #[kani::proof]
    fn check_vbmi_dec_masked_step() {
        let len: usize = kani::any();
        kani::assume(len <= MAX_LEN);

        let (done, dst_off, rem) = any_dec_state(len);
        // Guard `if rem >= 8`, entered only once the single tier has stopped.
        kani::assume(rem >= DEC_MASKED_MIN && rem < DEC_SINGLE_MIN);
        let cap = dec_cap(len);

        let take = (rem - DEC_LEAD) & !(DEC_GROUP - 1);
        let out = take / DEC_GROUP * 3;

        assert!(
            (DEC_GROUP..=DEC_VEC_IN).contains(&take),
            "load mask shift out of range"
        );
        assert!(
            (3..=DEC_VEC_OUT).contains(&out),
            "store mask shift out of range"
        );

        assert!(done + take <= len, "masked load leaves input");
        assert!(dst_off + out <= cap, "masked store leaves output");

        assert_eq!(take % DEC_GROUP, 0);
        assert_eq!(dst_off + out, 3 * ((done + take) / DEC_GROUP));
    }

    /// Exit case, and the property the whole decoder design rests on: **every
    /// tier that runs leaves at least a full group unconsumed**. The vector
    /// tiers have no padding logic at all, so if one of them could swallow the
    /// final group it would reject every legally padded input; the scalar tail
    /// has to be the one that sees it. Each tier's guard is written as "what it
    /// consumes, plus [`DEC_LEAD`]", and this is where that pays off.
    #[kani::proof]
    fn check_vbmi_dec_tail_slack() {
        let rem: usize = kani::any();
        kani::assume(rem <= MAX_LEN);

        if rem >= DEC_QUAD_MIN {
            assert!(
                rem - DEC_QUAD_IN >= DEC_LEAD,
                "quad tier ate the last group"
            );
        }
        if rem >= DEC_SINGLE_MIN {
            assert!(
                rem - DEC_VEC_IN >= DEC_LEAD,
                "single tier ate the last group"
            );
        }
        if rem >= DEC_MASKED_MIN {
            let take = (rem - DEC_LEAD) & !(DEC_GROUP - 1);
            let left = rem - take;
            assert!(take >= DEC_GROUP, "masked tier makes no progress");
            assert!(left >= DEC_LEAD, "masked tier ate the last group");
            // ...and it cannot run twice, which is why the kernel writes it as
            // an `if` rather than a `while`.
            assert!(left < DEC_MASKED_MIN, "masked tier would run again");
        }
    }

    // Layer 2 — kernel proofs: run the real code over symbolic bytes (the
    // gather/multishift bit extraction, the alphabet permute, the reverse LUT,
    // the validity accumulator). Layer 1 owns the loop arithmetic, so each of
    // these needs only enough length to reach its tiers once. Buffers are the
    // exact public-API capacities, so any real overrun fails.
    //
    // The quad tiers are deliberately absent: 256 symbolic characters through
    // four `vpermi2b` lookups is out of CBMC's reach, and a quad iteration is
    // four copies of the vector these harnesses do cover, with the offsets that
    // Layer 1 proves. That gap is stated in the README rather than papered over.

    /// One single-tier vector (48 bytes), one masked vector (15 bytes), and a
    /// 1-byte scalar tail — every encode tier below the quad, in one harness.
    const ENC_KERNEL_LEN: usize = 64;
    /// One single-tier vector (64 characters) and a 4-character scalar tail.
    const DEC_KERNEL_LEN: usize = 68;
    /// One masked vector (8 characters) and a 4-character scalar tail.
    const DEC_MASKED_KERNEL_LEN: usize = 12;
    /// One full masked encode vector, whose 64 characters then decode through
    /// one masked decode vector plus a scalar group.
    const ROUNDTRIP_LEN: usize = 48;

    // Guard: a length below its tier's threshold would prove nothing but the
    // scalar fallback. Fail the build rather than quietly verify less.
    const _: () = assert!(
        ENC_KERNEL_LEN >= ENC_SINGLE_MIN
            && ENC_KERNEL_LEN < ENC_QUAD_MIN
            && ENC_KERNEL_LEN - ENC_VEC_IN >= ENC_GROUP
            && ENC_KERNEL_LEN % ENC_GROUP != 0,
        "ENC_KERNEL_LEN must run a single-tier vector, then a masked vector, \
         then leave a partial group for scalar"
    );
    const _: () = assert!(
        DEC_KERNEL_LEN >= DEC_SINGLE_MIN
            && DEC_KERNEL_LEN < DEC_QUAD_MIN
            && DEC_KERNEL_LEN % DEC_GROUP == 0,
        "DEC_KERNEL_LEN must run a single-tier vector and still be decodable"
    );
    const _: () = assert!(
        DEC_MASKED_KERNEL_LEN >= DEC_MASKED_MIN
            && DEC_MASKED_KERNEL_LEN < DEC_SINGLE_MIN
            && DEC_MASKED_KERNEL_LEN % DEC_GROUP == 0,
        "DEC_MASKED_KERNEL_LEN must run the masked tier and still be decodable"
    );
    const _: () = assert!(
        ROUNDTRIP_LEN < ENC_SINGLE_MIN && ROUNDTRIP_LEN % ENC_GROUP == 0,
        "ROUNDTRIP_LEN must be a whole number of groups in the masked tier"
    );

    const ENC_KERNEL_CAP: usize = TURBO_STANDARD.encoded_len(ENC_KERNEL_LEN);
    const DEC_KERNEL_CAP: usize = TURBO_STANDARD.estimate_decoded_len(DEC_KERNEL_LEN);
    const DEC_MASKED_KERNEL_CAP: usize = TURBO_STANDARD.estimate_decoded_len(DEC_MASKED_KERNEL_LEN);
    const ROUNDTRIP_ENC_CAP: usize = TURBO_STANDARD.encoded_len(ROUNDTRIP_LEN);
    const ROUNDTRIP_DEC_CAP: usize = TURBO_STANDARD.estimate_decoded_len(ROUNDTRIP_ENC_CAP);

    /// The vectorized encoder agrees with the scalar one on every input of this
    /// length. `crate::scalar` is `#![forbid(unsafe_code)]` and separately
    /// tested, so it is the natural oracle — and a stronger one than a
    /// round-trip, which cannot see an encode bug that the decoder inverts.
    fn encode_matches_scalar(url_safe: bool) {
        let config = Config {
            url_safe,
            padding: true,
        };
        let input: [u8; ENC_KERNEL_LEN] = kani::any();

        let mut vbmi_out = [0u8; ENC_KERNEL_CAP];
        let mut scalar_out = [0u8; ENC_KERNEL_CAP];

        unsafe { encode_slice_avx512_vbmi(&config, &input, &mut vbmi_out) };
        crate::scalar::encode_slice(&config, &input, &mut scalar_out);

        assert_eq!(
            vbmi_out, scalar_out,
            "kernel and scalar encoded differently"
        );
    }

    #[kani::proof]
    #[kani::stub(_mm512_permutexvar_epi8, m::permutexvar_epi8_model)]
    #[kani::stub(_mm512_permutex2var_epi8, m::permutex2var_epi8_model)]
    #[kani::stub(_mm512_multishift_epi64_epi8, m::multishift_epi64_epi8_model)]
    #[kani::stub(_mm512_maddubs_epi16, m::maddubs_epi16_model)]
    #[kani::stub(_mm512_madd_epi16, m::madd_epi16_model)]
    #[kani::stub(_mm512_ternarylogic_epi32, m::ternarylogic_epi32_model)]
    #[kani::stub(_mm512_movepi8_mask, m::movepi8_mask_model)]
    #[kani::stub(_mm512_mask_loadu_epi8, m::mask_loadu_epi8_model)]
    #[kani::stub(_mm512_maskz_loadu_epi8, m::maskz_loadu_epi8_model)]
    #[kani::stub(_mm512_mask_storeu_epi8, m::mask_storeu_epi8_model)]
    fn check_vbmi_encode_matches_scalar_standard() {
        encode_matches_scalar(false);
    }

    #[kani::proof]
    #[kani::stub(_mm512_permutexvar_epi8, m::permutexvar_epi8_model)]
    #[kani::stub(_mm512_permutex2var_epi8, m::permutex2var_epi8_model)]
    #[kani::stub(_mm512_multishift_epi64_epi8, m::multishift_epi64_epi8_model)]
    #[kani::stub(_mm512_maddubs_epi16, m::maddubs_epi16_model)]
    #[kani::stub(_mm512_madd_epi16, m::madd_epi16_model)]
    #[kani::stub(_mm512_ternarylogic_epi32, m::ternarylogic_epi32_model)]
    #[kani::stub(_mm512_movepi8_mask, m::movepi8_mask_model)]
    #[kani::stub(_mm512_mask_loadu_epi8, m::mask_loadu_epi8_model)]
    #[kani::stub(_mm512_maskz_loadu_epi8, m::maskz_loadu_epi8_model)]
    #[kani::stub(_mm512_mask_storeu_epi8, m::mask_storeu_epi8_model)]
    fn check_vbmi_encode_matches_scalar_url_safe() {
        encode_matches_scalar(true);
    }

    /// The vectorized decoder agrees with the scalar one over every input of
    /// length `N`, which pins *rejection* as well as value — including VBMI's
    /// second rejection route, where a byte >= 0x80 aliases into the 128-entry
    /// table via bit 6 and has to be caught by the accumulator's other half.
    /// Symbolic bytes cover all 256 values in every lane at once.
    ///
    /// Error *kinds* are deliberately not compared: the vector path ORs every
    /// lane's verdict into one accumulator and reports it before the scalar
    /// tail ever runs, so an input that is both mis-sized and mis-charactered
    /// can legitimately be `InvalidLength` for one and `InvalidCharacter` for
    /// the other. Rejecting it at all is the contract.
    fn decode_matches_scalar<const N: usize, const CAP: usize>() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; N] = kani::any();

        let mut vbmi_out = [0u8; CAP];
        let mut scalar_out = [0u8; CAP];

        let vbmi = unsafe { decode_slice_avx512_vbmi(&config, &input, &mut vbmi_out) };
        let scalar = crate::scalar::decode_slice(&config, &input, &mut scalar_out);

        match scalar {
            Ok(n) => {
                assert_eq!(vbmi, Ok(n), "scalar accepted an input the kernel rejected");
                assert_eq!(
                    &vbmi_out[..n],
                    &scalar_out[..n],
                    "kernel and scalar decoded to different bytes"
                );
            }
            Err(_) => assert!(vbmi.is_err(), "kernel accepted an input scalar rejected"),
        }
    }

    #[kani::proof]
    #[kani::stub(_mm512_permutexvar_epi8, m::permutexvar_epi8_model)]
    #[kani::stub(_mm512_permutex2var_epi8, m::permutex2var_epi8_model)]
    #[kani::stub(_mm512_multishift_epi64_epi8, m::multishift_epi64_epi8_model)]
    #[kani::stub(_mm512_maddubs_epi16, m::maddubs_epi16_model)]
    #[kani::stub(_mm512_madd_epi16, m::madd_epi16_model)]
    #[kani::stub(_mm512_ternarylogic_epi32, m::ternarylogic_epi32_model)]
    #[kani::stub(_mm512_movepi8_mask, m::movepi8_mask_model)]
    #[kani::stub(_mm512_mask_loadu_epi8, m::mask_loadu_epi8_model)]
    #[kani::stub(_mm512_maskz_loadu_epi8, m::maskz_loadu_epi8_model)]
    #[kani::stub(_mm512_mask_storeu_epi8, m::mask_storeu_epi8_model)]
    fn check_vbmi_decode_matches_scalar() {
        decode_matches_scalar::<DEC_KERNEL_LEN, DEC_KERNEL_CAP>();
    }

    #[kani::proof]
    #[kani::stub(_mm512_permutexvar_epi8, m::permutexvar_epi8_model)]
    #[kani::stub(_mm512_permutex2var_epi8, m::permutex2var_epi8_model)]
    #[kani::stub(_mm512_multishift_epi64_epi8, m::multishift_epi64_epi8_model)]
    #[kani::stub(_mm512_maddubs_epi16, m::maddubs_epi16_model)]
    #[kani::stub(_mm512_madd_epi16, m::madd_epi16_model)]
    #[kani::stub(_mm512_ternarylogic_epi32, m::ternarylogic_epi32_model)]
    #[kani::stub(_mm512_movepi8_mask, m::movepi8_mask_model)]
    #[kani::stub(_mm512_mask_loadu_epi8, m::mask_loadu_epi8_model)]
    #[kani::stub(_mm512_maskz_loadu_epi8, m::maskz_loadu_epi8_model)]
    #[kani::stub(_mm512_mask_storeu_epi8, m::mask_storeu_epi8_model)]
    fn check_vbmi_decode_matches_scalar_masked() {
        decode_matches_scalar::<DEC_MASKED_KERNEL_LEN, DEC_MASKED_KERNEL_CAP>();
    }

    /// `Decode(Encode(x)) == x` over every input of [`ROUNDTRIP_LEN`] bytes,
    /// through both kernels end to end.
    #[kani::proof]
    #[kani::stub(_mm512_permutexvar_epi8, m::permutexvar_epi8_model)]
    #[kani::stub(_mm512_permutex2var_epi8, m::permutex2var_epi8_model)]
    #[kani::stub(_mm512_multishift_epi64_epi8, m::multishift_epi64_epi8_model)]
    #[kani::stub(_mm512_maddubs_epi16, m::maddubs_epi16_model)]
    #[kani::stub(_mm512_madd_epi16, m::madd_epi16_model)]
    #[kani::stub(_mm512_ternarylogic_epi32, m::ternarylogic_epi32_model)]
    #[kani::stub(_mm512_movepi8_mask, m::movepi8_mask_model)]
    #[kani::stub(_mm512_mask_loadu_epi8, m::mask_loadu_epi8_model)]
    #[kani::stub(_mm512_maskz_loadu_epi8, m::maskz_loadu_epi8_model)]
    #[kani::stub(_mm512_mask_storeu_epi8, m::mask_storeu_epi8_model)]
    fn check_vbmi_roundtrip_standard() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        let input: [u8; ROUNDTRIP_LEN] = kani::any();

        let mut enc_buf = [0u8; ROUNDTRIP_ENC_CAP];
        let mut dec_buf = [0u8; ROUNDTRIP_DEC_CAP];

        unsafe {
            encode_slice_avx512_vbmi(&config, &input, &mut enc_buf);
            let dec_len = decode_slice_avx512_vbmi(&config, &enc_buf, &mut dec_buf)
                .expect("valid encoding failed to decode");
            assert_eq!(dec_len, ROUNDTRIP_LEN);
            assert_eq!(&dec_buf[..dec_len], &input, "roundtrip mismatch");
        }
    }
}

/// Rust models of the AVX-512 instructions the VBMI kernels cannot execute
/// symbolically.
///
/// Each is a transcription of the `<operation>` pseudocode published in the
/// Intel Intrinsics Guide (data version 3.6.9), quoted line for line in the
/// comments with the Rust statement it became directly beneath it. Nothing is
/// paraphrased, condensed or "improved": if Intel writes a bit-by-bit loop where
/// a rotate would do, so does the model, because the comments are the
/// specification these proofs are checked against and a reader has to be able to
/// diff them against the guide symbol by symbol.
///
/// One systematic departure, and only one. Intel addresses vectors by **bit**
/// offset — `i := j*8`, then `dst[i+7:i]` for the byte at that offset. Rust
/// indexes bytes, so every transcription keeps Intel's bit-offset variables
/// verbatim and divides by 8 at the point of access (`dst[i / 8]`). Where Intel
/// addresses an individual bit, [`bit`] does it. Lines carrying no Intel text
/// are marked `NOTE:`.
///
/// Two consumers with different appetites. Miri implements almost all of these
/// itself, so it takes only the three byte permutes it lacks, through the
/// `cfg(miri)` shims in the parent module — everything else it executes for
/// real, and the semantics are the Miri developers' problem rather than ours.
/// Kani cannot execute any of them, so its proofs name the whole set in
/// `#[kani::stub(...)]`.
///
/// Two things hold the models up. First, [`avx512_vbmi_stub_equivalence`] runs
/// every one of them against the real instruction, on any host that has the
/// subsets — it skips rather than fails elsewhere, so it costs nothing on a
/// plain runner and checks everything on a VBMI one. Second, and where that
/// check has not run, each model is a mechanical transcription of Intel's own
/// definition of the instruction, kept beside the pseudocode it came from so a
/// reader can diff the two by eye; a CPU that disagrees with that pseudocode
/// has an erratum, not a bug this crate could have anticipated. The three Miri
/// takes are additionally exercised against the `base64` oracle by every run of
/// the Miri coverage suite below.
// Every consumer of this module is invisible to rustc's dead-code pass: the
// Kani proofs reach the models through `#[kani::stub(...)]` attribute
// arguments, and Miri reaches only the three byte permutes, through the
// `cfg(miri)` shims in the parent module. So which models look "used" depends
// on which harness is being compiled, and the answer is never the whole set.
#[allow(dead_code)]
#[allow(non_snake_case)]
// As the AVX2 models: literal-transcription lints, disabled rather than let
// them reshape the pseudocode.
#[allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_const_for_fn,
    clippy::missing_transmute_annotations,
    clippy::needless_late_init,
    clippy::needless_range_loop
)]
pub(super) mod intrinsic_models {
    use super::*;
    use std::mem::transmute;

    // NOTE: scaffolding, not from Intel. Reads bit `n` of a little-endian byte
    // vector, for the places the pseudocode indexes a single bit.
    fn bit(v: &[u8; 64], n: usize) -> u8 {
        (v[n / 8] >> (n % 8)) & 1
    }

    // NOTE: scaffolding, not from Intel. The `Saturate16` helper the pseudocode
    // calls by name.
    fn Saturate16(x: i32) -> i16 {
        x.clamp(-32768, 32767) as i16
    }

    // STUB: _mm512_permutexvar_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutexvar_epi8
    pub(in crate::simd::avx512_vbmi) unsafe fn permutexvar_epi8_model(
        idx: __m512i,
        a: __m512i,
    ) -> __m512i {
        let idx: [u8; 64] = unsafe { transmute(idx) };
        let a: [u8; 64] = unsafe { transmute(a) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // 	i := j*8
            let i = j * 8;
            // 	id := idx[i+5:i]*8
            let id = usize::from(idx[i / 8] & 0x3F) * 8;
            // 	dst[i+7:i] := a[id+7:id]
            dst[i / 8] = a[id / 8];
        }
        // ENDFOR
        // dst[MAX:512] := 0
        // NOTE: `__m512i` is exactly 512 bits; there is nothing above to zero.

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_permutex2var_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutex2var_epi8
    pub(in crate::simd::avx512_vbmi) unsafe fn permutex2var_epi8_model(
        a: __m512i,
        idx: __m512i,
        b: __m512i,
    ) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let idx: [u8; 64] = unsafe { transmute(idx) };
        let b: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // 	i := j*8
            let i = j * 8;
            // 	off := 8*idx[i+5:i]
            let off = 8 * usize::from(idx[i / 8] & 0x3F);
            // 	dst[i+7:i] := idx[i+6] ? b[off+7:off] : a[off+7:off]
            dst[i / 8] = if bit(&idx, i + 6) == 1 {
                b[off / 8]
            } else {
                a[off / 8]
            };
        }
        // ENDFOR
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_multishift_epi64_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_multishift_epi64_epi8
    pub(in crate::simd::avx512_vbmi) unsafe fn multishift_epi64_epi8_model(
        a: __m512i,
        b: __m512i,
    ) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR i := 0 to 7
        for i in 0..8 {
            // 	q := i * 64
            let q = i * 64;
            // 	FOR j := 0 to 7
            for j in 0..8 {
                // 		tmp8 := 0
                let mut tmp8: u8 = 0;
                // 		ctrl := a[q+j*8+7:q+j*8] & 63
                let ctrl = usize::from(a[(q + j * 8) / 8]) & 63;
                // 		FOR l := 0 to 7
                for l in 0..8 {
                    // 			tmp8[l] := b[q+((ctrl+l) & 63)]
                    tmp8 |= bit(&b, q + ((ctrl + l) & 63)) << l;
                }
                // 		ENDFOR
                // 		dst[q+j*8+7:q+j*8] := tmp8[7:0]
                dst[(q + j * 8) / 8] = tmp8;
            }
            // 	ENDFOR
        }
        // ENDFOR
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maddubs_epi16
    pub(in crate::simd::avx512_vbmi) unsafe fn maddubs_epi16_model(
        a: __m512i,
        b: __m512i,
    ) -> __m512i {
        // NOTE: `a` holds unsigned bytes, `b` signed ones.
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [i8; 64] = unsafe { transmute(b) };
        let mut dst = [0i16; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // 	i := j*16
            let i = j * 16;
            // 	dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[i / 16] = Saturate16(
                i32::from(a[(i + 8) / 8]) * i32::from(b[(i + 8) / 8])
                    + i32::from(a[i / 8]) * i32::from(b[i / 8]),
            );
        }
        // ENDFOR
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_madd_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_madd_epi16
    pub(in crate::simd::avx512_vbmi) unsafe fn madd_epi16_model(a: __m512i, b: __m512i) -> __m512i {
        let a: [i16; 32] = unsafe { transmute(a) };
        let b: [i16; 32] = unsafe { transmute(b) };
        let mut dst = [0i32; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
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
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_ternarylogic_epi32
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_ternarylogic_epi32
    //
    // NOTE: this is the one entry whose pseudocode the guide does not print in
    // full. `TernaryOP` is a 256-arm `CASE` and Intel elides arms 2..=253 behind
    // a literal `// ...`, so a line-for-line transcription is not available and
    // the definition below is reconstructed. It is the standard truth-table
    // encoding: arm `n` returns bit `(a<<2)|(b<<1)|c` of `n`. The four arms
    // Intel does print are quoted verbatim and are checked against this
    // reconstruction by `ternarylogic_matches_intels_printed_cases`.
    //
    // DEFINE TernaryOP(imm8, a, b, c) {
    // 	CASE imm8[7:0] OF
    // 	0: dst[0] := 0                   // imm8[7:0] := 0
    // 	1: dst[0] := NOT (a OR b OR c)   // imm8[7:0] := NOT (_MM_TERNLOG_A OR _MM_TERNLOG_B OR _MM_TERNLOG_C)
    // 	// ...
    // 	254: dst[0] := a OR b OR c       // imm8[7:0] := _MM_TERNLOG_A OR _MM_TERNLOG_B OR _MM_TERNLOG_C
    // 	255: dst[0] := 1                 // imm8[7:0] := 1
    // 	ESAC
    // }
    pub(in crate::simd::avx512_vbmi) fn TernaryOP(imm8: i32, a: u32, b: u32, c: u32) -> u32 {
        ((imm8 as u32) >> ((a << 2) | (b << 1) | c)) & 1
    }

    pub(in crate::simd::avx512_vbmi) unsafe fn ternarylogic_epi32_model<const IMM8: i32>(
        a: __m512i,
        b: __m512i,
        c: __m512i,
    ) -> __m512i {
        let a: [u32; 16] = unsafe { transmute(a) };
        let b: [u32; 16] = unsafe { transmute(b) };
        let c: [u32; 16] = unsafe { transmute(c) };
        let mut dst = [0u32; 16];

        // imm8[7:0] = LogicExp(_MM_TERNLOG_A, _MM_TERNLOG_B, _MM_TERNLOG_C)
        let imm8 = IMM8;

        // FOR j := 0 to 15
        for j in 0..16 {
            // 	i := j*32
            let i = j * 32;
            // 	FOR h := 0 to 31
            for h in 0..32 {
                // 		dst[i+h] := TernaryOP(imm8[7:0], a[i+h], b[i+h], c[i+h])
                dst[i / 32] |= TernaryOP(
                    imm8 & 0xFF,
                    (a[i / 32] >> h) & 1,
                    (b[i / 32] >> h) & 1,
                    (c[i / 32] >> h) & 1,
                ) << h;
            }
            // 	ENDFOR
        }
        // ENDFOR
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_movepi8_mask
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_movepi8_mask
    pub(in crate::simd::avx512_vbmi) unsafe fn movepi8_mask_model(a: __m512i) -> u64 {
        let a: [u8; 64] = unsafe { transmute(a) };
        let mut k = 0u64;

        // FOR j := 0 to 63
        for j in 0..64 {
            // 	i := j*8
            let i = j * 8;
            // 	IF a[i+7]
            if bit(&a, i + 7) == 1 {
                // 		k[j] := 1
                k |= 1u64 << j;
            // 	ELSE
            } else {
                // 		k[j] := 0
                k &= !(1u64 << j);
            }
            // 	FI
        }
        // ENDFOR
        // k[MAX:64] := 0

        k
    }

    // STUB: _mm512_mask_loadu_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi8
    pub(in crate::simd::avx512_vbmi) unsafe fn mask_loadu_epi8_model(
        src: __m512i,
        k: u64,
        mem_addr: *const i8,
    ) -> __m512i {
        let src: [u8; 64] = unsafe { transmute(src) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // 	i := j*8
            let i = j * 8;
            // 	IF k[j]
            if (k >> j) & 1 == 1 {
                // 		dst[i+7:i] := MEM[mem_addr+i+7:mem_addr+i]
                dst[i / 8] = unsafe { mem_addr.add(i / 8).read_unaligned() }.cast_unsigned();
            // 	ELSE
            } else {
                // 		dst[i+7:i] := src[i+7:i]
                dst[i / 8] = src[i / 8];
            }
            // 	FI
        }
        // ENDFOR
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_maskz_loadu_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi8
    //
    // NOTE: as `mask_loadu_epi8_model`, the masked-off lanes are never read.
    pub(in crate::simd::avx512_vbmi) unsafe fn maskz_loadu_epi8_model(
        k: u64,
        mem_addr: *const i8,
    ) -> __m512i {
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // 	i := j*8
            let i = j * 8;
            // 	IF k[j]
            if (k >> j) & 1 == 1 {
                // 		dst[i+7:i] := MEM[mem_addr+i+7:mem_addr+i]
                dst[i / 8] = unsafe { mem_addr.add(i / 8).read_unaligned() }.cast_unsigned();
            // 	ELSE
            } else {
                // 		dst[i+7:i] := 0
                dst[i / 8] = 0;
            }
            // 	FI
        }
        // ENDFOR
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_mask_storeu_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi8
    //
    // NOTE: as `mask_loadu_epi8_model`, the masked-off lanes are never written.
    pub(in crate::simd::avx512_vbmi) unsafe fn mask_storeu_epi8_model(
        mem_addr: *mut i8,
        k: u64,
        a: __m512i,
    ) {
        let a: [u8; 64] = unsafe { transmute(a) };

        // FOR j := 0 to 63
        for j in 0..64 {
            // 	i := j*8
            let i = j * 8;
            // 	IF k[j]
            if (k >> j) & 1 == 1 {
                // 		MEM[mem_addr+i+7:mem_addr+i] := a[i+7:i]
                unsafe { mem_addr.add(i / 8).write_unaligned(a[i / 8].cast_signed()) };
            }
            // 	FI
        }
        // ENDFOR
    }
}

/// The one reconstruction in [`intrinsic_models`] — `TernaryOP`, whose 256-arm
/// `CASE` the Intel guide abbreviates — checked against the four arms the guide
/// does print, over every combination of the three input bits.
#[cfg(test)]
mod ternarylogic_reconstruction {
    use super::intrinsic_models::TernaryOP;

    #[test]
    fn ternarylogic_matches_intels_printed_cases() {
        for a in 0..2u32 {
            for b in 0..2u32 {
                for c in 0..2u32 {
                    // 0: dst[0] := 0
                    assert_eq!(TernaryOP(0, a, b, c), 0, "case 0 at {a}{b}{c}");
                    // 1: dst[0] := NOT (a OR b OR c)
                    assert_eq!(
                        TernaryOP(1, a, b, c),
                        1 - (a | b | c),
                        "case 1 at {a}{b}{c}"
                    );
                    // 254: dst[0] := a OR b OR c
                    assert_eq!(TernaryOP(254, a, b, c), a | b | c, "case 254 at {a}{b}{c}");
                    // 255: dst[0] := 1
                    assert_eq!(TernaryOP(255, a, b, c), 1, "case 255 at {a}{b}{c}");
                }
            }
        }
    }
}

/// Checks every model in [`intrinsic_models`] against the real instruction on
/// AVX-512-VBMI hardware, under plain `cargo test`.
///
/// Opportunistic by design: it skips, loudly, on a host without the subsets, so
/// it is free to run everywhere and does the real work wherever the silicon
/// turns up — a developer machine, the benchmark box, or a CI runner that
/// happens to land on VBMI-capable hardware. The `simd-avx512-vbmi` job in
/// `tests.yml` runs it for exactly that reason and reports which case it hit.
#[cfg(all(test, not(miri)))]
mod avx512_vbmi_stub_equivalence {
    use super::intrinsic_models as model;
    use super::*;

    /// Saturation and sign boundaries, the bit-6/bit-7 selectors the VBMI
    /// permutes key off, index-shaped bytes, and deterministic noise.
    fn probes() -> Vec<[u8; 64]> {
        let byte = |i: usize| u8::try_from(i).expect("index below the 64-byte vector width");

        let mut out = vec![[0x00; 64], [0xFF; 64], [0x80; 64], [0x7F; 64], [0x40; 64]];
        out.push(core::array::from_fn(byte));
        out.push(core::array::from_fn(|i| byte(i) | 0x80));
        out.push(core::array::from_fn(|i| byte(i) | 0x40));
        out.push(core::array::from_fn(|i| byte(i % 16)));
        out.push(core::array::from_fn(|i| 0xFF - byte(i)));

        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..8 {
            out.push(core::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                u8::try_from(state >> 56).expect("shifted down to 8 bits")
            }));
        }
        out
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    unsafe fn compare_all() {
        use std::mem::transmute;

        let byte = |i: usize| u8::try_from(i).expect("index below the 64-byte vector width");
        let probes = probes();
        // SAFETY: `__m512i` has no invalid bit patterns, so it and `[u8; 64]`
        // are freely transmutable both ways.
        let bytes = |v: __m512i| -> [u8; 64] { unsafe { transmute::<__m512i, [u8; 64]>(v) } };
        let zmm = |b: [u8; 64]| -> __m512i { unsafe { transmute::<[u8; 64], __m512i>(b) } };

        // Each arm: `real(a, b)` must equal `model(a, b)` for every probe pair.
        macro_rules! same2 {
            ($real:ident, $model:ident, $wrap:expr) => {
                for x in &probes {
                    for y in &probes {
                        let (a, b) = (zmm(*x), zmm(*y));
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

        same2!(_mm512_permutexvar_epi8, permutexvar_epi8_model, bytes);
        same2!(
            _mm512_multishift_epi64_epi8,
            multishift_epi64_epi8_model,
            bytes
        );
        same2!(_mm512_maddubs_epi16, maddubs_epi16_model, bytes);
        same2!(_mm512_madd_epi16, madd_epi16_model, bytes);

        for x in &probes {
            let a = zmm(*x);
            assert_eq!(
                _mm512_movepi8_mask(a),
                unsafe { model::movepi8_mask_model(a) },
                "_mm512_movepi8_mask: a={x:02x?}"
            );

            for y in &probes {
                let b = zmm(*y);
                assert_eq!(
                    bytes(_mm512_permutex2var_epi8(a, b, a)),
                    bytes(unsafe { model::permutex2var_epi8_model(a, b, a) }),
                    "_mm512_permutex2var_epi8: a={x:02x?} idx={y:02x?}"
                );
                assert_eq!(
                    bytes(_mm512_ternarylogic_epi32::<0xFE>(a, b, a)),
                    bytes(unsafe { model::ternarylogic_epi32_model::<0xFE>(a, b, a) }),
                    "_mm512_ternarylogic_epi32: a={x:02x?} b={y:02x?}"
                );
            }
        }

        // Masked memory ops, over a mask that exercises both halves and both
        // ends of the vector.
        for &k in &[
            0u64,
            1,
            0xFFFF_FFFF_FFFF_FFFF,
            0x0000_FFFF_FFFF_FFFF,
            0xAAAA_AAAA_AAAA_AAAA,
        ] {
            let src_bytes: [u8; 64] = core::array::from_fn(|i| byte(i) ^ 0x5A);
            let fill = zmm([0x11; 64]);

            assert_eq!(
                bytes(unsafe { _mm512_mask_loadu_epi8(fill, k, src_bytes.as_ptr().cast()) }),
                bytes(unsafe { model::mask_loadu_epi8_model(fill, k, src_bytes.as_ptr().cast()) }),
                "_mm512_mask_loadu_epi8: k={k:#018x}"
            );
            assert_eq!(
                bytes(unsafe { _mm512_maskz_loadu_epi8(k, src_bytes.as_ptr().cast()) }),
                bytes(unsafe { model::maskz_loadu_epi8_model(k, src_bytes.as_ptr().cast()) }),
                "_mm512_maskz_loadu_epi8: k={k:#018x}"
            );

            let value = zmm(core::array::from_fn(|i| byte(i).wrapping_mul(3)));
            let mut real_dst = [0u8; 64];
            let mut model_dst = [0u8; 64];
            unsafe { _mm512_mask_storeu_epi8(real_dst.as_mut_ptr().cast(), k, value) };
            unsafe { model::mask_storeu_epi8_model(model_dst.as_mut_ptr().cast(), k, value) };
            assert_eq!(real_dst, model_dst, "_mm512_mask_storeu_epi8: k={k:#018x}");
        }
    }

    #[test]
    fn avx512_vbmi_models_match_hardware() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi"))
        {
            eprintln!("skipping: host CPU lacks AVX-512-VBMI");
            return;
        }
        unsafe { compare_all() };
    }
}

#[cfg(all(test, miri))]
mod miri_avx512_vbmi_coverage {
    use super::*;
    use crate::simd::testutil::{check_decode, check_decode_exact, check_encode};
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD};

    fn enc(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_encode(config, oracle, encode_slice_avx512_vbmi, len);
    }
    fn dec(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_decode(config, oracle, decode_slice_avx512_vbmi, len);
    }
    fn exact(config: &Config, oracle: &impl base64::Engine, len: usize) {
        check_decode_exact(config, oracle, decode_slice_avx512_vbmi, len);
    }

    const STD: Config = Config {
        url_safe: false,
        padding: true,
    };
    const URL: Config = Config {
        url_safe: true,
        padding: true,
    };
    const NO_PAD: Config = Config {
        url_safe: false,
        padding: false,
    };
    const NO_PAD_URL: Config = Config {
        url_safe: true,
        padding: false,
    };

    /// Tier boundaries, encode. The vector path now runs down to 3 bytes, so
    /// scalar only ever sees a final 1-2 byte group: quad at >= 256, single at
    /// >= 64 (a plain load reads 64 to consume 48), masked below that.
    #[test]
    fn miri_avx512_vbmi_encode_tier_boundaries() {
        for &len in &[0, 1, 2, 3, 4, 5, 47, 48, 49, 63, 64, 65, 66, 95, 96] {
            enc(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_encode_quad_tier_boundaries() {
        for &len in &[190, 192, 255, 256, 257, 259, 448] {
            enc(&STD, &STANDARD, len);
        }
    }

    /// The masked tier runs twice whenever the remainder exceeds 48 bytes,
    /// which is every length whose `len % 48` lands in 49..63 after the single
    /// tier stops.
    #[test]
    fn miri_avx512_vbmi_encode_masked_tier_two_passes() {
        for &len in &[51, 54, 60, 62, 63] {
            enc(&STD, &STANDARD, len);
            enc(&NO_PAD, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_encode_url_safe() {
        enc(&URL, &URL_SAFE, 100);
        for &len in &[3, 47, 63, 100, 259] {
            enc(&NO_PAD_URL, &URL_SAFE_NO_PAD, len);
        }
    }

    /// Tier boundaries, decode, in *decoded* byte lengths. The character
    /// thresholds are 260 / 68 / 8; every tier stops 4 characters short so the
    /// only group that may carry '=' is always the scalar tail's.
    #[test]
    fn miri_avx512_vbmi_decode_tier_boundaries() {
        for &len in &[0, 1, 2, 3, 4, 5, 6, 45, 48, 49, 50, 51, 52, 66, 96] {
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_quad_tier_boundaries() {
        for &len in &[144, 192, 193, 194, 195, 196, 255, 300] {
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_url_safe() {
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_";
        let mut dst = [0u8; 64];
        unsafe {
            decode_slice_avx512_vbmi(&NO_PAD_URL, input, &mut dst).unwrap();
        }
        dec(&URL, &URL_SAFE, 100);
    }

    /// Invalid bytes must be caught in every tier and the scalar tail. Unlike
    /// the other paths, VBMI has a second rejection route: bytes >= 128 alias
    /// into the LUT via bit 6, so the accumulator must catch them too. The
    /// accumulator spans the whole call, so a byte anywhere in the vector
    /// region still fails the single test after the loops.
    #[test]
    fn miri_avx512_vbmi_decode_error_detection() {
        let mut dst = [0u8; 512];
        for &(len, bad_at, byte, where_) in &[
            (260, 0, b'$', "sentinel, quad tier, first byte"),
            (260, 255, b'$', "sentinel, quad tier, last byte"),
            (68, 63, b'?', "sentinel, single tier"),
            (12, 5, b'?', "sentinel, masked tier"),
            (260, 0, 0x80u8, "high bit, quad tier"),
            (68, 0, 0xFF, "high bit, single tier"),
            (12, 0, 0x80u8, "high bit, masked tier"),
            (68, 65, b'?', "scalar tail"),
        ] {
            let mut input = vec![b'A'; len];
            input[bad_at] = byte;
            let res = unsafe { decode_slice_avx512_vbmi(&STD, &input, &mut dst) };
            assert!(res.is_err(), "missed invalid byte in {where_}");
        }
    }

    #[test]
    fn miri_avx512_vbmi_roundtrip_standard() {
        for &len in &[48, 96, 192, 193, 240, 384] {
            enc(&STD, &STANDARD, len);
            dec(&STD, &STANDARD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_no_padding() {
        for &len in &[1, 3, 48, 49, 63, 96, 192, 193] {
            enc(&NO_PAD, &STANDARD_NO_PAD, len);
        }
        for &len in &[3, 6, 48, 49, 51, 96, 192, 193] {
            dec(&NO_PAD, &STANDARD_NO_PAD, len);
        }
    }

    /// Masked-store regression: every chunk-boundary length must decode into an
    /// exactly-sized buffer without overrunning, for both alphabets and
    /// padded/unpadded input.
    #[test]
    fn miri_avx512_vbmi_decode_exact_buffer_boundaries() {
        for &len in &[3, 6, 45, 48, 51, 96, 192, 193, 195, 240, 384, 1000, 1001] {
            exact(&STD, &STANDARD, len);
            exact(&URL, &URL_SAFE, len);
            exact(&NO_PAD, &STANDARD_NO_PAD, len);
        }
    }
}

#[cfg(all(test, not(miri)))]
mod avx512_vbmi_hardware_coverage {
    use super::*;
    use crate::simd::testutil::check_decode_exact;
    use base64::engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE};

    /// The same exact-buffer masked-store regression the Miri suite runs, but on
    /// real AVX-512-VBMI silicon (skipped when the host CPU lacks it).
    #[test]
    fn hw_avx512_vbmi_decode_exact_buffer_boundaries() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi"))
        {
            eprintln!("skipping: host CPU lacks AVX-512-VBMI");
            return;
        }

        let standard = Config {
            url_safe: false,
            padding: true,
        };
        let url_safe = Config {
            url_safe: true,
            padding: true,
        };
        let no_pad = Config {
            url_safe: false,
            padding: false,
        };

        for &len in &[3, 6, 45, 48, 51, 96, 192, 193, 195, 240, 384, 1000, 1001] {
            check_decode_exact(&standard, &STANDARD, decode_slice_avx512_vbmi, len);
            check_decode_exact(&url_safe, &URL_SAFE, decode_slice_avx512_vbmi, len);
            check_decode_exact(&no_pad, &STANDARD_NO_PAD, decode_slice_avx512_vbmi, len);
        }
    }
}
