# Safety & Verification

**Philosophy:** `Safety > Performance > Convenience`

This library achieves its performance through `unsafe` SIMD intrinsics and raw pointer arithmetic. Rather than relying on manual review alone to justify that tradeoff, every `unsafe` code path is checked by multiple independent verification layers: Kani's formal model checker, MIRI's Undefined Behavior interpreter, MemorySanitizer, and continuous fuzzing.

## Verification Status Matrix

We use a "Swiss Cheese" model where multiple layers of verification cover each other's blind spots.

| Architecture | MIRI (UB Check) | MSan (Uninit Check) | Kani (Math Proof) | Fuzzing (2.5B+) | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Scalar** | ✅ Passed | ✅ Passed | ✅ **Proven** (CI) | ✅ Passed | **Formally Verified** |
| **AVX2** | ✅ Passed | ✅ Passed | ✅ **Proven** (CI) | ✅ Passed | **Formally Verified** |
| **AVX512** | ✅ Passed | ✅ Passed | ✅ **Proven** (local) | ✅ Passed | **Formally Verified** |
| **AVX512-VBMI** | ✅ Passed | ✅ Passed | ❌ N/A | ✅ Passed | **MIRI Verified** |
| **NEON** | ✅ Passed | ✅ Passed | ❌ N/A | ❌ N/A | **MIRI Verified** |

`AVX512` and `AVX512-VBMI` are two distinct code paths (see [Architecture & Design](design.md)):
plain AVX512 (`avx512f`+`avx512bw`) and a separate, faster VBMI path (`avx512vbmi`,
`vpermb`/`vpermi2b`) that the runtime dispatcher only picks on CPUs that support it.
They are verified differently — see [Kani Coverage: AVX512 vs. AVX512-VBMI](#kani-coverage-avx512-vs-avx512-vbmi)
below for exactly what "Proven (local)" and "N/A" mean here.

## Kani Coverage: AVX512 vs. AVX512-VBMI

The dispatcher's "AVX512" tier is actually two separate compiled code paths, and they
are **not** verified to the same standard:

*   **Plain AVX512** (`encode_slice_avx512` / `decode_slice_avx512`, requires `avx512f`
    + `avx512bw`) has a full Kani proof (`kani_verification_avx512`), the same as Scalar
    and AVX2.
*   **AVX512-VBMI** (`encode_slice_avx512_vbmi` / `decode_slice_avx512_vbmi`, requires
    the additional `avx512vbmi` feature) has **no Kani proof**. It relies on `vpermb`
    (`_mm512_permutexvar_epi8`) and `vpermi2b` (`_mm512_permutex2var_epi8`), and we not yet 
    added stubbing support for those two intrinsics today.

### CI vs. local Kani execution

GitHub Actions' `verification.yml` workflow only runs the **Scalar** and **AVX2** Kani
harnesses (`kani_verification_scalar`, `kani_verification_avx2`). The **AVX512** proof
(`kani_verification_avx512`) is not run in CI — its induction length and symbolic-byte
state space made it too slow for GitHub Actions' runners/time budget in practice. It is
run and re-verified locally by the maintainer before each release, and it passes.

This isn't a gap you have to take on faith: the harness is checked into the repository
and anyone can reproduce the proof on their own machine with:

```sh
cargo kani --unstable stubbing --harness kani_verification_avx512
```

## Deep Dive: The Kani Proofs (Proof by Induction)

The most distinctive part of `base64-turbo`'s verification is the use of the **Kani Model Checker** to prove the correctness of our SIMD logic for entire classes of input, rather than relying on individual test cases.

### The Challenge
It is impossible to verify an input of "infinite length" using standard testing, and even a bounded model checker like Kani/CBMC cannot exhaustively explore arbitrarily large buffers — the state space grows too fast.

### The Solution: Structural Induction
Base64 encoding/decoding is a linear, block-based operation, so we do not need to check every length. It is enough to prove the logic holds for a single fixed length that is chosen to exercise:
1.  **The loop body, at least twice:** proving that the pointer/state one iteration hands off is itself a valid entry state for the next iteration.
2.  **The transition:** the handover from the SIMD loop to the Scalar fallback.
3.  **The tail:** a non-empty, non-aligned scalar remainder, so the SIMD-to-scalar handoff is exercised too.

If Kani proves those properties hold for one such length with fully symbolic input, the result generalizes by induction to all inputs from length `0` to `usize::MAX`.

### The Proof Harness
`encode_slice_avx2` only enters its SIMD path once `len >= 32`, and then processes input in 24-byte rounds; `decode_slice_avx2` processes input in 32-byte single-vector passes, with a separate, faster 128-byte "quad" tier used for larger inputs. The induction-length constants in `src/simd/avx2.rs` are derived directly from that structure:

*   `ENC_INDUCTION_LEN = 53` — two 24-byte AVX2 rounds (48 bytes) plus a 5-byte scalar remainder.
*   `DEC_INDUCTION_LEN = 69` — two 32-byte AVX2 single-vector passes (64 bytes) plus a 5-byte scalar remainder.
*   `QUAD_ENC_INDUCTION_LEN = 125` — the smallest length that triggers exactly one pass of the decoder's 4x-unrolled quad tier. This is checked by a separate harness, `check_avx2_quad_tier_roundtrip`, since the quad tier's own unrolled pointer arithmetic is a distinct concern from the single-vector loop's.

An earlier version of this proof used `ENC_INDUCTION_LEN = 29`. That value was incorrect: `encode_slice_avx2`'s `len >= 32` guard means any length below 32, including 29, skips the SIMD path entirely, so the proof was silently exercising only the scalar fallback and never touched the AVX2 encoder it was meant to verify.

By making the input **symbolic** (`kani::any()`), Kani explores every possible bit combination for that fixed length — 2^(53*8) possibilities for the encoder proof — rather than sampling individual cases.

#### Proof Harness (Encode -> Decode Roundtrip)
Simplified from the actual harness in `src/simd/avx2.rs` (which also stubs several AVX2 intrinsics for Kani — see the FAQ below for why that's a valid substitution):

```rust
#[kani::proof]
fn check_avx2_roundtrip_correctness() {
    // `kani::any()` represents ANY possible byte sequence of this length —
    // a symbolic value, not a random sample.
    let config = Config { url_safe: kani::any(), padding: true };
    let input: [u8; ENC_INDUCTION_LEN] = kani::any();

    let mut enc_buf = [0u8; 128];
    let mut dec_buf = [0u8; 128];

    unsafe {
        // Kani verifies this pointer write never goes out of bounds.
        encode_slice_avx2(&config, &input, enc_buf.as_mut_ptr());

        let enc_len = encoded_size(ENC_INDUCTION_LEN, config.padding);
        let encoded_slice = &enc_buf[..enc_len];

        // For ANY valid encoded output, decoding must succeed.
        let dec_len = decode_slice_avx2(&config, encoded_slice, dec_buf.as_mut_ptr())
            .expect("Valid encoding failed to decode");

        assert_eq!(dec_len, ENC_INDUCTION_LEN);
        assert_eq!(&dec_buf[..dec_len], &input, "Roundtrip mismatch");
    }
}

const ENC_INDUCTION_LEN: usize = 53;
```

## The Toolchain

### 1. MIRI (Undefined Behavior Analysis)
We run our comprehensive deterministic test suite under [MIRI](https://github.com/rust-lang/miri), an interpreter that checks for Undefined Behavior according to the strict Rust memory model.

*   **Checks Performed:** Strict provenance tracking, alignment checks, out-of-bounds pointer arithmetic, and data races.
*   **Coverage:** Every distinct code path (single-vector loop, quad-vector loop, and scalar-tail fallback) for **Scalar, AVX2, AVX512, and AVX512-VBMI** is exercised at least once — this is branch coverage, not exhaustive input coverage (that's what the Kani proofs above are for, where they exist — see [Kani Coverage: AVX512 vs. AVX512-VBMI](#kani-coverage-avx512-vs-avx512-vbmi)).
*   **Strategy:** We utilize deterministic input generation to force the engine into every possible boundary condition (e.g., buffer lengths of `0`, `1`, `31`, `32`, `33`, `63`, `64`, `65`...) to prove safe handling of pointers at register boundaries.

### 2. MemorySanitizer (MSan)
While MIRI checks for validity, **MemorySanitizer (MSan)** checks for **Initialization**.

*   **The Threat:** In high-performance code, reading uninitialized memory (padding bytes) is a common source of non-deterministic bugs and security leaks (Information Disclosure).
*   **The Check:** We recompile the **entire Rust Standard Library** from source with MSan instrumentation (`-Z build-std -Z sanitizer=memory`). This allows us to track the "definedness" of every single bit of memory.
*   **Guarantee:** We ensure that our SIMD algorithms (including AVX512's extensive masking operations) never perform logic on garbage data derived from uninitialized buffers.

## FAQ

### Q: Does this crate use `unsafe` Rust?
**A:** Yes, extensively. We use pointers and SIMD intrinsics to achieve speed. All `unsafe` blocks are encapsulated behind a Safe API and covered by the verification layers described above.

### Q: Is it safe to use in production?
**A:** For Scalar, AVX2, and plain AVX512, yes — those paths are Kani-proven in addition to passing MIRI and MSan (see [CI vs. local Kani execution](#ci-vs-local-kani-execution) for the AVX512 caveat). AVX512-VBMI and NEON have MIRI and fuzzing coverage but no Kani proof; we consider both production-ready on that basis, though it is a lower bar than the Kani-proven paths.

### Q: How do I know your SIMD stubs are correct?
**A:** We use literal translation: stub implementations copy the exact variable names and logic flow from the [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html), replicating documented hardware behaviors (saturation, masking) exactly, so they can be checked side-by-side against the reference.
