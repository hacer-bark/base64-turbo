# 🛡️ Safety & Verification

**Philosophy:** `Safety > Performance > Convenience`

At `base64-turbo`, we believe that speed is meaningless if it compromises stability. While this library achieves extreme performance by leveraging `unsafe` SIMD intrinsics and pointer arithmetic, we do not rely on "hope" or "good practices" to prevent crashes.

Instead, we rely on **Mathematical Proofs**, **Strict Formal Audits**, and **Deterministic Analysis**.

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

The most distinctive part of `base64-turbo`'s verification is the use of the **Kani Model Checker** to mathematically prove the correctness of our SIMD logic, rather than relying on test cases alone.

### The Challenge
It is impossible to verify an input of "infinite length" using standard testing. Even symbolic execution engines cannot check a 1GB buffer because the state space is too large.

### The Solution: Structural Induction
Since Base64 encoding/decoding is a linear, block-based operation, we do not need to check infinite lengths. We only need to prove that the logic holds for **one full cycle + the boundaries**.

If we prove:
1.  **The Loop Body:** One full SIMD vector iteration is correct and memory-safe.
2.  **The Transition:** The handover from the SIMD loop to the Scalar fallback is correct.
3.  **The Tail:** The Scalar fallback handles the remaining 0-3 bytes correctly.

Then, by **Mathematical Induction**, we have proven safety for **all** inputs from length `0` to `usize::MAX`.

### The Proof Harness
We utilize a "Magic Number" constant for verification: `ENC_INDUCTION_LEN = 29`.

*   **28 Bytes:** Ensures we trigger exactly one full AVX2 Loop iteration.
*   **+1 Byte:** Forces the code to break out of the SIMD loop and execute the **Scalar Transition** logic.

By making the input **Symbolic** (using `kani::any()`), Kani explores **every possible bit combination** (2^(29*8) possibilities) for that length.

#### Actual Verification Code
Here is the harness that proves the AVX2 Roundtrip (`encode -> decode == input`):

```rust
#[kani::proof]
fn check_avx2_roundtrip_correctness() {
    // 1. Create Symbolic Input
    // `kani::any()` represents ANY possible byte sequence of this length.
    // It is not a random generator; it is a mathematical symbol.
    let config = Config { url_safe: kani::any(), padding: true };
    let input: [u8; ENC_INDUCTION_LEN] = kani::any();

    // 2. Setup Buffers
    let mut enc_buf = [0u8; 128];
    let mut dec_buf = [0u8; 128];

    unsafe {
        // 3. Execute AVX2 Encode (Unsafe Intrinsic)
        // Kani verifies that this POINTER write never goes out of bounds.
        encode_slice_avx2(&config, &input, enc_buf.as_mut_ptr());

        // Calculate expected length
        let enc_len = encoded_size(ENC_INDUCTION_LEN, config.padding);
        let encoded_slice = &enc_buf[..enc_len];

        // 4. Execute AVX2 Decode
        // We assert that for ANY valid encoded output, decoding MUST succeed.
        let dec_len = decode_slice_avx2(&config, encoded_slice, dec_buf.as_mut_ptr())
            .expect("Valid encoding failed to decode");

        // 5. Verify Logic
        // If this assertion passes, it is mathematically impossible for
        // the algorithm to produce the wrong result for this block size.
        assert_eq!(dec_len, ENC_INDUCTION_LEN);
        assert_eq!(&dec_buf[..dec_len], &input, "Roundtrip mismatch");
    }
}

// Why 29?
// Encoder Induction Size: 28 (Satisfies 1 AVX2 Loop) + 1 (Forces Scalar Transition)
const ENC_INDUCTION_LEN: usize = 29;
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

## ❓ FAQ

**Q: Does this crate use `unsafe` Rust?**
**A:** Yes, extensively. We use pointers and SIMD intrinsics to achieve speed. However, all `unsafe` blocks are encapsulated behind a Safe API and have been formally audited.

**Q: Is it safe to use in Production?**
**A:** For Scalar, AVX2, and (plain) AVX512, yes — those paths are Kani-proven (symbolic
execution, not just testing) in addition to passing MIRI and MSan; note that the AVX512
proof runs locally rather than in CI (see
[Kani Coverage: AVX512 vs. AVX512-VBMI](#kani-coverage-avx512-vs-avx512-vbmi)). AVX512-VBMI
and NEON currently have MIRI + fuzzing coverage but no Kani proof (see the matrix above);
we consider both production-ready based on that coverage, but it is a strictly lower bar
than the Kani-proven paths.

**Q: How do I know your SIMD stubs are correct?**
**A:** We use **"Literal Translation."** We copy the exact variable names and logic flow from the [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html), replicating specific hardware behaviors (saturation, masking) exactly as documented, allowing side-by-side verification.
