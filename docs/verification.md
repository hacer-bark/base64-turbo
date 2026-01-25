# Safety & Formal Verification

**Philosophy:** `Security > Performance > Convenience`

At `base64-turbo`, we believe that speed is meaningless if it compromises stability. While this library achieves extreme performance by leveraging `unsafe` SIMD intrinsics and pointer arithmetic, we do not rely on "hope" or "good practices" to prevent crashes.

Instead, we rely on **Mathematical Proofs** and **Strict Formal Audits**.

We have rigorously audited the codebase to the point where "unsafe code" effectively becomes "safe code." We guarantee that for the supported architectures, **no input exists**—valid or invalid—that can cause Undefined Behavior (UB), Segfaults, or Panics via the public API.

## Verification Status Matrix

Not all architectures support the same verification tooling. We rely on a "Swiss Cheese" model where multiple layers of verification (MIRI + Kani + Fuzzing) cover each other's blind spots.

| Architecture | MIRI (UB Check) | Kani (Math Proof) | Fuzzing (2.5B+) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Scalar** | ✅ Passed | ✅ Passed | ✅ Passed | **100% Formally Verified** |
| **AVX2** | ✅ Passed | ✅ Passed | ✅ Passed | **100% Formally Verified** |
| **SSE4.1** | ✅ Passed | 🚧 In Progress | ✅ Passed | **Memory Safe (MIRI Verified)** |
| **AVX512** | ❌ Not Supported | 🚧 In Progress | ✅ Passed | **Empirically Safe** (Fuzz-Tested) |

> **Note on AVX512:** Since MIRI does not support AVX512 intrinsics, and Kani support is in progress, the `avx512` feature flag is **disabled by default**. We do not enable code by default unless it has passed MIRI or Kani analysis.

## The Toolchain

### 1. Kani Model Checker (Mathematical Proofs)
We use [Kani](https://github.com/model-checking/kani), the same formal verification tool used by Amazon Web Services to audit **Firecracker VM**.

*   **How it works:** Unlike testing, which tries *some* inputs, Kani uses symbolic execution to analyze *all possible* execution paths.
*   **The Guarantee:** We have mathematically proven that for any byte array of any size (from 0 to Infinity), the encoding/decoding loop will **never** read out of bounds or overflow.
*   **SIMD Verification:** Since Kani does not natively support Intel intrinsics, we have written custom semantic stubs based on Intel documentation to verify the logic of our SIMD implementations.

### 2. MIRI (Undefined Behavior Analysis)
We run our test suite under [MIRI](https://github.com/rust-lang/miri), an interpreter that checks for Undefined Behavior according to the Rust memory model.

*   **Checks Performed:** Strict provenance tracking, alignment checks, out-of-bounds pointer arithmetic, and data races.
*   **Coverage:** Covers Scalar, SSE4.1, and AVX2 paths.

### 3. Supply Chain Security (GitHub Policy)
Security is not just about code; it is about process. This repository adheres to strict **Supply Chain Security** protocols to prevent malicious code injection.

1.  **No Direct Commits:** Not even the repo owner can commit to `main`. All changes must go through a Pull Request (PR).
2.  **Required Checks:** A PR cannot be merged unless it passes 3 mandatory gates:
    *   ✅ **Kani Verification**
    *   ✅ **MIRI Audit**
    *   ✅ **Logic/Unit Tests**
3.  **GPG Signing:** All commits in `main` and PRs are cryptographically signed with GPG keys. You can verify the signature of every line of code in this crate.

## Threat Model & Guarantees

### What we Guarantee
We guarantee that for any usage of the **Public Safe API** (`encode`, `decode`, `encode_into`, etc.):
*   **Input Resilience:** You can pass **ANY** slice of bytes (`&[u8]`) of **ANY** length.
*   **Content Agnostic:** You can pass valid Base64, garbage binary data, random noise, or malicious payloads.
*   **Result:** The program will **NEVER** Segfault, Panic, or trigger Undefined Behavior (UB).

### What is Out of Scope (Contract Violations)
The library exposes internal `unsafe` functions for users who need to bypass bounds checks for performance.
*   **Contract Violation:** If you use `unsafe` functions and pass a raw pointer with a length of `0` while the pointer is invalid (null/dangling), you have violated the safety contract documented in the code.
*   **Responsibility:** We do not verify against contract violations in `unsafe` blocks. If you bypass the Safe API, you are responsible for maintaining memory invariants.

## FAQ

**Q: Does this crate use `unsafe` Rust?**
**A:** Yes, extensively. We use pointers and SIMD intrinsics to achieve speed. However, all `unsafe` blocks are encapsulated behind a Safe API and have been formally audited.

**Q: Is it safe to use?**
**A:** Yes. It is **mathematically proven** to be safe for the verified architectures (Scalar/AVX2). "Safe" here isn't an opinion; it's a result of symbolic execution analysis.

**Q: Can I trust you?**
**A:** **No, you should not.** Do not trust the author's words. Trust the cryptographic proofs and the CI logs. You are encouraged to visit the [GitHub Actions](https://github.com/hacer-bark/base64-turbo/actions) tab and inspect the Kani and MIRI logs yourself.

**Q: How do you verify SIMD with Kani?**
**A:** We wrote custom semantic stubs for unsupported SIMD functions (e.g., `_mm256_shuffle_epi8`) that emulate the behavior of the CPU instructions in pure Rust. Kani verifies the logic using these stubs. You can inspect the implementation by yourself.

**Q: Why is AVX512 disabled by default?**
**A:** MIRI does not support AVX512. While we have fuzzed it (2.5B+ ops), we hold ourselves to a standard where "Fuzzing is not enough." Until we can formally verify it with Kani or MIRI, it remains opt-in.
