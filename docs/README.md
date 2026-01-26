# 📚 Technical Documentation

This directory contains detailed technical reports, formal verification proofs, and architectural decision records for `base64-turbo`.

## 📂 Index

### 🛡️ [Safety & Verification](verification.md)
**Target Audience:** Security Auditors, Systems Engineers
*   **Formal Verification:** How we use Kani to mathematically prove the absence of panics/overflows.
*   **UB Checks:** Details on MIRI usage and strict provenance.
*   **Threat Model:** What we protect against and our trust boundaries.

### ⚡ [Benchmarks & Methodology](benchmarks)
**Target Audience:** HFT Developers, Performance Engineers
*   **Methodology:** How we measure throughput and latency (CPU pinning, cache warming).
*   **Hardware Specs:** Detailed breakdown of the test environments (Intel Xeon, Apple M3, etc.).
*   **Reproduction:** Scripts to run the benchmarks yourself.

### 🏗️ [Architecture & Design](design.md)
**Target Audience:** Contributors, Curious Developers
*   **SIMD Selection:** How the runtime detection logic works.
*   **Data Flow:** How bytes move from the API to the CPU registers.
*   **Fallback Strategies:** How we handle architectures without AVX2/SSE4.1.

### ⚖️ [Ecosystem Comparison](ecosystem_comparison.md)
**Target Audience:** Architects, CTOs
*   **Rust vs C:** Detailed breakdown of why C libraries are faster but riskier.
*   **Competitor Analysis:** Feature matrix comparing `base64-turbo` against `base64-simd`, `base64` (std), and `Turbo-Base64`.
*   **Alternatives:** When to use other libraries (e.g., for custom alphabets or WASM).

### ❓ [Frequently Asked Questions (FAQ)](faq.md)
**Target Audience:** All Users
*   **Integration:** Using `no_std` and embedded environments.
*   **Edge Cases:** Handling non-standard padding or whitespace.
*   **Troubleshooting:** Common compile-time or runtime issues.
