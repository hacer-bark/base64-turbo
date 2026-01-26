# ❓ Frequently Asked Questions

## 🛡️ Safety & Verification

### Q: The crate uses `unsafe`. How can you claim it is safe?
**A:** We distinguish between "Safe Rust" (the compiler checks it) and "Memory Safe" (mathematically proven to be correct).
While we use `unsafe` pointers and intrinsics for speed, we have established a **Formal Verification Pipeline** using Kani and MIRI. We have mathematically proven that for the verified paths (Scalar, AVX2), there is **no possible input** (from empty strings to infinite streams) that can trigger a buffer overflow, segfault, or panic via the public API.

**[Read the Verification Report](safety/verification.md)**

### Q: Can I crash the library by passing garbage data?
**A:** **No.**
The decoder is resilient. If you pass invalid Base64 strings, random binary noise, or malicious payloads, the library will return a standard `Result::Err`. It will **never** panic or cause Undefined Behavior (UB).

### Q: What happens if I violate safety contracts in the internal `unsafe` API?
**A:** If you bypass the public API (`encode`/`decode`) and call internal `unsafe` functions directly, **you are responsible for the invariants.**
For example, if you pass a null pointer with a non-zero length to an internal function, you have violated the documented safety contract. We do not verify against contract violations in `unsafe` blocks; we verify that our Safe API never violates them.

### Q: Why is the `avx512` feature disabled by default?
**A:** Limitations in the verification toolchain.
While our AVX512 implementation has passed over **2.5 Billion Fuzzing Iterations** and is empirically stable, the **MIRI** interpreter does not yet support AVX512 intrinsics. Strict adherence to our "Security > Performance" philosophy dictates that we do not enable code by default unless it is fully audit-ready by our entire toolchain.

## ⚡ Performance & Usage

### Q: Does this work on `no_std` / Embedded systems?
**A:** **Yes.**
Disable the default `std` feature in your `Cargo.toml`.
```toml
[dependencies]
base64-turbo = { version = "0.1", default-features = false }
```

### Q: Why is `parallel` (Rayon) disabled by default?
**A:** To prevent **Jitter** and **Bloat**.
1.  **Bloat:** We strive to be dependency-free. Enabling parallel pulls in `rayon`.
2.  **Jitter:** In HFT or real-time web servers, creating thread pools can cause non-deterministic latency spikes.
Enable `parallel` only if you are processing massive files (>1MB) and can tolerate the context-switching overhead.

### Q: Does this work on ARM (Apple Silicon / Raspberry Pi)?
**A:** **Not-yet.**
The library uses **Runtime Feature Detection**.
*   On **x86_64:** It detects AVX512/AVX2/SSE4.1.
*   On **ARM:** It would detect NEON when we will add support, for now it falls back to our optimized SWAR Scalar implementation.
The binary is portable; you can move it between CPUs of the same architecture family without crashing.

### Q: Why is this faster than standard look-up tables?
**A:** **Logic > Memory.**
Modern CPUs hate random memory access (which causes cache misses) and random data (which causes branch mispredictions). We replace memory lookups with **Vectorized Arithmetic**. We calculate the Base64 indices using CPU logic gates rather than fetching them from RAM. This keeps the CPU pipeline saturated regardless of the input data entropy.

## 📦 Comparisons & Ecosystem

### Q: Why should I use this over the C library (`turbo-base64`)?
**A:** **Safety.**
The C library is faster (~29 GiB/s vs our ~12 GiB/s), but it is **Unsafe**. It relies on unchecked pointer arithmetic. `base64-turbo` offers the highest possible performance while maintaining **100% Rust Memory Safety guarantees**.

### Q: Is this a fork of `base64-simd`?
**A:** **No.**
This is a clean-sheet design engineered from the ground up to exploit modern micro-architecture features (like AVX512 masking and port saturation) that other crates do not utilize.

### Q: How can I trust this code?
**A:** **Don't.**
Trust the math.
1.  Check our **[GitHub Actions](https://github.com/hacer-bark/base64-turbo/actions)** to see the Kani/MIRI logs.
2.  Verify the **GPG Signatures** on our commits.
3.  Read the code. We have commented extensively on the `unsafe` blocks explaining the invariants.
