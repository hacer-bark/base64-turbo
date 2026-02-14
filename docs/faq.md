# ❓ Frequently Asked Questions

## 🛡️ Safety & Verification

### Q: The crate uses `unsafe`. How can you claim it is safe?
**A:** We distinguish between "Safe Rust" (compiler-checked) and "Memory Safe" (mathematically proven).
While we use `unsafe` pointers and intrinsics to achieve raw speed, we rely on a **Formal Verification Pipeline**. We have mathematically proven via Kani and MIRI that for the verified paths, **no possible input** (from empty strings to infinite streams) can trigger a buffer overflow, segfault, or panic via the public API.

**[Read the Verification Report](./verification.md)**

### Q: Can I crash the library by passing garbage data?
**A:** **No.**
The decoder is resilient. If you pass invalid Base64 strings, random binary noise, or malicious payloads, the library simply returns a `Result::Err`. It will **never** panic or cause Undefined Behavior (UB) as long as you use the Safe API.

### Q: What happens if I violate safety contracts in the internal `unsafe` API?
**A:** **You are responsible for the crash.**
The `unsafe` internal functions (exposed via `unstable`) are raw tools for those who need to bypass bounds checks for absolute maximum performance. If you pass a null pointer or an invalid length to these functions, you violate the contract. We verify that *our* Safe API never violates these contracts, but we cannot protect you if you bypass the API.

### Q: Is AVX512 enabled by default?
**A:** **Yes.**
We detect CPU features at runtime. If your CPU supports AVX512, we use it. If not, we gracefully downgrade to AVX2, SSE4.1, or Scalar. There is no need to manually enable feature flags for SIMD support.

## ⚡ Performance & Usage

### Q: Does this work on ARM (Apple Silicon / Raspberry Pi)?
**A:** **Yes, currently via the Scalar backend.**
The library compiles and runs perfectly on ARM.
*   **x86_64:** Automatically uses AVX512 / AVX2 / SSE4.1.
*   **ARM:** Currently falls back to our optimized Scalar implementation.
*   **Roadmap:** Native NEON support is planned for a future release.

### Q: How do I calculate the buffer size for `encode_into`?
**A:** Use the helper functions.
If you are avoiding allocations, you cannot guess the size.
```rust
// For Encoding:
let needed = STANDARD.encoded_size(input.len());
let mut buf = vec![0u8; needed];

// For Decoding:
let max_needed = STANDARD.estimate_decoded_len(input.len());
let mut buf = vec![0u8; max_needed];
```

### Q: Is the Scalar fallback slow?
**A:** **No, it is highly optimized.**
Even without SIMD, our scalar implementation eliminates many bounds checks found in the standard library. While it won't hit the 20 GiB/s of AVX512, it is competitive with other standard Base64 implementations on ARM.

### Q: Does this work on `no_std` / Embedded systems?
**A:** **Yes.**
Simply disable the default `std` feature in your `Cargo.toml`. The library does not require a heap allocator if you use the `_into` (slice-based) APIs.
```toml
[dependencies]
base64-turbo = { version = "0.1", default-features = false }
```

## 🔌 Compatibility & Ecosystem

### Q: Is the output compatible with the standard `base64` crate?
**A:** **Yes.**
We fully conform to **RFC 4648**.
*   `STANDARD` engine: Matches standard Base64 (output ends with `=`).
*   `URL_SAFE` engine: Matches URL-safe Base64 (uses `-` and `_`).
You can swap `base64-turbo` into any project using standard Base64 without breaking data compatibility.

### Q: Do you support `serde`?
**A:** **Not directly (yet).**
To keep compile times low and dependencies minimal, we do not include `serde` implementations by default. However, you can easily use `base64-turbo` inside a custom `serde` serializer/deserializer.

### Q: Why should I use this over the C library (`turbo-base64`)?
**A:** **Memory Safety.**
The C library is faster (~29 GiB/s vs our ~12 GiB/s), but it is **Unsafe**. It relies on unchecked pointer arithmetic. `base64-turbo` offers a strategic compromise: saturating memory bandwidth while maintaining **100% Rust Memory Safety guarantees**.

### Q: How can I trust this code?
**A:** **Trust the math, not the author.**
1.  Check the **[GitHub Actions](https://github.com/hacer-bark/base64-turbo/actions)** to see the live Kani/MIRI logs.
2.  Inspect the **GPG Signatures** on our commits.
3.  Read the **[Verification Report](./verification.md)** to understand our audit methodology.
