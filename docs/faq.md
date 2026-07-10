# Frequently Asked Questions

## Safety & Verification

### Q: The crate uses `unsafe`. How can you claim it is safe?
**A:** We distinguish between "Safe Rust" (compiler-checked) and "memory safe" (Kani-proven and MIRI-checked). For the verified paths (see the matrix in [Safety & Verification](./verification.md)), Kani mathematically proves and MIRI's UB checker confirms that no input via the public API can trigger a buffer overflow, segfault, or panic.

### Q: Can I crash the library by passing garbage data?
**A:** No. The decoder is resilient: invalid Base64 strings, random binary noise, or malicious payloads simply return a `Result::Err`. It will never panic or cause Undefined Behavior (UB) as long as you use the Safe API.

### Q: What happens if I violate safety contracts in the internal `unsafe` API?
**A:** You are responsible for the resulting crash. The `unsafe` internal functions (exposed via the `unstable` feature) are raw tools for bypassing bounds checks when every cycle matters. If you pass a null pointer or an invalid length to these functions, you violate their contract. We verify that *our* Safe API never violates these contracts, but we cannot protect you if you call the unsafe internals directly.

### Q: Is AVX512 enabled by default?
**A:** Yes. We detect CPU features at runtime: if your CPU supports AVX512, we use it; otherwise we fall back to AVX2 or Scalar. No feature flags are required to get SIMD acceleration.

## Performance & Usage

### Q: Does this work on ARM (Apple Silicon / Raspberry Pi)?
**A:** Yes, with native NEON acceleration.
*   **x86_64:** Automatically uses AVX512 / AVX2.
*   **ARM (aarch64):** Uses NEON SIMD instructions (compile-time dispatch, no runtime detection needed).
*   **Other:** Falls back to our optimized Scalar implementation.

### Q: How do I calculate the buffer size for `encode_into`?
**A:** Use the helper functions rather than guessing, particularly if you are avoiding allocations.
```rust
// For Encoding:
let needed = STANDARD.encoded_size(input.len());
let mut buf = vec![0u8; needed];

// For Decoding:
let max_needed = STANDARD.estimate_decoded_len(input.len());
let mut buf = vec![0u8; max_needed];
```

### Q: Is the Scalar fallback slow?
**A:** No, it is highly optimized. Even without SIMD, our scalar implementation eliminates many bounds checks found in the standard library. It won't reach AVX512 throughput, but it is competitive with other standard Base64 implementations on architectures without vector support.

### Q: Does this work on `no_std` / Embedded systems?
**A:** Yes. Disable the default `std` feature in your `Cargo.toml`. The library does not require a heap allocator if you use the `_into` (slice-based) APIs.
```toml
[dependencies]
base64-turbo = { version = "0.2", default-features = false }
```

## Compatibility & Ecosystem

### Q: Is the output compatible with the standard `base64` crate?
**A:** Yes. We fully conform to RFC 4648.
*   `STANDARD` engine: Matches standard Base64 (output ends with `=`).
*   `URL_SAFE` engine: Matches URL-safe Base64 (uses `-` and `_`).
You can swap `base64-turbo` into any project using standard Base64 without breaking data compatibility.

### Q: Do you support `serde`?
**A:** Not directly, yet. To keep compile times low and dependencies minimal, we do not include `serde` implementations by default. You can use `base64-turbo` inside a custom `serde` serializer/deserializer.

### Q: Why should I use this over the C library (`turbo-base64`)?
**A:** Formally verified memory safety, at a moderate speed cost. The C library is faster (~29 GiB/s vs our ~12-20 GiB/s) and relies on unchecked pointer arithmetic with no published safety audits. `base64-turbo` trades some of that throughput for `unsafe` paths that are Kani-proven and MIRI-checked (see [Safety & Verification](./verification.md) for exactly what that covers).

### Q: How can I trust this code?
**A:** Don't take our word for it — check the evidence directly.
1.  Check **[GitHub Actions](https://github.com/hacer-bark/base64-turbo/actions)** for the live Kani/MIRI/fuzzing CI logs.
2.  Read the **[Verification Report](./verification.md)** to see exactly which architectures are Kani-proven versus MIRI-only, and what each check actually covers.
3.  Read the `unsafe` code yourself — every `unsafe` block is documented with the safety contract it relies on.
