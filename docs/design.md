# 🏗️ Architecture & Design

**Goal:** Saturate Memory Bandwidth.
**Method:** Extreme Hardware Sympathy.

`base64-turbo` is not merely a loop over a lookup table. It is an engine engineered to exploit the micro-architectural mechanics of modern x86 processors. By aligning software logic with hardware capabilities, we achieve maximum **Instruction Level Parallelism (ILP)**.

## Design Philosophy: Logic > Memory

Traditional Base64 implementations rely heavily on byte-by-byte memory lookup tables. This creates two problems for modern CPUs:
1.  **Pipeline Stalls:** Processing one byte at a time creates a tight dependency chain, preventing the CPU from using its superscalar capabilities.
2.  **Branch Misprediction:** Naive loops often incur penalties when handling padding or invalid characters.

**The Turbo Approach:**
We replace byte-level processing with **Vectorized Data Movement**.
*   Instead of asking memory *"What is the character for this byte?"* one by one, we load machine-word sized chunks (64-bit or 256-bit) and process them in parallel.
*   **Benefit:** This maximizes "Instructions Per Cycle" (IPC). The CPU pipeline remains full, processing 8 to 32 bytes simultaneously.

## Scalar Implementation (Wide-LUT)

Even without SIMD, our Scalar fallback is significantly faster than standard implementations.

We utilize a technique known as **Wide-Scalar Processing**.
*   **Data Size:** Instead of processing bytes (`u8`), we cast data to `u32` or `u64` to move 4-8 bytes at a time.
*   **Loop Unrolling:** We manually unroll loops to reduce branch prediction overhead.
*   **Table-Based Lookups:** We utilize 64-bit registers to construct 8 output bytes from 6 input bytes in a single logical block, leveraging the CPU's L1 cache efficiently.
*   **Safety:** This relies on `unsafe` pointer arithmetic and `read_unaligned` calls, but is rigorously bounded and verified by Kani and MSan to never read beyond the allocated slice or leak uninitialized memory.

## AVX2 Implementation

The AVX2 path is the workhorse of this library. It is optimized based on the specific cost of CPU instructions (Cycles vs. Ports).

### 1. Execution Port Saturation
Modern Intel CPUs have multiple "Ports" to execute instructions.
*   **Multiplication:** Runs on Ports 0 & 1 (Cost: ~5 cycles).
*   **Shuffle:** Runs on Port 5 (Cost: ~1 cycle).

We explicitly designed the algorithm to balance the load across these ports. We utilize **Register-based Look-Up Tables (LUTs)** via `vpshufb` to offload work to Port 5. This prevents any single execution port from becoming a bottleneck, maximizing Superscalar throughput.

### 2. The "Lane Stitching" Problem
Standard AVX2 instructions (`vpshufb`) are restricted to 128-bit "lanes." You cannot easily move a byte from the lower 128-bits of a register to the upper 128-bits. This breaks Base64, which requires a sliding bit-window.

**Our Solution:**
We utilize "Double-Load" and permutation intrinsics to bridge this gap. By carefully stitching the lanes together, we utilize the full 32-byte width of the YMM registers without incurring pipeline stalls associated with cross-lane data movement.

## AVX512 Implementation

`base64-turbo` features what is likely the **first pure-Rust AVX512 Base64 implementation**.

This is not an auto-generated port of the AVX2 code. It is hand-written to exploit specific AVX512 features present in Zen 4 and Ice Lake CPUs.

*   **Zero-Cost Masking:** We utilize Mask Registers (`k`) to handle partial data chunks without the overhead of prologue/epilogue code.
*   **Note on Verification:** While the AVX2/Scalar paths are MIRI and MSan verified, the AVX512 path is currently verified via Fuzzing and Kani only, as standard tooling does not yet support these intrinsics. See [Safety & Verification](./verification.md) for details.

## Runtime Dispatch & Fallback

To ensure the library is safe to use on **any** hardware (from embedded ARM to Server-Grade Xeons), we utilize **Dynamic Feature Detection** at runtime.

The library compiles multiple implementation paths into the binary and selects the fastest valid one during the first function call.

**The Priority Chain:**
1.  **AVX512** (If feature enabled + CPU supports it)
2.  **AVX2** (If CPU supports it)
3.  **SSE4.1** (If CPU supports it)
4.  **Scalar** (Universal Fallback)

**Safety Guarantee:**
This detection prevents `SIGILL` (Illegal Instruction) crashes. If you try to run this library on an ARM Raspberry Pi or an old Intel Atom, it will gracefully degrade to the highly-optimized **Scalar** path.

> **Trade-off:** This adds a small increase to binary size (as code for all architectures is included) but ensures universal portability and safety.

## 🏆 Summary of Claims

Due to these architectural decisions and rigorous verification strategies, `base64-turbo` asserts the following positions in the ecosystem:

1.  **The Fastest Base64 Crate in Rust:**
    Outperforming the previous standard (`base64-simd`) by ~2x in decoding throughput.

2.  **The World's Fastest *Memory-Safe* Implementation:**
    While unchecked C libraries (like `turbo-base64`) may achieve higher raw throughput (~29 GiB/s), they lack memory safety guarantees. Among all **memory-safe** implementations (Rust, Java, Go), `base64-turbo` is the fastest.

3.  **First Rust AVX512 Base64:**
    We provide the first known production-ready implementation of Base64 leveraging AVX512 intrinsics in the Rust ecosystem, utilizing mask registers for zero-cost edge handling.

> **Note on Safety:**
> While the architecture relies heavily on `unsafe` intrinsics and raw pointers to achieve this speed, the logic is encapsulated and formally verified. See [Safety & Verification](./verification.md) for details on how we prove this architecture is sound.
