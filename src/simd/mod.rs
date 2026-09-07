// `x86_simd` (from build.rs) is already "x86 with some AVX kernel", so each arm
// only needs to add its own feature.
#[cfg(all(x86_simd, feature = "avx2"))]
mod avx2;
#[cfg(all(x86_simd, feature = "avx512-vbmi"))]
mod avx512_vbmi;

#[cfg(all(x86_simd, feature = "avx2"))]
pub(crate) use avx2::{decode_slice_avx2, encode_slice_avx2};
#[cfg(all(x86_simd, feature = "avx512-vbmi"))]
pub(crate) use avx512_vbmi::{decode_slice_avx512_vbmi, encode_slice_avx512_vbmi};

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) use neon::{decode_slice_neon, encode_slice_neon};

#[cfg(test)]
mod testutil;

// The naive oracle the Kani kernel proofs check the vector kernels against.
// Compiled under `test` as well so the ordinary build still type-checks it and
// its own differential test can pin it to the scalar kernel. Under Miri that
// differential test is skipped -- it is a pure-logic check with nothing for Miri
// to find -- and it is the oracle's only non-Kani caller, so the module comes out
// with it rather than being dead code.
#[cfg(any(kani, all(test, not(miri))))]
mod refcodec;

/// Fraction of the last-level cache above which streaming stores start paying,
/// as `NONTEMPORAL_LLC_RATIO / NONTEMPORAL_LLC_DIV`.
///
/// Whether `vmovntdq` helps depends on something a kernel cannot see: whether
/// the destination was going to stay in cache. Both cases were measured on
/// Zen 5 (c8a.large, 8 MiB L3), racing a kernel against a twin with the tier
/// disabled, each variant reading its own slot of a 640 MiB pool so neither
/// inherits a buffer the other just warmed:
///
/// * **Cold destination** — a large payload streamed through once. Streaming
///   wins at every size from 512 KiB up, +11% to +24%, and never loses: the
///   output is never read again, so read-for-ownership traffic is pure waste.
/// * **Resident destination** — one buffer encoded over and over, which is what
///   a working set that fits in cache looks like. Streaming *loses* heavily
///   below ~0.6x LLC — -39% (encode) and -36% (decode) at 1 MiB — because it
///   throws away a destination that would have stayed in L2/L3, and wins above
///   it, rising to +22% at 64 MiB.
///
/// The two agree above the crossover and conflict below it, so the gate goes
/// where the resident case turns: 5/8 of the last-level cache. Below it the
/// resident case decides, because its penalty is the larger of the two — -39%
/// against the -13% the cold case pays for not streaming.
///
/// This is a ratio and not a length because the crossover tracks the cache
/// rather than the machine. Both x86 kernels share it; what differs is the
/// floor each passes to [`nontemporal_min`].
#[cfg(all(x86_simd, not(any(miri, kani))))]
const NONTEMPORAL_LLC_RATIO: usize = 5;
#[cfg(all(x86_simd, not(any(miri, kani))))]
const NONTEMPORAL_LLC_DIV: usize = 8;

/// Input length at which a kernel may switch its top tier to non-temporal
/// stores: 5/8 of the last-level cache, never below `floor`.
///
/// Both kernels write more than they read, so once the working set no longer
/// fits in cache most of the cost is read-for-ownership traffic on a
/// destination whose old contents are dead, and `vmovntdq` skips it. Below that
/// point the same instruction is a large *loss* — see
/// [`NONTEMPORAL_LLC_RATIO`].
///
/// `floor` is the caller's own lower bound, and it is a floor rather than the
/// whole threshold: each kernel's floor carries preconditions its proofs and
/// tests are stated against (AVX-512's alignment peel needs a whole quad step
/// after it; AVX2's hardware test enters the tier at exactly its floor), so
/// scaling may raise the gate but never lower it below what those assume.
#[cfg(x86_simd)]
#[inline]
fn nontemporal_min(floor: usize) -> usize {
    // Neither Miri nor Kani runs `cpuid`, and neither reaches a real threshold
    // — Miri's suites are tiny, Kani's proofs are symbolic and assume only
    // `rem >= floor`. Returning the floor is what keeps the streaming tier
    // reachable under Miri and the proofs' gate exactly the one they state.
    #[cfg(any(miri, kani))]
    {
        floor
    }
    #[cfg(not(any(miri, kani)))]
    {
        // Resolved once inside `cpu::llc_bytes`; what is left here is a load
        // and a multiply. With no cache size to scale against, the floor is the
        // threshold.
        let Some(llc) = crate::cpu::llc_bytes() else {
            return floor;
        };
        (llc / NONTEMPORAL_LLC_DIV)
            .saturating_mul(NONTEMPORAL_LLC_RATIO)
            .max(floor)
    }
}

/// Shared SIMD -> scalar handoff. Each backend runs its vectorized loops, then
/// calls these with the pointer/offset state they left off at; `src` points at
/// the first unconsumed input byte and `dst_off` is how many bytes the loops
/// already wrote.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
mod tail {
    use crate::{Config, Error, scalar};

    /// # Safety
    /// `src` must point within `input`.
    pub(super) unsafe fn encode(
        config: &Config,
        input: &[u8],
        src: *const u8,
        dst: &mut [u8],
        dst_off: usize,
    ) {
        let done = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
        if done < input.len() {
            scalar::encode_slice(config, &input[done..], &mut dst[dst_off..]);
        }
    }

    /// Returns the total bytes written (`dst_off` plus the scalar remainder).
    ///
    /// # Safety
    /// `src` must point within `input`.
    pub(super) unsafe fn decode(
        config: &Config,
        input: &[u8],
        src: *const u8,
        dst: &mut [u8],
        dst_off: usize,
    ) -> Result<usize, Error> {
        let done = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
        if done < input.len() {
            Ok(dst_off + scalar::decode_slice(config, &input[done..], &mut dst[dst_off..])?)
        } else {
            Ok(dst_off)
        }
    }
}

#[cfg(all(x86_simd, feature = "avx2"))]
const PACK_L1: [i8; 32] = [
    0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01,
    0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01,
];

#[cfg(all(x86_simd, feature = "avx2"))]
const PACK_L2: [i16; 16] = [
    0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001,
    0x1000, 0x0001, 0x1000, 0x0001,
];

// These are used by the AVX2 packer; the VBMI kernel builds its multipliers
// from immediates and does its own permute, so all three are absent from a
// VBMI-only build.
#[cfg(all(x86_simd, feature = "avx2"))]
const PACK_SHUFFLE: [i8; 32] = [
    2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1, 2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12,
    -1, -1, -1, -1,
];
