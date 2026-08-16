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
