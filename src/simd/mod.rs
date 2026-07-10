#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use avx2::{decode_slice_avx2, encode_slice_avx2};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use avx512::{
    decode_slice_avx512, decode_slice_avx512_vbmi, encode_slice_avx512, encode_slice_avx512_vbmi,
};

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
pub(crate) use neon::{decode_slice_neon, encode_slice_neon};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const PACK_L1: [i8; 32] = [
    0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01,
    0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01,
];

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const PACK_L2: [i16; 16] = [
    0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001, 0x1000, 0x0001,
    0x1000, 0x0001, 0x1000, 0x0001,
];

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const PACK_SHUFFLE: [i8; 32] = [
    2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1, 2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12,
    -1, -1, -1, -1,
];
