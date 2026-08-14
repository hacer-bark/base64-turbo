//! AVX-512-VBMI Base64: `vpermb`/`vpermi2b` table lookups replace the plain
//! AVX-512F/BW path's arithmetic character mapping. The 6-bit index extraction
//! is shared in spirit with [`super::avx512`]; only the alphabet/reverse-LUT
//! lookups differ.

use super::{PACK_L1, PACK_L2};
use crate::{Config, Error};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m512i, _kor_mask64, _mm_loadu_si128, _mm512_and_si512, _mm512_broadcast_i32x4,
    _mm512_cmpeq_epi8_mask, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16,
    _mm512_mask_storeu_epi8, _mm512_movepi8_mask, _mm512_set1_epi8, _mm512_set1_epi32,
    _mm512_setr_epi32, _mm512_sllv_epi16, _mm512_srlv_epi16, _mm512_storeu_si512,
    _mm512_ternarylogic_epi32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m512i, _kor_mask64, _mm_loadu_si128, _mm512_and_si512, _mm512_broadcast_i32x4,
    _mm512_cmpeq_epi8_mask, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16,
    _mm512_mask_storeu_epi8, _mm512_movepi8_mask, _mm512_set1_epi8, _mm512_set1_epi32,
    _mm512_setr_epi32, _mm512_sllv_epi16, _mm512_srlv_epi16, _mm512_storeu_si512,
    _mm512_ternarylogic_epi32,
};

#[cfg(all(not(miri), target_arch = "x86"))]
use std::arch::x86::{_mm512_permutex2var_epi8, _mm512_permutexvar_epi8};
#[cfg(all(not(miri), target_arch = "x86_64"))]
use std::arch::x86_64::{_mm512_permutex2var_epi8, _mm512_permutexvar_epi8};

// --- Compile-time lookup tables ---

/// Base64 alphabet for the `vpermb` encoder lookup.
const VBMI_ENCODE_STANDARD: [u8; 64] =
    *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const VBMI_ENCODE_URL_SAFE: [u8; 64] =
    *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// 128-byte reverse lookup for the `vpermi2b` decoder: ASCII 0-127 -> 6-bit
/// index, `0xFF` for invalid.
const VBMI_DECODE_STANDARD: [u8; 128] = build_decode_lut(&VBMI_ENCODE_STANDARD);
const VBMI_DECODE_URL_SAFE: [u8; 128] = build_decode_lut(&VBMI_ENCODE_URL_SAFE);

#[allow(clippy::cast_possible_truncation)] // `i` is always < 64, fits in u8
const fn build_decode_lut(alphabet: &[u8; 64]) -> [u8; 128] {
    let mut t = [0xFFu8; 128];
    let mut i = 0;
    while i < 64 {
        t[alphabet[i] as usize] = i as u8;
        i += 1;
    }
    t
}

// ======================================================================
// Miri-compatible VBMI shims
// ======================================================================

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn zmm_permutexvar_epi8(idx: __m512i, a: __m512i) -> __m512i {
    #[cfg(miri)]
    {
        // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutexvar_epi8
        let idx: [u8; 64] = unsafe { std::mem::transmute(idx) };
        let a: [u8; 64] = unsafe { std::mem::transmute(a) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            // (In Rust we access bytes 'j' so '*8' offset is not needed)
            let i = j;

            // id := idx[i+5:i]*8
            // (In Rust we index byte-wise, so no additional *8 byte-offset is needed)
            let id = usize::from(idx[i] & 0x3F);
            // dst[i+7:i] := a[id+7:id]
            dst[i] = a[id];
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { std::mem::transmute(dst) }
    }
    #[cfg(not(miri))]
    {
        _mm512_permutexvar_epi8(idx, a)
    }
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn zmm_permutex2var_epi8(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    #[cfg(miri)]
    {
        // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutex2var_epi8
        let a: [u8; 64] = unsafe { std::mem::transmute(a) };
        let idx: [u8; 64] = unsafe { std::mem::transmute(idx) };
        let b: [u8; 64] = unsafe { std::mem::transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            let i = j;

            // off := idx[i+5:i]*8
            let off = usize::from(idx[i] & 0x3F);
            // IF idx[i+6]
            if (idx[i] & 0x40) != 0 {
                // dst[i+7:i] := b[off+7:off]
                dst[i] = b[off];
            // ELSE
            } else {
                // dst[i+7:i] := a[off+7:off]
                dst[i] = a[off];
                // FI
            }
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { std::mem::transmute(dst) }
    }
    #[cfg(not(miri))]
    {
        _mm512_permutex2var_epi8(a, idx, b)
    }
}

// --- VBMI encoder ---

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn encode_slice_avx512_vbmi(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    // One vpermb byte-reorder from a raw 64-byte load, replacing the plain
    // path's permutexvar_epi32 + shuffle_epi8 (verified equal byte-for-byte).
    let shuffle_input = _mm512_setr_epi32(
        0x0102_0001,
        0x0405_0304,
        0x0708_0607,
        0x0a0b_090a,
        0x0d0e_0c0d,
        0x1011_0f10,
        0x1314_1213,
        0x1617_1516,
        0x191a_1819,
        0x1c1d_1b1c,
        0x1f20_1e1f,
        0x2223_2122,
        0x2526_2425,
        0x2829_2728,
        0x2b2c_2a2b,
        0x2e2f_2d2e,
    );

    // Full 64-byte alphabet in one ZMM; vpermb selects by each index's low 6 bits.
    let alphabet = if config.url_safe {
        unsafe { _mm512_loadu_si512(VBMI_ENCODE_URL_SAFE.as_ptr().cast()) }
    } else {
        unsafe { _mm512_loadu_si512(VBMI_ENCODE_STANDARD.as_ptr().cast()) }
    };

    macro_rules! encode_vec_vbmi {
        ($in_vec:expr) => {{
            // 3 bytes -> 4 six-bit indices via AVX-512F/BW variable shifts.
            let t0 = _mm512_and_si512($in_vec, _mm512_set1_epi32(0x0fc0_fc00));
            let t1 = _mm512_srlv_epi16(t0, _mm512_set1_epi32(0x0006_000a));
            let t2 = _mm512_sllv_epi16($in_vec, _mm512_set1_epi32(0x0008_0004));
            let indices = _mm512_ternarylogic_epi32::<0xca>(_mm512_set1_epi32(0x3f00_3f00), t2, t1);

            // One vpermb maps all indices to characters (replaces 8 instrs).
            unsafe { zmm_permutexvar_epi8(indices, alphabet) }
        }};
    }

    macro_rules! load_48_bytes {
        ($ptr:expr) => {{
            let v = unsafe { _mm512_loadu_si512($ptr.cast()) };
            unsafe { zmm_permutexvar_epi8(shuffle_input, v) }
        }};
    }

    // Quad tier: 192 input bytes -> 256 output.
    let safe_len_192 = len.saturating_sub(16);
    let aligned_len_192 = safe_len_192 - (safe_len_192 % 192);
    let src_end_192 = unsafe { src.add(aligned_len_192) };

    while src < src_end_192 {
        let v0 = load_48_bytes!(src);
        let v1 = load_48_bytes!(src.add(48));
        let v2 = load_48_bytes!(src.add(96));
        let v3 = load_48_bytes!(src.add(144));

        let i0 = encode_vec_vbmi!(v0);
        let i1 = encode_vec_vbmi!(v1);
        let i2 = encode_vec_vbmi!(v2);
        let i3 = encode_vec_vbmi!(v3);

        unsafe { _mm512_storeu_si512(dst.cast(), i0) };
        unsafe { _mm512_storeu_si512(dst.add(64).cast(), i1) };
        unsafe { _mm512_storeu_si512(dst.add(128).cast(), i2) };
        unsafe { _mm512_storeu_si512(dst.add(192).cast(), i3) };

        src = unsafe { src.add(192) };
        dst = unsafe { dst.add(256) };
    }

    // Single tier: 48 input bytes -> 64 output.
    let safe_len_single = len.saturating_sub(16);
    let aligned_len_single = safe_len_single - (safe_len_single % 48);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let v = load_48_bytes!(src);
        let res = encode_vec_vbmi!(v);
        unsafe { _mm512_storeu_si512(dst.cast(), res) };

        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::encode(config, input, src, dst_slice, dst_off) };
}

// --- VBMI decoder ---

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn decode_slice_avx512_vbmi(
    config: &Config,
    input: &[u8],
    dst_slice: &mut [u8],
) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;

    // 128-byte reverse LUT across two ZMMs; vpermi2b picks the register by bit
    // 6 and the byte by the low 6 bits, covering ASCII 0-127 in one lookup.
    let lut = if config.url_safe {
        &VBMI_DECODE_URL_SAFE
    } else {
        &VBMI_DECODE_STANDARD
    };
    let lut_lo = unsafe { _mm512_loadu_si512(lut.as_ptr().cast()) };
    let lut_hi = unsafe { _mm512_loadu_si512(lut.as_ptr().add(64).cast()) };

    let invalid = _mm512_set1_epi8(-1); // 0xFF sentinel
    let pack_l1 =
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_L1.as_ptr().cast::<__m128i>())) };
    let pack_l2 =
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_L2.as_ptr().cast::<__m128i>())) };
    let pack = _mm512_setr_epi32(
        0x0600_0102,
        0x090a_0405,
        0x0c0d_0e08,
        0x1610_1112,
        0x191a_1415,
        0x1c1d_1e18,
        0x2620_2122,
        0x292a_2425,
        0x2c2d_2e28,
        0x3630_3132,
        0x393a_3435,
        0x3c3d_3e38,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
    );

    // Validate + decode one vector via the 128-byte reverse LUT.
    macro_rules! decode_vec_vbmi {
        ($input:expr) => {{
            // One vpermi2b maps every byte through the reverse LUT.
            let indices = unsafe { zmm_permutex2var_epi8(lut_lo, $input, lut_hi) };

            // Invalid iff the LUT returned the 0xFF sentinel, or the byte was
            // >= 128 (bit 7 set) and vpermi2b aliased it into the table.
            let is_invalid = _mm512_cmpeq_epi8_mask(indices, invalid);
            let is_high_bit = _mm512_movepi8_mask($input);
            let err_mask = _kor_mask64(is_invalid, is_high_bit);

            (indices, err_mask)
        }};
    }

    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            let m = _mm512_maddubs_epi16($indices, pack_l1);
            let p = _mm512_madd_epi16(m, pack_l2);
            let packed = unsafe { zmm_permutexvar_epi8(pack, p) };

            // Only the low 48 bytes are real output; the mask suppresses the
            // 16 high bytes so an exactly-sized buffer isn't overrun.
            unsafe {
                _mm512_mask_storeu_epi8($dst_ptr.cast::<i8>(), 0x0000_FFFF_FFFF_FFFF, packed)
            };
        }};
    }

    // Quad tier: 256 input bytes -> 192 output.
    let safe_len_256 = len.saturating_sub(4);
    let aligned_len_256 = safe_len_256 - (safe_len_256 % 256);
    let src_end_256 = unsafe { src.add(aligned_len_256) };

    while src < src_end_256 {
        let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(64).cast::<__m512i>()) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(128).cast::<__m512i>()) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(192).cast::<__m512i>()) };

        let (i0, e0) = decode_vec_vbmi!(v0);
        let (i1, e1) = decode_vec_vbmi!(v1);
        let (i2, e2) = decode_vec_vbmi!(v2);
        let (i3, e3) = decode_vec_vbmi!(v3);

        if (e0 | e1 | e2 | e3) != 0 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(i0, dst);
        pack_and_store!(i1, dst.add(48));
        pack_and_store!(i2, dst.add(96));
        pack_and_store!(i3, dst.add(144));

        src = unsafe { src.add(256) };
        dst = unsafe { dst.add(192) };
    }

    // Single tier: 64 input bytes -> 48 output.
    let safe_len_64 = len.saturating_sub(4);
    let aligned_len_64 = safe_len_64 - (safe_len_64 % 64);
    let src_end_64 = unsafe { input.as_ptr().add(aligned_len_64) };

    while src < src_end_64 {
        let v = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let (idx, err_mask) = decode_vec_vbmi!(v);

        if err_mask != 0 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(idx, dst);

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

// Verification: Miri + hardware coverage suites.
#[cfg(test)]
mod verify;
