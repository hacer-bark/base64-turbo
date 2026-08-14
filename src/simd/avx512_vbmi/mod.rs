//! AVX-512-VBMI Base64. Two VBMI-only instructions carry this path: `vpermb` /
//! `vpermi2b` replace the plain AVX-512F/BW arithmetic character mapping, and
//! `vpmultishiftqb` replaces the encoder's shift/mask chain outright.
//!
//! Both kernels are bound by port 5 (every byte permute issues there), so the
//! design goal is simply to retire fewer ops per vector:
//!
//! * encode: gather -> `vpmultishiftqb` -> alphabet `vpermb` is 3 ops per
//!   48-byte vector, down from 6.
//! * decode: validity is folded into a `vpternlogd` OR tree, one op per vector,
//!   replacing the per-vector compare/movemask/kor trio.
//!
//! Both also run their remainder through masked vector passes rather than
//! handing tens of bytes to the scalar kernel; a masked `vmovdqu8` cannot fault
//! on a masked-off element, so the loops need no read-ahead slack and scalar
//! only ever sees the final partial group.

use crate::{Config, Error};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m512i, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_mask_loadu_epi8,
    _mm512_mask_storeu_epi8, _mm512_maskz_loadu_epi8, _mm512_movepi8_mask, _mm512_set1_epi8,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_storeu_si512, _mm512_ternarylogic_epi32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m512i, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_mask_loadu_epi8,
    _mm512_mask_storeu_epi8, _mm512_maskz_loadu_epi8, _mm512_movepi8_mask, _mm512_set1_epi8,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_si512,
    _mm512_storeu_si512, _mm512_ternarylogic_epi32,
};

#[cfg(all(not(miri), target_arch = "x86"))]
use std::arch::x86::{
    _mm512_multishift_epi64_epi8, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
};
#[cfg(all(not(miri), target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm512_multishift_epi64_epi8, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
};

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

/// `vpermb` control that gathers 48 input bytes into 8 qwords laid out
/// `[b2,b1,b0, b5,b4,b3, x,x]`. That puts one big-endian input triple in each
/// qword's bits 0..23 and the next in bits 24..47, which is what makes all
/// eight 6-bit fields *contiguous* bit runs and so reachable by a single
/// `vpmultishiftqb`. In the natural little-endian byte order they are not:
/// the second index of a triple straddles bits 0..1 and 12..15.
#[allow(clippy::cast_possible_truncation)] // `q * 6 + 5` is at most 47
const fn build_encode_gather() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut q = 0;
    while q < 8 {
        let b = (q * 6) as u8;
        let o = q * 8;
        t[o] = b + 2;
        t[o + 1] = b + 1;
        t[o + 2] = b;
        t[o + 3] = b + 5;
        t[o + 4] = b + 4;
        t[o + 5] = b + 3;
        // Byte 6 only ever feeds the top 2 bits of an extracted index, which
        // `vpermb` discards; byte 7 is never read at all.
        t[o + 6] = b + 3;
        t[o + 7] = b + 3;
        q += 1;
    }
    t
}
const VBMI_ENCODE_GATHER: [u8; 64] = build_encode_gather();

/// `vpmultishiftqb` controls: bit offsets 18/12/6/0 for the low triple's four
/// 6-bit fields and 42/36/30/24 for the high one, in output byte order.
const VBMI_MULTISHIFT: i64 = 0x181E_242A_0006_0C12_u64.cast_signed();

/// `vpmaddubsw` multiplier that folds each index pair into one 12-bit value:
/// `even * 64 + odd`. The 32-byte constant it replaces was a repeating
/// `0x40, 0x01`, which is exactly this broadcast.
const VBMI_PACK_L1: i16 = 0x0140;

/// `vpmaddwd` multiplier that folds each 12-bit pair into one 24-bit value.
const VBMI_PACK_L2: i32 = 0x0001_1000;

/// `vpermb` control that compresses the 16 packed dwords (each holding a
/// big-endian 24-bit triple in its low 3 bytes) into 48 contiguous output
/// bytes. The top 16 lanes are unused.
const VBMI_PACK_SHUFFLE: [i32; 16] = [
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
    0,
    0,
    0,
    0,
];

/// Store mask selecting the low 48 bytes of a decoded vector.
const LOW_48: u64 = 0x0000_FFFF_FFFF_FFFF;

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

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn zmm_multishift_epi64_epi8(a: __m512i, b: __m512i) -> __m512i {
    #[cfg(miri)]
    {
        // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_multishift_epi64_epi8
        let a: [u8; 64] = unsafe { std::mem::transmute(a) };
        let b: [u64; 8] = unsafe { std::mem::transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 7
        for j in 0..8 {
            // FOR k := 0 to 7
            for k in 0..8 {
                // ctrl := a[j][k] & 63
                let ctrl = u32::from(a[j * 8 + k] & 63);
                // dst[j][k] := (b[j] >> ctrl) | (b[j] << (64 - ctrl))
                // (expressed as a rotate, which is what that pair of shifts is
                // and which stays defined when `ctrl` is 0)
                dst[j * 8 + k] = (b[j].rotate_right(ctrl) & 0xFF) as u8;
                // ENDFOR
            }
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { std::mem::transmute(dst) }
    }
    #[cfg(not(miri))]
    {
        _mm512_multishift_epi64_epi8(a, b)
    }
}

// --- VBMI encoder ---

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn encode_slice_avx512_vbmi(config: &Config, input: &[u8], dst_slice: &mut [u8]) {
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;
    let mut rem = input.len();

    let gather = unsafe { _mm512_loadu_si512(VBMI_ENCODE_GATHER.as_ptr().cast()) };
    let shifts = _mm512_set1_epi64(VBMI_MULTISHIFT);

    // Full 64-byte alphabet in one ZMM; vpermb selects by each index's low 6
    // bits, so the garbage in each index's top 2 bits needs no masking.
    let alphabet = if config.url_safe {
        unsafe { _mm512_loadu_si512(VBMI_ENCODE_URL_SAFE.as_ptr().cast()) }
    } else {
        unsafe { _mm512_loadu_si512(VBMI_ENCODE_STANDARD.as_ptr().cast()) }
    };

    /// 48 input bytes in a ZMM -> 64 output characters, in three port-5 ops.
    macro_rules! encode_vec {
        ($v:expr) => {{
            let raw = $v;
            let g = unsafe { zmm_permutexvar_epi8(gather, raw) };
            let indices = unsafe { zmm_multishift_epi64_epi8(shifts, g) };
            unsafe { zmm_permutexvar_epi8(indices, alphabet) }
        }};
    }
    macro_rules! load_48 {
        ($off:expr) => {{ unsafe { _mm512_loadu_si512(src.add($off).cast()) } }};
    }

    // Quad tier: 192 input bytes -> 256 output. The last load starts 144 bytes
    // in and reads 64, so 208 <= 256 bytes are always in bounds.
    while rem >= 256 {
        let r0 = encode_vec!(load_48!(0));
        let r1 = encode_vec!(load_48!(48));
        let r2 = encode_vec!(load_48!(96));
        let r3 = encode_vec!(load_48!(144));
        unsafe { _mm512_storeu_si512(dst.cast(), r0) };
        unsafe { _mm512_storeu_si512(dst.add(64).cast(), r1) };
        unsafe { _mm512_storeu_si512(dst.add(128).cast(), r2) };
        unsafe { _mm512_storeu_si512(dst.add(192).cast(), r3) };
        src = unsafe { src.add(192) };
        dst = unsafe { dst.add(256) };
        rem -= 192;
    }

    // Single tier: 48 input bytes -> 64 output. A plain load reads 64 bytes to
    // consume 48, so it needs 64 to exist.
    while rem >= 64 {
        let r = encode_vec!(load_48!(0));
        unsafe { _mm512_storeu_si512(dst.cast(), r) };
        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
        rem -= 48;
    }

    // Masked tier: whole triples only, so no padding logic lands here. `rem` is
    // now < 64 and `take` is capped at 48, so this runs at most twice.
    while rem >= 3 {
        let take = (rem - rem % 3).min(48);
        let out = take / 3 * 4;
        let v = unsafe { _mm512_maskz_loadu_epi8(u64::MAX >> (64 - take), src.cast()) };
        let chars = encode_vec!(v);
        unsafe { _mm512_mask_storeu_epi8(dst.cast::<i8>(), u64::MAX >> (64 - out), chars) };
        src = unsafe { src.add(take) };
        dst = unsafe { dst.add(out) };
        rem -= take;
    }

    // Scalar now sees at most the final 1-2 bytes, plus whatever padding the
    // config asks for.
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
    let mut src = input.as_ptr();
    let dst_start = dst_slice.as_mut_ptr();
    let mut dst = dst_start;
    let mut rem = input.len();

    // 128-byte reverse LUT across two ZMMs; vpermi2b picks the register by bit
    // 6 and the byte by the low 6 bits, covering ASCII 0-127 in one lookup.
    let lut = if config.url_safe {
        &VBMI_DECODE_URL_SAFE
    } else {
        &VBMI_DECODE_STANDARD
    };
    let lut_lo = unsafe { _mm512_loadu_si512(lut.as_ptr().cast()) };
    let lut_hi = unsafe { _mm512_loadu_si512(lut.as_ptr().add(64).cast()) };

    let pack_l1 = _mm512_set1_epi16(VBMI_PACK_L1);
    let pack_l2 = _mm512_set1_epi32(VBMI_PACK_L2);
    let pack = unsafe { _mm512_loadu_si512(VBMI_PACK_SHUFFLE.as_ptr().cast()) };

    // A character is bad iff its input byte had bit 7 set (>= 0x80, which
    // vpermi2b silently aliases into the 128-entry table) or the LUT answered
    // with the 0xFF sentinel. Either way bit 7 of `input | index` is set, so
    // OR-ing every input and every index into one accumulator and testing its
    // sign bits once validates the whole buffer. That is one `vpternlogd` per
    // vector in place of a compare, a movemask and a mask-OR.
    let mut bad = _mm512_setzero_si512();

    macro_rules! pack_vec {
        ($idx:expr) => {{
            let m = _mm512_maddubs_epi16($idx, pack_l1);
            let p = _mm512_madd_epi16(m, pack_l2);
            unsafe { zmm_permutexvar_epi8(pack, p) }
        }};
    }

    // Quad tier: 256 input characters -> 192 output bytes. Every tier stops at
    // least 4 characters short of the end so the final group -- the only one
    // that may legally carry '=' -- is always decided by the scalar tail, which
    // owns the padding and length rules.
    while rem >= 260 {
        let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(64).cast::<__m512i>()) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(128).cast::<__m512i>()) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(192).cast::<__m512i>()) };

        let i0 = unsafe { zmm_permutex2var_epi8(lut_lo, v0, lut_hi) };
        let i1 = unsafe { zmm_permutex2var_epi8(lut_lo, v1, lut_hi) };
        let i2 = unsafe { zmm_permutex2var_epi8(lut_lo, v2, lut_hi) };
        let i3 = unsafe { zmm_permutex2var_epi8(lut_lo, v3, lut_hi) };

        let p0 = pack_vec!(i0);
        let p1 = pack_vec!(i1);
        let p2 = pack_vec!(i2);
        let p3 = pack_vec!(i3);

        // 0xFE is the 3-input OR; four of them fold all eight vectors in.
        let t0 = _mm512_ternarylogic_epi32::<0xFE>(v0, i0, v1);
        let t1 = _mm512_ternarylogic_epi32::<0xFE>(i1, v2, i2);
        let t2 = _mm512_ternarylogic_epi32::<0xFE>(v3, i3, t0);
        bad = _mm512_ternarylogic_epi32::<0xFE>(bad, t1, t2);

        // Only the last store needs masking: each of the first three overhangs
        // its 48 bytes by 16, and the very next store in this same iteration
        // rewrites exactly that overhang.
        unsafe { _mm512_storeu_si512(dst.cast(), p0) };
        unsafe { _mm512_storeu_si512(dst.add(48).cast(), p1) };
        unsafe { _mm512_storeu_si512(dst.add(96).cast(), p2) };
        unsafe { _mm512_mask_storeu_epi8(dst.add(144).cast::<i8>(), LOW_48, p3) };

        src = unsafe { src.add(256) };
        dst = unsafe { dst.add(192) };
        rem -= 256;
    }

    // Single tier: 64 input characters -> 48 output bytes.
    while rem >= 68 {
        let v = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let idx = unsafe { zmm_permutex2var_epi8(lut_lo, v, lut_hi) };
        bad = _mm512_ternarylogic_epi32::<0xFE>(bad, v, idx);
        let p = pack_vec!(idx);
        unsafe { _mm512_mask_storeu_epi8(dst.cast::<i8>(), LOW_48, p) };
        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
        rem -= 64;
    }

    // Masked tier: the lanes past the end are backfilled with 'A', which decodes
    // to index 0, so they cannot trip validation.
    if rem >= 8 {
        let take = (rem - 4) & !3;
        let out = take / 4 * 3;
        let v = unsafe {
            _mm512_mask_loadu_epi8(
                _mm512_set1_epi8(b'A'.cast_signed()),
                u64::MAX >> (64 - take),
                src.cast(),
            )
        };
        let idx = unsafe { zmm_permutex2var_epi8(lut_lo, v, lut_hi) };
        bad = _mm512_ternarylogic_epi32::<0xFE>(bad, v, idx);
        let p = pack_vec!(idx);
        unsafe { _mm512_mask_storeu_epi8(dst.cast::<i8>(), u64::MAX >> (64 - out), p) };
        src = unsafe { src.add(take) };
        dst = unsafe { dst.add(out) };
    }

    if _mm512_movepi8_mask(bad) != 0 {
        return Err(Error::InvalidCharacter);
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

// Verification: Miri + hardware coverage suites.
#[cfg(test)]
mod verify;
