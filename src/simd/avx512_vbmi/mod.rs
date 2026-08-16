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

// --- Stride constants ---
//
// The Kani index proofs in `verify` reason over this same arithmetic
// symbolically, and import these rather than restating them, so a stride that
// changes here changes the proofs too instead of silently drifting out from
// under them. The derived ones (`*_MIN`) are the point: writing the tier guards
// as "what the tier consumes, plus the read-ahead margin" is what makes the
// scalar tail's slack a consequence of the constants rather than a coincidence
// of three hand-picked literals.

/// Bytes a full-width load reads or a full-width store writes.
const ENC_VEC: usize = 64;
/// Input bytes one encode vector consumes.
const ENC_VEC_IN: usize = 48;
/// Characters one encode vector produces.
const ENC_VEC_OUT: usize = 64;
/// Vectors per iteration of the encoder's quad tier.
const ENC_UNROLL: usize = 4;
/// Input bytes per quad-tier iteration.
const ENC_QUAD_IN: usize = ENC_VEC_IN * ENC_UNROLL;
/// Characters per quad-tier iteration.
const ENC_QUAD_OUT: usize = ENC_VEC_OUT * ENC_UNROLL;
/// Quad-tier guard. The binding requirement is only that the last load
/// (starting 144 bytes in, reading 64) stays in bounds, i.e. 208; this is the
/// output-sized round number above it, and Layer 1 proves it suffices.
const ENC_QUAD_MIN: usize = 256;
/// Single-tier guard: a plain load reads a whole vector to consume 48 of it.
const ENC_SINGLE_MIN: usize = ENC_VEC;
/// Input bytes per Base64 group; the masked tier handles whole groups only.
const ENC_GROUP: usize = 3;

/// Characters one decode vector consumes, which is also its load width.
const DEC_VEC_IN: usize = 64;
/// Bytes one decode vector produces.
const DEC_VEC_OUT: usize = 48;
/// Vectors per iteration of the decoder's quad tier.
const DEC_UNROLL: usize = 4;
/// Characters per quad-tier iteration.
const DEC_QUAD_IN: usize = DEC_VEC_IN * DEC_UNROLL;
/// Bytes per quad-tier iteration.
const DEC_QUAD_OUT: usize = DEC_VEC_OUT * DEC_UNROLL;
/// Characters per Base64 group.
const DEC_GROUP: usize = 4;
/// Characters every decode tier stops short of the end, so that the final
/// group — the only one that may legally carry `'='` — is always decided by the
/// scalar tail, which owns the padding and length rules.
const DEC_LEAD: usize = 4;
/// Quad-tier guard: what it consumes, plus the margin.
const DEC_QUAD_MIN: usize = DEC_QUAD_IN + DEC_LEAD;
/// Single-tier guard: what it consumes, plus the margin.
const DEC_SINGLE_MIN: usize = DEC_VEC_IN + DEC_LEAD;
/// Masked-tier guard: one group, plus the margin.
const DEC_MASKED_MIN: usize = DEC_GROUP + DEC_LEAD;

/// Store mask selecting the low 48 bytes of a decoded vector.
const LOW_48: u64 = (1u64 << DEC_VEC_OUT) - 1;

// ======================================================================
// Miri-compatible VBMI shims
// ======================================================================

#[cfg(miri)]
use self::verify::intrinsic_models as m;

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn zmm_permutexvar_epi8(idx: __m512i, a: __m512i) -> __m512i {
    #[cfg(miri)]
    {
        unsafe { m::permutexvar_epi8_model(idx, a) }
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
        unsafe { m::permutex2var_epi8_model(a, idx, b) }
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
        unsafe { m::multishift_epi64_epi8_model(a, b) }
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
    while rem >= ENC_QUAD_MIN {
        let r0 = encode_vec!(load_48!(0));
        let r1 = encode_vec!(load_48!(ENC_VEC_IN));
        let r2 = encode_vec!(load_48!(2 * ENC_VEC_IN));
        let r3 = encode_vec!(load_48!(3 * ENC_VEC_IN));
        unsafe { _mm512_storeu_si512(dst.cast(), r0) };
        unsafe { _mm512_storeu_si512(dst.add(ENC_VEC_OUT).cast(), r1) };
        unsafe { _mm512_storeu_si512(dst.add(2 * ENC_VEC_OUT).cast(), r2) };
        unsafe { _mm512_storeu_si512(dst.add(3 * ENC_VEC_OUT).cast(), r3) };
        src = unsafe { src.add(ENC_QUAD_IN) };
        dst = unsafe { dst.add(ENC_QUAD_OUT) };
        rem -= ENC_QUAD_IN;
    }

    // Single tier: 48 input bytes -> 64 output. A plain load reads 64 bytes to
    // consume 48, so it needs 64 to exist.
    while rem >= ENC_SINGLE_MIN {
        let r = encode_vec!(load_48!(0));
        unsafe { _mm512_storeu_si512(dst.cast(), r) };
        src = unsafe { src.add(ENC_VEC_IN) };
        dst = unsafe { dst.add(ENC_VEC_OUT) };
        rem -= ENC_VEC_IN;
    }

    // Masked tier: whole triples only, so no padding logic lands here. `rem` is
    // now < 64 and `take` is capped at 48, so this runs at most twice.
    while rem >= ENC_GROUP {
        let take = (rem - rem % ENC_GROUP).min(ENC_VEC_IN);
        let out = take / ENC_GROUP * 4;
        let v = unsafe { _mm512_maskz_loadu_epi8(u64::MAX >> (ENC_VEC - take), src.cast()) };
        let chars = encode_vec!(v);
        unsafe { _mm512_mask_storeu_epi8(dst.cast::<i8>(), u64::MAX >> (ENC_VEC - out), chars) };
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
    while rem >= DEC_QUAD_MIN {
        let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(DEC_VEC_IN).cast::<__m512i>()) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(2 * DEC_VEC_IN).cast::<__m512i>()) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(3 * DEC_VEC_IN).cast::<__m512i>()) };

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
        unsafe { _mm512_storeu_si512(dst.add(DEC_VEC_OUT).cast(), p1) };
        unsafe { _mm512_storeu_si512(dst.add(2 * DEC_VEC_OUT).cast(), p2) };
        unsafe { _mm512_mask_storeu_epi8(dst.add(3 * DEC_VEC_OUT).cast::<i8>(), LOW_48, p3) };

        src = unsafe { src.add(DEC_QUAD_IN) };
        dst = unsafe { dst.add(DEC_QUAD_OUT) };
        rem -= DEC_QUAD_IN;
    }

    // Single tier: 64 input characters -> 48 output bytes.
    while rem >= DEC_SINGLE_MIN {
        let v = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let idx = unsafe { zmm_permutex2var_epi8(lut_lo, v, lut_hi) };
        bad = _mm512_ternarylogic_epi32::<0xFE>(bad, v, idx);
        let p = pack_vec!(idx);
        unsafe { _mm512_mask_storeu_epi8(dst.cast::<i8>(), LOW_48, p) };
        src = unsafe { src.add(DEC_VEC_IN) };
        dst = unsafe { dst.add(DEC_VEC_OUT) };
        rem -= DEC_VEC_IN;
    }

    // Masked tier: the lanes past the end are backfilled with 'A', which decodes
    // to index 0, so they cannot trip validation.
    if rem >= DEC_MASKED_MIN {
        let take = (rem - DEC_LEAD) & !(DEC_GROUP - 1);
        let out = take / DEC_GROUP * 3;
        let v = unsafe {
            _mm512_mask_loadu_epi8(
                _mm512_set1_epi8(b'A'.cast_signed()),
                u64::MAX >> (DEC_VEC_IN - take),
                src.cast(),
            )
        };
        let idx = unsafe { zmm_permutex2var_epi8(lut_lo, v, lut_hi) };
        bad = _mm512_ternarylogic_epi32::<0xFE>(bad, v, idx);
        let p = pack_vec!(idx);
        unsafe { _mm512_mask_storeu_epi8(dst.cast::<i8>(), u64::MAX >> (DEC_VEC_IN - out), p) };
        src = unsafe { src.add(take) };
        dst = unsafe { dst.add(out) };
    }

    if _mm512_movepi8_mask(bad) != 0 {
        return Err(Error::InvalidCharacter);
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

// Verification: Kani proofs, Intel-pseudocode intrinsic models, and the Miri +
// hardware coverage suites.
#[cfg(any(kani, test, miri))]
mod verify;
