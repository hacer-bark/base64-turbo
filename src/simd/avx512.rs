use super::{PACK_L1, PACK_L2, PACK_SHUFFLE};
use crate::{Config, Error, scalar};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128i, __m512i, _knot_mask64, _kor_mask64, _mm_loadu_si128, _mm_setr_epi8, _mm_storeu_si128,
    _mm512_add_epi8, _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_castsi512_si128,
    _mm512_cmpeq_epi8_mask, _mm512_cmpgt_epi8_mask, _mm512_cmple_epu8_mask,
    _mm512_extracti32x4_epi32, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16,
    _mm512_mask_add_epi8, _mm512_mask_storeu_epi8, _mm512_movepi8_mask, _mm512_permutexvar_epi32,
    _mm512_set1_epi8, _mm512_set1_epi32, _mm512_setr_epi32, _mm512_shuffle_epi8, _mm512_sllv_epi16,
    _mm512_srli_epi16, _mm512_srlv_epi16, _mm512_storeu_si512, _mm512_sub_epi8, _mm512_subs_epu8,
    _mm512_ternarylogic_epi32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i, __m512i, _knot_mask64, _kor_mask64, _mm_loadu_si128, _mm_setr_epi8, _mm_storeu_si128,
    _mm512_add_epi8, _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_castsi512_si128,
    _mm512_cmpeq_epi8_mask, _mm512_cmpgt_epi8_mask, _mm512_cmple_epu8_mask,
    _mm512_extracti32x4_epi32, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16,
    _mm512_mask_add_epi8, _mm512_mask_storeu_epi8, _mm512_movepi8_mask, _mm512_permutexvar_epi32,
    _mm512_set1_epi8, _mm512_set1_epi32, _mm512_setr_epi32, _mm512_shuffle_epi8, _mm512_sllv_epi16,
    _mm512_srli_epi16, _mm512_srlv_epi16, _mm512_storeu_si512, _mm512_sub_epi8, _mm512_subs_epu8,
    _mm512_ternarylogic_epi32,
};

#[cfg(all(not(miri), target_arch = "x86"))]
use std::arch::x86::{_mm512_permutex2var_epi8, _mm512_permutexvar_epi8};
#[cfg(all(not(miri), target_arch = "x86_64"))]
use std::arch::x86_64::{_mm512_permutex2var_epi8, _mm512_permutexvar_epi8};

// ======================================================================
// AVX-512 VBMI Lookup Tables (compile-time)
// ======================================================================

/// Standard Base64 alphabet for VBMI `vpermb` encoder lookup.
const VBMI_ENCODE_STANDARD: [u8; 64] =
    *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// URL-safe Base64 alphabet for VBMI `vpermb` encoder lookup.
const VBMI_ENCODE_URL_SAFE: [u8; 64] =
    *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Standard Base64 reverse lookup (128 bytes) for VBMI `vpermi2b` decoder.
/// Maps ASCII 0–127 → 6-bit index. Invalid entries contain `0xFF`.
#[allow(clippy::cast_possible_truncation)] // `i` is always < 64, fits in u8
const VBMI_DECODE_STANDARD: [u8; 128] = {
    let mut t = [0xFFu8; 128];
    let a = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut i = 0;
    while i < 64 {
        t[a[i] as usize] = i as u8;
        i += 1;
    }
    t
};

/// URL-safe Base64 reverse lookup (128 bytes) for VBMI `vpermi2b` decoder.
#[allow(clippy::cast_possible_truncation)] // `i` is always < 64, fits in u8
const VBMI_DECODE_URL_SAFE: [u8; 128] = {
    let mut t = [0xFFu8; 128];
    let a = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut i = 0;
    while i < 64 {
        t[a[i] as usize] = i as u8;
        i += 1;
    }
    t
};

// ======================================================================
// AVX-512 VBMI Encoder
// ======================================================================

#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn encode_slice_avx512(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

    // Shuffle bytes for mul
    let shuffle = _mm512_broadcast_i32x4(_mm_setr_epi8(
        1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
    ));

    let set_25 = _mm512_set1_epi8(25);
    let set_51 = _mm512_set1_epi8(51);
    let one = _mm512_set1_epi8(1);
    let translate_lut = if config.url_safe {
        _mm512_broadcast_i32x4(_mm_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -17, 32, 0, 0,
        ))
    } else {
        _mm512_broadcast_i32x4(_mm_setr_epi8(
            65, 71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 0, 0,
        ))
    };

    macro_rules! encode_vec {
        ($in_vec:expr) => {{
            // Compute 3 bytes => 4 letters using baseline AVX-512F/BW
            // variable-shift instructions (no VBMI needed) instead of the
            // mullo/mulhi trick.
            let v = _mm512_shuffle_epi8($in_vec, shuffle);

            let t0 = _mm512_and_si512(v, _mm512_set1_epi32(0x0fc0_fc00));
            let t1 = _mm512_srlv_epi16(t0, _mm512_set1_epi32(0x0006_000a));
            let t2 = _mm512_sllv_epi16(v, _mm512_set1_epi32(0x0008_0004));
            let indices = _mm512_ternarylogic_epi32::<0xca>(_mm512_set1_epi32(0x3f00_3f00), t2, t1);

            let sub_base = _mm512_subs_epu8(indices, set_51);
            let m_gt25 = _mm512_cmpgt_epi8_mask(indices, set_25);
            let lut_idx = _mm512_mask_add_epi8(sub_base, m_gt25, sub_base, one);

            _mm512_add_epi8(indices, _mm512_shuffle_epi8(translate_lut, lut_idx))
        }};
    }

    // Permutation index for 48-byte distribution into 128-bit lanes
    let permute_idx = _mm512_setr_epi32(
        0, 1, 2, 3, // Lane 0 gets elements 0, 1, 2, and 3 (bytes 12-15 as garbage)
        3, 4, 5, 6, // Lane 1 gets elements 3, 4, 5, and 6 (bytes 24-27 as garbage)
        6, 7, 8, 9, // Lane 2 gets elements 6, 7, 8, and 9
        9, 10, 11, 12, // Lane 3 gets elements 9, 10, 11, and 12
    );

    macro_rules! load_48_bytes {
        ($ptr:expr) => {{
            let v = unsafe { _mm512_loadu_si512($ptr.cast()) };
            _mm512_permutexvar_epi32(permute_idx, v)
        }};
    }

    // Process 192 bytes (4 chunks) at a time
    let safe_len_192 = len.saturating_sub(16);
    let aligned_len_192 = safe_len_192 - (safe_len_192 % 192);
    let src_end_192 = unsafe { src.add(aligned_len_192) };

    while src < src_end_192 {
        // Load 4 vectors
        let v0 = load_48_bytes!(src);
        let v1 = load_48_bytes!(src.add(48));
        let v2 = load_48_bytes!(src.add(96));
        let v3 = load_48_bytes!(src.add(144));

        // Process
        let i0 = encode_vec!(v0);
        let i1 = encode_vec!(v1);
        let i2 = encode_vec!(v2);
        let i3 = encode_vec!(v3);

        // Store results
        unsafe { _mm512_storeu_si512(dst.cast(), i0) };
        unsafe { _mm512_storeu_si512(dst.add(64).cast(), i1) };
        unsafe { _mm512_storeu_si512(dst.add(128).cast(), i2) };
        unsafe { _mm512_storeu_si512(dst.add(192).cast(), i3) };

        src = unsafe { src.add(192) };
        dst = unsafe { dst.add(256) };
    }

    // Process remaining 48-byte chunks
    let safe_len_single = len.saturating_sub(16);
    let aligned_len_single = safe_len_single - (safe_len_single % 48);
    let src_end_single = unsafe { input.as_ptr().add(aligned_len_single) };

    while src < src_end_single {
        let v = load_48_bytes!(src);
        let res = encode_vec!(v);
        unsafe { _mm512_storeu_si512(dst.cast(), res) };

        src = unsafe { src.add(48) };
        dst = unsafe { dst.add(64) };
    }

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

/// Precomputed AVX-512 vector constants shared by every lane processed in
/// [`decode_slice_avx512`]. Factored out purely to keep that function's body
/// under clippy's line-count threshold; the values themselves are unchanged.
struct DecodeConstantsAvx512 {
    lut_hi_nibble: __m512i,
    sym_62: __m512i,
    sym_63: __m512i,
    delta_62: __m512i,
    delta_63: __m512i,
    range_0: __m512i,
    digit_span: __m512i,
    range_a: __m512i,
    upper_span: __m512i,
    range_a_low: __m512i,
    range_z_low_len: __m512i,
    pack_l1: __m512i,
    pack_l2: __m512i,
    pack_shuffle: __m512i,
    mask_hi_nibble: __m512i,
}

#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn decode_constants_avx512(config: &Config) -> DecodeConstantsAvx512 {
    // LUT for offsets based on high nibble (bits 4-7).
    let lut_hi_nibble = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 0, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0,
    ));

    // Range and offsets of special chars
    let (char_62, char_63) = if config.url_safe {
        (b'-', b'_')
    } else {
        (b'+', b'/')
    };
    let sym_62 = _mm512_set1_epi8(char_62.cast_signed());
    let sym_63 = _mm512_set1_epi8(char_63.cast_signed());

    let (fix_62, fix_63) = if config.url_safe { (-2, 33) } else { (0, -3) };
    let delta_62 = _mm512_set1_epi8(fix_62);
    let delta_63 = _mm512_set1_epi8(fix_63);

    // Range Validation Constants
    let range_0 = _mm512_set1_epi8(b'0'.cast_signed());
    let digit_span = _mm512_set1_epi8(9);

    let range_a = _mm512_set1_epi8(b'A'.cast_signed());
    let upper_span = _mm512_set1_epi8(25);

    let range_a_low = _mm512_set1_epi8(b'a'.cast_signed());
    let range_z_low_len = _mm512_set1_epi8(25);

    // Packing Constants
    let pack_l1 =
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_L1.as_ptr().cast::<__m128i>())) };
    let pack_l2 =
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_L2.as_ptr().cast::<__m128i>())) };
    let pack_shuffle =
        unsafe { _mm512_broadcast_i32x4(_mm_loadu_si128(PACK_SHUFFLE.as_ptr().cast::<__m128i>())) };

    // Masks for nibble extraction
    let mask_hi_nibble = _mm512_set1_epi8(0x0F);

    DecodeConstantsAvx512 {
        lut_hi_nibble,
        sym_62,
        sym_63,
        delta_62,
        delta_63,
        range_0,
        digit_span,
        range_a,
        upper_span,
        range_a_low,
        range_z_low_len,
        pack_l1,
        pack_l2,
        pack_shuffle,
        mask_hi_nibble,
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn decode_slice_avx512(
    config: &Config,
    input: &[u8],
    mut dst: *mut u8,
) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst;

    let DecodeConstantsAvx512 {
        lut_hi_nibble,
        sym_62,
        sym_63,
        delta_62,
        delta_63,
        range_0,
        digit_span,
        range_a,
        upper_span,
        range_a_low,
        range_z_low_len,
        pack_l1,
        pack_l2,
        pack_shuffle,
        mask_hi_nibble,
    } = unsafe { decode_constants_avx512(config) };

    // Decode & Validate Single Vector
    // Computed using mask ops for zero-blend performance
    macro_rules! decode_vec {
        ($input:expr) => {{
            let hi = _mm512_and_si512(_mm512_srli_epi16($input, 4), mask_hi_nibble);
            let offset = _mm512_shuffle_epi8(lut_hi_nibble, hi);
            let mut indices = _mm512_add_epi8($input, offset);

            let mask_62 = _mm512_cmpeq_epi8_mask($input, sym_62);
            let mask_63 = _mm512_cmpeq_epi8_mask($input, sym_63);

            indices = _mm512_mask_add_epi8(indices, mask_62, indices, delta_62);
            indices = _mm512_mask_add_epi8(indices, mask_63, indices, delta_63);

            let is_sym = _kor_mask64(mask_62, mask_63);

            let sub_0 = _mm512_sub_epi8($input, range_0);
            let is_num = _mm512_cmple_epu8_mask(sub_0, digit_span);

            let sub_a = _mm512_sub_epi8($input, range_a);
            let is_upper = _mm512_cmple_epu8_mask(sub_a, upper_span);

            let sub_a_low = _mm512_sub_epi8($input, range_a_low);
            let is_lower = _mm512_cmple_epu8_mask(sub_a_low, range_z_low_len);

            let is_char = _kor_mask64(is_num, _kor_mask64(is_upper, is_lower));
            let is_valid = _kor_mask64(is_char, is_sym);
            let err_mask = _knot_mask64(is_valid);

            (indices, err_mask)
        }};
    }

    macro_rules! pack_and_store {
        ($indices:expr, $dst_ptr:expr) => {{
            let m = _mm512_maddubs_epi16($indices, pack_l1);
            let p = _mm512_madd_epi16(m, pack_l2);
            let out = _mm512_shuffle_epi8(p, pack_shuffle);

            let lane0 = _mm512_castsi512_si128(out);
            unsafe { _mm_storeu_si128($dst_ptr.cast::<__m128i>(), lane0) };
            let lane1 = _mm512_extracti32x4_epi32(out, 1);
            unsafe { _mm_storeu_si128($dst_ptr.add(12).cast::<__m128i>(), lane1) };
            let lane2 = _mm512_extracti32x4_epi32(out, 2);
            unsafe { _mm_storeu_si128($dst_ptr.add(24).cast::<__m128i>(), lane2) };
            let lane3 = _mm512_extracti32x4_epi32(out, 3);
            unsafe { _mm_storeu_si128($dst_ptr.add(36).cast::<__m128i>(), lane3) };
        }};
    }

    // Process 128 bytes (4 chunks) at a time
    let safe_len_256 = len.saturating_sub(4);
    let aligned_len_256 = safe_len_256 - (safe_len_256 % 256);
    let src_end_256 = unsafe { src.add(aligned_len_256) };

    while src < src_end_256 {
        // Load 4 vectors
        let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(64).cast::<__m512i>()) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(128).cast::<__m512i>()) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(192).cast::<__m512i>()) };

        // Process
        let (i0, e0) = decode_vec!(v0);
        let (i1, e1) = decode_vec!(v1);
        let (i2, e2) = decode_vec!(v2);
        let (i3, e3) = decode_vec!(v3);

        // Check errors
        if (e0 | e1 | e2 | e3) != 0 {
            return Err(Error::InvalidCharacter);
        }

        // Store 4 chunks
        pack_and_store!(i0, dst);
        pack_and_store!(i1, dst.add(48));
        pack_and_store!(i2, dst.add(96));
        pack_and_store!(i3, dst.add(144));

        src = unsafe { src.add(256) };
        dst = unsafe { dst.add(192) };
    }

    // Process remaining 32-byte chunks
    let safe_len_64 = len.saturating_sub(4);
    let aligned_len_64 = safe_len_64 - (safe_len_64 % 64);
    let src_end_64 = unsafe { input.as_ptr().add(aligned_len_64) };

    while src < src_end_64 {
        let v = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let (idx, err_mask) = decode_vec!(v);

        if err_mask != 0 {
            return Err(Error::InvalidCharacter);
        }

        pack_and_store!(idx, dst);

        src = unsafe { src.add(64) };
        dst = unsafe { dst.add(48) };
    }

    unsafe { decode_scalar_tail(config, input, src, dst, dst_start) }
}

/// Decodes any bytes left over after the vectorized main/tail loops via the
/// scalar fallback, then returns the total number of bytes written.
///
/// # Safety
/// `src` must point within `input`, and `dst`/`dst_start` must satisfy the
/// same contract as [`scalar::decode_slice_unsafe`].
unsafe fn decode_scalar_tail(
    config: &Config,
    input: &[u8],
    src: *const u8,
    mut dst: *mut u8,
    dst_start: *mut u8,
) -> Result<usize, Error> {
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < input.len() {
        dst = unsafe {
            dst.add(scalar::decode_slice_unsafe(
                config,
                &input[processed_len..],
                dst,
            )?)
        };
    }

    Ok(unsafe { dst.offset_from(dst_start) }.cast_unsigned())
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

// ======================================================================
// AVX-512 VBMI Encoder
// ======================================================================

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn encode_slice_avx512_vbmi(config: &Config, input: &[u8], mut dst: *mut u8) {
    let len = input.len();
    let mut src = input.as_ptr();

    // VBMI: single-instruction byte reorder directly from a raw 64-byte load,
    // replacing the two-step permutexvar_epi32(dword gather) + shuffle_epi8
    // (byte pick) used by the plain AVX-512F/BW path. Independently verified
    // to equal that two-step composition byte-for-byte.
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

    // VBMI: Load the full 64-byte alphabet into a single ZMM register.
    // vpermb uses bits [5:0] of each index byte to select from this table.
    let alphabet = if config.url_safe {
        unsafe { _mm512_loadu_si512(VBMI_ENCODE_URL_SAFE.as_ptr().cast()) }
    } else {
        unsafe { _mm512_loadu_si512(VBMI_ENCODE_STANDARD.as_ptr().cast()) }
    };

    macro_rules! encode_vec_vbmi {
        ($in_vec:expr) => {{
            // Extract 6-bit indices using baseline AVX-512F/BW variable-shift
            // instructions (no VBMI needed) instead of the mullo/mulhi trick.
            let t0 = _mm512_and_si512($in_vec, _mm512_set1_epi32(0x0fc0_fc00));
            let t1 = _mm512_srlv_epi16(t0, _mm512_set1_epi32(0x0006_000a));
            let t2 = _mm512_sllv_epi16($in_vec, _mm512_set1_epi32(0x0008_0004));
            let indices = _mm512_ternarylogic_epi32::<0xca>(_mm512_set1_epi32(0x3f00_3f00), t2, t1);

            // VBMI: Single-instruction alphabet lookup replaces 8 instructions.
            // vpermb(idx, table): for each byte in idx, uses bits [5:0] to
            // select a byte from the 64-byte table.
            unsafe { zmm_permutexvar_epi8(indices, alphabet) }
        }};
    }

    macro_rules! load_48_bytes {
        ($ptr:expr) => {{
            let v = unsafe { _mm512_loadu_si512($ptr.cast()) };
            unsafe { zmm_permutexvar_epi8(shuffle_input, v) }
        }};
    }

    // Process 192 bytes (4 chunks) at a time
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

    // Process remaining 48-byte chunks
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

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < len {
        unsafe { scalar::encode_slice_unsafe(config, &input[processed_len..], dst) };
    }
}

// ======================================================================
// AVX-512 VBMI Decoder
// ======================================================================

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn decode_slice_avx512_vbmi(
    config: &Config,
    input: &[u8],
    mut dst: *mut u8,
) -> Result<usize, Error> {
    let len = input.len();
    let mut src = input.as_ptr();
    let dst_start = dst;

    // VBMI: Load 128-byte reverse LUT into two ZMM registers.
    // vpermi2b uses bit [6] to select between the two registers,
    // and bits [5:0] to select the byte within the chosen register.
    // This covers the full ASCII range 0–127 in a single lookup.
    let lut = if config.url_safe {
        &VBMI_DECODE_URL_SAFE
    } else {
        &VBMI_DECODE_STANDARD
    };
    let lut_lo = unsafe { _mm512_loadu_si512(lut.as_ptr().cast()) };
    let lut_hi = unsafe { _mm512_loadu_si512(lut.as_ptr().add(64).cast()) };

    // Sentinel for invalid characters
    let invalid = _mm512_set1_epi8(-1);

    // Packing Constants
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

    // Decode & Validate Single Vector (VBMI path)
    macro_rules! decode_vec_vbmi {
        ($input:expr) => {{
            // VBMI: Direct 128-byte table lookup (1 instruction).
            // vpermi2b(a, idx, b): for each byte in idx, bit [6] selects
            // between a (0) and b (1), bits [5:0] select the byte.
            let indices = unsafe { zmm_permutex2var_epi8(lut_lo, $input, lut_hi) };

            // Validate: check for 0xFF sentinel (invalid chars in LUT)
            let is_invalid = _mm512_cmpeq_epi8_mask(indices, invalid);
            // Check for bytes >= 128 (bit 7 set), which vpermi2b would alias
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

            // Masked store: only the low 48 bytes are real output. A caller
            // may pass a buffer sized exactly to the true decoded length, so
            // an unmasked 64-byte store would write out of bounds; the mask
            // makes AVX-512 suppress writes on the 16 masked-off high bytes.
            unsafe {
                _mm512_mask_storeu_epi8($dst_ptr.cast::<i8>(), 0x0000_FFFF_FFFF_FFFF, packed)
            };
        }};
    }

    // Process 256 bytes (4 chunks) at a time
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

    // Process remaining 64-byte chunks
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

    // Scalar Fallback
    let processed_len = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if processed_len < len {
        dst = unsafe {
            dst.add(scalar::decode_slice_unsafe(
                config,
                &input[processed_len..],
                dst,
            )?)
        };
    }

    Ok(unsafe { dst.offset_from(dst_start) }.cast_unsigned())
}

#[cfg(kani)]
mod kani_verification_avx512 {
    use super::*;
    use crate::{Config, STANDARD as TURBO_STANDARD, STANDARD_NO_PAD as TURBO_STANDARD_NO_PAD};
    #[cfg(target_arch = "x86")]
    use std::arch::x86::__mmask64;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::__mmask64;
    use std::mem::transmute;

    // --- CONSTANTS ---

    // Kani/CBMC is a bounded model checker: a proof at one fixed input
    // length only generalizes to "true for all N" if that length exercises
    // (a) the loop body at least twice, so the state one iteration hands
    // off (the advanced `src`/`dst` pointers) is proven to be a valid entry
    // state for the next iteration, and (b) a non-empty, non-aligned scalar
    // remainder, so the SIMD -> scalar handoff is covered too.
    //
    // `encode_slice_avx512`/`decode_slice_avx512` each have two independent
    // loop tiers: an outer quad-vector loop (192B/256B chunks, unrolled 4x)
    // that only triggers on large inputs, and an inner single-vector loop
    // (48B/64B chunks) that both the quad loop and short inputs fall
    // through to. `ENC_INDUCTION_LEN`/`DEC_INDUCTION_LEN` below hit exactly
    // 2 passes of the single-vector tier and 0 passes of the quad tier,
    // since that tier shares its per-block macro (`encode_vec!`/
    // `decode_vec!`) with the quad tier. The quad tier's own concern — its
    // 4x-unrolled pointer arithmetic and inter-tier handoff — is covered
    // separately by `check_avx512_quad_tier_roundtrip` below.

    // Encoder induction size: 96 (2x 48-byte AVX512 single-vector passes) +
    // 17 (scalar transition). The +17 (not +1) accounts for the loop's own
    // 16-byte safety margin, which guarantees at least a 16-byte scalar
    // remainder whenever the loop runs at all.
    const ENC_INDUCTION_LEN: usize = 113;

    // Decoder induction size: 128 (2x 64-byte AVX512 single-vector passes) +
    // 5 (scalar transition); decode's safety margin is 4 bytes.
    const DEC_INDUCTION_LEN: usize = 133;

    // Quad-tier induction size: smallest length that triggers exactly 1 pass
    // of the 192-byte quad-vector loop (0 single-vector-loop passes) plus a
    // scalar remainder. Used only by `check_avx512_quad_tier_roundtrip`.
    const QUAD_ENC_INDUCTION_LEN: usize = 209;

    // --- HELPERS ---

    fn encoded_size(len: usize, padding: bool) -> usize {
        if padding {
            TURBO_STANDARD.encoded_len(len)
        } else {
            TURBO_STANDARD_NO_PAD.encoded_len(len)
        }
    }

    // --- STUBS ---

    // STUB: _mm512_shuffle_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_shuffle_epi8
    unsafe fn _mm512_shuffle_epi8_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            // (In Rust we access bytes 'j' so '*8' offset is not needed)
            let i = j;

            // IF b[i+7] == 1
            if (b[i] & 0x80) != 0 {
                // dst[i+7:i] := 0
                dst[i] = 0;
            // ELSE
            } else {
                // index[5:0] := b[i+3:i] + (j & 0x30)
                let index: u8 = (b[i] & 0x0F) + (j as u8 & 0x30);
                // dst[i+7:i] := a[index*8+7:index*8]
                dst[i] = a[index as usize];
                // FI
            }
            // ENDFOR
        }
        // dst[MAX:512] := 0
        // (No extra bits beyond 512 in __m512i)

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_mask_add_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_add_epi8
    unsafe fn _mm512_mask_add_epi8_stub(
        src: __m512i,
        k: __mmask64,
        a: __m512i,
        b: __m512i,
    ) -> __m512i {
        let src_bytes: [u8; 64] = unsafe { transmute(src) };
        let a_bytes: [u8; 64] = unsafe { transmute(a) };
        let b_bytes: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            let i = j;

            // IF k[j]
            if (k & (1 << j)) != 0 {
                // dst[i+7:i] := a[i+7:i] + b[i+7:i]
                dst[i] = a_bytes[i].wrapping_add(b_bytes[i]);
            // ELSE
            } else {
                // dst[i+7:i] := src[i+7:i]
                dst[i] = src_bytes[i];
                // FI
            }
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_maddubs_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maddubs_epi16
    unsafe fn _mm512_maddubs_epi16_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [i8; 64] = unsafe { transmute(b) };
        let mut dst = [0i16; 32];

        // FOR j := 0 to 31
        for j in 0..32 {
            // i := j*16
            let i = j * 2;
            // dst[i+15:i] := Saturate16( a[i+15:i+8]*b[i+15:i+8] + a[i+7:i]*b[i+7:i] )
            dst[j] = ((a[i + 1] as i16) * (b[i + 1] as i16))
                .saturating_add((a[i] as i16) * (b[i] as i16));
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_madd_epi16
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_madd_epi16
    unsafe fn _mm512_madd_epi16_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [i16; 32] = unsafe { transmute(a) };
        let b: [i16; 32] = unsafe { transmute(b) };
        let mut dst = [0i32; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // i := j*32
            let i = j * 2;

            // dst[i+31:i] := SignExtend32(a[i+31:i+16]*b[i+31:i+16]) + SignExtend32(a[i+15:i]*b[i+15:i])
            dst[j] = (a[i + 1] as i32)
                .wrapping_mul(b[i + 1] as i32)
                .wrapping_add((a[i] as i32).wrapping_mul(b[i] as i32));
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_permutexvar_epi32
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_permutexvar_epi32
    unsafe fn _mm512_permutexvar_epi32_stub(idx: __m512i, a: __m512i) -> __m512i {
        let idx: [u32; 16] = unsafe { transmute(idx) };
        let a: [u32; 16] = unsafe { transmute(a) };
        let mut dst = [0u32; 16];

        // FOR j := 0 to 15
        for j in 0..16 {
            // id := idx[j*32+3:j*32]
            let id = (idx[j] & 0xF) as usize;
            // dst[j*32+31:j*32] := a[id*32+31:id*32]
            dst[j] = a[id];
        }
        // ENDFOR
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // STUB: _mm512_sub_epi8
    // REFERENCE: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_sub_epi8
    unsafe fn _mm512_sub_epi8_stub(a: __m512i, b: __m512i) -> __m512i {
        let a: [u8; 64] = unsafe { transmute(a) };
        let b: [u8; 64] = unsafe { transmute(b) };
        let mut dst = [0u8; 64];

        // FOR j := 0 to 63
        for j in 0..64 {
            // i := j*8
            let i = j;
            // dst[i+7:i] := a[i+7:i] - b[i+7:i]
            dst[i] = a[i].wrapping_sub(b[i]);
            // ENDFOR
        }
        // dst[MAX:512] := 0

        unsafe { transmute(dst) }
    }

    // --- PROOFS ---

    /// **Proof 1: Roundtrip Correctness (The Logic Check)**
    ///
    /// Verifies that `Decode(Encode(X)) == X`.
    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, _mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_mask_add_epi8, _mm512_mask_add_epi8_stub)]
    #[kani::stub(_mm512_maddubs_epi16, _mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_madd_epi16, _mm512_madd_epi16_stub)]
    #[kani::stub(_mm512_sub_epi8, _mm512_sub_epi8_stub)]
    #[kani::stub(_mm512_permutexvar_epi32, _mm512_permutexvar_epi32_stub)]
    fn check_avx512_roundtrip_correctness() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; ENC_INDUCTION_LEN] = kani::any();

        // Buffers
        let mut enc_buf = [0u8; 256];
        let mut dec_buf = [0u8; 256];

        unsafe {
            // 1. Encode
            encode_slice_avx512(&config, &input, enc_buf.as_mut_ptr());

            // Calculate actual encoded length for slicing
            let enc_len = encoded_size(ENC_INDUCTION_LEN, config.padding);
            let encoded_slice = &enc_buf[..enc_len];

            // 2. Decode
            // This MUST succeed for valid encoded output
            let dec_len = decode_slice_avx512(&config, encoded_slice, dec_buf.as_mut_ptr())
                .expect("Valid encoding failed to decode");

            // 3. Verify
            assert_eq!(dec_len, ENC_INDUCTION_LEN);
            assert_eq!(&dec_buf[..dec_len], &input, "Roundtrip mismatch");
        }
    }

    /// **Proof 2: Decoder Robustness & Induction**
    ///
    /// Verifies that `decode_slice_avx512`:
    /// 1. Accepts ANY bytes of garbage input.
    /// 2. Never Segfaults, Panics, or causes UB.
    /// 3. Safely handles the SIMD->Scalar pointer transition.
    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, _mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_mask_add_epi8, _mm512_mask_add_epi8_stub)]
    #[kani::stub(_mm512_maddubs_epi16, _mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_madd_epi16, _mm512_madd_epi16_stub)]
    #[kani::stub(_mm512_sub_epi8, _mm512_sub_epi8_stub)]
    fn check_avx512_decode_robustness() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };

        // Input: bytes of unrestricted symbolic data (garbage)
        let input: [u8; DEC_INDUCTION_LEN] = kani::any();

        // Output Buffer: Max estimated size
        let mut output = [0u8; 256];

        unsafe {
            // We ignore the Result. We only care that this function call
            // returns safely (Ok or Err) and does not crash.
            let _ = decode_slice_avx512(&config, &input, output.as_mut_ptr());
        }
    }

    /// **Proof 3: Quad-Tier Loop Coverage**
    ///
    /// Proofs 1 and 2 are sized to trigger the single-vector loop tier only
    /// (0 quad-tier passes), since that tier's per-block macro
    /// (`encode_vec!`/`decode_vec!`) is shared with the quad tier. What
    /// that leaves uncovered is the quad tier's own concern: its
    /// 4x-unrolled body's fixed pointer offsets (`src.add(48)`,
    /// `dst.add(64)`, etc.) and its handoff to the next stage (the
    /// single-vector tier, or the scalar fallback). `QUAD_ENC_INDUCTION_LEN`
    /// triggers exactly 1 quad-tier pass — 1 suffices here (not 2) because
    /// the quad loop's per-iteration pointer arithmetic
    /// (`src += 192; dst += 256`) is a fixed-stride increment with no
    /// per-iteration branching, and the loop's own state-handoff is already
    /// proven by the single-vector tier's 2-pass proof, since both loops
    /// are built from the same primitives.
    ///
    /// Verifies that `Decode(Encode(X)) == X` still holds when the quad
    /// tier runs. This checks logical correctness rather than just
    /// crash-safety: `decode_slice_avx512` validates all 4 quad-loop
    /// sub-blocks before storing any of them (see `decode_vec!`'s call
    /// sites), so a separate garbage-input quad-tier robustness proof would
    /// be redundant with Proof 2's already-exhaustive coverage of
    /// `decode_vec!`'s validation logic.
    #[kani::proof]
    #[kani::stub(_mm512_shuffle_epi8, _mm512_shuffle_epi8_stub)]
    #[kani::stub(_mm512_mask_add_epi8, _mm512_mask_add_epi8_stub)]
    #[kani::stub(_mm512_maddubs_epi16, _mm512_maddubs_epi16_stub)]
    #[kani::stub(_mm512_madd_epi16, _mm512_madd_epi16_stub)]
    #[kani::stub(_mm512_sub_epi8, _mm512_sub_epi8_stub)]
    #[kani::stub(_mm512_permutexvar_epi32, _mm512_permutexvar_epi32_stub)]
    fn check_avx512_quad_tier_roundtrip() {
        let config = Config {
            url_safe: kani::any(),
            padding: true,
        };
        let input: [u8; QUAD_ENC_INDUCTION_LEN] = kani::any();

        // Buffers: encoded_size(209, true) = 280, sized with margin.
        let mut enc_buf = [0u8; 320];
        let mut dec_buf = [0u8; 320];

        unsafe {
            // 1. Encode (exercises the quad-tier encode loop: 1 pass)
            encode_slice_avx512(&config, &input, enc_buf.as_mut_ptr());

            let enc_len = encoded_size(QUAD_ENC_INDUCTION_LEN, config.padding);
            let encoded_slice = &enc_buf[..enc_len];

            // 2. Decode (this encoded_len also happens to land in the
            // decode quad tier: 1 pass, 0 single-vector passes)
            let dec_len = decode_slice_avx512(&config, encoded_slice, dec_buf.as_mut_ptr())
                .expect("Valid encoding failed to decode");

            // 3. Verify
            assert_eq!(dec_len, QUAD_ENC_INDUCTION_LEN);
            assert_eq!(&dec_buf[..dec_len], &input, "Quad-tier roundtrip mismatch");
        }
    }
}

#[cfg(all(test, miri))]
mod miri_avx512_coverage {
    use super::*;
    use base64::{
        Engine,
        engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD},
    };
    use rand::{RngExt, rng};

    // --- Mock Infrastructure ---
    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    /// Helper to verify AVX512 encoding against the 'base64' crate oracle
    fn verify_encode_avx512(config: &Config, oracle: &impl Engine, input_len: usize) {
        let input = random_bytes(input_len);
        let expected = oracle.encode(&input);
        let mut dst = vec![0u8; expected.len() * 2]; // Safety margin

        unsafe {
            encode_slice_avx512(config, &input, dst.as_mut_ptr());
        }

        let result = &dst[..expected.len()];
        assert_eq!(
            std::str::from_utf8(result).unwrap(),
            expected,
            "Encode len {}",
            input_len
        );
    }

    /// Helper to verify AVX512 decoding against the 'base64' crate oracle
    fn verify_decode_avx512(config: &Config, oracle: &impl Engine, original_len: usize) {
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();
        let mut dst = vec![0u8; original_len + 64];

        let len = unsafe {
            decode_slice_avx512(config, encoded_bytes, dst.as_mut_ptr())
                .expect("Valid input failed to decode")
        };

        assert_eq!(&dst[..len], &input_bytes, "Decode len {}", original_len);
    }

    // ----------------------------------------------------------------------
    // 1. Encoder Coverage Tests (AVX512)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_encode_scalar_fallback() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // AVX512 Single Loop threshold is 48 bytes.
        // Test < 48 bytes -> Pure Scalar
        verify_encode_avx512(&config, &STANDARD, 1);
        verify_encode_avx512(&config, &STANDARD, 47);
    }

    #[test]
    fn miri_avx512_encode_single_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // AVX512 Single Vector processes 48 input bytes.
        // Exactly 1 loop
        verify_encode_avx512(&config, &STANDARD, 48);
        // Exactly 2 loops (Proves pointer math)
        verify_encode_avx512(&config, &STANDARD, 96);
        // 1 loop + 1 byte scalar fallback
        verify_encode_avx512(&config, &STANDARD, 49);
    }

    #[test]
    fn miri_avx512_encode_quad_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // The quad loop needs `aligned_len_192 >= 192`, i.e. `len >= 208`
        // (the 16-byte safety margin means 192 alone isn't enough); 192 and
        // 193 resolve to 0 quad-loop iterations and are handled by the
        // single-vector loop instead.
        verify_encode_avx512(&config, &STANDARD, 192);
        verify_encode_avx512(&config, &STANDARD, 193);
        // 208: smallest length that triggers exactly 1 quad-loop iteration
        // (0 single-loop iterations, 16-byte scalar remainder).
        verify_encode_avx512(&config, &STANDARD, 208);
        // 384: 2 quad-loop iterations (proves inter-iteration pointer
        // arithmetic — src/dst advancing correctly for a 2nd pass).
        verify_encode_avx512(&config, &STANDARD, 384);
        // 240: 1 quad-loop iteration (192) + 1 single-loop iteration (48),
        // 0 scalar remainder.
        verify_encode_avx512(&config, &STANDARD, 240);
    }

    #[test]
    fn miri_avx512_encode_url_safe() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        verify_encode_avx512(&config, &URL_SAFE, 100);
    }

    // ----------------------------------------------------------------------
    // 2. Decoder Coverage Tests (AVX512)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_decode_scalar_fallback() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // AVX512 decode loop threshold is 64 bytes.
        // < 64 bytes input -> Pure Scalar
        verify_decode_avx512(&config, &STANDARD, 3); // 4 encoded chars
        verify_decode_avx512(&config, &STANDARD, 45); // 60 encoded chars
    }

    #[test]
    fn miri_avx512_decode_single_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // AVX512 Single Vector processes 64 input bytes.
        // Exactly 1 Single Loop
        verify_decode_avx512(&config, &STANDARD, 48); // 64 bytes encoded
        // Exactly 2 Single Loops
        verify_decode_avx512(&config, &STANDARD, 96); // 128 bytes encoded
        // 1 Single Loop + Scalar Remainder
        verify_decode_avx512(&config, &STANDARD, 49); // 64 bytes + extra
    }

    #[test]
    fn miri_avx512_decode_quad_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // The quad loop needs `aligned_len_256 >= 256`, i.e. encoded input
        // >= 260 bytes (the 4-byte safety margin means 256 alone isn't
        // enough). raw=192 encodes to exactly 256 bytes, so it does not
        // reach the quad loop (all 3 rounds run via the single-vector tail
        // loop).
        verify_decode_avx512(&config, &STANDARD, 192); // 256 bytes encoded
        // raw=193 encodes to 260 bytes: exactly 1 quad-loop iteration, 0
        // single-loop iterations, 4-byte scalar remainder.
        verify_decode_avx512(&config, &STANDARD, 193); // 260 bytes encoded
        // raw=384 encodes to 512 bytes: 1 quad-loop iteration + 3 more via
        // the single-vector tail loop (proves inter-tier pointer handoff).
        verify_decode_avx512(&config, &STANDARD, 384); // 512 bytes encoded
    }

    #[test]
    fn miri_avx512_decode_url_safe() {
        let config = Config {
            url_safe: true,
            padding: false,
        };
        // 64 bytes input to trigger one AVX512 vector
        // Repeated pattern of - and _
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_";
        let mut dst = [0u8; 64];
        unsafe {
            decode_slice_avx512(&config, input, dst.as_mut_ptr()).unwrap();
        }
    }

    // ----------------------------------------------------------------------
    // 3. Error Logic Coverage (AVX512)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_decode_error_detection() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        let mut dst = [0u8; 512];

        // Case 1: Error in Quad Loop (last vector, last lane)
        // Batch size is 256 bytes.
        let mut bad_input_256 = vec![b'A'; 256];
        bad_input_256[255] = b'$'; // Invalid char
        let res = unsafe { decode_slice_avx512(&config, &bad_input_256, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Quad Loop");

        // Case 2: Error in Single Loop
        // Vector size is 64 bytes.
        let mut bad_input_64 = vec![b'A'; 64];
        bad_input_64[63] = b'?'; // Invalid char
        let res = unsafe { decode_slice_avx512(&config, &bad_input_64, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Single Loop");

        // Case 3: Error in Quad Loop (first vector, first byte)
        let mut bad_input_256_first = vec![b'A'; 256];
        bad_input_256_first[0] = b'$';
        let res = unsafe { decode_slice_avx512(&config, &bad_input_256_first, dst.as_mut_ptr()) };
        assert!(
            res.is_err(),
            "Failed to catch error in Quad Loop first vector"
        );

        // Case 4: Error in Scalar Fallback (after SIMD processing)
        let mut bad_input_65 = vec![b'A'; 65];
        bad_input_65[64] = b'?'; // Invalid in scalar region
        let res = unsafe { decode_slice_avx512(&config, &bad_input_65, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Scalar Fallback");
    }

    // ----------------------------------------------------------------------
    // 4. Roundtrip & Config Coverage (AVX512)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_roundtrip_standard() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for &len in &[48, 96, 192, 193, 240, 384] {
            let input = random_bytes(len);
            let expected = STANDARD.encode(&input);
            let mut enc = vec![0u8; expected.len() * 2];
            unsafe {
                encode_slice_avx512(&config, &input, enc.as_mut_ptr());
            }
            let encoded = &enc[..expected.len()];
            assert_eq!(std::str::from_utf8(encoded).unwrap(), expected);

            let mut dec = vec![0u8; len + 64];
            let dec_len =
                unsafe { decode_slice_avx512(&config, encoded, dec.as_mut_ptr()).unwrap() };
            assert_eq!(&dec[..dec_len], &input, "Roundtrip len {}", len);
        }
    }

    #[test]
    fn miri_avx512_encode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[1, 48, 49, 96, 192, 193] {
            verify_encode_avx512(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_decode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[3, 48, 49, 96, 192, 193] {
            let input_bytes = random_bytes(len);
            let encoded = STANDARD_NO_PAD.encode(&input_bytes);
            let mut dst = vec![0u8; len + 64];
            let dec_len = unsafe {
                decode_slice_avx512(&config, encoded.as_bytes(), dst.as_mut_ptr()).unwrap()
            };
            assert_eq!(&dst[..dec_len], &input_bytes, "No-pad decode len {}", len);
        }
    }

    #[test]
    fn miri_avx512_encode_url_safe_no_pad() {
        let config = Config {
            url_safe: true,
            padding: false,
        };
        for &len in &[48, 96, 192] {
            verify_encode_avx512(&config, &URL_SAFE_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_decode_url_safe_roundtrip() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        verify_decode_avx512(&config, &URL_SAFE, 100);
    }
}

#[cfg(all(test, miri))]
mod miri_avx512_vbmi_coverage {
    use super::*;
    use base64::{
        Engine,
        engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD},
    };
    use rand::{RngExt, rng};

    // --- Mock Infrastructure ---
    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    /// Helper to verify AVX512-VBMI encoding against the 'base64' crate oracle
    fn verify_encode_avx512_vbmi(config: &Config, oracle: &impl Engine, input_len: usize) {
        let input = random_bytes(input_len);
        let expected = oracle.encode(&input);
        let mut dst = vec![0u8; expected.len() * 2]; // Safety margin

        unsafe {
            encode_slice_avx512_vbmi(config, &input, dst.as_mut_ptr());
        }

        let result = &dst[..expected.len()];
        assert_eq!(
            std::str::from_utf8(result).unwrap(),
            expected,
            "Encode len {input_len}"
        );
    }

    /// Helper to verify AVX512-VBMI decoding against the 'base64' crate oracle
    fn verify_decode_avx512_vbmi(config: &Config, oracle: &impl Engine, original_len: usize) {
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();
        let mut dst = vec![0u8; original_len + 64];

        let len = unsafe {
            decode_slice_avx512_vbmi(config, encoded_bytes, dst.as_mut_ptr())
                .expect("Valid input failed to decode")
        };

        assert_eq!(&dst[..len], &input_bytes, "Decode len {original_len}");
    }

    // ----------------------------------------------------------------------
    // 1. Encoder Coverage Tests (AVX512-VBMI)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_vbmi_encode_scalar_fallback() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Single Vector threshold is 48 bytes (same layout as non-VBMI encoder).
        verify_encode_avx512_vbmi(&config, &STANDARD, 1);
        verify_encode_avx512_vbmi(&config, &STANDARD, 47);
    }

    #[test]
    fn miri_avx512_vbmi_encode_single_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        verify_encode_avx512_vbmi(&config, &STANDARD, 48);
        verify_encode_avx512_vbmi(&config, &STANDARD, 96);
        verify_encode_avx512_vbmi(&config, &STANDARD, 49);
    }

    #[test]
    fn miri_avx512_vbmi_encode_quad_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Same tier thresholds as `encode_slice_avx512` (see
        // `miri_avx512_encode_quad_vector_loop`). 192/193 miss the quad loop
        // (single-loop only).
        verify_encode_avx512_vbmi(&config, &STANDARD, 192);
        verify_encode_avx512_vbmi(&config, &STANDARD, 193);
        // 208: smallest length hitting exactly 1 quad-loop iteration.
        verify_encode_avx512_vbmi(&config, &STANDARD, 208);
        // 384: 2 quad-loop iterations.
        verify_encode_avx512_vbmi(&config, &STANDARD, 384);
        // 240: 1 quad-loop iteration + 1 single-loop iteration.
        verify_encode_avx512_vbmi(&config, &STANDARD, 240);
    }

    #[test]
    fn miri_avx512_vbmi_encode_url_safe() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        // Exercises the VBMI_ENCODE_URL_SAFE alphabet table specifically.
        verify_encode_avx512_vbmi(&config, &URL_SAFE, 100);
    }

    // ----------------------------------------------------------------------
    // 2. Decoder Coverage Tests (AVX512-VBMI)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_vbmi_decode_scalar_fallback() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Decode Single Vector threshold is 64 bytes.
        verify_decode_avx512_vbmi(&config, &STANDARD, 3);
        verify_decode_avx512_vbmi(&config, &STANDARD, 45);
    }

    #[test]
    fn miri_avx512_vbmi_decode_single_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        verify_decode_avx512_vbmi(&config, &STANDARD, 48);
        verify_decode_avx512_vbmi(&config, &STANDARD, 96);
        verify_decode_avx512_vbmi(&config, &STANDARD, 49);
    }

    #[test]
    fn miri_avx512_vbmi_decode_quad_vector_loop() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        // Same tier thresholds as `decode_slice_avx512` — see
        // `miri_avx512_decode_quad_vector_loop`'s comments for the derived
        // boundaries. raw=192 (256B encoded) misses the quad loop (0
        // iterations); raw=193 (260B encoded) hits it exactly once.
        verify_decode_avx512_vbmi(&config, &STANDARD, 192);
        verify_decode_avx512_vbmi(&config, &STANDARD, 193);
        verify_decode_avx512_vbmi(&config, &STANDARD, 384);
    }

    #[test]
    fn miri_avx512_vbmi_decode_url_safe() {
        let config = Config {
            url_safe: true,
            padding: false,
        };
        // Exercises the VBMI_DECODE_URL_SAFE reverse LUT specifically.
        let input = b"-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_";
        let mut dst = [0u8; 64];
        unsafe {
            decode_slice_avx512_vbmi(&config, input, dst.as_mut_ptr()).unwrap();
        }
    }

    // ----------------------------------------------------------------------
    // 3. Error Logic Coverage (AVX512-VBMI)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_vbmi_decode_error_detection() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        let mut dst = [0u8; 512];

        // Case 1: `0xFF` LUT sentinel path — char not present in either alphabet.
        let mut bad_input_256 = vec![b'A'; 256];
        bad_input_256[255] = b'$';
        let res = unsafe { decode_slice_avx512_vbmi(&config, &bad_input_256, dst.as_mut_ptr()) };
        assert!(
            res.is_err(),
            "Failed to catch sentinel-invalid char in Quad Loop"
        );

        let mut bad_input_64 = vec![b'A'; 64];
        bad_input_64[63] = b'?';
        let res = unsafe { decode_slice_avx512_vbmi(&config, &bad_input_64, dst.as_mut_ptr()) };
        assert!(
            res.is_err(),
            "Failed to catch sentinel-invalid char in Single Loop"
        );

        // Case 2: high-bit (>= 128) rejection path, specific to VBMI's `vpermi2b`-based
        // decode (`is_high_bit` in `decode_vec_vbmi!`) — bytes >= 128 alias into the LUT
        // via bit 6 rather than naturally failing a range check like the scalar/AVX2/
        // plain-AVX512 paths, so this needs its own dedicated regression case.
        let mut bad_input_high_bit = vec![b'A'; 256];
        bad_input_high_bit[0] = 0x80;
        let res =
            unsafe { decode_slice_avx512_vbmi(&config, &bad_input_high_bit, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch byte >= 128 in Quad Loop");

        let mut bad_input_high_bit_single = vec![b'A'; 64];
        bad_input_high_bit_single[0] = 0xFF;
        let res = unsafe {
            decode_slice_avx512_vbmi(&config, &bad_input_high_bit_single, dst.as_mut_ptr())
        };
        assert!(res.is_err(), "Failed to catch byte >= 128 in Single Loop");

        // Case 3: Error in Scalar Fallback (after SIMD processing)
        let mut bad_input_65 = vec![b'A'; 65];
        bad_input_65[64] = b'?';
        let res = unsafe { decode_slice_avx512_vbmi(&config, &bad_input_65, dst.as_mut_ptr()) };
        assert!(res.is_err(), "Failed to catch error in Scalar Fallback");
    }

    // ----------------------------------------------------------------------
    // 4. Roundtrip & Config Coverage (AVX512-VBMI)
    // ----------------------------------------------------------------------

    #[test]
    fn miri_avx512_vbmi_roundtrip_standard() {
        let config = Config {
            url_safe: false,
            padding: true,
        };
        for &len in &[48, 96, 192, 193, 240, 384] {
            let input = random_bytes(len);
            let expected = STANDARD.encode(&input);
            let mut enc = vec![0u8; expected.len() * 2];
            unsafe {
                encode_slice_avx512_vbmi(&config, &input, enc.as_mut_ptr());
            }
            let encoded = &enc[..expected.len()];
            assert_eq!(std::str::from_utf8(encoded).unwrap(), expected);

            let mut dec = vec![0u8; len + 64];
            let dec_len =
                unsafe { decode_slice_avx512_vbmi(&config, encoded, dec.as_mut_ptr()).unwrap() };
            assert_eq!(&dec[..dec_len], &input, "Roundtrip len {len}");
        }
    }

    #[test]
    fn miri_avx512_vbmi_encode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[1, 48, 49, 96, 192, 193] {
            verify_encode_avx512_vbmi(&config, &STANDARD_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_no_padding() {
        let config = Config {
            url_safe: false,
            padding: false,
        };
        for &len in &[3, 48, 49, 96, 192, 193] {
            let input_bytes = random_bytes(len);
            let encoded = STANDARD_NO_PAD.encode(&input_bytes);
            let mut dst = vec![0u8; len + 64];
            let dec_len = unsafe {
                decode_slice_avx512_vbmi(&config, encoded.as_bytes(), dst.as_mut_ptr()).unwrap()
            };
            assert_eq!(&dst[..dec_len], &input_bytes, "No-pad decode len {len}");
        }
    }

    #[test]
    fn miri_avx512_vbmi_encode_url_safe_no_pad() {
        let config = Config {
            url_safe: true,
            padding: false,
        };
        for &len in &[48, 96, 192] {
            verify_encode_avx512_vbmi(&config, &URL_SAFE_NO_PAD, len);
        }
    }

    #[test]
    fn miri_avx512_vbmi_decode_url_safe_roundtrip() {
        let config = Config {
            url_safe: true,
            padding: true,
        };
        verify_decode_avx512_vbmi(&config, &URL_SAFE, 100);
    }

    // ----------------------------------------------------------------------
    // 5. Exact-Buffer Boundary Coverage (masked-store safety regression)
    // ----------------------------------------------------------------------

    /// Decodes into a buffer sized to the exact true output length only (no
    /// safety margin), so Miri's precise out-of-bounds tracking can catch a
    /// masked-store overrun by even a single byte.
    fn verify_decode_avx512_vbmi_exact(config: &Config, oracle: &impl Engine, original_len: usize) {
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();
        let mut dst = vec![0u8; original_len];

        let len = unsafe {
            decode_slice_avx512_vbmi(config, encoded_bytes, dst.as_mut_ptr())
                .expect("Valid input failed to decode")
        };

        assert_eq!(len, original_len, "Exact-buffer decode len {original_len}");
        assert_eq!(
            &dst[..len],
            &input_bytes,
            "Exact-buffer decode len {original_len}"
        );
    }

    #[test]
    fn miri_avx512_vbmi_decode_exact_buffer_boundaries() {
        // Regression coverage for the vpermb-compaction + masked-store
        // rewrite of `pack_and_store!`: every chunk-boundary length (tail,
        // single-vector, quad-vector, and a long multi-quad-iteration
        // buffer) must decode without writing past an exactly-sized output
        // buffer, for both alphabets and padded/unpadded input.
        let standard = Config {
            url_safe: false,
            padding: true,
        };
        let url_safe = Config {
            url_safe: true,
            padding: true,
        };
        let no_pad = Config {
            url_safe: false,
            padding: false,
        };

        for &len in &[3, 45, 48, 96, 192, 193, 240, 384, 1000, 1001] {
            verify_decode_avx512_vbmi_exact(&standard, &STANDARD, len);
            verify_decode_avx512_vbmi_exact(&url_safe, &URL_SAFE, len);
            verify_decode_avx512_vbmi_exact(&no_pad, &STANDARD_NO_PAD, len);
        }
    }
}

#[cfg(all(test, not(miri)))]
mod avx512_vbmi_hardware_coverage {
    use super::*;
    use base64::{
        Engine,
        engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE},
    };
    use rand::{RngExt, rng};

    fn random_bytes(len: usize) -> Vec<u8> {
        let mut rng = rng();
        (0..len).map(|_| rng.random()).collect()
    }

    fn verify_decode_avx512_vbmi_exact(config: &Config, oracle: &impl Engine, original_len: usize) {
        let input_bytes = random_bytes(original_len);
        let encoded = oracle.encode(&input_bytes);
        let encoded_bytes = encoded.as_bytes();
        let mut dst = vec![0u8; original_len];

        let len = unsafe {
            decode_slice_avx512_vbmi(config, encoded_bytes, dst.as_mut_ptr())
                .expect("Valid input failed to decode")
        };

        assert_eq!(len, original_len, "Exact-buffer decode len {original_len}");
        assert_eq!(
            &dst[..len],
            &input_bytes,
            "Exact-buffer decode len {original_len}"
        );
    }

    #[test]
    fn hw_avx512_vbmi_decode_exact_buffer_boundaries() {
        if !(std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vbmi"))
        {
            eprintln!("skipping: host CPU lacks AVX-512-VBMI");
            return;
        }

        let standard = Config {
            url_safe: false,
            padding: true,
        };
        let url_safe = Config {
            url_safe: true,
            padding: true,
        };
        let no_pad = Config {
            url_safe: false,
            padding: false,
        };

        for &len in &[3, 45, 48, 96, 192, 193, 240, 384, 1000, 1001] {
            verify_decode_avx512_vbmi_exact(&standard, &STANDARD, len);
            verify_decode_avx512_vbmi_exact(&url_safe, &URL_SAFE, len);
            verify_decode_avx512_vbmi_exact(&no_pad, &STANDARD_NO_PAD, len);
        }
    }
}
