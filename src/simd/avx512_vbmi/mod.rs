//! AVX-512-VBMI Base64. Two VBMI-only instructions carry this path: `vpermb` /
//! `vpermi2b` replace the plain AVX-512F/BW arithmetic character mapping, and
//! `vpmultishiftqb` replaces the encoder's shift/mask chain outright.
//!
//! The design goal is to retire fewer ops per vector:
//!
//! * encode: gather -> `vpmultishiftqb` -> alphabet `vpermb` is 3 ops per
//!   48-byte vector, down from 6.
//! * decode: validity is folded into a `vpternlogd` OR tree, one op per vector,
//!   replacing the per-vector compare/movemask/kor trio.
//!
//! Both also run their remainder through masked vector passes rather than
//! handing tens of bytes to the scalar kernel; a masked `vmovdqu8` cannot fault
//! on a masked-off element, so the loops need no read-ahead slack and scalar
//! only ever sees the final group — and even that is placed here, by
//! [`encode_final_group`] / [`decode_final_group`], because entering the scalar
//! kernel for one group cost a flat ~20 cycles (decode) and ~6-9 (encode),
//! which is half the cost of a 128-byte decode and 40% of a 512-byte one.
//!
//! Both permutes read their control vector — the 64-byte alphabet for encode,
//! the low 128 entries of the reverse table for decode — straight out of the
//! `Config`'s [`Alphabet`](crate::Alphabet), so this kernel serves a custom
//! alphabet at exactly the speed it serves the built-ins.
//!
//! Where the tuning stops is microarchitecture-specific, and the two vendors
//! measured so far do not agree:
//!
//! * **Sapphire Rapids** — both kernels are bound by port 5, where every byte
//!   permute issues, dispatching ~1.0 port-5 uops per cycle on an L1-resident
//!   input (`uops_dispatched.port_5_11` within 2% of `cycles`).
//! * **Zen 5** (EPYC 9R45) — port 5 is an Intel structure and the model does not
//!   carry over. The decoder is *FP-dispatch* bound: ~8 FP ops per 64-character
//!   vector at 3.98 dispatched per cycle, a flat ceiling. The encoder is neither
//!   — it is **store bound**, spending 33% of its cycles in dispatch stalls on
//!   store-queue tokens (`de_dispatch_stall_cycle_dynamic_tokens_part1.store_queue_rsrc_stall`)
//!   because it writes 4/3 of what it reads, while FP-scheduler stalls are under
//!   0.5%. Unroll factor was swept there (1/2/4/8) and 4 is the best of them.
//!
//! Either way the shuffle chain itself has no slack left to reclaim, and past
//! the last-level cache neither kernel is compute-bound at all: both sit on
//! memory bandwidth, encode moving 7/3 bytes of traffic per input byte and
//! decode 7/4. On the Zen 5 lab box one core tops out at ~56 GiB/s of total
//! traffic (measured with a hand-rolled non-temporal `memcpy`, against a
//! ~45 GiB/s pure-read ceiling), which puts a hard roof of ~24 GiB/s on encode
//! and ~32 GiB/s on decode at 100 MiB no matter what the kernels do — they
//! reach ~22 and ~29. Unroll factor and software prefetch were both swept again
//! in that regime and every variant landed within 2% of the shipped loop, so
//! the only thing that actually moves at these sizes is the write stream — see
//! [`nontemporal_min`].

use crate::{Config, Error};

#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m512i, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_mask_loadu_epi8,
    _mm512_mask_storeu_epi8, _mm512_maskz_loadu_epi8, _mm512_movepi8_mask, _mm512_or_si512,
    _mm512_set1_epi8, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_storeu_si512, _mm512_ternarylogic_epi32,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m512i, _mm512_loadu_si512, _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_mask_loadu_epi8,
    _mm512_mask_storeu_epi8, _mm512_maskz_loadu_epi8, _mm512_movepi8_mask, _mm512_or_si512,
    _mm512_set1_epi8, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_set1_epi64,
    _mm512_setzero_si512, _mm512_storeu_si512, _mm512_ternarylogic_epi32,
};

// `_mm512_stream_si512`/`_mm_sfence` lower to inline `asm!`, which Miri never
// executes; the Miri build routes around them (see `zmm_stream`/`sfence`).
#[cfg(all(not(miri), target_arch = "x86"))]
use std::arch::x86::{
    _mm_sfence, _mm512_multishift_epi64_epi8, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
    _mm512_stream_si512,
};
#[cfg(all(not(miri), target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm_sfence, _mm512_multishift_epi64_epi8, _mm512_permutex2var_epi8, _mm512_permutexvar_epi8,
    _mm512_stream_si512,
};

// --- Compile-time lookup tables ---

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

/// `vpermi2b` controls for the decoder's streaming packer.
///
/// The ordinary packer turns each 64-character vector into 48 bytes with its
/// own `vpermb` and stores it 48 wide, which `vmovntdq` cannot do — a streaming
/// store is a whole vector or nothing. Four packed vectors are exactly 192
/// bytes, so each 64-byte output is drawn from the *pair* of packed vectors
/// that straddles it: still one shuffle, but three of them instead of four, and
/// three whole stores instead of three overhanging ones plus a masked one.
///
/// It is a worse packer everywhere else — `vpermi2b` costs more port 5 than
/// `vpermb`, measured -16% on an L1-resident input — so it lives only in the
/// streaming loop, which is waiting on memory anyway.
#[allow(clippy::cast_possible_truncation)] // every index is < 128 by construction
const fn build_stream_pack(which: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        // Decoded byte `g` of the 192-byte group lives in packed vector `g / 48`
        // at index `g % 48`, and inside a packed vector byte `d` is byte
        // `2 - d % 3` of dword `d / 3` (the triples are big-endian). Bit 6 of a
        // `vpermi2b` index picks the second source register, so a byte from the
        // straddling vector is the same index plus 64.
        let g = which * 64 + i;
        let d = g % DEC_VEC_OUT;
        let src = 4 * (d / 3) + (2 - d % 3);
        t[i] = (src + if g / DEC_VEC_OUT == which { 0 } else { 64 }) as u8;
        i += 1;
    }
    t
}
const VBMI_STREAM_PACK: [[u8; 64]; 3] = [
    build_stream_pack(0),
    build_stream_pack(1),
    build_stream_pack(2),
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

// --- Non-temporal threshold ---

/// Floor on the input length at which either kernel may switch its top tier to
/// non-temporal stores, and the bound the Kani stream-peel proofs are stated
/// against.
///
/// The tuned threshold is [`nontemporal_min`], which is derived from the
/// last-level cache and is never below this; the proofs assume only this
/// weaker gate, so any larger runtime value is covered by them.
///
/// The binding requirement is that the gate leave a whole quad step after the
/// worst-case alignment peel. Otherwise the decoder's peel loop — which stops
/// if `rem` falls under the quad guard — could exit with `dst` still unaligned,
/// and the streaming loop it exists for would be skipped entirely. The static
/// assertion below makes that a consequence of the constants rather than a
/// coincidence.
#[cfg(not(miri))]
const NONTEMPORAL_FLOOR: usize = 512 * 1024;
/// Lowered under Miri, whose suites run at tiny lengths and would otherwise
/// never reach the streaming tier or its alignment peel at all. This is the
/// value that actually needs the assertion below checking.
#[cfg(miri)]
const NONTEMPORAL_FLOOR: usize = 1024;

/// Fraction of the last-level cache above which streaming stores start paying,
/// as `RATIO / DIV`.
///
/// Whether `vmovntdq` helps depends on something the kernel cannot see: whether
/// the destination was going to stay in cache. Both cases were measured on
/// Zen 5 (c8a.large, 8 MiB L3), racing this kernel against a twin with the tier
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
/// The two agree above the crossover and conflict below it, so the gate is put
/// where the resident case turns: 5/8 of the last-level cache, ~5 MiB here,
/// which is where the output stops co-residing with the input. Above it
/// streaming is right for both. Below it the resident case is the one that
/// decides, because its penalty is the larger of the two — -39% against the
/// -13% the cold case pays for not streaming.
///
/// This is a ratio and not a length because the crossover tracks the cache
/// rather than the machine. The value it replaces was a flat 512 KiB, tuned on
/// Sapphire Rapids (c7i.large) against a flushed destination — the cold case
/// only, which as measured above has no crossover to find. Carried onto a Zen 5
/// box with an 8 MiB L3 it put the gate at a sixteenth of where the resident
/// case turns, which is the 3.6x cliff at 1 MiB this replaces. Sapphire Rapids
/// has not been re-measured under the resident case; if its crossover also
/// tracks its last-level cache, this ratio covers it, and that is the claim
/// worth re-checking first on any new microarchitecture.
#[cfg(not(miri))]
const NONTEMPORAL_LLC_RATIO: usize = 5;
#[cfg(not(miri))]
const NONTEMPORAL_LLC_DIV: usize = 8;

/// Input length at which both kernels switch their top tier to non-temporal
/// stores.
///
/// Both write more than they read, so once the working set no longer fits in
/// cache most of the cost is read-for-ownership traffic on a destination whose
/// old contents are dead, and `vmovntdq` skips it. Below that point the same
/// instruction is a large *loss*, because it throws away a cache hit the
/// regular store would have got.
///
/// Where that crossover sits is a property of the machine, not a constant: it
/// tracks the last-level cache, which is why this reads the cache size rather
/// than hard-coding a length. See [`NONTEMPORAL_LLC_RATIO`] for the measurement
/// behind the ratio.
///
/// One wrinkle worth knowing: the result is compared against `rem`, which is
/// input *bytes* in the encoder but input *characters* in the decoder, so one
/// threshold gates two different working-set footprints (7/3·E against 7/4·C).
/// The measured crossover lands at the same *input* length for both, so that is
/// the unit it is expressed in.
#[inline]
fn nontemporal_min() -> usize {
    // Miri runs neither `cpuid` nor inputs anywhere near a real threshold, so
    // there it is the floor that makes the streaming tier reachable at all.
    #[cfg(miri)]
    {
        NONTEMPORAL_FLOOR
    }
    #[cfg(not(miri))]
    {
        // Resolved once inside `cpu::llc_bytes`; what is left here is a load
        // and a multiply, on an input already known to be a quad step long.
        // With no cache size to scale against, the floor is the threshold.
        let Some(llc) = crate::cpu::llc_bytes() else {
            return NONTEMPORAL_FLOOR;
        };
        (llc / NONTEMPORAL_LLC_DIV)
            .saturating_mul(NONTEMPORAL_LLC_RATIO)
            .max(NONTEMPORAL_FLOOR)
    }
}

/// Worst-case input each kernel's alignment peel consumes before the streaming
/// loop starts.
///
/// The encoder's peel is one masked step of at most 60 output characters, so 45
/// input bytes. The decoder's `head` grows by a vector until it is also a whole
/// number of 3-byte groups, so it tops out at `3 * 64 - 3` output bytes — 252
/// characters, four masked steps.
const ENC_PEEL_MAX: usize = (ENC_VEC - 4) / 4 * ENC_GROUP;
const DEC_PEEL_MAX: usize = (3 * DEC_VEC_IN - 3) / 3 * DEC_GROUP;

// The Kani proofs in `verify` re-derive the same bound symbolically.
const _: () = assert!(
    NONTEMPORAL_FLOOR >= ENC_QUAD_MIN + ENC_PEEL_MAX
        && NONTEMPORAL_FLOOR >= DEC_QUAD_MIN + DEC_PEEL_MAX,
    "NONTEMPORAL_FLOOR must leave a full quad step after the worst-case alignment peel"
);

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

/// `vmovntdq`, routed through an ordinary store under Miri, whose interpreter
/// will not execute the real instruction. Exact rather than an approximation:
/// streaming is a caching hint, not a difference in the value stored or in what
/// a single-threaded reader observes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn zmm_stream(dst: *mut __m512i, val: __m512i) {
    #[cfg(miri)]
    unsafe {
        _mm512_storeu_si512(dst, val);
    }
    #[cfg(not(miri))]
    unsafe {
        _mm512_stream_si512(dst, val);
    }
}

/// `sfence`, which orders the non-temporal stores above against later loads.
/// With none of those under Miri, it is a no-op there too.
#[inline]
fn sfence() {
    #[cfg(not(miri))]
    unsafe {
        _mm_sfence();
    }
}

/// Encodes the trailing one- or two-byte group without entering the scalar
/// kernel.
///
/// The masked tier consumes every whole triple, so this is all that can be left,
/// and the handoff costs more than the work: entering `scalar::encode_slice` to
/// place two bytes measured 6-9 cycles on Zen 5, which a 128-byte encode has
/// nothing to hide behind.
///
/// Returns `false` without writing if `dst` has no room, leaving the caller to
/// fall back to the scalar kernel.
///
/// # Safety
/// `src` must have `left` readable bytes, and `left` must be 1 or 2.
#[inline]
unsafe fn encode_final_group(
    config: &Config,
    src: *const u8,
    dst: *mut u8,
    left: usize,
    room: usize,
) -> bool {
    let out = if config.padding { 4 } else { left + 1 };
    if out > room {
        return false;
    }
    let a = config.alphabet.as_bytes();
    let b0 = unsafe { *src };
    let b1 = if left == 2 { unsafe { *src.add(1) } } else { 0 };
    let n = (u32::from(b0) << 16) | (u32::from(b1) << 8);
    unsafe {
        *dst = a[(n >> 18) as usize & 0x3F];
        *dst.add(1) = a[(n >> 12) as usize & 0x3F];
        if left == 2 {
            *dst.add(2) = a[(n >> 6) as usize & 0x3F];
        }
        if config.padding {
            if left == 1 {
                *dst.add(2) = b'=';
            }
            *dst.add(3) = b'=';
        }
    }
    true
}

/// Decodes the trailing four-character group without entering the scalar kernel.
///
/// Every tier stops at least [`DEC_LEAD`] short of the end, so the scalar kernel
/// is otherwise entered on *every* call just to place one group -- a flat ~20
/// cycles on Zen 5, which is half the cost of a 128-byte decode and 40% of a
/// 512-byte one.
///
/// Only the plain four-character shape is handled, which is what every
/// well-formed padded input ends with. Anything else -- an unpadded remainder, a
/// misplaced `=`, a destination with no room -- returns `None` so the caller
/// falls back to the scalar kernel, which keeps sole ownership of the padding
/// and length rules.
///
/// On success returns the number of bytes written (1, 2 or 3).
///
/// # Safety
/// `src` must have four readable characters.
#[inline]
unsafe fn decode_final_group(
    config: &Config,
    src: *const u8,
    dst: *mut u8,
    room: usize,
) -> Option<Result<usize, Error>> {
    let c0 = unsafe { *src };
    let c1 = unsafe { *src.add(1) };
    let c2 = unsafe { *src.add(2) };
    let c3 = unsafe { *src.add(3) };
    // `=` is legal only as the last one or two characters, and only for a config
    // that pads at all.
    let pad = usize::from(c2 == b'=') + usize::from(c3 == b'=');
    let shape_ok = c0 != b'=' && c1 != b'=' && (c2 != b'=' || c3 == b'=');
    if !shape_ok || (pad > 0 && !config.padding) || 3 - pad > room {
        return None;
    }
    let t = config.alphabet.decode_table();
    let d0 = t[usize::from(c0)];
    let d1 = t[usize::from(c1)];
    let d2 = if c2 == b'=' { 0 } else { t[usize::from(c2)] };
    let d3 = if c3 == b'=' { 0 } else { t[usize::from(c3)] };
    // Valid entries are 0..=63 and the sentinel is 0xFF, so one test on the OR
    // covers all four.
    if (d0 | d1 | d2 | d3) >= 0x80 {
        return Some(Err(Error::InvalidCharacter));
    }
    // A padded group must be canonical: the bits the dropped byte(s) would have
    // carried have to be zero. `XX==` keeps only the top two bits of `d1`, and
    // `XXX=` only the top four of `d2`. The scalar kernel rejects a group that
    // sets the rest, so this path has to as well or the two disagree.
    if (pad == 2 && d1 & 0x0F != 0) || (pad == 1 && d2 & 0x03 != 0) {
        return Some(Err(Error::InvalidCharacter));
    }
    let n = (u32::from(d0) << 18) | (u32::from(d1) << 12) | (u32::from(d2) << 6) | u32::from(d3);
    // The triple sits in bits 0..23, so big-endian bytes 1..3 are exactly the
    // output and no truncating cast is needed.
    let b = n.to_be_bytes();
    unsafe {
        *dst = b[1];
        if pad < 2 {
            *dst.add(1) = b[2];
        }
        if pad == 0 {
            *dst.add(2) = b[3];
        }
    }
    Some(Ok(3 - pad))
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

    // Full 64-byte alphabet in one ZMM, straight out of the `Alphabet`; vpermb
    // selects by each index's low 6 bits, so the garbage in each index's top 2
    // bits needs no masking. Any alphabet works here — it is pure data.
    let alphabet =
        unsafe { _mm512_loadu_si512(config.alphabet.as_bytes().as_ptr().cast::<__m512i>()) };

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

    /// Quad tier body: 192 input bytes -> 256 output. The last load starts 144
    /// bytes in and reads 64, so 208 <= 256 bytes are always in bounds.
    macro_rules! encode_quad {
        ($store:ident) => {{
            let r0 = encode_vec!(load_48!(0));
            let r1 = encode_vec!(load_48!(ENC_VEC_IN));
            let r2 = encode_vec!(load_48!(2 * ENC_VEC_IN));
            let r3 = encode_vec!(load_48!(3 * ENC_VEC_IN));
            unsafe { $store(dst.cast(), r0) };
            unsafe { $store(dst.add(ENC_VEC_OUT).cast(), r1) };
            unsafe { $store(dst.add(2 * ENC_VEC_OUT).cast(), r2) };
            unsafe { $store(dst.add(3 * ENC_VEC_OUT).cast(), r3) };
            src = unsafe { src.add(ENC_QUAD_IN) };
            dst = unsafe { dst.add(ENC_QUAD_OUT) };
            rem -= ENC_QUAD_IN;
        }};
    }

    /// One masked step consuming `take` input bytes, which must be a whole
    /// number of triples and at most 48. Shared by the masked tier and by the
    /// streaming tier's alignment peel.
    macro_rules! encode_masked {
        ($take:expr) => {{
            let take = $take;
            let out = take / ENC_GROUP * 4;
            let v = unsafe { _mm512_maskz_loadu_epi8(u64::MAX >> (ENC_VEC - take), src.cast()) };
            let chars = encode_vec!(v);
            unsafe {
                _mm512_mask_storeu_epi8(dst.cast::<i8>(), u64::MAX >> (ENC_VEC - out), chars)
            };
            src = unsafe { src.add(take) };
            dst = unsafe { dst.add(out) };
            rem -= take;
        }};
    }

    // Both size gates are nested inside the test the quad tier has to make
    // anyway, so an input with no whole quad in it reaches the single tier after
    // one compare and pays nothing for either.
    if rem >= ENC_QUAD_MIN {
        // Streaming tier. `vmovntdq` faults on an unaligned address and every
        // quad step advances `dst` by 256, so the alignment is decided once, up
        // front, by encoding whole groups until `dst` reaches a boundary. That
        // is only reachable at all when `dst` is 4-byte aligned: a group always
        // emits 4 characters, so `dst % 4` is invariant and an output that
        // starts at an odd address can never reach a 64-byte boundary.
        if rem >= nontemporal_min() && dst.addr().is_multiple_of(4) {
            let head = (ENC_VEC - (dst.addr() & (ENC_VEC - 1))) & (ENC_VEC - 1);
            if head > 0 {
                encode_masked!(head / 4 * ENC_GROUP);
            }
            debug_assert!(dst.addr().is_multiple_of(ENC_VEC));
            // The alignment is re-tested rather than assumed, so a peel that
            // somehow missed cannot turn into a faulting store.
            while dst.addr().is_multiple_of(ENC_VEC) && rem >= ENC_QUAD_MIN {
                encode_quad!(zmm_stream);
            }
            sfence();
        }

        while rem >= ENC_QUAD_MIN {
            encode_quad!(_mm512_storeu_si512);
        }
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
        encode_masked!((rem - rem % ENC_GROUP).min(ENC_VEC_IN));
    }

    // Scalar now sees at most the final 1-2 bytes, plus whatever padding the
    // config asks for.
    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();

    // The masked tier above runs until fewer than three bytes are left, so `rem`
    // already *is* the size of the final partial group. Reading it beats
    // recovering the same number from the pointers, which is what the scalar
    // handoff does on entry.
    debug_assert!(rem < ENC_GROUP);
    if rem == 0 {
        return;
    }
    if unsafe { encode_final_group(config, src, dst, rem, dst_slice.len() - dst_off) } {
        return;
    }
    unsafe { super::tail::encode(config, input, src, dst_slice, dst_off) };
}

// --- VBMI decoder ---

/// The lookup vectors the decoder builds once from its `Config` and both of its
/// top tiers then share.
#[derive(Clone, Copy)]
struct DecodeLuts {
    /// Low and high halves of the 128-byte reverse LUT, for `vpermi2b`.
    lut_lo: __m512i,
    lut_hi: __m512i,
    /// `vpmaddubsw` / `vpmaddwd` multipliers that fold four 6-bit indices into
    /// one 24-bit triple.
    pack_l1: __m512i,
    pack_l2: __m512i,
    /// `vpermb` control that compresses those triples into 48 output bytes.
    pack: __m512i,
    /// The alphabet's index-0 character, used to backfill the lanes a masked
    /// step does not read. It has to come from the alphabet in play: a fixed
    /// `'A'` would decode to the 0xFF sentinel under an alphabet that lacks it
    /// and fail validation on padding lanes that carry no input.
    fill: __m512i,
}

/// One masked decode step, consuming `take` characters — a whole number of
/// groups, at most 64 — and writing the `take / 4 * 3` bytes they decode to.
///
/// The lanes past `take` are backfilled with the alphabet's index-0 character,
/// so they cannot trip validation. Returns this step's validity evidence for the
/// caller to fold in.
///
/// # Safety
/// `src` must have `take` characters and `dst` room for `take / 4 * 3` bytes.
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn decode_masked_step(
    luts: &DecodeLuts,
    src: *const u8,
    dst: *mut u8,
    take: usize,
) -> __m512i {
    let out = take / DEC_GROUP * 3;
    let v =
        unsafe { _mm512_mask_loadu_epi8(luts.fill, u64::MAX >> (DEC_VEC_IN - take), src.cast()) };
    let idx = unsafe { zmm_permutex2var_epi8(luts.lut_lo, v, luts.lut_hi) };
    let m = _mm512_maddubs_epi16(idx, luts.pack_l1);
    let p = unsafe { zmm_permutexvar_epi8(luts.pack, _mm512_madd_epi16(m, luts.pack_l2)) };
    unsafe { _mm512_mask_storeu_epi8(dst.cast::<i8>(), u64::MAX >> (DEC_VEC_IN - out), p) };
    _mm512_ternarylogic_epi32::<0xFE>(_mm512_setzero_si512(), v, idx)
}

/// Where the streaming loop left off.
struct StreamState {
    src: *const u8,
    dst: *mut u8,
    rem: usize,
    /// Validity evidence for the characters this loop consumed; the caller ORs
    /// it into its own accumulator.
    bad: __m512i,
}

/// The decoder's non-temporal top tier: whole 64-byte `vmovntdq` stores, which
/// need the three-vector packer described on [`VBMI_STREAM_PACK`].
///
/// Out of line because it is entered at most once per call, above
/// [`nontemporal_min`], so the call costs nothing measurable — and keeping it
/// out of the main kernel keeps that function's register pressure and its
/// length where they were.
///
/// # Safety
/// `src`/`dst` must have `rem` characters and the corresponding output bytes
/// available, and `rem` must be at least [`NONTEMPORAL_FLOOR`].
#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn decode_stream_loop(
    luts: &DecodeLuts,
    mut src: *const u8,
    mut dst: *mut u8,
    mut rem: usize,
) -> StreamState {
    let mut bad = _mm512_setzero_si512();

    // Alignment peel. `vmovntdq` faults on an unaligned address and every step
    // below advances `dst` by 192, so the alignment is decided once, here, by
    // decoding whole groups until `dst` reaches a boundary. Unlike the encoder
    // there is no precondition on where `dst` starts: a group emits 3 bytes and
    // 3 is coprime with 64, so some whole number of groups reaches a boundary
    // from *any* address — at most three 64-byte steps of it, hence `head < 192`
    // and fewer than 256 characters consumed.
    let mut head = (DEC_VEC_IN - (dst.addr() & (DEC_VEC_IN - 1))) & (DEC_VEC_IN - 1);
    while !head.is_multiple_of(3) {
        head += DEC_VEC_IN;
    }
    let mut chars = head / 3 * DEC_GROUP;
    while chars > 0 && rem >= DEC_QUAD_MIN {
        let take = chars.min(DEC_VEC_IN);
        bad = _mm512_or_si512(bad, unsafe { decode_masked_step(luts, src, dst, take) });
        src = unsafe { src.add(take) };
        dst = unsafe { dst.add(take / DEC_GROUP * 3) };
        rem -= take;
        chars -= take;
    }
    debug_assert!(dst.addr().is_multiple_of(DEC_VEC_IN));

    let pack0 = unsafe { _mm512_loadu_si512(VBMI_STREAM_PACK[0].as_ptr().cast()) };
    let pack1 = unsafe { _mm512_loadu_si512(VBMI_STREAM_PACK[1].as_ptr().cast()) };
    let pack2 = unsafe { _mm512_loadu_si512(VBMI_STREAM_PACK[2].as_ptr().cast()) };

    // The alignment is re-tested rather than assumed, so a peel that somehow
    // missed cannot turn into a faulting store.
    while dst.addr().is_multiple_of(DEC_VEC_IN) && rem >= DEC_QUAD_MIN {
        let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
        let v1 = unsafe { _mm512_loadu_si512(src.add(DEC_VEC_IN).cast::<__m512i>()) };
        let v2 = unsafe { _mm512_loadu_si512(src.add(2 * DEC_VEC_IN).cast::<__m512i>()) };
        let v3 = unsafe { _mm512_loadu_si512(src.add(3 * DEC_VEC_IN).cast::<__m512i>()) };

        let i0 = unsafe { zmm_permutex2var_epi8(luts.lut_lo, v0, luts.lut_hi) };
        let i1 = unsafe { zmm_permutex2var_epi8(luts.lut_lo, v1, luts.lut_hi) };
        let i2 = unsafe { zmm_permutex2var_epi8(luts.lut_lo, v2, luts.lut_hi) };
        let i3 = unsafe { zmm_permutex2var_epi8(luts.lut_lo, v3, luts.lut_hi) };

        let t0 = _mm512_ternarylogic_epi32::<0xFE>(v0, i0, v1);
        let t1 = _mm512_ternarylogic_epi32::<0xFE>(i1, v2, i2);
        let t2 = _mm512_ternarylogic_epi32::<0xFE>(v3, i3, t0);
        bad = _mm512_ternarylogic_epi32::<0xFE>(bad, t1, t2);

        let fold = |idx| _mm512_madd_epi16(_mm512_maddubs_epi16(idx, luts.pack_l1), luts.pack_l2);
        let f0 = fold(i0);
        let f1 = fold(i1);
        let f2 = fold(i2);
        let f3 = fold(i3);

        // 192 bytes as three whole vectors, each drawn from the pair it
        // straddles.
        let o0 = unsafe { zmm_permutex2var_epi8(f0, pack0, f1) };
        let o1 = unsafe { zmm_permutex2var_epi8(f1, pack1, f2) };
        let o2 = unsafe { zmm_permutex2var_epi8(f2, pack2, f3) };
        unsafe { zmm_stream(dst.cast(), o0) };
        unsafe { zmm_stream(dst.add(DEC_VEC_IN).cast(), o1) };
        unsafe { zmm_stream(dst.add(2 * DEC_VEC_IN).cast(), o2) };

        src = unsafe { src.add(DEC_QUAD_IN) };
        dst = unsafe { dst.add(DEC_QUAD_OUT) };
        rem -= DEC_QUAD_IN;
    }
    // Streaming stores are only ordered against a fence.
    sfence();

    StreamState { src, dst, rem, bad }
}

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

    // The low 128 entries of the alphabet's reverse table, across two ZMMs;
    // vpermi2b picks the register by bit 6 and the byte by the low 6 bits,
    // covering ASCII 0-127 in one lookup. `Alphabet::new` caps characters at
    // 0x7E, so those 128 entries hold every valid character of any alphabet and
    // 0xFF everywhere else.
    let lut = config.alphabet.decode_table();
    let lut_lo = unsafe { _mm512_loadu_si512(lut.as_ptr().cast()) };
    let lut_hi = unsafe { _mm512_loadu_si512(lut.as_ptr().add(64).cast()) };

    let pack_l1 = _mm512_set1_epi16(VBMI_PACK_L1);
    let pack_l2 = _mm512_set1_epi32(VBMI_PACK_L2);
    let pack = unsafe { _mm512_loadu_si512(VBMI_PACK_SHUFFLE.as_ptr().cast()) };
    let luts = DecodeLuts {
        lut_lo,
        lut_hi,
        pack_l1,
        pack_l2,
        pack,
        fill: _mm512_set1_epi8(config.alphabet.as_bytes()[0].cast_signed()),
    };

    // A character is bad iff its input byte had bit 7 set (>= 0x80, which
    // vpermi2b silently aliases into the 128-entry table) or the LUT answered
    // with the 0xFF sentinel. Either way bit 7 of `input | index` is set, so
    // OR-ing every input and every index into one accumulator and testing its
    // sign bits once validates the whole buffer. That is one `vpternlogd` per
    // vector in place of a compare, a movemask and a mask-OR.
    let mut bad = _mm512_setzero_si512();

    macro_rules! fold_vec {
        ($idx:expr) => {{
            let m = _mm512_maddubs_epi16($idx, pack_l1);
            _mm512_madd_epi16(m, pack_l2)
        }};
    }
    macro_rules! pack_vec {
        ($idx:expr) => {{ unsafe { zmm_permutexvar_epi8(pack, fold_vec!($idx)) } }};
    }

    /// The half of a quad step both top tiers share: four loads, four reverse
    /// lookups, and the validity fold. Yields the four index vectors.
    macro_rules! decode_quad_front {
        () => {{
            let v0 = unsafe { _mm512_loadu_si512(src.cast::<__m512i>()) };
            let v1 = unsafe { _mm512_loadu_si512(src.add(DEC_VEC_IN).cast::<__m512i>()) };
            let v2 = unsafe { _mm512_loadu_si512(src.add(2 * DEC_VEC_IN).cast::<__m512i>()) };
            let v3 = unsafe { _mm512_loadu_si512(src.add(3 * DEC_VEC_IN).cast::<__m512i>()) };

            let i0 = unsafe { zmm_permutex2var_epi8(lut_lo, v0, lut_hi) };
            let i1 = unsafe { zmm_permutex2var_epi8(lut_lo, v1, lut_hi) };
            let i2 = unsafe { zmm_permutex2var_epi8(lut_lo, v2, lut_hi) };
            let i3 = unsafe { zmm_permutex2var_epi8(lut_lo, v3, lut_hi) };

            // 0xFE is the 3-input OR; four of them fold all eight vectors in.
            let t0 = _mm512_ternarylogic_epi32::<0xFE>(v0, i0, v1);
            let t1 = _mm512_ternarylogic_epi32::<0xFE>(i1, v2, i2);
            let t2 = _mm512_ternarylogic_epi32::<0xFE>(v3, i3, t0);
            bad = _mm512_ternarylogic_epi32::<0xFE>(bad, t1, t2);

            (i0, i1, i2, i3)
        }};
    }

    // Quad tier: 256 input characters -> 192 output bytes. Every tier stops at
    // least 4 characters short of the end so the final group -- the only one
    // that may legally carry '=' -- is always decided by the scalar tail, which
    // owns the padding and length rules.
    //
    // As in the encoder, the streaming gate is nested inside the quad tier's own
    // test, so nothing below 260 characters pays for it.
    if rem >= DEC_QUAD_MIN {
        // Streaming tier. Unlike the encoder there is no alignment precondition:
        // a group emits 3 bytes and 3 is coprime with 64, so some whole number
        // of groups reaches a boundary from *any* destination address. At most
        // three 64-byte steps of peel are needed, hence `head < 192`.
        if rem >= nontemporal_min() {
            let stream = unsafe { decode_stream_loop(&luts, src, dst, rem) };
            src = stream.src;
            dst = stream.dst;
            rem = stream.rem;
            bad = _mm512_or_si512(bad, stream.bad);
        }

        while rem >= DEC_QUAD_MIN {
            let (i0, i1, i2, i3) = decode_quad_front!();

            let p0 = pack_vec!(i0);
            let p1 = pack_vec!(i1);
            let p2 = pack_vec!(i2);
            let p3 = pack_vec!(i3);

            // Only the last store needs masking: each of the first three
            // overhangs its 48 bytes by 16, and the very next store in this same
            // iteration rewrites exactly that overhang.
            unsafe { _mm512_storeu_si512(dst.cast(), p0) };
            unsafe { _mm512_storeu_si512(dst.add(DEC_VEC_OUT).cast(), p1) };
            unsafe { _mm512_storeu_si512(dst.add(2 * DEC_VEC_OUT).cast(), p2) };
            unsafe { _mm512_mask_storeu_epi8(dst.add(3 * DEC_VEC_OUT).cast::<i8>(), LOW_48, p3) };

            src = unsafe { src.add(DEC_QUAD_IN) };
            dst = unsafe { dst.add(DEC_QUAD_OUT) };
            rem -= DEC_QUAD_IN;
        }
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

    // Masked tier.
    if rem >= DEC_MASKED_MIN {
        let take = (rem - DEC_LEAD) & !(DEC_GROUP - 1);
        bad = _mm512_or_si512(bad, unsafe { decode_masked_step(&luts, src, dst, take) });
        src = unsafe { src.add(take) };
        dst = unsafe { dst.add(take / DEC_GROUP * 3) };
    }

    if _mm512_movepi8_mask(bad) != 0 {
        return Err(Error::InvalidCharacter);
    }

    let dst_off = unsafe { dst.offset_from(dst_start) }.cast_unsigned();
    let done = unsafe { src.offset_from(input.as_ptr()) }.cast_unsigned();
    if input.len() - done == DEC_GROUP
        && let Some(r) = unsafe { decode_final_group(config, src, dst, dst_slice.len() - dst_off) }
    {
        return r.map(|written| dst_off + written);
    }
    unsafe { super::tail::decode(config, input, src, dst_slice, dst_off) }
}

// Verification: Kani proofs, Intel-pseudocode intrinsic models, and the Miri +
// hardware coverage suites.
#[cfg(any(kani, test, miri))]
mod verify;
