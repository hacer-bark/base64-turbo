//! Runtime CPU capability detection for the x86 kernels, resolved once and cached.
//!
//! Feature bits come from [`cpufeatures`], which reads `CPUID` directly — no
//! `std`, and it checks `XCR0` for the AVX-512 state as well as the feature
//! bits, so a kernel that the OS has not enabled is never selected.
//!
//! The one thing it does not answer is how large the last-level cache is, which
//! the non-temporal store gate needs; that stays a `CPUID` walk of our own.

use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering::Relaxed};

#[cfg(b64_avx2)]
cpufeatures::new!(avx2_cpuid, "avx2");
#[cfg(b64_avx512)]
cpufeatures::new!(avx512_vbmi_cpuid, "avx512f", "avx512bw", "avx512vbmi");

// Tier levels, ordered least- to most-capable so callers compare with `>=`.
// Each level exists only when its kernel was compiled in, which keeps the
// detection and dispatch arms in lockstep with the selected backend.
#[cfg(b64_avx2)]
pub(crate) const AVX2: u8 = 1;
#[cfg(b64_avx512)]
pub(crate) const AVX512_VBMI: u8 = 2;

/// No tier resolved yet. `0` already means "scalar", so the sentinel has to sit
/// outside the range of real answers.
const TIER_UNINIT: u8 = u8::MAX;

fn detect() -> u8 {
    #[cfg(b64_avx512)]
    if avx512_vbmi_cpuid::get() {
        return AVX512_VBMI;
    }
    #[cfg(b64_avx2)]
    if avx2_cpuid::get() {
        return AVX2;
    }
    0 // scalar
}

/// The best compiled-in kernel tier the current CPU supports. Detected on the
/// first call and cached for the lifetime of the process.
///
/// `cpufeatures` caches each feature separately; collapsing them into one byte
/// here makes the whole *tier* decision a single relaxed load on every call
/// after the first, rather than one per feature. Relaxed is enough because the
/// value is derived from immutable hardware state, so a racing pair of callers
/// can only write the same answer twice.
#[inline]
pub(crate) fn tier() -> u8 {
    static CACHE: AtomicU8 = AtomicU8::new(TIER_UNINIT);

    match CACHE.load(Relaxed) {
        TIER_UNINIT => {
            let tier = detect();
            CACHE.store(tier, Relaxed);
            tier
        }
        tier => tier,
    }
}

/// Size in bytes of the largest data cache `CPUID` reports, or `None` when
/// it reports none — an emulator, or a hypervisor that masks the leaves.
///
/// Intel's leaf 4 and AMD's leaf `0x8000_001D` encode a cache's geometry in
/// exactly the same EAX/EBX/ECX layout, so one walk serves both vendors and
/// only the leaf number differs. Sub-leaves are enumerated until one reports
/// type 0; instruction caches are skipped, since the thing being sized is a
/// data working set.
///
/// Absent under Miri and Kani. Neither executes `cpuid` — Miri interprets,
/// Kani is symbolic — and neither needs to: the only caller, the AVX-512
/// streaming gate, takes its floor when no cache size is available, which
/// is exactly the bound the stream-peel proofs are stated against.
///
/// `__cpuid_count` became a safe function in 1.94; at this crate's 1.93 MSRV
/// it is still `unsafe`. The blocks are what 1.93 needs, and the `allow` is
/// what stops 1.94-and-later failing the crate's `unused` deny over blocks
/// it considers redundant. Both halves are load-bearing until the MSRV
/// reaches 1.94.
#[cfg(not(any(miri, kani)))]
#[allow(unused_unsafe)]
fn detect_llc() -> Option<usize> {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::__cpuid_count;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::__cpuid_count;

    // Both leaves have to be checked for existence first: reading an
    // unimplemented leaf returns another leaf's contents, not zeros.
    let leaf = if unsafe { __cpuid_count(0x8000_0000, 0) }.eax >= 0x8000_001D {
        0x8000_001D
    } else if unsafe { __cpuid_count(0, 0) }.eax >= 4 {
        4
    } else {
        return None;
    };

    let mut best: u64 = 0;
    for sub in 0..16 {
        let r = unsafe { __cpuid_count(leaf, sub) };
        match r.eax & 0x1f {
            0 => break,    // no cache at this sub-leaf, and none after it
            2 => continue, // instruction cache
            _ => {}
        }
        let ways = u64::from((r.ebx >> 22) & 0x3ff) + 1;
        let partitions = u64::from((r.ebx >> 12) & 0x3ff) + 1;
        let line = u64::from(r.ebx & 0xfff) + 1;
        let sets = u64::from(r.ecx) + 1;
        best = best.max(ways * partitions * line * sets);
    }
    (best > 0).then(|| usize::try_from(best).unwrap_or(LLC_MAX))
}

/// No cache size resolved yet, and the largest one that can be reported. A real
/// last-level cache is nowhere near either, so clamping the answer to `LLC_MAX`
/// costs nothing and keeps the sentinel outside the range of real answers.
#[cfg(not(any(miri, kani)))]
const LLC_UNINIT: usize = usize::MAX;
#[cfg(not(any(miri, kani)))]
const LLC_MAX: usize = usize::MAX - 1;

/// Size of the last-level data cache, detected once and cached. `None` if
/// `CPUID` does not report one; callers pick their own fallback.
#[cfg(not(any(miri, kani)))]
#[inline]
pub(crate) fn llc_bytes() -> Option<usize> {
    // `0` is the "CPUID reported none" answer, which is why `None` can share the
    // cache with a real size rather than needing a slot of its own.
    static CACHE: AtomicUsize = AtomicUsize::new(LLC_UNINIT);

    let cached = match CACHE.load(Relaxed) {
        LLC_UNINIT => {
            let bytes = detect_llc().unwrap_or(0);
            CACHE.store(bytes, Relaxed);
            bytes
        }
        bytes => bytes,
    };
    (cached > 0).then_some(cached)
}
