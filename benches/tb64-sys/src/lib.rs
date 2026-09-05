//! Minimal FFI shim over [Turbo-Base64](https://github.com/powturbo/Turbo-Base64), used
//! only by the `base64-turbo` benchmarks.
//!
//! No Turbo-Base64 code lives in this repository; the build script compiles a checkout the
//! user supplies. When that checkout is absent the whole crate degrades to [`AVAILABLE`]
//! being `false` and the benchmark skips its tb64 candidates.
//!
//! The calls here go straight to `_tb64e` / `_tb64d`, the function pointers upstream
//! documents as the fastest entry point ("saving a function call + a check instruction"),
//! rather than the `tb64enc` / `tb64dec` wrappers. Nothing is validated on this side
//! either: bounds checks the C caller would not pay for are not added here.

#![allow(unsafe_code)]

/// Whether the C library was found and linked in.
pub const AVAILABLE: bool = cfg!(tb64);

/// Trailing bytes every buffer handed to [`encode`] and [`decode`] must have beyond the
/// length the codec actually needs.
///
/// Upstream's own driver allocates its *input* buffer at the encoded length, so the vector
/// kernels are never run against an exactly sized buffer, and the short-string variants are
/// documented to read up to 32 bytes past the input end. 64 covers both with a cache line
/// to spare.
pub const SLACK: usize = 64;

/// Base64 length of `n` input bytes, padded — matches `tb64enclen`.
#[must_use]
pub const fn encoded_len(n: usize) -> usize {
    n.div_ceil(3) * 4
}

#[cfg(tb64)]
mod ffi {
    use core::ffi::c_char;

    /// `size_t (*)(const unsigned char *, size_t, unsigned char *)`.
    pub type Func = unsafe extern "C" fn(*const u8, usize, *mut u8) -> usize;

    #[allow(non_upper_case_globals)]
    unsafe extern "C" {
        pub static _tb64e: Func;
        pub static _tb64d: Func;
        pub static tb64_floor: Func;

        pub fn tb64ini(id: u32, isshort: u32);
        pub fn cpuini(cpuisa: u32) -> u32;
        pub fn cpustr(cpuisa: u32) -> *mut c_char;
    }
}

/// Runs the CPU detection that installs the kernels behind `_tb64e` / `_tb64d`.
///
/// Must be called once before [`encode`] or [`decode`]; until then both point at the
/// scalar fallback. `isshort` is left at 0: the short-string variants skip validation,
/// which would not be a like-for-like comparison against decoders that check their input.
pub fn init() {
    #[cfg(tb64)]
    // SAFETY: `tb64ini` only writes tb64's own globals. Called before any codec call.
    unsafe {
        ffi::tb64ini(0, 0);
    }
}

/// The instruction set tb64 selected, e.g. `"avx512_vbmi"` — for the benchmark report.
#[must_use]
pub fn isa() -> String {
    #[cfg(tb64)]
    // SAFETY: `cpustr` returns a pointer to a static string literal inside the library.
    let name = unsafe {
        core::ffi::CStr::from_ptr(ffi::cpustr(ffi::cpuini(0)))
            .to_string_lossy()
            .into_owned()
    };
    #[cfg(not(tb64))]
    let name = String::from("unavailable");
    name
}

/// Encodes `src` into `dst`, returning the number of bytes written.
///
/// # Safety
///
/// [`init`] must have been called, and `dst` must be at least
/// [`encoded_len`]`(src.len()) + `[`SLACK`] bytes long.
#[must_use]
pub unsafe fn encode(src: &[u8], dst: &mut [u8]) -> usize {
    #[cfg(tb64)]
    // SAFETY: the caller guarantees `dst` is large enough, including the slack the vector
    // kernels may touch past the encoded length.
    let written = unsafe { (ffi::_tb64e)(src.as_ptr(), src.len(), dst.as_mut_ptr()) };
    #[cfg(not(tb64))]
    let written = unavailable(src, dst);
    written
}

/// Decodes `src` into `dst`, returning the number of bytes written, or 0 on invalid input.
///
/// # Safety
///
/// [`init`] must have been called, and `dst` must be at least `src.len() / 4 * 3 + `
/// [`SLACK`] bytes long.
#[must_use]
pub unsafe fn decode(src: &[u8], dst: &mut [u8]) -> usize {
    #[cfg(tb64)]
    // SAFETY: as `encode`.
    let written = unsafe { (ffi::_tb64d)(src.as_ptr(), src.len(), dst.as_mut_ptr()) };
    #[cfg(not(tb64))]
    let written = unavailable(src, dst);
    written
}

/// The cost of reaching C and coming back, with no codec work in between.
///
/// See `src/ffi_floor.c`. Returns `src.len()` so the call cannot be optimised away.
///
/// # Safety
///
/// `dst` is never written, but is taken by `&mut` to keep the call shape identical to
/// [`encode`].
#[must_use]
pub unsafe fn ffi_floor(src: &[u8], dst: &mut [u8]) -> usize {
    #[cfg(tb64)]
    // SAFETY: `floor_impl` touches neither pointer.
    let n = unsafe { (ffi::tb64_floor)(src.as_ptr(), src.len(), dst.as_mut_ptr()) };
    #[cfg(not(tb64))]
    let n = unavailable(src, dst);
    n
}

#[cfg(not(tb64))]
fn unavailable(_src: &[u8], _dst: &mut [u8]) -> usize {
    panic!("tb64-sys was built without the Turbo-Base64 source; guard calls on `AVAILABLE`")
}
