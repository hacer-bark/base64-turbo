# tb64-sys

A dev-only FFI shim that lets the `base64-turbo` benchmarks call
[Turbo-Base64](https://github.com/powturbo/Turbo-Base64) as a competitor.

**No Turbo-Base64 code is vendored here.** Turbo-Base64 is GPL-3; `base64-turbo` is 0BSD.
This directory contains only original 0BSD code: a build script, four `extern "C"`
declarations, and one small C file (`src/ffi_floor.c`) used as a measurement control.

## Enabling the tb64 benchmark candidates

Clone the library into `target/` (git-ignored, never packaged, never distributed):

```sh
git clone --depth 1 https://github.com/powturbo/Turbo-Base64 target/tb64-src
```

Set `TB64_SRC` to use a checkout elsewhere. Without it the shim compiles to stubs and
the benchmark simply skips the tb64 candidates.

The linked benchmark binary is a combined work with GPL-3 code, so it stays in `target/`:
do not distribute it or publish it as a CI artifact. Measurements are facts and are not
covered by the GPL.
