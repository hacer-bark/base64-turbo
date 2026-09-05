/* Measurement control for the Turbo-Base64 benchmark candidates.
 *
 * tb64's fast entry points are global function pointers set by tb64ini, so every call
 * from Rust is an opaque, non-inlinable, indirect call that no LTO can remove. This does
 * the same thing through the same kind of pointer while doing no work at all, which gives
 * the benchmark a floor: no tb64 measurement in this harness can go below it, and at the
 * smallest sizes that floor is a visible part of the number.
 */
#include <stddef.h>

typedef size_t (*tb64_floor_func)(const unsigned char *in, size_t n, unsigned char *out);

static size_t floor_impl(const unsigned char *in, size_t n, unsigned char *out) {
  (void)in;
  (void)out;
  return n;
}

tb64_floor_func tb64_floor = floor_impl;
