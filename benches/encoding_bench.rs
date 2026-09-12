//! Throughput benchmarks comparing `base64-turbo` against the `base64`, `base64-simd`
//! and `base64-ng` crates, and against the Turbo-Base64 C library when a checkout of it
//! is present (see `benches/tb64-sys/README.md`).
//!
//! Every candidate is driven through its fastest zero-allocation slice API, writing into
//! a pre-sized buffer that is reused across iterations, so the numbers reflect codec work
//! rather than allocator behaviour.
//!
//! Two rules keep the C competitor on equal footing:
//!
//! * every buffer carries `tb64_sys::SLACK` trailing bytes and every candidate is handed
//!   an exact-length slice of it. tb64's vector kernels may touch past the end of a
//!   buffer — upstream's own driver never runs them against an exactly sized one — so
//!   without the slack tb64 would be unsound here, and giving the slack to tb64 alone
//!   would change its cache footprint relative to everyone else;
//!
//! Because linking the C library moves every symbol in the binary, comparisons are only
//! meaningful within one build: keep the tb64 checkout in place for every run you compare,
//! and select candidates with `BENCH_TARGET`, not by rebuilding without it.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    missing_docs,
    clippy::too_many_lines
)]

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, SamplingMode, Throughput,
    criterion_group, criterion_main,
};
use rand::RngExt;
use std::env;
use std::hint::black_box;
use std::time::Duration;

use base64_turbo::STANDARD as TURBO_ENGINE;

// Competitors: the standard `base64` crate, `base64-simd`, and `base64-ng`.
use base64::{
    Engine as _,
    engine::{GeneralPurposeConfig, Simd},
};
use base64_ng::STANDARD as NG_ENGINE;
use base64_simd::{AsOut, STANDARD as SIMD_ENGINE};
use tb64_sys as tb64;

/// Allocates `len` usable bytes plus the trailing slack every candidate is given.
fn slack_buf(len: usize) -> Vec<u8> {
    vec![0u8; len + tb64::SLACK]
}

/// Helper to check if a specific engine should be benchmarked based on ENV vars.
/// Usage: `BENCH_TARGET=turbo cargo bench` or `BENCH_TARGET=all cargo bench`
/// Targets: `turbo`, `std`, `simd`, `ng`, `tb64`, and the control `memcpy`
/// (byte-copy roofline), which is not a codec.
fn should_run(target_name: &str) -> bool {
    let var = env::var("BENCH_TARGET").unwrap_or_else(|_| "turbo".to_string());
    let targets: Vec<String> = var.split(',').map(|s| s.trim().to_lowercase()).collect();
    if targets.contains(&"all".to_string()) {
        return true;
    }
    targets.contains(&target_name.to_lowercase())
}

/// Runs every candidate once through the benchmarked API and checks it against the
/// reference encoding, so a benchmark can never time an error path or a short write.
fn verify_candidates(
    std_engine: &Simd,
    input: &[u8],
    encoded: &[u8],
    encode_buf: &mut [u8],
    decode_buf: &mut [u8],
) {
    let expect_enc = |written: usize, buf: &[u8]| {
        assert_eq!(&buf[..written], encoded);
    };
    let expect_dec = |written: usize, buf: &[u8]| {
        assert_eq!(&buf[..written], input);
    };

    expect_enc(
        TURBO_ENGINE.encode_slice(input, encode_buf).unwrap(),
        encode_buf,
    );
    expect_enc(
        std_engine.encode_slice(input, encode_buf).unwrap(),
        encode_buf,
    );
    expect_enc(
        SIMD_ENGINE.encode(input, encode_buf.as_out()).len(),
        encode_buf,
    );
    expect_enc(
        NG_ENGINE.encode_slice(input, encode_buf).unwrap(),
        encode_buf,
    );

    let src = encoded;
    expect_dec(
        TURBO_ENGINE.decode_slice(src, decode_buf).unwrap(),
        decode_buf,
    );
    expect_dec(
        std_engine.decode_slice(src, decode_buf).unwrap(),
        decode_buf,
    );
    expect_dec(
        SIMD_ENGINE.decode(src, decode_buf.as_out()).unwrap().len(),
        decode_buf,
    );
    expect_dec(NG_ENGINE.decode_slice(src, decode_buf).unwrap(), decode_buf);

    if tb64::AVAILABLE {
        // SAFETY: `tb64::init` ran in `bench_comparison`, and both buffers were allocated
        // with `tb64::SLACK` trailing bytes.
        unsafe {
            expect_enc(tb64::encode(input, encode_buf), encode_buf);
            expect_dec(tb64::decode(src, decode_buf), decode_buf);
        }
    }
}

fn bench_comparison(c: &mut Criterion) {
    // Runtime-detects AVX2/NEON once; matches STANDARD's alphabet/padding.
    let std_engine = Simd::standard(GeneralPurposeConfig::new());

    // tb64 dispatches through globals that stay at the scalar fallback until this runs.
    // Once, outside every timed loop.
    if tb64::AVAILABLE {
        tb64::init();
        println!("tb64: linked, simd set = {}", tb64::isa());
    } else {
        println!("tb64: not linked, candidates skipped (see benches/tb64-sys/README.md)");
    }

    let mut group = c.benchmark_group("Base64_Performances");

    // Logarithmic scaling to view 32 B and 10 MB on the same axis.
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.warm_up_time(Duration::from_secs(5));
    group.noise_threshold(0.05);
    group.confidence_level(0.99);
    group.significance_level(0.01);

    let sizes = [
        32,               // 32 B
        512,              // 512 B
        4 * 1024,         // 4 KB
        64 * 1024,        // 64 KB
        512 * 1024,       // 512 KB
        1024 * 1024,      // 1 MB
        10 * 1024 * 1024, // 10 MB
    ];

    for size in &sizes {
        // Source buffers carry the same trailing slack as the destinations, so a candidate
        // that reads past the end of its input stays in bounds. Every candidate is handed
        // the exact-length slice, so nobody sees a longer input than anybody else.
        let mut input_store = slack_buf(*size);
        rand::rng().fill(&mut input_store[..*size]);
        let input = &input_store[..*size];

        let encoded_str = std_engine.encode(input);
        let mut encoded_store = slack_buf(encoded_str.len());
        encoded_store[..encoded_str.len()].copy_from_slice(encoded_str.as_bytes());
        let encoded = &encoded_store[..encoded_str.len()];

        // Shared destination buffers, allocated once per size and reused by every
        // candidate so no engine pays for a `Vec`/`String` inside the timed loop.
        let mut encode_buf = slack_buf(TURBO_ENGINE.encoded_len(*size));
        let mut decode_buf = slack_buf(TURBO_ENGINE.decoded_len_estimate(encoded.len()));

        // Guard against timing a silently failing call: every candidate must reproduce the
        // reference result through the exact API and buffers the benchmark uses.
        verify_candidates(
            &std_engine,
            input,
            encoded,
            &mut encode_buf,
            &mut decode_buf,
        );

        // Per-size measurement knobs. A single setting cannot serve both a 22 ns and a
        // ~1 ms iteration, so each tier gets what it actually needs:
        //
        // * tiny/small — dominated by timer resolution and code alignment, so take many
        //   samples and let Criterion's linear regression average the per-call overhead out;
        // * mid — still cheap enough for linear sampling, fewer samples suffice;
        // * large — bandwidth bound and thermally sensitive. Linear sampling would ramp the
        //   iteration count quadratically and spend most of the budget on the longest
        //   samples, so use flat sampling: equal iterations per sample, evenly spread over
        //   the measurement window.
        let (samples, measurement_secs, mode) = match *size {
            0..=4096 => (300, 8, SamplingMode::Linear),
            4097..=524_288 => (200, 10, SamplingMode::Linear),
            _ => (100, 12, SamplingMode::Flat),
        };
        group.sample_size(samples);
        group.measurement_time(Duration::from_secs(measurement_secs));
        group.sampling_mode(mode);

        // Encode.
        group.throughput(Throughput::Bytes(*size as u64));

        // Roofline reference: a plain byte copy of the same input, so every encode number
        // can be read against the cost of just moving the bytes.
        if should_run("memcpy") {
            group.bench_with_input(BenchmarkId::new("Encode/Memcpy", size), &input, |b, d| {
                b.iter(|| {
                    let src = black_box(d);
                    black_box(&mut encode_buf)[..src.len()].copy_from_slice(src);
                });
            });
        }

        if should_run("turbo") {
            group.bench_with_input(BenchmarkId::new("Encode/Turbo", size), &input, |b, d| {
                b.iter(|| TURBO_ENGINE.encode_slice(black_box(d), black_box(&mut encode_buf)));
            });
        }

        // Turbo-Base64 (C). `_tb64e` is the entry point upstream documents as fastest,
        // skipping the wrapper call and its dispatch check.
        if tb64::AVAILABLE && should_run("tb64") {
            group.bench_with_input(BenchmarkId::new("Encode/Tb64", size), &input, |b, d| {
                // SAFETY: `encode_buf` carries `tb64::SLACK` trailing bytes.
                b.iter(|| unsafe { tb64::encode(black_box(d), black_box(&mut encode_buf)) });
            });
        }

        // base64 (std)
        if should_run("std") || should_run("base64") {
            group.bench_with_input(BenchmarkId::new("Encode/Std", size), &input, |b, d| {
                b.iter(|| std_engine.encode_slice(black_box(d), black_box(&mut encode_buf)));
            });
        }

        // base64-simd
        if should_run("simd") {
            group.bench_with_input(BenchmarkId::new("Encode/Simd", size), &input, |b, d| {
                b.iter(|| {
                    SIMD_ENGINE
                        .encode(black_box(d), black_box(&mut encode_buf).as_out())
                        .len()
                });
            });
        }

        // base64-ng
        if should_run("ng") {
            group.bench_with_input(BenchmarkId::new("Encode/Ng", size), &input, |b, d| {
                b.iter(|| NG_ENGINE.encode_slice(black_box(d), black_box(&mut encode_buf)));
            });
        }

        // Decode.

        // Throughput is measured against the encoded (input) text size.
        group.throughput(Throughput::Bytes(encoded.len() as u64));

        // Same reference for the decode side: `encode_buf` is exactly the encoded length,
        // so this moves the same byte count the decoders read.
        if should_run("memcpy") {
            group.bench_with_input(BenchmarkId::new("Decode/Memcpy", size), &encoded, |b, s| {
                b.iter(|| {
                    let src = black_box(s);
                    black_box(&mut encode_buf)[..src.len()].copy_from_slice(src);
                });
            });
        }

        if should_run("turbo") {
            group.bench_with_input(BenchmarkId::new("Decode/Turbo", size), &encoded, |b, s| {
                b.iter(|| TURBO_ENGINE.decode_slice(black_box(s), black_box(&mut decode_buf)));
            });
        }

        // Turbo-Base64 (C); see the encode side.
        if tb64::AVAILABLE && should_run("tb64") {
            group.bench_with_input(BenchmarkId::new("Decode/Tb64", size), &encoded, |b, s| {
                // SAFETY: `decode_buf` carries `tb64::SLACK` trailing bytes.
                b.iter(|| unsafe { tb64::decode(black_box(s), black_box(&mut decode_buf)) });
            });
        }

        // base64 (std)
        if should_run("std") || should_run("base64") {
            group.bench_with_input(BenchmarkId::new("Decode/Std", size), &encoded, |b, s| {
                b.iter(|| std_engine.decode_slice(black_box(s), black_box(&mut decode_buf)));
            });
        }

        // base64-simd
        if should_run("simd") {
            group.bench_with_input(BenchmarkId::new("Decode/Simd", size), &encoded, |b, s| {
                b.iter(|| {
                    SIMD_ENGINE
                        .decode(black_box(s), black_box(&mut decode_buf).as_out())
                        .map(|out| out.len())
                });
            });
        }

        // base64-ng
        if should_run("ng") {
            group.bench_with_input(BenchmarkId::new("Decode/Ng", size), &encoded, |b, s| {
                b.iter(|| NG_ENGINE.decode_slice(black_box(s), black_box(&mut decode_buf)));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
