//! Throughput benchmarks comparing `base64-turbo` against the `base64`, `base64-simd`
//! and `base64-ng` crates.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    missing_docs,
    clippy::too_many_lines
)]

use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, Throughput, criterion_group,
    criterion_main,
};
use rand::RngExt;
use std::env;
use std::hint::black_box;
use std::time::Duration;

use base64_turbo::{STANDARD as TURBO_ENGINE, decoded_len_estimate, encoded_len};

// Competitors: the standard `base64` crate, `base64-simd`, and `base64-ng`.
use base64::{
    Engine as _,
    engine::{GeneralPurposeConfig, Simd},
};
use base64_simd::STANDARD as SIMD_ENGINE;
use base64_ng::STANDARD as NG_ENGINE;

fn generate_random_data(size: usize) -> Vec<u8> {
    let mut data = vec![0u8; size];
    rand::rng().fill(&mut data[..]);
    data
}

/// Helper to check if a specific engine should be benchmarked based on ENV vars.
/// Usage: `BENCH_TARGET=turbo cargo bench` or `BENCH_TARGET=all cargo bench`
fn should_run(target_name: &str) -> bool {
    let var = env::var("BENCH_TARGET").unwrap_or_else(|_| "turbo".to_string());
    let targets: Vec<String> = var.split(',').map(|s| s.trim().to_lowercase()).collect();
    if targets.contains(&"all".to_string()) {
        return true;
    }
    targets.contains(&target_name.to_lowercase())
}

fn bench_comparison(c: &mut Criterion) {
    // Runtime-detects AVX2/NEON once; matches STANDARD's alphabet/padding.
    let std_engine = Simd::standard(GeneralPurposeConfig::new());

    let mut group = c.benchmark_group("Base64_Performances");

    // Logarithmic scaling to view 32 B and 10 MB on the same axis.
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(5));
    group.noise_threshold(0.05);

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
        let input_data = generate_random_data(*size);

        // Fewer samples for large inputs to keep the bench time reasonable.
        if *size > 1_000_000 {
            group.sample_size(50);
        } else {
            group.sample_size(250);
        }

        // --- Encode ---
        group.throughput(Throughput::Bytes(*size as u64));

        // Turbo (allocating)
        if should_run("turbo") {
            group.bench_with_input(
                BenchmarkId::new("Encode/Turbo", size),
                &input_data,
                |b, d| {
                    b.iter(|| TURBO_ENGINE.encode(black_box(d)));
                },
            );
        }

        // Turbo (zero-allocation)
        if should_run("turbo-buff") {
            let encoded_len = encoded_len(*size, true).unwrap();
            let mut output_buffer = vec![0u8; encoded_len];

            group.bench_with_input(
                BenchmarkId::new("Encode/TurboBuff", size),
                &input_data,
                |b, d| {
                    b.iter(|| {
                        TURBO_ENGINE.encode_slice(black_box(d), black_box(&mut output_buffer))
                    });
                },
            );
        }

        // base64 (std)
        if should_run("std") || should_run("base64") {
            group.bench_with_input(BenchmarkId::new("Encode/Std", size), &input_data, |b, d| {
                b.iter(|| std_engine.encode(black_box(d)));
            });
        }

        // base64-simd
        if should_run("simd") {
            group.bench_with_input(
                BenchmarkId::new("Encode/Simd", size),
                &input_data,
                |b, d| {
                    b.iter(|| SIMD_ENGINE.encode_to_string(black_box(d)));
                },
            );
        }

        // base64-ng
        if should_run("ng") {
            group.bench_with_input(BenchmarkId::new("Encode/Ng", size), &input_data, |b, d| {
                b.iter(|| NG_ENGINE.encode_string(black_box(d)));
            });
        }

        // --- Decode ---

        let encoded_str = std_engine.encode(&input_data);

        // Throughput is measured against the encoded (input) text size.
        group.throughput(Throughput::Bytes(encoded_str.len() as u64));

        // Turbo (allocating)
        if should_run("turbo") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Turbo", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| TURBO_ENGINE.decode(black_box(s)));
                },
            );
        }

        // Turbo (zero-allocation)
        if should_run("turbo-buff") {
            let decoded_len = decoded_len_estimate(encoded_str.len());
            let mut output_buffer = vec![0u8; decoded_len];

            group.bench_with_input(
                BenchmarkId::new("Decode/TurboBuff", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| {
                        TURBO_ENGINE
                            .decode_slice(black_box(s.as_bytes()), black_box(&mut output_buffer))
                    });
                },
            );
        }

        // base64 (std)
        if should_run("std") || should_run("base64") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Std", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| std_engine.decode(black_box(s)));
                },
            );
        }

        // base64-simd
        if should_run("simd") {
            group.bench_with_input(
                BenchmarkId::new("Decode/Simd", size),
                &encoded_str,
                |b, s| {
                    b.iter(|| SIMD_ENGINE.decode_to_vec(black_box(s)));
                },
            );
        }

        // base64-ng
        if should_run("ng") {
            group.bench_with_input(BenchmarkId::new("Decode/Ng", size), &encoded_str, |b, s| {
                b.iter(|| NG_ENGINE.decode_vec(black_box(s.as_bytes())));
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
