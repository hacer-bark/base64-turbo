use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use std::hint::black_box;
use rand::Rng;
use std::env;
use std::time::Duration;

// 1. The Base64-turbo
use base64_turbo::STANDARD as TURBO_ENGINE;

// 2. Competitor 1: The standard 'base64' crate
use base64::{prelude::BASE64_STANDARD as STD_ENGINE, Engine as _};

// 3. Competitor 2: The 'base64-simd' crate
use base64_simd::STANDARD as SIMD_ENGINE;

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
    let mut group = c.benchmark_group("Base64_Performances");

    // Logarithmic scaling is essential for viewing 32B vs 10MB
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

    for size in sizes.iter() {
        let input_data = generate_random_data(*size);

        // Dynamic configuration: Reduce sample count for large files to keep bench time reasonable
        if *size > 1_000_000 {
            group.sample_size(50);
        } else {
            group.sample_size(250);
        }

        // ======================================================================
        // ENCODE
        // ======================================================================
        group.throughput(Throughput::Bytes(*size as u64));

        // 1a. Base64 Turbo (Allocating)
        if should_run("turbo") {
            group.bench_with_input(BenchmarkId::new("Encode/Turbo", size), &input_data, |b, d| {
                b.iter(|| TURBO_ENGINE.encode(black_box(d)))
            });
        }

        // 1b. Base64 Turbo (Buff / No-Alloc)
        if should_run("turbo-buff") {
            let encoded_len = TURBO_ENGINE.encoded_len(*size);
            let mut output_buffer = vec![0u8; encoded_len];

            group.bench_with_input(BenchmarkId::new("Encode/TurboBuff", size), &input_data, |b, d| {
                b.iter(|| TURBO_ENGINE.encode_into(black_box(d), black_box(&mut output_buffer)))
            });
        }

        // 2. Base64 Standard
        if should_run("std") || should_run("base64") {
            group.bench_with_input(BenchmarkId::new("Encode/Std", size), &input_data, |b, d| {
                b.iter(|| STD_ENGINE.encode(black_box(d)))
            });
        }

        // 3. Base64 SIMD
        if should_run("simd") {
            group.bench_with_input(BenchmarkId::new("Encode/Simd", size), &input_data, |b, d| {
                b.iter(|| SIMD_ENGINE.encode_to_string(black_box(d)))
            });
        }

        // ======================================================================
        // DECODE
        // ======================================================================

        // Prepare valid Base64 string for decoding
        let encoded_str = STD_ENGINE.encode(&input_data);

        // We measure throughput based on the INPUT text size (bytes processed per second)
        group.throughput(Throughput::Bytes(encoded_str.len() as u64));

        // 1a. Base64 Turbo Decode (Allocating)
        if should_run("turbo") {
            group.bench_with_input(BenchmarkId::new("Decode/Turbo", size), &encoded_str, |b, s| {
                b.iter(|| TURBO_ENGINE.decode(black_box(s)))
            });
        }

        // 1b. Base64 Turbo Decode (Buff / No-Alloc)
        if should_run("turbo-buff") {
            let decoded_len = TURBO_ENGINE.estimate_decoded_len(encoded_str.len());
            let mut output_buffer = vec![0u8; decoded_len];

            group.bench_with_input(BenchmarkId::new("Decode/TurboBuff", size), &encoded_str, |b, s| {
                b.iter(|| TURBO_ENGINE.decode_into(black_box(s.as_bytes()), black_box(&mut output_buffer)))
            });
        }

        // 2. Base64 Standard Decode
        if should_run("std") || should_run("base64") {
            group.bench_with_input(BenchmarkId::new("Decode/Std", size), &encoded_str, |b, s| {
                b.iter(|| STD_ENGINE.decode(black_box(s)))
            });
        }

        // 3. Base64 SIMD Decode
        if should_run("simd") {
            group.bench_with_input(BenchmarkId::new("Decode/Simd", size), &encoded_str, |b, s| {
                b.iter(|| SIMD_ENGINE.decode_to_vec(black_box(s)))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_comparison);
criterion_main!(benches);
