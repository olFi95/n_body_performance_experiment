use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, AxisScale};
use monte_carlo_root::nbody::*;
use std::time::Duration;

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(5))
}

fn benchmark_scaling_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("#Bodies Scaling: All Implementations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for n in [1000, 2500, 5_000, 10_000, 25000, 50_000, 100_000].iter() {
        let bodies = utils::generate_random_bodies(*n, 100.0);
        let params = SimulationParams::default();

        group.bench_with_input(BenchmarkId::new("CPU Single", n), n, |b, _| {
            let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
            b.iter(|| {
                sim.step(black_box(1));
            });
        });

        group.bench_with_input(BenchmarkId::new("CPU Rayon", n), n, |b, _| {
            let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
            b.iter(|| {
                sim.step(black_box(1));
            });
        });

        group.bench_with_input(BenchmarkId::new("SIMD Single", n), n, |b, _| {
            let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
            b.iter(|| {
                sim.step(black_box(1));
            });
        });

        group.bench_with_input(BenchmarkId::new("SIMD Rayon", n), n, |b, _| {
            let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
            b.iter(|| {
                sim.step(black_box(1));
            });
        });

        // GPU Benchmark - Initialisierung außerhalb der Benchmark-Schleife
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut gpu_sim = rt.block_on(async {
            GpuSimulator::new(bodies.clone(), params).await
        });

        group.bench_with_input(BenchmarkId::new("GPU WGPU", n), n, |b, _| {
            b.iter(|| {
                gpu_sim.step(black_box(1));
            });
        });
    }

    group.finish();
}

fn benchmark_step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("#Steps Scaling: All Implementations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    const NUM_BODIES: usize = 10_000;
    let bodies = utils::generate_random_bodies(NUM_BODIES, 100.0);
    let params = SimulationParams::default();

    for steps in [1, 10, 50, 100, 250, 500].iter() {
        group.bench_with_input(BenchmarkId::new("CPU Single", steps), steps, |b, &s| {
            let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
            b.iter(|| {
                sim.step(black_box(s));
            });
        });

        group.bench_with_input(BenchmarkId::new("CPU Rayon", steps), steps, |b, &s| {
            let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
            b.iter(|| {
                sim.step(black_box(s));
            });
        });

        group.bench_with_input(BenchmarkId::new("SIMD Single", steps), steps, |b, &s| {
            let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
            b.iter(|| {
                sim.step(black_box(s));
            });
        });

        group.bench_with_input(BenchmarkId::new("SIMD Rayon", steps), steps, |b, &s| {
            let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
            b.iter(|| {
                sim.step(black_box(s));
            });
        });

        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut gpu_sim = rt.block_on(async {
            GpuSimulator::new(bodies.clone(), params).await
        });

        group.bench_with_input(BenchmarkId::new("GPU WGPU", steps), steps, |b, &s| {
            b.iter(|| {
                gpu_sim.step(black_box(s));
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = configure_criterion();
    targets = benchmark_scaling_comparison, benchmark_step_scaling
);

criterion_main!(benches);
