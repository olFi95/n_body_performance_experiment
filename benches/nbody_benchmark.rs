use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration, AxisScale};
use monte_carlo_root::nbody::*;
use std::time::Duration;

// Konfiguration für bessere Plots
fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(5))
}

fn benchmark_scaling_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaling: All Implementations");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // Deutlich größere Körperanzahlen für realistische Benchmarks
    for n in [1000, 2500, 5000, 10000, 25000, 50000].iter() {
        let bodies = utils::generate_random_bodies(*n, 100.0);
        let params = SimulationParams::default();

        group.bench_with_input(BenchmarkId::new("CPU Single", n), n, |b, _| {
            b.iter(|| {
                let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
                sim.step(black_box(1));
            });
        });

        group.bench_with_input(BenchmarkId::new("CPU Rayon", n), n, |b, _| {
            b.iter(|| {
                let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
                sim.step(black_box(1));
            });
        });

        group.bench_with_input(BenchmarkId::new("SIMD Single", n), n, |b, _| {
            b.iter(|| {
                let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
                sim.step(black_box(1));
            });
        });

        group.bench_with_input(BenchmarkId::new("SIMD Rayon", n), n, |b, _| {
            b.iter(|| {
                let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
                sim.step(black_box(1));
            });
        });
    }

    group.finish();
}

fn benchmark_cpu_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPU Single-threaded");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for n in [1000, 5000, 10000, 25000, 50000].iter() {
        let bodies = utils::generate_random_bodies(*n, 100.0);
        let params = SimulationParams::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
                sim.step(black_box(1));
            });
        });
    }

    group.finish();
}

fn benchmark_cpu_rayon(c: &mut Criterion) {
    let mut group = c.benchmark_group("CPU Rayon Multi-threaded");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for n in [1000, 5000, 10000, 25000, 50000].iter() {
        let bodies = utils::generate_random_bodies(*n, 100.0);
        let params = SimulationParams::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
                sim.step(black_box(1));
            });
        });
    }

    group.finish();
}

fn benchmark_simd_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Single-threaded");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // SIMD funktioniert am besten mit Vielfachen von 8
    for n in [1024, 2048, 5000, 10000, 25000, 50000].iter() {
        let bodies = utils::generate_random_bodies(*n, 100.0);
        let params = SimulationParams::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
                sim.step(black_box(1));
            });
        });
    }

    group.finish();
}

fn benchmark_simd_rayon(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Rayon Multi-threaded");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for n in [1024, 2048, 5000, 10000, 25000, 50000].iter() {
        let bodies = utils::generate_random_bodies(*n, 100.0);
        let params = SimulationParams::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            b.iter(|| {
                let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
                sim.step(black_box(1));
            });
        });
    }

    group.finish();
}

fn benchmark_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU WGPU");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    // GPU funktioniert am besten mit größeren Workloads
    for n in [1000, 5000, 10000, 25000, 50000].iter() {
        let bodies = utils::generate_random_bodies(*n, 100.0);
        let params = SimulationParams::default();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, _| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let mut sim = rt.block_on(async {
                GpuSimulator::new(bodies.clone(), params).await
            });

            b.iter(|| {
                sim.step(black_box(1));
            });
        });
    }

    group.finish();
}

fn benchmark_gpu_vs_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU vs CPU/SIMD - 10000 bodies");
    let bodies = utils::generate_random_bodies(10000, 100.0);
    let params = SimulationParams::default();

    group.bench_function("CPU Single", |b| {
        b.iter(|| {
            let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("CPU Rayon", |b| {
        b.iter(|| {
            let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("SIMD Single", |b| {
        b.iter(|| {
            let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("SIMD Rayon", |b| {
        b.iter(|| {
            let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("GPU WGPU", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut sim = rt.block_on(async {
            GpuSimulator::new(bodies.clone(), params).await
        });

        b.iter(|| {
            sim.step(black_box(1));
        });
    });

    group.finish();
}

fn benchmark_gpu_multiple_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU Multiple Steps - 10000 bodies");
    let n = 10000;
    let bodies = utils::generate_random_bodies(n, 100.0);
    let params = SimulationParams::default();

    for steps in [1, 5, 10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(steps),
            steps,
            |b, &steps| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let mut sim = rt.block_on(async {
                    GpuSimulator::new(bodies.clone(), params).await
                });

                b.iter(|| {
                    sim.step(black_box(steps));
                });
            },
        );
    }

    group.finish();
}

fn benchmark_multiple_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("Multiple Steps - 10000 bodies");
    let n = 10000;
    let bodies = utils::generate_random_bodies(n, 100.0);
    let params = SimulationParams::default();

    for steps in [1, 5, 10, 25, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("CPU Single", steps),
            steps,
            |b, &steps| {
                b.iter(|| {
                    let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
                    sim.step(black_box(steps));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("CPU Rayon", steps),
            steps,
            |b, &steps| {
                b.iter(|| {
                    let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
                    sim.step(black_box(steps));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SIMD Single", steps),
            steps,
            |b, &steps| {
                b.iter(|| {
                    let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
                    sim.step(black_box(steps));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SIMD Rayon", steps),
            steps,
            |b, &steps| {
                b.iter(|| {
                    let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
                    sim.step(black_box(steps));
                });
            },
        );
    }

    group.finish();
}

fn benchmark_direct_comparison_small(c: &mut Criterion) {
    let mut group = c.benchmark_group("Direct Comparison - 1000 bodies, 1 step");
    let bodies = utils::generate_random_bodies(1000, 100.0);
    let params = SimulationParams::default();

    group.bench_function("CPU Single", |b| {
        b.iter(|| {
            let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("CPU Rayon", |b| {
        b.iter(|| {
            let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("SIMD Single", |b| {
        b.iter(|| {
            let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("SIMD Rayon", |b| {
        b.iter(|| {
            let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.finish();
}

fn benchmark_direct_comparison_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("Direct Comparison - 10000 bodies, 1 step");
    let bodies = utils::generate_random_bodies(10000, 100.0);
    let params = SimulationParams::default();

    group.bench_function("CPU Single", |b| {
        b.iter(|| {
            let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("CPU Rayon", |b| {
        b.iter(|| {
            let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("SIMD Single", |b| {
        b.iter(|| {
            let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("SIMD Rayon", |b| {
        b.iter(|| {
            let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.finish();
}

fn benchmark_direct_comparison_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("Direct Comparison - 50000 bodies, 1 step");
    let bodies = utils::generate_random_bodies(50000, 100.0);
    let params = SimulationParams::default();

    group.bench_function("CPU Single", |b| {
        b.iter(|| {
            let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("CPU Rayon", |b| {
        b.iter(|| {
            let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("SIMD Single", |b| {
        b.iter(|| {
            let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.bench_function("SIMD Rayon", |b| {
        b.iter(|| {
            let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.finish();
}

fn benchmark_speedup_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Speedup Analysis - 25000 bodies");
    let bodies = utils::generate_random_bodies(25000, 100.0);
    let params = SimulationParams::default();

    // Baseline: CPU Single
    group.bench_function("1_Baseline_CPU_Single", |b| {
        b.iter(|| {
            let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    // Rayon Parallelisierung
    group.bench_function("2_Rayon_Only", |b| {
        b.iter(|| {
            let mut sim = CpuMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    // SIMD Vektorisierung
    group.bench_function("3_SIMD_Only", |b| {
        b.iter(|| {
            let mut sim = SimdSingleThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    // Kombiniert: SIMD + Rayon
    group.bench_function("4_SIMD_Plus_Rayon", |b| {
        b.iter(|| {
            let mut sim = SimdMultiThreaded::new(bodies.clone(), params);
            sim.step(black_box(1));
        });
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = configure_criterion();
    targets =
        benchmark_scaling_comparison,
        benchmark_speedup_analysis,
        benchmark_cpu_single,
        benchmark_cpu_rayon,
        benchmark_simd_single,
        benchmark_simd_rayon,
        benchmark_gpu,
        benchmark_gpu_vs_all,
        benchmark_gpu_multiple_steps,
        benchmark_multiple_steps,
        benchmark_direct_comparison_small,
        benchmark_direct_comparison_medium,
        benchmark_direct_comparison_large,
);

criterion_main!(benches);
