// Comparison tests - all implementations against CPU Single as reference
use crate::nbody::*;
use approx::assert_relative_eq;
use crate::nbody::tests::integration_tests::compare_bodies;

const EPSILON: f32 = 0.01; // Tolerance for floating point comparisons

#[test]
fn test_cpu_single_vs_cpu_rayon() {
    let bodies = utils::generate_random_bodies(50, 100.0);
    let params = SimulationParams::default();

    let mut sim1 = CpuSingleThreaded::new(bodies.clone(), params);
    let mut sim2 = CpuMultiThreaded::new(bodies.clone(), params);

    sim1.step(5);
    sim2.step(5);

    let result1 = sim1.get_bodies();
    let result2 = sim2.get_bodies();

    compare_bodies(&result1, &result2, EPSILON);
}

#[test]
fn test_cpu_single_vs_simd_single() {
    // Use a count divisible by 8 for optimal SIMD performance
    let bodies = utils::generate_random_bodies(64, 100.0);
    let params = SimulationParams::default();

    let mut sim1 = CpuSingleThreaded::new(bodies.clone(), params);
    let mut sim2 = SimdSingleThreaded::new(bodies.clone(), params);

    sim1.step(5);
    sim2.step(5);

    let result1 = sim1.get_bodies();
    let result2 = sim2.get_bodies();

    compare_bodies(&result1, &result2, EPSILON);
}

#[test]
fn test_cpu_single_vs_simd_rayon() {
    let bodies = utils::generate_random_bodies(64, 100.0);
    let params = SimulationParams::default();

    let mut sim1 = CpuSingleThreaded::new(bodies.clone(), params);
    let mut sim2 = SimdMultiThreaded::new(bodies.clone(), params);

    sim1.step(5);
    sim2.step(5);

    let result1 = sim1.get_bodies();
    let result2 = sim2.get_bodies();

    compare_bodies(&result1, &result2, EPSILON);
}

#[test]
fn test_simd_single_vs_simd_rayon() {
    let bodies = utils::generate_random_bodies(64, 100.0);
    let params = SimulationParams::default();

    let mut sim1 = SimdSingleThreaded::new(bodies.clone(), params);
    let mut sim2 = SimdMultiThreaded::new(bodies.clone(), params);

    sim1.step(5);
    sim2.step(5);

    let result1 = sim1.get_bodies();
    let result2 = sim2.get_bodies();

    compare_bodies(&result1, &result2, EPSILON);
}

#[tokio::test]
async fn test_cpu_single_vs_gpu() {
    let bodies = utils::generate_random_bodies(11, 100.0);
    let params = SimulationParams::default();

    let mut sim1 = CpuSingleThreaded::new(bodies.clone(), params);
    let mut sim2 = GpuSimulator::new(bodies.clone(), params).await;

    sim1.step(5);
    sim2.step(5);

    let result1 = sim1.get_bodies();
    let result2 = sim2.get_bodies();

    // GPU can have slightly different results due to floating-point arithmetic
    compare_bodies(&result1, &result2, 0.1); // Tolerance for GPU floating-point differences
}

#[tokio::test]
async fn test_gpu_basic() {
    // Simple test with only 2 bodies
    let bodies = utils::generate_two_body_system();
    let params = SimulationParams::default();

    let mut sim = GpuSimulator::new(bodies.clone(), params).await;

    // A single step
    sim.step(1);
    let result = sim.get_bodies();

    // Check that bodies have moved
    assert_ne!(result[0].position, bodies[0].position);
    assert_ne!(result[1].position, bodies[1].position);

    // Compare with CPU
    let mut cpu_sim = CpuSingleThreaded::new(bodies.clone(), params);
    cpu_sim.step(1);
    let cpu_result = cpu_sim.get_bodies();

    // With 1 step they should be very similar
    compare_bodies(&cpu_result, &result, 0.01);
}

#[test]
fn test_all_implementations_consistency() {
    // Test with all CPU implementations
    let bodies = utils::generate_random_bodies(64, 100.0);
    let params = SimulationParams::default();

    let mut sim_cpu_single = CpuSingleThreaded::new(bodies.clone(), params);
    let mut sim_cpu_rayon = CpuMultiThreaded::new(bodies.clone(), params);
    let mut sim_simd_single = SimdSingleThreaded::new(bodies.clone(), params);
    let mut sim_simd_rayon = SimdMultiThreaded::new(bodies.clone(), params);

    sim_cpu_single.step(3);
    sim_cpu_rayon.step(3);
    sim_simd_single.step(3);
    sim_simd_rayon.step(3);

    let result_cpu_single = sim_cpu_single.get_bodies();
    let result_cpu_rayon = sim_cpu_rayon.get_bodies();
    let result_simd_single = sim_simd_single.get_bodies();
    let result_simd_rayon = sim_simd_rayon.get_bodies();

    // All should have identical results
    compare_bodies(&result_cpu_single, &result_cpu_rayon, EPSILON);
    compare_bodies(&result_cpu_single, &result_simd_single, EPSILON);
    compare_bodies(&result_cpu_single, &result_simd_rayon, EPSILON);
}

#[tokio::test]
async fn test_exact_symmetry_gpu() {
    // Two identical bodies, symmetrically placed
    let bodies = vec![
        Body::new([-1.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([1.0, 0.0], [0.0, 0.0], 1.0),
    ];

    let params = SimulationParams {
        dt: 0.1,
        epsilon: 0.0,
        g_constant: 1.0,
    };

    let mut sim = GpuSimulator::new(bodies.clone(), params).await;
    sim.step(1);
    let result = sim.get_bodies();

    // Due to symmetry, both bodies should have exactly the same velocities (magnitude)
    // (GPU can have minimal deviations, so slightly larger tolerance)
    assert_relative_eq!(result[0].velocity[0].abs(), result[1].velocity[0].abs(),
        epsilon = 1e-4, max_relative = 1e-4);

    // Body 0 moves to the right, Body 1 to the left
    assert!(result[0].velocity[0] > 0.0);
    assert!(result[1].velocity[0] < 0.0);

    // Y-velocities should remain close to 0
    assert!(result[0].velocity[1].abs() < 1e-4);
    assert!(result[1].velocity[1].abs() < 1e-4);
}
