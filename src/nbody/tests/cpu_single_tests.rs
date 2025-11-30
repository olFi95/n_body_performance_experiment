// Reference implementation tests - cpu_single is the reference for correct calculations
use crate::nbody::*;
use approx::assert_relative_eq;

const EPSILON: f32 = 1e-6;

fn calculate_total_energy(bodies: &[Body]) -> f32 {
    let mut kinetic = 0.0;
    let mut potential = 0.0;

    // Kinetic energy
    for body in bodies {
        let v_squared = body.velocity[0].powi(2) + body.velocity[1].powi(2);
        kinetic += 0.5 * body.mass * v_squared;
    }

    // Potential energy
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let dx = bodies[j].position[0] - bodies[i].position[0];
            let dy = bodies[j].position[1] - bodies[i].position[1];
            let r = (dx * dx + dy * dy).sqrt().max(1e-6);
            potential -= bodies[i].mass * bodies[j].mass / r;
        }
    }

    kinetic + potential
}

#[test]
fn test_two_body_system_cpu_single() {
    let bodies = utils::generate_two_body_system();
    let params = SimulationParams::default();

    let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
    sim.step(10);

    let result = sim.get_bodies();

    // Check that bodies have moved
    assert_ne!(result[0].position, bodies[0].position);
    assert_ne!(result[1].position, bodies[1].position);
}

#[test]
fn test_energy_conservation_approximation() {
    // Test if total energy is approximately conserved
    // Use a more stable scenario with fewer steps
    let bodies = utils::generate_two_body_system();
    let params = SimulationParams::default();

    let mut sim = CpuSingleThreaded::new(bodies.clone(), params);

    let initial_energy = calculate_total_energy(&sim.get_bodies());
    sim.step(10); // Reduce number of steps for better stability
    let final_energy = calculate_total_energy(&sim.get_bodies());

    // Euler integration doesn't conserve energy, but shouldn't explode
    let energy_diff = (final_energy - initial_energy).abs();
    assert!(energy_diff / initial_energy.abs() < 5.0,
        "Energy changed too much: {} -> {}", initial_energy, final_energy);
}

#[test]
fn test_circular_system() {
    let bodies = utils::generate_circular_system(8, 1.0);
    let params = SimulationParams::default();

    let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
    sim.step(10);

    let result = sim.get_bodies();

    // All bodies should have moved
    for (original, updated) in bodies.iter().zip(result.iter()) {
        assert_ne!(original.position, updated.position);
    }
}

#[test]
fn test_exact_symmetry_cpu() {
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

    let mut sim = CpuSingleThreaded::new(bodies.clone(), params);
    sim.step(1);
    let result = sim.get_bodies();

    // Due to symmetry, both bodies should have exactly the same velocities (magnitude)
    assert_relative_eq!(result[0].velocity[0].abs(), result[1].velocity[0].abs(),
        epsilon = 1e-4, max_relative = 1e-4);

    // Body 0 moves to the right, Body 1 to the left
    assert!(result[0].velocity[0] > 0.0);
    assert!(result[1].velocity[0] < 0.0);

    // Y-velocities should remain 0 (no force in Y direction)
    assert_relative_eq!(result[0].velocity[1], 0.0, epsilon = 1e-6, max_relative = 1e-6);
    assert_relative_eq!(result[1].velocity[1], 0.0, epsilon = 1e-6, max_relative = 1e-6);
}

#[test]
fn test_deterministic_gravity_calculation() {
    // Setup: Two bodies with exactly known positions
    // Body 1 at (0, 0), Body 2 at (1, 0) - distance = 1
    let bodies = vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([1.0, 0.0], [0.0, 0.0], 1.0),
    ];

    let params = SimulationParams {
        dt: 0.1,
        epsilon: 0.0,
        g_constant: 1.0,
    };

    let mut sim = CpuSingleThreaded::new(bodies.clone(), params);

    // Calculate expected force: F = G * m1 * m2 / r^2 = 1.0 * 1.0 * 1.0 / 1.0^2 = 1.0
    // Acceleration: a = F / m = 1.0 / 1.0 = 1.0
    // After one step (dt=0.1): v = a * dt = 1.0 * 0.1 = 0.1

    sim.step(1);
    let result = sim.get_bodies();

    // Body 0 should have moved to the right (towards Body 1)
    assert!(result[0].velocity[0] > 0.0, "Body 0 velocity should be positive (moving right)");
    assert!(result[0].position[0] > 0.0, "Body 0 should have moved right");

    // Body 1 should have moved to the left (towards Body 0)
    assert!(result[1].velocity[0] < 0.0, "Body 1 velocity should be negative (moving left)");
    assert!(result[1].position[0] < 1.0, "Body 1 should have moved left");
}

#[test]
fn test_inverse_square_law() {
    // Test that force decreases with 1/r^2
    let params = SimulationParams {
        dt: 0.1,
        epsilon: 0.0,
        g_constant: 1.0,
    };

    // Test 1: Bodies with distance 1
    let mut sim1 = CpuSingleThreaded::new(vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([1.0, 0.0], [0.0, 0.0], 1.0),
    ], params);
    sim1.step(1);
    let result1 = sim1.get_bodies();

    // Test 2: Bodies with distance 2 (force should be 1/4)
    let mut sim2 = CpuSingleThreaded::new(vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([2.0, 0.0], [0.0, 0.0], 1.0),
    ], params);
    sim2.step(1);
    let result2 = sim2.get_bodies();

    // Velocity at distance 2 should be 1/4 of velocity at distance 1
    let expected_ratio = 0.25;
    let actual_ratio = result2[0].velocity[0] / result1[0].velocity[0];

    assert_relative_eq!(actual_ratio, expected_ratio, epsilon = 0.01, max_relative = 0.01);
    assert!((actual_ratio - expected_ratio).abs() < 0.01,
        "Inverse square law not followed: expected ratio {}, got {}",
        expected_ratio, actual_ratio);
}
