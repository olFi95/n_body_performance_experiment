// Integration tests - set_params and set_bodies for all implementations
use crate::nbody::*;
use approx::assert_relative_eq;
use crate::nbody::shader_types::nbody::{Body, SimulationParams};

pub fn compare_bodies(bodies1: &[Body], bodies2: &[Body], tolerance: f32) {
    assert_eq!(bodies1.len(), bodies2.len(), "Number of bodies doesn't match");

    for (_, (b1, b2)) in bodies1.iter().zip(bodies2.iter()).enumerate() {
        assert_relative_eq!(b1.position[0], b2.position[0], epsilon = tolerance,
            max_relative = tolerance);
        assert_relative_eq!(b1.position[1], b2.position[1], epsilon = tolerance,
            max_relative = tolerance);
        assert_relative_eq!(b1.velocity[0], b2.velocity[0], epsilon = tolerance,
            max_relative = tolerance);
        assert_relative_eq!(b1.velocity[1], b2.velocity[1], epsilon = tolerance,
            max_relative = tolerance);
        assert_relative_eq!(b1.mass, b2.mass, epsilon = tolerance,
            max_relative = tolerance);
    }
}

// ========== set_params Tests ==========

#[test]
fn test_set_params_cpu_single() {
    let bodies = utils::generate_random_bodies(10, 100.0);
    let params = SimulationParams::default();

    let mut sim = CpuSingleThreaded::new(bodies.clone(), params);

    // Step 1: One step with default params
    sim.step(1);
    let bodies_after_step1 = sim.get_bodies();

    // Step 2: Set new parameters
    let new_params = SimulationParams {
        dt: 0.032,  // Double timestep
        epsilon: 1e-5,
        g_constant: 2.0,  // Double gravity
    };
    sim.set_params(new_params);

    // Step 3: Read parameters back and verify
    let read_params = sim.get_params();
    assert_eq!(read_params.dt, new_params.dt);
    assert_eq!(read_params.epsilon, new_params.epsilon);
    assert_eq!(read_params.g_constant, new_params.g_constant);

    // Step 4: Another step with new params
    sim.step(1);
    let bodies_after_step2 = sim.get_bodies();

    // Bodies should have moved differently
    assert_ne!(bodies_after_step1[0].position, bodies_after_step2[0].position);
}

#[tokio::test]
async fn test_set_params_gpu() {
    let bodies = utils::generate_random_bodies(10, 100.0);
    let params = SimulationParams::default();

    let mut sim = GpuSimulator::new(bodies.clone(), params).await;

    // Step 1: One step with default params
    sim.step(1);
    let bodies_after_step1 = sim.get_bodies();

    // Step 2: Set new parameters
    let new_params = SimulationParams {
        dt: 0.032,  // Double timestep
        epsilon: 1e-5,
        g_constant: 2.0,  // Double gravity
    };
    sim.set_params(new_params);

    // Step 3: Read parameters back and verify
    let read_params = sim.get_params();
    assert_eq!(read_params.dt, new_params.dt);
    assert_eq!(read_params.epsilon, new_params.epsilon);
    assert_eq!(read_params.g_constant, new_params.g_constant);

    // Step 4: Another step with new params
    sim.step(1);
    let bodies_after_step2 = sim.get_bodies();

    // Bodies should have moved differently
    assert_ne!(bodies_after_step1[0].position, bodies_after_step2[0].position);
}

// ========== set_bodies Tests ==========

#[test]
fn test_set_bodies_cpu_single() {
    let bodies = utils::generate_random_bodies(10, 100.0);
    let params = SimulationParams::default();

    let mut sim = CpuSingleThreaded::new(bodies.clone(), params);

    // Step 1: One step
    sim.step(1);

    // Step 2: Set new bodies
    let new_bodies = utils::generate_two_body_system();
    sim.set_bodies(new_bodies.clone());

    // Step 3: Read bodies back and verify
    let read_bodies = sim.get_bodies();
    assert_eq!(read_bodies.len(), new_bodies.len());
    compare_bodies(&read_bodies, &new_bodies, 1e-6);

    // Step 4: Another step with new bodies
    sim.step(1);
    let bodies_after_step = sim.get_bodies();

    // Bodies should have moved
    assert_ne!(bodies_after_step[0].position, new_bodies[0].position);
    assert_ne!(bodies_after_step[1].position, new_bodies[1].position);
}

#[tokio::test]
async fn test_set_bodies_gpu() {
    let bodies = utils::generate_random_bodies(10, 100.0);
    let params = SimulationParams::default();

    let mut sim = GpuSimulator::new(bodies.clone(), params).await;

    // Step 1: One step
    sim.step(1);

    // Step 2: Set new bodies
    let new_bodies = utils::generate_two_body_system();
    sim.set_bodies(new_bodies.clone());

    // Step 3: Read bodies back and verify
    let read_bodies = sim.get_bodies();
    assert_eq!(read_bodies.len(), new_bodies.len());
    compare_bodies(&read_bodies, &new_bodies, 1e-6);

    // Step 4: Another step with new bodies
    sim.step(1);
    let bodies_after_step = sim.get_bodies();

    // Bodies should have moved
    assert_ne!(bodies_after_step[0].position, new_bodies[0].position);
    assert_ne!(bodies_after_step[1].position, new_bodies[1].position);
}

// ========== Deterministic Tests ==========

#[test]
fn test_set_params_deterministic_cpu() {
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

    sim.step(1);
    let result = sim.get_bodies();

    // Body 0 should have moved to the right (towards Body 1)
    assert!(result[0].velocity[0] > 0.0, "Body 0 velocity should be positive (moving right)");
    assert!(result[0].position[0] > 0.0, "Body 0 should have moved right");

    // Body 1 should have moved to the left (towards Body 0)
    assert!(result[1].velocity[0] < 0.0, "Body 1 velocity should be negative (moving left)");
    assert!(result[1].position[0] < 1.0, "Body 1 should have moved left");

    // Now change parameters: double gravity
    let new_params = SimulationParams {
        dt: 0.1,
        epsilon: 0.0,
        g_constant: 2.0,  // Double force
    };
    sim.set_params(new_params);

    // Reset bodies
    sim.set_bodies(bodies.clone());
    sim.step(1);
    let result_doubled = sim.get_bodies();

    // With doubled gravity, velocity should be significantly higher
    // Original: v ≈ 0.1, New: v ≈ 0.2
    assert!(result_doubled[0].velocity[0] > result[0].velocity[0] * 1.5,
        "Doubled gravity should produce significantly higher velocity: {} vs {}",
        result_doubled[0].velocity[0], result[0].velocity[0]);
}

#[tokio::test]
async fn test_set_params_deterministic_gpu() {
    // Setup: Two bodies with exactly known positions
    let bodies = vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
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

    // Body 0 should have moved to the right
    assert!(result[0].velocity[0] > 0.0, "Body 0 velocity should be positive (moving right)");
    assert!(result[0].position[0] > 0.0, "Body 0 should have moved right");

    // Body 1 should have moved to the left (towards Body 0)
    assert!(result[1].velocity[0] < 0.0, "Body 1 velocity should be negative (moving left)");
    assert!(result[1].position[0] < 1.0, "Body 1 should have moved left");

    // Now change parameters: double gravity
    let new_params = SimulationParams {
        dt: 0.1,
        epsilon: 0.0,
        g_constant: 2.0,
    };
    sim.set_params(new_params);

    // Reset bodies
    sim.set_bodies(bodies.clone());
    sim.step(1);
    let result_doubled = sim.get_bodies();

    // With doubled gravity, velocity should be about twice as high
    assert!(result_doubled[0].velocity[0] > result[0].velocity[0] * 1.5,
        "Doubled gravity should produce significantly higher velocity: {} vs {}",
        result_doubled[0].velocity[0], result[0].velocity[0]);
}

#[test]
fn test_set_bodies_deterministic_cpu() {
    // Start with 3 bodies
    let initial_bodies = vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([1.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([0.5, 1.0], [0.0, 0.0], 1.0),
    ];

    let params = SimulationParams {
        dt: 0.1,
        epsilon: 0.0,
        g_constant: 1.0,
    };

    let mut sim = CpuSingleThreaded::new(initial_bodies.clone(), params);
    sim.step(1);

    // Replace with only 2 bodies at greater distance
    let new_bodies = vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([2.0, 0.0], [0.0, 0.0], 1.0),  // Distance = 2
    ];
    sim.set_bodies(new_bodies.clone());

    // Verify that only 2 bodies exist
    let read_bodies = sim.get_bodies();
    assert_eq!(read_bodies.len(), 2, "Should have exactly 2 bodies after set_bodies");

    sim.step(1);
    let result = sim.get_bodies();

    // For comparison: simulate with distance 1
    let mut sim_close = CpuSingleThreaded::new(vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([1.0, 0.0], [0.0, 0.0], 1.0),
    ], params);
    sim_close.step(1);
    let result_close = sim_close.get_bodies();

    // Velocity at distance 2 should be significantly lower than at distance 1
    assert!(result[0].velocity[0] < result_close[0].velocity[0] * 0.5,
        "Bodies with distance 2 should have much lower velocity than distance 1: {} vs {}",
        result[0].velocity[0], result_close[0].velocity[0]);
}

#[tokio::test]
async fn test_set_bodies_deterministic_gpu() {
    // Start with 3 bodies
    let initial_bodies = vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([1.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([0.5, 1.0], [0.0, 0.0], 1.0),
    ];

    let params = SimulationParams {
        dt: 0.1,
        epsilon: 0.0,
        g_constant: 1.0,
    };

    let mut sim = GpuSimulator::new(initial_bodies.clone(), params).await;
    sim.step(1);

    // Replace with only 2 bodies at greater distance
    let new_bodies = vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([2.0, 0.0], [0.0, 0.0], 1.0),  // Distance = 2
    ];
    sim.set_bodies(new_bodies.clone());

    // Verify that only 2 bodies exist
    let read_bodies = sim.get_bodies();
    assert_eq!(read_bodies.len(), 2, "Should have exactly 2 bodies after set_bodies");

    sim.step(1);
    let result = sim.get_bodies();

    // For comparison: simulate with distance 1
    let mut sim_close = GpuSimulator::new(vec![
        Body::new([0.0, 0.0], [0.0, 0.0], 1.0),
        Body::new([1.0, 0.0], [0.0, 0.0], 1.0),
    ], params).await;
    sim_close.step(1);
    let result_close = sim_close.get_bodies();

    // Velocity at distance 2 should be significantly lower
    assert!(result[0].velocity[0] < result_close[0].velocity[0] * 0.5,
        "Bodies with distance 2 should have much lower velocity than distance 1: {} vs {}",
        result[0].velocity[0], result_close[0].velocity[0]);
}

#[tokio::test]
async fn test_params_synchronization_gpu() {
    // This test verifies that params actually reach the GPU
    let bodies = utils::generate_two_body_system();
    let params = SimulationParams::default();

    // Two simulations: one with default params, one with modified
    let mut sim1 = GpuSimulator::new(bodies.clone(), params).await;
    let mut sim2 = GpuSimulator::new(bodies.clone(), params).await;

    // sim2 gets different parameters
    let modified_params = SimulationParams {
        dt: 0.032,
        epsilon: params.epsilon,
        g_constant: 5.0,  // Very different
    };
    sim2.set_params(modified_params);

    // Both take one step
    sim1.step(1);
    sim2.step(1);

    let result1 = sim1.get_bodies();
    let result2 = sim2.get_bodies();

    // Results MUST be different (different g_constant)
    let pos_diff = ((result1[0].position[0] - result2[0].position[0]).powi(2)
        + (result1[0].position[1] - result2[0].position[1]).powi(2)).sqrt();

    assert!(pos_diff > 0.01,
        "GPU params not applied correctly - positions too similar: diff = {}", pos_diff);
}
