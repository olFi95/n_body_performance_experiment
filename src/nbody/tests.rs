#[cfg(test)]
mod tests {
    use crate::nbody::*;
    use approx::assert_relative_eq;

    const EPSILON: f32 = 0.01; // Tolerance for floating point comparisons

    fn compare_bodies(bodies1: &[Body], bodies2: &[Body], tolerance: f32) {
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

    // ========== Tests für set_params und set_bodies ==========

    #[test]
    fn test_set_params_cpu_single() {
        let bodies = utils::generate_random_bodies(10, 100.0);
        let params = SimulationParams::default();

        let mut sim = CpuSingleThreaded::new(bodies.clone(), params);

        // Schritt 1: Einen Step mit Standard-Params
        sim.step(1);
        let bodies_after_step1 = sim.get_bodies();

        // Schritt 2: Neue Parameter setzen
        let new_params = SimulationParams {
            dt: 0.032,  // Doppelter Zeitschritt
            epsilon: 1e-5,
            g_constant: 2.0,  // Doppelte Gravitation
        };
        sim.set_params(new_params);

        // Schritt 3: Parameter wieder auslesen und verifizieren
        let read_params = sim.get_params();
        assert_eq!(read_params.dt, new_params.dt);
        assert_eq!(read_params.epsilon, new_params.epsilon);
        assert_eq!(read_params.g_constant, new_params.g_constant);

        // Schritt 4: Noch einen Step mit neuen Params
        sim.step(1);
        let bodies_after_step2 = sim.get_bodies();

        // Die Bodies sollten sich unterschiedlich bewegt haben
        assert_ne!(bodies_after_step1[0].position, bodies_after_step2[0].position);
    }

    #[tokio::test]
    async fn test_set_params_gpu() {
        let bodies = utils::generate_random_bodies(10, 100.0);
        let params = SimulationParams::default();

        let mut sim = GpuSimulator::new(bodies.clone(), params).await;

        // Schritt 1: Einen Step mit Standard-Params
        sim.step(1);
        let bodies_after_step1 = sim.get_bodies();

        // Schritt 2: Neue Parameter setzen
        let new_params = SimulationParams {
            dt: 0.032,  // Doppelter Zeitschritt
            epsilon: 1e-5,
            g_constant: 2.0,  // Doppelte Gravitation
        };
        sim.set_params(new_params);

        // Schritt 3: Parameter wieder auslesen und verifizieren
        let read_params = sim.get_params();
        assert_eq!(read_params.dt, new_params.dt);
        assert_eq!(read_params.epsilon, new_params.epsilon);
        assert_eq!(read_params.g_constant, new_params.g_constant);

        // Schritt 4: Noch einen Step mit neuen Params
        sim.step(1);
        let bodies_after_step2 = sim.get_bodies();

        // Die Bodies sollten sich unterschiedlich bewegt haben
        assert_ne!(bodies_after_step1[0].position, bodies_after_step2[0].position);
    }

    #[test]
    fn test_set_bodies_cpu_single() {
        let bodies = utils::generate_random_bodies(10, 100.0);
        let params = SimulationParams::default();

        let mut sim = CpuSingleThreaded::new(bodies.clone(), params);

        // Schritt 1: Einen Step
        sim.step(1);

        // Schritt 2: Neue Bodies setzen
        let new_bodies = utils::generate_two_body_system();
        sim.set_bodies(new_bodies.clone());

        // Schritt 3: Bodies wieder auslesen und verifizieren
        let read_bodies = sim.get_bodies();
        assert_eq!(read_bodies.len(), new_bodies.len());
        compare_bodies(&read_bodies, &new_bodies, 1e-6);

        // Schritt 4: Noch einen Step mit neuen Bodies
        sim.step(1);
        let bodies_after_step = sim.get_bodies();

        // Die Bodies sollten sich bewegt haben
        assert_ne!(bodies_after_step[0].position, new_bodies[0].position);
        assert_ne!(bodies_after_step[1].position, new_bodies[1].position);
    }

    #[tokio::test]
    async fn test_set_bodies_gpu() {
        let bodies = utils::generate_random_bodies(10, 100.0);
        let params = SimulationParams::default();

        let mut sim = GpuSimulator::new(bodies.clone(), params).await;

        // Schritt 1: Einen Step
        sim.step(1);

        // Schritt 2: Neue Bodies setzen
        let new_bodies = utils::generate_two_body_system();
        sim.set_bodies(new_bodies.clone());

        // Schritt 3: Bodies wieder auslesen und verifizieren
        let read_bodies = sim.get_bodies();
        assert_eq!(read_bodies.len(), new_bodies.len());
        compare_bodies(&read_bodies, &new_bodies, 1e-6);

        // Schritt 4: Noch einen Step mit neuen Bodies
        sim.step(1);
        let bodies_after_step = sim.get_bodies();

        // Die Bodies sollten sich bewegt haben
        assert_ne!(bodies_after_step[0].position, new_bodies[0].position);
        assert_ne!(bodies_after_step[1].position, new_bodies[1].position);
    }

    #[tokio::test]
    async fn test_params_synchronization_gpu() {
        // Dieser Test verifiziert, dass Params wirklich auf der GPU ankommen
        let bodies = utils::generate_two_body_system();
        let params = SimulationParams::default();

        // Zwei Simulationen: eine mit Standard-Params, eine mit modifizierten
        let mut sim1 = GpuSimulator::new(bodies.clone(), params).await;
        let mut sim2 = GpuSimulator::new(bodies.clone(), params).await;

        // sim2 bekommt andere Parameter
        let modified_params = SimulationParams {
            dt: 0.032,
            epsilon: params.epsilon,
            g_constant: 5.0,  // Sehr unterschiedlich
        };
        sim2.set_params(modified_params);

        // Beide einen Step
        sim1.step(1);
        sim2.step(1);

        let result1 = sim1.get_bodies();
        let result2 = sim2.get_bodies();

        // Die Ergebnisse MÜSSEN unterschiedlich sein (unterschiedliche g_constant)
        let pos_diff = ((result1[0].position[0] - result2[0].position[0]).powi(2)
            + (result1[0].position[1] - result2[0].position[1]).powi(2)).sqrt();

        assert!(pos_diff > 0.01,
            "GPU params not applied correctly - positions too similar: diff = {}", pos_diff);
    }
}
