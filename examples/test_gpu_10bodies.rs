use monte_carlo_root::nbody::*;

#[tokio::main]
async fn main() {
    // Teste mit mehr Bodies
    let bodies = utils::generate_random_bodies(10, 100.0);
    let params = SimulationParams::default();

    println!("=== INITIAL (10 bodies) ===");
    for (i, b) in bodies.iter().take(3).enumerate() {
        println!("Body {}: pos=[{:.3}, {:.3}], vel=[{:.3}, {:.3}]",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1]);
    }

    // CPU-Referenz
    let mut cpu_sim = CpuSingleThreaded::new(bodies.clone(), params);
    cpu_sim.step(1);
    let cpu_result = cpu_sim.get_bodies();

    println!("\n=== CPU AFTER 1 STEP ===");
    for (i, b) in cpu_result.iter().take(3).enumerate() {
        println!("Body {}: pos=[{:.3}, {:.3}], vel=[{:.3}, {:.3}]",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1]);
    }

    // GPU
    let mut gpu_sim = GpuSimulator::new(bodies.clone(), params).await;
    gpu_sim.step(1);
    let gpu_result = gpu_sim.get_bodies();

    println!("\n=== GPU AFTER 1 STEP ===");
    for (i, b) in gpu_result.iter().take(3).enumerate() {
        println!("Body {}: pos=[{:.3}, {:.3}], vel=[{:.3}, {:.3}]",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1]);
    }

    println!("\n=== COMPARISON ===");
    let mut max_pos_err = 0.0f32;
    let mut max_vel_err = 0.0f32;

    for i in 0..cpu_result.len() {
        let pos_err = ((cpu_result[i].position[0] - gpu_result[i].position[0]).powi(2) +
                       (cpu_result[i].position[1] - gpu_result[i].position[1]).powi(2)).sqrt();
        let vel_err = ((cpu_result[i].velocity[0] - gpu_result[i].velocity[0]).powi(2) +
                       (cpu_result[i].velocity[1] - gpu_result[i].velocity[1]).powi(2)).sqrt();

        max_pos_err = max_pos_err.max(pos_err);
        max_vel_err = max_vel_err.max(vel_err);

        if i < 3 {
            println!("Body {}: pos_err={:.6}, vel_err={:.6}", i, pos_err, vel_err);
        }
    }

    println!("\nMax pos error: {:.6}", max_pos_err);
    println!("Max vel error: {:.6}", max_vel_err);

    if max_pos_err < 0.01 && max_vel_err < 0.01 {
        println!("✓ GPU implementation is CORRECT!");
    } else {
        println!("✗ GPU implementation has ERRORS!");
    }
}

