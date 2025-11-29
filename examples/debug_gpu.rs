use monte_carlo_root::nbody::*;

#[tokio::main]
async fn main() {
    // Einfaches 2-Körper-System für Debugging
    let bodies = utils::generate_two_body_system();
    let params = SimulationParams::default();

    println!("Initial bodies:");
    for (i, b) in bodies.iter().enumerate() {
        println!("  Body {}: pos={:?}, vel={:?}, mass={}", i, b.position, b.velocity, b.mass);
    }

    // CPU-Referenz
    let mut cpu_sim = CpuSingleThreaded::new(bodies.clone(), params);
    cpu_sim.step(1);
    let cpu_result = cpu_sim.get_bodies();

    println!("\nCPU after 1 step:");
    for (i, b) in cpu_result.iter().enumerate() {
        println!("  Body {}: pos={:?}, vel={:?}, mass={}", i, b.position, b.velocity, b.mass);
    }

    // GPU
    let mut gpu_sim = GpuSimulator::new(bodies.clone(), params).await;
    gpu_sim.step(1);
    let gpu_result = gpu_sim.get_bodies();

    println!("\nGPU after 1 step:");
    for (i, b) in gpu_result.iter().enumerate() {
        println!("  Body {}: pos={:?}, vel={:?}, mass={}", i, b.position, b.velocity, b.mass);
    }

    println!("\nDifferences:");
    for i in 0..cpu_result.len() {
        let pos_diff = [
            cpu_result[i].position[0] - gpu_result[i].position[0],
            cpu_result[i].position[1] - gpu_result[i].position[1],
        ];
        let vel_diff = [
            cpu_result[i].velocity[0] - gpu_result[i].velocity[0],
            cpu_result[i].velocity[1] - gpu_result[i].velocity[1],
        ];
        println!("  Body {}: pos_diff={:?}, vel_diff={:?}", i, pos_diff, vel_diff);
    }
}

