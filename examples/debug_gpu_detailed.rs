use monte_carlo_root::nbody::*;

#[tokio::main]
async fn main() {
    // Sehr einfaches System: 2 Körper
    let bodies = utils::generate_two_body_system();
    let params = SimulationParams::default();

    println!("=== INITIAL STATE ===");
    for (i, b) in bodies.iter().enumerate() {
        println!("Body {}: pos=[{:.6}, {:.6}], vel=[{:.6}, {:.6}], mass={}",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1], b.mass);
    }

    // GPU-Simulation - 0 Schritte (nur Initialisierung testen)
    let mut gpu_sim = GpuSimulator::new(bodies.clone(), params).await;
    let gpu_initial = gpu_sim.get_bodies();

    println!("\n=== GPU INITIAL (nach get_bodies) ===");
    for (i, b) in gpu_initial.iter().enumerate() {
        println!("Body {}: pos=[{:.6}, {:.6}], vel=[{:.6}, {:.6}], mass={}",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1], b.mass);
    }

    // Vergleiche
    println!("\n=== INITIAL COMPARISON ===");
    for i in 0..bodies.len() {
        if bodies[i].position != gpu_initial[i].position ||
           bodies[i].velocity != gpu_initial[i].velocity {
            println!("ERROR: Body {} wurde während Initialisierung verändert!", i);
        } else {
            println!("OK: Body {} ist korrekt initialisiert", i);
        }
    }

    // Jetzt 1 Schritt
    println!("\n=== RUNNING 1 STEP ===");
    gpu_sim.step(1);
    let gpu_result = gpu_sim.get_bodies();

    println!("\n=== GPU AFTER 1 STEP ===");
    for (i, b) in gpu_result.iter().enumerate() {
        println!("Body {}: pos=[{:.6}, {:.6}], vel=[{:.6}, {:.6}], mass={}",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1], b.mass);
    }

    // CPU-Referenz
    let mut cpu_sim = CpuSingleThreaded::new(bodies.clone(), params);
    cpu_sim.step(1);
    let cpu_result = cpu_sim.get_bodies();

    println!("\n=== CPU AFTER 1 STEP ===");
    for (i, b) in cpu_result.iter().enumerate() {
        println!("Body {}: pos=[{:.6}, {:.6}], vel=[{:.6}, {:.6}], mass={}",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1], b.mass);
    }

    println!("\n=== COMPARISON ===");
    for i in 0..cpu_result.len() {
        let pos_err = ((cpu_result[i].position[0] - gpu_result[i].position[0]).powi(2) +
                       (cpu_result[i].position[1] - gpu_result[i].position[1]).powi(2)).sqrt();
        let vel_err = ((cpu_result[i].velocity[0] - gpu_result[i].velocity[0]).powi(2) +
                       (cpu_result[i].velocity[1] - gpu_result[i].velocity[1]).powi(2)).sqrt();

        println!("Body {}: pos_error={:.6}, vel_error={:.6}", i, pos_err, vel_err);

        if pos_err > 0.01 || vel_err > 0.01 {
            println!("  ERROR: Difference too large!");
        }
    }
}

