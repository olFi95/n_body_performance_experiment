use monte_carlo_root::nbody::*;
use std::mem;

fn main() {
    println!("Body size: {} bytes", mem::size_of::<Body>());
    println!("Body alignment: {} bytes", mem::align_of::<Body>());
    println!("SimulationParams size: {} bytes", mem::size_of::<SimulationParams>());
    println!("SimulationParams alignment: {} bytes", mem::align_of::<SimulationParams>());

    let body = Body::new([1.0, 2.0], [3.0, 4.0], 5.0);
    println!("\nBody layout:");
    println!("  position: {:?}", body.position);
    println!("  velocity: {:?}", body.velocity);
    println!("  mass: {}", body.mass);
    println!("  _padding: {:?}", body._padding);

    let params = SimulationParams::default();
    println!("\nParams:");
    println!("  dt: {}", params.dt);
    println!("  epsilon: {}", params.epsilon);
    println!("  g_constant: {}", params.g_constant);
}

