use monte_carlo_root::nbody::*;
use std::mem::{size_of, offset_of};

fn main() {
    println!("=== RUST STRUCT LAYOUT ===");
    println!("Body size: {} bytes", size_of::<Body>());
    println!("Body alignment: {} bytes", std::mem::align_of::<Body>());

    println!("\nField offsets:");
    println!("  position offset: {} bytes", offset_of!(Body, position));
    println!("  velocity offset: {} bytes", offset_of!(Body, velocity));
    println!("  mass offset: {} bytes", offset_of!(Body, mass));
    println!("  _padding offset: {} bytes", offset_of!(Body, _padding));

    println!("\nField sizes:");
    println!("  position: {} bytes (2 x f32)", size_of::<[f32; 2]>());
    println!("  velocity: {} bytes (2 x f32)", size_of::<[f32; 2]>());
    println!("  mass: {} bytes (1 x f32)", size_of::<f32>());
    println!("  _padding: {} bytes (3 x f32)", size_of::<[f32; 3]>());

    println!("\n=== WGSL EXPECTED LAYOUT ===");
    println!("struct Body {{");
    println!("    position: vec2<f32>,  // offset 0, size 8");
    println!("    velocity: vec2<f32>,  // offset 8, size 8");
    println!("    mass: f32,            // offset 16, size 4");
    println!("    _padding: vec3<f32>,  // offset 20, size 12");
    println!("}}");
    println!("Total: 32 bytes");

    println!("\n=== SimulationParams ===");
    println!("Size: {} bytes", size_of::<SimulationParams>());
    println!("Field offsets:");
    println!("  dt offset: {} bytes", offset_of!(SimulationParams, dt));
    println!("  epsilon offset: {} bytes", offset_of!(SimulationParams, epsilon));
    println!("  g_constant offset: {} bytes", offset_of!(SimulationParams, g_constant));
    println!("  _padding offset: {} bytes", offset_of!(SimulationParams, _padding));

    // Test mit tatsächlichen Daten
    let body = Body::new([1.5, 2.5], [3.5, 4.5], 100.0);
    let bytes = bytemuck::bytes_of(&body);

    println!("\n=== ACTUAL BINARY LAYOUT ===");
    println!("Body as bytes (hex):");
    for (i, chunk) in bytes.chunks(16).enumerate() {
        print!("  {:02x}: ", i * 16);
        for byte in chunk {
            print!("{:02x} ", byte);
        }
        println!();
    }

    // Interpretiere die Bytes
    println!("\nInterpreting as f32 values:");
    for (i, chunk) in bytes.chunks(4).enumerate() {
        if chunk.len() == 4 {
            let value = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            println!("  Offset {}: {}", i * 4, value);
        }
    }
}

