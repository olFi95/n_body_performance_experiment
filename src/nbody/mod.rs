pub mod types;
pub mod simulator;
pub mod simulation_state;
pub mod simulation_trait;  // Zentraler Simulation Trait
pub mod cpu_single;
pub mod cpu_rayon;
pub mod cpu_core;      // Gemeinsame CPU-Funktionen
pub mod simd_single;
pub mod simd_rayon;
pub mod simd_core;     // Gemeinsame SIMD-Funktionen
pub mod gpu;

#[cfg(test)]
mod tests;

// Re-exports für einfacheren Zugriff
pub use types::{Body, SimulationParams};
pub use simulator::{NBodySimulator, utils};
pub use simulation_state::SimulationState;
pub use simulation_trait::Simulation;
pub use cpu_single::CpuSingleThreaded;
pub use cpu_rayon::CpuMultiThreaded;
pub use simd_single::SimdSingleThreaded;
pub use simd_rayon::SimdMultiThreaded;
pub use gpu::GpuSimulator;
