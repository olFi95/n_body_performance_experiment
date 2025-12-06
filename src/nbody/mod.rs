pub mod simulator;
pub mod simulation_state;
pub mod simulation_trait;  // Central Simulation Trait
pub mod cpu_single;
pub mod cpu_rayon;
pub mod cpu_core;      // Shared CPU functions
pub mod simd_single;
pub mod simd_rayon;
pub mod simd_core;     // Shared SIMD functions
pub mod gpu;

#[cfg(test)]
mod tests;
pub mod simd_alligned_core;
pub mod shader_types;

pub use cpu_rayon::CpuMultiThreaded;
pub use cpu_single::CpuSingleThreaded;
pub use gpu::GpuSimulator;
pub use simd_rayon::SimdMultiThreaded;
pub use simd_single::SimdSingleThreaded;
pub use simulation_state::SimulationState;
pub use simulation_trait::Simulation;
pub use simulator::{utils, NBodySimulator};
