/// Gemeinsame Datentypen für alle N-Body Implementierungen
use bytemuck::{Pod, Zeroable};

/// Repräsentiert einen Körper in der 2D-Simulation
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Body {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub mass: f32,
    pub _padding: [f32; 3], // Für GPU-Alignment
}

impl Body {
    pub fn new(position: [f32; 2], velocity: [f32; 2], mass: f32) -> Self {
        Self {
            position,
            velocity,
            mass,
            _padding: [0.0; 3],
        }
    }
}

/// Simulationsparameter
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct SimulationParams {
    pub dt: f32,           // Zeitschritt
    pub epsilon: f32,      // Softening parameter zur Vermeidung von Singularitäten
    pub g_constant: f32,   // Gravitationskonstante
    pub _padding: f32,     // Für GPU-Alignment
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            dt: 0.016,        // ~60 FPS
            epsilon: 1e-6,
            g_constant: 1.0,
            _padding: 0.0,
        }
    }
}
