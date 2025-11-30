use bytemuck::{Pod, Zeroable};

/// IMPORTANT: Layout must match exactly with WGSL shader
/// - #[repr(C)]: C-compatible memory layout
/// - #[repr(align(16))]: Alignment at 16-byte boundary
/// - Explicit padding required for Pod trait
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(16))]
pub struct Body {
    pub position: [f32; 2],   // Offset 0, Size 8
    pub velocity: [f32; 2],   // Offset 8, Size 8
    pub mass: f32,            // Offset 16, Size 4
    _padding: [f32; 3],       // Offset 20, Size 12 -> Total 32 Bytes (erforderlich für Pod)
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

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct SimulationParams {
    pub dt: f32,           // Time step
    pub epsilon: f32,      // Softening parameter to avoid singularities
    pub g_constant: f32,   // Gravitational constant
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            dt: 0.016,        // ~60 FPS
            epsilon: 1e-6,
            g_constant: 1.0,
        }
    }
}
