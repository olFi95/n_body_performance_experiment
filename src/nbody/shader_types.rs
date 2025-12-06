#![allow(unused, non_snake_case, non_camel_case_types, non_upper_case_globals)]
include!(concat!(
env!("OUT_DIR"),
"/shaders_types.rs"
));

use crate::nbody::shader_types::nbody::SimulationParams;

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            dt: 0.016,        // ~60 FPS
            epsilon: 1e-6,
            g_constant: 1.0,
        }
    }
}

