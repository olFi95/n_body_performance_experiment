/// Multi-threaded CPU N-Body Simulation mit Rayon
use crate::nbody::simulator::NBodySimulator;
use crate::nbody::types::{Body, SimulationParams};
use rayon::prelude::*;

pub struct CpuMultiThreaded {
    bodies: Vec<Body>,
    params: SimulationParams,
}

impl CpuMultiThreaded {
    pub fn new(bodies: Vec<Body>, params: SimulationParams) -> Self {
        Self { bodies, params }
    }
}

impl NBodySimulator for CpuMultiThreaded {
    fn step(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step_once();
        }
    }

    fn get_bodies(&self) -> Vec<Body> {
        self.bodies.clone()
    }

    fn get_params(&self) -> SimulationParams {
        self.params
    }

    fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.bodies = bodies;
    }
}

impl CpuMultiThreaded {
    #[inline]
    fn step_once(&mut self) {
        let n = self.bodies.len();
        let bodies_ref = &self.bodies;
        let params = self.params;

        // Parallele Berechnung mit Rayon
        let new_bodies: Vec<Body> = (0..n)
            .into_par_iter()
            .map(|i| {
                let current = bodies_ref[i];
                let mut force = [0.0f32; 2];

                // Berechne Kraft von allen anderen Körpern
                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let other = bodies_ref[j];
                    let r_vec = [
                        other.position[0] - current.position[0],
                        other.position[1] - current.position[1],
                    ];

                    let r_squared = (r_vec[0].powi(2) + r_vec[1].powi(2))
                        .max(params.epsilon);
                    let r_distance = r_squared.sqrt();

                    // F = G * m1 * m2 / r^2
                    let force_magnitude =
                        params.g_constant * current.mass * other.mass / r_squared;

                    let force_direction = [r_vec[0] / r_distance, r_vec[1] / r_distance];

                    force[0] += force_magnitude * force_direction[0];
                    force[1] += force_magnitude * force_direction[1];
                }

                // Berechne Beschleunigung: a = F / m
                let acceleration = [force[0] / current.mass, force[1] / current.mass];

                // Update Geschwindigkeit: v = v + a * dt
                let new_velocity = [
                    current.velocity[0] + acceleration[0] * params.dt,
                    current.velocity[1] + acceleration[1] * params.dt,
                ];

                // Update Position: x = x + v * dt
                let new_position = [
                    current.position[0] + new_velocity[0] * params.dt,
                    current.position[1] + new_velocity[1] * params.dt,
                ];

                Body::new(new_position, new_velocity, current.mass)
            })
            .collect();

        self.bodies = new_bodies;
    }
}

