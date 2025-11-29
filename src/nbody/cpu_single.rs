/// Single-threaded CPU N-Body Simulation
use crate::nbody::simulator::NBodySimulator;
use crate::nbody::types::{Body, SimulationParams};

pub struct CpuSingleThreaded {
    bodies: Vec<Body>,
    params: SimulationParams,
}

impl CpuSingleThreaded {
    pub fn new(bodies: Vec<Body>, params: SimulationParams) -> Self {
        Self { bodies, params }
    }
}

impl NBodySimulator for CpuSingleThreaded {
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

impl CpuSingleThreaded {
    #[inline]
    fn step_once(&mut self) {
        let n = self.bodies.len();
        let mut new_bodies = vec![Body::new([0.0; 2], [0.0; 2], 0.0); n];

        for i in 0..n {
            let current = self.bodies[i];
            let mut force = [0.0f32; 2];

            // Berechne Kraft von allen anderen Körpern
            for j in 0..n {
                if i == j {
                    continue;
                }

                let other = self.bodies[j];
                let r_vec = [
                    other.position[0] - current.position[0],
                    other.position[1] - current.position[1],
                ];

                let r_squared = (r_vec[0].powi(2) + r_vec[1].powi(2))
                    .max(self.params.epsilon);
                let r_distance = r_squared.sqrt();

                // F = G * m1 * m2 / r^2
                let force_magnitude =
                    self.params.g_constant * current.mass * other.mass / r_squared;

                let force_direction = [r_vec[0] / r_distance, r_vec[1] / r_distance];

                force[0] += force_magnitude * force_direction[0];
                force[1] += force_magnitude * force_direction[1];
            }

            // Berechne Beschleunigung: a = F / m
            let acceleration = [force[0] / current.mass, force[1] / current.mass];

            // Update Geschwindigkeit: v = v + a * dt
            let new_velocity = [
                current.velocity[0] + acceleration[0] * self.params.dt,
                current.velocity[1] + acceleration[1] * self.params.dt,
            ];

            // Update Position: x = x + v * dt
            let new_position = [
                current.position[0] + new_velocity[0] * self.params.dt,
                current.position[1] + new_velocity[1] * self.params.dt,
            ];

            new_bodies[i] = Body::new(new_position, new_velocity, current.mass);
        }

        self.bodies = new_bodies;
    }
}

