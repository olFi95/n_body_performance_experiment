use std::simd::{f32x8, StdFloat};
use std::simd::prelude::SimdFloat;
use crate::nbody::types::{Body, SimulationParams};
use crate::nbody::Simulation;

#[derive(Default)]
pub struct SimdAlignedNBodyCore {
    position_x: Vec<f32>,
    position_y: Vec<f32>,
    velocity_x: Vec<f32>,
    velocity_y: Vec<f32>,
    mass: Vec<f32>,
    params: SimulationParams,
}

impl SimdAlignedNBodyCore {
    pub fn new(bodies: Vec<Body>) -> Self {
        let mut ret = SimdAlignedNBodyCore {
            position_x: Vec::with_capacity(0),
            position_y: Vec::with_capacity(0),
            velocity_x: Vec::with_capacity(0),
            velocity_y: Vec::with_capacity(0),
            mass: Vec::with_capacity(0),
            params: Default::default(),
        };
        ret.set_bodies(bodies);
        ret
    }
    #[inline]
    fn simd_step_once(&mut self) {
        let n = self.mass.len();

        // temporäre Arrays für neue Zustände
        let mut new_pos_x = vec![0.0f32; n];
        let mut new_pos_y = vec![0.0f32; n];
        let mut new_vel_x = vec![0.0f32; n];
        let mut new_vel_y = vec![0.0f32; n];

        let g = f32x8::splat(self.params.g_constant);
        let eps = f32x8::splat(self.params.epsilon);
        let dt = self.params.dt;

        let chunks = n / 8;

        for i in 0..n {
            let px = f32x8::splat(self.position_x[i]);
            let py = f32x8::splat(self.position_y[i]);
            let m_i = f32x8::splat(self.mass[i]);

            let mut fx = f32x8::splat(0.0);
            let mut fy = f32x8::splat(0.0);

            // SIMD loops
            for chunk in 0..chunks {
                let base = chunk * 8;

                if base <= i && i < base + 8 {
                    // self interaction in scalar
                    for j in base..base + 8 {
                        if j == i { continue; }

                        let dx = self.position_x[j] - self.position_x[i];
                        let dy = self.position_y[j] - self.position_y[i];

                        let r2 = (dx*dx + dy*dy).max(self.params.epsilon);
                        let r = r2.sqrt();
                        let f = self.params.g_constant * self.mass[i] * self.mass[j] / r2;
                        fx[0] += f * dx / r;
                        fy[0] += f * dy / r;
                    }
                    continue;
                }

                // SIMD load
                let ox = f32x8::from_slice(&self.position_x[base..base+8]);
                let oy = f32x8::from_slice(&self.position_y[base..base+8]);
                let om = f32x8::from_slice(&self.mass[base..base+8]);

                let dx = ox - px;
                let dy = oy - py;

                let r2 = (dx*dx + dy*dy).simd_max(eps);
                let r = r2.sqrt();

                let f = g * m_i * om / r2;

                fx += f * dx / r;
                fy += f * dy / r;
            }

            // scalar tail
            for j in chunks * 8 .. n {
                if j == i { continue; }

                let dx = self.position_x[j] - self.position_x[i];
                let dy = self.position_y[j] - self.position_y[i];

                let r2 = (dx*dx + dy*dy).max(self.params.epsilon);
                let r = r2.sqrt();
                let f = self.params.g_constant * self.mass[i] * self.mass[j] / r2;

                fx[0] += f * dx / r;
                fy[0] += f * dy / r;
            }

            // reduce SIMD vector to scalars
            let fx_sum = fx.reduce_sum();
            let fy_sum = fy.reduce_sum();

            let ax = fx_sum / self.mass[i];
            let ay = fy_sum / self.mass[i];

            let nvx = self.velocity_x[i] + ax * dt;
            let nvy = self.velocity_y[i] + ay * dt;

            let npx = self.position_x[i] + nvx * dt;
            let npy = self.position_y[i] + nvy * dt;

            new_vel_x[i] = nvx;
            new_vel_y[i] = nvy;
            new_pos_x[i] = npx;
            new_pos_y[i] = npy;
        }

        // swap new state in
        self.position_x = new_pos_x;
        self.position_y = new_pos_y;
        self.velocity_x = new_vel_x;
        self.velocity_y = new_vel_y;
    }
}


impl Simulation for SimdAlignedNBodyCore{
    fn step(&mut self, steps: usize) {
        for _ in 0..steps {
            self.simd_step_once();
        }
    }
    fn get_bodies(&self) -> Vec<Body> {
        let len = self.mass.len();
        let mut bodies = Vec::with_capacity(len);

        for i in 0..len {
            bodies.push(Body::new(
                [self.position_x[i], self.position_y[i]],
                [self.velocity_x[i], self.velocity_y[i]],
                self.mass[i])
            );
        }

        bodies
    }
    fn set_bodies(&mut self, bodies: Vec<Body>) {
        let len = bodies.len();

        self.position_x = vec![0.0; len];
        self.position_y = vec![0.0; len];
        self.velocity_x = vec![0.0; len];
        self.velocity_y = vec![0.0; len];
        self.mass = vec![0.0; len];

        for (i, body) in bodies.iter().enumerate() {
            self.position_x[i] = body.position[0];
            self.position_y[i] = body.position[1];
            self.velocity_x[i] = body.velocity[0];
            self.velocity_y[i] = body.velocity[1];
            self.mass[i] = body.mass;
        }
    }
    fn get_params(&self) -> &SimulationParams {
        &self.params
    }

    fn set_params(&mut self, simulation_params: SimulationParams) {
        self.params = simulation_params;
    }
}