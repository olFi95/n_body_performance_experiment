use crate::nbody::types::{Body, SimulationParams};
use crate::nbody::simulation_state::SimulationState;
use crate::nbody::simulation_trait::Simulation;
use crate::nbody::cpu_core;
use rayon::prelude::*;

pub struct CpuMultiThreaded {
    state: SimulationState,
}

impl CpuMultiThreaded {
    pub fn new(bodies: Vec<Body>, params: SimulationParams) -> Self {
        Self {
            state: SimulationState::new(bodies, params),
        }
    }

    #[inline]
    fn step_once(&mut self) {
        let n = self.state.len();
        let bodies_ref = self.state.bodies();
        let params = self.state.params;

        let new_bodies: Vec<Body> = (0..n)
            .into_par_iter()
            .map(|i| {
                cpu_core::compute_body_update(i, bodies_ref, &params)
            })
            .collect();

        self.state.update_bodies(new_bodies);
    }
}

impl Simulation for CpuMultiThreaded {
    fn step(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step_once();
        }
    }

    fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.state.set_bodies(bodies);
    }

    fn get_bodies(&self) -> Vec<Body> {
        self.state.get_bodies()
    }

    fn get_params(&self) -> &SimulationParams {
        self.state.get_params()
    }

    fn set_params(&mut self, simulation_params: SimulationParams) {
        self.state.params = simulation_params;
    }
}
