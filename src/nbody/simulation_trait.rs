use crate::nbody::types::{Body, SimulationParams};
use crate::nbody::simulation_state::SimulationState;

pub trait Simulation {
    fn step(&mut self, steps: usize);

    fn state(&self) -> &SimulationState;

    fn state_mut(&mut self) -> &mut SimulationState;

    fn get_bodies(&self) -> Vec<Body> {
        self.state().get_bodies()
    }

    fn get_params(&self) -> SimulationParams {
        self.state().get_params()
    }

    fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.state_mut().set_bodies(bodies);
    }

    fn len(&self) -> usize {
        self.state().len()
    }

    fn is_empty(&self) -> bool {
        self.state().is_empty()
    }
}

