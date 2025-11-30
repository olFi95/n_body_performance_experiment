use crate::nbody::types::{Body, SimulationParams};
use crate::nbody::simulation_state::SimulationState;

pub trait Simulation {
    fn step(&mut self, steps: usize);

    fn state(&self) -> &SimulationState;

    /// Returns a copy of the bodies
    fn get_bodies(&self) -> Vec<Body> {
        self.state().get_bodies()
    }

    /// Returns the simulation parameters
    fn get_params(&self) -> SimulationParams {
        self.state().get_params()
    }

    /// Sets new bodies - guarantees synchronization with GPU buffers
    fn set_bodies(&mut self, bodies: Vec<Body>);

    fn len(&self) -> usize {
        self.state().len()
    }

    fn is_empty(&self) -> bool {
        self.state().is_empty()
    }
}
