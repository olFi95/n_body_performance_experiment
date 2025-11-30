use crate::nbody::types::{Body, SimulationParams};
use crate::nbody::simulation_state::SimulationState;

pub trait Simulation {
    fn step(&mut self, steps: usize);

    fn state(&self) -> &SimulationState;

    /// Gibt eine Kopie der Bodies zurück
    fn get_bodies(&self) -> Vec<Body> {
        self.state().get_bodies()
    }

    /// Gibt die Simulationsparameter zurück
    fn get_params(&self) -> SimulationParams {
        self.state().get_params()
    }

    /// Setzt neue Bodies - garantiert Synchronisation mit GPU-Buffern
    fn set_bodies(&mut self, bodies: Vec<Body>);

    fn len(&self) -> usize {
        self.state().len()
    }

    fn is_empty(&self) -> bool {
        self.state().is_empty()
    }
}
