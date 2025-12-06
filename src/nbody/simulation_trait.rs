use crate::nbody::shader_types::nbody::{Body, SimulationParams};

pub trait Simulation {
    fn step(&mut self, steps: usize);

    fn get_bodies(&self) -> Vec<Body>;

    fn set_bodies(&mut self, bodies: Vec<Body>);

    fn get_params(&self) -> &SimulationParams;
    fn set_params(&mut self, simulation_params: SimulationParams);
}
