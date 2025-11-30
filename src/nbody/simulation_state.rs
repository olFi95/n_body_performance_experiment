use crate::nbody::types::{Body, SimulationParams};

pub struct SimulationState {
    bodies: Vec<Body>,
    pub(crate) params: SimulationParams,
}

impl SimulationState {
    pub fn new(bodies: Vec<Body>, params: SimulationParams) -> Self {
        Self { bodies, params }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    #[inline]
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    #[inline]
    pub fn get_bodies(&self) -> Vec<Body> {
        self.bodies.clone()
    }

    #[inline]
    pub fn get_params(&self) -> &SimulationParams {
        &self.params
    }

    #[inline]
    pub fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.bodies = bodies;
    }

    #[inline]
    pub fn update_bodies(&mut self, new_bodies: Vec<Body>) {
        self.bodies = new_bodies;
    }
}

