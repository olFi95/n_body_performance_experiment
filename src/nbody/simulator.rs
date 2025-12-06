use crate::nbody::shader_types::nbody::{Body, SimulationParams};

pub trait NBodySimulator: Send {
    fn step(&mut self, steps: usize);

    fn get_bodies(&self) -> Vec<Body>;

    fn get_params(&self) -> SimulationParams;

    fn set_bodies(&mut self, bodies: Vec<Body>);
}

pub mod utils {
    use rand::Rng;
    use crate::nbody::shader_types::nbody::Body;

    pub fn generate_random_bodies(n: usize, mass: f32) -> Vec<Body> {
        let mut rng = rand::rng();
        (0..n)
            .map(|_| {
                Body::new(
                    [
                        rng.random_range(-1.0..=1.0),
                        rng.random_range(-1.0..=1.0),
                    ],
                    [0.0, 0.0],
                    mass,
                )
            })
            .collect()
    }

    pub fn generate_two_body_system() -> Vec<Body> {
        vec![
            Body::new([0.0, 1.0], [0.5, 0.0], 100.0),
            Body::new([0.0, -1.0], [-0.5, 0.0], 100.0),
        ]
    }

    pub fn generate_circular_system(n: usize, radius: f32) -> Vec<Body> {
        (0..n)
            .map(|i| {
                let angle = 2.0 * std::f32::consts::PI * (i as f32) / (n as f32);
                Body::new(
                    [radius * angle.cos(), radius * angle.sin()],
                    [0.0, 0.0],
                    100.0,
                )
            })
            .collect()
    }
}

