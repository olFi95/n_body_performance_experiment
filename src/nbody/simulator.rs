use crate::nbody::types::{Body, SimulationParams};

/// Trait für alle N-Body Simulator-Backends
pub trait NBodySimulator: Send {
    /// Führt eine bestimmte Anzahl von Simulationsschritten durch
    fn step(&mut self, steps: usize);
    
    /// Gibt die aktuellen Körper zurück
    fn get_bodies(&self) -> Vec<Body>;
    
    /// Gibt die Simulationsparameter zurück
    fn get_params(&self) -> SimulationParams;
    
    /// Setzt neue Körper
    fn set_bodies(&mut self, bodies: Vec<Body>);
}

/// Hilfsfunktionen für die Simulation
pub mod utils {
    use crate::nbody::types::Body;
    use rand::Rng;

    /// Generiert Testdaten mit zufälligen Positionen
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

    /// Generiert ein stabiles Zwei-Körper-System für Tests
    pub fn generate_two_body_system() -> Vec<Body> {
        vec![
            Body::new([0.0, 1.0], [0.5, 0.0], 100.0),
            Body::new([0.0, -1.0], [-0.5, 0.0], 100.0),
        ]
    }

    /// Generiert ein kreisförmiges System
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

