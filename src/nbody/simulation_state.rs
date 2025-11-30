/// Zentraler Simulationszustand für alle N-Body Implementierungen
use crate::nbody::types::{Body, SimulationParams};

/// Gemeinsamer Zustand für alle Simulationen
pub struct SimulationState {
    pub bodies: Vec<Body>,
    pub params: SimulationParams,
}

impl SimulationState {
    /// Erstellt einen neuen Simulationszustand
    pub fn new(bodies: Vec<Body>, params: SimulationParams) -> Self {
        Self { bodies, params }
    }

    /// Gibt die Anzahl der Körper zurück
    #[inline]
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Prüft, ob die Simulation leer ist
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Gibt eine Referenz auf die Bodies zurück
    #[inline]
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// Gibt eine Kopie der Bodies zurück
    #[inline]
    pub fn get_bodies(&self) -> Vec<Body> {
        self.bodies.clone()
    }

    /// Gibt die Simulationsparameter zurück
    #[inline]
    pub fn get_params(&self) -> SimulationParams {
        self.params
    }

    /// Setzt neue Bodies
    #[inline]
    pub fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.bodies = bodies;
    }

    /// Aktualisiert die Bodies mit einem neuen Vektor
    #[inline]
    pub fn update_bodies(&mut self, new_bodies: Vec<Body>) {
        self.bodies = new_bodies;
    }
}

