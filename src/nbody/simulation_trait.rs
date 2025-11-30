/// Zentraler Trait für N-Body Simulationen mit SimulationState
use crate::nbody::types::{Body, SimulationParams};
use crate::nbody::simulation_state::SimulationState;

/// Trait für N-Body Simulationen basierend auf SimulationState
pub trait Simulation {
    /// Führt eine bestimmte Anzahl von Simulationsschritten aus
    fn step(&mut self, steps: usize);

    /// Gibt eine Referenz auf den Simulationszustand zurück
    fn state(&self) -> &SimulationState;

    /// Gibt eine mutable Referenz auf den Simulationszustand zurück
    fn state_mut(&mut self) -> &mut SimulationState;

    /// Gibt eine Kopie der Bodies zurück
    fn get_bodies(&self) -> Vec<Body> {
        self.state().get_bodies()
    }

    /// Gibt die Simulationsparameter zurück
    fn get_params(&self) -> SimulationParams {
        self.state().get_params()
    }

    /// Setzt neue Bodies
    fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.state_mut().set_bodies(bodies);
    }

    /// Gibt die Anzahl der Körper zurück
    fn len(&self) -> usize {
        self.state().len()
    }

    /// Prüft, ob die Simulation leer ist
    fn is_empty(&self) -> bool {
        self.state().is_empty()
    }
}

