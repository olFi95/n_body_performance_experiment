/// Gemeinsame SIMD-Kernfunktionen für Single- und Multi-threaded Implementierungen
use crate::nbody::types::{Body, SimulationParams};
use std::simd::{f32x8, num::SimdFloat, StdFloat};

/// Berechnet ein Update für einen einzelnen Körper unter Verwendung von SIMD
/// für die Interaktionen mit allen anderen Körpern
#[inline]
pub fn compute_body_update(
    index: usize,
    all_bodies: &[Body],
    params: &SimulationParams,
) -> Body {
    let n = all_bodies.len();
    let current = &all_bodies[index];
    let mut force = [0.0f32; 2];

    // SIMD-beschleunigte Kraftberechnung
    let current_pos_x = f32x8::splat(current.position[0]);
    let current_pos_y = f32x8::splat(current.position[1]);
    let current_mass = f32x8::splat(current.mass);
    let epsilon = f32x8::splat(params.epsilon);
    let g_constant = f32x8::splat(params.g_constant);

    let mut force_x = f32x8::splat(0.0);
    let mut force_y = f32x8::splat(0.0);

    // Verarbeite 8 Körper gleichzeitig mit SIMD
    let chunks = n / 8;
    for chunk in 0..chunks {
        let base_idx = chunk * 8;

        // Überspringe den aktuellen Körper
        if base_idx <= index && index < base_idx + 8 {
            for j in base_idx..base_idx + 8 {
                if index == j {
                    continue;
                }
                compute_force_scalar(current, &all_bodies[j], params, &mut force);
            }
            continue;
        }

        // Lade 8 Körper
        let mut other_pos_x = [0.0f32; 8];
        let mut other_pos_y = [0.0f32; 8];
        let mut other_mass = [0.0f32; 8];

        for k in 0..8 {
            let idx = base_idx + k;
            other_pos_x[k] = all_bodies[idx].position[0];
            other_pos_y[k] = all_bodies[idx].position[1];
            other_mass[k] = all_bodies[idx].mass;
        }

        let other_pos_x_simd = f32x8::from_array(other_pos_x);
        let other_pos_y_simd = f32x8::from_array(other_pos_y);
        let other_mass_simd = f32x8::from_array(other_mass);

        // Berechne Distanzvektoren
        let r_vec_x = other_pos_x_simd - current_pos_x;
        let r_vec_y = other_pos_y_simd - current_pos_y;

        // Berechne Distanz^2
        let r_squared = (r_vec_x * r_vec_x + r_vec_y * r_vec_y).simd_max(epsilon);
        let r_distance = r_squared.sqrt();

        // Berechne Kraft: F = G * m1 * m2 / r^2
        let force_magnitude = g_constant * current_mass * other_mass_simd / r_squared;

        // Kraftrichtung
        let force_dir_x = r_vec_x / r_distance;
        let force_dir_y = r_vec_y / r_distance;

        // Akkumuliere Kraft
        force_x += force_magnitude * force_dir_x;
        force_y += force_magnitude * force_dir_y;
    }

    // Summiere SIMD-Vektoren
    let force_x_array = force_x.to_array();
    let force_y_array = force_y.to_array();
    for k in 0..8 {
        force[0] += force_x_array[k];
        force[1] += force_y_array[k];
    }

    // Verarbeite verbleibende Körper (nicht durch 8 teilbar)
    for j in (chunks * 8)..n {
        if index == j {
            continue;
        }
        compute_force_scalar(current, &all_bodies[j], params, &mut force);
    }

    // Update Geschwindigkeit und Position
    let acceleration = [force[0] / current.mass, force[1] / current.mass];
    let new_velocity = [
        current.velocity[0] + acceleration[0] * params.dt,
        current.velocity[1] + acceleration[1] * params.dt,
    ];
    let new_position = [
        current.position[0] + new_velocity[0] * params.dt,
        current.position[1] + new_velocity[1] * params.dt,
    ];

    Body::new(new_position, new_velocity, current.mass)
}

/// Skalare Kraftberechnung für einzelne Körperpaare
#[inline]
fn compute_force_scalar(
    current: &Body,
    other: &Body,
    params: &SimulationParams,
    force: &mut [f32; 2],
) {
    let r_vec = [
        other.position[0] - current.position[0],
        other.position[1] - current.position[1],
    ];

    let r_squared = (r_vec[0].powi(2) + r_vec[1].powi(2)).max(params.epsilon);
    let r_distance = r_squared.sqrt();

    let force_magnitude = params.g_constant * current.mass * other.mass / r_squared;
    let force_direction = [r_vec[0] / r_distance, r_vec[1] / r_distance];

    force[0] += force_magnitude * force_direction[0];
    force[1] += force_magnitude * force_direction[1];
}
