use crate::nbody::shader_types::nbody::{Body, SimulationParams};

#[inline]
pub fn compute_body_update(
    index: usize,
    all_bodies: &[Body],
    params: &SimulationParams,
) -> Body {
    let n = all_bodies.len();
    let current = &all_bodies[index];
    let mut force = [0.0f32; 2];

    // Calculate force from all other bodies
    for j in 0..n {
        if index == j {
            continue;
        }

        let other = &all_bodies[j];
        let r_vec = [
            other.position[0] - current.position[0],
            other.position[1] - current.position[1],
        ];

        let r_squared = (r_vec[0].powi(2) + r_vec[1].powi(2)).max(params.epsilon);
        let r_distance = r_squared.sqrt();

        // F = G * m1 * m2 / r^2
        let force_magnitude = params.g_constant * current.mass * other.mass / r_squared;
        let force_direction = [r_vec[0] / r_distance, r_vec[1] / r_distance];

        force[0] += force_magnitude * force_direction[0];
        force[1] += force_magnitude * force_direction[1];
    }

    // Berechne Beschleunigung: a = F / m
    let acceleration = [force[0] / current.mass, force[1] / current.mass];

    // Update Geschwindigkeit: v = v + a * dt
    let new_velocity = [
        current.velocity[0] + acceleration[0] * params.dt,
        current.velocity[1] + acceleration[1] * params.dt,
    ];

    // Update Position: x = x + v * dt
    let new_position = [
        current.position[0] + new_velocity[0] * params.dt,
        current.position[1] + new_velocity[1] * params.dt,
    ];

    Body::new(new_position, new_velocity, current.mass)
}
