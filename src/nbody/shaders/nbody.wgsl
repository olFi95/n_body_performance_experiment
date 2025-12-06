// N-Body Simulation GPU Shader - Correctly aligned for WGSL Storage Buffer

struct Body {
    position: vec2<f32>,
    velocity: vec2<f32>,
    mass: f32,
    padding0: f32,
    padding1: f32,
    padding2: f32,
}

struct SimulationParams {
    dt: f32,
    epsilon: f32,
    g_constant: f32,
}

@group(0) @binding(0)
var<storage, read> bodies_in: array<Body>;

@group(0) @binding(1)
var<storage, read_write> bodies_out: array<Body>;

@group(0) @binding(2)
var<uniform> params: SimulationParams;

@group(0) @binding(3)
var<uniform> n_bodies: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if (i >= n_bodies) {
        return;
    }

    let current = bodies_in[i];
    var force = vec2<f32>(0.0, 0.0);

    for (var j = 0u; j < n_bodies; j = j + 1u) {
        if (i == j) {
            continue;
        }

        let other = bodies_in[j];
        let r_vec = other.position - current.position;
        let r_squared = max(dot(r_vec, r_vec), params.epsilon);
        let r_distance = sqrt(r_squared);
        let force_magnitude = params.g_constant * current.mass * other.mass / r_squared;
        force = force + force_magnitude * (r_vec / r_distance);
    }

    let acceleration = force / current.mass;

    let new_velocity = current.velocity + acceleration * params.dt;

    let new_position = current.position + new_velocity * params.dt;

    bodies_out[i].position = new_position;
    bodies_out[i].velocity = new_velocity;
    bodies_out[i].mass = current.mass;
    bodies_out[i].padding0 = 0.0;
    bodies_out[i].padding1 = 0.0;
    bodies_out[i].padding2 = 0.0;
}
