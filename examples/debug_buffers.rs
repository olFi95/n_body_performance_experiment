use monte_carlo_root::nbody::*;
use wgpu::util::DeviceExt;
use wgpu::wgt::PollType;

#[tokio::main]
async fn main() {
    let bodies = utils::generate_two_body_system();
    let params = SimulationParams::default();

    println!("=== INITIAL ===");
    for (i, b) in bodies.iter().enumerate() {
        println!("Body {}: pos=[{:.6}, {:.6}], vel=[{:.6}, {:.6}]",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1]);
    }

    // Erstelle GPU-Setup manuell
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .unwrap();

    // Erstelle Buffer A und B
    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer A"),
        contents: bytemuck::cast_slice(&bodies),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });

    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer B"),
        contents: bytemuck::cast_slice(&bodies),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });

    // Hilfsfunktion zum Lesen eines Buffers
    let read_buffer = |device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, size: usize| -> Vec<Body> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size as u64);
        let idx = queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(PollType::Wait { submission_index: Some(idx), timeout: None }).unwrap();
        futures::executor::block_on(rx).unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<Body> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    };

    let size = bodies.len() * std::mem::size_of::<Body>();

    println!("\n=== BUFFER A (initial) ===");
    let a_initial = read_buffer(&device, &queue, &buffer_a, size);
    for (i, b) in a_initial.iter().enumerate() {
        println!("Body {}: pos=[{:.6}, {:.6}], vel=[{:.6}, {:.6}]",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1]);
    }

    println!("\n=== BUFFER B (initial) ===");
    let b_initial = read_buffer(&device, &queue, &buffer_b, size);
    for (i, b) in b_initial.iter().enumerate() {
        println!("Body {}: pos=[{:.6}, {:.6}], vel=[{:.6}, {:.6}]",
                 i, b.position[0], b.position[1], b.velocity[0], b.velocity[1]);
    }
}
