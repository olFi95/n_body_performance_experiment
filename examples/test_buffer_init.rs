use monte_carlo_root::nbody::*;
use wgpu::util::DeviceExt;

#[tokio::main]
async fn main() {
    // Erstelle minimale GPU-Setup
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("Failed to find adapter");

    println!("GPU Adapter: {:?}", adapter.get_info());

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("Failed to create device");

    // Einfacher Test: Erstelle einen Body und kopiere ihn durch den Shader
    let body = Body::new([1.0, 2.0], [3.0, 4.0], 5.0);
    let bodies = vec![body];

    println!("\nInput body: pos={:?}, vel={:?}, mass={}", body.position, body.velocity, body.mass);

    // Erstelle Input Buffer
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input"),
        contents: bytemuck::cast_slice(&bodies),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Erstelle Output Buffer (leer)
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: std::mem::size_of::<Body>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Lese den Output Buffer OHNE Shader (sollte 0 sein)
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size: std::mem::size_of::<Body>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, std::mem::size_of::<Body>() as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::wgt::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    futures::executor::block_on(rx).unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<Body> = bytemuck::cast_slice(&data).to_vec();
    println!("\nOutput buffer (uninitialized): pos={:?}, vel={:?}, mass={}",
             result[0].position, result[0].velocity, result[0].mass);
    drop(data);
    staging.unmap();
}

