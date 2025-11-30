/// GPU N-Body Simulation mit WGPU - Double-Buffering wie CPU-Version
use crate::nbody::types::{Body, SimulationParams};
use crate::nbody::simulation_state::SimulationState;
use crate::nbody::simulation_trait::Simulation;
use wgpu::util::DeviceExt;
use wgpu::wgt::PollType;

pub struct GpuSimulator {
    state: SimulationState,
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bodies_buffer_a: wgpu::Buffer,
    bodies_buffer_b: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    n_bodies_buffer: wgpu::Buffer,
    current_buffer_is_a: bool,
}

impl GpuSimulator {
    pub async fn new(bodies: Vec<Body>, params: SimulationParams) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to create device");

        // Erstelle Buffer A (initial mit Daten)
        let bodies_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bodies Buffer A"),
            contents: bytemuck::cast_slice(&bodies),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Erstelle Buffer B (initial mit den gleichen Daten initialisiert)
        // WICHTIG: Beide Buffer müssen initialisiert sein für Double-Buffering
        let bodies_buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bodies Buffer B"),
            contents: bytemuck::cast_slice(&bodies),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let n_bodies = bodies.len() as u32;
        let n_bodies_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("N Bodies Buffer"),
            contents: bytemuck::bytes_of(&n_bodies),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        println!("GPU Setup: {} bodies, buffer size A: {} bytes, buffer size B: {} bytes",
                 bodies.len(),
                 bodies.len() * std::mem::size_of::<Body>(),
                 bodies.len() * std::mem::size_of::<Body>());

        // DEBUG: Lese Buffer B direkt nach Initialisierung aus
        {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Debug Staging"),
                size: (bodies.len() * std::mem::size_of::<Body>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(&bodies_buffer_b, 0, &staging, 0, (bodies.len() * std::mem::size_of::<Body>()) as u64);
            let idx = queue.submit(Some(encoder.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = futures_channel::oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            device.poll(wgpu::wgt::PollType::Wait { submission_index: Some(idx), timeout: None }).unwrap();
            futures::executor::block_on(rx).unwrap().unwrap();

            staging.unmap();
        }

        // Shader Module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("N-Body Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nbody.wgsl").into()),
        });

        // Bind Group Layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("N-Body Bind Group Layout"),
            entries: &[
                // Input Buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output Buffer (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Anzahl der Bodies
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Pipeline Layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("N-Body Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Compute Pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("N-Body Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            state: SimulationState::new(bodies.clone(), params),
            device,
            queue,
            compute_pipeline,
            bind_group_layout,
            bodies_buffer_a,
            bodies_buffer_b,
            params_buffer,
            n_bodies_buffer,
            current_buffer_is_a: true,
        }
    }

    #[inline]
    fn get_active_buffer(&self) -> &wgpu::Buffer {
        if self.current_buffer_is_a {
            &self.bodies_buffer_a
        } else {
            &self.bodies_buffer_b
        }
    }

    #[inline]
    fn get_inactive_buffer(&self) -> &wgpu::Buffer {
        if self.current_buffer_is_a {
            &self.bodies_buffer_b
        } else {
            &self.bodies_buffer_a
        }
    }
}

// Implementierung des zentralen Simulation Traits
impl Simulation for GpuSimulator {
    fn step(&mut self, steps: usize) {
        let n = self.state.len() as u32;
        let workgroup_size = 256;
        let num_workgroups = (n + workgroup_size - 1) / workgroup_size;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("N-Body Compute Encoder"),
        });

        for _ in 0..steps {
            let input_buffer = self.get_active_buffer();
            let output_buffer = self.get_inactive_buffer();

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("N-Body Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.n_bodies_buffer.as_entire_binding(),
                    },
                ],
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("N-Body Compute Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
            }

            self.current_buffer_is_a = !self.current_buffer_is_a;
        }

        let command_buffer = encoder.finish();
        let submission_index = self.queue.submit(Some(command_buffer));

        self.device.poll(PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        }).expect("Failed to poll device");

        self.state.update_bodies(self.get_bodies());
    }

    fn get_bodies(&self) -> Vec<Body> {
        let source_buffer = self.get_active_buffer();

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.state.len() * std::mem::size_of::<Body>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            source_buffer, 0, &staging_buffer, 0,
            (self.state.len() * std::mem::size_of::<Body>()) as u64,
        );

        let submission_index = self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = futures_channel::oneshot::channel();
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device.poll(PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        }).expect("Failed to poll device");

        futures::executor::block_on(receiver)
            .expect("Communication failed")
            .expect("Buffer reading failed");

        let data = staging_buffer.slice(..).get_mapped_range();
        let bodies: Vec<Body> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        bodies
    }

    fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.state.set_bodies(bodies.clone());

        let target_buffer = self.get_active_buffer();
        self.queue.write_buffer(target_buffer, 0, bytemuck::cast_slice(&bodies));

        // WICHTIG: Auch n_bodies_buffer aktualisieren!
        let n_bodies = bodies.len() as u32;
        self.queue.write_buffer(&self.n_bodies_buffer, 0, bytemuck::bytes_of(&n_bodies));
    }

    fn get_params(&self) -> &SimulationParams {
        self.state.get_params()
    }

    fn set_params(&mut self, simulation_params: SimulationParams) {
        self.state.params = simulation_params;

        // Aktualisiere den GPU-Buffer mit den neuen Parametern
        self.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&simulation_params)
        );
    }
}
