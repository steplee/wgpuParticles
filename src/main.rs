use std::{
    sync::{Arc, Mutex},
    thread,
};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferAsyncError, BufferBindingType,
    BufferDescriptor, BufferSize, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipeline, ComputePipelineDescriptor, Device, MapMode, PipelineCompilationOptions,
    PipelineLayoutDescriptor, Queue, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

mod setup;
use setup::{WebGpuCtx, setup};

// const MaxParticlesSqrt: u64 = 256;
const MaxParticlesSqrt: u64 = 256;
const MaxParticles: u64 = MaxParticlesSqrt*MaxParticlesSqrt;
const ParticlesPerGroup: u64 = 32;
const FrameWorkGroupSize: u64 = 16;
// const FrameW: u64 = 1920;
// const FrameH: u64 = 1080;
const FrameW: u64 = 1280;
const FrameH: u64 = 720;

// https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html#x=5d00000100e100000000000000003d888b0237284ce7dce121b384fd72bd9a1ff901e6abc5860a8aab2b748a2f3fbc2dca897cd5cc036480d1f4e50913dd8a6d45a2b935e6a3e3540a7b4907cc3a21aa8b1b0bef4daf9ebe127e4f6eda4c885b12cea0b7c858107e112a5eaaec2f3636dd194dd383565b3dbd03913941e8be64550fccc2539215021287c596c9204174b45ea9bba988d9fef564aa00
#[derive(Clone, Copy, Debug)]
struct Particle {
    x: f32,
    y: f32,
    z: f32,
    pad: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    intensity: f32,
    size: f32,
    pad2: f32,
    pad3: f32,
    pad4: f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct SimParams {
    time: f32,
    speed: f32,
    chaos: f32,
    w: u32,
    h: u32,
}

unsafe impl Zeroable for Particle {}
unsafe impl Pod for Particle {}

unsafe impl Zeroable for SimParams {}
unsafe impl Pod for SimParams {}

struct Scene {
    wctx: WebGpuCtx,

    sim_params: SimParams,
    sim_ubo: Buffer,

    particle_bgs: Vec<BindGroup>,
    particle_buffers: Vec<Buffer>,

    frame_bgs: Vec<BindGroup>,
    frame_buffers: Vec<Buffer>,
    frame_map_buffer: Buffer,

    sim_pipeline: ComputePipeline,
    frame_pipeline: ComputePipeline,
    sim_work_group_count: u32,
    // frame_work_group_count: u32,
}

fn to_bind_group_entries<'a>(items: &[&'a Buffer]) -> Vec<BindGroupEntry<'a>> {
    return items
        .iter()
        .enumerate()
        .map(|(i, item)| BindGroupEntry {
            binding: i as u32,
            resource: (*item).as_entire_binding(),
        })
        .collect();
}

fn go() -> Scene {
    let wctx = setup();
    let device = &wctx.device;
    let queue = &wctx.queue;

    let sim_work_group_count = ((MaxParticles as f32) / (ParticlesPerGroup as f32)).ceil() as u32;

    // ------------------------------------------------------------------------------------------
    // Create Sim stuff.
    //  - Shader
    //  - BindGroupLayout
    //  - PipelineLayout & ComputePipeline
    //  - (Double-) Buffers & BindGroups

    let a = simple!("a");
    println!("a: {}", a);
    // panic!();
    let sim_shader_desc = include_combined_wgsl_2!("sim"; "common.wgsl", "sim.wgsl");
    let sim_shader = device.create_shader_module(sim_shader_desc);

    let sim_params = SimParams {
        time: 0.,
        speed: 1.,
        chaos: 1.,
        w: FrameW as u32,
        h: FrameH as u32,
    };
    let sim_ubo = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("SimUbo"),
        contents: &bytemuck::cast_slice(&[sim_params]),
        usage: BufferUsages::UNIFORM,
    });

    let sim_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<SimParams>() as _),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(
                        MaxParticles * (size_of::<Particle>() as u64),
                    ),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(
                        MaxParticles * (size_of::<Particle>() as u64),
                    ),
                },
                count: None,
            },
        ],
        label: Some("SimBGL"),
    });

    let sim_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        bind_group_layouts: &[&sim_bgl],
        label: Some("SimPipelineLayout"),
        push_constant_ranges: &[],
    });

    let sim_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("SimPipeline"),
        layout: Some(&sim_pl),
        module: &sim_shader,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let mut ini_particles = vec![];
    for i in 0..MaxParticlesSqrt {
        for j in 0..MaxParticlesSqrt {
            ini_particles.push(Particle::zeroed());
            ini_particles[(i*MaxParticlesSqrt+j) as usize].x = ((j as f32) - (MaxParticlesSqrt as f32)*0.5) / (MaxParticlesSqrt as f32) * 1.5;
            ini_particles[(i*MaxParticlesSqrt+j) as usize].y = ((i as f32) - (MaxParticlesSqrt as f32)*0.5) / (MaxParticlesSqrt as f32) * 1.5;
            let t = (j as f32) * 12.2 + (i as f32) * 9.1;
            let r = f32::sin(t) * 93.321;
            ini_particles[(i*MaxParticlesSqrt+j) as usize].intensity = f32::ceil(r) - r;
        }
    }

    let mut particle_buffers = vec![];
    let mut particle_bgs = vec![];
    for i in 0..2 {
        particle_buffers.push(device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("ParticleBuf{}", i)),
            contents: bytemuck::cast_slice(&ini_particles),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
    }

    for i in 0..2 {
        // NOTE: On even iterations we write to particle_buffers[1] and read from
        // particle_buffers[0]. And the opposite for odd iterations.
        particle_bgs.push(device.create_bind_group(&BindGroupDescriptor {
            entries: &to_bind_group_entries(&[
                &sim_ubo,
                &particle_buffers[(i + 0) % 2],
                &particle_buffers[(i + 1) % 2],
            ]),
            label: Some(&format!("ParticleBg{}", i)),
            layout: &sim_bgl,
        }));
    }

    // Test that we can read the buffer properly.
    // ... Looks good.
    {
        // I won't actually need this because we'll map the rendered frame buffer instead.
        let map_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("ParticleMapBuffer"),
            mapped_at_creation: false,
            size: (MaxParticles * (size_of::<Particle>() as u64)),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });

        {
            let mut ce = device.create_command_encoder(&CommandEncoderDescriptor::default());
            ce.copy_buffer_to_buffer(
                &particle_buffers[0],
                0u32.into(),
                &map_buffer,
                0u32.into(),
                particle_buffers[0].size(),
            );
            queue.submit([ce.finish()]);
        }

        // let slice = particle_buffers[0].slice(..);
        let slice = map_buffer.slice(..);
        // let mut container = Arc::new(Mutex::new(Vec::<Result<(), BufferAsyncError>>::new()));
        let mut container = Arc::new(Mutex::new(Option::<Result<(), BufferAsyncError>>::None));
        let mut container2 = container.clone();
        slice.map_async(MapMode::Read, move |v| {
            container2.lock().unwrap().insert(v);
        });

        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        for i in 1..100 {
            if container.lock().unwrap().is_some() {
                println!("GOOD");
                let view = slice.get_mapped_range(); // BufferView derefences as &[u8]
                let particles: &[Particle] = bytemuck::cast_slice(&view);
                println!("First few particles:");
                for j in 0..5 {
                    println!("{}: {:?}", i, particles[j]);
                }
                break;
            } else {
                println!("WAITING {} ms", i);
            }
            std::thread::sleep(std::time::Duration::from_millis(i));
        }
    }

    // ------------------------------------------------------------------------------------------
    // Create Frame stuff.
    //  - Shader
    //  - BindGroupLayout
    //  - PipelineLayout & ComputePipeline
    //  - (Double-) Buffers & BindGroups
    //  - Frame Map Buffer (for getting results to cpu)

    let frame_shader_desc = include_combined_wgsl_2!("frame"; "common.wgsl", "frame.wgsl");
    let frame_shader = device.create_shader_module(frame_shader_desc);

    let frame_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(size_of::<SimParams>() as _),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(MaxParticles * size_of::<Particle>() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(
                        FrameW * FrameH * 4 * size_of::<f32>() as u64,
                    ),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(
                        FrameW * FrameH * 4 * size_of::<f32>() as u64,
                    ),
                },
                count: None,
            },
        ],
        label: Some("FrameBGL"),
    });

    let frame_pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        bind_group_layouts: &[&frame_bgl],
        label: Some("FramePipelineLayout"),
        push_constant_ranges: &[],
    });

    let frame_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("FramePipeline"),
        layout: Some(&frame_pl),
        module: &frame_shader,
        entry_point: Some("main"),
        compilation_options: PipelineCompilationOptions::default(),
        cache: None,
    });

    let ini_frame = vec![0.0_f32; (FrameW * FrameH * 4) as usize];
    let mut frame_buffers = vec![];
    let mut frame_bgs = vec![];

    for i in 0..2 {
        frame_buffers.push(device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("FrameBuf{}", i)),
            contents: bytemuck::cast_slice(&ini_frame),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        }));
    }

    for i in 0..2 {
        frame_bgs.push(device.create_bind_group(&BindGroupDescriptor {
            // NOTE: When writing to frame_buffer[1] we read from particle_buffer[1] and
            //                    to frame_buffer[0] we read from particle_buffer[0].
            entries: &to_bind_group_entries(&[
                &sim_ubo,
                &particle_buffers[(i + 1) % 2],
                &frame_buffers[(i + 0) % 2],
                &frame_buffers[(i + 1) % 2],
            ]),
            label: Some(&format!("FrameBg{}", i)),
            layout: &frame_bgl,
        }));
        println!("frame_bg[{}] [ubo, pbuf[{}], framebuf[{}], framebuf[{}]", i, (i+1)%2, (i+0)%2, (i+1)%2);
    }

    let frame_map_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("FrameMapBuffer"),
        mapped_at_creation: false,
        size: (FrameW * FrameH * 4 * (size_of::<f32>() as u64)),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    });

    return Scene {
        wctx,
        sim_params,
        sim_ubo,
        particle_bgs,
        particle_buffers,
        frame_bgs,
        frame_buffers,
        frame_map_buffer,
        sim_pipeline,
        frame_pipeline,
        sim_work_group_count,
    };
}

fn read_frame_buffer(dst: &mut[u8], scene: &Scene, idx: usize) {
        {
            let mut ce = scene.wctx.device.create_command_encoder(&CommandEncoderDescriptor::default());
            ce.copy_buffer_to_buffer(
                &scene.frame_buffers[idx],
                0u32.into(),
                &scene.frame_map_buffer,
                0u32.into(),
                scene.frame_buffers[idx].size(),
            );
            scene.wctx.queue.submit([ce.finish()]);
        }

        let slice = scene.frame_map_buffer.slice(..);
        let container = Arc::new(Mutex::new(Option::<Result<(), BufferAsyncError>>::None));
        let container2 = container.clone();
        slice.map_async(MapMode::Read, move |v| {
            let _ = container2.lock().unwrap().insert(v);
        });

        scene.wctx.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        for i in 1..100 {
            if container.lock().unwrap().is_some() {
                assert!(container.lock().unwrap().as_ref().unwrap().is_ok());
                println!("GOOD");

                let view = slice.get_mapped_range(); // BufferView derefences as &[u8]
                let pixels: &[f32] = bytemuck::cast_slice(&view);

                assert!(dst.len() >= pixels.len(), "expected {} >= {}", dst.len(), pixels.len());

                for j in 0..pixels.len() {
                    dst[j] = (pixels[j] * (255.5)) as u8;
                }

                drop(view);
                scene.frame_map_buffer.unmap();

                break;
            } else {
                println!("WAITING {} ms", i);
            }
            std::thread::sleep(std::time::Duration::from_millis(i));
        }

}

struct App {
    scene: Scene,
    frame_cntr: usize,
}

impl App {
    fn step(&mut self, render: bool) {
        let scene = &self.scene;
        let device = &scene.wctx.device;
        let queue = &scene.wctx.queue;


        // Step sim
        {
            let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("simStep"), });
            command_encoder.push_debug_group("sim");
            {
                let mut cpass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("simComputePass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&scene.sim_pipeline);
                cpass.set_bind_group(0, &scene.particle_bgs[self.frame_cntr % 2], &[]);
                cpass.dispatch_workgroups(((MaxParticles + ParticlesPerGroup - 1) / ParticlesPerGroup) as u32, 1, 1);
            }
            command_encoder.pop_debug_group();
            queue.submit(Some(command_encoder.finish()));
            device.poll(wgpu::Maintain::wait()).panic_on_timeout();
            println!("stepped sim.");
        }

        // Render
        if render {
            let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("render"), });
            command_encoder.push_debug_group("render");
            {
                let mut cpass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("renderComputePass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&scene.frame_pipeline);
                cpass.set_bind_group(0, &scene.frame_bgs[self.frame_cntr % 2], &[]);
                cpass.dispatch_workgroups(((FrameW + FrameWorkGroupSize-1) / FrameWorkGroupSize) as u32, ((FrameH + FrameWorkGroupSize-1) / FrameWorkGroupSize) as u32, 1);
                // cpass.dispatch_workgroups(4,4,1);
            }
            command_encoder.pop_debug_group();
            queue.submit(Some(command_encoder.finish()));
            device.poll(wgpu::Maintain::wait()).panic_on_timeout();

            let mut img = vec![0u8; (FrameW*FrameH*4) as usize];
            // read_frame_buffer(img.as_mut_slice(), scene, 0);
            read_frame_buffer(img.as_mut_slice(), scene, (self.frame_cntr+1) % 2);

            {
                // let fp = std::fs::File::open("/tmp/img.bin");
                // let fp = std::fs::OpenOptions::new().write(true).open("/tmp/img.bin").unwrap();
                use std::fs::File;
                use std::io::Write;
                let mut fp = std::fs::File::create("/tmp/img.bin").unwrap();
                fp.write(img.as_slice()).ok();
                println!("wrote '/tmp/img.bin'");
            }
        }

        self.frame_cntr += 1;
    }
}

fn main() {
    env_logger::init();

    let mut scene = go();
    let mut app = App { scene, frame_cntr: 0};
    app.step(false);
    app.step(false);
    app.step(false);
    app.step(true);
    app.step(true);
    app.step(true);
    app.step(true);
    app.step(true);
    app.step(true);

    println!("Hello, world!");
}
