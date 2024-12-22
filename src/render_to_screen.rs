use winit::{
    application::ApplicationHandler, dpi::LogicalSize, event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::Window
};
use wgpu::{
    include_wgsl, util::{BufferInitDescriptor, DeviceExt}, Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferAsyncError, BufferBindingType, BufferDescriptor, BufferSize, BufferUsages, ColorTargetState, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, FragmentState, MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PrimitiveState, Queue, RenderPipeline, RenderPipelineDescriptor, ShaderStages, VertexState
};

use crate::setup::{to_bind_group_entries, WebGpuCtx};

/*
struct ScreenCtx {
    event_loop: EventLoop,
    window: Window,
}
*/

// fn run_loop(f: &dyn Fn() -> ()) {
fn run_loop(mut f: Box<dyn ApplicationHandler>) {
    let event_loop = EventLoop::new().unwrap();
    let mut attr = winit::window::Window::default_attributes()
        .with_title("Shader")
        .with_inner_size(LogicalSize::new(512, 512));
    let window = event_loop.create_window(attr).unwrap();

    // event_loop.set_control_flow(ControlFlow::Poll); // loop fast, don't wait for IO
    event_loop.set_control_flow(ControlFlow::Wait); // wait for IO.

    event_loop.run_app(&mut f);
}

pub struct RenderToScreen {
    // screen_ctx: ScreenCtx,
    bgs: Vec<BindGroup>,
    pipeline: RenderPipeline,
    frame_w: u32,
    frame_h: u32,
}

// @group(0) @binding(0) var<storage, read> frameIn: array<vec4f>;

impl RenderToScreen {

    pub fn new(wctx: WebGpuCtx, w:u32, h:u32, framebuffers: &Vec<Buffer>) -> Self {
        let device = &wctx.device;
        let queue = &wctx.queue;

        let render_to_screen_shader_desc = include_wgsl!("renderToScreen.wgsl");
        let render_to_screen_shader = device.create_shader_module(render_to_screen_shader_desc);

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(((w*h*4) as u64)*(size_of::<f32>() as u64)),
                    },
                    count: None,
                },
            ],
            label: Some("renderToScreenBGL"),
        });

        let pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&bgl],
            label: Some("RenderToScreenPL"),
            push_constant_ranges: &[],
        });


            // module: &render_to_screen_shader,
            // entry_point: Some("main"),

        let vertex = VertexState {
            module: &render_to_screen_shader,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[],
        };

        let fragment = FragmentState {
            module: &render_to_screen_shader,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[
                Some(ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8Unorm,
                    blend: None,
                    write_mask: Default::default(),
                })
            ],
        };

        let mut primitive = PrimitiveState::default();
        primitive.topology = wgpu::PrimitiveTopology::TriangleList;
        primitive.cull_mode = None;

        let multisample = Default::default();

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("RenderToScreenPipeline"),
            layout: Some(&pl),

            vertex: vertex,
            primitive: primitive,
            depth_stencil: None,
            multisample: multisample,
            fragment: Some(fragment),
            multiview: None,
            cache: None,

        });

        let mut bgs = vec![];
        for i in 0..2 {
            // NOTE: On even iterations we write to particle_buffers[1] and read from
            // particle_buffers[0]. And the opposite for odd iterations.
            bgs.push(device.create_bind_group(&BindGroupDescriptor {
                entries: &to_bind_group_entries(&[
                    &framebuffers[(i + 0) % 2],
                ]),
                label: Some(&format!("ParticleBg{}", i)),
                layout: &bgl,
            }));
        }


        return RenderToScreen {
            bgs, pipeline, frame_w: w, frame_h: h
        }
    }

    pub fn show_storage_buffer(&mut self, buf: &Buffer) {
    }
}
