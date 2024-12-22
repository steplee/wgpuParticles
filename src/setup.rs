use futures::executor::block_on;
use wgpu::{Adapter, BindGroup, BindGroupEntry, Buffer, ComputePipeline, Device, Queue, ShaderModuleDescriptor};

pub struct WebGpuCtx {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub fn setup() -> WebGpuCtx {
    let instance_descriptor = wgpu::InstanceDescriptor::default();
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    println!("adapters:");
    adapters
        .iter()
        .enumerate()
        .for_each(|(i, a)| println!("{}: {:?}", i, a));
    let adapter = adapters.into_iter().nth(0).unwrap();

    async fn get_it(adapter: &Adapter) -> (Device, Queue) {
        let trace_dir = std::env::var("WGPU_TRACE");

        // let needed_limits = E::required_limits().using_resolution(adapter.limits());
        let needed_limits = wgpu::Limits::default();
        let optional_features = wgpu::Features::default();
        let required_features = wgpu::Features::default();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: (optional_features & adapter.features()) | required_features,
                    required_limits: needed_limits,
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        return (device, queue);
    }

    let (device, queue) = block_on(get_it(&adapter));

    return WebGpuCtx {
        instance,
        adapter,
        device,
        queue,
    };
}

pub fn to_bind_group_entries<'a>(items: &[&'a Buffer]) -> Vec<BindGroupEntry<'a>> {
    return items
        .iter()
        .enumerate()
        .map(|(i, item)| BindGroupEntry {
            binding: i as u32,
            resource: (*item).as_entire_binding(),
        })
        .collect();
}

/*
pub fn include_combined_wgsls<'a>(label: &str, paths: &[&'a str]) -> ShaderModuleDescriptor<'a> {

    let mut source = String::new();
    for path in paths {
        // source += &format!("{}", include_str!(path));
        // source += &format!("{}", "a");
    }

    /*

            $crate::ShaderModuleDescriptor {
                label: Some($($token)*),
                source: $crate::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!($($token)*))),
            }
    */
    todo!()
}
*/

#[macro_export]
macro_rules! include_combined_wgsl_2 {
    ($label:tt; $a:tt, $b:tt) => {
        {
            let src = include_str!($a).to_owned()
                    + include_str!($b);

            wgpu::ShaderModuleDescriptor {
                label: Some($label),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(src))
            }
        }
    }
}
#[macro_export]
macro_rules! simple {
    ($label:literal) => {
        {
            stringify!($label)
        }
    }
}
