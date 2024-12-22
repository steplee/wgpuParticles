struct SimParams {
    time: f32,
    speed: f32,
    chaos: f32,
	frame_w: u32,
	frame_h: u32,
}

@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read> frameIn: array<vec4f>;

struct VertexOutput {
	@builtin(position) pos: vec4f,
	@location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
	let inds = array<u32,6>(0u, 1u, 2u, 2u, 3u, 0u);
	let poss = array<vec2f,4>(
	vec2f(-1., -1.),
	vec2f( 1., -1.),
	vec2f( 1.,  1.),
	vec2f(-1.,  1.));

	let ind = inds[vid];

	let p = vec4f(poss[ind], 0., 1.);
	let uv = p.xy * .5 + .5;

	return VertexOutput(p,uv);
}

@fragment
fn fs_main(vo: VertexOutput) -> @location(0) vec4f {
	let fx = u32(vo.uv.x * f32(sim_params.frame_w) + .5);
	let fy = u32(vo.uv.y * f32(sim_params.frame_h) + .5);
	let c = frameIn[fy*sim_params.frame_w+fx];
	return c;
}
