@group(0) @binding(0) var<uniform> sim_params: SimParams;

@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;

@group(0) @binding(2) var<storage, read> frameIn: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> frameOut: array<vec4f>;

@group(0) @binding(4) var<storage, read> frameStateIn: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> frameStateOut: array<vec4f>;

struct FrameState {
	top: vec4u, // tracking info
	rstate: u32, // prng state
}


@compute
@workgroup_size(16,16)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let npart = arrayLength(&particlesIn);
	let fw = sim_params.frame_w;
	let fh = sim_params.frame_h;
	let ffw = f32(sim_params.frame_w);
	let ffh = f32(sim_params.frame_h);

	let px = global_invocation_id.x;
	let py = global_invocation_id.y;
	let screenp = vec2f(f32(px),f32(py));


	let u = (f32(global_invocation_id.x) - ffw * .5) / (ffh * .5);
	let v = (f32(global_invocation_id.y) - ffh * .5) / (ffh * .5);
	let uv = vec2f(u,v);

	if px >= fw || py >= fh { return; }

	var color = vec4f(0.);

	for (var pi: u32 = 0; pi < npart; pi += u32(1)) {
		let p = particlesIn[pi];
		let pp = p.p;
		// let d = length(pp.xy - screenp);
		let d = length(pp.xy - uv);
		// if d < 2.5 {
		// if d < .15 {
			color += vec4f(.1, .8*(1.-abs(cos(p.intensity))),abs(cos(p.intensity)), 1.) * exp(-d*6550.);
			// color += vec4f(.1,.2,1., 1.) * exp(-d*6550.);
		// }
	}

	// color.r += noise13(vec3(uv,1.) * 150.);

	color = vec4f(color.rgb/(.000001+color.a), color.a);

	let a = mulberry32(px*4000+py);
	let b = mulberry32(a.y);
	let c = mulberry32(b.y);
	color.r = f32(c.x % 255) / 255.;

	// color += frameIn[py*fw+px] * .9;


	frameOut[py*fw+px] = color;
}

