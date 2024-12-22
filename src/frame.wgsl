@group(0) @binding(0) var<uniform> sim_params: SimParams;

@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;

@group(0) @binding(2) var<storage, read> frameIn: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> frameOut: array<vec4f>;

@group(0) @binding(4) var<storage, read> frameStateIn: array<FrameState>;
@group(0) @binding(5) var<storage, read_write> frameStateOut: array<FrameState>;

// Note: as a compute shader, no need for `top` to be limited to vec4u. It could be an array.
struct FrameState {
	top: vec4u, // tracking info
	rstate: u32, // prng state
}

struct Item {
	i: u32,
	d: f32
}

// ------------------------------------------------------------------------------------------------------------------------
//
// Version 1, wherein I loop over all particles to render them (slow)
//

/*
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
		let d = length(pp.xy - uv);
		color += vec4f(.1, .8*(1.-abs(cos(p.intensity))),abs(cos(p.intensity)), 1.) * exp(-d*6550.);
	}

	// color.r += noise13(vec3(uv,1.) * 150.);

	color = vec4f(color.rgb/(.000001+color.a), color.a);

	{
		let rng_tuple = mulberry32(frameStateIn[py*fw+px].rstate);
		let r = rng_tuple.x;
		let nrstate = rng_tuple.y;
		color.r += f32(r % 4096) / 4096.;
		frameStateOut[py*fw+px].rstate = nrstate;
	}

	color += frameIn[py*fw+px] * .9;


	frameOut[py*fw+px] = color;
}
*/

// ------------------------------------------------------------------------------------------------------------------------
//
// Version 2, wherein I sample a few and propogate spatially.
//

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

	/*
	for (var pi: u32 = 0; pi < npart; pi += u32(1)) {
		let p = particlesIn[pi];
		let pp = p.p;
		let d = length(pp.xy - uv);
		color += vec4f(.1, .8*(1.-abs(cos(p.intensity))),abs(cos(p.intensity)), 1.) * exp(-d*6550.);
	}
	*/

	// color.r += noise13(vec3(uv,1.) * 150.);


	let fstate0 = frameStateIn[py*fw+px];
	var rstate = fstate0.rstate;

	var items : array<Item,4> = array<Item,4>(
		Item (fstate0.top[0], length(particlesIn[fstate0.top[0]].p.xy - uv)),
		Item (fstate0.top[1], length(particlesIn[fstate0.top[1]].p.xy - uv)),
		Item (fstate0.top[2], length(particlesIn[fstate0.top[2]].p.xy - uv)),
		Item (fstate0.top[3], length(particlesIn[fstate0.top[3]].p.xy - uv))
	);

	let n_particles_to_sample = 50u;
	let ddy = array<i32,4>(0,0,1,-1);
	let ddx = array<i32,4>(1,-1,0,0);

	for (var ii = 0u; ii < n_particles_to_sample + 4u; ii += 1u) {
		var pi : u32;

		if (ii < n_particles_to_sample) {
			// Random selection
			let rng_tuple = mulberry32(rstate);
			rstate = rng_tuple.y;
			pi = rng_tuple.x % npart;
		} else {
			// Propagate spatially.
			let iii = ii - n_particles_to_sample;
			let neighbory = clamp(u32(i32(py) + ddy[iii]), 0u, fh);
			let neighborx = clamp(u32(i32(px) + ddx[iii]), 0u, fw);
			pi = frameStateIn[neighbory*fw+neighborx].top.x;
		}

		let part = particlesIn[pi];

		let d = length(part.p.xy - uv);

		for (var j = 0u; j < 4; j++) {
			let dj = items[j].d;
			if (d < dj) {
				for (var k = j+1; k < 4; k++) {
					items[k] = items[k-1];
				}
				items[j] = Item(pi, d);
				break;
			}
		}
	}

	for (var j = 0u; j < 4; j++) {
		let p = particlesIn[items[j].i];
		let d = length(p.p.xy - uv);
		// color += vec4f(.1, .8*(1.-abs(cos(p.intensity))),abs(cos(p.intensity)), 1.) * exp(-d*6550.);
		color += vec4f(.1, .8*(1.-abs(cos(p.intensity))),abs(cos(p.intensity)), 1.) * exp(-d*3650.);
	}


	color = vec4f(color.rgb/(.000001+color.a), color.a);
	color += frameIn[py*fw+px] * .9;

	frameOut[py*fw+px] = color;

	let ntop = vec4u(
		items[0].i,
		items[1].i,
		items[2].i,
		items[3].i);
	frameStateOut[py*fw+px] = FrameState(ntop, rstate);
}
