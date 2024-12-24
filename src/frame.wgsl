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

	// De-duplicate
	for (var ii = 0u; ii < 4u; ii += 1u) {
		for (var jj = ii+1; jj < 4u; jj += 1u) {
			if (items[ii].i == items[jj].i) {
				items[jj].i = 0u;
				items[jj].d = 999999.;
			}
		}
	}
	
	// bubble sort
	for (var ii = 0u; ii < 3u; ii += 1u) {
		for (var jj = ii; jj < 3u; jj += 1u) {
			let kk = jj + 1;
			if (items[jj].d > items[kk].d) {
				let tmp = items[jj];
				items[jj] = items[kk];
				items[kk] = tmp;
			}
		}
	}

	// Show duplicates.
	for (var ii = 0u; ii < 1u; ii += 1u) {
		for (var jj = ii+1; jj < 4u; jj += 1u) {
			if (items[ii].i == items[jj].i) {color.b += 10.;}
		}
	}

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


		// Prevent duplicate (is this necessary?)
		for (var j = 0u; j < 4; j++) { if (items[j].i == pi) {continue;} }

		let part = particlesIn[pi];

		let d = length(part.p.xy - uv);

		for (var j = 0u; j < 4; j++) {
			let dj = items[j].d;
			if (d <= dj) {
				if (d < dj) {
					for (var k = j+1; k < 4; k++) {
						items[k] = items[k-1];
					}
				}
				items[j] = Item(pi, d);
				break;
			}
		}
	}

	// Very unlikely we have > 1 particle near us (us being the pixel)
	for (var j = 0u; j < 4; j++) {
		let p = particlesIn[items[j].i];
		let d = length(p.p.xy - uv);
		// color += vec4f(.1, .8*(1.-abs(cos(p.intensity))),abs(cos(p.intensity)), 1.) * exp(-d*5650.) * .0001;
		// color += vec4f(.1, .8*(1.-abs(cos(p.intensity))),abs(cos(p.intensity)), 1.) * exp(-d*650.) * .01;

		let s0 = exp(-d*4550.) * 2.;
		color += vec4f(vec3f(.5, 1., 1.) * s0, s0);

		// let s = exp(-d*250.) * .5;
		let s = .000002 / (.000001 + pow(d, 7.)) * (array<f32,4>(1.,.7,.5,.2))[j];
		// color += vec4f(.1, .8*(1.-abs(cos(p.intensity))),abs(cos(p.intensity)), 1.) * vec4f(s,s,s,1.);
	}

	{
		let p0 = particlesIn[items[0].i];
		let d0 = length(p0.p.xy - uv);
		let p1 = particlesIn[items[1].i];
		let d1 = length(p1.p.xy - uv);
		let p2 = particlesIn[items[2].i];
		let d2 = length(p2.p.xy - uv);
		let p3 = particlesIn[items[3].i];
		let d3 = length(p3.p.xy - uv);

		// let ll = cross(vec3(p1.p.xy, 1.), vec3(p2.p.xy, 1.));
		// let l = cross(ll, vec3(0.,0.,1.));
		// let d = abs(dot(l, vec3(uv - p1.p.xy,0.))) * 2.;
		// color += vec4f(d, d,d, d);

		// color += vec4f(1.,0.,0.,1.) * exp(-d3 * 1900.) * .0001;
		// color += vec4f(0.,1.,0.,1.) * exp(-d2 * 1950.) * .00003;
		// color += vec4f(0.,0.,1.,1.) * exp(-d3 * 100.);
	}

	// for (var j = 0u; j < 3; j++) {
	for (var j = 0u; j < 3; j++) {
	for (var k = j+1u; k < 4; k++) {
	{
		let pp0 = particlesIn[items[j].i].p;
		let pp1 = particlesIn[items[k].i].p;

		let ll = normalize(cross(vec3(pp0.xy, 1.), vec3(pp1.xy, 1.)));
		let d = abs(dot(ll, vec3(uv - pp0.xy, 0.)));
		var thresh = (1. - (f32(1 + j) * f32(1 + k)) / 25.) * .001;
		let z = .5 + (pp0.z * pp1.z);
		// let blur = 20.*abs(z);
		let blur = distance(pp0,pp1) * 10.;
		thresh *= blur + 1.;

		// let thresh = .004;
		if (d < thresh) {
			let dir = normalize(pp1.xy - pp0.xy);
			let t = dot(uv - pp0.xy, dir);
			if t > 0. && t < 1. {
			var s = 1. - (f32(1 + j) * f32(1 + k)) / 25. + .05;
			s = s * smoothstep(thresh, 0., d);
			// s *= 1. /blur;
			s *= 1./(.5+blur);
			color.g += s*2.;
			color.b += s*abs(sin(pp0.z*10.));
			color.a += 2.;
			}
		}
	}
	}}


	color = vec4f(color.rgb/(.000001+color.a), color.a);
	// color += frameIn[py*fw+px] * .95;
	color += frameIn[py*fw+px] * .85;

	// frameOut[py*fw+px] = vec4f(sin(3.141*f32(px)),f32(py)*.001,0.,1.);
	frameOut[py*fw+px] = color;

	let ntop = vec4u(
		items[0].i,
		items[1].i,
		items[2].i,
		items[3].i);
	frameStateOut[py*fw+px] = FrameState(ntop, rstate);
}
