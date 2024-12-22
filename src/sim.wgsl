@group(0) @binding(0) var<uniform> sim_params: SimParams;
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesOut: array<Particle>;

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let npart = arrayLength(&particlesIn);
	let idx = global_invocation_id.x;
	if idx >= npart { return; }

	var part = particlesIn[idx];

	let p0 = part.p;
	let v0 = part.v;

	var a = get_random_acc(p0 * 15.5);
	a += -v0 * .7; // drag

	// let dt = .1;
	let dt = .01;

	part.v = v0 + a * dt;
	part.p = p0 + part.v * dt + a * .5 * dt * dt;

	particlesOut[idx] = part;
}
