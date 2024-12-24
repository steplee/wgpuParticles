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

	// let dt = .1;
	var dt = .01;
	dt *= 1.;

	let p0 = part.p;
	let v0 = part.v;

	// var a = get_random_acc(p0 * 15.5);
	// var a = get_random_acc(p0 * vec3(15.5, 15.5, 6.5));

	var a = get_random_acc(p0 * vec3(15.5, 15.5, 9.5));
	// var a = get_random_acc(p0 * vec3((abs(p0.z)+abs(p0.y+1.)+1.)*15.5, (abs(p0.z)+abs(p0.y+1.)+1.)*15.5, .5));

	// a = a*abs(a);

	// a += -v0 * .97; // drag
	a += -v0 * (.0097/dt); // drag



	part.v = v0 + a * dt;
	part.p = p0 + part.v * dt + a * .5 * dt * dt;

	particlesOut[idx] = part;
}
