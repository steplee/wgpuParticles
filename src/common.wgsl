struct SimParams {
    time: f32,
    speed: f32,
    chaos: f32,
	frame_w: u32,
	frame_h: u32,
}

struct Particle {
	p: vec3f,
	v: vec3f,
    intensity: f32,
    size: f32,
}


fn rand11(a: f32) -> f32 {
	return fract(sin(a*1737.33-20.01)*73.1);
}

fn rand13(a: vec3f) -> f32 {
	return fract(sin(dot(a, vec3f(3.133, -1.991, 6.1)-1.01))*693.1739);
}

fn smoo(a: vec3f) -> vec3f {
	return vec3f(
	3.*a.x*a.x - 2.*a.x*a.x*a.x,
	3.*a.y*a.y - 2.*a.y*a.y*a.y,
	3.*a.z*a.z - 2.*a.z*a.z*a.z);
}

fn noise13(a: vec3f) -> f32 {
	let ia = floor(a);
	let fa = fract(a);

	/*
	for (var dz=0; dz<2; dz++) {
		for (var dy=0; dy<2; dy++) {
			for (var dx=0; dx<2; dx++) {
				let wz = fa.z;
			}
		}
	}
	*/

	let d = vec2f(0.,1.);
	let s_000 = rand13(ia + d.xxx);
	let s_100 = rand13(ia + d.yxx);
	let s_110 = rand13(ia + d.yyx);
	let s_010 = rand13(ia + d.xyx);
	let s_001 = rand13(ia + d.xxy);
	let s_101 = rand13(ia + d.yxy);
	let s_111 = rand13(ia + d.yyy);
	let s_011 = rand13(ia + d.xyy);

	let w1 = smoo(fa);
	let w0 = 1. - w1;

	let o =
		s_000 * w0.x * w0.y * w0.z +
		s_100 * w1.x * w0.y * w0.z +
		s_110 * w1.x * w1.y * w0.z +
		s_010 * w0.x * w1.y * w0.z +
		s_001 * w0.x * w0.y * w1.z +
		s_101 * w1.x * w0.y * w1.z +
		s_111 * w1.x * w1.y * w1.z +
		s_011 * w0.x * w1.y * w1.z;

	return o;
}

fn get_random_acc(a: vec3f) -> vec3f {
	let x = noise13(a);
	let y = noise13(a + vec3f(-9.4, 99.1, -12.2));
	let z = noise13(a + vec3f(42.2, -13.5, 22.23));
	return vec3f(x,y,z) - .5;
}


// https://stackoverflow.com/questions/17035441/looking-for-decent-quality-prng-with-only-32-bits-of-state
fn mulberry32(state: u32) -> vec2u {
	let nstate = state + 0x6D2B79F5u;
	
    var z = (state ^ state >> 15) * (1 | state);
    z ^= z + (z ^ z >> 7) * (61 | z);
	z = z ^ z >> 14;
	return vec2u(z, nstate);
}
