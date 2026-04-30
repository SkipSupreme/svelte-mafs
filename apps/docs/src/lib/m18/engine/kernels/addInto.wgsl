// In-place accumulator: b += a (elementwise). Distinct buffers required —
// WGSL forbids the same buffer being bound as read AND read_write within a
// single compute pass, so we can't fold this into residualAdd(a, b, a).

@group(0) @binding(0) var<storage, read>          a:    array<f32>;
@group(0) @binding(1) var<storage, read_write>    b:    array<f32>;
@group(0) @binding(2) var<uniform>                dims: vec4<u32>; // (N, _, _, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.x) { return; }
  b[i] = b[i] + a[i];
}
