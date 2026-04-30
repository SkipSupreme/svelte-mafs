// residualAdd: out = a + b elementwise.

@group(0) @binding(0) var<storage, read>          a:    array<f32>;
@group(0) @binding(1) var<storage, read>          b:    array<f32>;
@group(0) @binding(2) var<storage, read_write>    out:  array<f32>;
@group(0) @binding(3) var<uniform>                dims: vec4<u32>; // (N, _, _, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.x) { return; }
  out[i] = a[i] + b[i];
}
