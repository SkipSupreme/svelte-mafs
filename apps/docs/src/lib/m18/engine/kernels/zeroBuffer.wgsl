// Fill a buffer with zeros. Used to clear gradient buffers between steps.

@group(0) @binding(0) var<storage, read_write>    g:    array<f32>;
@group(0) @binding(1) var<uniform>                dims: vec4<u32>; // (N, _, _, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.x) { return; }
  g[i] = 0.0;
}
