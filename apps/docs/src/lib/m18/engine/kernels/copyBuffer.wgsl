// One-line memcpy kernel for f32 buffers. WGSL compute passes can't issue
// buffer-to-buffer copies natively (that's an encoder-level command), so this
// fills the gap when a forward / backward step needs to fork an intermediate
// into a separate persistent slot.

@group(0) @binding(0) var<storage, read>          src:  array<f32>;
@group(0) @binding(1) var<storage, read_write>    dst:  array<f32>;
@group(0) @binding(2) var<uniform>                dims: vec4<u32>; // (N, _, _, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.x) { return; }
  dst[i] = src[i];
}
