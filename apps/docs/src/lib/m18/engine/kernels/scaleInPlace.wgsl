// Multiply a buffer in place by a scalar held in a uniform.
//   Inputs : g [N], scale (uniform)
//   Output : g[i] *= scale
// Used for grad clipping: after the global-norm readback, the driver computes
// scale = min(1, maxNorm / sqrt(Σ Σ g²)) and dispatches this kernel for every
// parameter buffer.

@group(0) @binding(0) var<storage, read_write>    g: array<f32>;
@group(0) @binding(1) var<uniform>                u: vec4<u32>; // (N, _, _, _)
@group(0) @binding(2) var<uniform>                s: vec4<f32>; // (scale, _, _, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.x) { return; }
  g[i] = g[i] * s.x;
}
