// gelu (tanh approximation): y = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
// Matches torch.nn.GELU(approximate='tanh') and the M18 nanoGPT reference.

const C: f32 = 0.7978845608028654; // sqrt(2 / pi)

@group(0) @binding(0) var<storage, read>          x:    array<f32>;
@group(0) @binding(1) var<storage, read_write>    y:    array<f32>;
@group(0) @binding(2) var<uniform>                dims: vec4<u32>; // (N, _, _, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.x) { return; }
  let v = x[i];
  let u = C * (v + 0.044715 * v * v * v);
  y[i] = 0.5 * v * (1.0 + tanh(u));
}
