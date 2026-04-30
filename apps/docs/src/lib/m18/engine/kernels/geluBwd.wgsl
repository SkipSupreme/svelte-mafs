// gelu backward (tanh approximation, matching the forward gelu kernel).
//   Forward: y = 0.5 x (1 + tanh(u)),  u = c (x + a x³)
//   dy/dx   = 0.5 (1 + tanh u) + 0.5 x · (1 − tanh² u) · c (1 + 3 a x²)
// Elementwise: thread i writes dx[i] = dy[i] · dy/dx.

const C: f32 = 0.7978845608028654; // sqrt(2 / pi)
const A_COEF: f32 = 0.044715;

@group(0) @binding(0) var<storage, read>          dy:   array<f32>;
@group(0) @binding(1) var<storage, read>          x:    array<f32>;
@group(0) @binding(2) var<storage, read_write>    dx:   array<f32>;
@group(0) @binding(3) var<uniform>                dims: vec4<u32>; // (N, _, _, _)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= dims.x) { return; }
  let v = x[i];
  let u = C * (v + A_COEF * v * v * v);
  let th = tanh(u);
  let dudx = C * (1.0 + 3.0 * A_COEF * v * v);
  let dydx = 0.5 * (1.0 + th) + 0.5 * v * (1.0 - th * th) * dudx;
  dx[i] = dy[i] * dydx;
}
