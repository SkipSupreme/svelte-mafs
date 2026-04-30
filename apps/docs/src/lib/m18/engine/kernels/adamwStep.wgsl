// AdamW one-step (PyTorch ordering: decoupled weight decay → m/v update →
// bias-corrected step). Mirrors torch.optim.AdamW's per-element math line for
// line so the curve matches the reference oracle in docs/research/.
//
//   1. θ ← θ · (1 − lr · λ)
//   2. m ← β₁ m + (1 − β₁) g
//   3. v ← β₂ v + (1 − β₂) g²
//   4. θ ← θ − (lr / (1 − β₁ᵗ)) · m / (√(v / (1 − β₂ᵗ)) + ε)
//
// One thread per parameter element. The driver passes (lr, β₁, β₂, ε, λ,
// 1−β₁ᵗ, 1−β₂ᵗ) in a single uniform block.

struct HyperParams {
  lr:     f32,
  beta1:  f32,
  beta2:  f32,
  eps:    f32,
  lambda: f32,
  bc1:    f32, // 1 − β₁ᵗ (precomputed by driver)
  bc2:    f32, // 1 − β₂ᵗ
  _pad:   f32,
};

@group(0) @binding(0) var<storage, read_write>    theta: array<f32>;
@group(0) @binding(1) var<storage, read>          grad:  array<f32>;
@group(0) @binding(2) var<storage, read_write>    m:     array<f32>;
@group(0) @binding(3) var<storage, read_write>    v:     array<f32>;
@group(0) @binding(4) var<uniform>                u:     vec4<u32>; // (N, _, _, _)
@group(0) @binding(5) var<uniform>                p:     HyperParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.x) { return; }

  // 1. decoupled weight decay
  var t = theta[i] * (1.0 - p.lr * p.lambda);

  // 2. m, v running averages
  let g = grad[i];
  let mNew = p.beta1 * m[i] + (1.0 - p.beta1) * g;
  let vNew = p.beta2 * v[i] + (1.0 - p.beta2) * g * g;
  m[i] = mNew;
  v[i] = vNew;

  // 3. bias-corrected step
  let stepSize = p.lr / p.bc1;
  let denom = sqrt(vNew) / sqrt(p.bc2) + p.eps;
  t = t - stepSize * mNew / denom;

  theta[i] = t;
}
