// LayerNorm backward — dx branch.
//   Inputs : dy [rows, d], x [rows, d], gamma [d]
//   Outputs: dx [rows, d]
//   Math   : g_k = γ_k · dy_k. dx_k = (1/σ) · (g_k − meanG − z_k · meanGz),
//            where z_k = (x_k − μ)/σ, meanG = (1/d)Σ g_j, meanGz = (1/d)Σ g_j z_j.
// One workgroup per row. Threads stride k by WG.

const WG: u32 = 64u;
const EPS: f32 = 1e-5;

@group(0) @binding(0) var<storage, read>          dy:    array<f32>;
@group(0) @binding(1) var<storage, read>          x:     array<f32>;
@group(0) @binding(2) var<storage, read>          gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write>    dx:    array<f32>;
@group(0) @binding(4) var<uniform>                dims:  vec4<u32>; // (rows, d, _, _)

var<workgroup> red: array<f32, WG>;
var<workgroup> muS: f32;
var<workgroup> invS: f32;
var<workgroup> mGS: f32;
var<workgroup> mGzS: f32;

fn wgSum(local: u32, val: f32) -> f32 {
  red[local] = val;
  workgroupBarrier();
  var step: u32 = WG / 2u;
  loop {
    if (step == 0u) { break; }
    if (local < step) { red[local] = red[local] + red[local + step]; }
    workgroupBarrier();
    step = step / 2u;
  }
  return red[0];
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let r = wid.x; if (r >= dims.x) { return; }
  let d = dims.y; let local = lid.x;

  // mean of x
  var partial: f32 = 0.0;
  var k: u32 = local;
  loop { if (k >= d) { break; } partial = partial + x[r * d + k]; k = k + WG; }
  let mu = wgSum(local, partial) / f32(d);
  if (local == 0u) { muS = mu; }
  workgroupBarrier();

  // variance of x
  partial = 0.0;
  k = local;
  loop {
    if (k >= d) { break; }
    let z = x[r * d + k] - muS;
    partial = partial + z * z;
    k = k + WG;
  }
  let varv = wgSum(local, partial) / f32(d);
  if (local == 0u) { invS = 1.0 / sqrt(varv + EPS); }
  workgroupBarrier();

  // mean(g) and mean(g·z) — g = γ · dy, z = (x − μ)/σ
  partial = 0.0;
  k = local;
  loop {
    if (k >= d) { break; }
    partial = partial + gamma[k] * dy[r * d + k];
    k = k + WG;
  }
  let mG = wgSum(local, partial) / f32(d);
  if (local == 0u) { mGS = mG; }
  workgroupBarrier();

  partial = 0.0;
  k = local;
  loop {
    if (k >= d) { break; }
    let z = (x[r * d + k] - muS) * invS;
    let g = gamma[k] * dy[r * d + k];
    partial = partial + g * z;
    k = k + WG;
  }
  let mGz = wgSum(local, partial) / f32(d);
  if (local == 0u) { mGzS = mGz; }
  workgroupBarrier();

  // write dx
  k = local;
  loop {
    if (k >= d) { break; }
    let z = (x[r * d + k] - muS) * invS;
    let g = gamma[k] * dy[r * d + k];
    dx[r * d + k] = invS * (g - mGS - z * mGzS);
    k = k + WG;
  }
}
