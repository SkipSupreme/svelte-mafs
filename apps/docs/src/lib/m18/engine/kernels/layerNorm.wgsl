// layerNorm: per row r, y[r,k] = gamma[k] * (x[r,k] - mu) / sqrt(var + eps) + beta[k]
// Dispatch: rows workgroups of 64 threads each. Workgroup cooperates on the
// row-reduction; threads stride k by workgroup_size.

const WG: u32 = 64u;
const EPS: f32 = 1e-5;

@group(0) @binding(0) var<storage, read>          x:     array<f32>; // [rows, d]
@group(0) @binding(1) var<storage, read>          gamma: array<f32>; // [d]
@group(0) @binding(2) var<storage, read>          beta:  array<f32>; // [d]
@group(0) @binding(3) var<storage, read_write>    y:     array<f32>; // [rows, d]
@group(0) @binding(4) var<uniform>                dims:  vec4<u32>;  // (rows, d, _, _)

var<workgroup> sumScratch: array<f32, WG>;

fn wgReduce(local: u32, val: f32) -> f32 {
  sumScratch[local] = val;
  workgroupBarrier();
  var step: u32 = WG / 2u;
  loop {
    if (step == 0u) { break; }
    if (local < step) { sumScratch[local] = sumScratch[local] + sumScratch[local + step]; }
    workgroupBarrier();
    step = step / 2u;
  }
  return sumScratch[0];
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let r = wid.x; let local = lid.x;
  let d = dims.y;
  if (r >= dims.x) { return; }

  // pass 1: mean
  var partial: f32 = 0.0;
  var k: u32 = local;
  loop { if (k >= d) { break; } partial = partial + x[r * d + k]; k = k + WG; }
  let mu = wgReduce(local, partial) / f32(d);

  // pass 2: variance
  workgroupBarrier();
  partial = 0.0;
  k = local;
  loop { if (k >= d) { break; } let z = x[r * d + k] - mu; partial = partial + z * z; k = k + WG; }
  let varv = wgReduce(local, partial) / f32(d);
  let inv = 1.0 / sqrt(varv + EPS);

  // pass 3: write
  workgroupBarrier();
  k = local;
  loop {
    if (k >= d) { break; }
    y[r * d + k] = gamma[k] * (x[r * d + k] - mu) * inv + beta[k];
    k = k + WG;
  }
}
