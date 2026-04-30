// softmax + cross-entropy backward.
//   Inputs : logits [rows, V] (read), targets [rows] (read)
//   Outputs: dlogits [rows, V] (write)
//   Math   : dlogits[i, v] = (softmax(logits[i])[v] − 1[v == targets[i]]) / rows.
// One workgroup per row; one thread per vocab entry. WG size 64 strides over V
// when V > 64 (here V=65, so a single stride is enough).
// Computes the loss accumulator on the side: each WG writes
//   loss[r] = log Σ_v exp(z_v − maxz) + (maxz − z_target).
// The driver mean-reduces loss[] on the CPU side.

const WG: u32 = 64u;

@group(0) @binding(0) var<storage, read>          logits:  array<f32>;
@group(0) @binding(1) var<storage, read>          targets: array<i32>;
@group(0) @binding(2) var<storage, read_write>    dlogits: array<f32>;
@group(0) @binding(3) var<storage, read_write>    lossPer: array<f32>; // [rows] per-row NLL
@group(0) @binding(4) var<uniform>                dims:    vec4<u32>;  // (rows, V, _, _)

var<workgroup> mxScratch: array<f32, WG>;
var<workgroup> sumScratch: array<f32, WG>;
var<workgroup> logSum: f32;
var<workgroup> rowMax: f32;

// 'target' is a reserved word in WGSL (texture sampling target etc.) — use
// 'tgt' here to avoid the parser rejecting the line.

fn wgMax(local: u32, val: f32) -> f32 {
  mxScratch[local] = val;
  workgroupBarrier();
  var step: u32 = WG / 2u;
  loop {
    if (step == 0u) { break; }
    if (local < step) {
      let other = mxScratch[local + step];
      if (other > mxScratch[local]) { mxScratch[local] = other; }
    }
    workgroupBarrier();
    step = step / 2u;
  }
  return mxScratch[0];
}

fn wgSum(local: u32, val: f32) -> f32 {
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
  let rows = dims.x; let V = dims.y;
  let r = wid.x; if (r >= rows) { return; }
  let local = lid.x;
  let inv = 1.0 / f32(rows);

  // Pass 1: per-row max.
  var mx: f32 = -1e30;
  var v: u32 = local;
  loop {
    if (v >= V) { break; }
    let z = logits[r * V + v];
    if (z > mx) { mx = z; }
    v = v + WG;
  }
  let rmax = wgMax(local, mx);
  if (local == 0u) { rowMax = rmax; }
  workgroupBarrier();

  // Pass 2: per-row exp-sum.
  var s: f32 = 0.0;
  v = local;
  loop {
    if (v >= V) { break; }
    s = s + exp(logits[r * V + v] - rowMax);
    v = v + WG;
  }
  let rsum = wgSum(local, s);
  if (local == 0u) { logSum = log(rsum); }
  workgroupBarrier();

  // Pass 3: write dlogits and the per-row loss.
  let tgt = targets[r];
  v = local;
  loop {
    if (v >= V) { break; }
    let p = exp(logits[r * V + v] - rowMax) / rsum;
    let one = select(0.0, 1.0, i32(v) == tgt);
    dlogits[r * V + v] = (p - one) * inv;
    v = v + WG;
  }
  if (local == 0u) {
    let zt = logits[r * V + u32(tgt)];
    lossPer[r] = logSum + rowMax - zt;
  }
}
