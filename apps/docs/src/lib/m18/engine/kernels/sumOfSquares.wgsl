// Sum-of-squares reduction over a single buffer.
//   Input : g [N]  — a parameter's gradient
//   Output: out[outIdx]  — single f32 slot in a small global "norm² of every
//                          parameter" buffer the driver readbacks once per step.
// Workgroup-local two-stage reduction: each WG sums its slice of g into a
// shared partial; WG 0 then writes to out[outIdx]. To keep this simple at our
// scale, we use *one* workgroup per call and stride threads through N. With
// our largest grad buffer being d_model · d_ff = 64·256 = 16,384 f32, a single
// 64-thread workgroup loops 256 times — fine.

const WG: u32 = 64u;

@group(0) @binding(0) var<storage, read>          g:    array<f32>;
@group(0) @binding(1) var<storage, read_write>    out:  array<f32>;
@group(0) @binding(2) var<uniform>                dims: vec4<u32>; // (N, outIdx, _, _)

var<workgroup> red: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let local = lid.x;
  let N = dims.x; let outIdx = dims.y;

  var partial: f32 = 0.0;
  var i: u32 = local;
  loop {
    if (i >= N) { break; }
    let v = g[i];
    partial = partial + v * v;
    i = i + WG;
  }
  red[local] = partial;
  workgroupBarrier();

  var step: u32 = WG / 2u;
  loop {
    if (step == 0u) { break; }
    if (local < step) { red[local] = red[local] + red[local + step]; }
    workgroupBarrier();
    step = step / 2u;
  }

  if (local == 0u) { out[outIdx] = red[0]; }
}
