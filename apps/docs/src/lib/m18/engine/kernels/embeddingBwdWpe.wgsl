// Embedding backward — wpe branch.
//   Inputs : dEmb [N, d] (N = B·T)
//   Output : dWpe [T, d]
//
// One workgroup per position p ∈ [0, T). Thread k sums dEmb[i, k] over all i
// with (i mod T) == p — equivalently i ∈ {p, p+T, p+2T, …, p+(B−1)T}.
// Each WG owns a unique output row → no cross-WG races, no atomics needed.

const WG: u32 = 64u;

@group(0) @binding(0) var<storage, read>          dEmb: array<f32>;
@group(0) @binding(1) var<storage, read_write>    dWpe: array<f32>;
@group(0) @binding(2) var<uniform>                dims: vec4<u32>; // (N, T, d, _)

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let p = wid.x; let k = lid.x;
  let N = dims.x; let T = dims.y; let d = dims.z;
  if (p >= T || k >= d) { return; }
  var acc: f32 = 0.0;
  var i: u32 = p;
  loop {
    if (i >= N) { break; }
    acc = acc + dEmb[i * d + k];
    i = i + T;
  }
  dWpe[p * d + k] = acc;
}
