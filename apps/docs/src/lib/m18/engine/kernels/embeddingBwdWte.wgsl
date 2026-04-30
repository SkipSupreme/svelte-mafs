// Embedding backward — wte branch.
//   Inputs : dEmb [N, d] (upstream gradient on the embedded vector), tokIds [N]
//   Output : dWte [V, d] (read-modify-write: ADDS to existing content so the
//                         tied-unembedding gradient already in dWte is preserved)
//
// One workgroup per vocab entry v ∈ [0, V); WG size = d, one thread per channel.
// Each thread sums dEmb[i, k] over all i with tokIds[i] == v. Avoids float
// atomicAdd by partitioning the output rows across workgroups.

const WG: u32 = 64u;

@group(0) @binding(0) var<storage, read>          dEmb:   array<f32>;
@group(0) @binding(1) var<storage, read>          tokIds: array<i32>;
@group(0) @binding(2) var<storage, read_write>    dWte:   array<f32>;
@group(0) @binding(3) var<uniform>                dims:   vec4<u32>; // (N, V, d, _)

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let v = wid.x; let k = lid.x;
  let N = dims.x; let V = dims.y; let d = dims.z;
  if (v >= V || k >= d) { return; }
  var acc: f32 = 0.0;
  for (var i: u32 = 0u; i < N; i = i + 1u) {
    if (u32(tokIds[i]) == v) { acc = acc + dEmb[i * d + k]; }
  }
  // Add (don't assign): dWte may already hold the unembedding-backward
  // contribution for this row. Tied weights → both gradients accumulate.
  dWte[v * d + k] = dWte[v * d + k] + acc;
}
