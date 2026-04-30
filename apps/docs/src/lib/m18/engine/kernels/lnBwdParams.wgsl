// LayerNorm backward — dGamma / dBeta branch.
//   Inputs : dy [rows, d], x [rows, d]
//   Outputs: dGamma [d] (write), dBeta [d] (write)
// Each thread handles one column k, summing over all rows. Dispatch ceil(d/WG)
// workgroups; with d=64 a single workgroup is enough.

const WG: u32 = 64u;
const EPS: f32 = 1e-5;

@group(0) @binding(0) var<storage, read>          dy:     array<f32>;
@group(0) @binding(1) var<storage, read>          x:      array<f32>;
@group(0) @binding(2) var<storage, read_write>    dGamma: array<f32>;
@group(0) @binding(3) var<storage, read_write>    dBeta:  array<f32>;
@group(0) @binding(4) var<uniform>                dims:   vec4<u32>; // (rows, d, _, _)

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  let rows = dims.x; let d = dims.y;
  if (k >= d) { return; }

  var ag: f32 = 0.0;
  var ab: f32 = 0.0;
  for (var r: u32 = 0u; r < rows; r = r + 1u) {
    // Recompute z = (x − μ)/σ on the fly. This duplicates work with lnBwd
    // (which also computes μ and σ⁻¹ per row), but keeps this kernel
    // self-contained and avoids allocating a per-row μ/σ scratch buffer.
    var mu: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) { mu = mu + x[r * d + j]; }
    mu = mu / f32(d);
    var varv: f32 = 0.0;
    for (var j: u32 = 0u; j < d; j = j + 1u) {
      let z = x[r * d + j] - mu;
      varv = varv + z * z;
    }
    varv = varv / f32(d);
    let inv = 1.0 / sqrt(varv + EPS);

    let gy = dy[r * d + k];
    let zk = (x[r * d + k] - mu) * inv;
    ag = ag + gy * zk;
    ab = ab + gy;
  }
  dGamma[k] = ag;
  dBeta[k]  = ab;
}
