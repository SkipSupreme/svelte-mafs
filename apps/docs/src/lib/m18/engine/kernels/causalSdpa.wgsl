// Causal scaled-dot-product attention. One workgroup per (b, head, query-position).
// Layout assumptions: qkv is [B*T, 3*d] row-major; within each row the slab
// order is [Q | K | V], each slab of length d. Heads occupy contiguous strips
// of length dHead inside Q, K, V. Workgroup size = 64; this implementation
// supports T ≤ 64 and dHead ≤ 64 (our locked config: T=64, dHead=16).

const WG: u32 = 64u;
// Large-magnitude negative used as a softmax mask. Avoids the literal
// -3.4028235e38 which the WGSL parser rounds to slightly more than f32::MIN
// and rejects on Chromium 124+. -1e30 is plenty for exp(-1e30) → 0.
const NEG_INF: f32 = -1e30;

@group(0) @binding(0) var<storage, read>          qkv:  array<f32>; // [B*T, 3*d]
@group(0) @binding(1) var<storage, read_write>    out:  array<f32>; // [B*T, d]
@group(0) @binding(2) var<uniform>                cfg:  vec4<u32>;  // (B, T, d, nHead)

var<workgroup> scores: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let B = cfg.x; let T = cfg.y; let d = cfg.z; let nHead = cfg.w;
  let tq = wid.x; let h = wid.y; let b = wid.z;
  if (b >= B || h >= nHead || tq >= T) { return; }

  let dHead = d / nHead;
  let scale = 1.0 / sqrt(f32(dHead));
  let i = lid.x;

  // 1. Compute one score per thread (one t' = i). Mask t' > tq to -∞.
  if (i < T) {
    if (i <= tq) {
      var s: f32 = 0.0;
      for (var c: u32 = 0u; c < dHead; c = c + 1u) {
        let qi = (b * T + tq) * 3u * d + 0u * d + h * dHead + c;
        let ki = (b * T + i ) * 3u * d + 1u * d + h * dHead + c;
        s = s + qkv[qi] * qkv[ki];
      }
      scores[i] = s * scale;
    } else {
      scores[i] = NEG_INF;
    }
  }
  workgroupBarrier();

  // 2. Softmax along T. Single-thread reduction (T ≤ 64).
  if (i == 0u) {
    var mx: f32 = scores[0];
    for (var t: u32 = 1u; t < T; t = t + 1u) { if (scores[t] > mx) { mx = scores[t]; } }
    var sum: f32 = 0.0;
    for (var t: u32 = 0u; t < T; t = t + 1u) {
      scores[t] = exp(scores[t] - mx);
      sum = sum + scores[t];
    }
    let inv = 1.0 / sum;
    for (var t: u32 = 0u; t < T; t = t + 1u) { scores[t] = scores[t] * inv; }
  }
  workgroupBarrier();

  // 3. Output: thread i (i < dHead) computes one channel of this head.
  if (i < dHead) {
    var acc: f32 = 0.0;
    for (var tk: u32 = 0u; tk < T; tk = tk + 1u) {
      let vi = (b * T + tk) * 3u * d + 2u * d + h * dHead + i;
      acc = acc + scores[tk] * qkv[vi];
    }
    out[(b * T + tq) * d + h * dHead + i] = acc;
  }
}
