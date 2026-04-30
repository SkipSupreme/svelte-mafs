// Causal scaled-dot-product attention backward.
//   Inputs : qkv  [B*T, 3*d] (forward inputs, read)
//            dout [B*T, d]   (upstream gradient, read)
//   Output : dqkv [B*T, 3*d] (write — dQ, dK, dV interleaved as in forward)
//
// One workgroup per (b, head). Loops over tq sequentially inside the
// workgroup so all dK[tk] and dV[tk] writes for a given (b, head) are
// serialized and race-free without atomics. Workgroup size = 64 covers
// T ≤ 64 (one thread per tk for the softmax / dV / dK paths) and
// dHead ≤ 64 (one thread per channel for the dQ reduction).
//
// Recomputes attention probabilities on the fly to skip the [B,H,T,T]
// activation cache. Cost: one extra QKᵀ pass per backward step.

const WG: u32 = 64u;
// Same softmax-mask trick as the forward kernel — see causalSdpa.wgsl.
const NEG_INF: f32 = -1e30;

@group(0) @binding(0) var<storage, read>          qkv:  array<f32>;
@group(0) @binding(1) var<storage, read>          dout: array<f32>;
@group(0) @binding(2) var<storage, read_write>    dqkv: array<f32>;
@group(0) @binding(3) var<uniform>                cfg:  vec4<u32>; // (B, T, d, nHead)

var<workgroup> P:   array<f32, WG>;
var<workgroup> dP:  array<f32, WG>;
var<workgroup> dS:  array<f32, WG>;
var<workgroup> red: array<f32, WG>;

fn wgMax(local: u32, val: f32) -> f32 {
  red[local] = val;
  workgroupBarrier();
  var step: u32 = WG / 2u;
  loop {
    if (step == 0u) { break; }
    if (local < step) {
      let other = red[local + step];
      if (other > red[local]) { red[local] = other; }
    }
    workgroupBarrier();
    step = step / 2u;
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

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
  let r = red[0];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let B = cfg.x; let T = cfg.y; let d = cfg.z; let nHead = cfg.w;
  let h = wid.x; let b = wid.y;
  if (b >= B || h >= nHead) { return; }

  let dHead = d / nHead;
  let scale = 1.0 / sqrt(f32(dHead));
  let i = lid.x;

  for (var tq: u32 = 0u; tq < T; tq = tq + 1u) {
    // ── softmax row ─────────────────────────────────────────────────────────
    var sCur: f32 = NEG_INF;
    if (i < T) {
      if (i <= tq) {
        var s: f32 = 0.0;
        for (var c: u32 = 0u; c < dHead; c = c + 1u) {
          let qi = (b * T + tq) * 3u * d + 0u * d + h * dHead + c;
          let ki = (b * T + i ) * 3u * d + 1u * d + h * dHead + c;
          s = s + qkv[qi] * qkv[ki];
        }
        sCur = s * scale;
      }
    }
    let mx  = wgMax(i, sCur);
    let e   = select(0.0, exp(sCur - mx), i < T && i <= tq);
    let sum = wgSum(i, e);
    if (i < T) { P[i] = e / sum; }
    workgroupBarrier();

    // ── dP[tk] = Σ_c dout[tq, h, c] · V[tk, h, c]  for tk ≤ tq ─────────────
    if (i < T) {
      if (i <= tq) {
        var dp: f32 = 0.0;
        for (var c: u32 = 0u; c < dHead; c = c + 1u) {
          let doIdx = (b * T + tq) * d + h * dHead + c;
          let vi    = (b * T + i ) * 3u * d + 2u * d + h * dHead + c;
          dp = dp + dout[doIdx] * qkv[vi];
        }
        dP[i] = dp;
      } else {
        dP[i] = 0.0;
      }
    }
    workgroupBarrier();

    // ── dV[tk] += P[tk] · dout[tq] ────────────────────────────────────────
    if (i < T && i <= tq) {
      for (var c: u32 = 0u; c < dHead; c = c + 1u) {
        let vi = (b * T + i) * 3u * d + 2u * d + h * dHead + c;
        let doIdx = (b * T + tq) * d + h * dHead + c;
        dqkv[vi] = dqkv[vi] + P[i] * dout[doIdx];
      }
    }
    workgroupBarrier();

    // ── softmax Jacobian: dS = P ⊙ (dP − Σ P·dP) ──────────────────────────
    let dotPart = select(0.0, P[i] * dP[i], i < T && i <= tq);
    let dotSum  = wgSum(i, dotPart);
    if (i < T) {
      if (i <= tq) { dS[i] = P[i] * (dP[i] - dotSum); }
      else { dS[i] = 0.0; }
    }
    workgroupBarrier();

    // ── dK[tk] += scale · dS[tk] · Q[tq] ──────────────────────────────────
    if (i < T && i <= tq) {
      for (var c: u32 = 0u; c < dHead; c = c + 1u) {
        let qi = (b * T + tq) * 3u * d + 0u * d + h * dHead + c;
        let ki = (b * T + i ) * 3u * d + 1u * d + h * dHead + c;
        dqkv[ki] = dqkv[ki] + scale * dS[i] * qkv[qi];
      }
    }
    workgroupBarrier();

    // ── dQ[tq, c] += scale · Σ_tk dS[tk] · K[tk, c] ───────────────────────
    if (i < dHead) {
      var dq: f32 = 0.0;
      for (var tk: u32 = 0u; tk <= tq; tk = tk + 1u) {
        let ki = (b * T + tk) * 3u * d + 1u * d + h * dHead + i;
        dq = dq + dS[tk] * qkv[ki];
      }
      let qi = (b * T + tq) * 3u * d + 0u * d + h * dHead + i;
      dqkv[qi] = dqkv[qi] + scale * dq;
    }
    workgroupBarrier();
  }
}
