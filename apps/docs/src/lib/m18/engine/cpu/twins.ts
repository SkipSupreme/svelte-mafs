// CPU twins for the 10 forward kernels. Each twin is the math reference the
// WGSL kernel must match to ~1e-5 on random inputs (Slice 2 oracle).
//
// Conventions: every tensor is a Float32Array (row-major). Shapes are tuples
// passed alongside. No allocations beyond the output. f32 only.

export type F32 = Float32Array;

// 1. embeddingGather: tokIds [B*T] + posIds [T] + wte [V,d] + wpe [T,d] → out [B*T,d]
export function embeddingGather(
  tokIds: Int32Array, T: number, wte: F32, wpe: F32, V: number, d: number,
): F32 {
  const N = tokIds.length;
  const out = new Float32Array(N * d);
  for (let i = 0; i < N; i++) {
    const tok = tokIds[i]; const pos = i % T;
    for (let k = 0; k < d; k++) out[i * d + k] = wte[tok * d + k] + wpe[pos * d + k];
  }
  return out;
}

// 2. layerNorm: per row (zero-mean, unit-var) → gamma * normalized + beta
export function layerNorm(x: F32, gamma: F32, beta: F32, rows: number, d: number, eps = 1e-5): F32 {
  const out = new Float32Array(rows * d);
  for (let r = 0; r < rows; r++) {
    let mu = 0; for (let k = 0; k < d; k++) mu += x[r * d + k]; mu /= d;
    let v = 0; for (let k = 0; k < d; k++) { const z = x[r * d + k] - mu; v += z * z; } v /= d;
    const inv = 1 / Math.sqrt(v + eps);
    for (let k = 0; k < d; k++) out[r * d + k] = gamma[k] * (x[r * d + k] - mu) * inv + beta[k];
  }
  return out;
}

// 3. matmul: A [M,K] × B [K,N] → C [M,N]. Used for QKV, attnOut, FFN1, FFN2, unembed.
export function matmul(A: F32, B: F32, M: number, K: number, N: number): F32 {
  const C = new Float32Array(M * N);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < K; k++) {
      const a = A[m * K + k]; if (a === 0) continue;
      for (let n = 0; n < N; n++) C[m * N + n] += a * B[k * N + n];
    }
  }
  return C;
}

// 4. causalSdpa: qkv [B*T, 3d] → out [B*T, d]. nHead heads; dHead = d/nHead.
// Splits qkv along feature dim as [Q | K | V], heads as inner-strided.
export function causalSdpa(
  qkv: F32, B: number, T: number, d: number, nHead: number,
): F32 {
  const dHead = d / nHead; const scale = 1 / Math.sqrt(dHead);
  const out = new Float32Array(B * T * d);
  // qkv layout: [B*T, 3*d]; within row: q[0..d), k[d..2d), v[2d..3d)
  // head h occupies feature range [h*dHead, (h+1)*dHead) inside each of q/k/v.
  const scratch = new Float32Array(T); // attn scores per query position
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < nHead; h++) {
      for (let tq = 0; tq < T; tq++) {
        // scores[t] = q · k[t]   for t ≤ tq, else -inf
        let mx = -Infinity;
        for (let tk = 0; tk <= tq; tk++) {
          let s = 0;
          for (let c = 0; c < dHead; c++) {
            const qi = (b * T + tq) * 3 * d + 0 * d + h * dHead + c;
            const ki = (b * T + tk) * 3 * d + 1 * d + h * dHead + c;
            s += qkv[qi] * qkv[ki];
          }
          s *= scale; scratch[tk] = s; if (s > mx) mx = s;
        }
        // softmax over [0..tq]
        let sum = 0;
        for (let tk = 0; tk <= tq; tk++) { scratch[tk] = Math.exp(scratch[tk] - mx); sum += scratch[tk]; }
        for (let tk = 0; tk <= tq; tk++) scratch[tk] /= sum;
        // out[tq, head] = sum_t scratch[t] * v[t, head]
        for (let c = 0; c < dHead; c++) {
          let acc = 0;
          for (let tk = 0; tk <= tq; tk++) {
            const vi = (b * T + tk) * 3 * d + 2 * d + h * dHead + c;
            acc += scratch[tk] * qkv[vi];
          }
          out[(b * T + tq) * d + h * dHead + c] = acc;
        }
      }
    }
  }
  return out;
}

// 5. residualAdd: c = a + b elementwise.
export function residualAdd(a: F32, b: F32): F32 {
  const N = a.length; const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = a[i] + b[i];
  return out;
}

// 6. gelu: tanh approximation. Matches nanoGPT(approximate='tanh') and torch.nn.GELU(approximate='tanh').
const GELU_C = Math.sqrt(2 / Math.PI);
export function gelu(x: F32): F32 {
  const N = x.length; const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const v = x[i]; const u = GELU_C * (v + 0.044715 * v * v * v);
    out[i] = 0.5 * v * (1 + Math.tanh(u));
  }
  return out;
}

// 7. softmax over last dim of [rows, cols]. Used by sampler / dev-page assertion.
export function softmax(x: F32, rows: number, cols: number): F32 {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let mx = -Infinity;
    for (let c = 0; c < cols; c++) { const v = x[r * cols + c]; if (v > mx) mx = v; }
    let sum = 0;
    for (let c = 0; c < cols; c++) { const e = Math.exp(x[r * cols + c] - mx); out[r * cols + c] = e; sum += e; }
    const inv = 1 / sum;
    for (let c = 0; c < cols; c++) out[r * cols + c] *= inv;
  }
  return out;
}

// Unembedding is tied with the token-embedding matrix wte. The forward op is
// logits = x · wteᵀ. We expose a helper that does exactly that.
export function unembedding(x: F32, wte: F32, rows: number, d: number, V: number): F32 {
  // x: [rows, d]; wte: [V, d]. We need x · wteᵀ → [rows, V] which is
  // out[r, v] = sum_k x[r, k] * wte[v, k].
  const out = new Float32Array(rows * V);
  for (let r = 0; r < rows; r++) {
    for (let v = 0; v < V; v++) {
      let s = 0; for (let k = 0; k < d; k++) s += x[r * d + k] * wte[v * d + k];
      out[r * V + v] = s;
    }
  }
  return out;
}

// ── Backward twins (Slice 3) ──────────────────────────────────────────────────
//
// Each takes the same inputs the WGSL kernel does — forward inputs (cached
// activations) plus the upstream gradient — and returns the local gradients.
// All in pure TS so a vitest run can pin the math without WebGPU.

// 1. matmul backward, dA branch.   dA[m,k] = Σ_n dC[m,n] · B[k,n]
// Equivalent to dA = dC · Bᵀ, mirrors the matmulRhsT WGSL kernel.
export function matmulBwdA(dC: F32, B: F32, M: number, K: number, N: number): F32 {
  const dA = new Float32Array(M * K);
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      const dc = dC[m * N + n]; if (dc === 0) continue;
      for (let k = 0; k < K; k++) dA[m * K + k] += dc * B[k * N + n];
    }
  }
  return dA;
}

// 2. matmul backward, dB branch.   dB[k,n] = Σ_m A[m,k] · dC[m,n]
// Equivalent to dB = Aᵀ · dC, mirrors the new matmulLhsT WGSL kernel.
export function matmulBwdB(A: F32, dC: F32, M: number, K: number, N: number): F32 {
  const dB = new Float32Array(K * N);
  for (let m = 0; m < M; m++) {
    for (let k = 0; k < K; k++) {
      const a = A[m * K + k]; if (a === 0) continue;
      for (let n = 0; n < N; n++) dB[k * N + n] += a * dC[m * N + n];
    }
  }
  return dB;
}

// 3. softmax + cross-entropy backward.
//    Given logits [rows, V] and target indices [rows], compute
//    dlogits[i, v] = (softmax(logits[i])[v] − 1[v == targets[i]]) / rows.
// `rows` is the batch dimension B*T; the 1/rows averaging matches PyTorch's
// F.cross_entropy(reduction='mean') over the flattened (B*T, V) logits.
export function softmaxCrossEntropyBwd(
  logits: F32, targets: Int32Array, rows: number, V: number,
): F32 {
  const dlogits = new Float32Array(rows * V);
  const inv = 1 / rows;
  for (let r = 0; r < rows; r++) {
    let mx = -Infinity;
    for (let v = 0; v < V; v++) { const z = logits[r * V + v]; if (z > mx) mx = z; }
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(logits[r * V + v] - mx);
    const t = targets[r];
    for (let v = 0; v < V; v++) {
      const p = Math.exp(logits[r * V + v] - mx) / sum;
      dlogits[r * V + v] = (p - (v === t ? 1 : 0)) * inv;
    }
  }
  return dlogits;
}

// 3b. cross-entropy loss itself, used by tests and by the train driver to read
//     the eval metric back to JS without re-running softmax twice.
export function crossEntropyLoss(
  logits: F32, targets: Int32Array, rows: number, V: number,
): number {
  let sum = 0;
  for (let r = 0; r < rows; r++) {
    let mx = -Infinity;
    for (let v = 0; v < V; v++) { const z = logits[r * V + v]; if (z > mx) mx = z; }
    let lse = 0;
    for (let v = 0; v < V; v++) lse += Math.exp(logits[r * V + v] - mx);
    const logZ = mx + Math.log(lse);
    sum += logZ - logits[r * V + targets[r]];
  }
  return sum / rows;
}

// 4. gelu backward (tanh approximation).
//    y = 0.5 x (1 + tanh(u)),  u = c (x + a x³),  c = √(2/π), a = 0.044715
//    dy/dx = 0.5 (1 + tanh u) + 0.5 x · sech²(u) · du/dx
//    where du/dx = c (1 + 3 a x²) and sech² = 1 − tanh².
export function geluBwd(dy: F32, x: F32): F32 {
  const N = x.length; const out = new Float32Array(N);
  const a = 0.044715;
  for (let i = 0; i < N; i++) {
    const v = x[i];
    const u = GELU_C * (v + a * v * v * v);
    const th = Math.tanh(u);
    const dudx = GELU_C * (1 + 3 * a * v * v);
    const dydx = 0.5 * (1 + th) + 0.5 * v * (1 - th * th) * dudx;
    out[i] = dy[i] * dydx;
  }
  return out;
}

// 5. residualAdd backward — gradient flows identically to both inputs. We
//    expose this as an explicit twin even though the WGSL path collapses it
//    into the consumer's read.
export function residualAddBwd(dy: F32): { da: F32; db: F32 } {
  return { da: dy.slice(), db: dy.slice() };
}

// 6. layerNorm backward — the 3-term in-shader form (research §5).
//    Given x, gamma, dy, returns dx, dGamma, dBeta. Reductions are per row
//    for dx, summed across rows for dGamma/dBeta.
export function layerNormBwd(
  dy: F32, x: F32, gamma: F32, rows: number, d: number, eps = 1e-5,
): { dx: F32; dGamma: F32; dBeta: F32 } {
  const dx = new Float32Array(rows * d);
  const dGamma = new Float32Array(d);
  const dBeta = new Float32Array(d);

  for (let r = 0; r < rows; r++) {
    let mu = 0; for (let k = 0; k < d; k++) mu += x[r * d + k]; mu /= d;
    let varv = 0;
    for (let k = 0; k < d; k++) { const z = x[r * d + k] - mu; varv += z * z; }
    varv /= d;
    const inv = 1 / Math.sqrt(varv + eps);

    // Per-row reductions for the dx 3-term form. g_k = γ_k · dy_k.
    let sumG = 0; let sumGz = 0;
    for (let k = 0; k < d; k++) {
      const z = (x[r * d + k] - mu) * inv;
      const g = gamma[k] * dy[r * d + k];
      sumG += g;
      sumGz += g * z;
    }
    for (let k = 0; k < d; k++) {
      const z = (x[r * d + k] - mu) * inv;
      const g = gamma[k] * dy[r * d + k];
      dx[r * d + k] = inv * (g - sumG / d - z * sumGz / d);
      dGamma[k] += dy[r * d + k] * z;
      dBeta[k]  += dy[r * d + k];
    }
  }
  return { dx, dGamma, dBeta };
}

// 7. causal SDPA backward.
//    Inputs:  qkv [B*T, 3d], dout [B*T, d].
//    Outputs: dqkv [B*T, 3d] in the same [Q | K | V] interleaved layout.
//    We recompute the attention probabilities on the fly to avoid caching the
//    per-head [T, T] matrix; this matches the WGSL kernel's strategy.
export function causalSdpaBwd(
  qkv: F32, dout: F32, B: number, T: number, d: number, nHead: number,
): F32 {
  const dHead = d / nHead;
  const scale = 1 / Math.sqrt(dHead);
  const dqkv = new Float32Array(B * T * 3 * d);

  const P = new Float32Array(T);    // softmax row for one (b, h, tq)
  const dP = new Float32Array(T);   // upstream into softmax
  const dS = new Float32Array(T);   // grad w.r.t. pre-softmax scores

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < nHead; h++) {
      for (let tq = 0; tq < T; tq++) {
        // Recompute softmax row for this (b, h, tq).
        let mx = -Infinity;
        for (let tk = 0; tk <= tq; tk++) {
          let s = 0;
          for (let c = 0; c < dHead; c++) {
            const qi = (b * T + tq) * 3 * d + 0 * d + h * dHead + c;
            const ki = (b * T + tk) * 3 * d + 1 * d + h * dHead + c;
            s += qkv[qi] * qkv[ki];
          }
          s *= scale; P[tk] = s; if (s > mx) mx = s;
        }
        let sum = 0;
        for (let tk = 0; tk <= tq; tk++) { P[tk] = Math.exp(P[tk] - mx); sum += P[tk]; }
        for (let tk = 0; tk <= tq; tk++) P[tk] /= sum;
        for (let tk = tq + 1; tk < T; tk++) P[tk] = 0;

        // dP[tk] = Σ_c dout[b, tq, h*dHead + c] · V[b, tk, h*dHead + c]
        // dV[b, tk, h*dHead + c] += P[tk] · dout[b, tq, h*dHead + c]
        for (let tk = 0; tk <= tq; tk++) dP[tk] = 0;
        for (let c = 0; c < dHead; c++) {
          const doIdx = (b * T + tq) * d + h * dHead + c;
          const dy = dout[doIdx];
          for (let tk = 0; tk <= tq; tk++) {
            const vi = (b * T + tk) * 3 * d + 2 * d + h * dHead + c;
            dP[tk] += dy * qkv[vi];
            dqkv[vi] += P[tk] * dy;
          }
        }

        // Softmax Jacobian: dS = P ⊙ (dP - Σ P·dP)
        let dot = 0;
        for (let tk = 0; tk <= tq; tk++) dot += P[tk] * dP[tk];
        for (let tk = 0; tk <= tq; tk++) dS[tk] = P[tk] * (dP[tk] - dot);

        // dQ[b, tq, h, c] += scale · Σ_tk dS[tk] · K[b, tk, h, c]
        // dK[b, tk, h, c] += scale · dS[tk] · Q[b, tq, h, c]
        for (let c = 0; c < dHead; c++) {
          const qi = (b * T + tq) * 3 * d + 0 * d + h * dHead + c;
          let dq = 0;
          for (let tk = 0; tk <= tq; tk++) {
            const ki = (b * T + tk) * 3 * d + 1 * d + h * dHead + c;
            dq += dS[tk] * qkv[ki];
            dqkv[ki] += scale * dS[tk] * qkv[qi];
          }
          dqkv[qi] += scale * dq;
        }
      }
    }
  }
  return dqkv;
}

// 8. embedding backward — scatter-add the per-token gradient back into wte and
//    wpe. Caller is responsible for ADDING to dWte if it has prior content
//    from the unembedding backward (tied weights).
export function embeddingBwd(
  dEmb: F32, tokIds: Int32Array, T: number, V: number, d: number,
): { dWte: F32; dWpe: F32 } {
  const N = tokIds.length;
  const dWte = new Float32Array(V * d);
  const dWpe = new Float32Array(T * d);
  for (let i = 0; i < N; i++) {
    const tok = tokIds[i]; const pos = i % T;
    for (let k = 0; k < d; k++) {
      const g = dEmb[i * d + k];
      dWte[tok * d + k] += g;
      dWpe[pos * d + k] += g;
    }
  }
  return { dWte, dWpe };
}

// 9. gradient clipping (for tests / driver). Computes the global L2 norm over
//    a list of grad arrays, returns scale = min(1, maxNorm/norm). Caller then
//    scales each grad in place.
export function gradClipScale(grads: F32[], maxNorm: number): number {
  let sq = 0;
  for (const g of grads) for (let i = 0; i < g.length; i++) sq += g[i] * g[i];
  const norm = Math.sqrt(sq);
  if (norm <= maxNorm) return 1;
  return maxNorm / norm;
}

// 10. AdamW one-step (PyTorch ordering: decoupled decay → m/v update →
//     bias-corrected step). Mutates theta, m, v in place. Returns nothing.
export function adamwStep(
  theta: F32, grad: F32, m: F32, v: F32,
  lr: number, beta1: number, beta2: number, eps: number, lambda: number, t: number,
): void {
  const bc1 = 1 - Math.pow(beta1, t);
  const bc2 = 1 - Math.pow(beta2, t);
  const stepSize = lr / bc1;
  const bc2Sqrt = Math.sqrt(bc2);
  const decay = 1 - lr * lambda;
  for (let i = 0; i < theta.length; i++) {
    theta[i] *= decay;
    const g = grad[i];
    m[i] = beta1 * m[i] + (1 - beta1) * g;
    v[i] = beta2 * v[i] + (1 - beta2) * g * g;
    const denom = Math.sqrt(v[i]) / bc2Sqrt + eps;
    theta[i] -= stepSize * m[i] / denom;
  }
}
