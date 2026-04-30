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
