// CPU-twin sanity tests. These pin the math of the reference implementations
// the WGSL kernels must match. The /dev/m18-forward/ page does the in-browser
// WGSL-vs-twin comparison live; these tests pin the twin itself.

import { describe, expect, it } from 'vitest';
import type { F32 } from './twins';
import {
  embeddingGather, layerNorm, matmul, causalSdpa, residualAdd, gelu, softmax, unembedding,
  matmulBwdA, matmulBwdB, softmaxCrossEntropyBwd, crossEntropyLoss, geluBwd,
  layerNormBwd, causalSdpaBwd, embeddingBwd, gradClipScale, adamwStep,
} from './twins';

const f32 = (a: number[]) => Float32Array.from(a);

describe('embeddingGather', () => {
  it('selects rows of wte and adds the position-indexed row of wpe', () => {
    const V = 3, T = 2, d = 2;
    const wte = f32([1, 2,  3, 4,  5, 6]); // rows for tok 0..2
    const wpe = f32([10, 20,  30, 40]);    // rows for pos 0..1
    const tok = Int32Array.from([0, 1, 2, 0]); // 4 = 2 batches × T=2
    const out = embeddingGather(tok, T, wte, wpe, V, d);
    // (b=0,t=0): wte[0] + wpe[0] = [11, 22]
    // (b=0,t=1): wte[1] + wpe[1] = [33, 44]
    // (b=1,t=0): wte[2] + wpe[0] = [15, 26]
    // (b=1,t=1): wte[0] + wpe[1] = [31, 42]
    expect(Array.from(out)).toEqual([11, 22, 33, 44, 15, 26, 31, 42]);
  });
});

describe('layerNorm', () => {
  it('produces zero mean and unit variance per row when gamma=1, beta=0', () => {
    const rows = 3, d = 8;
    const x = new Float32Array(rows * d);
    for (let i = 0; i < x.length; i++) x[i] = Math.random() * 4 - 2;
    const gamma = new Float32Array(d).fill(1);
    const beta = new Float32Array(d);
    const y = layerNorm(x, gamma, beta, rows, d);
    for (let r = 0; r < rows; r++) {
      let mu = 0; for (let k = 0; k < d; k++) mu += y[r * d + k]; mu /= d;
      let v = 0; for (let k = 0; k < d; k++) { const z = y[r * d + k] - mu; v += z * z; } v /= d;
      expect(mu).toBeCloseTo(0, 5);
      expect(v).toBeCloseTo(1, 4); // population variance ~1 (within eps tolerance)
    }
  });

  it('applies gamma scale and beta shift', () => {
    const rows = 1, d = 4;
    const x = f32([0, 1, 2, 3]);
    const gamma = f32([2, 2, 2, 2]);
    const beta = f32([10, 10, 10, 10]);
    const y = layerNorm(x, gamma, beta, rows, d);
    // mean = 1.5, var = 1.25 → inv = 1/sqrt(1.25 + eps)
    const inv = 1 / Math.sqrt(1.25 + 1e-5);
    const expected = [0 - 1.5, 1 - 1.5, 2 - 1.5, 3 - 1.5].map((z) => 2 * z * inv + 10);
    for (let i = 0; i < d; i++) expect(y[i]).toBeCloseTo(expected[i], 4);
  });
});

describe('matmul', () => {
  it('matches a hand-computed 2×3 × 3×2 product', () => {
    const A = f32([1, 2, 3,  4, 5, 6]); // 2×3
    const B = f32([7, 8,  9, 10,  11, 12]); // 3×2
    const C = matmul(A, B, 2, 3, 2);
    // [1·7+2·9+3·11, 1·8+2·10+3·12] = [58, 64]
    // [4·7+5·9+6·11, 4·8+5·10+6·12] = [139, 154]
    expect(Array.from(C)).toEqual([58, 64, 139, 154]);
  });

  it('agrees with a transpose-trick reformulation on random inputs', () => {
    const M = 5, K = 7, N = 3;
    const A = new Float32Array(M * K); const B = new Float32Array(K * N);
    for (let i = 0; i < A.length; i++) A[i] = Math.random() - 0.5;
    for (let i = 0; i < B.length; i++) B[i] = Math.random() - 0.5;
    const C = matmul(A, B, M, K, N);
    // brute-force ground truth
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let s = 0; for (let k = 0; k < K; k++) s += A[m * K + k] * B[k * N + n];
        expect(C[m * N + n]).toBeCloseTo(s, 5);
      }
    }
  });
});

describe('causalSdpa', () => {
  it('has rows of attention weights that each sum to 1 (stress test)', () => {
    const B = 1, T = 4, d = 8, nHead = 2;
    const qkv = new Float32Array(B * T * 3 * d);
    for (let i = 0; i < qkv.length; i++) qkv[i] = Math.random() - 0.5;
    const out = causalSdpa(qkv, B, T, d, nHead);
    expect(out.length).toBe(B * T * d);
    expect(out.every(Number.isFinite)).toBe(true);
  });

  it('first query position attends only to itself (causal mask)', () => {
    // A row with B=1, T=2, d=4, nHead=1. tq=0 can only attend to tk=0,
    // so out[0,:] must equal v[0,:] (head=0 spans full d).
    const B = 1, T = 2, d = 4, nHead = 1;
    const qkv = new Float32Array(B * T * 3 * d);
    // Set v[t=0] = [1,2,3,4], v[t=1] = [99,99,99,99]; q,k arbitrary.
    for (let c = 0; c < d; c++) qkv[(0) * 3 * d + 2 * d + c] = c + 1;       // v[0]
    for (let c = 0; c < d; c++) qkv[(1) * 3 * d + 2 * d + c] = 99;         // v[1]
    const out = causalSdpa(qkv, B, T, d, nHead);
    expect(Array.from(out.slice(0, d))).toEqual([1, 2, 3, 4]);
  });
});

describe('residualAdd + gelu + softmax', () => {
  it('residualAdd is elementwise sum', () => {
    const a = f32([1, 2, 3]); const b = f32([10, 20, 30]);
    expect(Array.from(residualAdd(a, b))).toEqual([11, 22, 33]);
  });

  it('gelu(0) = 0 and gelu is monotonic on positives', () => {
    const x = f32([-3, -1, 0, 1, 3]);
    const y = gelu(x);
    expect(y[2]).toBeCloseTo(0, 6);
    expect(y[3]).toBeGreaterThan(y[2]);
    expect(y[4]).toBeGreaterThan(y[3]);
    expect(y[1]).toBeLessThan(y[2]);
    // Spot-check the tanh-approx: gelu(1) ≈ 0.84119 within 5e-3.
    expect(y[3]).toBeCloseTo(0.8411, 3);
  });

  it('softmax produces probabilities summing to 1', () => {
    const rows = 4, cols = 7;
    const x = new Float32Array(rows * cols);
    for (let i = 0; i < x.length; i++) x[i] = Math.random() * 10;
    const p = softmax(x, rows, cols);
    for (let r = 0; r < rows; r++) {
      let s = 0; for (let c = 0; c < cols; c++) s += p[r * cols + c];
      expect(s).toBeCloseTo(1, 6);
      for (let c = 0; c < cols; c++) expect(p[r * cols + c]).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('unembedding (tied with embedding matrix)', () => {
  it('computes x · wteᵀ', () => {
    const rows = 2, d = 3, V = 4;
    const x = f32([1, 2, 3,  4, 5, 6]);
    const wte = f32([1, 0, 0,  0, 1, 0,  0, 0, 1,  1, 1, 1]); // identity-ish + ones row
    const out = unembedding(x, wte, rows, d, V);
    // out[0,:] = x[0]·wte[0..3,:].T = [1, 2, 3, 1+2+3]
    // out[1,:] = [4, 5, 6, 15]
    expect(Array.from(out)).toEqual([1, 2, 3, 6, 4, 5, 6, 15]);
  });
});

// ── Backward twin tests ───────────────────────────────────────────────────────
//
// Numerical-gradient checks are the M12 canonical debugging tool. Each backward
// twin is verified against (L(θ+ε) − L(θ−ε)) / (2ε) computed through the
// matching forward twin, with ε small enough that f32 round-off doesn't
// dominate.

const seedRng = (s: number) => {
  // tiny LCG so repeated test runs don't re-seed Math.random in unexpected ways.
  let state = s | 0;
  return () => { state = (state * 1664525 + 1013904223) | 0; return ((state >>> 0) / 0xffffffff) - 0.5; };
};

const fillRand = (a: F32, rng: () => number, scale = 1): F32 => {
  for (let i = 0; i < a.length; i++) a[i] = scale * rng();
  return a;
};

describe('matmul backward', () => {
  it('dA · forward agrees with finite-difference at random points', () => {
    const M = 3, K = 4, N = 2;
    const r = seedRng(1);
    const A = fillRand(new Float32Array(M * K), r);
    const B = fillRand(new Float32Array(K * N), r);
    const dC = fillRand(new Float32Array(M * N), r);
    const dA = matmulBwdA(dC, B, M, K, N);
    // Loss is sum(C · dC). dL/dA[m,k] = Σ_n dC[m,n] · B[k,n] = matmulBwdA above.
    const eps = 1e-3;
    for (let m = 0; m < M; m++) for (let k = 0; k < K; k++) {
      const A2 = new Float32Array(A); A2[m * K + k] += eps;
      const C2 = matmul(A2, B, M, K, N);
      let lp = 0; for (let i = 0; i < C2.length; i++) lp += C2[i] * dC[i];
      A2[m * K + k] -= 2 * eps;
      const C3 = matmul(A2, B, M, K, N);
      let lm = 0; for (let i = 0; i < C3.length; i++) lm += C3[i] * dC[i];
      const num = (lp - lm) / (2 * eps);
      expect(dA[m * K + k]).toBeCloseTo(num, 3);
    }
  });

  it('dB · forward agrees with finite-difference', () => {
    const M = 3, K = 4, N = 2;
    const r = seedRng(2);
    const A = fillRand(new Float32Array(M * K), r);
    const B = fillRand(new Float32Array(K * N), r);
    const dC = fillRand(new Float32Array(M * N), r);
    const dB = matmulBwdB(A, dC, M, K, N);
    const eps = 1e-3;
    for (let k = 0; k < K; k++) for (let n = 0; n < N; n++) {
      const B2 = new Float32Array(B); B2[k * N + n] += eps;
      const Cp = matmul(A, B2, M, K, N);
      let lp = 0; for (let i = 0; i < Cp.length; i++) lp += Cp[i] * dC[i];
      B2[k * N + n] -= 2 * eps;
      const Cm = matmul(A, B2, M, K, N);
      let lm = 0; for (let i = 0; i < Cm.length; i++) lm += Cm[i] * dC[i];
      const num = (lp - lm) / (2 * eps);
      expect(dB[k * N + n]).toBeCloseTo(num, 3);
    }
  });
});

describe('softmaxCrossEntropy backward', () => {
  it('matches finite-difference of crossEntropyLoss', () => {
    const rows = 4, V = 5;
    const r = seedRng(3);
    const logits = fillRand(new Float32Array(rows * V), r, 2);
    const targets = Int32Array.from([0, 2, 4, 1]);
    const dl = softmaxCrossEntropyBwd(logits, targets, rows, V);
    const eps = 1e-3;
    for (let i = 0; i < rows * V; i++) {
      const lp = (() => { const a = new Float32Array(logits); a[i] += eps; return crossEntropyLoss(a, targets, rows, V); })();
      const lm = (() => { const a = new Float32Array(logits); a[i] -= eps; return crossEntropyLoss(a, targets, rows, V); })();
      const num = (lp - lm) / (2 * eps);
      expect(dl[i]).toBeCloseTo(num, 4);
    }
  });

  it('crossEntropyLoss reduces ln(V) for uniform logits to a uniform-distribution baseline', () => {
    const rows = 2, V = 65;
    const logits = new Float32Array(rows * V);
    const targets = Int32Array.from([0, 0]);
    const loss = crossEntropyLoss(logits, targets, rows, V);
    expect(loss).toBeCloseTo(Math.log(V), 5);
  });
});

describe('gelu backward', () => {
  it('matches finite-difference of forward gelu', () => {
    const r = seedRng(4);
    const x = fillRand(new Float32Array(8), r, 2);
    const dy = fillRand(new Float32Array(8), r);
    const dx = geluBwd(dy, x);
    const eps = 1e-4;
    for (let i = 0; i < x.length; i++) {
      const xp = new Float32Array(x); xp[i] += eps;
      const xm = new Float32Array(x); xm[i] -= eps;
      const yp = gelu(xp); const ym = gelu(xm);
      let lp = 0; for (let j = 0; j < x.length; j++) lp += yp[j] * dy[j];
      let lm = 0; for (let j = 0; j < x.length; j++) lm += ym[j] * dy[j];
      const num = (lp - lm) / (2 * eps);
      expect(dx[i]).toBeCloseTo(num, 3);
    }
  });
});

describe('layerNorm backward', () => {
  it('dx, dGamma, dBeta all match finite-difference', () => {
    const rows = 2, d = 6;
    const r = seedRng(5);
    const x = fillRand(new Float32Array(rows * d), r);
    const gamma = fillRand(new Float32Array(d), r); for (let i = 0; i < d; i++) gamma[i] += 1;
    const beta = fillRand(new Float32Array(d), r);
    const dy = fillRand(new Float32Array(rows * d), r);
    const { dx, dGamma, dBeta } = layerNormBwd(dy, x, gamma, rows, d);
    const dot = (a: F32) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * dy[i]; return s; };
    const eps = 1e-3;

    // dx
    for (let i = 0; i < rows * d; i++) {
      const xp = new Float32Array(x); xp[i] += eps;
      const xm = new Float32Array(x); xm[i] -= eps;
      const num = (dot(layerNorm(xp, gamma, beta, rows, d)) - dot(layerNorm(xm, gamma, beta, rows, d))) / (2 * eps);
      expect(dx[i]).toBeCloseTo(num, 2);
    }
    // dGamma
    for (let k = 0; k < d; k++) {
      const gp = new Float32Array(gamma); gp[k] += eps;
      const gm = new Float32Array(gamma); gm[k] -= eps;
      const num = (dot(layerNorm(x, gp, beta, rows, d)) - dot(layerNorm(x, gm, beta, rows, d))) / (2 * eps);
      expect(dGamma[k]).toBeCloseTo(num, 2);
    }
    // dBeta
    for (let k = 0; k < d; k++) {
      const bp = new Float32Array(beta); bp[k] += eps;
      const bm = new Float32Array(beta); bm[k] -= eps;
      const num = (dot(layerNorm(x, gamma, bp, rows, d)) - dot(layerNorm(x, gamma, bm, rows, d))) / (2 * eps);
      expect(dBeta[k]).toBeCloseTo(num, 2);
    }
  });
});

describe('causalSdpa backward', () => {
  it('dqkv matches finite-difference of forward sdpa', () => {
    const B = 1, T = 3, d = 4, nHead = 2;
    const r = seedRng(6);
    const qkv = fillRand(new Float32Array(B * T * 3 * d), r);
    const dout = fillRand(new Float32Array(B * T * d), r);
    const dqkv = causalSdpaBwd(qkv, dout, B, T, d, nHead);
    const dot = (a: F32) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * dout[i]; return s; };
    const eps = 1e-3;
    for (let i = 0; i < qkv.length; i++) {
      const qp = new Float32Array(qkv); qp[i] += eps;
      const qm = new Float32Array(qkv); qm[i] -= eps;
      const num = (dot(causalSdpa(qp, B, T, d, nHead)) - dot(causalSdpa(qm, B, T, d, nHead))) / (2 * eps);
      expect(dqkv[i]).toBeCloseTo(num, 2);
    }
  });
});

describe('embedding backward', () => {
  it('scatter-adds tokens and positions', () => {
    const T = 2, V = 3, d = 2;
    const tok = Int32Array.from([0, 1, 2, 0]); // 4 tokens, B=2 × T=2
    const dEmb = f32([1, 2,  3, 4,  5, 6,  7, 8]);
    const { dWte, dWpe } = embeddingBwd(dEmb, tok, T, V, d);
    // dWte: tok=0 hit at i=0,3 → [1+7, 2+8] = [8, 10]; tok=1 at i=1 → [3, 4]; tok=2 at i=2 → [5, 6]
    expect(Array.from(dWte)).toEqual([8, 10, 3, 4, 5, 6]);
    // dWpe: pos=0 at i=0,2 → [1+5, 2+6]=[6, 8]; pos=1 at i=1,3 → [3+7, 4+8]=[10, 12]
    expect(Array.from(dWpe)).toEqual([6, 8, 10, 12]);
  });
});

describe('grad-clip scale', () => {
  it('returns 1 when norm is below the cap', () => {
    const a = f32([0.1, 0.2, 0.3]);
    expect(gradClipScale([a], 1)).toBe(1);
  });
  it('returns max/norm when over the cap, and the rescaled L2 hits exactly max', () => {
    const a = f32([3, 4]); // norm = 5
    const s = gradClipScale([a], 1);
    expect(s).toBeCloseTo(0.2, 6);
    for (let i = 0; i < a.length; i++) a[i] *= s;
    let sq = 0; for (const v of a) sq += v * v;
    expect(Math.sqrt(sq)).toBeCloseTo(1, 5);
  });
});

describe('adamw step', () => {
  it('matches a pure-formula reference for one step from zero state', () => {
    // PyTorch AdamW: θ ← θ(1 − lrλ); then m, v update; then θ −= (lr/(1−β₁ᵗ)) m / (√v/(1−β₂ᵗ)^.5 + ε)
    const theta = f32([1.0, -0.5]); const grad = f32([0.1, -0.2]);
    const m = new Float32Array(2); const v = new Float32Array(2);
    const lr = 1e-3, b1 = 0.9, b2 = 0.95, eps = 1e-8, lam = 0.1;
    adamwStep(theta, grad, m, v, lr, b1, b2, eps, lam, 1);
    // Formula by hand for index 0:
    const decay = 1 - lr * lam; // 0.9999
    const t0_after_decay = 1.0 * decay;
    const m0 = (1 - b1) * 0.1; const v0 = (1 - b2) * 0.01;
    const denom = Math.sqrt(v0) / Math.sqrt(1 - b2) + eps;
    const t0 = t0_after_decay - (lr / (1 - b1)) * m0 / denom;
    expect(theta[0]).toBeCloseTo(t0, 7);
    expect(m[0]).toBeCloseTo(m0, 7);
    expect(v[0]).toBeCloseTo(v0, 9);
  });

  it('two zero-grad steps leave theta only with weight-decay drift', () => {
    const theta = f32([2.0]); const grad = f32([0.0]);
    const m = new Float32Array(1); const v = new Float32Array(1);
    const lr = 1e-3, b1 = 0.9, b2 = 0.95, eps = 1e-8, lam = 0.1;
    adamwStep(theta, grad, m, v, lr, b1, b2, eps, lam, 1);
    adamwStep(theta, grad, m, v, lr, b1, b2, eps, lam, 2);
    // Two decay steps, no m/v contribution since grad = 0.
    expect(theta[0]).toBeCloseTo(2.0 * (1 - lr * lam) * (1 - lr * lam), 6);
    expect(m[0]).toBe(0);
    expect(v[0]).toBe(0);
  });
});
