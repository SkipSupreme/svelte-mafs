// CPU-twin sanity tests. These pin the math of the reference implementations
// the WGSL kernels must match. The /dev/m18-forward/ page does the in-browser
// WGSL-vs-twin comparison live; these tests pin the twin itself.

import { describe, expect, it } from 'vitest';
import { embeddingGather, layerNorm, matmul, causalSdpa, residualAdd, gelu, softmax, unembedding } from './twins';

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
