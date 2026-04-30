// CPU reference forward — composes the per-op twins into a full transformer
// forward pass. Used by the /dev/m18-forward/ smoke widget to compare against
// the WGSL forward at ~1e-4 tolerance.

import type { ModelConfig } from '../config';
import type { F32, ModelParamData } from '../engine';
import { embeddingGather, layerNorm, matmul, causalSdpa, residualAdd, gelu, unembedding } from './twins';

export function cpuForward(weights: ModelParamData, tokenIds: Int32Array, cfg: ModelConfig, batch: number): F32 {
  const T = cfg.contextLen; const d = cfg.dModel; const dFF = cfg.dFF;
  const N = batch * T;

  let x = embeddingGather(tokenIds, T, weights.wte, weights.wpe, cfg.vocabSize, d);

  for (const blk of weights.blocks) {
    // Pre-LN attention.
    const ln1 = layerNorm(x, blk.ln1Gamma, blk.ln1Beta, N, d);
    const qkv = matmul(ln1, blk.wQKV, N, d, 3 * d);
    const attn = causalSdpa(qkv, batch, T, d, cfg.nHead);
    const proj = matmul(attn, blk.wAttnOut, N, d, d);
    x = residualAdd(x, proj);

    // Pre-LN FFN.
    const ln2 = layerNorm(x, blk.ln2Gamma, blk.ln2Beta, N, d);
    const ff1 = matmul(ln2, blk.wFFN1, N, d, dFF);
    const ff1a = gelu(ff1);
    const ff2 = matmul(ff1a, blk.wFFN2, N, dFF, d);
    x = residualAdd(x, ff2);
  }

  const xn = layerNorm(x, weights.lnFGamma, weights.lnFBeta, N, d);
  return unembedding(xn, weights.wte, N, d, cfg.vocabSize);
}
