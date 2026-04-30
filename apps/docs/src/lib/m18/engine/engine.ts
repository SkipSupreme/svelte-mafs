// Browser-side wrapper around EngineCore. Imports the WGSL kernels via Vite's
// `?raw` query so they ship as inlined source in the bundle.

import type { ModelConfig } from './config';
import { EngineCore, type F32, type ModelParamData, type KernelSources } from './core';
import { seededRng, fillNormal } from './rng';

import embeddingGatherWgsl from './kernels/embeddingGather.wgsl?raw';
import layerNormWgsl       from './kernels/layerNorm.wgsl?raw';
import matmulWgsl          from './kernels/matmul.wgsl?raw';
import matmulRhsTWgsl      from './kernels/matmulRhsT.wgsl?raw';
import matmulLhsTWgsl      from './kernels/matmulLhsT.wgsl?raw';
import causalSdpaWgsl      from './kernels/causalSdpa.wgsl?raw';
import causalSdpaBwdWgsl   from './kernels/causalSdpaBwd.wgsl?raw';
import residualAddWgsl     from './kernels/residualAdd.wgsl?raw';
import geluWgsl            from './kernels/gelu.wgsl?raw';
import geluBwdWgsl         from './kernels/geluBwd.wgsl?raw';
import lnBwdWgsl           from './kernels/lnBwd.wgsl?raw';
import lnBwdParamsWgsl     from './kernels/lnBwdParams.wgsl?raw';
import softmaxCEBwdWgsl    from './kernels/softmaxCrossEntropyBwd.wgsl?raw';
import embeddingBwdWteWgsl from './kernels/embeddingBwdWte.wgsl?raw';
import embeddingBwdWpeWgsl from './kernels/embeddingBwdWpe.wgsl?raw';
import sumOfSquaresWgsl    from './kernels/sumOfSquares.wgsl?raw';
import scaleInPlaceWgsl    from './kernels/scaleInPlace.wgsl?raw';
import zeroBufferWgsl      from './kernels/zeroBuffer.wgsl?raw';
import copyBufferWgsl      from './kernels/copyBuffer.wgsl?raw';
import addIntoWgsl         from './kernels/addInto.wgsl?raw';
import adamwStepWgsl       from './kernels/adamwStep.wgsl?raw';

const SOURCES: KernelSources = {
  embeddingGather: embeddingGatherWgsl,
  layerNorm:       layerNormWgsl,
  matmul:          matmulWgsl,
  matmulRhsT:      matmulRhsTWgsl,
  matmulLhsT:      matmulLhsTWgsl,
  causalSdpa:      causalSdpaWgsl,
  causalSdpaBwd:   causalSdpaBwdWgsl,
  residualAdd:     residualAddWgsl,
  gelu:            geluWgsl,
  geluBwd:         geluBwdWgsl,
  lnBwd:           lnBwdWgsl,
  lnBwdParams:     lnBwdParamsWgsl,
  softmaxCEBwd:    softmaxCEBwdWgsl,
  embeddingBwdWte: embeddingBwdWteWgsl,
  embeddingBwdWpe: embeddingBwdWpeWgsl,
  sumOfSquares:    sumOfSquaresWgsl,
  scaleInPlace:    scaleInPlaceWgsl,
  zeroBuffer:      zeroBufferWgsl,
  copyBuffer:      copyBufferWgsl,
  addInto:         addIntoWgsl,
  adamwStep:       adamwStepWgsl,
};

export class Engine extends EngineCore {
  static async create(cfg: ModelConfig): Promise<Engine> {
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not available — try Chrome / Edge / Firefox 141+ / Safari 18+ on desktop.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter.');
    const device = await adapter.requestDevice();
    return new Engine(device, cfg);
  }

  static fromDevice(device: GPUDevice, cfg: ModelConfig): Engine {
    return new Engine(device, cfg);
  }

  constructor(device: GPUDevice, cfg: ModelConfig) {
    super(device, cfg, SOURCES);
  }
}

// Re-export the public surface so existing call sites that imported from
// engine.ts directly keep working.
export type { F32, Tensor, BlockParams, ModelParams, BlockParamData, ModelParamData, KernelSources } from './core';

// Seeded init that matches the nanoGPT reference: every Linear weight ~
// N(0, 0.02), every Embedding weight ~ N(0, 0.02), LN gamma=1 / beta=0.
// We can't reproduce torch.manual_seed's exact sequence in JS, so the
// convergence gate is a trajectory-shape match rather than a bit match.
export function seededInitWeights(cfg: ModelConfig, seed: number | string): ModelParamData {
  const rng = seededRng(seed);
  const norm = (n: number, std: number): Float32Array => {
    const a = new Float32Array(n); fillNormal(a, std, rng); return a;
  };
  const ones = (n: number): Float32Array => { const a = new Float32Array(n); a.fill(1); return a; };
  const zeros = (n: number): Float32Array => new Float32Array(n);

  return {
    wte: norm(cfg.vocabSize * cfg.dModel, 0.02),
    wpe: norm(cfg.contextLen * cfg.dModel, 0.02),
    blocks: Array.from({ length: cfg.nLayer }, () => ({
      ln1Gamma: ones(cfg.dModel), ln1Beta: zeros(cfg.dModel),
      wQKV:     norm(cfg.dModel * 3 * cfg.dModel, 0.02),
      wAttnOut: norm(cfg.dModel * cfg.dModel, 0.02),
      ln2Gamma: ones(cfg.dModel), ln2Beta: zeros(cfg.dModel),
      wFFN1:    norm(cfg.dModel * cfg.dFF, 0.02),
      wFFN2:    norm(cfg.dFF * cfg.dModel, 0.02),
    })),
    lnFGamma: ones(cfg.dModel), lnFBeta: zeros(cfg.dModel),
  };
}

// Cosine LR schedule with linear warmup. Matches scripts/m18/nanogpt_reference.py:
//   warmup: lr_max · (step + 1) / warmup_iters    for step < warmup_iters
//   cosine: lr_min + 0.5(lr_max − lr_min)(1 + cos(π·p))   p = (step − warmup) / (total − warmup)
export function cosineLR(step: number, warmupIters: number, totalIters: number, lrMax: number, lrMin: number): number {
  if (step < warmupIters) return lrMax * (step + 1) / warmupIters;
  const p = (step - warmupIters) / Math.max(1, totalIters - warmupIters);
  return lrMin + 0.5 * (lrMax - lrMin) * (1 + Math.cos(Math.PI * p));
}

// Deterministic, seed-free init used for the dev-page smoke pass. Slice 4 will
// replace this with a sfc32-seeded He init threaded through a single PRNG.
export function debugInitWeights(cfg: ModelConfig): ModelParamData {
  const lin = (n: number, scale: number, offset = 0) => {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = scale * Math.sin(i + offset);
    return a;
  };
  const ones = (n: number) => { const a = new Float32Array(n); a.fill(1); return a; };
  const zeros = (n: number) => new Float32Array(n);

  return {
    wte: lin(cfg.vocabSize * cfg.dModel, 0.02),
    wpe: lin(cfg.contextLen * cfg.dModel, 0.02, 100),
    blocks: Array.from({ length: cfg.nLayer }, (_, l) => ({
      ln1Gamma: ones(cfg.dModel), ln1Beta: zeros(cfg.dModel),
      wQKV:     lin(cfg.dModel * 3 * cfg.dModel, 0.05, l * 7),
      wAttnOut: lin(cfg.dModel * cfg.dModel, 0.05, l * 11),
      ln2Gamma: ones(cfg.dModel), ln2Beta: zeros(cfg.dModel),
      wFFN1:    lin(cfg.dModel * cfg.dFF, 0.05, l * 13),
      wFFN2:    lin(cfg.dFF * cfg.dModel, 0.05, l * 17),
    })),
    lnFGamma: ones(cfg.dModel), lnFBeta: zeros(cfg.dModel),
  };
}
