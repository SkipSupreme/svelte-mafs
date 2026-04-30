// Browser-side wrapper around EngineCore. Imports the WGSL kernels via Vite's
// `?raw` query so they ship as inlined source in the bundle.

import type { ModelConfig } from './config';
import { EngineCore, type F32, type ModelParamData, type KernelSources } from './core';

import embeddingGatherWgsl from './kernels/embeddingGather.wgsl?raw';
import layerNormWgsl from './kernels/layerNorm.wgsl?raw';
import matmulWgsl from './kernels/matmul.wgsl?raw';
import matmulRhsTWgsl from './kernels/matmulRhsT.wgsl?raw';
import causalSdpaWgsl from './kernels/causalSdpa.wgsl?raw';
import residualAddWgsl from './kernels/residualAdd.wgsl?raw';
import geluWgsl from './kernels/gelu.wgsl?raw';

const SOURCES: KernelSources = {
  embeddingGather: embeddingGatherWgsl,
  layerNorm: layerNormWgsl,
  matmul: matmulWgsl,
  matmulRhsT: matmulRhsTWgsl,
  causalSdpa: causalSdpaWgsl,
  residualAdd: residualAddWgsl,
  gelu: geluWgsl,
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
