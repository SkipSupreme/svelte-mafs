export { M18_CONFIG, dHead } from './config';
export type { ModelConfig } from './config';

export { Engine, debugInitWeights, seededInitWeights, cosineLR } from './engine';
export type {
  Tensor, BlockParams, ModelParams, ModelParamData, BlockParamData, F32,
} from './engine';

export { seededRng, cyrb128, sfc32 } from './rng';
export { loadTinyShakespeare, getBatch } from './data';
export type { CorpusBundle } from './data';

export * as cpu from './cpu/twins';
export { cpuForward } from './cpu/forward';
