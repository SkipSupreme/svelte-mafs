// M18 capstone — locked architecture (see docs/plans/2026-04-29-m18-capstone-implementation.md §0).
// Trainable params ≈ 196,608 (12 · d² · n_layer = 12 · 64² · 4) plus 8,320 token-embed
// (tied with unembed) and 4,096 learned positional → ~209k total. The honest
// "200k-parameter" number we ship in lessons.

export interface ModelConfig {
  readonly vocabSize: number; // V
  readonly contextLen: number; // T
  readonly dModel: number; // d
  readonly nHead: number;
  readonly nLayer: number;
  readonly dFF: number; // 4 · d typically
}

export const M18_CONFIG: ModelConfig = {
  vocabSize: 65,
  contextLen: 64,
  dModel: 64,
  nHead: 4,
  nLayer: 4,
  dFF: 256,
} as const;

export function dHead(cfg: ModelConfig): number {
  return cfg.dModel / cfg.nHead;
}
