#!/usr/bin/env bun
/**
 * Headless WebGPU smoke runner for the M18 forward engine.
 *
 * Runs the same one-batch forward pass the in-browser /dev/m18-forward/ widget
 * runs, and prints a PASS/FAIL line plus the WGSL-vs-CPU drift. Bun is the
 * runtime of choice (built-in WebGPU); Node would need @webgpu/dawn — heavier
 * dep, no upside for this slice.
 *
 *     bun scripts/m18/gpu_smoke.ts
 *
 * Exits nonzero if any check fails.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { EngineCore, type KernelSources } from '../../apps/docs/src/lib/m18/engine/core';
import { M18_CONFIG } from '../../apps/docs/src/lib/m18/engine/config';
import { debugInitWeights } from '../../apps/docs/src/lib/m18/engine/engine';
import { cpuForward } from '../../apps/docs/src/lib/m18/engine/cpu/forward';
import { softmax } from '../../apps/docs/src/lib/m18/engine/cpu/twins';

const here = dirname(fileURLToPath(import.meta.url));
const kernelsDir = join(here, '..', '..', 'apps', 'docs', 'src', 'lib', 'm18', 'engine', 'kernels');
const read = (name: string): string => readFileSync(join(kernelsDir, name), 'utf8');

const sources: KernelSources = {
  embeddingGather: read('embeddingGather.wgsl'),
  layerNorm:       read('layerNorm.wgsl'),
  matmul:          read('matmul.wgsl'),
  matmulRhsT:      read('matmulRhsT.wgsl'),
  causalSdpa:      read('causalSdpa.wgsl'),
  residualAdd:     read('residualAdd.wgsl'),
  gelu:            read('gelu.wgsl'),
};

async function main(): Promise<number> {
  const gpu = (globalThis as unknown as { navigator?: { gpu?: GPU } }).navigator?.gpu
    ?? (globalThis as unknown as { GPU?: GPU }).GPU;
  if (!gpu) {
    console.error('SKIP — this Bun build does not expose WebGPU (navigator.gpu missing).');
    console.error('       Slice 2 canonical smoke: open /dev/m18-forward/ in a desktop browser.');
    console.error('       The CPU-twin vitest suite covers the math reference independently.');
    return 0; // not a hard failure — Slice 2 has two other diagnostics
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) { console.error('FAIL — no GPU adapter'); return 2; }
  const device = await adapter.requestDevice();

  const cfg = M18_CONFIG;
  const BATCH = 2;
  const N = BATCH * cfg.contextLen;
  const V = cfg.vocabSize;

  const tokens = new Int32Array(N);
  for (let i = 0; i < N; i++) tokens[i] = (i * 7 + 3) % V;
  const weights = debugInitWeights(cfg);

  const engine = new EngineCore(device, cfg, sources);
  engine.loadParameters(weights);

  const t0 = performance.now();
  const wgslLogits = await engine.forward(tokens, BATCH);
  const tWgsl = performance.now() - t0;

  // CPU reference.
  const t1 = performance.now();
  const cpuLogits = cpuForward(weights, tokens, cfg, BATCH);
  const tCpu = performance.now() - t1;

  // Drift.
  let mxDiff = 0; let sumDiff = 0;
  for (let i = 0; i < wgslLogits.length; i++) {
    const e = Math.abs(wgslLogits[i] - cpuLogits[i]);
    if (e > mxDiff) mxDiff = e; sumDiff += e;
  }
  const meanDiff = sumDiff / wgslLogits.length;

  // Softmax invariant.
  const probs = softmax(wgslLogits, N, V);
  let mxRowErr = 0;
  for (let r = 0; r < N; r++) {
    let s = 0; for (let c = 0; c < V; c++) s += probs[r * V + c];
    const e = Math.abs(s - 1); if (e > mxRowErr) mxRowErr = e;
  }

  const PASS_DRIFT = 1e-3;
  const PASS_SOFTMAX = 1e-4;
  const ok = mxDiff < PASS_DRIFT && mxRowErr < PASS_SOFTMAX;

  console.log('M18 forward smoke');
  console.log(`  cfg          n_layer=${cfg.nLayer} d_model=${cfg.dModel} d_ff=${cfg.dFF} n_head=${cfg.nHead} T=${cfg.contextLen} V=${V}`);
  console.log(`  shape        wgsl ${wgslLogits.length} === cpu ${cpuLogits.length}  (expected ${N * V})`);
  console.log(`  drift        max ${mxDiff.toExponential(2)}  mean ${meanDiff.toExponential(2)}  (gate ${PASS_DRIFT})`);
  console.log(`  softmax      max |Σp−1| ${mxRowErr.toExponential(2)}  (gate ${PASS_SOFTMAX})`);
  console.log(`  timings      wgsl ${tWgsl.toFixed(0)}ms  cpu ${tCpu.toFixed(0)}ms`);
  console.log(`  ${ok ? 'PASS' : 'FAIL'}`);
  return ok ? 0 : 1;
}

main().then((code) => process.exit(code)).catch((e) => { console.error(e); process.exit(2); });
