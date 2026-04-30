// Engine core. Pure WebGPU plumbing — no Vite-specific imports — so the same
// class can run inside the Astro app (browser) and inside Bun's headless
// WebGPU (the GPU smoke runner). Source strings are injected by the consumer.

import type { ModelConfig } from './config';

export type F32 = Float32Array;

export interface Tensor {
  buffer: GPUBuffer;
  shape: readonly number[];
  dtype: 'f32';
}

export interface BlockParams {
  ln1Gamma: Tensor; ln1Beta: Tensor;
  wQKV: Tensor;     // [d, 3d]
  wAttnOut: Tensor; // [d, d]
  ln2Gamma: Tensor; ln2Beta: Tensor;
  wFFN1: Tensor;    // [d, 4d]
  wFFN2: Tensor;    // [4d, d]
}

export interface ModelParams {
  wte: Tensor; wpe: Tensor;
  blocks: BlockParams[];
  lnFGamma: Tensor; lnFBeta: Tensor;
}

export interface KernelSources {
  embeddingGather: string;
  layerNorm:       string;
  matmul:          string;
  matmulRhsT:      string;
  causalSdpa:      string;
  residualAdd:     string;
  gelu:            string;
}

export interface Pipelines {
  embeddingGather: GPUComputePipeline;
  layerNorm:       GPUComputePipeline;
  matmul:          GPUComputePipeline;
  matmulRhsT:      GPUComputePipeline;
  causalSdpa:      GPUComputePipeline;
  residualAdd:     GPUComputePipeline;
  gelu:            GPUComputePipeline;
}

export interface BlockParamData {
  ln1Gamma: F32; ln1Beta: F32;
  wQKV: F32; wAttnOut: F32;
  ln2Gamma: F32; ln2Beta: F32;
  wFFN1: F32; wFFN2: F32;
}

export interface ModelParamData {
  wte: F32; wpe: F32;
  blocks: BlockParamData[];
  lnFGamma: F32; lnFBeta: F32;
}

export class EngineCore {
  readonly device: GPUDevice;
  readonly cfg: ModelConfig;
  readonly pipelines: Pipelines;
  params!: ModelParams;

  constructor(device: GPUDevice, cfg: ModelConfig, sources: KernelSources) {
    this.device = device; this.cfg = cfg;
    const compile = (code: string): GPUComputePipeline => {
      const m = device.createShaderModule({ code });
      return device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' } });
    };
    this.pipelines = {
      embeddingGather: compile(sources.embeddingGather),
      layerNorm:       compile(sources.layerNorm),
      matmul:          compile(sources.matmul),
      matmulRhsT:      compile(sources.matmulRhsT),
      causalSdpa:      compile(sources.causalSdpa),
      residualAdd:     compile(sources.residualAdd),
      gelu:            compile(sources.gelu),
    };
  }

  alloc(shape: readonly number[]): Tensor {
    const n = shape.reduce((a, b) => a * b, 1);
    const buffer = this.device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    return { buffer, shape, dtype: 'f32' };
  }

  upload(t: Tensor, data: F32): void {
    this.device.queue.writeBuffer(t.buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  uploadInt(buf: GPUBuffer, data: Int32Array | Uint32Array): void {
    this.device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  loadParameters(weights: ModelParamData): void {
    const cfg = this.cfg;
    const wte = this.alloc([cfg.vocabSize, cfg.dModel]); this.upload(wte, weights.wte);
    const wpe = this.alloc([cfg.contextLen, cfg.dModel]); this.upload(wpe, weights.wpe);
    const blocks: BlockParams[] = weights.blocks.map((b) => {
      const ln1Gamma = this.alloc([cfg.dModel]); this.upload(ln1Gamma, b.ln1Gamma);
      const ln1Beta  = this.alloc([cfg.dModel]); this.upload(ln1Beta, b.ln1Beta);
      const wQKV     = this.alloc([cfg.dModel, 3 * cfg.dModel]); this.upload(wQKV, b.wQKV);
      const wAttnOut = this.alloc([cfg.dModel, cfg.dModel]); this.upload(wAttnOut, b.wAttnOut);
      const ln2Gamma = this.alloc([cfg.dModel]); this.upload(ln2Gamma, b.ln2Gamma);
      const ln2Beta  = this.alloc([cfg.dModel]); this.upload(ln2Beta, b.ln2Beta);
      const wFFN1    = this.alloc([cfg.dModel, cfg.dFF]); this.upload(wFFN1, b.wFFN1);
      const wFFN2    = this.alloc([cfg.dFF, cfg.dModel]); this.upload(wFFN2, b.wFFN2);
      return { ln1Gamma, ln1Beta, wQKV, wAttnOut, ln2Gamma, ln2Beta, wFFN1, wFFN2 };
    });
    const lnFGamma = this.alloc([cfg.dModel]); this.upload(lnFGamma, weights.lnFGamma);
    const lnFBeta  = this.alloc([cfg.dModel]); this.upload(lnFBeta, weights.lnFBeta);
    this.params = { wte, wpe, blocks, lnFGamma, lnFBeta };
  }

  async forward(tokenIds: Int32Array, batch: number): Promise<F32> {
    const cfg = this.cfg; const T = cfg.contextLen; const d = cfg.dModel;
    const B = batch; const N = B * T;
    if (tokenIds.length !== N) throw new Error(`forward: expected ${N} token ids, got ${tokenIds.length}`);

    const tokBuf = this.device.createBuffer({
      size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.uploadInt(tokBuf, tokenIds);

    const x   = this.alloc([N, d]);
    const tmp = this.alloc([N, d]);
    const lnOut = this.alloc([N, d]);
    const qkv   = this.alloc([N, 3 * d]);
    const attn  = this.alloc([N, d]);
    const proj  = this.alloc([N, d]);
    const ffn1  = this.alloc([N, cfg.dFF]);
    const ffn1A = this.alloc([N, cfg.dFF]);
    const ffn2  = this.alloc([N, d]);
    const logits = this.alloc([N, cfg.vocabSize]);

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();

    this.dispatchEmbedding(pass, tokBuf, this.params.wte, this.params.wpe, x, N, d, T);

    for (const blk of this.params.blocks) {
      this.dispatchLayerNorm(pass, x, blk.ln1Gamma, blk.ln1Beta, lnOut, N, d);
      this.dispatchMatmul(pass, lnOut, blk.wQKV, qkv, N, d, 3 * d);
      this.dispatchSdpa(pass, qkv, attn, B, T, d, cfg.nHead);
      this.dispatchMatmul(pass, attn, blk.wAttnOut, proj, N, d, d);
      this.dispatchResidual(pass, x, proj, tmp, N * d);
      this.swap(x, tmp);

      this.dispatchLayerNorm(pass, x, blk.ln2Gamma, blk.ln2Beta, lnOut, N, d);
      this.dispatchMatmul(pass, lnOut, blk.wFFN1, ffn1, N, d, cfg.dFF);
      this.dispatchGelu(pass, ffn1, ffn1A, N * cfg.dFF);
      this.dispatchMatmul(pass, ffn1A, blk.wFFN2, ffn2, N, cfg.dFF, d);
      this.dispatchResidual(pass, x, ffn2, tmp, N * d);
      this.swap(x, tmp);
    }

    this.dispatchLayerNorm(pass, x, this.params.lnFGamma, this.params.lnFBeta, lnOut, N, d);
    this.dispatchMatmulRhsT(pass, lnOut, this.params.wte, logits, N, d, cfg.vocabSize);

    pass.end();

    const readback = this.device.createBuffer({
      size: N * cfg.vocabSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    enc.copyBufferToBuffer(logits.buffer, 0, readback, 0, N * cfg.vocabSize * 4);
    this.device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();

    [tokBuf, x.buffer, tmp.buffer, lnOut.buffer, qkv.buffer, attn.buffer,
     proj.buffer, ffn1.buffer, ffn1A.buffer, ffn2.buffer, logits.buffer, readback]
      .forEach((b) => { try { b.destroy(); } catch { /* ok */ } });

    return out;
  }

  // ── per-op dispatch helpers ────────────────────────────────────────────────

  private uniform(values: number[]): GPUBuffer {
    const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const arr = new Uint32Array(4);
    for (let i = 0; i < 4; i++) arr[i] = values[i] ?? 0;
    this.device.queue.writeBuffer(buf, 0, arr.buffer);
    return buf;
  }

  private bg(pipeline: GPUComputePipeline, entries: Array<{ binding: number; resource: GPUBuffer }>): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((e) => ({ binding: e.binding, resource: { buffer: e.resource } })),
    });
  }

  private dispatchEmbedding(pass: GPUComputePassEncoder, tokIds: GPUBuffer, wte: Tensor, wpe: Tensor, out: Tensor, rows: number, d: number, T: number): void {
    const u = this.uniform([rows, d, T]);
    const bg = this.bg(this.pipelines.embeddingGather, [
      { binding: 0, resource: tokIds }, { binding: 1, resource: wte.buffer },
      { binding: 2, resource: wpe.buffer }, { binding: 3, resource: out.buffer },
      { binding: 4, resource: u },
    ]);
    pass.setPipeline(this.pipelines.embeddingGather);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil((rows * d) / 64));
  }

  private dispatchLayerNorm(pass: GPUComputePassEncoder, x: Tensor, gamma: Tensor, beta: Tensor, y: Tensor, rows: number, d: number): void {
    const u = this.uniform([rows, d]);
    const bg = this.bg(this.pipelines.layerNorm, [
      { binding: 0, resource: x.buffer }, { binding: 1, resource: gamma.buffer },
      { binding: 2, resource: beta.buffer }, { binding: 3, resource: y.buffer },
      { binding: 4, resource: u },
    ]);
    pass.setPipeline(this.pipelines.layerNorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(rows);
  }

  private dispatchMatmul(pass: GPUComputePassEncoder, A: Tensor, B: Tensor, C: Tensor, M: number, K: number, N: number): void {
    const u = this.uniform([M, K, N]);
    const bg = this.bg(this.pipelines.matmul, [
      { binding: 0, resource: A.buffer }, { binding: 1, resource: B.buffer },
      { binding: 2, resource: C.buffer }, { binding: 3, resource: u },
    ]);
    pass.setPipeline(this.pipelines.matmul);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
  }

  private dispatchMatmulRhsT(pass: GPUComputePassEncoder, A: Tensor, B: Tensor, C: Tensor, M: number, K: number, N: number): void {
    const u = this.uniform([M, K, N]);
    const bg = this.bg(this.pipelines.matmulRhsT, [
      { binding: 0, resource: A.buffer }, { binding: 1, resource: B.buffer },
      { binding: 2, resource: C.buffer }, { binding: 3, resource: u },
    ]);
    pass.setPipeline(this.pipelines.matmulRhsT);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
  }

  private dispatchSdpa(pass: GPUComputePassEncoder, qkv: Tensor, out: Tensor, B: number, T: number, d: number, nHead: number): void {
    const u = this.uniform([B, T, d, nHead]);
    const bg = this.bg(this.pipelines.causalSdpa, [
      { binding: 0, resource: qkv.buffer }, { binding: 1, resource: out.buffer },
      { binding: 2, resource: u },
    ]);
    pass.setPipeline(this.pipelines.causalSdpa);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(T, nHead, B);
  }

  private dispatchResidual(pass: GPUComputePassEncoder, a: Tensor, b: Tensor, out: Tensor, n: number): void {
    const u = this.uniform([n]);
    const bg = this.bg(this.pipelines.residualAdd, [
      { binding: 0, resource: a.buffer }, { binding: 1, resource: b.buffer },
      { binding: 2, resource: out.buffer }, { binding: 3, resource: u },
    ]);
    pass.setPipeline(this.pipelines.residualAdd);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }

  private dispatchGelu(pass: GPUComputePassEncoder, x: Tensor, y: Tensor, n: number): void {
    const u = this.uniform([n]);
    const bg = this.bg(this.pipelines.gelu, [
      { binding: 0, resource: x.buffer }, { binding: 1, resource: y.buffer },
      { binding: 2, resource: u },
    ]);
    pass.setPipeline(this.pipelines.gelu);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }

  private swap(a: Tensor, b: Tensor): void {
    const t = a.buffer;
    (a as { buffer: GPUBuffer }).buffer = b.buffer;
    (b as { buffer: GPUBuffer }).buffer = t;
  }
}
