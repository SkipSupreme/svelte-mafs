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
  matmulLhsT:      string;
  causalSdpa:      string;
  causalSdpaBwd:   string;
  residualAdd:     string;
  gelu:            string;
  geluBwd:         string;
  lnBwd:           string;
  lnBwdParams:     string;
  softmaxCEBwd:    string;
  embeddingBwdWte: string;
  embeddingBwdWpe: string;
  sumOfSquares:    string;
  scaleInPlace:    string;
  zeroBuffer:      string;
  copyBuffer:      string;
  addInto:         string;
  adamwStep:       string;
}

export interface Pipelines {
  embeddingGather: GPUComputePipeline;
  layerNorm:       GPUComputePipeline;
  matmul:          GPUComputePipeline;
  matmulRhsT:      GPUComputePipeline;
  matmulLhsT:      GPUComputePipeline;
  causalSdpa:      GPUComputePipeline;
  causalSdpaBwd:   GPUComputePipeline;
  residualAdd:     GPUComputePipeline;
  gelu:            GPUComputePipeline;
  geluBwd:         GPUComputePipeline;
  lnBwd:           GPUComputePipeline;
  lnBwdParams:     GPUComputePipeline;
  softmaxCEBwd:    GPUComputePipeline;
  embeddingBwdWte: GPUComputePipeline;
  embeddingBwdWpe: GPUComputePipeline;
  sumOfSquares:    GPUComputePipeline;
  scaleInPlace:    GPUComputePipeline;
  zeroBuffer:      GPUComputePipeline;
  copyBuffer:      GPUComputePipeline;
  addInto:         GPUComputePipeline;
  adamwStep:       GPUComputePipeline;
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
    const compile = (code: string, name: string): GPUComputePipeline => {
      device.pushErrorScope('validation');
      const m = device.createShaderModule({ code, label: name });
      const pipe = device.createComputePipeline({
        layout: 'auto',
        label: name,
        compute: { module: m, entryPoint: 'main' },
      });
      device.popErrorScope().then((err) => {
        if (err) {
          // eslint-disable-next-line no-console
          console.error(`[m18 kernel ${name}] validation error: ${err.message}`);
        }
      });
      m.getCompilationInfo?.().then((info) => {
        for (const msg of info.messages) {
          if (msg.type === 'error') {
            // eslint-disable-next-line no-console
            console.error(`[m18 kernel ${name}] ${msg.message} @ line ${msg.lineNum}:${msg.linePos}`);
          }
        }
      });
      return pipe;
    };
    this.pipelines = {
      embeddingGather: compile(sources.embeddingGather, 'embeddingGather'),
      layerNorm:       compile(sources.layerNorm, 'layerNorm'),
      matmul:          compile(sources.matmul, 'matmul'),
      matmulRhsT:      compile(sources.matmulRhsT, 'matmulRhsT'),
      matmulLhsT:      compile(sources.matmulLhsT, 'matmulLhsT'),
      causalSdpa:      compile(sources.causalSdpa, 'causalSdpa'),
      causalSdpaBwd:   compile(sources.causalSdpaBwd, 'causalSdpaBwd'),
      residualAdd:     compile(sources.residualAdd, 'residualAdd'),
      gelu:            compile(sources.gelu, 'gelu'),
      geluBwd:         compile(sources.geluBwd, 'geluBwd'),
      lnBwd:           compile(sources.lnBwd, 'lnBwd'),
      lnBwdParams:     compile(sources.lnBwdParams, 'lnBwdParams'),
      softmaxCEBwd:    compile(sources.softmaxCEBwd, 'softmaxCEBwd'),
      embeddingBwdWte: compile(sources.embeddingBwdWte, 'embeddingBwdWte'),
      embeddingBwdWpe: compile(sources.embeddingBwdWpe, 'embeddingBwdWpe'),
      sumOfSquares:    compile(sources.sumOfSquares, 'sumOfSquares'),
      scaleInPlace:    compile(sources.scaleInPlace, 'scaleInPlace'),
      zeroBuffer:      compile(sources.zeroBuffer, 'zeroBuffer'),
      copyBuffer:      compile(sources.copyBuffer, 'copyBuffer'),
      addInto:         compile(sources.addInto, 'addInto'),
      adamwStep:       compile(sources.adamwStep, 'adamwStep'),
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

  /** Read a tensor's contents back to CPU as a Float32Array. */
  async readTensor(t: Tensor): Promise<F32> {
    const n = t.shape.reduce((a, b) => a * b, 1);
    const readback = this.device.createBuffer({
      size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(t.buffer, 0, readback, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.getMappedRange()).slice();
    readback.unmap();
    readback.destroy();
    return out;
  }

  /** Accessor for the gradient buffers populated by trainStep. Returns null
   * before initTraining(). */
  gradients(): ModelParams | null { return this.trainCtx?.grads ?? null; }

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

  // ── Training-step orchestration (Slice 3) ──────────────────────────────────

  private trainCtx: TrainContext | null = null;

  /** Allocate persistent training buffers + grads + AdamW state. */
  initTraining(batch: number): void {
    const cfg = this.cfg; const T = cfg.contextLen; const d = cfg.dModel;
    const B = batch; const N = B * T;

    const tokBuf = this.device.createBuffer({
      size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const tgtBuf = this.device.createBuffer({
      size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Forward activations.
    const acts: Activations = {
      tok: tokBuf, tgt: tgtBuf,
      emb: this.alloc([N, d]),
      blocks: Array.from({ length: cfg.nLayer }, () => ({
        xIn:     this.alloc([N, d]),
        ln1Out:  this.alloc([N, d]),
        qkv:     this.alloc([N, 3 * d]),
        attnOut: this.alloc([N, d]),
        projOut: this.alloc([N, d]),
        xMid:    this.alloc([N, d]),
        ln2Out:  this.alloc([N, d]),
        ffn1Pre: this.alloc([N, cfg.dFF]),
        ffn1Act: this.alloc([N, cfg.dFF]),
        ffn2Out: this.alloc([N, d]),
        xOut:    this.alloc([N, d]),
      })),
      lnFIn:  this.alloc([N, d]),
      lnFOut: this.alloc([N, d]),
      logits: this.alloc([N, cfg.vocabSize]),
    };

    // Backward scratch — gradient buffers reused across the chain.
    const back: BackwardScratch = {
      dlogits: this.alloc([N, cfg.vocabSize]),
      lossPer: this.alloc([N]),
      dRes:    this.alloc([N, d]),     // residual-stream gradient (mutates)
      dXIn:    this.alloc([N, d]),
      dProj:   this.alloc([N, d]),
      dAttn:   this.alloc([N, d]),
      dQkv:    this.alloc([N, 3 * d]),
      dLn1:    this.alloc([N, d]),
      dLn2:    this.alloc([N, d]),
      dFfn2:   this.alloc([N, d]),
      dFfn1A:  this.alloc([N, cfg.dFF]),
      dFfn1P:  this.alloc([N, cfg.dFF]),
      dLnF:    this.alloc([N, d]),
      dEmb:    this.alloc([N, d]),
      dResAcc: this.alloc([N, d]),     // helper for residual-add of LN-bwd output
    };

    // Per-parameter grads + AdamW (m, v) state.
    const grads = this.makeGradMatchingParams();
    const optM = this.makeGradMatchingParams();
    const optV = this.makeGradMatchingParams();

    // Flat list of (theta, grad, m, v, n) tuples — drives grad-clip + AdamW.
    const flat: ParamSlot[] = [];
    const push = (t: Tensor, g: Tensor, m: Tensor, v: Tensor): void => {
      flat.push({ theta: t, grad: g, m, v, n: t.shape.reduce((a, b) => a * b, 1) });
    };
    push(this.params.wte, grads.wte, optM.wte, optV.wte);
    push(this.params.wpe, grads.wpe, optM.wpe, optV.wpe);
    for (let l = 0; l < cfg.nLayer; l++) {
      const p = this.params.blocks[l]; const g = grads.blocks[l];
      const m = optM.blocks[l]; const v = optV.blocks[l];
      push(p.ln1Gamma, g.ln1Gamma, m.ln1Gamma, v.ln1Gamma);
      push(p.ln1Beta,  g.ln1Beta,  m.ln1Beta,  v.ln1Beta);
      push(p.wQKV,     g.wQKV,     m.wQKV,     v.wQKV);
      push(p.wAttnOut, g.wAttnOut, m.wAttnOut, v.wAttnOut);
      push(p.ln2Gamma, g.ln2Gamma, m.ln2Gamma, v.ln2Gamma);
      push(p.ln2Beta,  g.ln2Beta,  m.ln2Beta,  v.ln2Beta);
      push(p.wFFN1,    g.wFFN1,    m.wFFN1,    v.wFFN1);
      push(p.wFFN2,    g.wFFN2,    m.wFFN2,    v.wFFN2);
    }
    push(this.params.lnFGamma, grads.lnFGamma, optM.lnFGamma, optV.lnFGamma);
    push(this.params.lnFBeta,  grads.lnFBeta,  optM.lnFBeta,  optV.lnFBeta);

    // Buffer holding one f32 sum-of-squares per parameter (read back per step
    // so we can compute the global norm on the CPU side).
    const normBuf = this.device.createBuffer({
      size: flat.length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const normReadback = this.device.createBuffer({
      size: flat.length * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Host-side readback helpers for loss + diagnostics.
    const lossReadback = this.device.createBuffer({
      size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.trainCtx = { B, N, acts, back, grads, optM, optV, flat, normBuf, normReadback, lossReadback };
  }

  private makeGradMatchingParams(): ModelParams {
    const cfg = this.cfg;
    return {
      wte: this.alloc([cfg.vocabSize, cfg.dModel]),
      wpe: this.alloc([cfg.contextLen, cfg.dModel]),
      blocks: Array.from({ length: cfg.nLayer }, () => ({
        ln1Gamma: this.alloc([cfg.dModel]), ln1Beta: this.alloc([cfg.dModel]),
        wQKV:     this.alloc([cfg.dModel, 3 * cfg.dModel]),
        wAttnOut: this.alloc([cfg.dModel, cfg.dModel]),
        ln2Gamma: this.alloc([cfg.dModel]), ln2Beta: this.alloc([cfg.dModel]),
        wFFN1:    this.alloc([cfg.dModel, cfg.dFF]),
        wFFN2:    this.alloc([cfg.dFF, cfg.dModel]),
      })),
      lnFGamma: this.alloc([cfg.dModel]), lnFBeta: this.alloc([cfg.dModel]),
    };
  }

  /** One training step: forward → loss → backward → grad-clip → AdamW.
   *  Returns the per-batch mean cross-entropy loss. */
  async trainStep(
    tokens: Int32Array, targets: Int32Array,
    lr: number, t: number,
    hp: { beta1: number; beta2: number; eps: number; lambda: number; clipNorm: number },
  ): Promise<number> {
    const ctx = this.trainCtx; if (!ctx) throw new Error('initTraining() not called');
    const { B, N, acts, back, flat } = ctx;

    this.device.queue.writeBuffer(acts.tok, 0, tokens.buffer, tokens.byteOffset, tokens.byteLength);
    this.device.queue.writeBuffer(acts.tgt, 0, targets.buffer, targets.byteOffset, targets.byteLength);

    const enc = this.device.createCommandEncoder();
    {
      const pass = enc.beginComputePass();
      this.runForward(pass, B, N);
      this.runZeroGrads(pass, ctx);
      this.runBackward(pass, B, N);
      this.runGradNorms(pass, ctx);
      pass.end();
    }
    enc.copyBufferToBuffer(ctx.normBuf, 0, ctx.normReadback, 0, ctx.flat.length * 4);
    enc.copyBufferToBuffer(back.lossPer.buffer, 0, ctx.lossReadback, 0, N * 4);
    this.device.queue.submit([enc.finish()]);

    await Promise.all([ctx.normReadback.mapAsync(GPUMapMode.READ), ctx.lossReadback.mapAsync(GPUMapMode.READ)]);
    const norms = new Float32Array(ctx.normReadback.getMappedRange()).slice();
    const losses = new Float32Array(ctx.lossReadback.getMappedRange()).slice();
    ctx.normReadback.unmap(); ctx.lossReadback.unmap();

    let totalSq = 0;
    for (let i = 0; i < norms.length; i++) totalSq += norms[i];
    const norm = Math.sqrt(totalSq);
    const scale = norm > hp.clipNorm ? hp.clipNorm / norm : 1;

    let lossSum = 0;
    for (let i = 0; i < N; i++) lossSum += losses[i];
    const loss = lossSum / N;

    // Pass 2: scale grads (if needed) + AdamW step.
    const enc2 = this.device.createCommandEncoder();
    const pass2 = enc2.beginComputePass();
    if (scale !== 1) {
      for (const slot of flat) this.dispatchScale(pass2, slot.grad.buffer, slot.n, scale);
    }
    const bc1 = 1 - Math.pow(hp.beta1, t);
    const bc2 = 1 - Math.pow(hp.beta2, t);
    for (const slot of flat) {
      this.dispatchAdamw(pass2, slot, lr, hp.beta1, hp.beta2, hp.eps, hp.lambda, bc1, bc2);
    }
    pass2.end();
    this.device.queue.submit([enc2.finish()]);

    return loss;
  }

  /** Forward + cross-entropy only — no backward, no parameter update. */
  async valLoss(tokens: Int32Array, targets: Int32Array): Promise<number> {
    const ctx = this.trainCtx; if (!ctx) throw new Error('initTraining() not called');
    const { B, N, acts, back } = ctx;

    this.device.queue.writeBuffer(acts.tok, 0, tokens.buffer, tokens.byteOffset, tokens.byteLength);
    this.device.queue.writeBuffer(acts.tgt, 0, targets.buffer, targets.byteOffset, targets.byteLength);

    const enc = this.device.createCommandEncoder();
    {
      const pass = enc.beginComputePass();
      this.runForward(pass, B, N);
      // Loss-only path: fire softmaxCEBwd just so lossPer fills, but we don't
      // care about dlogits or anything below. Cheap.
      this.dispatchSoftmaxCEBwd(pass, acts.logits, acts.tgt, back.dlogits, back.lossPer, N, this.cfg.vocabSize);
      pass.end();
    }
    enc.copyBufferToBuffer(back.lossPer.buffer, 0, ctx.lossReadback, 0, N * 4);
    this.device.queue.submit([enc.finish()]);

    await ctx.lossReadback.mapAsync(GPUMapMode.READ);
    const losses = new Float32Array(ctx.lossReadback.getMappedRange()).slice();
    ctx.lossReadback.unmap();
    let s = 0; for (let i = 0; i < N; i++) s += losses[i];
    return s / N;
  }

  // ── Forward (training variant: writes into persistent activation cache) ────

  private runForward(pass: GPUComputePassEncoder, B: number, N: number): void {
    const cfg = this.cfg; const T = cfg.contextLen; const d = cfg.dModel;
    const ctx = this.trainCtx!; const { acts } = ctx;

    this.dispatchEmbedding(pass, acts.tok, this.params.wte, this.params.wpe, acts.emb, N, d, T);

    let prevX: Tensor = acts.emb;
    for (let l = 0; l < cfg.nLayer; l++) {
      const blk = this.params.blocks[l]; const a = acts.blocks[l];
      // Persist the block input as xIn (needed for residual + LN1 backward).
      this.copyTensor(pass, prevX, a.xIn, N * d);

      this.dispatchLayerNorm(pass, a.xIn, blk.ln1Gamma, blk.ln1Beta, a.ln1Out, N, d);
      this.dispatchMatmul(pass, a.ln1Out, blk.wQKV, a.qkv, N, d, 3 * d);
      this.dispatchSdpa(pass, a.qkv, a.attnOut, B, T, d, cfg.nHead);
      this.dispatchMatmul(pass, a.attnOut, blk.wAttnOut, a.projOut, N, d, d);
      this.dispatchResidual(pass, a.xIn, a.projOut, a.xMid, N * d);

      this.dispatchLayerNorm(pass, a.xMid, blk.ln2Gamma, blk.ln2Beta, a.ln2Out, N, d);
      this.dispatchMatmul(pass, a.ln2Out, blk.wFFN1, a.ffn1Pre, N, d, cfg.dFF);
      this.dispatchGelu(pass, a.ffn1Pre, a.ffn1Act, N * cfg.dFF);
      this.dispatchMatmul(pass, a.ffn1Act, blk.wFFN2, a.ffn2Out, N, cfg.dFF, d);
      this.dispatchResidual(pass, a.xMid, a.ffn2Out, a.xOut, N * d);

      prevX = a.xOut;
    }
    this.copyTensor(pass, prevX, acts.lnFIn, N * d);
    this.dispatchLayerNorm(pass, acts.lnFIn, this.params.lnFGamma, this.params.lnFBeta, acts.lnFOut, N, d);
    this.dispatchMatmulRhsT(pass, acts.lnFOut, this.params.wte, acts.logits, N, d, cfg.vocabSize);
  }

  // ── Backward chain ─────────────────────────────────────────────────────────

  private runBackward(pass: GPUComputePassEncoder, B: number, N: number): void {
    const cfg = this.cfg; const T = cfg.contextLen; const d = cfg.dModel; const V = cfg.vocabSize;
    const ctx = this.trainCtx!; const { acts, back } = ctx;

    // 1. Loss backward — dlogits.
    this.dispatchSoftmaxCEBwd(pass, acts.logits, acts.tgt, back.dlogits, back.lossPer, N, V);

    // 2. Unembedding backward.
    //    Forward: logits = lnFOut · wteᵀ  (matmulRhsT: A=[N,d], B=[V,d], C=[N,V])
    //    dA = dlogits · wte  (plain matmul: A=[N,V], B=[V,d], C=[N,d]) → dLnF
    //    dB = dlogitsᵀ · lnFOut  (matmulLhsT: A=[N,V], B=[N,d], C=[V,d]) → dWte (assigned)
    this.dispatchMatmul(pass, back.dlogits, this.params.wte, back.dLnF, N, V, d);
    this.dispatchMatmulLhsT(pass, back.dlogits, acts.lnFOut, ctx.grads.wte, N, V, d);

    // 3. Final LN backward.
    this.dispatchLnBwd(pass, back.dLnF, acts.lnFIn, this.params.lnFGamma, back.dRes, N, d);
    this.dispatchLnBwdParams(pass, back.dLnF, acts.lnFIn, ctx.grads.lnFGamma, ctx.grads.lnFBeta, N, d);

    // 4. Per block, walk in reverse.
    for (let l = cfg.nLayer - 1; l >= 0; l--) {
      const blk = this.params.blocks[l]; const a = acts.blocks[l]; const g = ctx.grads.blocks[l];

      // Residual2: x_out = x_mid + ffn2_out.
      // dx_mid = dRes (in place); dffn2 = dRes (copy out before overwriting).
      this.copyTensor(pass, back.dRes, back.dFfn2, N * d);

      // FFN2 matmul backward.
      // Forward: ffn2Out = ffn1Act · wFFN2  (A=[N,dFF], B=[dFF,d])
      // dA = dFfn2 · wFFN2ᵀ  (matmulRhsT: A=[N,d], B=[dFF,d], C=[N,dFF]) → dFfn1A
      // dB = ffn1Actᵀ · dFfn2  (matmulLhsT: A=[N,dFF], B=[N,d], C=[dFF,d]) → dWFFN2
      this.dispatchMatmulRhsT(pass, back.dFfn2, blk.wFFN2, back.dFfn1A, N, d, cfg.dFF);
      this.dispatchMatmulLhsT(pass, a.ffn1Act, back.dFfn2, g.wFFN2, N, cfg.dFF, d);

      // GELU backward: dffn1_pre = geluBwd(dffn1_act, ffn1_pre).
      this.dispatchGeluBwd(pass, back.dFfn1A, a.ffn1Pre, back.dFfn1P, N * cfg.dFF);

      // FFN1 matmul backward.
      // Forward: ffn1Pre = ln2Out · wFFN1  (A=[N,d], B=[d,dFF])
      // dA = dFfn1P · wFFN1ᵀ  (matmulRhsT: A=[N,dFF], B=[d,dFF], C=[N,d]) → dLn2
      // dB = ln2Outᵀ · dFfn1P  (matmulLhsT: A=[N,d], B=[N,dFF], C=[d,dFF]) → dWFFN1
      this.dispatchMatmulRhsT(pass, back.dFfn1P, blk.wFFN1, back.dLn2, N, cfg.dFF, d);
      this.dispatchMatmulLhsT(pass, a.ln2Out, back.dFfn1P, g.wFFN1, N, d, cfg.dFF);

      // LN2 backward — writes into a separate buffer, then add to dRes (residual fan-in).
      this.dispatchLnBwd(pass, back.dLn2, a.xMid, blk.ln2Gamma, back.dResAcc, N, d);
      this.dispatchLnBwdParams(pass, back.dLn2, a.xMid, g.ln2Gamma, g.ln2Beta, N, d);
      this.dispatchAddInto(pass, back.dResAcc, back.dRes, N * d);

      // Residual1: x_mid = x_in + proj_out → dproj = dRes; dxIn = dRes.
      this.copyTensor(pass, back.dRes, back.dProj, N * d);

      // Attn-out matmul backward.
      // Forward: projOut = attnOut · wAttnOut  (A=[N,d], B=[d,d])
      // dA = dProj · wAttnOutᵀ  (matmulRhsT: A=[N,d], B=[d,d], C=[N,d]) → dAttn
      // dB = attnOutᵀ · dProj  (matmulLhsT: A=[N,d], B=[N,d], C=[d,d]) → dWAttnOut
      this.dispatchMatmulRhsT(pass, back.dProj, blk.wAttnOut, back.dAttn, N, d, d);
      this.dispatchMatmulLhsT(pass, a.attnOut, back.dProj, g.wAttnOut, N, d, d);

      // SDPA backward — writes dQkv from qkv + dAttn (must zero dQkv first since it accumulates).
      this.dispatchZero(pass, back.dQkv.buffer, N * 3 * d);
      this.dispatchSdpaBwd(pass, a.qkv, back.dAttn, back.dQkv, B, T, d, cfg.nHead);

      // QKV matmul backward.
      // Forward: qkv = ln1Out · wQKV  (A=[N,d], B=[d,3d])
      // dA = dQkv · wQKVᵀ  (matmulRhsT: A=[N,3d], B=[d,3d], C=[N,d]) → dLn1
      // dB = ln1Outᵀ · dQkv  (matmulLhsT: A=[N,d], B=[N,3d], C=[d,3d]) → dWQKV
      this.dispatchMatmulRhsT(pass, back.dQkv, blk.wQKV, back.dLn1, N, 3 * d, d);
      this.dispatchMatmulLhsT(pass, a.ln1Out, back.dQkv, g.wQKV, N, d, 3 * d);

      // LN1 backward — writes to dResAcc, then add to dRes.
      this.dispatchLnBwd(pass, back.dLn1, a.xIn, blk.ln1Gamma, back.dResAcc, N, d);
      this.dispatchLnBwdParams(pass, back.dLn1, a.xIn, g.ln1Gamma, g.ln1Beta, N, d);
      this.dispatchAddInto(pass, back.dResAcc, back.dRes, N * d);
    }

    // 5. Embedding backward — dRes is now the gradient on the post-embedding tensor.
    //    dWte already holds the unembedding-bwd contribution; embeddingBwdWte
    //    ADDS the scatter-add contribution. dWpe is fresh (assignment).
    this.copyTensor(pass, back.dRes, back.dEmb, N * d);
    this.dispatchEmbeddingBwdWte(pass, back.dEmb, acts.tok, ctx.grads.wte, N, V, d);
    this.dispatchEmbeddingBwdWpe(pass, back.dEmb, ctx.grads.wpe, N, T, d);
  }

  private runZeroGrads(pass: GPUComputePassEncoder, ctx: TrainContext): void {
    // Most grad buffers are written via assignment by their producing kernels,
    // so they don't need clearing. The exceptions are wte (tied — accumulates
    // unembedding bwd + embedding bwd), wpe (single writer but clearer to
    // zero), and dQkv (sdpaBwd accumulates). Zero everything for simplicity;
    // total bytes ≈ 800 KB so the pass is cheap.
    for (const slot of ctx.flat) this.dispatchZero(pass, slot.grad.buffer, slot.n);
  }

  private runGradNorms(pass: GPUComputePassEncoder, ctx: TrainContext): void {
    for (let i = 0; i < ctx.flat.length; i++) {
      const s = ctx.flat[i];
      this.dispatchSumOfSquares(pass, s.grad.buffer, ctx.normBuf, s.n, i);
    }
  }

  // ── Dispatch helpers for the new pipelines ─────────────────────────────────

  private uniformF32(values: number[]): GPUBuffer {
    const buf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const arr = new Float32Array(4);
    for (let i = 0; i < 4; i++) arr[i] = values[i] ?? 0;
    this.device.queue.writeBuffer(buf, 0, arr.buffer);
    return buf;
  }

  private uniformAdam(p: { lr: number; b1: number; b2: number; eps: number; lam: number; bc1: number; bc2: number }): GPUBuffer {
    const buf = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const arr = new Float32Array(8);
    arr[0] = p.lr; arr[1] = p.b1; arr[2] = p.b2; arr[3] = p.eps;
    arr[4] = p.lam; arr[5] = p.bc1; arr[6] = p.bc2; arr[7] = 0;
    this.device.queue.writeBuffer(buf, 0, arr.buffer);
    return buf;
  }

  private dispatchMatmulLhsT(pass: GPUComputePassEncoder, A: Tensor, B: Tensor, C: Tensor, M: number, K: number, N: number): void {
    const u = this.uniform([M, K, N]);
    const bg = this.bg(this.pipelines.matmulLhsT, [
      { binding: 0, resource: A.buffer }, { binding: 1, resource: B.buffer },
      { binding: 2, resource: C.buffer }, { binding: 3, resource: u },
    ]);
    pass.setPipeline(this.pipelines.matmulLhsT);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(K / 16));
  }

  private dispatchSoftmaxCEBwd(pass: GPUComputePassEncoder, logits: Tensor, targets: GPUBuffer, dlogits: Tensor, lossPer: Tensor, rows: number, V: number): void {
    const u = this.uniform([rows, V]);
    const bg = this.bg(this.pipelines.softmaxCEBwd, [
      { binding: 0, resource: logits.buffer }, { binding: 1, resource: targets },
      { binding: 2, resource: dlogits.buffer }, { binding: 3, resource: lossPer.buffer },
      { binding: 4, resource: u },
    ]);
    pass.setPipeline(this.pipelines.softmaxCEBwd);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(rows);
  }

  private dispatchGeluBwd(pass: GPUComputePassEncoder, dy: Tensor, x: Tensor, dx: Tensor, n: number): void {
    const u = this.uniform([n]);
    const bg = this.bg(this.pipelines.geluBwd, [
      { binding: 0, resource: dy.buffer }, { binding: 1, resource: x.buffer },
      { binding: 2, resource: dx.buffer }, { binding: 3, resource: u },
    ]);
    pass.setPipeline(this.pipelines.geluBwd);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }

  private dispatchLnBwd(pass: GPUComputePassEncoder, dy: Tensor, x: Tensor, gamma: Tensor, dx: Tensor, rows: number, d: number): void {
    const u = this.uniform([rows, d]);
    const bg = this.bg(this.pipelines.lnBwd, [
      { binding: 0, resource: dy.buffer }, { binding: 1, resource: x.buffer },
      { binding: 2, resource: gamma.buffer }, { binding: 3, resource: dx.buffer },
      { binding: 4, resource: u },
    ]);
    pass.setPipeline(this.pipelines.lnBwd);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(rows);
  }

  private dispatchLnBwdParams(pass: GPUComputePassEncoder, dy: Tensor, x: Tensor, dGamma: Tensor, dBeta: Tensor, rows: number, d: number): void {
    const u = this.uniform([rows, d]);
    const bg = this.bg(this.pipelines.lnBwdParams, [
      { binding: 0, resource: dy.buffer }, { binding: 1, resource: x.buffer },
      { binding: 2, resource: dGamma.buffer }, { binding: 3, resource: dBeta.buffer },
      { binding: 4, resource: u },
    ]);
    pass.setPipeline(this.pipelines.lnBwdParams);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(d / 64));
  }

  private dispatchSdpaBwd(pass: GPUComputePassEncoder, qkv: Tensor, dout: Tensor, dqkv: Tensor, B: number, T: number, d: number, nHead: number): void {
    const u = this.uniform([B, T, d, nHead]);
    const bg = this.bg(this.pipelines.causalSdpaBwd, [
      { binding: 0, resource: qkv.buffer }, { binding: 1, resource: dout.buffer },
      { binding: 2, resource: dqkv.buffer }, { binding: 3, resource: u },
    ]);
    pass.setPipeline(this.pipelines.causalSdpaBwd);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(nHead, B);
  }

  private dispatchEmbeddingBwdWte(pass: GPUComputePassEncoder, dEmb: Tensor, tokIds: GPUBuffer, dWte: Tensor, N: number, V: number, d: number): void {
    const u = this.uniform([N, V, d]);
    const bg = this.bg(this.pipelines.embeddingBwdWte, [
      { binding: 0, resource: dEmb.buffer }, { binding: 1, resource: tokIds },
      { binding: 2, resource: dWte.buffer }, { binding: 3, resource: u },
    ]);
    pass.setPipeline(this.pipelines.embeddingBwdWte);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(V);
  }

  private dispatchEmbeddingBwdWpe(pass: GPUComputePassEncoder, dEmb: Tensor, dWpe: Tensor, N: number, T: number, d: number): void {
    const u = this.uniform([N, T, d]);
    const bg = this.bg(this.pipelines.embeddingBwdWpe, [
      { binding: 0, resource: dEmb.buffer }, { binding: 1, resource: dWpe.buffer },
      { binding: 2, resource: u },
    ]);
    pass.setPipeline(this.pipelines.embeddingBwdWpe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(T);
  }

  private dispatchSumOfSquares(pass: GPUComputePassEncoder, g: GPUBuffer, out: GPUBuffer, N: number, outIdx: number): void {
    const u = this.uniform([N, outIdx]);
    const bg = this.bg(this.pipelines.sumOfSquares, [
      { binding: 0, resource: g }, { binding: 1, resource: out }, { binding: 2, resource: u },
    ]);
    pass.setPipeline(this.pipelines.sumOfSquares);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
  }

  private dispatchScale(pass: GPUComputePassEncoder, g: GPUBuffer, n: number, scale: number): void {
    const uN = this.uniform([n]);
    const uS = this.uniformF32([scale]);
    const bg = this.bg(this.pipelines.scaleInPlace, [
      { binding: 0, resource: g }, { binding: 1, resource: uN }, { binding: 2, resource: uS },
    ]);
    pass.setPipeline(this.pipelines.scaleInPlace);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }

  private dispatchZero(pass: GPUComputePassEncoder, g: GPUBuffer, n: number): void {
    const u = this.uniform([n]);
    const bg = this.bg(this.pipelines.zeroBuffer, [
      { binding: 0, resource: g }, { binding: 1, resource: u },
    ]);
    pass.setPipeline(this.pipelines.zeroBuffer);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }

  private dispatchAdamw(pass: GPUComputePassEncoder, slot: ParamSlot, lr: number, b1: number, b2: number, eps: number, lam: number, bc1: number, bc2: number): void {
    const uN = this.uniform([slot.n]);
    const uH = this.uniformAdam({ lr, b1, b2, eps, lam, bc1, bc2 });
    const bg = this.bg(this.pipelines.adamwStep, [
      { binding: 0, resource: slot.theta.buffer }, { binding: 1, resource: slot.grad.buffer },
      { binding: 2, resource: slot.m.buffer },     { binding: 3, resource: slot.v.buffer },
      { binding: 4, resource: uN }, { binding: 5, resource: uH },
    ]);
    pass.setPipeline(this.pipelines.adamwStep);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(slot.n / 64));
  }

  /** target += delta (elementwise). Uses addInto kernel — distinct buffers
   *  required (WGSL forbids same-buffer read+read_write in one pass). */
  private dispatchAddInto(pass: GPUComputePassEncoder, delta: Tensor, target: Tensor, n: number): void {
    const u = this.uniform([n]);
    const bg = this.bg(this.pipelines.addInto, [
      { binding: 0, resource: delta.buffer }, { binding: 1, resource: target.buffer },
      { binding: 2, resource: u },
    ]);
    pass.setPipeline(this.pipelines.addInto);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }

  private copyTensor(pass: GPUComputePassEncoder, src: Tensor, dst: Tensor, n: number): void {
    const u = this.uniform([n]);
    const bg = this.bg(this.pipelines.copyBuffer, [
      { binding: 0, resource: src.buffer }, { binding: 1, resource: dst.buffer },
      { binding: 2, resource: u },
    ]);
    pass.setPipeline(this.pipelines.copyBuffer);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
  }
}

// ── Training-context types ────────────────────────────────────────────────────

interface BlockActivations {
  xIn: Tensor; ln1Out: Tensor; qkv: Tensor; attnOut: Tensor; projOut: Tensor;
  xMid: Tensor; ln2Out: Tensor; ffn1Pre: Tensor; ffn1Act: Tensor; ffn2Out: Tensor;
  xOut: Tensor;
}

interface Activations {
  tok: GPUBuffer; tgt: GPUBuffer;
  emb: Tensor;
  blocks: BlockActivations[];
  lnFIn: Tensor; lnFOut: Tensor; logits: Tensor;
}

interface BackwardScratch {
  dlogits: Tensor; lossPer: Tensor;
  dRes: Tensor; dXIn: Tensor; dProj: Tensor; dAttn: Tensor; dQkv: Tensor;
  dLn1: Tensor; dLn2: Tensor; dFfn2: Tensor; dFfn1A: Tensor; dFfn1P: Tensor;
  dLnF: Tensor; dEmb: Tensor;
  dResAcc: Tensor;
}

interface ParamSlot {
  theta: Tensor; grad: Tensor; m: Tensor; v: Tensor; n: number;
}

interface TrainContext {
  B: number; N: number;
  acts: Activations;
  back: BackwardScratch;
  grads: ModelParams; optM: ModelParams; optV: ModelParams;
  flat: ParamSlot[];
  normBuf: GPUBuffer; normReadback: GPUBuffer;
  lossReadback: GPUBuffer;
}
