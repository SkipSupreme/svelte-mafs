<script lang="ts">
  // M18 train smoke — runs the full 2,000-iter convergence gate live in the
  // browser. Pulls the reference CSV from /m18/nanogpt-reference.csv, plots the
  // WGSL train_nll/val_nll on a Canvas next to the reference trajectory, and
  // raises a pass / fail badge if the WGSL curve stays within ±0.1 nat at every
  // 100-iter checkpoint.
  //
  // Pre-flight: a numerical-gradient sanity check at small scale, comparing the
  // analytical AdamW+backward path to (L(θ+ε) − L(θ−ε))/2ε on a tiny subset of
  // parameters. M12 canonical debugging tool — protects the gate against
  // backward bugs the per-kernel CPU twins miss.

  import { onMount } from 'svelte';
  import {
    Engine, M18_CONFIG, seededInitWeights, cosineLR,
    loadTinyShakespeare, getBatch, seededRng, cpu,
    type CorpusBundle,
  } from '../../lib/m18/engine';

  const cfg = M18_CONFIG;
  const BATCH = 32;
  const T = cfg.contextLen;
  const N = BATCH * T;
  const TOTAL_ITERS = 2000;
  const WARMUP = 100;
  const LR_MAX = 3e-4;
  const LR_MIN = 3e-5;
  const WEIGHT_DECAY = 0.1;
  const GRAD_CLIP = 1.0;
  const EVAL_EVERY = 100;
  const EVAL_BATCHES = 20;
  const SEED = 1337;
  const ENVELOPE_NATS = 0.1;

  type Phase = 'idle' | 'loading' | 'compiling' | 'preflight' | 'training' | 'done' | 'error' | 'fail';

  interface RefRow { iter: number; trainNll: number; valNll: number; lr: number; }
  interface RunRow { iter: number; trainNll: number; valNll: number; }

  let phase: Phase = $state('idle');
  let progress: number = $state(0);
  let currentIter: number = $state(0);
  let runningTrain: number = $state(0);
  let runningVal: number = $state(0);
  let currentLR: number = $state(0);
  let itersPerSec: number = $state(0);
  let elapsedSec: number = $state(0);
  let errorMsg: string = $state('');
  let preflightDelta: number = $state(0);
  let preflightTolerance: number = $state(5e-2);
  let preflightReport: string = $state('');
  let lastFailIter: number | null = $state(null);
  let lastFailDelta: number | null = $state(null);

  const refRows: RefRow[] = $state([]);
  const wgslRows: RunRow[] = $state([]);
  let canvas: HTMLCanvasElement | undefined = $state();

  function parseRefCsv(text: string): RefRow[] {
    const lines = text.trim().split('\n').slice(1);
    return lines.map((l) => {
      const [it, tn, vn, lr] = l.split(',');
      return { iter: +it, trainNll: +tn, valNll: +vn, lr: +lr };
    });
  }

  function refAt(iter: number): RefRow | undefined {
    return refRows.find((r) => r.iter === iter);
  }

  function checkEnvelope(iter: number, trainNll: number, valNll: number): { ok: boolean; delta: number } {
    const r = refAt(iter); if (!r) return { ok: true, delta: 0 };
    const dT = Math.abs(trainNll - r.trainNll);
    const dV = Math.abs(valNll - r.valNll);
    const delta = Math.max(dT, dV);
    return { ok: delta <= ENVELOPE_NATS, delta };
  }

  function drawCurve(): void {
    if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    const w = canvas.width; const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // axes and frame
    const pad = { l: 38, r: 14, t: 14, b: 26 };
    const x0 = pad.l, y0 = h - pad.b;
    const x1 = w - pad.r, y1 = pad.t;
    const yMin = 1.8, yMax = 4.4;
    const xMin = 0, xMax = TOTAL_ITERS;
    const xs = (it: number) => x0 + (it - xMin) / (xMax - xMin) * (x1 - x0);
    const ys = (n: number) => y0 - (n - yMin) / (yMax - yMin) * (y0 - y1);

    ctx.strokeStyle = 'rgba(60, 65, 80, 0.18)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.stroke();
    // y-axis ticks
    ctx.fillStyle = 'rgba(60, 65, 80, 0.55)';
    ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const yv of [2, 2.5, 3, 3.5, 4]) {
      const py = ys(yv);
      ctx.fillText(yv.toFixed(1), x0 - 4, py);
      ctx.strokeStyle = 'rgba(60, 65, 80, 0.08)';
      ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x1, py); ctx.stroke();
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const xv of [0, 500, 1000, 1500, 2000]) ctx.fillText(String(xv), xs(xv), y0 + 6);

    // reference envelope (±0.1 nat band on val_nll).
    if (refRows.length) {
      ctx.fillStyle = 'rgba(70, 130, 180, 0.10)';
      ctx.beginPath();
      ctx.moveTo(xs(refRows[0].iter), ys(refRows[0].valNll - ENVELOPE_NATS));
      for (const r of refRows) ctx.lineTo(xs(r.iter), ys(r.valNll - ENVELOPE_NATS));
      for (let i = refRows.length - 1; i >= 0; i--) {
        ctx.lineTo(xs(refRows[i].iter), ys(refRows[i].valNll + ENVELOPE_NATS));
      }
      ctx.closePath();
      ctx.fill();
    }

    // reference curves
    if (refRows.length) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(70, 130, 180, 0.55)';  // val ref
      ctx.beginPath();
      refRows.forEach((r, i) => { const px = xs(r.iter); const py = ys(r.valNll); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
      ctx.stroke();
      ctx.strokeStyle = 'rgba(70, 130, 180, 0.32)'; // train ref
      ctx.beginPath();
      refRows.forEach((r, i) => { const px = xs(r.iter); const py = ys(r.trainNll); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
      ctx.stroke();
    }

    // wgsl run
    if (wgslRows.length) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#1c4f74';
      ctx.beginPath();
      wgslRows.forEach((r, i) => { const px = xs(r.iter); const py = ys(r.valNll); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
      ctx.stroke();
      ctx.strokeStyle = 'rgba(28, 79, 116, 0.55)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      wgslRows.forEach((r, i) => { const px = xs(r.iter); const py = ys(r.trainNll); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
      ctx.stroke();
    }
  }

  function ewma(prev: number | null, x: number, decay = 0.9): number {
    return prev == null ? x : decay * prev + (1 - decay) * x;
  }

  async function preflightGradCheck(engine: Engine, corpus: CorpusBundle): Promise<{ delta: number; ok: boolean; report: string }> {
    // Numerical gradient check: pick one parameter element per layer-type,
    // perturb by eps, compare (L(θ+ε) − L(θ−ε)) / (2ε) to the analytical
    // gradient. M12 canonical debugging tool — catches a bad backward kernel
    // that the per-kernel CPU twins missed.
    const rng = seededRng('preflight');
    const { x, y } = getBatch(corpus.trainIds, BATCH, T, rng);

    // 1. Snapshot all params to CPU.
    const p = engine.params;
    const snapshots = {
      wte: await engine.readTensor(p.wte),
      wQKV: await engine.readTensor(p.blocks[0].wQKV),
      wFFN1: await engine.readTensor(p.blocks[0].wFFN1),
      ln1Gamma: await engine.readTensor(p.blocks[0].ln1Gamma),
      lnFGamma: await engine.readTensor(p.lnFGamma),
    };

    // 2. Single trainStep to populate gradients (also runs AdamW; we restore after).
    await engine.trainStep(x, y, 0, 1, { beta1: 0.9, beta2: 0.95, eps: 1e-8, lambda: 0, clipNorm: 1e9 });
    // lr=0 + lambda=0 + clipNorm huge → AdamW makes only the bias-correction-related step.
    // To be safe we also restore params from the snapshot.
    engine.upload(p.wte, snapshots.wte);
    engine.upload(p.blocks[0].wQKV, snapshots.wQKV);
    engine.upload(p.blocks[0].wFFN1, snapshots.wFFN1);
    engine.upload(p.blocks[0].ln1Gamma, snapshots.ln1Gamma);
    engine.upload(p.lnFGamma, snapshots.lnFGamma);

    // 3. Read analytical gradients.
    const grads = engine.gradients(); if (!grads) throw new Error('no grads');
    const gWte = await engine.readTensor(grads.wte);
    const gQKV = await engine.readTensor(grads.blocks[0].wQKV);
    const gFFN1 = await engine.readTensor(grads.blocks[0].wFFN1);
    const gLn1Gamma = await engine.readTensor(grads.blocks[0].ln1Gamma);
    const gLnFGamma = await engine.readTensor(grads.lnFGamma);

    const eps = 1e-3;
    // Pick one element per layer-type. Stride mod array length so we don't
    // hammer the same axis position.
    const probes: { name: string; tensor: typeof p.wte; snap: Float32Array; grad: Float32Array; idx: number }[] = [
      { name: 'wte',      tensor: p.wte,                 snap: snapshots.wte,      grad: gWte,      idx: 17 * cfg.dModel + 5 },
      { name: 'wQKV[0]',  tensor: p.blocks[0].wQKV,      snap: snapshots.wQKV,     grad: gQKV,      idx: 3 * cfg.dModel + 7 },
      { name: 'wFFN1[0]', tensor: p.blocks[0].wFFN1,     snap: snapshots.wFFN1,    grad: gFFN1,     idx: 11 * cfg.dFF + 31 },
      { name: 'ln1G[0]',  tensor: p.blocks[0].ln1Gamma,  snap: snapshots.ln1Gamma, grad: gLn1Gamma, idx: 13 },
      { name: 'lnFG',     tensor: p.lnFGamma,            snap: snapshots.lnFGamma, grad: gLnFGamma, idx: 19 },
    ];

    // f32 finite-difference noise floor: a loss near 4 has ~4e-7 absolute f32
    // precision; divided by 2·ε that's a 2e-4 noise floor on the *gradient*
    // estimate. Probes whose analytical gradient is well below that floor
    // can't be checked numerically; we accept either a small relative error
    // or a small absolute error.
    const ABS_FLOOR = 1e-3; // absolute |numeric − analytic| acceptable
    const REL_TOL = 5e-2;   // relative |numeric − analytic| / max(|.|) acceptable
    let maxBadRel = 0; let report = '';
    for (const probe of probes) {
      const orig = probe.snap[probe.idx];
      const plus = new Float32Array(probe.snap); plus[probe.idx] = orig + eps;
      engine.upload(probe.tensor, plus);
      const lp = await engine.valLoss(x, y);
      const minus = new Float32Array(probe.snap); minus[probe.idx] = orig - eps;
      engine.upload(probe.tensor, minus);
      const lm = await engine.valLoss(x, y);
      engine.upload(probe.tensor, probe.snap);

      const numeric = (lp - lm) / (2 * eps);
      const analytic = probe.grad[probe.idx];
      const absDiff = Math.abs(numeric - analytic);
      const rel = absDiff / Math.max(Math.abs(analytic), Math.abs(numeric), 1e-12);
      const ok = absDiff < ABS_FLOOR || rel < REL_TOL;
      if (!ok && rel > maxBadRel) maxBadRel = rel;
      report += `${probe.name}: numeric=${numeric.toExponential(2)} analytic=${analytic.toExponential(2)} relΔ=${rel.toExponential(2)} ${ok ? 'ok' : 'FAIL'}\n`;
    }
    return { delta: maxBadRel, ok: maxBadRel === 0, report };
  }

  async function run(): Promise<void> {
    errorMsg = '';
    if (!('gpu' in navigator)) {
      phase = 'error';
      errorMsg = 'WebGPU is not available — try Chrome / Edge / Firefox 141+ on desktop.';
      return;
    }
    phase = 'loading';
    const t0 = performance.now();
    try {
      // Reference CSV.
      const csv = await fetch('/m18/nanogpt-reference.csv').then((r) => r.text());
      refRows.length = 0; refRows.push(...parseRefCsv(csv));

      // Corpus.
      const corpus = await loadTinyShakespeare();
      if (corpus.vocabSize !== cfg.vocabSize) {
        throw new Error(`vocab mismatch: corpus has ${corpus.vocabSize}, model expects ${cfg.vocabSize}`);
      }

      phase = 'compiling';
      const engine = await Engine.create(cfg);
      const weights = seededInitWeights(cfg, SEED);
      engine.loadParameters(weights);
      engine.initTraining(BATCH);

      phase = 'preflight';
      const pre = await preflightGradCheck(engine, corpus);
      preflightDelta = pre.delta;
      preflightReport = pre.report;
      if (!pre.ok) {
        phase = 'fail';
        errorMsg = `preflight gradient check failed: max relΔ = ${pre.delta.toExponential(2)} (tolerance ${preflightTolerance.toExponential(0)})\n${pre.report}`;
        drawCurve();
        return;
      }
      // Re-init training state for the actual run (preflight clobbered AdamW state).
      engine.initTraining(BATCH);

      phase = 'training';
      const trainRng = seededRng(SEED);
      const valRng = seededRng(SEED + 1);
      let runAvg: number | null = null;
      let lastEvalT = performance.now();
      let lastEvalIter = 0;

      for (let step = 0; step < TOTAL_ITERS; step++) {
        const lr = cosineLR(step, WARMUP, TOTAL_ITERS, LR_MAX, LR_MIN);
        currentLR = lr; currentIter = step;
        const { x, y } = getBatch(corpus.trainIds, BATCH, T, trainRng);
        const loss = await engine.trainStep(x, y, lr, step + 1, {
          beta1: 0.9, beta2: 0.95, eps: 1e-8, lambda: WEIGHT_DECAY, clipNorm: GRAD_CLIP,
        });
        runAvg = ewma(runAvg, loss);
        runningTrain = runAvg!;

        const isCheckpoint = step % EVAL_EVERY === 0 || step === TOTAL_ITERS - 1;
        if (isCheckpoint) {
          let valSum = 0;
          for (let b = 0; b < EVAL_BATCHES; b++) {
            const { x: vx, y: vy } = getBatch(corpus.valIds, BATCH, T, valRng);
            valSum += await engine.valLoss(vx, vy);
          }
          const valNll = valSum / EVAL_BATCHES;
          runningVal = valNll;
          wgslRows.push({ iter: step, trainNll: runAvg!, valNll });

          const env = checkEnvelope(step, runAvg!, valNll);
          if (!env.ok) {
            lastFailIter = step; lastFailDelta = env.delta;
          }

          progress = (step + 1) / TOTAL_ITERS;
          drawCurve();

          const now = performance.now();
          const itersDelta = step - lastEvalIter;
          const dt = (now - lastEvalT) / 1000;
          if (dt > 0) itersPerSec = itersDelta / dt;
          lastEvalT = now; lastEvalIter = step;
          elapsedSec = (now - t0) / 1000;

          // yield to the event loop so the canvas paints during long runs.
          await new Promise((r) => setTimeout(r, 0));
        }
      }
      phase = lastFailIter == null ? 'done' : 'fail';
      progress = 1;
    } catch (e) {
      phase = 'error';
      errorMsg = e instanceof Error ? e.message : String(e);
    }
  }

  onMount(() => { drawCurve(); });

  $effect(() => { drawCurve(); });

  const phaseLabel = $derived(({
    idle: '…', loading: 'loading reference + corpus', compiling: 'compiling shaders',
    preflight: 'preflight gradient check', training: 'training (live)', done: 'gate passed',
    error: 'error', fail: 'gate failed',
  } as const)[phase]);

  const allChecks = $derived(refRows.filter((r) => wgslRows.some((w) => w.iter === r.iter)).length);
  const passChecks = $derived(wgslRows.filter((w) => {
    const env = checkEnvelope(w.iter, w.trainNll, w.valNll); return env.ok;
  }).length);

  const fmt = (x: number, p: number) => x.toFixed(p);
  const sci = (x: number) => x.toExponential(2);
</script>

<div class="card">
  <header class="head">
    <span class="badge" class:pass={phase === 'done'} class:fail={phase === 'fail' || phase === 'error'}>
      {phaseLabel}
    </span>
    <button type="button" class="rerun" onclick={run} disabled={phase === 'training' || phase === 'compiling' || phase === 'loading' || phase === 'preflight'}>
      {phase === 'idle' ? 'start' : 're-run'}
    </button>
  </header>

  <dl class="kv">
    <dt>config</dt>
    <dd>n_layer={cfg.nLayer} · n_head={cfg.nHead} · d_model={cfg.dModel} · d_ff={cfg.dFF} · T={cfg.contextLen} · vocab={cfg.vocabSize}</dd>
    <dt>iter</dt>
    <dd>
      {currentIter} / {TOTAL_ITERS}
      <span class="muted"> · lr {sci(currentLR)} · {fmt(itersPerSec, 1)} it/s · elapsed {fmt(elapsedSec, 0)} s</span>
    </dd>
    <dt>WGSL train / val NLL</dt>
    <dd>
      {fmt(runningTrain, 4)} / {fmt(runningVal, 4)}
    </dd>
    <dt>reference at this iter</dt>
    <dd>
      {#if refAt(currentIter)}
        {fmt(refAt(currentIter)!.trainNll, 4)} / {fmt(refAt(currentIter)!.valNll, 4)}
      {:else}—{/if}
    </dd>
    <dt>checkpoints within ±0.1 nat</dt>
    <dd>
      <span class="check" class:pass={passChecks === allChecks && allChecks > 0} class:fail={passChecks < allChecks}>
        {passChecks} / {allChecks}
      </span>
      {#if lastFailIter != null}<span class="muted"> · last fail: iter {lastFailIter} (|Δ| = {fmt(lastFailDelta!, 3)})</span>{/if}
    </dd>
    <dt>preflight |relΔ|</dt>
    <dd>
      {#if preflightDelta > 0}
        <span class="check" class:pass={preflightDelta < preflightTolerance} class:fail={preflightDelta >= preflightTolerance}>
          {sci(preflightDelta)}
        </span>
        <span class="muted"> · tolerance {sci(preflightTolerance)}</span>
      {:else}—{/if}
    </dd>
  </dl>

  <canvas bind:this={canvas} width="720" height="280" class="plot"></canvas>

  {#if errorMsg}
    <p class="err">{errorMsg}</p>
  {/if}

  <p class="caption">
    Live 2,000-iter training run. Light blue = nanoGPT reference (val_nll = solid,
    train_nll = faded); the band around it is the ±0.1-nat envelope. Dark blue
    = the WGSL engine's curve. The gate passes when every WGSL checkpoint
    sits inside the band. If it fails despite the per-kernel CPU twins all
    passing, do <em>not</em> silently switch runtimes — flag it and stop.
  </p>
</div>

<style>
  .card {
    display: flex; flex-direction: column; gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(0.85rem, 2vw, 1.4rem);
    color: var(--site-fg);
    box-shadow: 0 1px 0 rgba(0,0,0,0.04),
                0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 30%, transparent);
  }
  .head { display: flex; align-items: center; gap: 0.6rem; justify-content: space-between; }
  .badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    border-radius: 999px; padding: 0.3rem 0.85rem;
    font-family: var(--font-mono); font-size: 0.78rem;
    background: color-mix(in srgb, var(--site-fg) 8%, transparent);
    color: var(--site-fg);
  }
  .badge.pass { background: color-mix(in srgb, var(--ink-sea) 18%, transparent); color: var(--ink-sea); }
  .badge.fail { background: color-mix(in srgb, var(--ink-red) 18%, transparent); color: var(--ink-red); }
  .rerun {
    cursor: pointer;
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.3rem 0.9rem; font-family: inherit;
    font-size: 0.82rem; font-weight: 600;
    transition: background 160ms ease, border-color 160ms ease;
  }
  .rerun:hover:not(:disabled) { border-color: var(--site-fg); }
  .rerun:disabled { opacity: 0.5; cursor: default; }

  .kv {
    display: grid; grid-template-columns: max-content 1fr; gap: 0.45rem 1rem;
    margin: 0; font-size: 0.92rem;
  }
  .kv dt {
    font-family: var(--font-mono); font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--site-fg-muted);
    padding-top: 0.15rem;
  }
  .kv dd { margin: 0; font-variant-numeric: tabular-nums; }

  .check { font-family: var(--font-mono); font-weight: 700; }
  .check.pass { color: var(--ink-sea); }
  .check.fail { color: var(--ink-red); }
  .muted { color: var(--site-fg-muted); }

  .plot {
    width: 100%; height: auto; max-width: 720px;
    background: color-mix(in srgb, var(--site-bg) 98%, var(--site-fg) 2%);
    border: 1px solid var(--demo-card-border);
    border-radius: 12px;
  }

  .err {
    margin: 0; padding: 0.7rem 0.9rem;
    border: 1px solid color-mix(in srgb, var(--ink-red) 35%, transparent);
    background: color-mix(in srgb, var(--ink-red) 8%, transparent);
    border-radius: 10px;
    color: var(--ink-red); font-size: 0.88rem;
  }

  .caption { margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55; }
</style>
