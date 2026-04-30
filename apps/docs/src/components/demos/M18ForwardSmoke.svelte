<script lang="ts">
  // M18 forward smoke — a /dev/-only widget that runs ONE forward pass through
  // the WGSL engine on a hand-crafted batch, runs the same forward through the
  // CPU twin, and reports:
  //   • WebGPU availability + adapter name
  //   • logit shape (B·T, V)
  //   • softmax-sums-to-1 invariant on each row
  //   • max-abs WGSL-vs-CPU drift
  //
  // It is the "Slice 2 didn't break the math" assertion, made tangible.

  import { onMount } from 'svelte';
  import { Engine, M18_CONFIG, debugInitWeights, cpuForward, cpu } from '../../lib/m18/engine';

  const cfg = M18_CONFIG;
  const BATCH = 2;
  const N = BATCH * cfg.contextLen;
  const V = cfg.vocabSize;

  type Status = 'idle' | 'compiling' | 'forwarding' | 'comparing' | 'done' | 'error';

  let status: Status = $state('idle');
  let adapter: string = $state('');
  let dispatchCount: number = $state(0);
  let logitShape: string = $state('');
  let softmaxRowsChecked: number = $state(0);
  let softmaxRowsOk: number = $state(0);
  let softmaxMaxAbsErr: number = $state(0);
  let wgslVsCpuMax: number = $state(0);
  let wgslVsCpuMean: number = $state(0);
  let elapsedMs: number = $state(0);
  let errorMsg: string = $state('');

  function buildBatch(): { tokens: Int32Array; weights: ReturnType<typeof debugInitWeights> } {
    const tokens = new Int32Array(N);
    // A deterministic, structured batch: stride 7 mod V across both batches.
    for (let i = 0; i < N; i++) tokens[i] = (i * 7 + 3) % V;
    const weights = debugInitWeights(cfg);
    return { tokens, weights };
  }

  async function run(): Promise<void> {
    errorMsg = '';
    if (!('gpu' in navigator)) {
      status = 'error';
      errorMsg = 'WebGPU is not available in this browser. Try Chrome / Edge / Firefox 141+ on desktop.';
      return;
    }
    status = 'compiling';
    const t0 = performance.now();
    try {
      const adapterRaw = await navigator.gpu.requestAdapter();
      if (!adapterRaw) throw new Error('No WebGPU adapter.');
      const info = (adapterRaw as unknown as { info?: { vendor?: string; architecture?: string } }).info;
      adapter = [info?.vendor, info?.architecture].filter(Boolean).join(' / ') || 'WebGPU adapter';
      const device = await adapterRaw.requestDevice();
      const engine = Engine.fromDevice(device, cfg);

      const { tokens, weights } = buildBatch();
      engine.loadParameters(weights);
      // Forward dispatch budget per the plan §2: 10 ops × 4 blocks + 3 boundary ops = 43.
      dispatchCount = 10 * cfg.nLayer + 3;

      status = 'forwarding';
      const logits = await engine.forward(tokens, BATCH);
      logitShape = `${N} × ${V}`;

      // Softmax invariant per row.
      status = 'comparing';
      const probs = cpu.softmax(logits, N, V);
      let okRows = 0; let maxAbsErr = 0;
      for (let r = 0; r < N; r++) {
        let s = 0; for (let c = 0; c < V; c++) s += probs[r * V + c];
        const e = Math.abs(s - 1);
        if (e > maxAbsErr) maxAbsErr = e;
        if (e < 1e-3) okRows++;
      }
      softmaxRowsChecked = N;
      softmaxRowsOk = okRows;
      softmaxMaxAbsErr = maxAbsErr;

      // WGSL vs CPU twin max-abs diff over the full logits tensor.
      const cpuLogits = cpuForward(weights, tokens, cfg, BATCH);
      let mx = 0; let sum = 0;
      for (let i = 0; i < logits.length; i++) {
        const e = Math.abs(logits[i] - cpuLogits[i]);
        if (e > mx) mx = e; sum += e;
      }
      wgslVsCpuMax = mx;
      wgslVsCpuMean = sum / logits.length;

      elapsedMs = performance.now() - t0;
      status = 'done';
    } catch (e) {
      status = 'error';
      errorMsg = e instanceof Error ? e.message : String(e);
    }
  }

  onMount(() => { run(); });

  const passSoftmax = $derived(status === 'done' && softmaxRowsOk === softmaxRowsChecked);
  const passDrift = $derived(status === 'done' && wgslVsCpuMax < 1e-3);
  const allPass = $derived(passSoftmax && passDrift);
  const fmt = (x: number, p: number) => x.toFixed(p);
  const sci = (x: number) => x.toExponential(2);
</script>

<div class="card">
  <header class="head">
    <span class="badge" class:pass={allPass} class:fail={status === 'error'}>
      {#if status === 'idle'}…{/if}
      {#if status === 'compiling'}compiling shaders{/if}
      {#if status === 'forwarding'}running forward{/if}
      {#if status === 'comparing'}comparing vs cpu{/if}
      {#if status === 'done' && allPass}all checks passed{/if}
      {#if status === 'done' && !allPass}drift detected{/if}
      {#if status === 'error'}error{/if}
    </span>
    <button type="button" class="rerun" onclick={run} disabled={status !== 'done' && status !== 'error'}>re-run</button>
  </header>

  <dl class="kv">
    <dt>config</dt>
    <dd>n_layer={cfg.nLayer} · n_head={cfg.nHead} · d_model={cfg.dModel} · d_ff={cfg.dFF} · T={cfg.contextLen} · vocab={cfg.vocabSize}</dd>

    <dt>adapter</dt>
    <dd>{adapter || (status === 'error' ? '—' : 'detecting…')}</dd>

    <dt>dispatches / forward</dt>
    <dd>{dispatchCount || '—'} <span class="muted">(10 ops × {cfg.nLayer} blocks + 3 boundary)</span></dd>

    <dt>logit shape</dt>
    <dd>{logitShape || '—'}</dd>

    <dt>softmax row sum</dt>
    <dd>
      {#if status === 'done'}
        <span class="check" class:pass={passSoftmax} class:fail={!passSoftmax}>
          {softmaxRowsOk} / {softmaxRowsChecked} rows within 1e-3
        </span>
        <span class="muted"> · max |Σp − 1| = {sci(softmaxMaxAbsErr)}</span>
      {:else}—{/if}
    </dd>

    <dt>WGSL vs CPU twin</dt>
    <dd>
      {#if status === 'done'}
        <span class="check" class:pass={passDrift} class:fail={!passDrift}>
          max |Δ| = {sci(wgslVsCpuMax)}
        </span>
        <span class="muted"> · mean |Δ| = {sci(wgslVsCpuMean)}</span>
      {:else}—{/if}
    </dd>

    <dt>elapsed</dt>
    <dd>{status === 'done' ? `${fmt(elapsedMs, 0)} ms` : '—'}</dd>
  </dl>

  {#if errorMsg}
    <p class="err">{errorMsg}</p>
  {/if}

  <p class="caption">
    One forward pass through the M18 engine (apps/docs/src/lib/m18/engine).
    The hand-crafted batch is deterministic — re-running should reproduce the
    same logits to floating-point. The CPU twin runs the same math in pure TS
    and the two are compared elementwise. Slice 2 ships forward only; backward
    + training arrive in Slice 3.
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

  .err {
    margin: 0; padding: 0.7rem 0.9rem;
    border: 1px solid color-mix(in srgb, var(--ink-red) 35%, transparent);
    background: color-mix(in srgb, var(--ink-red) 8%, transparent);
    border-radius: 10px;
    color: var(--ink-red); font-size: 0.88rem;
  }

  .caption { margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55; }
</style>
