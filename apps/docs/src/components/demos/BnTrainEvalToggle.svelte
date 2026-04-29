<script lang="ts">
  interface Props {
    /** True feature distribution N(μ_true, σ_true). */
    muTrue?: number;
    sigmaTrue?: number;
    /** Test-point feature value to normalize. */
    testValue?: number;
    /** EMA momentum for the running stats. */
    alpha?: number;
  }

  let {
    muTrue = 4.5,
    sigmaTrue = 1.4,
    testValue = 7.0,
    alpha = 0.1,
  }: Props = $props();

  const EPS = 1e-5;

  let mode: 'train' | 'eval' = $state('train');
  let batchSize: number = $state(8);
  let batch: number[] = $state(sampleBatch(8));
  let muRun: number = $state(muTrue);
  let varRun: number = $state(sigmaTrue * sigmaTrue);
  let stepsTaken: number = $state(0);

  function sampleBatch(n: number): number[] {
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      // Box–Muller
      const u = Math.random() || 1e-9;
      const v = Math.random() || 1e-9;
      out[i] = muTrue + sigmaTrue * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    return out;
  }

  // Recompute the batch whenever the size changes.
  let lastBatchSize = batchSize;
  $effect(() => {
    if (batchSize !== lastBatchSize) {
      lastBatchSize = batchSize;
      batch = sampleBatch(batchSize);
    }
  });

  const muBatch = $derived.by(() => {
    if (batch.length === 0) return 0;
    return batch.reduce((a, b) => a + b, 0) / batch.length;
  });
  const varBatch = $derived.by(() => {
    if (batch.length === 0) return 0;
    const m = muBatch;
    return batch.reduce((s, x) => s + (x - m) * (x - m), 0) / batch.length;
  });

  // Normalized values
  const xhatTrain = $derived((testValue - muBatch) / Math.sqrt(varBatch + EPS));
  const xhatEval = $derived((testValue - muRun) / Math.sqrt(varRun + EPS));
  const xhatActive = $derived(mode === 'train' ? xhatTrain : xhatEval);

  function trainingStep(): void {
    // Apply EMA update of running stats from the current batch, then resample.
    muRun = (1 - alpha) * muRun + alpha * muBatch;
    varRun = (1 - alpha) * varRun + alpha * varBatch;
    stepsTaken += 1;
    batch = sampleBatch(batchSize);
  }

  function resetRunning(): void {
    muRun = 0;
    varRun = 1;
    stepsTaken = 0;
    batch = sampleBatch(batchSize);
  }

  function reseedBatch(): void {
    batch = sampleBatch(batchSize);
  }

  // Number-line geometry
  const X_MIN = 0;
  const X_MAX = 10;
  function xToPct(x: number): number {
    return Math.max(0, Math.min(100, ((x - X_MIN) / (X_MAX - X_MIN)) * 100));
  }

  const fmt = (n: number, d = 3) => {
    if (!Number.isFinite(n)) return '∞';
    if (Math.abs(n) > 1e4) return n.toExponential(2);
    return n.toFixed(d);
  };

  const collapsed = $derived(
    mode === 'train' && batchSize === 1 && Math.abs(xhatTrain) < 1e-2,
  );
</script>

<div class="widget">
  <div class="seg">
    <button
      type="button"
      class="seg-btn"
      class:active={mode === 'train'}
      onclick={() => (mode = 'train')}
    >train mode</button>
    <button
      type="button"
      class="seg-btn"
      class:active={mode === 'eval'}
      onclick={() => (mode = 'eval')}
    >eval mode</button>
  </div>

  <div class="strip">
    <div class="track" aria-hidden="true">
      {#each [0, 2, 4, 6, 8, 10] as tick}
        <div class="tick" style="left:{xToPct(tick)}%">
          <span class="tick-label">{tick}</span>
        </div>
      {/each}
      <!-- batch points -->
      {#each batch as b}
        <div class="dot dot-batch" style="left:{xToPct(b)}%" title={`batch sample = ${b.toFixed(2)}`}></div>
      {/each}
      <!-- batch mean marker -->
      <div class="marker marker-batch" style="left:{xToPct(muBatch)}%" title="μ_B">
        <span class="marker-label">μ_B</span>
      </div>
      <!-- running mean marker -->
      <div class="marker marker-run" style="left:{xToPct(muRun)}%" title="μ_run">
        <span class="marker-label">μ_run</span>
      </div>
      <!-- test point -->
      <div class="dot dot-test" style="left:{xToPct(testValue)}%" title={`test point x = ${testValue}`}>
        <span class="dot-label">x = {testValue}</span>
      </div>
    </div>
  </div>

  <div class="readouts">
    <div class="rd-col rd-batch" class:active={mode === 'train'}>
      <h4>From this batch</h4>
      <dl>
        <div class="rd-row"><dt>μ_B</dt><dd>{fmt(muBatch)}</dd></div>
        <div class="rd-row"><dt>σ²_B</dt><dd>{fmt(varBatch)}</dd></div>
        <div class="rd-row hi" class:collapsed>
          <dt>x̂ (train)</dt>
          <dd>{fmt(xhatTrain)}</dd>
        </div>
      </dl>
    </div>
    <div class="rd-col rd-run" class:active={mode === 'eval'}>
      <h4>Running stats (steps: {stepsTaken})</h4>
      <dl>
        <div class="rd-row"><dt>μ_run</dt><dd>{fmt(muRun)}</dd></div>
        <div class="rd-row"><dt>σ²_run</dt><dd>{fmt(varRun)}</dd></div>
        <div class="rd-row hi">
          <dt>x̂ (eval)</dt>
          <dd>{fmt(xhatEval)}</dd>
        </div>
      </dl>
    </div>
  </div>

  {#if collapsed}
    <p class="warn">
      <strong>Collapse.</strong>
      Batch size 1 in train mode: μ_B equals the single example, σ²_B is exactly 0.
      x̂ = (x − x)/√ε ≈ 0. The signal is gone.
    </p>
  {/if}

  <div class="controls">
    <label class="slider">
      <span class="slider-label">batch size = <strong>{batchSize}</strong></span>
      <input type="range" min="1" max="32" step="1" bind:value={batchSize} aria-label="Batch size" />
    </label>
    <div class="ctl-row">
      <button type="button" class="btn btn-primary" onclick={trainingStep}>training step</button>
      <button type="button" class="btn" onclick={reseedBatch}>resample batch</button>
      <button type="button" class="btn btn-ghost" onclick={resetRunning}>reset running stats</button>
    </div>
  </div>

  <p class="caption">
    The batch sits on the number line (small dots). <span class="lg-batch">μ_B</span>
    moves with each resample; <span class="lg-run">μ_run</span> moves only when
    you press <em>training step</em>. The active panel — train or eval — is
    what BatchNorm uses depending on which mode the framework is in. Drop the
    batch size to 1 in train mode and the output collapses to zero. Switch to
    eval and the running stats save you.
  </p>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(0.85rem, 2vw, 1.4rem);
    color: var(--site-fg);
    box-shadow:
      0 1px 0 rgba(0,0,0,0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 50%, transparent);
  }

  .seg {
    align-self: flex-start;
    display: inline-flex; gap: 0;
    background: color-mix(in srgb, var(--site-fg) 8%, transparent);
    border-radius: 999px; padding: 3px;
  }
  .seg-btn {
    border: none; background: transparent; color: var(--site-fg-muted);
    padding: 0.32rem 0.85rem; border-radius: 999px; cursor: pointer;
    font-size: 0.83rem; font-weight: 600;
    transition: background 160ms ease, color 160ms ease;
  }
  .seg-btn.active {
    background: var(--ink-sea); color: #fdfdfc;
  }

  .strip {
    background: var(--demo-stage); border-radius: 12px; padding: 1.5rem 1rem 2.4rem;
  }
  .track {
    position: relative;
    height: 60px;
    border-bottom: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
  }
  .tick {
    position: absolute; bottom: -1px;
    width: 1px; height: 6px;
    background: color-mix(in srgb, var(--site-fg) 24%, transparent);
    transform: translateX(-50%);
  }
  .tick-label {
    position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
    font-family: var(--font-mono); font-size: 0.7rem; color: var(--site-fg-muted);
  }
  .dot {
    position: absolute; bottom: 6px;
    transform: translateX(-50%);
    width: 8px; height: 8px;
    border-radius: 50%;
  }
  .dot-batch {
    background: color-mix(in srgb, var(--ink-sea) 75%, transparent);
    width: 7px; height: 7px;
  }
  .dot-test {
    background: var(--ink-red);
    width: 12px; height: 12px;
    bottom: 4px;
    border: 2px solid var(--demo-stage);
  }
  .dot-test .dot-label {
    position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-red);
    white-space: nowrap;
  }
  .marker {
    position: absolute; bottom: -16px; transform: translateX(-50%);
    font-family: var(--font-mono); font-size: 0.7rem;
    width: 1px; height: 18px;
  }
  .marker-batch { background: var(--ink-sea); }
  .marker-batch .marker-label { color: var(--ink-sea); }
  .marker-run { background: var(--site-fg-muted); }
  .marker-run .marker-label { color: var(--site-fg-muted); }
  .marker-label {
    position: absolute; top: 100%; left: 50%; transform: translateX(-50%);
    margin-top: 1px; white-space: nowrap;
  }

  .readouts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }
  @media (max-width: 540px) { .readouts { grid-template-columns: 1fr; } }
  .rd-col {
    padding: 0.55rem 0.7rem;
    border-radius: 10px;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border: 1px solid color-mix(in srgb, var(--site-fg) 10%, transparent);
    transition: border-color 160ms ease, background 160ms ease;
  }
  .rd-col.active {
    border-color: var(--ink-sea);
    background: color-mix(in srgb, var(--ink-sea) 6%, transparent);
  }
  .rd-col h4 {
    margin: 0 0 0.35rem;
    font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--site-fg-muted);
    font-weight: 600;
  }
  .rd-col dl { margin: 0; display: flex; flex-direction: column; gap: 0.2rem; }
  .rd-row { display: flex; justify-content: space-between; align-items: baseline;
    font-family: var(--font-mono); font-size: 0.85rem;
  }
  .rd-row dt { color: var(--site-fg-muted); }
  .rd-row dd { margin: 0; color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .rd-row.hi { padding-top: 0.3rem; border-top: 1px dashed color-mix(in srgb, var(--site-fg) 22%, transparent); margin-top: 0.15rem; }
  .rd-row.hi dt { color: var(--site-fg); font-weight: 600; }
  .rd-row.hi.collapsed dd { color: var(--ink-coral); }

  .warn {
    margin: 0; padding: 0.5rem 0.8rem;
    background: color-mix(in srgb, var(--ink-coral) 10%, transparent);
    border-radius: 8px;
    font-size: 0.85rem; color: var(--site-fg);
    line-height: 1.5;
  }
  .warn strong { color: var(--ink-coral); margin-right: 0.35rem; }

  .controls { display: flex; flex-direction: column; gap: 0.5rem; }
  .ctl-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.35rem 0.85rem;
    font-size: 0.83rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover { transform: translateY(-1px); border-color: var(--site-fg); }
  .btn-primary { background: var(--ink-sea); color: #fdfdfc; border-color: var(--ink-sea); }
  .btn-ghost { color: var(--site-fg-muted); font-weight: 500; }

  .slider { display: flex; flex-direction: column; gap: 0.2rem; }
  .slider-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted); }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input[type='range'] { width: 100%; accent-color: var(--ink-sea); }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg); font-style: normal;
    font-family: var(--font-mono); font-size: 0.85em;
  }
  .lg-batch { color: var(--ink-sea); font-family: var(--font-mono); font-size: 0.85em; }
  .lg-run { color: var(--site-fg); font-family: var(--font-mono); font-size: 0.85em; }
</style>
