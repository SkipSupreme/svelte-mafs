<script lang="ts">
  import { onDestroy } from 'svelte';

  interface Props {
    /** Words to count bigrams from. Closed corpus — not editable here. */
    corpus?: string[];
    /** Add-k smoothing strength on the count panel. */
    initialSmoothing?: number;
    /** L2 weight on the SGD panel. */
    initialLambda?: number;
    /** Learning rate for the SGD panel (full-batch step). */
    learningRate?: number;
    /** Show the smoothing/L2 control row. */
    showRegControls?: boolean;
  }

  let {
    corpus = DEFAULT_CORPUS(),
    initialSmoothing = 1,
    initialLambda = 0,
    learningRate = 30,
    showRegControls = true,
  }: Props = $props();

  function DEFAULT_CORPUS(): string[] {
    return [
      'emma','olivia','ava','isabella','sophia','charlotte','amelia','mia','harper',
      'evelyn','abigail','emily','ella','elizabeth','sofia','avery','scarlett','grace',
      'chloe','victoria','riley','aria','lily','aubrey','zoey','penelope','lillian',
      'addison','layla','natalie','camila','hannah','brooklyn','zoe','nora','leah',
      'savannah','audrey','claire','eleanor','skylar','ellie','samantha','stella',
      'paisley','violet','mila','allison','alexa','anna',
    ];
  }

  const VOCAB: string[] = ['.', ...'abcdefghijklmnopqrstuvwxyz'.split('')];
  const V = VOCAB.length; // 27
  const INDEX: Record<string, number> = Object.fromEntries(
    VOCAB.map((c, i) => [c, i]),
  );
  const display = (c: string) => (c === '.' ? '·' : c);

  // Counts (frozen for the corpus prop) — recomputed if corpus changes.
  const counts: number[][] = $derived.by(() => {
    const N: number[][] = Array.from({ length: V }, () => Array(V).fill(0));
    for (const w of corpus) {
      const cleaned = w.toLowerCase().replace(/[^a-z]/g, '');
      if (!cleaned) continue;
      const seq = '.' + cleaned + '.';
      for (let t = 0; t < seq.length - 1; t++) {
        const i = INDEX[seq[t]];
        const j = INDEX[seq[t + 1]];
        if (i != null && j != null) N[i][j] += 1;
      }
    }
    return N;
  });
  const rowSums: number[] = $derived(
    counts.map((r) => r.reduce((a, b) => a + b, 0)),
  );
  const totalPairs: number = $derived(
    rowSums.reduce((a, b) => a + b, 0),
  );

  // === SGD-trained one-layer NN: W ∈ R^{V×V}, logits = W[i, :] for input i. ===
  // Stored as flat Float64Array for performance.
  let smoothing: number = $state(initialSmoothing);
  let lambda: number = $state(initialLambda);
  let W: number[] = $state(initW());
  let steps: number = $state(0);
  let lastLoss: number = $state(NaN);
  let auto: boolean = $state(false);
  let rafId: number | null = null;

  function initW(): number[] {
    const arr = new Array(V * V);
    for (let k = 0; k < V * V; k++) {
      // Box–Muller, σ=0.01, so initial softmax(W) ≈ uniform.
      const u = Math.random() || 1e-9;
      const v = Math.random() || 1e-9;
      arr[k] = 0.01 * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    return arr;
  }

  // softmax(W) per row → V×V probabilities. Stable max-subtraction.
  const sgdProbs: number[][] = $derived.by(() => {
    const out: number[][] = Array.from({ length: V }, () => Array(V).fill(0));
    for (let i = 0; i < V; i++) {
      let m = -Infinity;
      for (let j = 0; j < V; j++) if (W[i * V + j] > m) m = W[i * V + j];
      let s = 0;
      for (let j = 0; j < V; j++) {
        const e = Math.exp(W[i * V + j] - m);
        out[i][j] = e;
        s += e;
      }
      for (let j = 0; j < V; j++) out[i][j] /= s;
    }
    return out;
  });

  // Add-k smoothed count probabilities (the closed-form reference).
  // Guard against the degenerate empty-row + s=0 case → renders as zero,
  // matching the "log P = -∞" reality the smoothing knob is meant to fix.
  const countProbs: number[][] = $derived.by(() =>
    counts.map((row, i) => {
      const denom = rowSums[i] + V * smoothing;
      if (denom === 0) return row.map(() => 0);
      return row.map((c) => (c + smoothing) / denom);
    }),
  );

  // Mean absolute deviation between the two panels — convergence readout.
  const mad: number = $derived.by(() => {
    let s = 0;
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) s += Math.abs(countProbs[i][j] - sgdProbs[i][j]);
    }
    return s / (V * V);
  });

  // Loss of the SGD model on the full bigram dataset (per-pair NLL).
  // L = (1/N) Σ_i Σ_j N_ij · (-log p_ij) + (λ/2) ||W||²
  function computeLoss(): number {
    let nll = 0;
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) {
        if (counts[i][j] > 0) {
          // softmax(W)[i,j] guaranteed > 0 by construction.
          nll -= counts[i][j] * Math.log(sgdProbs[i][j]);
        }
      }
    }
    let reg = 0;
    if (lambda > 0) {
      for (let k = 0; k < V * V; k++) reg += W[k] * W[k];
      reg *= lambda / 2;
    }
    return totalPairs > 0 ? nll / totalPairs + reg : NaN;
  }

  // One full-batch GD step. Closed-form gradient for one-hot input + softmax NLL:
  //   ∂L/∂W[i,j] = (1/N) (rowSum[i] · p_ij − N_ij) + λ · W[i,j]
  function trainStep(): void {
    if (totalPairs === 0) return;
    const lrOverN = learningRate / totalPairs;
    const newW = W.slice();
    for (let i = 0; i < V; i++) {
      const rs = rowSums[i];
      for (let j = 0; j < V; j++) {
        const k = i * V + j;
        const grad = rs * sgdProbs[i][j] - counts[i][j];
        newW[k] -= lrOverN * grad;
        if (lambda > 0) newW[k] -= learningRate * lambda * W[k];
      }
    }
    W = newW;
    steps += 1;
    lastLoss = computeLoss();
  }

  function runMany(n: number): void {
    for (let k = 0; k < n; k++) trainStep();
  }

  function reset(): void {
    stopAuto();
    W = initW();
    steps = 0;
    lastLoss = NaN;
  }

  function startAuto(): void {
    if (auto) return;
    auto = true;
    const tick = () => {
      if (!auto) return;
      // ~6 steps per frame keeps animation watchable; convergence in ~5 seconds.
      runMany(6);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  function stopAuto(): void {
    auto = false;
    if (rafId != null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  }

  onDestroy(() => stopAuto());

  // Initial loss readout (computes once on mount).
  $effect(() => {
    if (Number.isNaN(lastLoss)) lastLoss = computeLoss();
  });

  // Color helpers — both panels share the same scale so visual comparison is honest.
  const maxProb: number = $derived.by(() => {
    let m = 0;
    for (let i = 0; i < V; i++) {
      for (let j = 0; j < V; j++) {
        if (countProbs[i][j] > m) m = countProbs[i][j];
      }
    }
    return m || 1;
  });
  function colorFor(p: number): string {
    if (p <= 0) return 'var(--cell-empty)';
    const t = Math.sqrt(p / maxProb); // sqrt lifts small values into view
    const pct = Math.max(2, Math.min(100, Math.round(t * 100)));
    return `color-mix(in srgb, var(--ink-red) ${pct}%, var(--cell-empty))`;
  }

  // Sample one name from the SGD model, just for fun.
  function sampleName(): string {
    let cur = 0; // start token '.'
    let out = '';
    for (let step = 0; step < 30; step++) {
      const p = sgdProbs[cur];
      const u = Math.random();
      let acc = 0;
      let pick = 0;
      for (let j = 0; j < V; j++) {
        acc += p[j];
        if (u <= acc) { pick = j; break; }
      }
      if (pick === 0) break;
      out += VOCAB[pick];
      cur = pick;
    }
    return out || '·';
  }
  let sampledName: string = $state('');
  function newSample(): void {
    sampledName = sampleName();
  }

  const fmt = (x: number, d = 3) =>
    Number.isNaN(x) ? '—' : x.toFixed(d);
  const fmtSmall = (x: number) =>
    x === 0 ? '0' : x < 0.001 ? x.toExponential(1) : x.toFixed(3);
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">corpus</span>
      <span class="meta-val">{corpus.length} words · {totalPairs} bigrams</span>
    </div>
    <div class="meta">
      <span class="meta-key">steps</span>
      <span class="meta-val">{steps}</span>
    </div>
    <div class="meta">
      <span class="meta-key">loss (SGD)</span>
      <span class="meta-val">{fmt(lastLoss, 3)}</span>
    </div>
    <div class="meta meta-converge" class:converged={mad < 0.005}>
      <span class="meta-key">|count − sgd|</span>
      <span class="meta-val">{fmt(mad, 4)}</span>
    </div>
  </header>

  <div class="panels">
    <section class="panel">
      <div class="panel-head">
        <span class="panel-title">Counts (smoothed)</span>
        <span class="panel-formula">P(j | i) = (N<sub>ij</sub> + s) / (Σ N<sub>i·</sub> + V·s)</span>
      </div>
      <div class="grid" role="figure" aria-label="Counts panel: smoothed bigram probabilities.">
        <div class="row col-headers" aria-hidden="true">
          <div class="corner" />
          {#each VOCAB as c}
            <div class="col-label">{display(c)}</div>
          {/each}
        </div>
        {#each VOCAB as rowChar, i}
          <div class="row">
            <div class="row-label">{display(rowChar)}</div>
            {#each VOCAB as _c, j}
              <div
                class="cell"
                style="background:{colorFor(countProbs[i][j])};"
                title={`${display(rowChar)} → ${display(VOCAB[j])}: ${fmtSmall(countProbs[i][j])}`}
              ></div>
            {/each}
          </div>
        {/each}
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <span class="panel-title">SGD on softmax(W)</span>
        <span class="panel-formula">P(j | i) = softmax(W[i,·])<sub>j</sub></span>
      </div>
      <div class="grid" role="figure" aria-label="SGD panel: softmax of trained weight matrix.">
        <div class="row col-headers" aria-hidden="true">
          <div class="corner" />
          {#each VOCAB as c}
            <div class="col-label">{display(c)}</div>
          {/each}
        </div>
        {#each VOCAB as rowChar, i}
          <div class="row">
            <div class="row-label">{display(rowChar)}</div>
            {#each VOCAB as _c, j}
              <div
                class="cell"
                style="background:{colorFor(sgdProbs[i][j])};"
                title={`${display(rowChar)} → ${display(VOCAB[j])}: ${fmtSmall(sgdProbs[i][j])}`}
              ></div>
            {/each}
          </div>
        {/each}
      </div>
    </section>
  </div>

  <div class="controls">
    <div class="ctl-row">
      <button type="button" class="btn btn-primary" onclick={trainStep} disabled={auto}>Step</button>
      <button type="button" class="btn" onclick={() => runMany(50)} disabled={auto}>Run 50</button>
      <button type="button" class="btn" onclick={() => runMany(500)} disabled={auto}>Run 500</button>
      <button
        type="button"
        class="btn btn-toggle"
        class:active={auto}
        onclick={() => (auto ? stopAuto() : startAuto())}
      >{auto ? 'Stop' : 'Auto'}</button>
      <button type="button" class="btn btn-ghost" onclick={reset}>Reset W</button>
    </div>

    {#if showRegControls}
      <div class="ctl-row sliders">
        <label class="slider">
          <span class="slider-label">smoothing s = <strong>{smoothing.toFixed(2)}</strong></span>
          <input
            type="range"
            min="0"
            max="20"
            step="0.5"
            bind:value={smoothing}
            aria-label="Add-k smoothing"
          />
        </label>
        <label class="slider">
          <span class="slider-label">L2 λ = <strong>{lambda.toFixed(3)}</strong></span>
          <input
            type="range"
            min="0"
            max="0.1"
            step="0.001"
            bind:value={lambda}
            aria-label="L2 regularization on SGD weights"
          />
        </label>
      </div>
    {/if}

    <div class="ctl-row sample">
      <button type="button" class="btn btn-sample" onclick={newSample}>Sample a name</button>
      <span class="sample-out" aria-live="polite">
        {#if sampledName}
          <code>{sampledName}</code>
        {:else}
          <span class="muted">click "sample" to draw from the SGD model →</span>
        {/if}
      </span>
    </div>
  </div>

  <p class="caption">
    Both panels are the same model after training, expressed two ways. The left
    panel computes probabilities from raw counts (with add-<em>s</em> smoothing).
    The right panel maintains weights <em>W</em> trained by gradient descent on
    NLL. Crank smoothing and L2 in tandem and watch them stay aligned —
    smoothing on the count side <em>is</em> L2 regularization on the SGD side.
  </p>
</div>

<style>
  .widget {
    --cell-size: clamp(9px, 1.4vw, 13px);
    --cell-empty: color-mix(in srgb, var(--site-fg) 6%, transparent);
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(0.85rem, 2vw, 1.4rem);
    color: var(--site-fg);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 50%, transparent);
  }

  .head {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.1rem;
    align-items: baseline;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
  .meta {
    display: inline-flex;
    align-items: baseline;
    gap: 0.4rem;
  }
  .meta-key {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
  }
  .meta-val {
    color: var(--site-fg);
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }
  .meta-converge.converged {
    color: var(--cta);
  }
  .meta-converge.converged .meta-val {
    color: var(--cta);
  }

  .panels {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.7rem;
  }
  @media (max-width: 640px) {
    .panels {
      grid-template-columns: 1fr;
    }
  }

  .panel {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    background: var(--demo-stage);
    border-radius: 12px;
    padding: 0.55rem 0.6rem 0.7rem;
    overflow-x: auto;
  }
  .panel-head {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
  }
  .panel-title {
    font-family: var(--font-display, Fraunces, Georgia, serif);
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--site-fg);
  }
  .panel-formula {
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.7rem;
    color: var(--site-fg-muted);
  }

  .grid {
    display: flex;
    flex-direction: column;
    gap: 1px;
    user-select: none;
    -webkit-user-select: none;
  }
  .row {
    display: grid;
    grid-template-columns: var(--cell-size) repeat(27, var(--cell-size));
    gap: 1px;
  }
  .corner,
  .col-label,
  .row-label {
    width: var(--cell-size);
    height: var(--cell-size);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.55rem;
    color: var(--site-fg-muted);
  }
  .cell {
    width: var(--cell-size);
    height: var(--cell-size);
    border-radius: 1px;
    transition: background 200ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }

  .controls {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .ctl-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
  }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent;
    color: var(--site-fg);
    border-radius: 999px;
    padding: 0.35rem 0.85rem;
    font-size: 0.83rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover:not(:disabled) {
    transform: translateY(-1px);
    border-color: var(--site-fg);
  }
  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .btn-primary {
    background: var(--ink-red);
    color: #fdfdfc;
    border-color: var(--ink-red);
  }
  .btn-primary:hover:not(:disabled) {
    background: color-mix(in srgb, var(--ink-red) 88%, black);
  }
  .btn-toggle.active {
    background: var(--cta);
    color: var(--cta-fg);
    border-color: var(--cta);
  }
  .btn-ghost {
    color: var(--site-fg-muted);
    font-weight: 500;
  }
  .btn-sample {
    background: color-mix(in srgb, var(--ink-sea) 14%, transparent);
    border-color: color-mix(in srgb, var(--ink-sea) 35%, transparent);
    color: var(--ink-sea);
  }

  .sliders {
    gap: 1rem;
    padding: 0.4rem 0;
    border-top: 1px dashed color-mix(in srgb, var(--site-fg) 14%, transparent);
    border-bottom: 1px dashed color-mix(in srgb, var(--site-fg) 14%, transparent);
  }
  .slider {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    flex: 1 1 220px;
  }
  .slider-label {
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
  .slider-label strong {
    color: var(--site-fg);
    font-variant-numeric: tabular-nums;
  }
  .slider input[type='range'] {
    width: 100%;
    accent-color: var(--ink-red);
  }

  .sample {
    align-items: baseline;
    gap: 0.7rem;
  }
  .sample-out code {
    background: var(--site-fg);
    color: var(--demo-card);
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.95rem;
  }
  .sample-out .muted {
    color: var(--site-fg-muted);
    font-size: 0.85rem;
  }

  .caption {
    margin: 0;
    font-size: 0.85rem;
    color: var(--site-fg-muted);
    line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg);
    font-style: normal;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.85em;
  }

  @media (prefers-reduced-motion: reduce) {
    .cell,
    .btn {
      transition: none;
    }
  }
</style>
