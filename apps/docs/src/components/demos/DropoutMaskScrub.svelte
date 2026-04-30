<script lang="ts">
  import { onDestroy } from 'svelte';

  interface Props {
    /** Number of activations to draw bars for. */
    units?: number;
    /** Initial dropout probability. */
    initialP?: number;
  }

  let { units = 8, initialP = 0.4 }: Props = $props();

  // Eval-mode "true" activations (gray). Fixed pattern for visual stability.
  const evalValues: number[] = $derived.by(() =>
    Array.from({ length: units }, (_, i) => 1.5 + Math.cos((i / units) * 2 * Math.PI) * 1.0 + 1.5),
  );

  let p: number = $state(initialP);
  let lastSample: (number | 0)[] = $state([]);
  let runningMean: number[] = $state(Array(units).fill(0));
  let nSamples: number = $state(0);
  let runRafId: number | null = null;
  let running: boolean = $state(false);

  const scale = $derived(1 / (1 - p));
  const maxBar = $derived.by(() => {
    let m = 0;
    for (const v of evalValues) if (v > m) m = v;
    // include scaled values too — at low survival, individual samples spike high
    const samplePeak = Math.max(...evalValues) * scale;
    return Math.max(m, samplePeak * 0.6);
  });

  function sampleOnce(): void {
    const next = new Array<number>(units);
    for (let i = 0; i < units; i++) {
      const survives = Math.random() > p;
      next[i] = survives ? evalValues[i] * scale : 0;
    }
    lastSample = next;
    // Update running mean.
    const newMean = new Array<number>(units);
    for (let i = 0; i < units; i++) {
      newMean[i] = (runningMean[i] * nSamples + next[i]) / (nSamples + 1);
    }
    runningMean = newMean;
    nSamples += 1;
  }

  function sampleN(n: number): void {
    for (let k = 0; k < n; k++) sampleOnce();
  }

  function reset(): void {
    stopAuto();
    lastSample = [];
    runningMean = Array(units).fill(0);
    nSamples = 0;
  }

  function startAuto(): void {
    if (running) return;
    running = true;
    const tick = () => {
      if (!running) return;
      sampleN(3);
      runRafId = requestAnimationFrame(tick);
    };
    runRafId = requestAnimationFrame(tick);
  }
  function stopAuto(): void {
    running = false;
    if (runRafId != null) {
      cancelAnimationFrame(runRafId);
      runRafId = null;
    }
  }
  onDestroy(() => stopAuto());

  // Reset running stats whenever p changes — old samples were drawn from a
  // different distribution and no longer make sense to average.
  let lastP = initialP;
  $effect(() => {
    if (p !== lastP) {
      lastP = p;
      reset();
    }
  });

  const fmt = (n: number, d = 2) => n.toFixed(d);

  // Bar geometry: heights scale to 100% of bar-track height.
  function barH(v: number): string {
    if (maxBar === 0) return '0%';
    return (Math.min(1, v / maxBar) * 100).toFixed(1) + '%';
  }

  function maxAbsErr(): number {
    if (nSamples === 0) return 0;
    let m = 0;
    for (let i = 0; i < units; i++) {
      const e = Math.abs(runningMean[i] - evalValues[i]);
      if (e > m) m = e;
    }
    return m;
  }
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">dropout p</span>
      <span class="meta-val">{fmt(p, 2)}</span>
    </div>
    <div class="meta">
      <span class="meta-key">scale 1/(1−p)</span>
      <span class="meta-val">{fmt(scale, 3)}</span>
    </div>
    <div class="meta">
      <span class="meta-key">samples</span>
      <span class="meta-val">{nSamples}</span>
    </div>
    <div class="meta meta-conv" class:converged={nSamples > 25 && maxAbsErr() < 0.1}>
      <span class="meta-key">|coral − gray|</span>
      <span class="meta-val">{fmt(maxAbsErr(), 3)}</span>
    </div>
  </header>

  <div class="stage">
    <div class="bars" role="figure" aria-label="Eval-mode activations (gray) versus running mean of training-mode samples (coral).">
      {#each evalValues as ev, i}
        <div class="bar-col">
          <div class="bar-track">
            <!-- Last sample (faint orange) -->
            {#if lastSample[i] != null && lastSample[i] > 0}
              <div class="bar bar-sample" style="height:{barH(lastSample[i])};"></div>
            {/if}
            <!-- Running mean (coral) -->
            {#if nSamples > 0}
              <div class="bar bar-mean" style="height:{barH(runningMean[i])};"></div>
            {/if}
            <!-- Eval-mode true value (gray, the target) -->
            <div class="bar bar-eval" style="height:{barH(ev)};"></div>
          </div>
          <span class="bar-label">{i + 1}</span>
        </div>
      {/each}
    </div>

    <div class="legend" aria-hidden="true">
      <span class="lg lg-eval">eval (target)</span>
      <span class="lg lg-mean">running mean of samples</span>
      <span class="lg lg-sample">last sample</span>
    </div>
  </div>

  <div class="controls">
    <div class="ctl-row">
      <button type="button" class="btn btn-primary" onclick={sampleOnce} disabled={running}>Sample × 1</button>
      <button type="button" class="btn" onclick={() => sampleN(50)} disabled={running}>Sample × 50</button>
      <button
        type="button"
        class="btn btn-toggle"
        class:active={running}
        onclick={() => (running ? stopAuto() : startAuto())}
      >{running ? 'Stop' : 'Auto'}</button>
      <button type="button" class="btn btn-ghost" onclick={reset}>Reset</button>
    </div>
    <label class="slider">
      <span class="slider-label">p = <strong>{p.toFixed(2)}</strong></span>
      <input type="range" min="0" max="0.9" step="0.05" bind:value={p} aria-label="Dropout probability" />
    </label>
  </div>

  <p class="caption">
    Each step: a binary mask drops each unit with probability <em>p</em>; surviving
    units multiply by 1/(1−<em>p</em>). The expectation lands on the gray bars by
    construction. With <em>p</em> = 0, scaling does nothing and samples equal
    eval. As <em>p</em> grows, individual samples get noisier — but their
    average still locks onto the eval-mode value within a few dozen draws.
  </p>
</div>

<style>
  .widget {
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-coral) 50%, transparent);
  }
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .meta-conv.converged .meta-val { color: var(--cta); }

  .stage {
    background: var(--demo-stage);
    border-radius: 12px;
    padding: 0.85rem 1rem 0.7rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .bars {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    height: 180px;
  }
  .bar-col {
    flex: 1 1 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
    height: 100%;
  }
  .bar-track {
    position: relative;
    width: 100%;
    height: 160px;
    display: flex;
    align-items: flex-end;
    justify-content: center;
  }
  .bar {
    position: absolute;
    bottom: 0;
    border-radius: 3px 3px 0 0;
    transition: height 220ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .bar-eval {
    width: 50%;
    left: 0;
    background: color-mix(in srgb, var(--site-fg) 28%, transparent);
  }
  .bar-mean {
    width: 50%;
    right: 0;
    background: var(--ink-coral);
  }
  .bar-sample {
    width: 50%;
    right: 0;
    background: color-mix(in srgb, var(--ink-coral) 30%, transparent);
    border: 1px dashed color-mix(in srgb, var(--ink-coral) 60%, transparent);
    border-bottom: none;
  }
  .bar-label {
    font-family: var(--font-mono); font-size: 0.68rem; color: var(--site-fg-muted);
  }

  .legend {
    display: flex; gap: 0.85rem; font-family: var(--font-mono);
    font-size: 0.7rem; color: var(--site-fg-muted);
  }
  .lg::before {
    content: ''; display: inline-block; width: 10px; height: 8px; margin-right: 4px;
    border-radius: 2px; vertical-align: middle;
  }
  .lg-eval::before { background: color-mix(in srgb, var(--site-fg) 28%, transparent); }
  .lg-mean::before { background: var(--ink-coral); }
  .lg-sample::before {
    background: color-mix(in srgb, var(--ink-coral) 30%, transparent);
    border: 1px dashed color-mix(in srgb, var(--ink-coral) 60%, transparent);
  }

  .controls { display: flex; flex-direction: column; gap: 0.5rem; }
  .ctl-row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.35rem 0.85rem;
    font-size: 0.83rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover:not(:disabled) { transform: translateY(-1px); border-color: var(--site-fg); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary {
    background: var(--ink-coral); color: var(--on-color-fg); border-color: var(--ink-coral);
  }
  .btn-toggle.active { background: var(--cta); color: var(--cta-fg); border-color: var(--cta); }
  .btn-ghost { color: var(--site-fg-muted); font-weight: 500; }

  .slider { display: flex; flex-direction: column; gap: 0.2rem; flex: 1 1 200px; }
  .slider-label {
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input[type='range'] { width: 100%; accent-color: var(--ink-coral); }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg); font-style: normal;
    font-family: var(--font-mono); font-size: 0.85em;
  }

  @media (prefers-reduced-motion: reduce) { .bar { transition: none; } }
</style>
