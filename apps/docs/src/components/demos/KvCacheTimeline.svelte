<script lang="ts">
  const T_MAX = 12;
  const D = 64; // notional per-head dimension; only the proportions matter

  let step = $state(0);
  let cache = $state(true);
  let pulseToken = $state<number | null>(null);

  // Each generated token contributes a (K, V) row at its position.
  // With the cache, that row is computed once and reused.
  // Without the cache, every prior K, V row is recomputed at every step.

  function flopsForStep(t: number, cached: boolean): number {
    // t is the index of the token being generated this step (0-indexed).
    // The query is one new vector. We score it against (t + 1) cached keys
    // and read (t + 1) values back, with d_k = D.
    const attentionWork = (t + 1) * D;
    if (cached) {
      // Compute one new (K, V) row: O(D) for the projections (we ignore the
      // model-side D^2 because it's constant either way).
      return attentionWork + D;
    }
    // No cache: recompute all (t + 1) keys and values from scratch.
    const recomputeKV = (t + 1) * D;
    return attentionWork + recomputeKV;
  }

  const stepFlopsCached = $derived(
    Array.from({ length: T_MAX }, (_, t) => flopsForStep(t, true))
  );
  const stepFlopsUncached = $derived(
    Array.from({ length: T_MAX }, (_, t) => flopsForStep(t, false))
  );

  const cumCached = $derived(
    stepFlopsCached.slice(0, step).reduce((a, b) => a + b, 0)
  );
  const cumUncached = $derived(
    stepFlopsUncached.slice(0, step).reduce((a, b) => a + b, 0)
  );
  const ratio = $derived(cumCached === 0 ? 1 : cumUncached / cumCached);

  function tick() {
    if (step >= T_MAX) return;
    pulseToken = step;
    step += 1;
    setTimeout(() => (pulseToken = null), 360);
  }
  function reset() {
    step = 0;
    pulseToken = null;
  }
</script>

<div class="widget">
  <div class="controls">
    <button type="button" class="primary" onclick={tick} disabled={step >= T_MAX}>
      ▶ generate token #{step + 1}
    </button>
    <button type="button" class="ghost" onclick={reset} disabled={step === 0}>reset</button>
    <div class="counter">
      step <strong>{step}</strong> / {T_MAX}
    </div>

    <div class="cache-toggle">
      <span class="label">cache</span>
      <div class="seg" role="group">
        <button
          type="button"
          class="seg-btn"
          class:active={!cache}
          onclick={() => (cache = false)}>off</button>
        <button
          type="button"
          class="seg-btn"
          class:active={cache}
          onclick={() => (cache = true)}>on</button>
      </div>
    </div>
  </div>

  <div class="track-block">
    <h4 class="track-title">K-cache (one row per token)</h4>
    <div class="track">
      {#each Array(T_MAX) as _, i}
        <div
          class="slot"
          class:filled={i < step}
          class:pulse={!cache && i < step && pulseToken === step - 1}
          class:fresh={cache && i === step - 1 && pulseToken === step - 1}
          aria-label="K row {i + 1} {i < step ? 'filled' : 'empty'}"
        >
          <span class="slot-num">k<sub>{i + 1}</sub></span>
        </div>
      {/each}
    </div>
  </div>

  <div class="track-block">
    <h4 class="track-title">V-cache (one row per token)</h4>
    <div class="track">
      {#each Array(T_MAX) as _, i}
        <div
          class="slot"
          class:filled={i < step}
          class:pulse={!cache && i < step && pulseToken === step - 1}
          class:fresh={cache && i === step - 1 && pulseToken === step - 1}
          aria-label="V row {i + 1} {i < step ? 'filled' : 'empty'}"
        >
          <span class="slot-num">v<sub>{i + 1}</sub></span>
        </div>
      {/each}
    </div>
  </div>

  <div class="explainer">
    {#if cache}
      <strong>cache on</strong> — each token's K and V are computed <em>once</em> at its
      step (yellow flash) and reused on every subsequent step. Past columns
      stay filled.
    {:else}
      <strong>cache off</strong> — every step recomputes K and V for the entire prefix.
      Watch all the past columns flash on each new generation: that work is being
      paid for, repeatedly, forever.
    {/if}
  </div>

  <div class="param-bar">
    <div class="metric">
      <span class="metric-label">cumulative work — cache on</span>
      <span class="metric-val">{cumCached.toLocaleString()} flops</span>
    </div>
    <div class="metric">
      <span class="metric-label">cumulative work — cache off</span>
      <span class="metric-val">{cumUncached.toLocaleString()} flops</span>
    </div>
    <div class="metric hi">
      <span class="metric-label">savings ratio</span>
      <span class="metric-val">{ratio.toFixed(2)}×</span>
    </div>
  </div>

  <p class="legend">
    work scales as <em>T²</em> with the cache, <em>T³</em> without. ratio grows linearly with sequence length — at <em>T = 1024</em> the cache buys roughly a 1000× speedup.
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
    padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 55%, transparent);
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.65rem;
  }
  .primary {
    appearance: none;
    border: 0;
    padding: 0.55rem 1rem;
    border-radius: 10px;
    background: var(--ink-sea);
    color: white;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 160ms ease;
  }
  .primary:disabled {
    background: color-mix(in srgb, var(--site-fg) 14%, transparent);
    color: var(--site-fg-muted);
    cursor: not-allowed;
  }
  .primary:not(:disabled):hover {
    background: color-mix(in srgb, var(--ink-sea) 85%, var(--site-fg));
  }
  .ghost {
    appearance: none;
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    background: transparent;
    color: var(--site-fg-muted);
    padding: 0.45rem 0.8rem;
    border-radius: 10px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    cursor: pointer;
  }
  .ghost:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .ghost:not(:disabled):hover {
    background: color-mix(in srgb, var(--site-fg) 5%, transparent);
  }
  .counter {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--site-fg-muted);
    margin-left: 0.4rem;
  }
  .counter strong {
    color: var(--site-fg);
    font-weight: 700;
  }

  .cache-toggle {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .label {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
  }
  .seg {
    display: inline-flex;
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    border-radius: 10px;
    overflow: hidden;
  }
  .seg-btn {
    appearance: none;
    border: 0;
    padding: 0.35rem 0.7rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    background: transparent;
    color: var(--site-fg-muted);
    cursor: pointer;
  }
  .seg-btn + .seg-btn {
    border-left: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
  }
  .seg-btn.active {
    background: color-mix(in srgb, var(--ink-sea) 18%, transparent);
    color: var(--site-fg);
  }

  .track-block {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }
  .track-title {
    margin: 0;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--site-fg-muted);
    font-weight: 700;
  }
  .track {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 0.3rem;
  }
  .slot {
    aspect-ratio: 1;
    border-radius: 8px;
    border: 1px dashed color-mix(in srgb, var(--site-fg) 14%, transparent);
    background: transparent;
    display: grid;
    place-items: center;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: color-mix(in srgb, var(--site-fg) 30%, transparent);
    transition: background 220ms ease, color 220ms ease, border-color 220ms ease, transform 160ms ease;
  }
  .slot.filled {
    background: color-mix(in srgb, var(--ink-sea) 22%, transparent);
    border: 1px solid var(--ink-sea);
    color: var(--site-fg);
  }
  .slot.fresh {
    animation: fresh 380ms ease;
  }
  .slot.pulse {
    animation: pulse 380ms ease;
  }
  @keyframes fresh {
    0% {
      background: color-mix(in srgb, var(--ink-sun) 70%, transparent);
      transform: scale(1.12);
    }
    100% {
      background: color-mix(in srgb, var(--ink-sea) 22%, transparent);
      transform: scale(1);
    }
  }
  @keyframes pulse {
    0% {
      background: color-mix(in srgb, var(--ink-coral) 60%, transparent);
    }
    100% {
      background: color-mix(in srgb, var(--ink-sea) 22%, transparent);
    }
  }

  .explainer {
    padding: 0.65rem 0.85rem;
    border-radius: 10px;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-left: 3px solid var(--ink-sea);
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: var(--site-fg);
    line-height: 1.55;
  }
  .explainer strong {
    font-weight: 700;
  }
  .explainer em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }

  .param-bar {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
  .metric {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    padding: 0.55rem 0.7rem;
    border-radius: 10px;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-left: 3px solid var(--site-fg-muted);
  }
  .metric.hi {
    border-left-color: var(--ink-sun);
    background: color-mix(in srgb, var(--ink-sun) 12%, transparent);
  }
  .metric-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--site-fg-muted);
  }
  .metric-val {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 1rem;
    font-weight: 700;
    color: var(--site-fg);
  }

  .legend {
    margin: 0;
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--site-fg-muted);
  }
  .legend em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
    color: var(--site-fg);
  }
</style>
