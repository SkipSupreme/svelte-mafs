<script lang="ts">
  interface Props {
    /** Length of the unrolled chain. */
    timesteps?: number;
    /** Initial spectral radius of W_h's Jacobian. */
    initialRho?: number;
  }

  let { timesteps = 30, initialRho = 0.85 }: Props = $props();

  let rho: number = $state(initialRho);

  // Gradient magnitude at each timestep, going backward from the loss at t = T.
  // ‖∂L/∂h_k‖ ≈ ρ^(T - k) (a clean upper bound on the per-layer Jacobian product).
  // Clamped to a representable floor so the visualization doesn't underflow visually.
  const FLOOR_LOG = -25;
  const grads: number[] = $derived.by(() => {
    const out: number[] = [];
    for (let k = 0; k < timesteps; k++) {
      const stepsBack = timesteps - 1 - k;
      const g = Math.pow(rho, stepsBack);
      out.push(g);
    }
    return out;
  });

  const logGrads: number[] = $derived(
    grads.map((g) => (g > 0 ? Math.max(FLOOR_LOG, Math.log10(g)) : FLOOR_LOG)),
  );

  // Memory horizon: rightmost (largest k) timestep where log10(g) drops below -6.
  // (Reading right-to-left from the loss: the leftmost surviving step.)
  const HORIZON_LOG = -6;
  const horizonStep: number | null = $derived.by(() => {
    for (let k = 0; k < timesteps; k++) {
      if (logGrads[k] >= HORIZON_LOG) return k;
    }
    return null;
  });

  // Color scheme:
  // - vanishing (ρ < 1): faded → bright as k → T
  // - balanced (ρ ≈ 1): uniform
  // - exploding (ρ > 1): bright everywhere; coral if log > 4
  function colorFor(g: number): string {
    if (g <= 0 || logGrads.length === 0) return 'var(--cell-empty)';
    const lg = Math.log10(g);
    if (lg > 3) return 'var(--ink-coral)';
    if (lg < HORIZON_LOG) return 'color-mix(in srgb, var(--site-fg) 4%, transparent)';
    // Map log10(g) ∈ [FLOOR_LOG, 0] to opacity 0 → 1
    const t = Math.max(0, Math.min(1, (lg - FLOOR_LOG) / -FLOOR_LOG));
    const pct = Math.round(t * 100);
    return `color-mix(in srgb, var(--ink-sea) ${pct}%, var(--cell-empty))`;
  }

  function fmt(g: number): string {
    if (!Number.isFinite(g)) return '∞';
    if (g === 0) return '0';
    if (g >= 0.001 && g <= 1000) return g.toFixed(3);
    return g.toExponential(1);
  }

  const dynamicMode: 'vanishing' | 'balanced' | 'exploding' = $derived(
    rho < 0.97 ? 'vanishing' : rho > 1.03 ? 'exploding' : 'balanced',
  );

  // For the Jacobian-as-ellipse visualization (left panel).
  // The ellipse is just a circle scaled by ρ on one axis — visually "stretched."
  const ellipseW = $derived(40 + rho * 18);
  const ellipseH = $derived(40);

  const presets = [
    { id: 'vanishing', label: 'vanishing (ρ = 0.5)', value: 0.5 },
    { id: 'balanced', label: 'balanced (ρ = 1.0)', value: 1.0 },
    { id: 'exploding', label: 'exploding (ρ = 1.15)', value: 1.15 },
  ];
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">spectral radius ρ</span>
      <span class="meta-val">{rho.toFixed(2)}</span>
    </div>
    <div class="meta">
      <span class="meta-key">‖∂L/∂h_0‖</span>
      <span class="meta-val" class:dead={grads[0] < 1e-6} class:hot={grads[0] > 1000}>
        {fmt(grads[0])}
      </span>
    </div>
    <div class="meta">
      <span class="meta-key">memory horizon</span>
      <span class="meta-val" class:dead={horizonStep == null}>
        {#if horizonStep == null}
          (none — gradient already vanished)
        {:else}
          t = {horizonStep}
        {/if}
      </span>
    </div>
    <div class="meta">
      <span class="meta-key">regime</span>
      <span class="meta-val regime-{dynamicMode}">{dynamicMode}</span>
    </div>
  </header>

  <div class="layout">
    <!-- Left: the Jacobian ellipse. Ellipse is a circle deformed by ρ. -->
    <div class="jacobian">
      <div class="jacobian-label">Jacobian of W<sub>h</sub></div>
      <svg viewBox="0 0 120 120" preserveAspectRatio="xMidYMid meet">
        <!-- unit-circle reference -->
        <circle cx="60" cy="60" r="30" fill="none" stroke="color-mix(in srgb, var(--site-fg) 18%, transparent)" stroke-dasharray="3 3" stroke-width="0.7" />
        <!-- the actual Jacobian's deformation ellipse -->
        <ellipse
          cx="60" cy="60"
          rx={ellipseW * 0.4}
          ry={ellipseH * 0.4}
          fill="color-mix(in srgb, var(--ink-sea) 14%, transparent)"
          stroke="var(--ink-sea)"
          stroke-width="2"
        />
        <!-- arrow showing magnitude -->
        <line x1="60" y1="60" x2={60 + ellipseW * 0.4} y2="60"
          stroke="var(--ink-red)" stroke-width="1.5" marker-end="url(#arrow)" />
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,1 L6,4 L0,7 Z" fill="var(--ink-red)" />
          </marker>
        </defs>
      </svg>
      <div class="jacobian-foot">
        <span class="jr-row"><span>ρ &lt; 1</span><span>contracts</span></span>
        <span class="jr-row"><span>ρ = 1</span><span>preserves</span></span>
        <span class="jr-row"><span>ρ &gt; 1</span><span>expands</span></span>
      </div>
    </div>

    <!-- Right: the timestep track. -->
    <div class="track-wrap">
      <div class="track-label">
        <span>t = 0 (input)</span>
        <span>← gradient flows back ←</span>
        <span>t = {timesteps - 1} (loss)</span>
      </div>
      <div class="track" role="figure" aria-label="Gradient magnitude at each timestep, going back from the loss.">
        {#each grads as g, k}
          <div
            class="cell"
            class:horizon={horizonStep === k}
            style="background:{colorFor(g)};"
            title={`t = ${k}, |grad| = ${fmt(g)}`}
          >
            {#if horizonStep === k}
              <span class="horizon-mark">↑ horizon</span>
            {/if}
          </div>
        {/each}
      </div>
      <div class="readouts">
        <div class="rd"><span>t = 0</span><strong>{fmt(grads[0])}</strong></div>
        <div class="rd"><span>t = {Math.floor(timesteps / 2)}</span><strong>{fmt(grads[Math.floor(timesteps / 2)])}</strong></div>
        <div class="rd"><span>t = {timesteps - 1}</span><strong>{fmt(grads[timesteps - 1])}</strong></div>
      </div>
    </div>
  </div>

  <div class="controls">
    <label class="slider">
      <span class="slider-label">spectral radius ρ = <strong>{rho.toFixed(2)}</strong></span>
      <input type="range" min="0.1" max="1.5" step="0.01" bind:value={rho} aria-label="Spectral radius" />
    </label>
    <div class="ctl-row">
      {#each presets as p}
        <button
          type="button"
          class="btn btn-ghost"
          class:active={Math.abs(rho - p.value) < 0.005}
          onclick={() => (rho = p.value)}
        >{p.label}</button>
      {/each}
    </div>
  </div>

  <p class="caption">
    Each box is one timestep. Brightness is ‖∂L/∂h<sub>t</sub>‖ on a log
    scale — bright = gradient survived, faded = gradient gone. The Jacobian
    of W<sub>h</sub> is an ellipse: ρ &lt; 1 squashes it (contracts, gradient
    fades exponentially as you go back); ρ &gt; 1 stretches it (explodes).
    For ρ = 0.7, the gradient at t = 0 from a loss at t = 29 is about
    0.7<sup>29</sup> ≈ 5×10<sup>−5</sup> — already past the horizon.
    There is no software fix for this; it's a property of repeated linear
    contraction.
  </p>
</div>

<style>
  .widget {
    --cell-empty: color-mix(in srgb, var(--site-fg) 4%, transparent);
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
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .meta-val.dead { color: var(--ink-coral); }
  .meta-val.hot { color: var(--ink-coral); }
  .meta-val.regime-vanishing { color: var(--ink-coral); }
  .meta-val.regime-balanced { color: var(--cta); }
  .meta-val.regime-exploding { color: var(--ink-coral); }

  .layout {
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 0.7rem;
    background: var(--demo-stage);
    border-radius: 12px;
    padding: 0.85rem;
  }
  @media (max-width: 540px) {
    .layout { grid-template-columns: 1fr; }
    .jacobian { justify-self: center; max-width: 200px; }
  }

  .jacobian {
    display: flex; flex-direction: column;
    gap: 0.35rem;
    align-items: center;
  }
  .jacobian-label {
    font-family: var(--font-mono); font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--site-fg-muted);
  }
  .jacobian svg { width: 100%; max-width: 130px; height: auto; }
  .jacobian-foot {
    display: flex; flex-direction: column; gap: 0.15rem;
    width: 100%;
    font-family: var(--font-mono); font-size: 0.65rem; color: var(--site-fg-muted);
  }
  .jr-row { display: flex; justify-content: space-between; }

  .track-wrap { display: flex; flex-direction: column; gap: 0.4rem; }
  .track-label {
    display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: 0.68rem; color: var(--site-fg-muted);
  }
  .track {
    display: grid;
    grid-template-columns: repeat(30, 1fr);
    gap: 2px;
    align-items: stretch;
  }
  .cell {
    height: 32px;
    border-radius: 2px;
    transition: background 200ms cubic-bezier(0.2, 0.8, 0.2, 1);
    position: relative;
  }
  .cell.horizon {
    box-shadow: inset 0 0 0 1.5px var(--ink-sun);
  }
  .horizon-mark {
    position: absolute; top: -16px; left: 50%; transform: translateX(-50%);
    font-family: var(--font-mono); font-size: 0.6rem;
    color: var(--ink-sun); white-space: nowrap;
  }
  .readouts {
    display: flex; justify-content: space-between;
    margin-top: 0.2rem;
  }
  .rd {
    display: flex; flex-direction: column; align-items: center; gap: 1px;
    font-family: var(--font-mono); font-size: 0.7rem; color: var(--site-fg-muted);
  }
  .rd strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }

  .controls { display: flex; flex-direction: column; gap: 0.5rem; }
  .ctl-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }
  .slider { display: flex; flex-direction: column; gap: 0.2rem; }
  .slider-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted); }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input[type='range'] { width: 100%; accent-color: var(--ink-sea); }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.32rem 0.85rem;
    font-size: 0.8rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover { transform: translateY(-1px); border-color: var(--site-fg); }
  .btn-ghost { color: var(--site-fg-muted); font-weight: 500; }
  .btn.active {
    background: var(--ink-sea); color: var(--on-color-fg); border-color: var(--ink-sea);
  }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }

  @media (prefers-reduced-motion: reduce) { .cell { transition: none; } }
</style>
