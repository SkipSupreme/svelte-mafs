<script lang="ts">
  import { Mafs, Coordinates, Point, Circle } from 'svelte-mafs';

  /**
   * GradientNoiseBall — run mini-batch SGD on a quadratic for many random
   * seeds, scatter the endpoints, and report the empirical radius.
   *
   * Loss: L(w) = ½ ||w||²  (gradient = w). Each step:
   *   w ← w - η (w + noise),  noise ~ N(0, σ_g² / |B|).
   *
   * Run for K steps, record final w, repeat for many seeds. Watch the
   * cloud shrink as |B| grows.
   */

  let batch = $state(4);
  let eta = $state(0.1);
  let sigmaG = $state(1);
  const steps = 80;
  const seeds = 120;

  // Box–Muller normal sample.
  function randn(): number {
    const u = 1 - Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  let regenKey = $state(0);
  function regen() { regenKey++; }

  type Pt = readonly [number, number];

  const endpoints = $derived.by<Pt[]>(() => {
    // Re-runs whenever batch / eta / sigmaG / regenKey changes.
    void regenKey;
    const out: Pt[] = [];
    const noiseStd = sigmaG / Math.sqrt(batch);
    for (let s = 0; s < seeds; s++) {
      let wx = 1.5;
      let wy = 0.8;
      for (let k = 0; k < steps; k++) {
        wx -= eta * (wx + noiseStd * randn());
        wy -= eta * (wy + noiseStd * randn());
      }
      out.push([wx, wy] as const);
    }
    return out;
  });

  // Empirical radius: mean Euclidean distance from origin.
  const radius = $derived(
    endpoints.reduce((acc, [x, y]) => acc + Math.hypot(x, y), 0) / endpoints.length,
  );

  // Theoretical noise-ball radius (rough): η σ_g / sqrt(|B|).
  const radiusTheory = $derived(eta * sigmaG / Math.sqrt(batch));
</script>

<div class="widget">
  <header class="header">
    <p class="title">SGD endpoint cloud · {seeds} seeds, {steps} steps each</p>
    <button type="button" class="reroll" onclick={regen}>reroll seeds</button>
  </header>

  <div class="stage">
    <Mafs width={460} height={340} viewBox={{ x: [-1, 1], y: [-0.7, 0.7] }}>
      <Coordinates.Cartesian />
      <!-- Theoretical noise ball -->
      <Circle center={[0, 0]} radius={radiusTheory} color="var(--ink-sea)" weight={1.5} fillOpacity={0.08} fillColor="var(--ink-sea)" opacity={0.5} />
      <!-- Endpoints -->
      {#each endpoints as p (p[0] * 1e6 + p[1])}
        <Point x={p[0]} y={p[1]} color="var(--ink-coral)" />
      {/each}
      <!-- Origin = the minimum -->
      <Point x={0} y={0} color="var(--ink-red)" />
    </Mafs>
  </div>

  <div class="controls">
    <label>
      <span>|B| = {batch}</span>
      <input type="range" min="1" max="64" step="1" bind:value={batch} aria-label="Batch size" />
    </label>
    <label>
      <span>η = {eta.toFixed(2)}</span>
      <input type="range" min="0.02" max="0.3" step="0.01" bind:value={eta} aria-label="Learning rate" />
    </label>
    <label>
      <span>σ_g = {sigmaG.toFixed(2)}</span>
      <input type="range" min="0.2" max="2" step="0.05" bind:value={sigmaG} aria-label="Per-example gradient noise" />
    </label>
  </div>

  <dl class="readout" aria-live="polite">
    <div>
      <dt>empirical radius</dt>
      <dd><strong>{radius.toFixed(3)}</strong></dd>
    </div>
    <div>
      <dt>theory: η σ / √|B|</dt>
      <dd>{radiusTheory.toFixed(3)}</dd>
    </div>
    <p class="caption">bigger batch → smaller cloud · smaller η → smaller cloud · this is a ball, not a point.</p>
  </dl>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: .85rem;
    background: var(--demo-card); border: 1px solid var(--demo-card-border);
    border-radius: 20px; padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-coral) 45%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .5rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .reroll { padding: .35rem .8rem; font-family: var(--font-mono); font-size: .8rem; border: 1px solid var(--site-border); border-radius: 999px; background: var(--demo-card); color: var(--site-fg); cursor: pointer; }
  .reroll:hover { border-color: var(--ink-coral); color: var(--ink-coral); }

  .stage { width: 100%; background: var(--demo-stage); border-radius: 12px; padding: .35rem; touch-action: none; }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

  .controls { display: grid; gap: .45rem; padding-top: .5rem; border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent); }
  .controls label { display: grid; grid-template-columns: 6.5rem 1fr; gap: .8rem; align-items: center; font-family: var(--font-mono); font-size: .85rem; }
  input[type="range"] { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; border-radius: 999px; background: color-mix(in srgb, var(--site-fg) 18%, transparent); outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 22px; height: 22px; border-radius: 999px; background: var(--ink-coral); border: 2px solid var(--demo-card); cursor: grab; }
  input[type="range"]::-moz-range-thumb { width: 22px; height: 22px; border-radius: 999px; background: var(--ink-coral); border: 2px solid var(--demo-card); cursor: grab; }

  .readout { margin: 0; display: flex; flex-wrap: wrap; gap: 1.2rem; padding: .55rem .85rem; background: color-mix(in srgb, var(--site-fg) 4%, transparent); border-radius: 10px; font-family: var(--font-mono); }
  .readout > div { display: flex; flex-direction: column; gap: .1rem; }
  .readout dt { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--site-fg-muted); }
  .readout dd { margin: 0; font-size: .98rem; color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .readout strong { color: var(--ink-coral); }
  .readout .caption { flex-basis: 100%; margin: .1rem 0 0; font-family: var(--font-body); font-size: .78rem; color: var(--site-fg-muted); }
</style>
