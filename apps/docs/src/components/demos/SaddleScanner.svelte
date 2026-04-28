<script lang="ts">
  import { Mafs, Coordinates, Plot, Line, Point } from 'svelte-mafs';

  /**
   * SaddleScanner — set the curvature signs (a, b) of f(x, y) = a x² + b y²,
   * then sweep a probe ray angle θ from the origin. The right panel plots
   * f along that ray as a function of distance t. The classification badge
   * (min / max / saddle) updates from the signs of a and b.
   */

  let a = $state(1);
  let b = $state(-1);
  let theta = $state(Math.PI / 4);

  const f = (x: number, y: number) => a * x * x + b * y * y;

  // Function along the ray (cos θ, sin θ): f(t cos θ, t sin θ)
  //   = (a cos²θ + b sin²θ) t²  → just a parabola in t with curvature
  const c = $derived(a * Math.cos(theta) ** 2 + b * Math.sin(theta) ** 2);
  const fAlongRay = (t: number) => c * t * t;

  // The (cos θ, sin θ) tip of the unit ray, for the left panel.
  const tipX = $derived(Math.cos(theta));
  const tipY = $derived(Math.sin(theta));

  const classify = $derived(
    a > 0 && b > 0 ? 'local minimum' :
    a < 0 && b < 0 ? 'local maximum' :
    Math.abs(a) < 0.05 || Math.abs(b) < 0.05 ? 'degenerate' :
    'saddle'
  );
  const classKind = $derived(
    classify === 'local minimum' ? 'min' :
    classify === 'local maximum' ? 'max' :
    classify === 'saddle' ? 'saddle' : 'deg'
  );

  // Y range for the side plot: cover both signs of f symmetrically.
  const yMag = $derived(Math.max(2, Math.abs(c) * 4 + 0.5));
</script>

<div class="widget">
  <header class="header">
    <p class="title">
      <em>f</em>(<em>x</em>, <em>y</em>) = <span class="mono">{a.toFixed(1)} <em>x</em>² + {b >= 0 ? '+' : ''}{b.toFixed(1)} <em>y</em>²</span>
    </p>
    <span class="badge badge-{classKind}">{classify}</span>
  </header>

  <div class="grid">
    <!-- Left: unit-circle compass with ray pointer -->
    <div class="stage">
      <Mafs width={300} height={300} viewBox={{ x: [-1.5, 1.5], y: [-1.5, 1.5] }}>
        <Coordinates.Cartesian />
        <!-- Unit circle: parametric -->
        <Plot.Parametric xy={(t) => [Math.cos(t), Math.sin(t)]} t={[0, 2 * Math.PI]} color="var(--site-fg-muted)" weight={1.25} opacity={0.45} />
        <Line.Segment point1={[0, 0]} point2={[tipX, tipY]} color="var(--ink-coral)" weight={2.5} />
        <Point x={tipX} y={tipY} color="var(--ink-coral)" />
      </Mafs>
      <p class="legend">θ = {(theta * 180 / Math.PI).toFixed(0)}°</p>
    </div>

    <!-- Right: f along the chosen ray -->
    <div class="stage">
      <Mafs width={300} height={300} viewBox={{ x: [-2, 2], y: [-yMag, yMag] }}>
        <Coordinates.Cartesian />
        <Plot.OfX y={fAlongRay} color="var(--ink-red)" weight={2.5} />
        <Point x={0} y={0} color="var(--ink-coral)" />
      </Mafs>
      <p class="legend">f along the ray (origin = 0)</p>
    </div>
  </div>

  <div class="controls">
    <label>
      <span><em>a</em> = {a.toFixed(2)}</span>
      <input type="range" min="-2" max="2" step="0.1" bind:value={a} aria-label="Coefficient a" />
    </label>
    <label>
      <span><em>b</em> = {b.toFixed(2)}</span>
      <input type="range" min="-2" max="2" step="0.1" bind:value={b} aria-label="Coefficient b" />
    </label>
    <label>
      <span>θ = {(theta * 180 / Math.PI).toFixed(0)}°</span>
      <input type="range" min="0" max={2 * Math.PI} step="0.02" bind:value={theta} aria-label="Ray angle" />
    </label>
  </div>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: .9rem;
    background: var(--demo-card); border: 1px solid var(--demo-card-border);
    border-radius: 20px; padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-red) 45%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .6rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .title em { font-style: italic; }
  .mono { font-family: var(--font-mono); }

  .badge { padding: .3rem .7rem; border-radius: 999px; font-family: var(--font-game); font-size: .78rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
  .badge-min { background: color-mix(in srgb, var(--cta) 18%, transparent); color: var(--cta-hover); }
  .badge-max { background: color-mix(in srgb, var(--ink-coral) 18%, transparent); color: var(--ink-coral); }
  .badge-saddle { background: color-mix(in srgb, var(--ink-sun) 22%, transparent); color: color-mix(in srgb, var(--ink-sun) 80%, var(--site-fg)); }
  .badge-deg { background: color-mix(in srgb, var(--site-fg) 14%, transparent); color: var(--site-fg-muted); }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: start; }
  @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
  .stage { width: 100%; background: var(--demo-stage); border-radius: 12px; padding: .35rem; overflow: hidden; touch-action: none; }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }
  .legend { margin: .25rem 0 0; text-align: center; font-family: var(--font-mono); font-size: .78rem; color: var(--site-fg-muted); }

  .controls { display: grid; gap: .5rem; padding-top: .5rem; border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent); }
  .controls label { display: grid; grid-template-columns: 8rem 1fr; gap: .8rem; align-items: center; font-family: var(--font-mono); font-size: .85rem; }
  .controls label em { font-style: italic; font-family: var(--font-display); font-weight: 600; }
  .controls span { color: var(--site-fg); }
  input[type="range"] { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; border-radius: 999px; background: color-mix(in srgb, var(--site-fg) 18%, transparent); outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 22px; height: 22px; border-radius: 999px; background: var(--ink-coral); border: 2px solid var(--demo-card); cursor: grab; }
  input[type="range"]::-moz-range-thumb { width: 22px; height: 22px; border-radius: 999px; background: var(--ink-coral); border: 2px solid var(--demo-card); cursor: grab; }
</style>
