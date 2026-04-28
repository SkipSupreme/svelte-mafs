<script lang="ts">
  import { Mafs, Coordinates, Plot, Point, Line } from 'svelte-mafs';

  /**
   * PartialSlicer — freeze one variable, plot the resulting 1D slice, and
   * read off its slope at the freeze point. That slope IS the partial
   * derivative.
   *
   *   f(x, y) = x² y + 3 y²
   *   ∂f/∂x = 2 x y           (treat y as constant)
   *   ∂f/∂y = x² + 6 y         (treat x as constant)
   */

  const f = (x: number, y: number) => x * x * y + 3 * y * y;
  const dfdx = (x: number, y: number) => 2 * x * y;
  const dfdy = (x: number, y: number) => x * x + 6 * y;

  let x0 = $state(1.5);
  let y0 = $state(1);

  // Slice along x at fixed y₀.
  const sliceX = (x: number) => f(x, y0);
  // Slice along y at fixed x₀.
  const sliceY = (y: number) => f(x0, y);

  // Tangent lines on each slice (1D linearization).
  const slopeX = $derived(dfdx(x0, y0));
  const slopeY = $derived(dfdy(x0, y0));
  const valHere = $derived(f(x0, y0));
  const tanX = (x: number) => valHere + slopeX * (x - x0);
  const tanY = (y: number) => valHere + slopeY * (y - y0);
</script>

<div class="widget">
  <header class="header">
    <p class="title"><em>f</em>(<em>x</em>, <em>y</em>) = <em>x</em>²<em>y</em> + 3<em>y</em>²</p>
    <p class="hint">freeze one variable · slope of the 1D slice = partial derivative</p>
  </header>

  <div class="grid">
    <div class="panel">
      <p class="panel-label">slice at y = {y0.toFixed(2)} → see ∂f/∂x</p>
      <div class="stage">
        <Mafs width={320} height={260} viewBox={{ x: [-2.5, 2.5], y: [-2, 12] }}>
          <Coordinates.Cartesian />
          <Plot.OfX y={sliceX} color="var(--ink-red)" weight={2.5} />
          <Plot.OfX y={tanX} color="var(--ink-coral)" weight={1.75} opacity={0.85} />
          <Point x={x0} y={valHere} color="var(--ink-coral)" />
        </Mafs>
      </div>
      <p class="readout">∂f/∂x at (x, y) = <strong>{slopeX.toFixed(2)}</strong></p>
    </div>

    <div class="panel">
      <p class="panel-label">slice at x = {x0.toFixed(2)} → see ∂f/∂y</p>
      <div class="stage">
        <Mafs width={320} height={260} viewBox={{ x: [-2.5, 2.5], y: [-2, 12] }}>
          <Coordinates.Cartesian />
          <Plot.OfX y={sliceY} color="var(--ink-sea)" weight={2.5} />
          <Plot.OfX y={tanY} color="var(--ink-coral)" weight={1.75} opacity={0.85} />
          <Point x={y0} y={valHere} color="var(--ink-coral)" />
        </Mafs>
      </div>
      <p class="readout">∂f/∂y at (x, y) = <strong>{slopeY.toFixed(2)}</strong></p>
    </div>
  </div>

  <div class="controls">
    <label>
      <span><em>x</em> = {x0.toFixed(2)}</span>
      <input type="range" min="-2.2" max="2.2" step="0.05" bind:value={x0} aria-label="x value" />
    </label>
    <label>
      <span><em>y</em> = {y0.toFixed(2)}</span>
      <input type="range" min="-1.5" max="1.5" step="0.05" bind:value={y0} aria-label="y value" />
    </label>
  </div>

  <p class="note">
    The coral line is the <strong>tangent</strong> to each slice. Its slope is the partial. As you drag, both partials update independently — the function has two answers, one per direction.
  </p>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: .85rem;
    background: var(--demo-card); border: 1px solid var(--demo-card-border);
    border-radius: 20px; padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-red) 45%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: .5rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .title em { font-style: italic; }
  .hint { margin: 0; font-family: var(--font-mono); font-size: .8rem; color: var(--site-fg-muted); }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
  .panel { display: flex; flex-direction: column; gap: .35rem; }
  .panel-label { margin: 0; font-family: var(--font-mono); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--site-fg-muted); }
  .stage { background: var(--demo-stage); border-radius: 12px; padding: .35rem; touch-action: none; }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }
  .readout { margin: 0; font-family: var(--font-mono); font-size: .9rem; color: var(--site-fg); }
  .readout strong { color: var(--ink-coral); }

  .controls { display: grid; gap: .45rem; padding-top: .5rem; border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent); }
  .controls label { display: grid; grid-template-columns: 6.5rem 1fr; gap: .8rem; align-items: center; font-family: var(--font-mono); font-size: .85rem; }
  .controls label em { font-style: italic; font-family: var(--font-display); }
  input[type="range"] { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; border-radius: 999px; background: color-mix(in srgb, var(--site-fg) 18%, transparent); outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 22px; height: 22px; border-radius: 999px; background: var(--ink-coral); border: 2px solid var(--demo-card); cursor: grab; }
  input[type="range"]::-moz-range-thumb { width: 22px; height: 22px; border-radius: 999px; background: var(--ink-coral); border: 2px solid var(--demo-card); cursor: grab; }

  .note { margin: 0; padding: .55rem .85rem; background: color-mix(in srgb, var(--site-fg) 4%, transparent); border-radius: 10px; font-family: var(--font-body); font-size: .82rem; color: var(--site-fg); }
  .note strong { color: var(--ink-coral); }
</style>
