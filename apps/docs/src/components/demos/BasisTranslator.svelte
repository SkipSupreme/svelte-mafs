<script lang="ts">
  import { Mafs, Coordinates, Vector, MovablePoint, Line, Point } from 'svelte-mafs';

  /**
   * BasisTranslator — same arrow, two coordinate addresses.
   *
   * A fixed point P sits in the plane (draggable). Two basis arrows v₁, v₂
   * are also draggable. The widget computes P's coordinates in the basis
   * {v₁, v₂} by inverting the basis matrix.
   *
   *   P = c₁ v₁ + c₂ v₂  →  [c₁, c₂] = B⁻¹ P, where B = [v₁ | v₂].
   *
   * Built-in payoff: when v₁ and v₂ are aligned with eigenvectors of some
   * underlying transformation, the new coordinates make a complicated map
   * look diagonal. (The lesson narrates this; the widget just shows the
   * coordinate translation.)
   */

  let v1x = $state(1);
  let v1y = $state(0);
  let v2x = $state(0);
  let v2y = $state(1);
  let px = $state(2);
  let py = $state(1);

  const detB = $derived(v1x * v2y - v1y * v2x);
  const singular = $derived(Math.abs(detB) < 0.02);

  // c₁, c₂ such that P = c₁ v₁ + c₂ v₂.
  const c1 = $derived(singular ? 0 : (px * v2y - py * v2x) / detB);
  const c2 = $derived(singular ? 0 : (-px * v1y + py * v1x) / detB);
</script>

<div class="widget">
  <header class="header">
    <p class="title">a fixed point, two coordinate systems</p>
    <p class="hint">drag the basis arrows · drag <em>P</em> · same arrow, two addresses</p>
  </header>

  <div class="stage">
    <Mafs width={560} height={380} viewBox={{ x: [-3.5, 3.5], y: [-2.4, 2.4] }}>
      <Coordinates.Cartesian />

      {#if !singular}
        <!-- Reconstruction lines: c₁ v₁ then c₂ v₂ → P -->
        <Line.Segment point1={[0, 0]} point2={[c1 * v1x, c1 * v1y]} color="var(--ink-red)" opacity={0.35} weight={1.25} />
        <Line.Segment point1={[c1 * v1x, c1 * v1y]} point2={[px, py]} color="var(--ink-sea)" opacity={0.35} weight={1.25} />
      {/if}

      <Vector tail={[0, 0]} tip={[v1x, v1y]} color="var(--ink-red)" weight={2.5} />
      <Vector tail={[0, 0]} tip={[v2x, v2y]} color="var(--ink-sea)" weight={2.5} />

      <MovablePoint bind:x={v1x} bind:y={v1y} color="var(--ink-red)" label="Basis vector v1" />
      <MovablePoint bind:x={v2x} bind:y={v2y} color="var(--ink-sea)" label="Basis vector v2" />
      <MovablePoint bind:x={px} bind:y={py} color="var(--ink-coral)" label="Fixed point P" />
    </Mafs>
  </div>

  <div class="readout" aria-live="polite">
    <div class="row">
      <span class="lab">P (standard)</span>
      <span class="val">({px.toFixed(2)}, {py.toFixed(2)})</span>
    </div>
    <div class="row alt">
      <span class="lab">P in basis {'{v₁, v₂}'}</span>
      {#if singular}
        <span class="val warn">basis collapsed — det = 0</span>
      {:else}
        <span class="val">({c1.toFixed(2)}, {c2.toFixed(2)})</span>
      {/if}
    </div>
    <p class="note">P never moved. Only the address changed.</p>
  </div>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: .85rem;
    background: var(--demo-card); border: 1px solid var(--demo-card-border);
    border-radius: 20px; padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-coral) 45%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: .6rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .hint { margin: 0; font-size: .82rem; color: var(--site-fg-muted); }
  .hint em { font-style: italic; font-family: var(--font-display); }

  .stage { width: 100%; background: var(--demo-stage); border-radius: 12px; overflow: hidden; user-select: none; touch-action: none; }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

  .readout { display: grid; gap: .4rem; padding: .55rem .85rem; background: color-mix(in srgb, var(--site-fg) 4%, transparent); border-radius: 10px; font-family: var(--font-mono); }
  .row { display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; font-size: .95rem; }
  .row.alt { padding-top: .35rem; border-top: 1px dashed color-mix(in srgb, var(--site-fg) 18%, transparent); }
  .lab { color: var(--site-fg-muted); }
  .val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .val.warn { color: var(--ink-coral); }
  .note { margin: .4rem 0 0; font-family: var(--font-body); font-size: .78rem; color: var(--site-fg-muted); }
</style>
