<script lang="ts">
  import { Mafs, Coordinates, Vector, MovablePoint, Polygon, Line } from 'svelte-mafs';

  /**
   * MatrixDraggable — the keystone widget for Module 7 (linear algebra).
   *
   * Two MovablePoints are literally the tips of the two column vectors of a
   * 2×2 matrix. Drag either one and the grid + unit square transform in
   * real time. Subsumes the determinantAreaScrubber (§4.5) idea via
   * orientation-flip coloring: blue for positive det, coral for flipped.
   *
   * Matrix convention:
   *   A = [ a  b ]
   *       [ c  d ]
   *   column 1 = (a, c)   column 2 = (b, d)
   *   det(A) = a d - b c  = signed area of the parallelogram spanned by the columns
   */

  // Default: identity — the unit square sits exactly on the unit square.
  let { initialA = 1, initialB = 0, initialC = 0, initialD = 1 } = $props<{
    initialA?: number;
    initialB?: number;
    initialC?: number;
    initialD?: number;
  }>();

  let px1 = $state(initialA);  // column 1 tip x
  let py1 = $state(initialC);  // column 1 tip y
  let px2 = $state(initialB);  // column 2 tip x
  let py2 = $state(initialD);  // column 2 tip y

  const a = $derived(px1);
  const c = $derived(py1);
  const b = $derived(px2);
  const d = $derived(py2);

  const det = $derived(a * d - b * c);
  const flipped = $derived(det < -0.005);
  const col1Mag = $derived(Math.hypot(a, c));
  const col2Mag = $derived(Math.hypot(b, d));
  const singular = $derived(Math.abs(det) < 0.015);
  const rank = $derived(
    !singular ? 2 :
      (col1Mag < 0.05 && col2Mag < 0.05 ? 0 : 1)
  );

  // Apply A to a point in the original (pre-transform) space.
  const apply = (x: number, y: number): [number, number] => [a * x + b * y, c * x + d * y];

  // Transformed unit square corners — the "shape" that deforms.
  const squareCorners = $derived<[number, number][]>([
    apply(0, 0),
    apply(1, 0),
    apply(1, 1),
    apply(0, 1),
  ]);

  // Reference grid lines, transformed. Each original gridline is still
  // straight after a linear map (that's the whole point of "linear"),
  // so we can draw it as a single Line.Segment between the two transformed
  // endpoints.
  const GRID_RANGE = 3;
  const GRID_INDICES = [-3, -2, -1, 0, 1, 2, 3] as const;
  type L = { a: [number, number]; b: [number, number] };
  const gridLines = $derived<L[]>([
    // verticals (original x = k, y varies)
    ...GRID_INDICES.map((k) => ({
      a: apply(k, -GRID_RANGE),
      b: apply(k, GRID_RANGE),
    })),
    // horizontals (original y = k, x varies)
    ...GRID_INDICES.map((k) => ({
      a: apply(-GRID_RANGE, k),
      b: apply(GRID_RANGE, k),
    })),
  ]);

  // Preset buttons — clicking sets the matrix to a famous transformation.
  function setMatrix(A: number, B: number, C: number, D: number) {
    px1 = A; py1 = C;
    px2 = B; py2 = D;
  }
  function reset() { setMatrix(1, 0, 0, 1); }
  function rotate90() { setMatrix(0, -1, 1, 0); }
  function shear() { setMatrix(1, 1, 0, 1); }
  function collapse() { setMatrix(1, 2, 0.5, 1); } // det = 0, columns collinear

  const fmt = (n: number, dp = 2) => (n >= 0 ? ' ' : '-') + Math.abs(n).toFixed(dp);
</script>

<div class="widget">
  <header class="header">
    <p class="title"><em>A</em> · drag the two column-arrows to define the matrix</p>
    <div class="presets">
      <button type="button" onclick={reset}>I</button>
      <button type="button" onclick={rotate90}>rot 90°</button>
      <button type="button" onclick={shear}>shear</button>
      <button type="button" onclick={collapse} class="danger">collapse</button>
    </div>
  </header>

  <div class="stage">
    <Mafs width={600} height={440} viewBox={{ x: [-3.5, 3.5], y: [-2.6, 2.6] }}>
      <Coordinates.Cartesian />

      <!-- Transformed reference grid. -->
      {#each gridLines as l}
        <Line.Segment
          point1={l.a}
          point2={l.b}
          color="var(--ink-red)"
          weight={0.8}
          opacity={0.22}
        />
      {/each}

      <!-- Transformed unit square. Orientation-aware coloring. -->
      <Polygon
        points={squareCorners}
        color={flipped ? 'var(--ink-coral)' : 'var(--ink-sea)'}
        weight={1.5}
        fillColor={flipped ? 'var(--ink-coral)' : 'var(--ink-sea)'}
        fillOpacity={singular ? 0 : 0.2}
      />

      <!-- Column-vector arrows. -->
      <Vector tail={[0, 0]} tip={[a, c]} color="var(--ink-red)" weight={2.75} />
      <Vector tail={[0, 0]} tip={[b, d]} color="var(--ink-coral)"  weight={2.75} />

      <!-- Draggable tips — the only objects the user controls. -->
      <MovablePoint bind:x={px1} bind:y={py1} color="var(--ink-red)" />
      <MovablePoint bind:x={px2} bind:y={py2} color="var(--ink-coral)"  />
    </Mafs>
  </div>

  <div class="readout" aria-live="polite">
    <div class="matrix">
      <span class="bracket">[</span>
      <div class="entries">
        <span class="ent e-a">{fmt(a)}</span>
        <span class="ent e-b">{fmt(b)}</span>
        <span class="ent e-c">{fmt(c)}</span>
        <span class="ent e-d">{fmt(d)}</span>
      </div>
      <span class="bracket">]</span>
    </div>
    <dl class="stats">
      <div class="stat">
        <dt>det</dt>
        <dd class:neg={flipped} class:zero={singular}>{fmt(det)}</dd>
      </div>
      <div class="stat">
        <dt>rank</dt>
        <dd class={`rank r${rank}`}>{rank}</dd>
      </div>
      {#if flipped}
        <div class="flag flag-coral">orientation flipped</div>
      {:else if singular}
        <div class="flag flag-warn">singular — space squashed</div>
      {/if}
    </dl>
  </div>
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 45%, transparent);
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
  }
  .title {
    margin: 0;
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 1rem;
    color: var(--site-fg);
  }
  .title em { font-style: italic; }
  .presets { display: flex; gap: 0.35rem; flex-wrap: wrap; }
  .presets button {
    padding: 0.35rem 0.7rem;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    border: 1px solid var(--site-border);
    border-radius: 999px;
    background: var(--demo-card);
    color: var(--site-fg);
    cursor: pointer;
  }
  .presets button:hover { border-color: var(--ink-red); color: var(--ink-red); }
  .presets .danger:hover { border-color: var(--ink-coral); color: var(--ink-coral); }

  .stage {
    width: 100%;
    background: var(--demo-stage);
    border-radius: 12px;
    overflow: hidden;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
  }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

  .readout {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 1rem;
    align-items: center;
    padding-top: 0.55rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
  @media (max-width: 560px) {
    .readout { grid-template-columns: 1fr; }
  }

  .matrix {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-family: var(--font-mono);
  }
  .bracket {
    font-size: 2.5rem;
    font-weight: 200;
    color: var(--site-fg-muted);
    line-height: 1;
  }
  .entries {
    display: grid;
    grid-template-columns: auto auto;
    gap: 0.1rem 0.8rem;
    padding: 0.25rem 0.1rem;
  }
  .ent {
    font-variant-numeric: tabular-nums;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--site-fg);
  }
  .ent.e-a, .ent.e-c { color: var(--ink-red); }
  .ent.e-b, .ent.e-d { color: var(--ink-coral); }

  .stats {
    display: flex;
    gap: 1.3rem;
    align-items: center;
    flex-wrap: wrap;
    margin: 0;
    font-family: var(--font-mono);
  }
  .stat { display: flex; flex-direction: column; gap: 0.1rem; }
  .stat dt {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--site-fg-muted);
  }
  .stat dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--site-fg);
  }
  .stat dd.neg  { color: var(--ink-coral); }
  .stat dd.zero { color: var(--ink-sun); }

  .rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.6rem;
    padding: 0.12rem 0.55rem;
    border-radius: 999px;
    font-size: 0.95rem;
  }
  .rank.r2 { background: color-mix(in srgb, var(--cta) 18%, transparent); color: var(--cta-hover); }
  .rank.r1 { background: color-mix(in srgb, var(--ink-sun) 22%, transparent); color: color-mix(in srgb, var(--ink-sun) 80%, var(--site-fg)); }
  .rank.r0 { background: color-mix(in srgb, var(--ink-coral) 18%, transparent); color: var(--ink-coral); }

  .flag {
    font-family: var(--font-body);
    font-size: 0.78rem;
    font-weight: 650;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
  }
  .flag-coral { background: color-mix(in srgb, var(--ink-coral) 18%, transparent); color: var(--ink-coral); }
  .flag-warn  { background: color-mix(in srgb, var(--ink-sun)   22%, transparent); color: color-mix(in srgb, var(--ink-sun) 80%, var(--site-fg)); }
</style>
