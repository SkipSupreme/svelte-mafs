<script lang="ts">
  import { Mafs, Coordinates, MovablePoint, Vector, Point, Line } from 'svelte-mafs';

  interface Props {
    /** Initial point cloud. */
    initialPoints?: [number, number][];
  }

  function defaultPoints(): [number, number][] {
    // Loosely linear by default — gives r ≈ 0.85, Cov > 0.
    return [
      [-2.5, -1.8],
      [-1.4, -1.1],
      [-0.6, -0.4],
      [0.2, 0.4],
      [1.1, 0.6],
      [2.2, 1.5],
    ];
  }

  let { initialPoints = defaultPoints() }: Props = $props();

  // Make each point a tuple of $state cells so MovablePoint's bind:x/bind:y can write back.
  // We use a single $state object with x/y so bind: works on properties.
  type Pt = { x: number; y: number };
  let points: Pt[] = $state(initialPoints.map(([x, y]) => ({ x, y })));

  function addPoint() {
    points = [...points, { x: 0, y: 0 }];
  }
  function removePoint() {
    if (points.length > 2) points = points.slice(0, -1);
  }
  function reset() {
    points = initialPoints.map(([x, y]) => ({ x, y }));
  }

  // ---- Stats ----
  const xbar = $derived(points.reduce((s, p) => s + p.x, 0) / points.length);
  const ybar = $derived(points.reduce((s, p) => s + p.y, 0) / points.length);

  // Sample covariance (with n-1 in the denominator — the unbiased estimator).
  const cov = $derived.by(() => {
    if (points.length < 2) return 0;
    let s = 0;
    for (const p of points) s += (p.x - xbar) * (p.y - ybar);
    return s / (points.length - 1);
  });

  // Sample standard deviations.
  const sx = $derived.by(() => {
    if (points.length < 2) return 0;
    let s = 0;
    for (const p of points) s += (p.x - xbar) ** 2;
    return Math.sqrt(s / (points.length - 1));
  });
  const sy = $derived.by(() => {
    if (points.length < 2) return 0;
    let s = 0;
    for (const p of points) s += (p.y - ybar) ** 2;
    return Math.sqrt(s / (points.length - 1));
  });

  // Pearson correlation = cosine similarity of centered data.
  const r = $derived(sx > 0 && sy > 0 ? cov / (sx * sy) : 0);

  // For an angular display: the angle whose cosine is r.
  const angleDeg = $derived((Math.acos(Math.max(-1, Math.min(1, r))) * 180) / Math.PI);

  function fmt(x: number, d = 2): string { return x.toFixed(d); }
</script>

<div class="widget">
  <div class="stage">
    <Mafs width={520} height={360} viewBox={{ x: [-4, 4], y: [-3, 3] }}>
      <Coordinates.Cartesian />

      <!-- Centered-vector arrows from centroid to each point -->
      {#each points as p, _i}
        <Vector
          tail={[xbar, ybar]}
          tip={[p.x, p.y]}
          color="var(--ink-coral)"
          opacity={0.35}
        />
      {/each}

      <!-- The centroid as a crosshair. -->
      <Line.Segment point1={[xbar, -3]} point2={[xbar, 3]} color="var(--ink-sea)" weight={0.75} opacity={0.4} />
      <Line.Segment point1={[-4, ybar]} point2={[4, ybar]} color="var(--ink-sea)" weight={0.75} opacity={0.4} />
      <Point x={xbar} y={ybar} color="var(--ink-sea)" />

      <!-- Movable data points -->
      {#each points as p, i (i)}
        <MovablePoint
          bind:x={points[i].x}
          bind:y={points[i].y}
          color="var(--ink-red)"
          label={`Data point ${i + 1}`}
        />
      {/each}
    </Mafs>
  </div>

  <div class="readout">
    <div class="stats">
      <div class="stat">
        <span class="k">x̄</span>
        <span class="v">{fmt(xbar)}</span>
      </div>
      <div class="stat">
        <span class="k">ȳ</span>
        <span class="v">{fmt(ybar)}</span>
      </div>
      <div class="sep" aria-hidden="true">·</div>
      <div class="stat">
        <span class="k">s<sub>x</sub></span>
        <span class="v">{fmt(sx)}</span>
      </div>
      <div class="stat">
        <span class="k">s<sub>y</sub></span>
        <span class="v">{fmt(sy)}</span>
      </div>
      <div class="sep" aria-hidden="true">·</div>
      <div class="stat highlight">
        <span class="k">Cov(X,Y)</span>
        <span class="v">{fmt(cov)}</span>
      </div>
      <div class="stat highlight">
        <span class="k">r</span>
        <span class="v">{fmt(r, 3)}</span>
      </div>
      <div class="stat">
        <span class="k">∠ = arccos r</span>
        <span class="v">{fmt(angleDeg, 1)}°</span>
      </div>
    </div>
    <div class="buttons">
      <button type="button" class="btn" onclick={addPoint}>Add point</button>
      <button type="button" class="btn" onclick={removePoint} disabled={points.length <= 2}>Remove point</button>
      <button type="button" class="btn ghost" onclick={reset}>Reset</button>
    </div>
  </div>
  <p class="hint">
    Drag points. The teal crosshair is the centroid <em>(x̄, ȳ)</em>. Coral arrows are the centered vectors <em>(x<sub>i</sub> − x̄, y<sub>i</sub> − ȳ)</em>. <strong>Cov(X, Y)</strong> is the average of the products of those arrow components; <strong>r</strong> is their cosine similarity — try pulling points into a tight diagonal and watch r approach ±1.
  </p>
</div>

<style>
  .widget {
    background: var(--demo-card);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 1px 0 rgba(0,0,0,0.04), 0 12px 32px -24px rgba(0,0,0,0.18);
  }
  .stage {
    background: var(--demo-stage);
    border-radius: 12px;
    overflow: hidden;
  }
  .readout {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-top: 12px;
    flex-wrap: wrap;
  }
  .stats {
    display: flex;
    gap: 14px;
    align-items: baseline;
    font-family: var(--font-mono);
    font-size: 13px;
    flex-wrap: wrap;
  }
  .stat { display: inline-flex; gap: 4px; align-items: baseline; }
  .stat .k { color: var(--site-fg-muted); }
  .stat .v { color: var(--site-fg); font-weight: 600; }
  .stat.highlight .k { color: color-mix(in srgb, var(--ink-red) 60%, var(--site-fg-muted)); }
  .stat.highlight .v { color: var(--ink-red); }
  .sep { color: color-mix(in srgb, var(--site-fg) 30%, transparent); }
  .buttons { display: flex; gap: 8px; flex-wrap: wrap; }
  .btn {
    font-family: var(--font-mono);
    font-size: 13px;
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: var(--demo-stage);
    color: var(--site-fg);
    cursor: pointer;
    transition: transform 120ms ease-out;
  }
  .btn:hover:not(:disabled) { transform: translateY(-1px); }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn.ghost { background: transparent; }
  .hint {
    margin: 10px 0 0;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--site-fg-muted);
  }
  .hint em { font-style: normal; color: var(--site-fg); }
  .hint strong { color: var(--ink-red); font-weight: 600; }
</style>
