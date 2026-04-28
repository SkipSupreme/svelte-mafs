<script lang="ts">
  import { Mafs, Coordinates, Ellipse, Vector, Point, Plot, MovablePoint, Line } from 'svelte-mafs';

  /**
   * GradientCompass — the keystone widget for Module 6 (multivariable calculus).
   *
   * Function: f(x, y) = x² + 2y²  (an elliptical bowl centered at origin).
   * Gradient: ∇f = (2x, 4y).
   * Level curves: x² + 2y² = c — ellipses with semi-axes (√c, √(c/2)).
   *
   * Pedagogical flow (per research brief §4.2):
   *   • Drag p anywhere on the contour plot.
   *   • See ∇f(p) as a violet arrow (lives in input space, perpendicular to
   *     the level curve through p).
   *   • Rotate the unit vector u via the angle slider.
   *   • Watch D_u f = ∇f · u change, tracing a cosine in the side plot.
   *   • Discover that the peak is at u = ∇f / |∇f|.
   */

  // Inner-product derivative: ∇f(p) · u.
  const f  = (x: number, y: number) => x * x + 2 * y * y;
  const gx = (x: number) => 2 * x;
  const gy = (y: number) => 4 * y;

  // The probe point p.
  let px = $state(1);
  let py = $state(0.6);

  // The direction angle θ (radians). User rotates u via slider or ±π/12 buttons.
  let theta = $state(Math.PI / 4);

  // Gradient at p.
  const grad = $derived<[number, number]>([gx(px), gy(py)]);
  const gradMag = $derived(Math.hypot(grad[0], grad[1]));
  const gradAngle = $derived(Math.atan2(grad[1], grad[0])); // the θ where D_u f peaks

  // Unit direction u (length 1 in user units — drawn as a visibly-sized arrow).
  const ux = $derived(Math.cos(theta));
  const uy = $derived(Math.sin(theta));

  // Directional derivative — the number the lesson is about.
  const duf = $derived(grad[0] * ux + grad[1] * uy);

  // For the cosine-curve side plot: D_u f as a function of θ.
  const dufAt = (t: number) => grad[0] * Math.cos(t) + grad[1] * Math.sin(t);

  // Level curves to draw (contours of x² + 2y²).
  const levels = [0.5, 1.25, 2.5, 4.5, 7];
  const levelRadii = levels.map((c) => [Math.sqrt(c), Math.sqrt(c / 2)] as [number, number]);

  const fmt = (n: number, d = 2) => (n >= 0 ? '+' : '') + n.toFixed(d);
  const fmtAngle = (r: number) => {
    const deg = ((r * 180) / Math.PI) % 360;
    const n = deg < 0 ? deg + 360 : deg;
    return n.toFixed(0) + '°';
  };

  function nudge(delta: number) {
    theta = (theta + delta + 2 * Math.PI) % (2 * Math.PI);
  }
  function snapToGradient() {
    theta = gradAngle;
  }
</script>

<div class="widget">
  <header class="header">
    <p class="title"><em>f</em>(<em>x</em>, <em>y</em>) = <em>x</em>² + 2<em>y</em>²</p>
    <p class="hint">drag <strong>p</strong> · rotate <em>u</em> · watch the cosine emerge</p>
  </header>

  <div class="grid">
    <!-- LEFT: contour plot with probe, gradient, u. -->
    <div class="stage">
      <Mafs width={420} height={420} viewBox={{ x: [-3, 3], y: [-2.3, 2.3] }}>
        <Coordinates.Cartesian />
        <!-- Level curves (ellipses): x² + 2y² = c -->
        {#each levelRadii as [rx, ry], i}
          <Ellipse
            center={[0, 0]}
            radius={[rx, ry]}
            color="var(--site-fg-muted)"
            weight={1.25}
            opacity={0.38}
          />
        {/each}

        <!-- Gradient arrow at p (scaled to fit the viewport visibly). -->
        <Vector
          tail={[px, py]}
          tip={[px + grad[0] * 0.35, py + grad[1] * 0.35]}
          color="var(--ink-red)"
          weight={2.75}
        />

        <!-- Unit direction arrow u at p. -->
        <Vector
          tail={[px, py]}
          tip={[px + ux, py + uy]}
          color="var(--ink-coral)"
          weight={2.5}
        />

        <MovablePoint bind:x={px} bind:y={py} color="var(--ink-red)" />
      </Mafs>
      <dl class="sidebar">
        <div>
          <dt><span class="dot violet"></span>∇<em>f</em>(<em>p</em>)</dt>
          <dd>({grad[0].toFixed(2)}, {grad[1].toFixed(2)})</dd>
          <dd class="sub">|∇<em>f</em>| = {gradMag.toFixed(2)}</dd>
        </div>
        <div>
          <dt><span class="dot coral"></span><em>u</em></dt>
          <dd>{fmtAngle(theta)} · ({ux.toFixed(2)}, {uy.toFixed(2)})</dd>
        </div>
        <div class="big">
          <dt>D<sub><em>u</em></sub><em>f</em></dt>
          <dd class="big-val">{fmt(duf)}</dd>
        </div>
      </dl>
    </div>

    <!-- RIGHT: the cosine curve D_u f vs θ. -->
    <div class="side">
      <p class="side-label">D<sub><em>u</em></sub><em>f</em> as you rotate <em>u</em>:</p>
      <div class="cos-stage">
        <Mafs width={420} height={260} viewBox={{ x: [0, 2 * Math.PI], y: [-gradMag - 0.5, gradMag + 0.5] }}>
          <Coordinates.Cartesian />
          <Plot.OfX y={dufAt} color="var(--ink-red)" weight={2.5} />
          <!-- peak marker (at θ = gradAngle, if in [0, 2π]) -->
          {@const peakT = gradAngle < 0 ? gradAngle + 2 * Math.PI : gradAngle}
          <Line.Segment point1={[peakT, -gradMag - 0.5]} point2={[peakT, gradMag + 0.5]} color="var(--ink-sea)" weight={1} opacity={0.35} />
          <Point x={peakT} y={gradMag} color="var(--ink-sea)" />

          <!-- current θ position on the cosine -->
          {@const curT = theta < 0 ? theta + 2 * Math.PI : theta}
          <Line.Segment point1={[curT, 0]} point2={[curT, duf]} color="var(--ink-coral)" weight={1.25} opacity={0.7} />
          <Point x={curT} y={duf} color="var(--ink-coral)" />
        </Mafs>
      </div>
      <p class="side-note">
        <span class="dot sea"></span>peak at
        <code>θ = {fmtAngle(gradAngle)}</code>
        — the angle of ∇<em>f</em>.
        Max value: <strong>{gradMag.toFixed(2)}</strong> = |∇<em>f</em>|.
      </p>
    </div>
  </div>

  <div class="controls">
    <label class="slider-row">
      <span class="slider-label">
        rotate <em>u</em>:
        <span class="val">{fmtAngle(theta)}</span>
      </span>
      <input
        type="range"
        min="0"
        max={2 * Math.PI}
        step="0.02"
        bind:value={theta}
        aria-label="Rotate the unit vector u"
      />
    </label>
    <div class="buttons">
      <button type="button" onclick={() => nudge(-Math.PI / 12)} aria-label="Rotate left">←</button>
      <button type="button" onclick={() => nudge(Math.PI / 12)} aria-label="Rotate right">→</button>
      <button type="button" class="snap" onclick={snapToGradient}>snap to ∇<em>f</em></button>
    </div>
  </div>
</div>

<style>
  .widget {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 55%, transparent);
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0;
  }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1.1rem; color: var(--site-fg); }
  .title em { font-style: italic; }
  .hint { margin: 0; font-size: 0.82rem; color: var(--site-fg-muted); }
  .hint em { font-style: italic; }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    align-items: start;
  }
  @media (max-width: 760px) {
    .grid { grid-template-columns: 1fr; }
  }

  .stage, .cos-stage {
    width: 100%;
    background: var(--demo-stage);
    border-radius: 12px;
    overflow: hidden;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
  }
  .stage :global(svg), .cos-stage :global(svg) {
    display: block; width: 100%; height: auto; max-width: 100%;
  }

  .sidebar {
    margin: 0.5rem 0 0;
    display: grid;
    gap: 0.4rem;
    padding: 0.55rem 0.75rem;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-radius: 10px;
    font-family: var(--font-mono);
  }
  .sidebar > div { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.6rem; }
  .sidebar dt {
    font-size: 0.82rem;
    color: var(--site-fg-muted);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .sidebar dt em { font-style: italic; font-family: var(--font-display); font-size: 1em; }
  .sidebar dd { margin: 0; font-size: 0.9rem; color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .sidebar .sub { color: var(--site-fg-muted); font-size: 0.8rem; }
  .sidebar .big { padding-top: 0.2rem; border-top: 1px dashed color-mix(in srgb, var(--site-fg) 15%, transparent); }
  .sidebar .big-val { font-size: 1.3rem; font-weight: 700; color: var(--ink-coral); }

  .dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 999px;
  }
  .dot.violet { background: var(--ink-red); }
  .dot.coral  { background: var(--ink-coral); }
  .dot.sea    { background: var(--ink-sea); }

  .side-label {
    margin: 0 0 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .side-label em { font-style: italic; font-family: var(--font-display); font-size: 1.1em; }
  .side-note {
    margin: 0.45rem 0 0;
    font-size: 0.8rem;
    color: var(--site-fg-muted);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    flex-wrap: wrap;
  }
  .side-note em { font-style: italic; }
  .side-note code {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    padding: 0.1rem 0.4rem;
    background: color-mix(in srgb, var(--ink-sea) 15%, transparent);
    color: var(--site-fg);
    border-radius: 4px;
  }

  .controls {
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 10%, transparent);
  }
  .slider-row {
    flex: 1 1 18rem;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.8rem;
    align-items: center;
  }
  .slider-label {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    color: var(--site-fg);
    min-width: 8rem;
  }
  .slider-label em { font-style: italic; font-family: var(--font-display); font-size: 1.1em; }
  .slider-label .val { color: var(--ink-coral); font-weight: 700; }
  input[type="range"] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 6px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--site-fg) 18%, transparent);
    outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 22px; height: 22px; border-radius: 999px;
    background: var(--ink-coral);
    border: 2px solid var(--demo-card);
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    cursor: grab;
  }
  input[type="range"]::-moz-range-thumb {
    width: 22px; height: 22px; border-radius: 999px;
    background: var(--ink-coral);
    border: 2px solid var(--demo-card);
    cursor: grab;
  }
  .buttons { display: flex; gap: 0.4rem; }
  .buttons button {
    padding: 0.4rem 0.7rem;
    font: inherit;
    font-size: 0.82rem;
    border: 1px solid var(--site-border);
    border-radius: var(--radius-pill);
    background: var(--demo-card);
    color: var(--site-fg);
    cursor: pointer;
    font-family: var(--font-mono);
  }
  .buttons button:hover { border-color: var(--ink-coral); color: var(--ink-coral); }
  .buttons .snap { font-family: var(--font-body); font-weight: 650; }
  .buttons .snap em { font-style: italic; font-family: var(--font-display); }
</style>
