<script lang="ts">
  import { Mafs, Coordinates, Plot, Point, Line } from 'svelte-mafs';

  /**
   * f(g(x)) = (sin(x))² — picked because:
   *   - both g and f have familiar, recognizable shapes,
   *   - g'(x) = cos(x) is clearly non-constant (so the two slopes actually
   *     DO multiply — not a degenerate 1× case),
   *   - the product simplifies to the identity sin(2x), so a learner can
   *     sanity-check by plotting sin(2x) and seeing it match the product.
   */
  const g  = (x: number) => Math.sin(x);
  const gp = (x: number) => Math.cos(x);
  const f  = (u: number) => u * u;
  const fp = (u: number) => 2 * u;

  let x = $state(1);
  const gx = $derived(g(x));
  const fgx = $derived(f(gx));
  const gpx = $derived(gp(x));
  const fpgx = $derived(fp(gx));
  const product = $derived(fpgx * gpx);

  // Tangent line endpoints for the inner function g on x∈[-3,3].
  const X_MIN = -3;
  const X_MAX = 3;
  const gTangentA = $derived<[number, number]>([X_MIN, gx + gpx * (X_MIN - x)]);
  const gTangentB = $derived<[number, number]>([X_MAX, gx + gpx * (X_MAX - x)]);

  // Tangent line endpoints for the outer function f on u∈[-1.2, 1.2].
  const U_MIN = -1.2;
  const U_MAX = 1.2;
  const fTangentA = $derived<[number, number]>([U_MIN, fgx + fpgx * (U_MIN - gx)]);
  const fTangentB = $derived<[number, number]>([U_MAX, fgx + fpgx * (U_MAX - gx)]);

  const fmt = (n: number, d = 2) => (n >= 0 ? '+' : '') + n.toFixed(d);
</script>

<div class="widget">
  <header class="header">
    <p class="title"><em>f</em>(<em>g</em>(<em>x</em>)) = (sin <em>x</em>)²</p>
    <p class="hint">drag the slider — watch the two slopes multiply</p>
  </header>

  <!-- Inner function: y = g(x) = sin(x) -->
  <div class="stage">
    <div class="stage-label">
      <span class="pill pill-violet">inner</span>
      <span class="formula">
        <em>g</em>(<em>x</em>) = sin <em>x</em>
      </span>
    </div>
    <Mafs width={640} height={200} viewBox={{ x: [X_MIN, X_MAX], y: [-1.4, 1.4] }}>
      <Coordinates.Cartesian />
      <Plot.OfX y={g} color="var(--ink-red)" weight={2.5} />
      <Line.Segment point1={gTangentA} point2={gTangentB} color="var(--ink-coral)" weight={2} opacity={0.9} />
      <Point x={x} y={gx} color="var(--ink-coral)" />
      <!-- Vertical guide down to x axis -->
      <Line.Segment point1={[x, 0]} point2={[x, gx]} color="var(--ink-sea)" weight={1} opacity={0.45} />
    </Mafs>
  </div>

  <!-- Outer function: z = f(u) = u² -->
  <div class="stage">
    <div class="stage-label">
      <span class="pill pill-sea">outer</span>
      <span class="formula">
        <em>f</em>(<em>u</em>) = <em>u</em>² &nbsp;·&nbsp; <em>u</em> = <em>g</em>(<em>x</em>) = {gx.toFixed(2)}
      </span>
    </div>
    <Mafs width={640} height={220} viewBox={{ x: [U_MIN, U_MAX], y: [-0.2, 1.5] }}>
      <Coordinates.Cartesian />
      <Plot.OfX y={f} color="var(--ink-sea)" weight={2.5} domain={[U_MIN, U_MAX]} />
      <Line.Segment point1={fTangentA} point2={fTangentB} color="var(--ink-coral)" weight={2} opacity={0.9} />
      <Point x={gx} y={fgx} color="var(--ink-coral)" />
      <!-- Vertical guide from u axis up to the point -->
      <Line.Segment point1={[gx, 0]} point2={[gx, fgx]} color="var(--ink-sea)" weight={1} opacity={0.45} />
    </Mafs>
  </div>

  <label class="slider">
    <span class="slider-label"><em>x</em> = {x.toFixed(2)}</span>
    <input
      type="range"
      min={X_MIN}
      max={X_MAX}
      step="0.01"
      bind:value={x}
      aria-label="Input value x"
    />
  </label>

  <dl class="readout" aria-live="polite">
    <div class="row">
      <dt><em>g</em>′(<em>x</em>)</dt>
      <dd class="val violet">{fmt(gpx)}</dd>
      <dt class="sub">rate of the inner</dt>
    </div>
    <div class="op">×</div>
    <div class="row">
      <dt><em>f</em>′(<em>g</em>(<em>x</em>))</dt>
      <dd class="val sea">{fmt(fpgx)}</dd>
      <dt class="sub">rate of the outer, at <em>u</em>={gx.toFixed(2)}</dt>
    </div>
    <div class="op">=</div>
    <div class="row sum">
      <dt>(<em>f</em>∘<em>g</em>)′(<em>x</em>)</dt>
      <dd class="val big">{fmt(product)}</dd>
      <dt class="sub">total rate — they multiplied</dt>
    </div>
  </dl>
</div>

<style>
  .widget {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 50%, transparent);
  }

  .header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0 0 0.2rem;
  }
  .header .title {
    margin: 0;
    font-family: var(--font-display);
    font-weight: 600;
    font-size: 1.1rem;
    color: var(--site-fg);
  }
  .header .title em { font-style: italic; }
  .header .hint {
    margin: 0;
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--site-fg-muted);
  }

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

  .stage-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.85rem;
    border-bottom: 1px solid color-mix(in srgb, var(--site-fg) 8%, transparent);
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--site-fg);
  }
  .pill {
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
  }
  .pill-violet { background: color-mix(in srgb, var(--ink-red) 18%, transparent); color: var(--ink-red); }
  .pill-sea    { background: color-mix(in srgb, var(--ink-sea) 18%, transparent);    color: var(--ink-sea); }
  .formula em { font-style: italic; font-family: var(--font-display); font-size: 1.05em; }

  .slider {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.9rem;
    align-items: center;
    padding: 0.3rem 0.2rem 0;
  }
  .slider-label {
    font-family: var(--font-mono);
    font-size: 1rem;
    color: var(--site-fg);
    min-width: 5.5rem;
  }
  .slider-label em { font-style: italic; color: var(--ink-red); font-family: var(--font-display); font-size: 1.1em; }
  input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--site-fg) 18%, transparent);
    outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 22px; height: 22px;
    border-radius: 999px;
    background: var(--ink-coral);
    border: 2px solid var(--demo-card);
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
    cursor: grab;
  }
  input[type="range"]::-moz-range-thumb {
    width: 22px; height: 22px;
    border-radius: 999px;
    background: var(--ink-coral);
    border: 2px solid var(--demo-card);
    cursor: grab;
  }

  .readout {
    margin: 0;
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1fr;
    gap: 0.75rem;
    align-items: center;
    padding-top: 0.75rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
  .row { display: flex; flex-direction: column; gap: 0.15rem; min-width: 0; }
  .row dt {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--site-fg);
  }
  .row dt em { font-style: italic; font-family: var(--font-display); font-size: 1.05em; }
  .row dt.sub {
    font-size: 0.72rem;
    color: var(--site-fg-muted);
    text-transform: none;
    letter-spacing: 0;
  }
  .row .val {
    margin: 0;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    font-size: 1.15rem;
  }
  .row .val.violet { color: var(--ink-red); }
  .row .val.sea    { color: var(--ink-sea); }
  .row .val.big {
    font-size: 1.4rem;
    color: var(--ink-coral);
  }
  .op {
    font-family: var(--font-display);
    font-size: 1.5rem;
    color: var(--site-fg-muted);
    text-align: center;
  }

  @media (max-width: 640px) {
    .readout {
      grid-template-columns: 1fr;
      gap: 0.4rem;
    }
    .op { text-align: left; font-size: 1rem; padding-left: 0.2rem; }
  }
</style>
