<script lang="ts">
  import { Mafs, Coordinates, Plot, Point, Line } from 'svelte-mafs';
  import { burst } from '../../lib/confetti';
  import { play } from '../../lib/sound';
  import { vibrate } from '../../lib/xp';
  import { TINKER_EVENT } from '../../lib/events';

  interface Props {
    /** Hide the slope readout — used for the opening "you can't do this yet" hook. */
    hideReadouts?: boolean;
    /** Lower the slider floor (e.g. 0) so the user can drive h all the way to the wall. */
    enforceH?: number;
    /** Fire confetti + sound + tinker:celebrate the first time h hits 0. */
    celebrateOnZero?: boolean;
    /** Initial value of h. */
    startH?: number;
    /** Where the fixed point P sits on the curve. */
    startP?: number;
    /** Which curve to show. */
    fn?: 'square' | 'cubic';
  }

  let {
    hideReadouts = false,
    enforceH,
    celebrateOnZero = false,
    startH = 1,
    startP = 1,
    fn = 'square',
  }: Props = $props();

  const f = (x: number) => (fn === 'cubic' ? x * x * x : x * x);
  const fnExp = fn === 'cubic' ? '³' : '²';

  const X0 = startP;
  const Y0 = f(X0);

  let h = $state(startH);
  const x2 = $derived(X0 + h);
  const y2 = $derived(f(x2));

  // Secant slope between (X0, Y0) and (X0+h, f(X0+h)).
  // When h is exactly 0 the slope is 0/0 — we surface that as NaN and refuse
  // to draw the secant line, which is the whole pedagogical point.
  const slope = $derived(h === 0 ? Number.NaN : (y2 - Y0) / h);
  const showSecant = $derived(Number.isFinite(slope));

  // Extend the secant across the viewport so it reads as a *line*, not a segment.
  const X_MIN = $derived(fn === 'cubic' ? 0.5 : -0.5);
  const X_MAX = $derived(fn === 'cubic' ? 4 : 3);
  const Y_MIN = $derived(fn === 'cubic' ? -1 : -0.5);
  const Y_MAX = $derived(fn === 'cubic' ? 16 : 5);
  const pA = $derived<[number, number]>([X_MIN, Y0 + slope * (X_MIN - X0)]);
  const pB = $derived<[number, number]>([X_MAX, Y0 + slope * (X_MAX - X0)]);

  const sliderMin = $derived(enforceH ?? 0.01);
  const sliderMax = $derived(fn === 'cubic' ? 1 : 2);

  const fmt = (n: number) =>
    Number.isFinite(n) ? (n >= 0 ? '+' : '') + n.toFixed(3) : '—';

  // Discovery celebration: the first time the user drags h all the way to 0.
  let stage: HTMLDivElement;
  let celebrated = $state(false);
  $effect(() => {
    if (!celebrateOnZero || celebrated) return;
    if (h !== 0) return;
    celebrated = true;
    play('ding');
    vibrate(20);
    if (stage) burst(stage, { count: 14, spread: 1.4 });
    window.dispatchEvent(
      new CustomEvent(TINKER_EVENT.celebrate, { detail: { level: 'step' } }),
    );
  });
</script>

<div class="widget">
  <div class="stage" bind:this={stage}>
    <Mafs width={560} height={360} viewBox={{ x: [X_MIN, X_MAX], y: [Y_MIN, Y_MAX] }}>
      <Coordinates.Cartesian />
      <Plot.OfX y={f} color="var(--ink-red)" weight={2.5} />

      {#if showSecant}
        <!-- The secant, extended as a line. -->
        <Line.Segment point1={pA} point2={pB} color="var(--ink-coral)" weight={2} opacity={0.9} />

        <!-- Rise/run helper: dashed horizontal then vertical from P to Q. -->
        <Line.Segment point1={[X0, Y0]} point2={[x2, Y0]} color="var(--ink-sea)"   weight={1.25} opacity={0.45} />
        <Line.Segment point1={[x2, Y0]} point2={[x2, y2]} color="var(--ink-coral)" weight={1.25} opacity={0.45} />
      {/if}

      <Point x={X0} y={Y0} color="var(--ink-red)" />
      {#if h !== 0}
        <Point x={x2} y={y2} color="var(--ink-coral)" />
      {/if}
    </Mafs>
  </div>

  <div class="controls">
    <label class="slider-row">
      <span class="label">
        <span class="name"><em>h</em></span>
        <span class="value">{h.toFixed(3)}</span>
      </span>
      <input
        type="range"
        min={sliderMin}
        max={sliderMax}
        step="0.01"
        bind:value={h}
        aria-label="Distance h between the two points"
      />
    </label>
    <button
      type="button"
      class="collapse"
      onclick={() => { h = sliderMin; }}
    >Shrink <em>h</em> → 0</button>
  </div>

  {#if !hideReadouts}
    <dl class="readout" aria-live="polite">
      <div class="row">
        <dt>secant slope</dt>
        <dd class="big">{fmt(slope)}</dd>
      </div>
      <div class="row sub">
        <dt>≈ (f({X0.toFixed(0)} + h) − f({X0.toFixed(0)})) / h</dt>
        <dd class="tiny">= (({(X0 + h).toFixed(2)}){fnExp} − {Y0.toFixed(0)}) / {h.toFixed(2)}</dd>
      </div>
    </dl>
  {/if}
</div>

<style>
  .widget {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(0.9rem, 2vw, 1.25rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-coral) 45%, transparent);
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

  .controls {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    flex-wrap: wrap;
  }
  .slider-row {
    flex: 1 1 18rem;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.8rem;
    align-items: center;
  }
  .label {
    display: inline-flex;
    align-items: baseline;
    gap: 0.5rem;
    min-width: 5rem;
  }
  .name em {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--ink-coral);
  }
  .value {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    color: var(--site-fg);
    font-size: 0.95rem;
  }
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
    width: 22px;
    height: 22px;
    border-radius: 999px;
    background: var(--ink-coral);
    border: 2px solid var(--demo-card);
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    cursor: grab;
  }
  input[type="range"]::-moz-range-thumb {
    width: 22px;
    height: 22px;
    border-radius: 999px;
    background: var(--ink-coral);
    border: 2px solid var(--demo-card);
    cursor: grab;
  }
  .collapse {
    padding: 0.55rem 0.9rem;
    border-radius: var(--radius-pill);
    border: 1.5px solid var(--ink-coral);
    color: var(--ink-coral);
    background: transparent;
    font: inherit;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
  }
  .collapse em { font-style: italic; font-family: var(--font-display); }
  .collapse:hover {
    background: color-mix(in srgb, var(--ink-coral) 10%, transparent);
  }

  .readout {
    margin: 0;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    color: var(--site-fg);
  }
  .row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 1rem;
  }
  .row dt {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .row dd {
    margin: 0;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    color: var(--site-fg);
  }
  .row dd.big {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--ink-coral);
  }
  .row.sub { margin-top: 0.3rem; }
  .row.sub dt, .row.sub dd { text-transform: none; letter-spacing: 0; font-size: 0.82rem; color: var(--site-fg-muted); }
  .row.sub dd.tiny { font-size: 0.82rem; }
</style>
