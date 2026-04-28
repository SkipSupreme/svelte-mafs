<script lang="ts">
  import {
    Mafs,
    Coordinates,
    Plot,
    MovablePoint,
    snapToCurve,
    Line,
  } from 'svelte-mafs';

  const f = (x: number) => Math.sin(x);
  const df = (x: number) => Math.cos(x);

  const X_MIN = -4.5;
  const X_MAX = 4.5;

  let x = $state(1);
  let y = $state(f(1));

  const slope = $derived(df(x));
  const yInt = $derived(y - slope * x);

  const tangentA = $derived<[number, number]>([X_MIN, slope * X_MIN + yInt]);
  const tangentB = $derived<[number, number]>([X_MAX, slope * X_MAX + yInt]);

  const onCurve = snapToCurve(
    (t) => [t, f(t)] as const,
    [X_MIN, X_MAX],
    400,
  );

  const fmt = (n: number) => (n >= 0 ? `+${n.toFixed(2)}` : n.toFixed(2));
</script>

<div class="demo">
  <div class="stage">
    <Mafs width={560} height={280} viewBox={{ x: [X_MIN, X_MAX], y: [-1.8, 1.8] }}>
      <Coordinates.Cartesian />
      <Plot.OfX y={f} color="var(--ink-red)" weight={2.5} />
      <Line.Segment
        point1={tangentA}
        point2={tangentB}
        color="var(--ink-coral)"
        weight={2}
        opacity={0.9}
      />
      <MovablePoint bind:x bind:y constrain={onCurve} color="var(--ink-coral)" />
    </Mafs>
  </div>
  <div class="readout" aria-live="polite">
    <div class="readout-item">
      <span class="label">point</span>
      <span class="value">({x.toFixed(2)}, {y.toFixed(2)})</span>
    </div>
    <div class="divider" aria-hidden="true"></div>
    <div class="readout-item">
      <span class="label">slope</span>
      <span class="value slope">{fmt(slope)}</span>
    </div>
    <p class="hint">drag the dot · or tab + arrow keys</p>
  </div>
</div>

<style>
  .demo {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 55%, transparent);
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
  .stage :global(svg) {
    display: block;
    width: 100%;
    height: auto;
    max-width: 100%;
  }

  .readout {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    padding-top: 0.35rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
    font-family: var(--font-mono);
    color: var(--site-fg);
  }
  .readout-item {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
  }
  .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--site-fg-muted);
  }
  .value {
    font-variant-numeric: tabular-nums;
    color: var(--site-fg);
    font-weight: 600;
    font-size: 1rem;
  }
  .value.slope {
    font-size: 1.25rem;
    color: var(--ink-coral);
  }
  .divider {
    width: 1px;
    align-self: stretch;
    background: color-mix(in srgb, var(--site-fg) 18%, transparent);
  }
  .hint {
    margin: 0 0 0 auto;
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
  @media (max-width: 520px) {
    .hint { margin-left: 0; width: 100%; }
  }
</style>
