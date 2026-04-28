<script lang="ts">
  import { Mafs, Coordinates, MovablePoint, Line } from 'svelte-mafs';

  let ax = $state(-2);
  let ay = $state(-1);
  let bx = $state(2);
  let by = $state(2);

  const rise = $derived(by - ay);
  const run = $derived(bx - ax);
  const slope = $derived(run === 0 ? NaN : rise / run);

  const fmt = (n: number) => (Number.isFinite(n) ? (n >= 0 ? '+' : '') + n.toFixed(2) : '—');
</script>

<div class="widget">
  <div class="stage">
    <Mafs width={560} height={340} viewBox={{ x: [-4.5, 4.5], y: [-3, 3] }}>
      <Coordinates.Cartesian />
      <Line.ThroughPoints
        point1={[ax, ay]}
        point2={[bx, by]}
        color="var(--ink-red)"
        weight={2}
        opacity={0.85}
      />
      <Line.Segment point1={[ax, ay]} point2={[bx, ay]} color="var(--ink-sea)"   weight={1.5} opacity={0.55} />
      <Line.Segment point1={[bx, ay]} point2={[bx, by]} color="var(--ink-coral)" weight={1.5} opacity={0.55} />
      <MovablePoint bind:x={ax} bind:y={ay} color="var(--ink-red)" />
      <MovablePoint bind:x={bx} bind:y={by} color="var(--ink-red)" />
    </Mafs>
  </div>
  <div class="readout" aria-live="polite">
    <div class="item">
      <span class="label">
        <span class="dot" style="background: var(--ink-coral)"></span>
        rise
      </span>
      <span class="value">{fmt(rise)}</span>
    </div>
    <div class="item">
      <span class="label">
        <span class="dot" style="background: var(--ink-sea)"></span>
        run
      </span>
      <span class="value">{fmt(run)}</span>
    </div>
    <div class="item big">
      <span class="label">slope</span>
      <span class="value accent">{fmt(slope)}</span>
    </div>
    <p class="hint">drag either endpoint · slope = rise / run</p>
  </div>
</div>

<style>
  .widget {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(0.9rem, 2vw, 1.25rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 50%, transparent);
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

  .readout {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    gap: 0.75rem 1.25rem;
    align-items: center;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    font-family: var(--font-mono);
    color: var(--site-fg);
  }
  .item { display: flex; flex-direction: column; gap: 0.15rem; }
  .item.big { grid-row: span 2; text-align: right; }
  .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--site-fg-muted);
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  .dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 999px;
  }
  .value {
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 1rem;
    color: var(--site-fg);
  }
  .item.big .value { font-size: 1.5rem; }
  .value.accent { color: var(--ink-red); }
  .hint {
    grid-column: 1 / -1;
    margin: 0.2rem 0 0;
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
  @media (max-width: 520px) {
    .readout { grid-template-columns: 1fr 1fr; }
    .item.big { grid-column: 1 / -1; grid-row: auto; text-align: left; }
  }
</style>
