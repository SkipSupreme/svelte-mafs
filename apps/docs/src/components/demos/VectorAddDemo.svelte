<script lang="ts">
  import { Mafs, Coordinates, MovablePoint, Vector, Line } from 'svelte-mafs';

  let ax = $state(2);
  let ay = $state(1);
  let bx = $state(1);
  let by = $state(2);

  const sx = $derived(ax + bx);
  const sy = $derived(ay + by);

  const magA = $derived(Math.hypot(ax, ay));
  const magB = $derived(Math.hypot(bx, by));
  const magS = $derived(Math.hypot(sx, sy));
</script>

<div class="demo">
  <div class="stage">
    <Mafs width={560} height={360} viewBox={{ x: [-4.5, 4.5], y: [-2, 4] }}>
      <Coordinates.Cartesian />

      <!-- Parallelogram helpers -->
      <Line.Segment point1={[ax, ay]} point2={[sx, sy]} color="var(--ink-sea)" opacity={0.35} weight={1.25} />
      <Line.Segment point1={[bx, by]} point2={[sx, sy]} color="var(--ink-red)" opacity={0.35} weight={1.25} />

      <Vector tail={[0, 0]} tip={[ax, ay]} color="var(--ink-red)" weight={2.5} />
      <Vector tail={[0, 0]} tip={[bx, by]} color="var(--ink-sea)" weight={2.5} />
      <Vector tail={[0, 0]} tip={[sx, sy]} color="var(--ink-coral)" weight={2.75} />

      <MovablePoint bind:x={ax} bind:y={ay} color="var(--ink-red)" />
      <MovablePoint bind:x={bx} bind:y={by} color="var(--ink-sea)" />
    </Mafs>
  </div>

  <div class="readout" aria-live="polite">
    <div class="row">
      <span class="swatch swatch-violet" aria-hidden="true"></span>
      <span class="name"><em>a</em></span>
      <span class="coord">({ax.toFixed(2)}, {ay.toFixed(2)})</span>
      <span class="mag">|{magA.toFixed(2)}|</span>
    </div>
    <div class="row">
      <span class="swatch swatch-sea" aria-hidden="true"></span>
      <span class="name"><em>b</em></span>
      <span class="coord">({bx.toFixed(2)}, {by.toFixed(2)})</span>
      <span class="mag">|{magB.toFixed(2)}|</span>
    </div>
    <div class="row sum">
      <span class="swatch swatch-coral" aria-hidden="true"></span>
      <span class="name"><em>a + b</em></span>
      <span class="coord">({sx.toFixed(2)}, {sy.toFixed(2)})</span>
      <span class="mag">|{magS.toFixed(2)}|</span>
    </div>
    <p class="hint">drag either arrow's dot — the sum follows</p>
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 55%, transparent);
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
    display: grid;
    gap: 0.35rem;
    padding-top: 0.45rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    font-family: var(--font-mono);
    font-size: 0.92rem;
    color: var(--site-fg);
  }
  .row {
    display: grid;
    grid-template-columns: auto auto 1fr auto;
    gap: 0.65rem;
    align-items: center;
    color: var(--site-fg);
  }
  .row.sum {
    padding-top: 0.2rem;
    border-top: 1px dashed color-mix(in srgb, var(--site-fg) 22%, transparent);
    margin-top: 0.15rem;
  }
  .name em {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 1.05rem;
    font-weight: 600;
  }
  .coord {
    font-variant-numeric: tabular-nums;
  }
  .mag {
    color: var(--site-fg-muted);
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
    justify-self: end;
  }
  .swatch {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 2px;
  }
  .swatch-violet { background: var(--ink-red); }
  .swatch-sea    { background: var(--ink-sea); }
  .swatch-coral  { background: var(--ink-coral); }

  .hint {
    margin: 0.3rem 0 0;
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
</style>
