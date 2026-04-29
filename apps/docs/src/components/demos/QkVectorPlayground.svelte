<script lang="ts">
  import { Mafs, Coordinates, MovablePoint, Vector } from 'svelte-mafs';

  let qx = $state(1.4);
  let qy = $state(0.6);
  let k1x = $state(2);
  let k1y = $state(0.2);
  let k2x = $state(-0.4);
  let k2y = $state(2);

  let mode = $state<'soft' | 'hard'>('soft');

  const s1 = $derived(qx * k1x + qy * k1y);
  const s2 = $derived(qx * k2x + qy * k2y);

  const softWeights = $derived.by(() => {
    const m = Math.max(s1, s2);
    const e1 = Math.exp(s1 - m);
    const e2 = Math.exp(s2 - m);
    const z = e1 + e2;
    return [e1 / z, e2 / z] as const;
  });

  const hardWeights = $derived(s1 >= s2 ? [1, 0] : [0, 1] as const);

  const weights = $derived(mode === 'soft' ? softWeights : hardWeights);

  const ox = $derived(weights[0] * k1x + weights[1] * k2x);
  const oy = $derived(weights[0] * k1y + weights[1] * k2y);

  const fmt = (n: number) =>
    Math.abs(n) < 0.005 ? '0.00' : (n > 0 ? '+' : '') + n.toFixed(2);
  const pct = (n: number) => `${(n * 100).toFixed(0)}%`;
</script>

<div class="widget">
  <div class="modebar">
    <span class="modelabel">lookup</span>
    <div class="seg" role="tablist" aria-label="Lookup mode">
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'hard'}
        class="seg-btn"
        class:active={mode === 'hard'}
        onclick={() => (mode = 'hard')}
      >
        hard (argmax)
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'soft'}
        class="seg-btn"
        class:active={mode === 'soft'}
        onclick={() => (mode = 'soft')}
      >
        soft (softmax)
      </button>
    </div>
  </div>

  <div class="stage">
    <Mafs width={560} height={360} viewBox={{ x: [-3, 3], y: [-2, 2.5] }}>
      <Coordinates.Cartesian />
      <Vector tail={[0, 0]} tip={[k1x, k1y]} color="var(--ink-sea)" weight={2.5} />
      <Vector tail={[0, 0]} tip={[k2x, k2y]} color="var(--ink-coral)" weight={2.5} />
      <Vector tail={[0, 0]} tip={[qx, qy]} color="var(--ink-red)" weight={2.75} />
      <Vector tail={[0, 0]} tip={[ox, oy]} color="var(--ink-sun)" weight={3.25} />
      <MovablePoint bind:x={k1x} bind:y={k1y} color="var(--ink-sea)" label="key 1 tip" />
      <MovablePoint bind:x={k2x} bind:y={k2y} color="var(--ink-coral)" label="key 2 tip" />
      <MovablePoint bind:x={qx} bind:y={qy} color="var(--ink-red)" label="query tip" />
    </Mafs>
  </div>

  <div class="readout" aria-live="polite">
    <dl class="scores">
      <div class="row sea">
        <dt><em>q</em>·<em>k</em><sub>1</sub></dt>
        <dd>{fmt(s1)}</dd>
      </div>
      <div class="row coral">
        <dt><em>q</em>·<em>k</em><sub>2</sub></dt>
        <dd>{fmt(s2)}</dd>
      </div>
    </dl>

    <div class="weights">
      <div class="weight-row">
        <span class="wlabel sea">α<sub>1</sub></span>
        <div class="bar">
          <div class="bar-fill sea-fill" style="width:{weights[0] * 100}%"></div>
        </div>
        <span class="wval">{pct(weights[0])}</span>
      </div>
      <div class="weight-row">
        <span class="wlabel coral">α<sub>2</sub></span>
        <div class="bar">
          <div class="bar-fill coral-fill" style="width:{weights[1] * 100}%"></div>
        </div>
        <span class="wval">{pct(weights[1])}</span>
      </div>
    </div>

    <div class="output-row">
      <span class="olabel"><em>o</em> = α<sub>1</sub><em>k</em><sub>1</sub> + α<sub>2</sub><em>k</em><sub>2</sub></span>
      <span class="oval">({fmt(ox)}, {fmt(oy)})</span>
    </div>

    <p class="hint">drag <span class="dot red"></span>q, <span class="dot sea"></span>k<sub>1</sub>, or <span class="dot coral"></span>k<sub>2</sub> — the <span class="dot sun"></span>output is α-weighted. for now, values are the keys themselves.</p>
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 55%, transparent);
  }

  .modebar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
  }
  .modelabel {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .seg {
    display: inline-flex;
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    border-radius: 10px;
    overflow: hidden;
  }
  .seg-btn {
    appearance: none;
    border: 0;
    padding: 0.35rem 0.7rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    background: transparent;
    color: var(--site-fg-muted);
    cursor: pointer;
    transition: background 160ms ease, color 160ms ease;
  }
  .seg-btn + .seg-btn {
    border-left: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
  }
  .seg-btn.active {
    background: color-mix(in srgb, var(--ink-sea) 18%, transparent);
    color: var(--site-fg);
  }
  .seg-btn:hover:not(.active) {
    background: color-mix(in srgb, var(--site-fg) 5%, transparent);
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
    flex-direction: column;
    gap: 0.7rem;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    color: var(--site-fg);
  }

  .scores {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.5rem 1rem;
    margin: 0;
    font-family: var(--font-mono);
    font-size: 0.95rem;
  }
  .scores .row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.4rem 0.65rem;
    border-radius: 10px;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-left: 3px solid var(--site-fg-muted);
  }
  .scores .row.sea { border-left-color: var(--ink-sea); }
  .scores .row.coral { border-left-color: var(--ink-coral); }
  .scores dt {
    color: var(--site-fg-muted);
    font-family: var(--font-mono);
  }
  .scores dt em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }
  .scores dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }

  .weights {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .weight-row {
    display: grid;
    grid-template-columns: 2.5rem 1fr 3rem;
    align-items: center;
    gap: 0.6rem;
    font-family: var(--font-mono);
    font-size: 0.9rem;
  }
  .wlabel {
    font-weight: 600;
  }
  .wlabel.sea { color: var(--ink-sea); }
  .wlabel.coral { color: var(--ink-coral); }
  .bar {
    height: 14px;
    background: color-mix(in srgb, var(--site-fg) 8%, transparent);
    border-radius: 7px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 7px;
    transition: width 220ms ease;
  }
  .sea-fill { background: var(--ink-sea); }
  .coral-fill { background: var(--ink-coral); }
  .wval {
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: var(--site-fg-muted);
  }

  .output-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.5rem 0.75rem;
    border-radius: 10px;
    background: color-mix(in srgb, var(--ink-sun) 14%, transparent);
    border-left: 3px solid var(--ink-sun);
    font-family: var(--font-mono);
    font-size: 0.95rem;
  }
  .olabel em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }
  .oval {
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }

  .hint {
    margin: 0;
    font-family: var(--font-body);
    font-size: 0.8rem;
    color: var(--site-fg-muted);
    line-height: 1.5;
  }
  .dot {
    display: inline-block;
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 50%;
    vertical-align: middle;
    margin-right: 0.15rem;
  }
  .dot.red { background: var(--ink-red); }
  .dot.sea { background: var(--ink-sea); }
  .dot.coral { background: var(--ink-coral); }
  .dot.sun { background: var(--ink-sun); }
</style>
