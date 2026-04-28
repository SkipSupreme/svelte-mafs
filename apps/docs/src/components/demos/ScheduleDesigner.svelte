<script lang="ts">
  import { Mafs, Coordinates, Plot, Line, Point } from 'svelte-mafs';

  /**
   * ScheduleDesigner — visualize a learning-rate schedule. Pick a kind
   * (cosine / linear / step), drag the warmup endpoint and the decay
   * endpoint, watch the η curve update.
   */

  type Kind = 'cosine' | 'linear' | 'step';
  let kind = $state<Kind>('cosine');

  // Warmup endpoint (T_w, η_max), decay endpoint (T, η_min).
  let Tw = $state(20);
  let etaMax = $state(0.8);
  let T = $state(100);
  let etaMin = $state(0.05);

  function clampT(t: number) { return Math.max(Tw + 1, Math.min(200, t)); }
  function clampTw(t: number) { return Math.max(0, Math.min(T - 1, t)); }
  function clampEta(e: number) { return Math.max(0, Math.min(1, e)); }

  // η(t) — schedule function.
  function eta(t: number): number {
    if (t <= Tw) {
      // linear warmup from 0 to etaMax
      return Tw <= 0 ? etaMax : etaMax * (t / Tw);
    }
    const u = (t - Tw) / (T - Tw); // [0, 1] over decay window
    if (u >= 1) return etaMin;
    if (kind === 'cosine') {
      return etaMin + 0.5 * (etaMax - etaMin) * (1 + Math.cos(Math.PI * u));
    }
    if (kind === 'linear') {
      return etaMax + (etaMin - etaMax) * u;
    }
    // step: drops to etaMin at t = Tw + (T - Tw) * 0.6, classic 10× cuts implied
    const dropAt = Tw + (T - Tw) * 0.6;
    return t < dropAt ? etaMax : etaMin;
  }
</script>

<div class="widget">
  <header class="header">
    <p class="title">learning-rate schedule</p>
    <div class="kinds">
      {#each ['cosine', 'linear', 'step'] as k}
        <button type="button" class:active={kind === k} onclick={() => (kind = k as Kind)}>{k}</button>
      {/each}
    </div>
  </header>

  <div class="stage">
    <Mafs width={560} height={300} viewBox={{ x: [-5, 210], y: [-0.05, 1.05] }}>
      <Coordinates.Cartesian
        xAxis={{ labels: false }}
        yAxis={{ labels: false }}
      />
      <Plot.OfX y={eta} color="var(--ink-red)" weight={2.5} />
      <Line.Segment point1={[Tw, 0]} point2={[Tw, etaMax]} color="var(--ink-sea)" weight={1.25} opacity={0.4} />
      <Line.Segment point1={[T, 0]} point2={[T, etaMin]} color="var(--ink-sea)" weight={1.25} opacity={0.4} />
      <Point x={Tw} y={etaMax} color="var(--ink-coral)" />
      <Point x={T} y={etaMin} color="var(--ink-coral)" />
    </Mafs>
  </div>

  <div class="controls">
    <label>
      <span>warmup end <em>T</em><sub>w</sub> = {Tw}</span>
      <input type="range" min="0" max="60" step="1" bind:value={Tw} oninput={() => { Tw = clampTw(Tw); }} aria-label="Warmup endpoint" />
    </label>
    <label>
      <span>peak η<sub>max</sub> = {etaMax.toFixed(2)}</span>
      <input type="range" min="0.1" max="1" step="0.01" bind:value={etaMax} oninput={() => { etaMax = clampEta(etaMax); }} aria-label="Peak learning rate" />
    </label>
    <label>
      <span>horizon <em>T</em> = {T}</span>
      <input type="range" min="40" max="200" step="1" bind:value={T} oninput={() => { T = clampT(T); }} aria-label="Decay horizon" />
    </label>
    <label>
      <span>final η<sub>min</sub> = {etaMin.toFixed(2)}</span>
      <input type="range" min="0" max="0.4" step="0.01" bind:value={etaMin} aria-label="Minimum learning rate" />
    </label>
  </div>

  <p class="note">
    <strong>cosine</strong> is the modern default — most of the budget at high η for exploration, smooth tail to fine-tune.
    Step decay is older, hand-tuned. Linear is a sanity baseline.
  </p>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: .85rem;
    background: var(--demo-card); border: 1px solid var(--demo-card-border);
    border-radius: 20px; padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-red) 40%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .5rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .kinds { display: flex; gap: .35rem; }
  .kinds button { padding: .35rem .75rem; font-family: var(--font-mono); font-size: .8rem; border: 1px solid var(--site-border); border-radius: 999px; background: var(--demo-card); color: var(--site-fg); cursor: pointer; text-transform: lowercase; }
  .kinds button:hover { border-color: var(--ink-red); color: var(--ink-red); }
  .kinds button.active { background: var(--ink-red); color: var(--cta-fg); border-color: var(--ink-red); }

  .stage { width: 100%; background: var(--demo-stage); border-radius: 12px; padding: .35rem; touch-action: none; }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

  .controls { display: grid; gap: .45rem; padding-top: .5rem; border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent); }
  .controls label { display: grid; grid-template-columns: 11rem 1fr; gap: .8rem; align-items: center; font-family: var(--font-mono); font-size: .85rem; }
  .controls em { font-style: italic; font-family: var(--font-display); }
  input[type="range"] { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; border-radius: 999px; background: color-mix(in srgb, var(--site-fg) 18%, transparent); outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 22px; height: 22px; border-radius: 999px; background: var(--ink-red); border: 2px solid var(--demo-card); cursor: grab; }
  input[type="range"]::-moz-range-thumb { width: 22px; height: 22px; border-radius: 999px; background: var(--ink-red); border: 2px solid var(--demo-card); cursor: grab; }

  .note { margin: 0; padding: .55rem .85rem; background: color-mix(in srgb, var(--site-fg) 4%, transparent); border-radius: 10px; font-family: var(--font-body); font-size: .82rem; color: var(--site-fg); }
  .note strong { color: var(--ink-red); }
</style>
