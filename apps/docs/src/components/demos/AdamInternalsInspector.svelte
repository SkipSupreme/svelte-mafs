<script lang="ts">
  /**
   * AdamInternalsInspector — scrub a step counter and watch the four Adam
   * statistics update. Synthetic gradient sequence g_t = 1 (constant), so:
   *   m_t  = 1 - β₁^t      (biased, slow warmup)
   *   v_t  = 1 - β₂^t      (more biased — β₂ is bigger)
   *   m̂_t  = m_t / (1 - β₁^t) = 1   (the bias correction's whole job)
   *   v̂_t  = 1
   * The teaching moment: corrected estimates are unit-scale from step 1;
   * uncorrected ones crawl up for thousands of steps when β₂ = 0.999.
   */

  let t = $state(1);
  let beta1 = $state(0.9);
  let beta2 = $state(0.999);

  // Adam stats with constant gradient g = 1 and zero init.
  const mRaw = $derived(1 - Math.pow(beta1, t));
  const vRaw = $derived(1 - Math.pow(beta2, t));
  const mHat = $derived(1); // m_t / (1 - β₁^t) = 1 exactly with constant g
  const vHat = $derived(1); // same
  const corr1 = $derived(1 / (1 - Math.pow(beta1, t)));
  const corr2 = $derived(1 / (1 - Math.pow(beta2, t)));

  // Normalize bar heights to [0, 1] for SVG.
  const W = 480;
  const H = 220;
  const PAD = 36;
  const barW = (W - PAD * 2) / 4 - 18;

  type Bar = { label: string; value: number; color: string };
  const bars = $derived<Bar[]>([
    { label: 'm_t (raw)', value: mRaw, color: 'var(--ink-red)' },
    { label: 'v_t (raw)', value: vRaw, color: 'var(--ink-sea)' },
    { label: 'm̂_t (corrected)', value: mHat, color: 'var(--cta-hover)' },
    { label: 'v̂_t (corrected)', value: vHat, color: 'var(--cta-hover)' },
  ]);
</script>

<div class="widget">
  <header class="header">
    <p class="title">scrub step <em>t</em> · watch raw vs bias-corrected</p>
    <p class="hint">grad = 1 each step · β₁ = {beta1.toFixed(2)} · β₂ = {beta2.toFixed(3)}</p>
  </header>

  <div class="stage">
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label="Adam statistics bars">
      <!-- Baseline -->
      <line x1={PAD} y1={H - 30} x2={W - PAD} y2={H - 30} stroke="color-mix(in srgb, var(--site-fg) 22%, transparent)" stroke-width="1" />
      <!-- Target line at value=1 -->
      <line x1={PAD} y1={30} x2={W - PAD} y2={30} stroke="color-mix(in srgb, var(--site-fg) 14%, transparent)" stroke-width="1" stroke-dasharray="3 3" />
      <text x={W - PAD + 4} y={34} class="axis-text">target = 1</text>

      {#each bars as bar, i}
        {@const x0 = PAD + i * ((W - PAD * 2) / 4) + 9}
        {@const h = Math.max(0, Math.min(1, bar.value)) * (H - 60)}
        {@const y0 = (H - 30) - h}
        <rect x={x0} y={y0} width={barW} height={h} fill={bar.color} rx="3" />
        <text x={x0 + barW / 2} y={H - 12} class="bar-label">{bar.label}</text>
        <text x={x0 + barW / 2} y={y0 - 6} class="bar-value">{bar.value.toFixed(2)}</text>
      {/each}
    </svg>
  </div>

  <div class="controls">
    <label>
      <span><em>t</em> = {t}</span>
      <input type="range" min="1" max="50" step="1" bind:value={t} aria-label="Step counter" />
    </label>
    <label>
      <span>β₁ = {beta1.toFixed(2)}</span>
      <input type="range" min="0.5" max="0.999" step="0.001" bind:value={beta1} aria-label="Beta 1" />
    </label>
    <label>
      <span>β₂ = {beta2.toFixed(3)}</span>
      <input type="range" min="0.9" max="0.9999" step="0.0001" bind:value={beta2} aria-label="Beta 2" />
    </label>
  </div>

  <p class="readout">
    <span>1 / (1 − β₁<sup>t</sup>) = <strong>{corr1.toFixed(2)}×</strong></span>
    <span>1 / (1 − β₂<sup>t</sup>) = <strong>{corr2.toFixed(2)}×</strong></span>
    <span class="cap">the bigger the correction factor, the more biased the raw estimate is.</span>
  </p>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: .85rem;
    background: var(--demo-card); border: 1px solid var(--demo-card-border);
    border-radius: 20px; padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 50%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: .5rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .title em { font-style: italic; }
  .hint { margin: 0; font-family: var(--font-mono); font-size: .8rem; color: var(--site-fg-muted); }

  .stage { background: var(--demo-stage); border-radius: 12px; padding: .5rem; }
  .stage svg { display: block; width: 100%; height: auto; max-height: 260px; }

  .axis-text { font-family: var(--font-mono); font-size: 9px; fill: var(--site-fg-muted); }
  .bar-label { font-family: var(--font-mono); font-size: 10px; fill: var(--site-fg-muted); text-anchor: middle; }
  .bar-value { font-family: var(--font-mono); font-size: 11px; font-weight: 700; fill: var(--site-fg); text-anchor: middle; }

  .controls { display: grid; gap: .45rem; padding-top: .5rem; border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent); }
  .controls label { display: grid; grid-template-columns: 6.5rem 1fr; gap: .8rem; align-items: center; font-family: var(--font-mono); font-size: .85rem; }
  .controls label em { font-style: italic; font-family: var(--font-display); }
  input[type="range"] { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; border-radius: 999px; background: color-mix(in srgb, var(--site-fg) 18%, transparent); outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 22px; height: 22px; border-radius: 999px; background: var(--ink-sea); border: 2px solid var(--demo-card); cursor: grab; }
  input[type="range"]::-moz-range-thumb { width: 22px; height: 22px; border-radius: 999px; background: var(--ink-sea); border: 2px solid var(--demo-card); cursor: grab; }

  .readout { margin: 0; display: flex; gap: 1rem; flex-wrap: wrap; padding: .55rem .85rem; background: color-mix(in srgb, var(--site-fg) 4%, transparent); border-radius: 10px; font-family: var(--font-mono); font-size: .85rem; }
  .readout strong { color: var(--ink-coral); }
  .readout .cap { color: var(--site-fg-muted); font-family: var(--font-body); font-size: .78rem; flex-basis: 100%; }
</style>
