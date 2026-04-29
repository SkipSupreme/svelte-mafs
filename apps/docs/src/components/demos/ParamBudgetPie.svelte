<script lang="ts">
  // Discrete d_model values that span small toy models → frontier.
  const D_OPTIONS = [64, 128, 256, 512, 768, 1024, 2048, 4096, 8192, 12288] as const;
  const N_MIN = 1;
  const N_MAX = 96;
  const V_OPTIONS = [256, 1024, 4096, 16384, 32768, 50257] as const;
  const FFN_RATIO_OPTIONS = [2, 4, 8] as const;
  const T_MAX = 2048;

  let dIndex = $state(4); // d=768
  let n = $state(12);
  let vIndex = $state(5); // V=50257
  let ffnRatioIndex = $state(1); // 4×
  let tied = $state(true);

  const d = $derived(D_OPTIONS[dIndex]);
  const v = $derived(V_OPTIONS[vIndex]);
  const ffnRatio = $derived(FFN_RATIO_OPTIONS[ffnRatioIndex]);

  // Components (no biases; small terms included for accounting completeness).
  const embedding = $derived(v * d);
  const positional = $derived(T_MAX * d);
  const lmHead = $derived(tied ? 0 : v * d);
  const attentionPerBlock = $derived(4 * d * d);
  const ffnPerBlock = $derived(2 * ffnRatio * d * d);
  const lnPerBlock = $derived(4 * d); // 2 LayerNorms × 2 params per dim
  const finalLn = $derived(2 * d);

  const attentionTotal = $derived(n * attentionPerBlock);
  const ffnTotal = $derived(n * ffnPerBlock);
  const lnTotal = $derived(n * lnPerBlock + finalLn);

  const total = $derived(
    embedding + positional + lmHead + attentionTotal + ffnTotal + lnTotal
  );

  const segments = $derived([
    { key: 'embed', label: 'token embed' + (tied ? ' (tied)' : ''), value: embedding, color: 'var(--ink-red)' },
    { key: 'pos', label: 'positional', value: positional, color: 'var(--ink-orange)' },
    ...(tied
      ? []
      : [{ key: 'lmhead', label: 'LM head', value: lmHead, color: 'var(--ink-pink)' }]),
    { key: 'attn', label: `attention × ${n}`, value: attentionTotal, color: 'var(--ink-sea)' },
    { key: 'ffn', label: `FFN × ${n}`, value: ffnTotal, color: 'var(--ink-coral)' },
    { key: 'ln', label: 'layer norms', value: lnTotal, color: 'var(--ink-teal)' },
  ]);

  function fmtParams(p: number): string {
    if (p >= 1e9) return `${(p / 1e9).toFixed(2)} B`;
    if (p >= 1e6) return `${(p / 1e6).toFixed(1)} M`;
    if (p >= 1e3) return `${(p / 1e3).toFixed(1)} K`;
    return `${p}`;
  }

  function pct(p: number): string {
    return `${((p / total) * 100).toFixed(1)}%`;
  }

  const presets = [
    { name: 'GPT-2 small', dIdx: 4, n: 12, vIdx: 5, ratioIdx: 1, tied: true },
    { name: 'GPT-2 medium', dIdx: 5, n: 24, vIdx: 5, ratioIdx: 1, tied: true },
    { name: 'GPT-2 XL', dIdx: 6, n: 48, vIdx: 5, ratioIdx: 1, tied: true },
    { name: 'GPT-3 175B', dIdx: 9, n: 96, vIdx: 5, ratioIdx: 1, tied: false },
  ];
  function loadPreset(p: (typeof presets)[number]) {
    dIndex = p.dIdx;
    n = p.n;
    vIndex = p.vIdx;
    ffnRatioIndex = p.ratioIdx;
    tied = p.tied;
  }
</script>

<div class="widget">
  <div class="presets">
    <span class="preset-label">presets</span>
    {#each presets as p}
      <button type="button" class="preset" onclick={() => loadPreset(p)}>{p.name}</button>
    {/each}
  </div>

  <div class="controls">
    <label class="ctrl">
      <span class="ctrl-label">d<sub>model</sub></span>
      <input type="range" min="0" max={D_OPTIONS.length - 1} step="1" bind:value={dIndex} />
      <output class="ctrl-val">{d.toLocaleString()}</output>
    </label>
    <label class="ctrl">
      <span class="ctrl-label">N (blocks)</span>
      <input type="range" min={N_MIN} max={N_MAX} step="1" bind:value={n} />
      <output class="ctrl-val">{n}</output>
    </label>
    <label class="ctrl">
      <span class="ctrl-label">|V|</span>
      <input type="range" min="0" max={V_OPTIONS.length - 1} step="1" bind:value={vIndex} />
      <output class="ctrl-val">{v.toLocaleString()}</output>
    </label>
    <label class="ctrl">
      <span class="ctrl-label">FFN ratio</span>
      <input type="range" min="0" max={FFN_RATIO_OPTIONS.length - 1} step="1" bind:value={ffnRatioIndex} />
      <output class="ctrl-val">{ffnRatio}×</output>
    </label>
    <label class="ctrl tying">
      <input type="checkbox" bind:checked={tied} />
      <span>weight tying</span>
    </label>
  </div>

  <div class="total">
    <span class="total-label">total parameters</span>
    <span class="total-value">{fmtParams(total)}</span>
  </div>

  <div class="bar" role="img" aria-label="parameter breakdown">
    {#each segments as seg}
      {#if seg.value > 0}
        <div
          class="seg"
          style="flex: {seg.value}; background: {seg.color}"
          title="{seg.label}: {fmtParams(seg.value)} ({pct(seg.value)})"
        ></div>
      {/if}
    {/each}
  </div>

  <ul class="legend">
    {#each segments as seg}
      {#if seg.value > 0}
        <li>
          <span class="dot" style="background: {seg.color}"></span>
          <span class="legend-label">{seg.label}</span>
          <span class="legend-val">{fmtParams(seg.value)}</span>
          <span class="legend-pct">{pct(seg.value)}</span>
        </li>
      {/if}
    {/each}
  </ul>

  <div class="footnote">
    per-block: <strong>{fmtParams(attentionPerBlock + ffnPerBlock)}</strong>
    ({(((attentionPerBlock) / (attentionPerBlock + ffnPerBlock)) * 100).toFixed(0)}% attention,
    {((ffnPerBlock / (attentionPerBlock + ffnPerBlock)) * 100).toFixed(0)}% FFN)
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

  .presets {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
  }
  .preset-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-right: 0.25rem;
  }
  .preset {
    appearance: none;
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    background: transparent;
    color: var(--site-fg);
    padding: 0.3rem 0.6rem;
    border-radius: 8px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    cursor: pointer;
    transition: background 160ms ease, border-color 160ms ease;
  }
  .preset:hover {
    background: color-mix(in srgb, var(--ink-sea) 12%, transparent);
    border-color: var(--ink-sea);
  }

  .controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.6rem;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
  .ctrl {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
  }
  .ctrl-label {
    color: var(--site-fg-muted);
    font-weight: 600;
    min-width: 4.5rem;
  }
  .ctrl-label sub {
    font-size: 0.75em;
  }
  .ctrl input[type='range'] {
    flex: 1;
    accent-color: var(--ink-sea);
    min-width: 0;
  }
  .ctrl-val {
    min-width: 4rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    color: var(--site-fg);
  }
  .ctrl.tying {
    grid-column: 1 / -1;
    justify-content: flex-start;
  }
  .ctrl.tying input {
    accent-color: var(--ink-sea);
  }

  .total {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 0.7rem 1rem;
    border-radius: 12px;
    background: color-mix(in srgb, var(--ink-sun) 14%, transparent);
    border-left: 4px solid var(--ink-sun);
  }
  .total-label {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
  }
  .total-value {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--site-fg);
  }

  .bar {
    display: flex;
    height: 32px;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid color-mix(in srgb, var(--site-fg) 8%, transparent);
  }
  .seg {
    transition: flex 220ms ease;
    min-width: 1px;
  }

  .legend {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.4rem 1rem;
  }
  .legend li {
    display: grid;
    grid-template-columns: 0.7rem 1fr auto auto;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.82rem;
  }
  .dot {
    width: 0.7rem;
    height: 0.7rem;
    border-radius: 3px;
  }
  .legend-label {
    color: var(--site-fg);
  }
  .legend-val {
    color: var(--site-fg-muted);
    font-variant-numeric: tabular-nums;
  }
  .legend-pct {
    color: var(--site-fg-muted);
    font-variant-numeric: tabular-nums;
    min-width: 3rem;
    text-align: right;
  }

  .footnote {
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: var(--site-fg-muted);
  }
  .footnote strong {
    color: var(--site-fg);
    font-weight: 700;
  }
</style>
