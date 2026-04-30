<script lang="ts">
  // NucleusCutoff — top-p (nucleus) sampling, where the math actually lives.
  //
  // Shows a sorted-descending categorical distribution as bars with their
  // cumulative-mass curve drawn behind. A draggable horizontal "p" line cuts
  // the curve. Bars whose cumulative mass is at or below the line stay; the
  // rest get masked. The kept bars renormalize to a new distribution.
  //
  // Two preset distributions echo Worked Example E6 (the brief's worked
  // example, p=0.9, nucleus size = 4) and Worked Example E9 (the long-tailed
  // step where top-k=10 and top-p=0.9 give very different cutoffs).

  type Preset = 'sharp' | 'flat' | 'custom';

  interface Props {
    initialPreset?: Preset;
    initialP?: number;
  }

  let { initialPreset = 'sharp', initialP = 0.9 }: Props = $props();

  // Preset distributions, each pre-sorted descending and summing to 1.
  const PRESETS: Record<Exclude<Preset, 'custom'>, { name: string; probs: number[] }> = {
    sharp: {
      name: 'confident step (E6)',
      probs: [0.50, 0.20, 0.15, 0.08, 0.05, 0.02],
    },
    flat: {
      name: 'long-tailed step (E9)',
      // 50 nearly-equal tokens summing to ~0.9, plus a tiny tail.
      probs: (() => {
        const top = Array(50).fill(0.018);
        const tail = Array(15).fill(0.0067);
        const all = [...top, ...tail];
        const s = all.reduce((a, b) => a + b, 0);
        return all.map((x) => x / s);
      })(),
    },
  };

  let preset: Preset = $state(initialPreset);
  let probs: number[] = $state([...PRESETS[initialPreset === 'custom' ? 'sharp' : initialPreset].probs]);
  let p: number = $state(initialP);
  let topK: number = $state(0); // 0 = off

  // Switch preset.
  function setPreset(name: Exclude<Preset, 'custom'>) {
    preset = name;
    probs = [...PRESETS[name].probs];
  }

  const N = $derived(probs.length);
  const cum: number[] = $derived.by(() => {
    const c: number[] = [];
    let s = 0;
    for (const x of probs) {
      s += x;
      c.push(s);
    }
    return c;
  });

  // Nucleus: smallest prefix with cum >= p.
  const nucleusSize: number = $derived.by(() => {
    for (let i = 0; i < N; i++) if (cum[i] >= p) return i + 1;
    return N;
  });

  const keptByK: boolean[] = $derived.by(() => {
    if (!topK) return Array(N).fill(true);
    return Array(N).fill(false).map((_, i) => i < topK);
  });

  // Combined keep mask (top-p AND optional top-k).
  const kept: boolean[] = $derived.by(() => {
    return Array(N).fill(false).map((_, i) => i < nucleusSize && keptByK[i]);
  });

  const keptMass: number = $derived(probs.reduce((a, b, i) => a + (kept[i] ? b : 0), 0));
  const renorm: number[] = $derived(
    keptMass > 0 ? probs.map((q, i) => (kept[i] ? q / keptMass : 0)) : probs.map(() => 0),
  );

  const keptCount: number = $derived(kept.filter(Boolean).length);

  // Geometry.
  const W = 460;
  const H = 220;
  const padL = 24;
  const padR = 14;
  const padT = 14;
  const padB = 30;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const barGap = 1;
  const barW = $derived((plotW - barGap * (N - 1)) / N);

  function barX(i: number): number {
    return padL + i * (barW + barGap);
  }
  // Bar height scales linearly to plotH so the y-axis means "probability".
  const yMax = $derived(Math.max(0.55, ...probs) * 1.06);
  function yFromProb(q: number): number {
    return padT + plotH - (q / yMax) * plotH;
  }
  function pToY(prob: number): number {
    return padT + plotH - (prob / yMax) * plotH;
  }
  // Cumulative path coordinates — drawn against the same y-axis (treating
  // cumulative mass like a probability that goes from 0 to 1).
  function cumPathD(): string {
    let d = '';
    for (let i = 0; i < N; i++) {
      const x = barX(i) + barW / 2;
      const y = padT + plotH - cum[i] * plotH;
      d += i === 0 ? `M ${x.toFixed(2)} ${y.toFixed(2)}` : ` L ${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    return d;
  }
  // The "p" line across the plot, in units of cumulative mass.
  const pLineY = $derived(padT + plotH - p * plotH);

  // Drag the p line.
  let draggingP: boolean = $state(false);
  function startDragP(e: PointerEvent) {
    draggingP = true;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    moveP(e);
  }
  function moveP(e: PointerEvent) {
    if (!draggingP) return;
    const svg = (e.currentTarget as Element).ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const yClient = e.clientY - rect.top;
    const yLocal = (yClient / rect.height) * H;
    const yInPlot = Math.max(0, Math.min(plotH, yLocal - padT));
    p = Math.max(0.01, Math.min(1, 1 - yInPlot / plotH));
  }
  function endDragP(e: PointerEvent) {
    if (draggingP) {
      try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch {}
    }
    draggingP = false;
  }

  const fmt3 = (x: number) => x.toFixed(3);
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">p</span>
      <span class="meta-val">{p.toFixed(2)}</span>
    </div>
    <div class="meta">
      <span class="meta-key">nucleus size</span>
      <span class="meta-val">{nucleusSize}</span>
    </div>
    {#if topK > 0}
      <div class="meta">
        <span class="meta-key">top-k</span>
        <span class="meta-val">{topK}</span>
      </div>
    {/if}
    <div class="meta">
      <span class="meta-key">tokens kept</span>
      <span class="meta-val">{keptCount} / {N}</span>
    </div>
    <div class="meta">
      <span class="meta-key">kept mass</span>
      <span class="meta-val">{fmt3(keptMass)}</span>
    </div>
  </header>

  <div class="preset-row">
    <button type="button" class="mode" class:active={preset === 'sharp'} onclick={() => setPreset('sharp')}>
      confident
    </button>
    <button type="button" class="mode" class:active={preset === 'flat'} onclick={() => setPreset('flat')}>
      long-tailed
    </button>
  </div>

  <div class="plot-wrap">
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img"
      aria-label="Sorted distribution with draggable top-p cutoff line.">
      <rect x="0" y="0" width={W} height={H} fill="var(--demo-stage)" rx="8" />

      <!-- horizontal gridlines for cum mass at 0.25, 0.5, 0.75 -->
      {#each [0.25, 0.5, 0.75, 1.0] as g}
        <line x1={padL} x2={W - padR}
          y1={padT + plotH - g * plotH} y2={padT + plotH - g * plotH}
          stroke="color-mix(in srgb, var(--site-fg) 8%, transparent)" stroke-width="0.5" />
        <text x={padL - 4} y={padT + plotH - g * plotH + 3} text-anchor="end"
          font-size="8" font-family="var(--font-mono)" fill="var(--site-fg-muted)">{g.toFixed(2)}</text>
      {/each}

      <!-- bars -->
      {#each probs as q, i}
        <g>
          <rect x={barX(i)} y={yFromProb(q)} width={barW} height={padT + plotH - yFromProb(q)}
            fill={kept[i] ? 'var(--ink-red)' : 'color-mix(in srgb, var(--site-fg) 14%, transparent)'}
            rx="1.5" />
          {#if kept[i]}
            <rect x={barX(i)} y={yFromProb(renorm[i])} width={barW} height="2"
              fill="var(--ink-coral)" opacity="0.9" />
          {/if}
        </g>
      {/each}

      <!-- cumulative mass curve (uses cumulative scale 0..1 across plotH) -->
      <path d={cumPathD()} stroke="var(--ink-sea)" stroke-width="1.5" fill="none"
        stroke-dasharray="3 2" />

      <!-- p line (draggable) -->
      <line x1={padL} x2={W - padR} y1={pLineY} y2={pLineY}
        stroke="var(--ink-sun)" stroke-width="2" />
      <text x={W - padR - 2} y={pLineY - 4} text-anchor="end"
        font-size="9" font-family="var(--font-mono)" fill="var(--ink-sun)" font-weight="700">
        p = {p.toFixed(2)}
      </text>

      <!-- drag handle: a wide invisible strip across the plot that captures pointer -->
      <rect x={padL} y={padT} width={plotW} height={plotH}
        fill="transparent"
        style="cursor: ns-resize; touch-action: none;"
        onpointerdown={startDragP}
        onpointermove={moveP}
        onpointerup={endDragP}
        onpointercancel={endDragP}
        aria-label="Drag vertically to set p" />

      <!-- legend -->
      <g transform={`translate(${padL + 4}, ${padT + 8})`} font-family="var(--font-mono)" font-size="9">
        <rect x="0" y="0" width="10" height="6" fill="var(--ink-red)" rx="1" />
        <text x="14" y="6" fill="var(--site-fg)">kept</text>
        <line x1="56" y1="3" x2="76" y2="3" stroke="var(--ink-sea)" stroke-width="1.5" stroke-dasharray="3 2" />
        <text x="80" y="6" fill="var(--site-fg)">cum mass</text>
        <rect x="142" y="2" width="10" height="2" fill="var(--ink-coral)" />
        <text x="156" y="6" fill="var(--site-fg)">renormalized</text>
      </g>
    </svg>
  </div>

  <div class="controls">
    <label class="slider">
      <span class="slider-label">top-p threshold p = <strong>{p.toFixed(2)}</strong></span>
      <input type="range" min="0.05" max="1" step="0.01" bind:value={p} aria-label="Top-p threshold" />
    </label>
    <label class="slider">
      <span class="slider-label">top-k = <strong>{topK === 0 ? 'off' : topK}</strong></span>
      <input type="range" min="0" max={N} step="1" bind:value={topK} aria-label="Top-k cutoff (0 = off)" />
    </label>
  </div>

  <p class="caption">
    Bars are the sorted descending probabilities. The dashed teal line is
    cumulative mass — it climbs from the first bar's probability to 1 by the
    last. Drag the gold p-line vertically: every bar whose <em>cumulative</em>
    mass is at or below it stays; the rest are masked. Coral ticks show the
    <em>renormalized</em> distribution over the kept (nucleus) tokens.
    The nucleus shrinks when the model is confident and grows when it isn't —
    that's what "top-p" means.
  </p>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(0.85rem, 2vw, 1.4rem);
    color: var(--site-fg);
    box-shadow:
      0 1px 0 rgba(0,0,0,0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-sun) 40%, transparent);
  }
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 700; }

  .preset-row { display: flex; gap: 0.4rem; }
  .mode {
    cursor: pointer;
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    border-radius: 999px; padding: 0.25rem 0.85rem;
    font-size: 0.83rem; font-weight: 600;
    background: transparent; color: var(--site-fg);
    font-family: inherit;
    transition: background 160ms ease, border-color 160ms ease;
  }
  .mode:hover { border-color: var(--site-fg); }
  .mode.active { background: var(--ink-red); color: #fdfdfc; border-color: var(--ink-red); }

  .plot-wrap {
    width: 100%; overflow: hidden; border-radius: 12px;
    user-select: none; -webkit-user-select: none;
  }
  .plot-wrap svg { width: 100%; height: auto; display: block; }

  .controls { display: flex; flex-direction: column; gap: 0.55rem; }
  .slider { display: flex; flex-direction: column; gap: 0.2rem; }
  .slider-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted); }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input { width: 100%; accent-color: var(--ink-sun); }

  .caption { margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55; }
</style>
