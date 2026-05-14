<script lang="ts">
  import { onDestroy } from 'svelte';
  import { play } from '../../lib/sound';

  interface Props {
    /** Initial probability vector. Length determines vocab size. Auto-normalized on load. */
    initialP?: number[];
    /** Token labels shown above each bar. */
    labels?: string[];
    /** Hide the "sample 200" auto button — used when the lesson wants a deliberate, one-drop tempo. */
    hideAuto?: boolean;
  }

  let {
    initialP = [0.1, 0.35, 0.25, 0.2, 0.1],
    labels = ['a', 'b', 'c', 'd', 'e'],
    hideAuto = false,
  }: Props = $props();

  const V = initialP.length;

  function normalize(arr: number[]): number[] {
    const s = arr.reduce((a, b) => a + b, 0);
    if (s <= 0) return arr.map(() => 1 / arr.length);
    return arr.map((x) => x / s);
  }

  let p: number[] = $state(normalize([...initialP]));
  let counts: number[] = $state(Array(V).fill(0));
  let nSamples: number = $state(0);
  let lastU: number | null = $state(null);
  let lastIdx: number | null = $state(null);
  let dropping = $state(false);
  let auto = $state(false);
  let rafId: number | null = null;

  // CDF: c_k = sum_{j <= k} p_j. The "staircase" we drop u onto.
  const cdf: number[] = $derived.by(() => {
    const out: number[] = [];
    let s = 0;
    for (const v of p) { s += v; out.push(s); }
    return out;
  });

  const empirical: number[] = $derived(
    nSamples > 0 ? counts.map((c) => c / nSamples) : Array(V).fill(0),
  );

  function sampleOnce(): number {
    const u = Math.random();
    let idx = V - 1;
    for (let i = 0; i < V; i++) {
      if (cdf[i] >= u) { idx = i; break; }
    }
    return (lastU = u, lastIdx = idx, idx);
  }

  function drop() {
    if (dropping) return;
    dropping = true;
    const idx = sampleOnce();
    counts = counts.map((c, i) => (i === idx ? c + 1 : c));
    nSamples += 1;
    play('tick');
    // Reset the "ball" state after the visual lands; the next drop replaces it.
    setTimeout(() => { dropping = false; }, 380);
  }

  function dropN(n: number) {
    const next = [...counts];
    for (let k = 0; k < n; k++) {
      const u = Math.random();
      let idx = V - 1;
      for (let i = 0; i < V; i++) {
        if (cdf[i] >= u) { idx = i; break; }
      }
      next[idx] += 1;
    }
    counts = next;
    nSamples += n;
  }

  function startAuto() {
    if (auto) return;
    auto = true;
    const tick = () => {
      if (!auto) return;
      dropN(8);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }
  function stopAuto() {
    auto = false;
    if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
  }
  function resetSamples() {
    counts = Array(V).fill(0);
    nSamples = 0;
    lastU = null;
    lastIdx = null;
  }
  onDestroy(() => stopAuto());

  // ---- SVG geometry ----
  // Left panel: pmf bars.    Center panel: stacked CDF + ball.   Right panel: empirical histogram.
  const W = 560;
  const H = 280;
  const padT = 18;
  const padB = 38;
  const innerH = H - padT - padB;

  // Panel layout (x positions).
  const colGap = 18;
  const pmfW = 200;
  const cdfW = 110;
  const histW = 200;
  const totalW = pmfW + cdfW + histW + colGap * 2 + 24;
  const offsetX = (W - totalW) / 2 + 12;
  const pmfX0 = offsetX;
  const cdfX0 = pmfX0 + pmfW + colGap;
  const histX0 = cdfX0 + cdfW + colGap;

  const barGap = 4;
  const pmfBarW = (pmfW - barGap * (V - 1)) / V;
  const histBarW = (histW - barGap * (V - 1)) / V;

  // Bars scale to a constant ceiling so the y-axis is stable when you drag.
  const Y_CEIL = 0.6;

  function pmfBarX(i: number): number { return pmfX0 + i * (pmfBarW + barGap); }
  function histBarX(i: number): number { return histX0 + i * (histBarW + barGap); }
  function barTop(prob: number): number {
    return padT + innerH - (Math.min(prob, Y_CEIL) / Y_CEIL) * innerH;
  }
  function barH(prob: number): number {
    return (Math.min(prob, Y_CEIL) / Y_CEIL) * innerH;
  }

  // CDF staircase: a single vertical bar at x = cdfX0..cdfX0+cdfW, stacked from bottom.
  // The top of bin k lives at height = (c_k / 1) * innerH from the bottom.
  function cdfBinTop(k: number): number {
    return padT + innerH - cdf[k] * innerH;
  }
  function cdfBinHeight(k: number): number {
    const prev = k === 0 ? 0 : cdf[k - 1];
    return (cdf[k] - prev) * innerH;
  }
  function uY(u: number): number {
    return padT + innerH - u * innerH;
  }

  // ---- Drag to edit a bar ----
  let dragging: number | null = null;

  function pointerDown(e: PointerEvent, i: number) {
    dragging = i;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    pointerDrag(e, i);
  }
  function pointerMove(e: PointerEvent, i: number) {
    if (dragging !== i) return;
    pointerDrag(e, i);
  }
  function pointerUp(e: PointerEvent, i: number) {
    if (dragging === i) {
      try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch { /* noop */ }
    }
    dragging = null;
  }
  function pointerDrag(e: PointerEvent, i: number) {
    const svg = (e.currentTarget as Element).ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const yClient = e.clientY - rect.top;
    const yLocal = (yClient / rect.height) * H;
    const yInPlot = Math.max(0, Math.min(innerH, yLocal - padT));
    const target = Math.max(0.005, Math.min(Y_CEIL, (1 - yInPlot / innerH) * Y_CEIL));
    // Hold the *ratio* of the other entries fixed; rescale them so the vector sums to 1.
    const others = p.reduce((acc, v, k) => acc + (k === i ? 0 : v), 0);
    const remaining = 1 - target;
    const next = p.map((v, k) => {
      if (k === i) return target;
      return others > 0 ? (v / others) * remaining : remaining / (V - 1);
    });
    p = normalize(next);
  }

  const fmt = (n: number) => n.toFixed(3);
</script>

<div class="widget">
  <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Categorical distribution and inverse-CDF sampler" class="stage">
    <!-- Panel labels -->
    <text x={pmfX0 + pmfW / 2} y={padT - 6} class="panel-label">p (drag the bars)</text>
    <text x={cdfX0 + cdfW / 2} y={padT - 6} class="panel-label">CDF + u</text>
    <text x={histX0 + histW / 2} y={padT - 6} class="panel-label">samples so far</text>

    <!-- ============ Left: pmf bars ============ -->
    {#each p as prob, i}
      {@const x = pmfBarX(i)}
      {@const y = barTop(prob)}
      {@const h = barH(prob)}
      {@const isLast = lastIdx === i}
      <g class="pmf-bar" class:active={isLast}>
        <rect
          x={x}
          y={y}
          width={pmfBarW}
          height={h}
          rx={3}
          ry={3}
          class="bar pmf"
        />
        <!-- Big invisible hit target for dragging the top of the bar. -->
        <rect
          x={x}
          y={padT}
          width={pmfBarW}
          height={innerH}
          fill="transparent"
          style="cursor: ns-resize; touch-action: none"
          onpointerdown={(e) => pointerDown(e, i)}
          onpointermove={(e) => pointerMove(e, i)}
          onpointerup={(e) => pointerUp(e, i)}
          onpointercancel={(e) => pointerUp(e, i)}
        />
        <text x={x + pmfBarW / 2} y={padT + innerH + 14} class="tick-label">{labels[i] ?? `x_${i}`}</text>
        <text x={x + pmfBarW / 2} y={padT + innerH + 30} class="tick-num">{fmt(prob)}</text>
      </g>
    {/each}

    <!-- Axis baseline under pmf -->
    <line
      x1={pmfX0 - 2} y1={padT + innerH}
      x2={pmfX0 + pmfW + 2} y2={padT + innerH}
      class="axis"
    />

    <!-- ============ Center: stacked CDF ============ -->
    {#each p as _, i}
      {@const yk = cdfBinTop(i)}
      {@const hk = cdfBinHeight(i)}
      {@const inBin = lastU !== null && (i === 0 ? lastU <= cdf[0] : lastU > cdf[i - 1] && lastU <= cdf[i])}
      <rect
        x={cdfX0}
        y={yk}
        width={cdfW}
        height={hk}
        class="bar cdf"
        class:landed={inBin}
      />
      <text
        x={cdfX0 + cdfW + 4}
        y={yk + hk / 2 + 4}
        class="cdf-label"
      >{labels[i] ?? `x_${i}`}</text>
    {/each}
    <!-- CDF outline -->
    <rect x={cdfX0} y={padT} width={cdfW} height={innerH} fill="none" class="axis" />

    <!-- The ball / u marker -->
    {#if lastU !== null}
      {@const uy = uY(lastU)}
      <line
        x1={cdfX0 - 8}
        y1={uy}
        x2={cdfX0 + cdfW + 8}
        y2={uy}
        class="u-line"
      />
      <circle
        cx={cdfX0 - 14}
        cy={uy}
        r={5}
        class="u-ball"
        class:settling={dropping}
      />
      <text
        x={cdfX0 - 22}
        y={uy + 4}
        text-anchor="end"
        class="u-num"
      >u = {fmt(lastU)}</text>
    {/if}

    <!-- y-axis ticks 0, 1 for the CDF -->
    <text x={cdfX0 - 4} y={padT + 4} text-anchor="end" class="tick-num">1</text>
    <text x={cdfX0 - 4} y={padT + innerH + 4} text-anchor="end" class="tick-num">0</text>

    <!-- ============ Right: empirical histogram ============ -->
    {#each empirical as q, i}
      {@const x = histBarX(i)}
      {@const y = barTop(q)}
      {@const h = barH(q)}
      <rect x={x} y={y} width={histBarW} height={h} rx={3} ry={3} class="bar hist" />
      <!-- True probability overlay: a thin line at p[i]. -->
      <line
        x1={x}
        x2={x + histBarW}
        y1={barTop(p[i])}
        y2={barTop(p[i])}
        class="true-mark"
      />
      <text x={x + histBarW / 2} y={padT + innerH + 14} class="tick-label">{labels[i] ?? `x_${i}`}</text>
    {/each}
    <line
      x1={histX0 - 2} y1={padT + innerH}
      x2={histX0 + histW + 2} y2={padT + innerH}
      class="axis"
    />
  </svg>

  <div class="controls">
    <div class="readout">
      {#if lastIdx !== null}
        <span class="result-label">drew <strong>{labels[lastIdx]}</strong></span>
      {:else}
        <span class="result-label muted">drop a ball</span>
      {/if}
      <span class="count">{nSamples} {nSamples === 1 ? 'sample' : 'samples'}</span>
    </div>
    <div class="buttons">
      <button type="button" class="btn primary" onclick={drop} disabled={auto}>Drop one</button>
      {#if !hideAuto}
        <button
          type="button"
          class="btn"
          onclick={() => (auto ? stopAuto() : startAuto())}
        >{auto ? 'Stop' : 'Drop 200'}</button>
      {/if}
      <button type="button" class="btn ghost" onclick={resetSamples}>Reset</button>
    </div>
  </div>
</div>

<style>
  .widget {
    --c-pmf:    var(--ink-red);
    --c-cdf:    var(--ink-sea);
    --c-hist:   var(--ink-teal);
    --c-landed: var(--ink-sun);
    --c-true:   var(--ink-coral);
    --c-axis:   color-mix(in srgb, var(--site-fg) 30%, transparent);
    --c-label:  color-mix(in srgb, var(--site-fg) 70%, transparent);

    background: var(--demo-card);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 1px 0 rgba(0,0,0,0.04), 0 12px 32px -24px rgba(0,0,0,0.18);
  }
  .stage {
    width: 100%;
    height: auto;
    display: block;
    background: var(--demo-stage);
    border-radius: 12px;
  }
  .panel-label {
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    text-anchor: middle;
    fill: var(--c-label);
  }
  .axis { stroke: var(--c-axis); stroke-width: 1; }
  .tick-label {
    font-family: var(--font-mono);
    font-size: 12px;
    text-anchor: middle;
    fill: var(--site-fg);
  }
  .tick-num {
    font-family: var(--font-mono);
    font-size: 11px;
    text-anchor: middle;
    fill: var(--c-label);
  }
  .cdf-label {
    font-family: var(--font-mono);
    font-size: 11px;
    fill: var(--c-label);
  }
  .bar.pmf  { fill: var(--c-pmf);  }
  .bar.cdf  { fill: color-mix(in srgb, var(--c-cdf) 75%, transparent); stroke: var(--c-cdf); stroke-width: 1; }
  .bar.cdf.landed { fill: var(--c-landed); stroke: var(--c-landed); }
  .bar.hist { fill: var(--c-hist); transition: y 120ms ease-out, height 120ms ease-out; }
  .pmf-bar.active .bar.pmf { filter: drop-shadow(0 0 6px color-mix(in srgb, var(--c-pmf) 60%, transparent)); }
  .true-mark { stroke: var(--c-true); stroke-width: 2; }

  .u-line {
    stroke: var(--c-landed);
    stroke-width: 1.5;
    stroke-dasharray: 4 3;
  }
  .u-ball {
    fill: var(--c-landed);
    stroke: color-mix(in srgb, var(--c-landed) 50%, #000);
    stroke-width: 1;
    transition: transform 120ms ease-out;
  }
  .u-num {
    font-family: var(--font-mono);
    font-size: 12px;
    fill: var(--site-fg);
    font-weight: 600;
  }

  .controls {
    margin-top: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }
  .readout {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .result-label {
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--site-fg);
  }
  .result-label.muted { opacity: 0.6; }
  .result-label strong { color: var(--c-pmf); }
  .count {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--c-label);
  }
  .buttons { display: flex; gap: 8px; flex-wrap: wrap; }
  .btn {
    font-family: var(--font-mono);
    font-size: 13px;
    padding: 6px 12px;
    border-radius: 8px;
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: var(--demo-stage);
    color: var(--site-fg);
    cursor: pointer;
    transition: transform 120ms ease-out, background 160ms ease-out;
  }
  .btn:hover:not(:disabled) { transform: translateY(-1px); }
  .btn:disabled { opacity: 0.5; cursor: default; }
  .btn.primary {
    background: var(--c-pmf);
    color: white;
    border-color: var(--c-pmf);
  }
  .btn.ghost { background: transparent; }
</style>
