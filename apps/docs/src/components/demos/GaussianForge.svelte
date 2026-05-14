<script lang="ts">
  import { play } from '../../lib/sound';

  interface Props {
    initialMu?: number;
    initialSigma?: number;
    /** Lock μ — used when only σ-driven shape changes are pedagogically interesting. */
    lockMu?: boolean;
    /** Hide the "Standardize → N(0,1)" affordance — for lessons before standardization is named. */
    hideStandardize?: boolean;
  }

  let {
    initialMu = 0,
    initialSigma = 1,
    lockMu = false,
    hideStandardize = false,
  }: Props = $props();

  let mu = $state(initialMu);
  let sigma = $state(Math.max(0.2, initialSigma));

  // ---- Math ----
  const X_MIN = -5;
  const X_MAX = 5;
  // We let Y go a bit past 1 so the "is this a probability?" moment lands when
  // σ shrinks and the peak rears past 1. DESIGN.md surprises live above 1.
  const Y_MIN = 0;
  const Y_MAX = 2.2;

  const gaussian = (x: number) =>
    (1 / (sigma * Math.sqrt(2 * Math.PI))) *
    Math.exp(-((x - mu) ** 2) / (2 * sigma ** 2));

  const peakDensity = $derived(1 / (sigma * Math.sqrt(2 * Math.PI)));

  // ---- Histogram ----
  const BINS = 36;
  const binWidth = (X_MAX - X_MIN) / BINS;
  let counts: number[] = $state(Array(BINS).fill(0));
  let total = $state(0);
  let emp_mean = $state(0);
  let emp_M2 = $state(0);

  function binCenter(i: number): number { return X_MIN + (i + 0.5) * binWidth; }
  function binDensity(c: number): number {
    return total > 0 ? c / total / binWidth : 0;
  }

  function sampleStandardNormal(): number {
    let u1 = 0;
    while (u1 === 0) u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  function sampleN(n: number) {
    const next = [...counts];
    let m = emp_mean;
    let m2 = emp_M2;
    let t = total;
    for (let k = 0; k < n; k++) {
      const x = mu + sigma * sampleStandardNormal();
      t += 1;
      const d = x - m;
      m += d / t;
      const d2 = x - m;
      m2 += d * d2;
      const idx = Math.floor((x - X_MIN) / binWidth);
      if (idx >= 0 && idx < BINS) next[idx] += 1;
    }
    counts = next;
    total = t;
    emp_mean = m;
    emp_M2 = m2;
    play('tick');
  }

  const emp_std = $derived(total > 1 ? Math.sqrt(emp_M2 / total) : 0);

  function reset() {
    counts = Array(BINS).fill(0);
    total = 0;
    emp_mean = 0;
    emp_M2 = 0;
  }
  function standardize() {
    mu = 0;
    sigma = 1;
  }

  // ---- SVG geometry ----
  const W = 560;
  const H = 320;
  const padL = 28;
  const padR = 16;
  const padT = 18;
  const padB = 36;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  function xToPx(x: number): number {
    return padL + ((x - X_MIN) / (X_MAX - X_MIN)) * plotW;
  }
  function yToPx(y: number): number {
    return padT + (1 - (y - Y_MIN) / (Y_MAX - Y_MIN)) * plotH;
  }

  // Curve path (sampled at high resolution).
  const curvePath = $derived.by(() => {
    const steps = 280;
    const dx = (X_MAX - X_MIN) / steps;
    let d = '';
    for (let i = 0; i <= steps; i++) {
      const x = X_MIN + i * dx;
      const y = gaussian(x);
      const px = xToPx(x);
      const py = yToPx(y);
      d += (i === 0 ? 'M' : 'L') + px.toFixed(2) + ',' + py.toFixed(2) + ' ';
    }
    return d.trim();
  });

  // ---- Handles ----
  // Peak handle: at (mu, peakDensity). Drag along x to change mu.
  const peakHX = $derived(xToPx(mu));
  const peakHY = $derived(yToPx(peakDensity));
  // Inflection handle: at (mu + sigma, peakDensity * exp(-1/2)). Drag along x to change sigma.
  const inflX = $derived(xToPx(mu + sigma));
  const inflY = $derived(yToPx(peakDensity * Math.exp(-0.5)));

  type DragMode = null | 'peak' | 'infl';
  let dragMode: DragMode = $state(null);

  function svgFromPointer(e: PointerEvent): [number, number] | null {
    const svg = (e.currentTarget as Element).ownerSVGElement;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const xLocal = ((e.clientX - rect.left) / rect.width) * W;
    const yLocal = ((e.clientY - rect.top) / rect.height) * H;
    return [xLocal, yLocal];
  }
  function pxToX(px: number): number {
    return X_MIN + ((px - padL) / plotW) * (X_MAX - X_MIN);
  }

  function startDrag(e: PointerEvent, mode: 'peak' | 'infl') {
    if (mode === 'peak' && lockMu) return;
    dragMode = mode;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    onDrag(e);
  }
  function onDrag(e: PointerEvent) {
    if (!dragMode) return;
    const p = svgFromPointer(e);
    if (!p) return;
    const xMath = Math.max(X_MIN + 0.1, Math.min(X_MAX - 0.1, pxToX(p[0])));
    if (dragMode === 'peak') {
      mu = xMath;
    } else {
      // Inflection lives at mu + sigma. Clamp sigma to a visible range.
      const s = Math.max(0.2, Math.min(3.0, xMath - mu));
      sigma = s;
    }
  }
  function endDrag(e: PointerEvent) {
    if (!dragMode) return;
    try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch { /* noop */ }
    dragMode = null;
  }
</script>

<div class="widget">
  <svg
    viewBox={`0 0 ${W} ${H}`}
    role="img"
    aria-label="Gaussian probability density with draggable μ and σ"
    class="stage"
  >
    <!-- y axis -->
    <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} class="axis" />
    <!-- y = 1 reference (the threshold density can cross past) -->
    <line
      x1={padL}
      y1={yToPx(1)}
      x2={padL + plotW}
      y2={yToPx(1)}
      class="ref"
    />
    <text x={padL - 4} y={yToPx(1) + 4} text-anchor="end" class="ref-label">1.0</text>

    <!-- x axis -->
    <line
      x1={padL}
      y1={padT + plotH}
      x2={padL + plotW}
      y2={padT + plotH}
      class="axis"
    />
    {#each [-4, -2, 0, 2, 4] as tick}
      <line
        x1={xToPx(tick)}
        y1={padT + plotH}
        x2={xToPx(tick)}
        y2={padT + plotH + 4}
        class="axis"
      />
      <text x={xToPx(tick)} y={padT + plotH + 18} text-anchor="middle" class="tick-num">{tick}</text>
    {/each}

    <!-- Histogram bins. -->
    {#each counts as c, i}
      {@const dens = binDensity(c)}
      {#if dens > 0}
        {@const left = xToPx(binCenter(i) - binWidth / 2)}
        {@const right = xToPx(binCenter(i) + binWidth / 2)}
        {@const top = yToPx(dens)}
        <rect
          x={left + 0.5}
          y={top}
          width={Math.max(0, right - left - 1)}
          height={Math.max(0, padT + plotH - top)}
          class="bin"
        />
      {/if}
    {/each}

    <!-- Gaussian curve. -->
    <path d={curvePath} class="curve" fill="none" />

    <!-- μ vertical guide -->
    <line
      x1={xToPx(mu)}
      x2={xToPx(mu)}
      y1={peakHY}
      y2={padT + plotH}
      class="guide mu"
    />
    <!-- σ horizontal guide (from peak x to inflection x) -->
    <line
      x1={peakHX}
      x2={inflX}
      y1={inflY}
      y2={inflY}
      class="guide sigma"
    />

    <!-- Peak handle (μ) -->
    <circle
      cx={peakHX}
      cy={peakHY}
      r={9}
      class="handle peak"
      class:locked={lockMu}
      style="touch-action: none; cursor: {lockMu ? 'default' : 'ew-resize'}"
      onpointerdown={(e) => startDrag(e, 'peak')}
      onpointermove={onDrag}
      onpointerup={endDrag}
      onpointercancel={endDrag}
      role="slider"
      tabindex={lockMu ? -1 : 0}
      aria-label="Peak (drag to change μ)"
      aria-valuemin={X_MIN}
      aria-valuemax={X_MAX}
      aria-valuenow={mu}
    />
    <text x={peakHX} y={peakHY - 14} text-anchor="middle" class="handle-label">μ</text>

    <!-- Inflection handle (σ) -->
    <circle
      cx={inflX}
      cy={inflY}
      r={9}
      class="handle infl"
      style="touch-action: none; cursor: ew-resize"
      onpointerdown={(e) => startDrag(e, 'infl')}
      onpointermove={onDrag}
      onpointerup={endDrag}
      onpointercancel={endDrag}
      role="slider"
      aria-label="Inflection point (drag to change σ)"
      aria-valuemin={0.2}
      aria-valuemax={3}
      aria-valuenow={sigma}
      tabindex={0}
    />
    <text x={inflX} y={inflY - 14} text-anchor="middle" class="handle-label sigma">σ</text>
  </svg>

  <div class="controls">
    <div class="readouts">
      <div class="param"><span class="k">μ</span><span class="v mu">{mu.toFixed(2)}</span></div>
      <div class="param"><span class="k">σ</span><span class="v sigma">{sigma.toFixed(2)}</span></div>
      <div class="sep" aria-hidden="true">·</div>
      <div class="param"><span class="k">x̄</span><span class="v">{total > 0 ? emp_mean.toFixed(3) : '—'}</span></div>
      <div class="param"><span class="k">s</span><span class="v">{total > 1 ? emp_std.toFixed(3) : '—'}</span></div>
      <div class="sep" aria-hidden="true">·</div>
      <div class="param peak-readout"><span class="k">peak</span><span class="v">{peakDensity.toFixed(3)}</span></div>
    </div>
    <div class="buttons">
      <button type="button" class="btn" onclick={() => sampleN(1)}>Sample 1</button>
      <button type="button" class="btn" onclick={() => sampleN(100)}>Sample 100</button>
      <button type="button" class="btn" onclick={() => sampleN(10000)}>Sample 10k</button>
      {#if !hideStandardize}
        <button type="button" class="btn ghost" onclick={standardize}>Standardize → N(0,1)</button>
      {/if}
      <button type="button" class="btn ghost" onclick={reset}>Reset samples</button>
    </div>
  </div>
</div>

<style>
  .widget {
    background: var(--demo-card);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 1px 0 rgba(0,0,0,0.04), 0 12px 32px -24px rgba(0,0,0,0.18);
  }
  .stage {
    display: block;
    width: 100%;
    height: auto;
    background: var(--demo-stage);
    border-radius: 12px;
  }
  .axis {
    stroke: color-mix(in srgb, var(--site-fg) 28%, transparent);
    stroke-width: 1;
  }
  .ref {
    stroke: color-mix(in srgb, var(--ink-sun) 60%, transparent);
    stroke-width: 1;
    stroke-dasharray: 4 4;
  }
  .ref-label {
    font-family: var(--font-mono);
    font-size: 10px;
    fill: color-mix(in srgb, var(--ink-sun) 80%, var(--site-fg));
  }
  .tick-num {
    font-family: var(--font-mono);
    font-size: 11px;
    fill: color-mix(in srgb, var(--site-fg) 70%, transparent);
  }
  .curve {
    stroke: var(--ink-red);
    stroke-width: 2.5;
    transition: d 80ms linear;
  }
  .bin {
    fill: color-mix(in srgb, var(--ink-teal) 55%, transparent);
    transition: y 160ms ease-out, height 160ms ease-out;
  }
  .guide.mu {
    stroke: color-mix(in srgb, var(--ink-red) 50%, transparent);
    stroke-width: 1;
    stroke-dasharray: 3 4;
  }
  .guide.sigma {
    stroke: color-mix(in srgb, var(--ink-sea) 70%, transparent);
    stroke-width: 1.5;
  }
  .handle {
    fill: var(--ink-red);
    stroke: white;
    stroke-width: 2;
    transition: r 120ms ease-out;
  }
  .handle:hover, .handle:focus { r: 11; outline: none; }
  .handle.peak.locked { fill: color-mix(in srgb, var(--ink-red) 40%, transparent); }
  .handle.infl { fill: var(--ink-sea); }
  .handle-label {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 14px;
    fill: var(--ink-red);
    font-weight: 700;
    pointer-events: none;
  }
  .handle-label.sigma { fill: var(--ink-sea); }

  .controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-top: 12px;
    flex-wrap: wrap;
  }
  .readouts {
    display: flex;
    gap: 14px;
    align-items: baseline;
    font-family: var(--font-mono);
    font-size: 13px;
  }
  .param { display: inline-flex; gap: 4px; align-items: baseline; }
  .param .k { color: var(--site-fg-muted); }
  .param .v { color: var(--site-fg); font-weight: 600; }
  .param .v.mu { color: var(--ink-red); }
  .param .v.sigma { color: var(--ink-sea); }
  .param.peak-readout .k { color: color-mix(in srgb, var(--ink-sun) 70%, var(--site-fg-muted)); }
  .param.peak-readout .v { color: var(--ink-sun); }
  .sep { color: color-mix(in srgb, var(--site-fg) 30%, transparent); }
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
  .btn:hover { transform: translateY(-1px); }
  .btn.ghost { background: transparent; }
</style>
