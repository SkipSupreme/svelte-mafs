<script lang="ts">
  /**
   * Natural-frequency grid for Bayes' theorem.
   * Population is rendered as four rectangles:
   *   row 1 = "sick" (height = prev),       row 2 = "well" (height = 1 - prev)
   *   within "sick":  TP (left, width = sens)         | FN (right, width = 1 - sens)
   *   within "well":  FP (left, width = 1 - spec)     | TN (right, width = spec)
   * The big readout is P(sick | +) = TP / (TP + FP).
   * Drag the horizontal divider to change prev; drag the vertical dividers to change sens/spec.
   */

  interface Props {
    initialPrev?: number;
    initialSens?: number;
    initialSpec?: number;
  }
  let {
    initialPrev = 0.01,
    initialSens = 0.95,
    initialSpec = 0.95,
  }: Props = $props();

  let prev = $state(clampUnit(initialPrev));
  let sens = $state(clampUnit(initialSens));
  let spec = $state(clampUnit(initialSpec));

  function clampUnit(x: number): number {
    return Math.max(0.001, Math.min(0.999, x));
  }

  const tp = $derived(prev * sens);
  const fn = $derived(prev * (1 - sens));
  const fp = $derived((1 - prev) * (1 - spec));
  const tn = $derived((1 - prev) * spec);

  const positiveMass = $derived(tp + fp);
  const posterior = $derived(positiveMass > 0 ? tp / positiveMass : 0);

  // ---- Geometry ----
  const W = 560;
  const H = 360;
  const padL = 36;
  const padR = 16;
  const padT = 18;
  const padB = 32;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // y_split: the y-pixel splitting sick (top) from well (bottom).
  // Math: row 1 occupies prev fraction of plot height.
  const ySplit = $derived(padT + plotH * prev);
  // x_split within sick row (top): sens fraction from the left.
  const xSplitSick = $derived(padL + plotW * sens);
  // x_split within well row (bottom): (1 - spec) fraction from the left.
  const xSplitWell = $derived(padL + plotW * (1 - spec));

  // ---- Drag interactions ----
  type DragMode = null | 'prev' | 'sens' | 'spec';
  let dragging: DragMode = $state(null);

  function svgPoint(e: PointerEvent): [number, number] | null {
    const svg = (e.currentTarget as Element).ownerSVGElement;
    if (!svg) return null;
    const r = svg.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const y = ((e.clientY - r.top) / r.height) * H;
    return [x, y];
  }

  function startDrag(e: PointerEvent, mode: Exclude<DragMode, null>) {
    dragging = mode;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    onDrag(e);
  }
  function onDrag(e: PointerEvent) {
    if (!dragging) return;
    const p = svgPoint(e);
    if (!p) return;
    if (dragging === 'prev') {
      const f = (p[1] - padT) / plotH;
      prev = clampUnit(f);
    } else if (dragging === 'sens') {
      const f = (p[0] - padL) / plotW;
      sens = clampUnit(f);
    } else if (dragging === 'spec') {
      const f = 1 - (p[0] - padL) / plotW;
      spec = clampUnit(f);
    }
  }
  function endDrag(e: PointerEvent) {
    if (!dragging) return;
    try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch { /* noop */ }
    dragging = null;
  }

  // ---- Display helpers ----
  function pct(x: number): string { return (x * 100).toFixed(1) + '%'; }
  function ppl(x: number): string {
    const n = x * 100000;
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return n < 10 ? n.toFixed(1) : n.toFixed(0);
  }
</script>

<div class="widget">
  <svg
    viewBox={`0 0 ${W} ${H}`}
    role="img"
    aria-label="Natural-frequency grid for Bayes' theorem"
    class="stage"
  >
    <!-- Frame -->
    <rect x={padL} y={padT} width={plotW} height={plotH} class="frame" />

    <!-- Sick row (top) -->
    <!-- TP (left) -->
    <rect
      x={padL}
      y={padT}
      width={Math.max(0, xSplitSick - padL)}
      height={Math.max(0, ySplit - padT)}
      class="block tp"
    />
    <!-- FN (right) -->
    <rect
      x={xSplitSick}
      y={padT}
      width={Math.max(0, padL + plotW - xSplitSick)}
      height={Math.max(0, ySplit - padT)}
      class="block fn"
    />

    <!-- Well row (bottom) -->
    <!-- FP (left) -->
    <rect
      x={padL}
      y={ySplit}
      width={Math.max(0, xSplitWell - padL)}
      height={Math.max(0, padT + plotH - ySplit)}
      class="block fp"
    />
    <!-- TN (right) -->
    <rect
      x={xSplitWell}
      y={ySplit}
      width={Math.max(0, padL + plotW - xSplitWell)}
      height={Math.max(0, padT + plotH - ySplit)}
      class="block tn"
    />

    <!-- Labels inside blocks (only when block is big enough to fit) -->
    {#if (ySplit - padT) > 22 && (xSplitSick - padL) > 30}
      <text x={padL + 8} y={padT + 16} class="block-label light">TP</text>
      <text x={padL + 8} y={padT + 32} class="block-num light">{ppl(tp)}</text>
    {/if}
    {#if (ySplit - padT) > 22 && (padL + plotW - xSplitSick) > 30}
      <text x={xSplitSick + 8} y={padT + 16} class="block-label">FN</text>
      <text x={xSplitSick + 8} y={padT + 32} class="block-num">{ppl(fn)}</text>
    {/if}
    {#if (padT + plotH - ySplit) > 22 && (xSplitWell - padL) > 30}
      <text x={padL + 8} y={ySplit + 16} class="block-label light">FP</text>
      <text x={padL + 8} y={ySplit + 32} class="block-num light">{ppl(fp)}</text>
    {/if}
    {#if (padT + plotH - ySplit) > 22 && (padL + plotW - xSplitWell) > 30}
      <text x={xSplitWell + 8} y={ySplit + 16} class="block-label">TN</text>
      <text x={xSplitWell + 8} y={ySplit + 32} class="block-num">{ppl(tn)}</text>
    {/if}

    <!-- "tests positive" highlight: left column (TP + FP) -->
    <rect
      x={padL - 6}
      y={padT - 6}
      width={6}
      height={Math.max(0, ySplit - padT)}
      class="hl positive"
    />
    <rect
      x={padL - 6}
      y={ySplit}
      width={6}
      height={Math.max(0, padT + plotH - ySplit)}
      class="hl positive"
    />
    <text
      x={padL - 10}
      y={padT - 6}
      text-anchor="end"
      class="axis-label"
    >tests + ←</text>

    <!-- Sick / Well row labels -->
    <text x={padL + plotW + 6} y={padT + (ySplit - padT) / 2 + 4} class="row-label">sick</text>
    <text x={padL + plotW + 6} y={ySplit + (padT + plotH - ySplit) / 2 + 4} class="row-label">well</text>

    <!-- Drag handles -->
    <!-- Prevalence: horizontal divider -->
    <line
      x1={padL}
      x2={padL + plotW}
      y1={ySplit}
      y2={ySplit}
      class="divider"
    />
    <rect
      x={padL}
      y={ySplit - 8}
      width={plotW}
      height={16}
      fill="transparent"
      style="cursor: ns-resize; touch-action: none"
      onpointerdown={(e) => startDrag(e, 'prev')}
      onpointermove={onDrag}
      onpointerup={endDrag}
      onpointercancel={endDrag}
      role="slider"
      tabindex={0}
      aria-label="Prevalence (drag vertically)"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={prev}
    />
    <circle cx={padL - 4} cy={ySplit} r={6} class="grip prev" pointer-events="none" />

    <!-- Sensitivity: vertical divider inside the sick row -->
    {#if ySplit - padT > 12}
      <line
        x1={xSplitSick}
        x2={xSplitSick}
        y1={padT}
        y2={ySplit}
        class="divider"
      />
      <rect
        x={xSplitSick - 8}
        y={padT}
        width={16}
        height={Math.max(0, ySplit - padT)}
        fill="transparent"
        style="cursor: ew-resize; touch-action: none"
        onpointerdown={(e) => startDrag(e, 'sens')}
        onpointermove={onDrag}
        onpointerup={endDrag}
        onpointercancel={endDrag}
        role="slider"
        tabindex={0}
        aria-label="Sensitivity (drag horizontally)"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={sens}
      />
      <circle cx={xSplitSick} cy={padT - 4} r={6} class="grip sens" pointer-events="none" />
    {/if}

    <!-- Specificity: vertical divider inside the well row -->
    {#if padT + plotH - ySplit > 12}
      <line
        x1={xSplitWell}
        x2={xSplitWell}
        y1={ySplit}
        y2={padT + plotH}
        class="divider"
      />
      <rect
        x={xSplitWell - 8}
        y={ySplit}
        width={16}
        height={Math.max(0, padT + plotH - ySplit)}
        fill="transparent"
        style="cursor: ew-resize; touch-action: none"
        onpointerdown={(e) => startDrag(e, 'spec')}
        onpointermove={onDrag}
        onpointerup={endDrag}
        onpointercancel={endDrag}
        role="slider"
        tabindex={0}
        aria-label="Specificity (drag horizontally)"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={spec}
      />
      <circle cx={xSplitWell} cy={padT + plotH + 4} r={6} class="grip spec" pointer-events="none" />
    {/if}

    <!-- Posterior readout floating top-right -->
    <g transform={`translate(${padL + plotW - 200}, ${padT + 8})`}>
      <rect x={0} y={0} width={196} height={64} rx={8} class="readout-bg" />
      <text x={12} y={20} class="readout-label">P(sick | +)</text>
      <text x={12} y={48} class="readout-num">{(posterior * 100).toFixed(2)}%</text>
    </g>
  </svg>

  <div class="controls">
    <div class="params">
      <div class="p">
        <span class="k">prevalence</span>
        <span class="v">{pct(prev)}</span>
      </div>
      <div class="p">
        <span class="k">sensitivity</span>
        <span class="v">{pct(sens)}</span>
      </div>
      <div class="p">
        <span class="k">specificity</span>
        <span class="v">{pct(spec)}</span>
      </div>
    </div>
    <div class="equation">
      <span class="num">{pct(prev)} × {pct(sens)}</span>
      <span class="div">÷ ({pct(prev)} × {pct(sens)} + {pct(1 - prev)} × {pct(1 - spec)}) =</span>
      <span class="ans">{(posterior * 100).toFixed(2)}%</span>
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
  .frame { fill: none; stroke: color-mix(in srgb, var(--site-fg) 18%, transparent); stroke-width: 1; }
  .block.tp { fill: var(--ink-red); }
  .block.fn { fill: color-mix(in srgb, var(--ink-red) 22%, transparent); }
  .block.fp { fill: color-mix(in srgb, var(--ink-coral) 70%, transparent); }
  .block.tn { fill: color-mix(in srgb, var(--ink-sea) 18%, transparent); }
  .block-label {
    font-family: var(--font-mono);
    font-size: 11px;
    fill: var(--site-fg);
    font-weight: 700;
    letter-spacing: 0.08em;
    pointer-events: none;
  }
  .block-num {
    font-family: var(--font-mono);
    font-size: 13px;
    fill: var(--site-fg);
    font-weight: 600;
    pointer-events: none;
  }
  .block-label.light, .block-num.light { fill: white; }
  .row-label {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 14px;
    fill: var(--site-fg-muted);
  }
  .axis-label {
    font-family: var(--font-mono);
    font-size: 10px;
    fill: var(--ink-sun);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .hl.positive { fill: var(--ink-sun); }
  .divider {
    stroke: color-mix(in srgb, var(--site-fg) 55%, transparent);
    stroke-width: 2;
    pointer-events: none;
  }
  .grip {
    fill: var(--site-fg);
    stroke: white;
    stroke-width: 1.5;
  }
  .grip.prev { fill: var(--ink-sea); }
  .grip.sens { fill: var(--ink-red); }
  .grip.spec { fill: var(--ink-sea); }
  .readout-bg {
    fill: color-mix(in srgb, var(--ink-sun) 14%, var(--demo-card));
    stroke: var(--ink-sun);
    stroke-width: 1.5;
  }
  .readout-label {
    font-family: var(--font-mono);
    font-size: 11px;
    fill: color-mix(in srgb, var(--ink-sun) 75%, var(--site-fg));
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .readout-num {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 24px;
    fill: var(--site-fg);
  }

  .controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-top: 12px;
    flex-wrap: wrap;
  }
  .params {
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    font-family: var(--font-mono);
    font-size: 13px;
  }
  .p .k { color: var(--site-fg-muted); margin-right: 6px; }
  .p .v { color: var(--site-fg); font-weight: 600; }
  .equation {
    display: flex;
    align-items: baseline;
    gap: 6px;
    flex-wrap: wrap;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--site-fg-muted);
  }
  .equation .ans {
    color: var(--ink-sun);
    font-weight: 700;
    font-size: 14px;
  }
</style>
