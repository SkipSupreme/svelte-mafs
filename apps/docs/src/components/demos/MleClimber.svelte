<script lang="ts">
  /**
   * Maximum likelihood for a Bernoulli, made tangible.
   * - A row of clickable H/T glyphs is the dataset. Toggle any flip.
   * - L(p) and ℓ(p) = log L(p) curves are plotted side-by-side.
   * - A draggable vertical line picks the candidate p̂.
   * - "Snap to MLE" jumps to the analytic optimum p̂ = n1/n.
   */

  interface Props {
    initialData?: ('H' | 'T')[];
  }

  let {
    initialData = ['H', 'H', 'T', 'H', 'H', 'T', 'H', 'H', 'T', 'H'],
  }: Props = $props();

  let data: ('H' | 'T')[] = $state([...initialData]);
  let phat = $state(0.5);

  const n = $derived(data.length);
  const n1 = $derived(data.filter((c) => c === 'H').length);
  const n0 = $derived(n - n1);
  const argmax = $derived(n > 0 ? n1 / n : 0.5);

  function toggle(i: number) {
    const next = [...data];
    next[i] = next[i] === 'H' ? 'T' : 'H';
    data = next;
  }
  function snap() { phat = argmax; }
  function addFlip(face: 'H' | 'T') {
    if (data.length >= 18) return;
    data = [...data, face];
  }
  function reset() { data = [...initialData]; phat = 0.5; }

  function safeLog(x: number): number {
    return x <= 0 ? -1e9 : Math.log(x);
  }

  function L(p: number): number {
    return Math.pow(p, n1) * Math.pow(1 - p, n0);
  }
  function logL(p: number): number {
    return n1 * safeLog(p) + n0 * safeLog(1 - p);
  }

  const Lhat = $derived(L(phat));
  const lhat = $derived(logL(phat));
  const LMax = $derived(L(argmax));
  const lMax = $derived(logL(argmax));

  // ---- Plot geometry ----
  const W = 560;
  const H = 320;
  const padL = 40;
  const padR = 16;
  const padT = 12;
  const padB = 28;
  const plotsGap = 22;
  const colW = (W - padL - padR - plotsGap) / 2;
  const plotH = H - padT - padB;

  // Likelihood plot: x = phat in [0,1], y in [0, max(L)].
  // For numerical headroom we scale to LMax so the curve is always visible.
  const LplotX0 = padL;
  const LplotY0 = padT;
  // Log-likelihood plot: x = phat in [0,1], y in [logL(0+ε), logL(argmax)].
  const RplotX0 = padL + colW + plotsGap;
  const RplotY0 = padT;

  const yLikeMax = $derived(Math.max(LMax, 1e-12));
  // Pick a sensible y-window for log-likelihood: from logL(argmax) - some_drop down to logL(argmax).
  // Clamp so we don't plot deep -∞ outliers.
  const yLogMax = $derived(lMax);
  const yLogMin = $derived(lMax - Math.max(8, 0.5 * n + 4));

  function L_xToPx(p: number): number { return LplotX0 + p * colW; }
  function L_yToPx(y: number): number {
    return LplotY0 + (1 - y / yLikeMax) * plotH;
  }
  function l_xToPx(p: number): number { return RplotX0 + p * colW; }
  function l_yToPx(y: number): number {
    const span = yLogMax - yLogMin;
    if (span <= 0) return RplotY0 + plotH;
    return RplotY0 + (1 - (y - yLogMin) / span) * plotH;
  }

  // Sampled curve paths.
  const Lpath = $derived.by(() => {
    const steps = 220;
    let d = '';
    for (let i = 0; i <= steps; i++) {
      const p = i / steps;
      const y = L(p);
      const px = L_xToPx(p).toFixed(2);
      const py = L_yToPx(y).toFixed(2);
      d += (i === 0 ? 'M' : 'L') + px + ',' + py + ' ';
    }
    return d.trim();
  });
  const lpath = $derived.by(() => {
    const steps = 220;
    let d = '';
    let started = false;
    for (let i = 0; i <= steps; i++) {
      const p = i / steps;
      const y = logL(p);
      if (!isFinite(y) || y < yLogMin - 0.5) { started = false; continue; }
      const px = l_xToPx(p).toFixed(2);
      const py = l_yToPx(y).toFixed(2);
      d += (started ? 'L' : 'M') + px + ',' + py + ' ';
      started = true;
    }
    return d.trim();
  });

  // ---- Drag the candidate ----
  let dragging = $state(false);
  function startDrag(e: PointerEvent) {
    dragging = true;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    onDrag(e);
  }
  function onDrag(e: PointerEvent) {
    if (!dragging) return;
    const svg = (e.currentTarget as Element).ownerSVGElement;
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    const xLocal = ((e.clientX - r.left) / r.width) * W;
    // Decide which column we're in.
    let p: number;
    if (xLocal < LplotX0 + colW + plotsGap / 2) {
      p = (xLocal - LplotX0) / colW;
    } else {
      p = (xLocal - RplotX0) / colW;
    }
    phat = Math.max(0.001, Math.min(0.999, p));
  }
  function endDrag(e: PointerEvent) {
    if (!dragging) return;
    try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch { /* noop */ }
    dragging = false;
  }
</script>

<div class="widget">
  <!-- Dataset -->
  <div class="data-row">
    <div class="data-label">Data ({n1} H / {n0} T) — click any flip to toggle</div>
    <div class="flips">
      {#each data as face, i (i)}
        <button
          type="button"
          class="flip"
          class:H={face === 'H'}
          class:T={face === 'T'}
          onclick={() => toggle(i)}
          aria-label={`Flip ${i + 1} is ${face}; click to toggle`}
        >{face}</button>
      {/each}
      <button type="button" class="flip add" onclick={() => addFlip('H')} aria-label="Add an H">+H</button>
      <button type="button" class="flip add" onclick={() => addFlip('T')} aria-label="Add a T">+T</button>
    </div>
  </div>

  <!-- Plots -->
  <svg
    viewBox={`0 0 ${W} ${H}`}
    role="img"
    aria-label="Likelihood and log-likelihood plots"
    class="stage"
    style="touch-action: none"
    onpointerdown={startDrag}
    onpointermove={onDrag}
    onpointerup={endDrag}
    onpointercancel={endDrag}
  >
    <!-- Likelihood pane -->
    <text x={LplotX0 + colW / 2} y={padT - 2} text-anchor="middle" class="pane-title">
      L(p) = p^{n1} (1−p)^{n0}
    </text>
    <line x1={LplotX0} y1={padT + plotH} x2={LplotX0 + colW} y2={padT + plotH} class="axis" />
    <line x1={LplotX0} y1={padT} x2={LplotX0} y2={padT + plotH} class="axis" />
    <!-- Curve -->
    <path d={Lpath} class="curve like" fill="none" />
    <!-- Candidate vertical line -->
    <line
      x1={L_xToPx(phat)}
      x2={L_xToPx(phat)}
      y1={padT}
      y2={padT + plotH}
      class="cand"
    />
    <circle cx={L_xToPx(phat)} cy={L_yToPx(Lhat)} r={5} class="cand-dot" />
    <!-- Argmax marker -->
    <line
      x1={L_xToPx(argmax)}
      x2={L_xToPx(argmax)}
      y1={padT + plotH - 8}
      y2={padT + plotH + 4}
      class="argmax-mark"
    />
    <text x={L_xToPx(argmax)} y={padT + plotH + 18} text-anchor="middle" class="argmax-label">n₁/n</text>

    <!-- p axis ticks -->
    {#each [0, 0.5, 1] as t}
      <text x={L_xToPx(t)} y={padT + plotH + 18} text-anchor="middle" class="tick">{t}</text>
    {/each}

    <!-- Log-likelihood pane -->
    <text x={RplotX0 + colW / 2} y={padT - 2} text-anchor="middle" class="pane-title">
      ℓ(p) = log L(p)
    </text>
    <line x1={RplotX0} y1={padT + plotH} x2={RplotX0 + colW} y2={padT + plotH} class="axis" />
    <line x1={RplotX0} y1={padT} x2={RplotX0} y2={padT + plotH} class="axis" />
    <path d={lpath} class="curve log" fill="none" />
    <line
      x1={l_xToPx(phat)}
      x2={l_xToPx(phat)}
      y1={padT}
      y2={padT + plotH}
      class="cand"
    />
    <circle cx={l_xToPx(phat)} cy={l_yToPx(lhat)} r={5} class="cand-dot" />
    <line
      x1={l_xToPx(argmax)}
      x2={l_xToPx(argmax)}
      y1={padT + plotH - 8}
      y2={padT + plotH + 4}
      class="argmax-mark"
    />
    <text x={l_xToPx(argmax)} y={padT + plotH + 18} text-anchor="middle" class="argmax-label">n₁/n</text>
    {#each [0, 0.5, 1] as t}
      <text x={l_xToPx(t)} y={padT + plotH + 18} text-anchor="middle" class="tick">{t}</text>
    {/each}
  </svg>

  <div class="readout">
    <div class="stats">
      <div class="stat"><span class="k">p̂</span><span class="v">{phat.toFixed(3)}</span></div>
      <div class="stat"><span class="k">L(p̂)</span><span class="v">{Lhat.toExponential(2)}</span></div>
      <div class="stat"><span class="k">ℓ(p̂)</span><span class="v">{isFinite(lhat) ? lhat.toFixed(3) : '−∞'}</span></div>
      <div class="sep" aria-hidden="true">·</div>
      <div class="stat hi"><span class="k">argmax = n₁/n</span><span class="v">{argmax.toFixed(3)}</span></div>
    </div>
    <div class="buttons">
      <button type="button" class="btn primary" onclick={snap}>Snap to MLE</button>
      <button type="button" class="btn ghost" onclick={reset}>Reset</button>
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
  .data-row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }
  .data-label {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .flips { display: flex; gap: 5px; flex-wrap: wrap; }
  .flip {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: var(--demo-stage);
    color: var(--site-fg);
    font-family: var(--font-mono);
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    transition: transform 120ms ease-out, background 160ms ease-out;
  }
  .flip.H { background: var(--ink-red); color: white; border-color: var(--ink-red); }
  .flip.T { background: var(--ink-sea); color: white; border-color: var(--ink-sea); }
  .flip.add {
    background: transparent;
    color: var(--site-fg-muted);
    font-size: 12px;
  }
  .flip:hover { transform: translateY(-1px); }

  .stage {
    display: block;
    width: 100%;
    height: auto;
    background: var(--demo-stage);
    border-radius: 12px;
    cursor: ew-resize;
  }
  .axis { stroke: color-mix(in srgb, var(--site-fg) 28%, transparent); stroke-width: 1; }
  .pane-title {
    font-family: var(--font-mono);
    font-size: 11px;
    fill: var(--site-fg-muted);
    letter-spacing: 0.06em;
  }
  .curve.like { stroke: var(--ink-red); stroke-width: 2.5; }
  .curve.log  { stroke: var(--ink-sea); stroke-width: 2.5; }
  .cand {
    stroke: var(--ink-sun);
    stroke-width: 1.5;
    stroke-dasharray: 4 4;
  }
  .cand-dot {
    fill: var(--ink-sun);
    stroke: white;
    stroke-width: 1.5;
  }
  .argmax-mark {
    stroke: color-mix(in srgb, var(--site-fg) 70%, transparent);
    stroke-width: 2;
  }
  .argmax-label {
    font-family: var(--font-mono);
    font-size: 10px;
    fill: var(--site-fg-muted);
  }
  .tick {
    font-family: var(--font-mono);
    font-size: 11px;
    fill: var(--site-fg-muted);
  }

  .readout {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-top: 12px;
    flex-wrap: wrap;
  }
  .stats {
    display: flex;
    gap: 14px;
    align-items: baseline;
    font-family: var(--font-mono);
    font-size: 13px;
    flex-wrap: wrap;
  }
  .stat { display: inline-flex; gap: 4px; align-items: baseline; }
  .stat .k { color: var(--site-fg-muted); }
  .stat .v { color: var(--site-fg); font-weight: 600; }
  .stat.hi .k { color: color-mix(in srgb, var(--ink-sun) 70%, var(--site-fg-muted)); }
  .stat.hi .v { color: var(--ink-sun); }
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
    transition: transform 120ms ease-out;
  }
  .btn:hover { transform: translateY(-1px); }
  .btn.primary { background: var(--ink-sun); color: var(--site-fg); border-color: var(--ink-sun); font-weight: 700; }
  .btn.ghost { background: transparent; }
</style>
