<script lang="ts">
  /**
   * 5x5 joint pmf heatmap.
   * - Click-drag on a cell to "paint" weight onto it; the joint is auto-normalized.
   * - Top strip = marginal p_X(x) (sum across rows).
   * - Right strip = marginal p_Y(y) (sum across columns).
   * - Click a row label to extract the conditional p(Y | X = x); same for column labels.
   * The conditional pops out as a separate bar chart below.
   */

  interface Props {
    /** Initial joint counts (5x5). Auto-normalized on load. */
    initialCounts?: number[][];
    /** Labels for X (columns) and Y (rows). */
    xLabels?: string[];
    yLabels?: string[];
  }

  function defaultJoint(): number[][] {
    // A loosely-correlated default: probability bunched near the diagonal.
    const N = 5;
    const out: number[][] = [];
    for (let i = 0; i < N; i++) {
      const row: number[] = [];
      for (let j = 0; j < N; j++) {
        const d = Math.abs(i - j);
        row.push(Math.max(0.05, 1 - d * 0.22 + (Math.random() - 0.5) * 0.05));
      }
      out.push(row);
    }
    return out;
  }

  let {
    initialCounts = defaultJoint(),
    xLabels = ['x₁', 'x₂', 'x₃', 'x₄', 'x₅'],
    yLabels = ['y₁', 'y₂', 'y₃', 'y₄', 'y₅'],
  }: Props = $props();

  const N = 5;

  // Joint stored as a flat 25-entry probability vector that sums to 1.
  // Index = row * 5 + col (row = y index, col = x index).
  function normalizeMatrix(m: number[][]): number[] {
    const flat: number[] = [];
    let s = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const v = Math.max(0, m[i]?.[j] ?? 0);
      flat.push(v); s += v;
    }
    if (s <= 0) return flat.map(() => 1 / (N * N));
    return flat.map((v) => v / s);
  }

  let joint: number[] = $state(normalizeMatrix(initialCounts));
  let conditionOn: { axis: 'row' | 'col'; idx: number } | null = $state(null);

  // ---- Derived marginals ----
  const pY: number[] = $derived.by(() => {
    // row marginal: sum across columns
    const out: number[] = Array(N).fill(0);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) out[i] += joint[i * N + j];
    return out;
  });
  const pX: number[] = $derived.by(() => {
    // column marginal: sum across rows
    const out: number[] = Array(N).fill(0);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) out[j] += joint[i * N + j];
    return out;
  });

  const maxCellProb = $derived.by(() => {
    let m = 0;
    for (const v of joint) if (v > m) m = v;
    return Math.max(m, 0.04);
  });
  const maxMarg = $derived(Math.max(...pX, ...pY, 0.1));

  // ---- Conditional ----
  const conditional = $derived.by(() => {
    if (!conditionOn) return null;
    if (conditionOn.axis === 'row') {
      const i = conditionOn.idx;
      const row: number[] = [];
      for (let j = 0; j < N; j++) row.push(joint[i * N + j]);
      const s = row.reduce((a, b) => a + b, 0);
      return { kind: 'row' as const, idx: i, vals: s > 0 ? row.map((v) => v / s) : row };
    } else {
      const j = conditionOn.idx;
      const col: number[] = [];
      for (let i = 0; i < N; i++) col.push(joint[i * N + j]);
      const s = col.reduce((a, b) => a + b, 0);
      return { kind: 'col' as const, idx: j, vals: s > 0 ? col.map((v) => v / s) : col };
    }
  });

  // ---- Drag-to-paint ----
  let painting = $state(false);
  let paintMode: 'up' | 'down' = $state('up');

  function paint(i: number, j: number, mode: 'up' | 'down') {
    const next = [...joint];
    const idx = i * N + j;
    // Operate on a non-normalized scratch vector: scale this cell up/down, then renormalize.
    const scratch = next.map((v) => v); // copy
    if (mode === 'up') scratch[idx] += 0.08;
    else scratch[idx] = Math.max(0, scratch[idx] - 0.08);
    const s = scratch.reduce((a, b) => a + b, 0);
    if (s > 0) joint = scratch.map((v) => v / s);
  }

  function onCellDown(e: PointerEvent, i: number, j: number) {
    painting = true;
    paintMode = e.button === 2 || e.shiftKey ? 'down' : 'up';
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    paint(i, j, paintMode);
  }
  function onCellEnter(_e: PointerEvent, i: number, j: number) {
    if (painting) paint(i, j, paintMode);
  }
  function onCellUp(e: PointerEvent, _i: number, _j: number) {
    painting = false;
    try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch { /* noop */ }
  }

  function pickRow(i: number) {
    if (conditionOn && conditionOn.axis === 'row' && conditionOn.idx === i) conditionOn = null;
    else conditionOn = { axis: 'row', idx: i };
  }
  function pickCol(j: number) {
    if (conditionOn && conditionOn.axis === 'col' && conditionOn.idx === j) conditionOn = null;
    else conditionOn = { axis: 'col', idx: j };
  }
  function clearCondition() { conditionOn = null; }
  function reset() { joint = normalizeMatrix(initialCounts); conditionOn = null; }

  // ---- Geometry ----
  const W = 560;
  const H = 360;
  const cellSize = 44;
  const padL = 50;
  const padT = 60;
  const heatW = cellSize * N;  // 220
  const heatH = cellSize * N;  // 220
  const margThickness = 36;

  function cellColor(p: number): string {
    if (p <= 0) return 'var(--demo-stage)';
    const t = Math.min(1, p / maxCellProb);
    return `color-mix(in srgb, var(--ink-red) ${Math.round(t * 100)}%, var(--demo-stage))`;
  }
  function isRowSelected(i: number): boolean { return conditionOn?.axis === 'row' && conditionOn.idx === i; }
  function isColSelected(j: number): boolean { return conditionOn?.axis === 'col' && conditionOn.idx === j; }

  function fmt(p: number): string { return p.toFixed(2); }
</script>

<div class="widget">
  <svg
    viewBox={`0 0 ${W} ${H}`}
    role="img"
    aria-label="Joint distribution heatmap with marginals and conditionals"
    class="stage"
  >
    <!-- Top marginal: p_X(x) -->
    {#each pX as p, j}
      {@const x = padL + j * cellSize}
      {@const barH = (p / maxMarg) * margThickness}
      <rect
        x={x + 4}
        y={padT - margThickness - 4}
        width={cellSize - 8}
        height={margThickness}
        class="marg-bg"
      />
      <rect
        x={x + 4}
        y={padT - 4 - barH}
        width={cellSize - 8}
        height={barH}
        class="marg-bar x"
        class:active={isColSelected(j)}
      />
      <text
        x={x + cellSize / 2}
        y={padT - margThickness - 8}
        text-anchor="middle"
        class="marg-num"
      >{fmt(p)}</text>
    {/each}
    <text x={padL + heatW / 2} y={14} text-anchor="middle" class="marg-title">p_X(x) — column sums</text>

    <!-- Right marginal: p_Y(y) -->
    {#each pY as p, i}
      {@const y = padT + i * cellSize}
      {@const barW = (p / maxMarg) * margThickness}
      <rect
        x={padL + heatW + 4}
        y={y + 4}
        width={margThickness}
        height={cellSize - 8}
        class="marg-bg"
      />
      <rect
        x={padL + heatW + 4}
        y={y + 4}
        width={barW}
        height={cellSize - 8}
        class="marg-bar y"
        class:active={isRowSelected(i)}
      />
      <text
        x={padL + heatW + margThickness + 12}
        y={y + cellSize / 2 + 4}
        class="marg-num"
      >{fmt(p)}</text>
    {/each}
    <text
      x={padL + heatW + margThickness / 2 + 4}
      y={padT + heatH + 18}
      text-anchor="middle"
      class="marg-title"
    >p_Y(y)</text>

    <!-- Column labels (X) -->
    {#each xLabels as label, j}
      {@const x = padL + j * cellSize + cellSize / 2}
      <text
        x={x}
        y={padT + heatH + 22}
        text-anchor="middle"
        class="ax-label"
        class:active={isColSelected(j)}
        style="cursor: pointer"
        onclick={() => pickCol(j)}
        role="button"
        tabindex={0}
      >{label}</text>
    {/each}
    <!-- Row labels (Y) -->
    {#each yLabels as label, i}
      {@const y = padT + i * cellSize + cellSize / 2 + 4}
      <text
        x={padL - 10}
        y={y}
        text-anchor="end"
        class="ax-label"
        class:active={isRowSelected(i)}
        style="cursor: pointer"
        onclick={() => pickRow(i)}
        role="button"
        tabindex={0}
      >{label}</text>
    {/each}

    <!-- Heatmap cells -->
    {#each Array(N) as _r, i}
      {#each Array(N) as _c, j}
        {@const x = padL + j * cellSize}
        {@const y = padT + i * cellSize}
        {@const p = joint[i * N + j]}
        <g class="cell"
          class:hilite={isRowSelected(i) || isColSelected(j)}
          style="touch-action: none"
          onpointerdown={(e) => onCellDown(e, i, j)}
          onpointerenter={(e) => onCellEnter(e, i, j)}
          onpointerup={(e) => onCellUp(e, i, j)}
          onpointercancel={(e) => onCellUp(e, i, j)}
          oncontextmenu={(e) => e.preventDefault()}
          role="gridcell"
        >
          <rect
            x={x + 1}
            y={y + 1}
            width={cellSize - 2}
            height={cellSize - 2}
            rx={4}
            ry={4}
            fill={cellColor(p)}
            stroke="color-mix(in srgb, var(--site-fg) 18%, transparent)"
            stroke-width="0.5"
          />
          <text
            x={x + cellSize / 2}
            y={y + cellSize / 2 + 4}
            text-anchor="middle"
            class="cell-num"
            class:dim={p < 0.02}
            class:light={p / maxCellProb > 0.55}
          >{fmt(p)}</text>
        </g>
      {/each}
    {/each}
  </svg>

  <div class="readout">
    {#if conditional}
      <div class="cond">
        <div class="cond-title">
          conditional on
          <strong class="strong">
            {conditional.kind === 'row' ? `Y = ${yLabels[conditional.idx]}` : `X = ${xLabels[conditional.idx]}`}
          </strong>
          —
          <span class="cond-desc">
            {conditional.kind === 'row' ? 'slice that row, renormalize to sum to 1' : 'slice that column, renormalize to sum to 1'}
          </span>
          <button type="button" class="btn ghost tiny" onclick={clearCondition}>clear</button>
        </div>
        <div class="cond-bars">
          {#each conditional.vals as v, k}
            {@const lbl = conditional.kind === 'row' ? xLabels[k] : yLabels[k]}
            <div class="cond-bar">
              <div class="cond-fill" style="height: {Math.round(v * 100)}%"></div>
              <div class="cond-num">{v.toFixed(2)}</div>
              <div class="cond-label">{lbl}</div>
            </div>
          {/each}
        </div>
      </div>
    {:else}
      <div class="hint">click a row label (yᵢ) or a column label (xⱼ) to take a conditional slice. drag on cells to repaint the joint. shift-drag to subtract.</div>
    {/if}
    <div class="actions">
      <button type="button" class="btn ghost" onclick={reset}>Reset joint</button>
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
  .marg-bg {
    fill: color-mix(in srgb, var(--site-fg) 5%, transparent);
  }
  .marg-bar.x { fill: var(--ink-sea); }
  .marg-bar.y { fill: var(--ink-coral); }
  .marg-bar.active { fill: var(--ink-sun); }
  .marg-num {
    font-family: var(--font-mono);
    font-size: 10px;
    fill: var(--site-fg-muted);
  }
  .marg-title {
    font-family: var(--font-mono);
    font-size: 10px;
    fill: var(--site-fg-muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .ax-label {
    font-family: var(--font-mono);
    font-size: 13px;
    fill: var(--site-fg-muted);
    font-weight: 600;
    user-select: none;
  }
  .ax-label.active { fill: var(--ink-sun); }
  .cell-num {
    font-family: var(--font-mono);
    font-size: 11px;
    fill: var(--site-fg);
    font-weight: 600;
    pointer-events: none;
  }
  .cell-num.dim { fill: var(--site-fg-muted); }
  .cell-num.light { fill: white; }
  .cell.hilite rect { stroke: var(--ink-sun); stroke-width: 2; }

  .readout { margin-top: 12px; display: flex; flex-direction: column; gap: 10px; }
  .hint {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--site-fg-muted);
  }
  .actions { display: flex; justify-content: flex-end; }
  .cond {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    background: var(--demo-stage);
    border-radius: 10px;
  }
  .cond-title {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--site-fg-muted);
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .strong { color: var(--ink-sun); }
  .cond-desc { font-style: italic; }
  .cond-bars {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    align-items: end;
    height: 110px;
  }
  .cond-bar {
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    justify-content: flex-end;
  }
  .cond-fill {
    width: 80%;
    background: var(--ink-sun);
    border-radius: 4px 4px 0 0;
    min-height: 2px;
    transition: height 160ms ease-out;
  }
  .cond-num {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--site-fg);
    margin-top: 3px;
  }
  .cond-label {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--site-fg-muted);
  }

  .btn {
    font-family: var(--font-mono);
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: var(--demo-stage);
    color: var(--site-fg);
    cursor: pointer;
  }
  .btn.tiny { padding: 2px 6px; font-size: 11px; }
  .btn.ghost { background: transparent; }
</style>
