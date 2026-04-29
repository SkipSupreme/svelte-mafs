<script lang="ts">
  const T = 6;
  const TOKENS = ['t₁', 't₂', 't₃', 't₄', 't₅', 't₆'];

  let mask = $state(false);
  let hoveredRow = $state<number | null>(null);

  // Deterministic pre-softmax scores. Hand-tuned so the unmasked pattern shows
  // some interesting structure (not uniform, not one-hot) — including a
  // long-range "coreference" hit at (5, 1) for an "it → animal" feel.
  const RAW: number[][] = [
    [1.4, -0.3, 0.1, 0.0, -0.5, 0.2],
    [0.6, 1.2, 0.4, -0.1, 0.3, -0.2],
    [0.2, 0.7, 1.1, 0.3, -0.4, 0.1],
    [-0.1, 0.5, 0.8, 1.3, 0.4, 0.0],
    [0.3, -0.2, 0.1, 0.6, 1.0, 0.5],
    [1.7, 0.3, -0.1, 0.4, 0.2, 1.0], // q₆ remembers q₁
  ];

  function softmaxRow(row: number[], maskFromCol: number | null): number[] {
    const masked = row.map((s, j) =>
      maskFromCol !== null && j > maskFromCol ? -Infinity : s
    );
    const m = Math.max(...masked.filter((v) => v > -Infinity));
    const exps = masked.map((s) => (s === -Infinity ? 0 : Math.exp(s - m)));
    const z = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / z);
  }

  const matrix = $derived(
    RAW.map((row, i) => softmaxRow(row, mask ? i : null))
  );
  const rowSums = $derived(matrix.map((row) => row.reduce((a, b) => a + b, 0)));
</script>

<div class="widget">
  <div class="controls">
    <span class="label">causal mask</span>
    <div class="seg" role="group">
      <button
        type="button"
        class="seg-btn"
        class:active={!mask}
        onclick={() => (mask = false)}>off</button>
      <button
        type="button"
        class="seg-btn"
        class:active={mask}
        onclick={() => (mask = true)}>on</button>
    </div>
    <span class="hint">
      {mask
        ? 'upper triangle masked → each row attends only to past + self'
        : 'every query sees every key (encoder-style)'}
    </span>
  </div>

  <div class="grid-wrap">
    <div class="corner"></div>
    <div class="col-labels">
      {#each TOKENS as t, j}
        <div class="lbl col">{t}<span class="muted"> (k)</span></div>
      {/each}
    </div>
    <div class="rowsum-label">Σ</div>

    {#each matrix as row, i}
      <div class="lbl row">{TOKENS[i]}<span class="muted"> (q)</span></div>
      <div
        class="row-cells"
        onmouseenter={() => (hoveredRow = i)}
        onmouseleave={() => (hoveredRow = null)}
        role="row"
      >
        {#each row as alpha, j}
          {@const dim = mask && j > i}
          <div
            class="cell"
            class:dim
            class:hi={hoveredRow === i}
            style="--a:{alpha}"
            title="α({TOKENS[i]} → {TOKENS[j]}) = {alpha.toFixed(3)}"
          >
            <span class="cell-num">{alpha < 0.005 ? '0' : alpha.toFixed(2)}</span>
          </div>
        {/each}
      </div>
      <div class="rowsum">{rowSums[i].toFixed(2)}</div>
    {/each}
  </div>

  <p class="legend">
    rows: queries · columns: keys · cell color = α<sub>ij</sub> ∈ [0, 1] · every row sums to 1
  </p>
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

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.65rem;
  }
  .label {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
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
  }
  .seg-btn + .seg-btn {
    border-left: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
  }
  .seg-btn.active {
    background: color-mix(in srgb, var(--ink-sea) 18%, transparent);
    color: var(--site-fg);
  }
  .hint {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--site-fg-muted);
    flex: 1 1 200px;
  }

  .grid-wrap {
    display: grid;
    grid-template-columns: 3.4rem 1fr 2.2rem;
    gap: 0.4rem;
    align-items: center;
  }

  .corner {
    grid-column: 1;
  }
  .col-labels {
    grid-column: 2;
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.25rem;
  }
  .col-labels .lbl {
    text-align: center;
  }
  .rowsum-label {
    grid-column: 3;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--site-fg-muted);
  }

  .lbl {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--site-fg);
    line-height: 1;
  }
  .lbl .muted {
    font-style: normal;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--site-fg-muted);
    margin-left: 0.15rem;
  }
  .lbl.row {
    text-align: right;
    padding-right: 0.4rem;
  }

  .row-cells {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.25rem;
  }
  .cell {
    aspect-ratio: 1;
    border-radius: 6px;
    background: color-mix(in srgb, var(--ink-sea) calc(var(--a) * 100%), transparent);
    display: grid;
    place-items: center;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
    color: color-mix(in srgb, var(--site-fg) calc(50% + var(--a) * 50%), transparent);
    border: 1px solid color-mix(in srgb, var(--site-fg) 6%, transparent);
    transition: background 220ms ease, color 220ms ease, transform 160ms ease;
  }
  .cell.dim {
    background: transparent;
    color: color-mix(in srgb, var(--site-fg) 25%, transparent);
    border-style: dashed;
  }
  .cell.hi {
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--ink-sun) 60%, transparent);
  }
  .cell-num {
    pointer-events: none;
  }

  .rowsum {
    text-align: center;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
    color: var(--site-fg);
    font-weight: 600;
    padding: 0.25rem;
    border-radius: 6px;
    background: color-mix(in srgb, var(--ink-sun) 12%, transparent);
  }

  .legend {
    margin: 0;
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
    padding-top: 0.4rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
</style>
