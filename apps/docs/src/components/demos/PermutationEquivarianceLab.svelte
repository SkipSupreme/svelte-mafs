<script lang="ts">
  // 4 tokens, d_model = 4. Fixed embeddings — orthogonal-ish so attention has
  // visible structure rather than collapsing to uniform.
  const TOKEN_LABELS = ['a', 'b', 'c', 'd'];
  const X: number[][] = [
    [1.0, 0.0, 0.0, 0.6],
    [0.0, 1.0, 0.5, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.0, 0.6, 0.0, 1.0],
  ];
  const D = 4;
  const T = 4;
  const SQRT_D = Math.sqrt(D);

  // Sinusoidal-style PE, scaled down so token semantics still dominate.
  const PE_SCALE = 0.6;
  const PE: number[][] = Array.from({ length: T }, (_, pos) => {
    const v = new Array(D);
    for (let i = 0; i < D; i++) {
      const w = 1 / Math.pow(10000, (2 * Math.floor(i / 2)) / D);
      v[i] = (i % 2 === 0 ? Math.sin(pos * w) : Math.cos(pos * w)) * PE_SCALE;
    }
    return v;
  });

  let order = $state<number[]>([0, 1, 2, 3]);
  let usePE = $state(false);
  let selected = $state<number | null>(null);

  function reset() {
    order = [0, 1, 2, 3];
    selected = null;
  }
  function clickChip(slot: number) {
    if (selected === null) {
      selected = slot;
      return;
    }
    if (selected === slot) {
      selected = null;
      return;
    }
    const next = [...order];
    [next[selected], next[slot]] = [next[slot], next[selected]];
    order = next;
    selected = null;
  }

  function buildX(orderedIdx: number[], pe: boolean): number[][] {
    return orderedIdx.map((tokenIdx, slot) => {
      const tok = X[tokenIdx];
      if (!pe) return [...tok];
      const peVec = PE[slot];
      return tok.map((v, i) => v + peVec[i]);
    });
  }

  function selfAttention(x: number[][]): { weights: number[][]; out: number[][] } {
    // Q = K = V = x (no projections)
    const scores: number[][] = [];
    for (let i = 0; i < T; i++) {
      const row: number[] = [];
      for (let j = 0; j < T; j++) {
        let dot = 0;
        for (let k = 0; k < D; k++) dot += x[i][k] * x[j][k];
        row.push(dot / SQRT_D);
      }
      scores.push(row);
    }
    const weights = scores.map((row) => {
      const m = Math.max(...row);
      const exps = row.map((s) => Math.exp(s - m));
      const z = exps.reduce((a, b) => a + b, 0);
      return exps.map((e) => e / z);
    });
    const out: number[][] = [];
    for (let i = 0; i < T; i++) {
      const o = new Array(D).fill(0);
      for (let j = 0; j < T; j++) {
        for (let k = 0; k < D; k++) o[k] += weights[i][j] * x[j][k];
      }
      out.push(o);
    }
    return { weights, out };
  }

  const original = $derived(selfAttention(buildX([0, 1, 2, 3], usePE)));
  const current = $derived(selfAttention(buildX(order, usePE)));

  // Equivariance check: is current.out equal to (P · original.out) ?
  // Where P is the permutation that maps slot i to order[i].
  // i.e., row `i` of permuted_original = original.out[order[i]].
  const permutedOriginal = $derived(order.map((tokenIdx) => original.out[tokenIdx]));
  const delta = $derived.by(() => {
    let s = 0;
    for (let i = 0; i < T; i++) {
      for (let k = 0; k < D; k++) {
        const d = current.out[i][k] - permutedOriginal[i][k];
        s += d * d;
      }
    }
    return Math.sqrt(s);
  });

  const isIdentity = $derived(order.every((v, i) => v === i));
  const equivalent = $derived(delta < 1e-9);
</script>

<div class="widget">
  <div class="controls">
    <div class="chips" role="group" aria-label="Token order">
      {#each order as tokenIdx, slot}
        <button
          type="button"
          class="chip"
          class:selected={selected === slot}
          onclick={() => clickChip(slot)}
          aria-pressed={selected === slot}
        >
          <span class="chip-pos">pos {slot}</span>
          <span class="chip-tok">{TOKEN_LABELS[tokenIdx]}</span>
        </button>
      {/each}
    </div>

    <div class="actions">
      <span class="hint">
        {selected === null
          ? 'tap two chips to swap their tokens'
          : `tap another chip to swap with pos ${selected}`}
      </span>
      <button type="button" class="ghost" onclick={reset} disabled={isIdentity}>reset</button>
    </div>

    <div class="pe-row">
      <span class="label">positional encoding</span>
      <div class="seg" role="group">
        <button
          type="button"
          class="seg-btn"
          class:active={!usePE}
          onclick={() => (usePE = false)}>off</button>
        <button
          type="button"
          class="seg-btn"
          class:active={usePE}
          onclick={() => (usePE = true)}>on</button>
      </div>
    </div>
  </div>

  <div class="panes">
    <div class="pane">
      <h4>original ordering [a, b, c, d]</h4>
      <div class="mini-grid">
        {#each original.weights as row}
          <div class="mini-row">
            {#each row as w}
              <div class="mini-cell" style="--a:{w}"></div>
            {/each}
          </div>
        {/each}
      </div>
    </div>

    <div class="arrow" aria-hidden="true">→</div>

    <div class="pane">
      <h4>
        current ordering [{order.map((i) => TOKEN_LABELS[i]).join(', ')}]
      </h4>
      <div class="mini-grid">
        {#each current.weights as row}
          <div class="mini-row">
            {#each row as w}
              <div class="mini-cell" style="--a:{w}"></div>
            {/each}
          </div>
        {/each}
      </div>
    </div>
  </div>

  <div class="verdict" data-state={isIdentity ? 'identity' : equivalent ? 'ok' : 'broken'}>
    <span class="verdict-icon" aria-hidden="true">
      {isIdentity ? '·' : equivalent ? '✓' : '✗'}
    </span>
    <span class="verdict-text">
      {#if isIdentity}
        identity ordering — swap two chips to test
      {:else if equivalent}
        outputs match permutation: <em>Y′ = P · Y</em> (delta {delta.toExponential(1)})
      {:else}
        outputs do not match: <em>Y′ ≠ P · Y</em> (delta {delta.toFixed(3)})
      {/if}
    </span>
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

  .controls {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .chips {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem;
  }
  .chip {
    appearance: none;
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    background: color-mix(in srgb, var(--site-fg) 3%, transparent);
    border-radius: 12px;
    padding: 0.7rem 0.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
    cursor: pointer;
    transition:
      transform 160ms ease,
      background 160ms ease,
      border-color 160ms ease;
  }
  .chip:hover {
    background: color-mix(in srgb, var(--site-fg) 6%, transparent);
  }
  .chip.selected {
    background: color-mix(in srgb, var(--ink-sun) 18%, transparent);
    border-color: var(--ink-sun);
    transform: translateY(-2px);
  }
  .chip-pos {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .chip-tok {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 700;
    font-size: 1.4rem;
    color: var(--site-fg);
    line-height: 1;
  }

  .actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .hint {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--site-fg-muted);
  }
  .ghost {
    appearance: none;
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    background: transparent;
    color: var(--site-fg-muted);
    padding: 0.35rem 0.7rem;
    border-radius: 10px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    cursor: pointer;
  }
  .ghost:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  .ghost:not(:disabled):hover {
    background: color-mix(in srgb, var(--site-fg) 5%, transparent);
  }

  .pe-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
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

  .panes {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 0.8rem;
    align-items: center;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
  .pane h4 {
    margin: 0 0 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--site-fg-muted);
    text-align: center;
    font-weight: 700;
  }
  .arrow {
    font-family: var(--font-display);
    font-size: 1.5rem;
    color: var(--site-fg-muted);
    text-align: center;
  }
  .mini-grid {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .mini-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 2px;
  }
  .mini-cell {
    aspect-ratio: 1;
    border-radius: 4px;
    background: color-mix(in srgb, var(--ink-sea) calc(var(--a) * 100%), transparent);
    border: 1px solid color-mix(in srgb, var(--site-fg) 8%, transparent);
    transition: background 220ms ease;
  }

  .verdict {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.65rem 0.85rem;
    border-radius: 12px;
    border-left: 4px solid var(--site-fg-muted);
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--site-fg);
  }
  .verdict-icon {
    font-size: 1.2rem;
    font-weight: 700;
    width: 1.4rem;
    text-align: center;
  }
  .verdict em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }
  .verdict[data-state='ok'] {
    border-left-color: var(--cta-hover);
    background: color-mix(in srgb, var(--cta-hover) 10%, transparent);
  }
  .verdict[data-state='ok'] .verdict-icon {
    color: var(--cta-hover);
  }
  .verdict[data-state='broken'] {
    border-left-color: var(--ink-coral);
    background: color-mix(in srgb, var(--ink-coral) 12%, transparent);
  }
  .verdict[data-state='broken'] .verdict-icon {
    color: var(--ink-coral);
  }
</style>
