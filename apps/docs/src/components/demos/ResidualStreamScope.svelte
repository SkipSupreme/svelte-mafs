<script lang="ts">
  // Tiny "model" with N=4 pre-LN blocks, T=5 tokens, d=16. All activations are
  // scripted deterministically so the same shape of stream is shown to every
  // learner. The pedagogical claim — "every block writes a delta into the
  // stream, the stream is the noun" — survives the absence of a real model.

  const T = 5;
  const D = 16;
  const N = 4;
  const TOKENS = ['h', 'e', 'l', 'l', 'o'];

  function mulberry32(a: number) {
    return () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng: () => number) {
    const u = 1 - rng();
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Build the embedding (h_0) and per-block deltas. Each block's delta is
  // sparse on a different subset of dims so the user, clicking around, sees
  // visibly different per-block contributions.
  function buildState() {
    const rng = mulberry32(11);
    const h0: number[][] = Array.from({ length: T }, () =>
      Array.from({ length: D }, () => gauss(rng) * 0.6)
    );

    // Each block has an "active" subset of dims it tends to write to.
    const blockSubsets = [
      [0, 1, 2, 3, 4, 5, 6, 7], // block 1: left half
      [8, 9, 10, 11, 12, 13, 14, 15], // block 2: right half
      [0, 4, 8, 12], // block 3: every 4th
      [2, 6, 10, 14], // block 4: every 4th, offset
    ];

    const attnDeltas: number[][][] = [];
    const ffnDeltas: number[][][] = [];
    for (let l = 0; l < N; l++) {
      const subset = blockSubsets[l];
      const aDelta: number[][] = Array.from({ length: T }, () => new Array(D).fill(0));
      const fDelta: number[][] = Array.from({ length: T }, () => new Array(D).fill(0));
      for (let pos = 0; pos < T; pos++) {
        // Attention delta: per-position, sparse on the subset
        for (const dim of subset) {
          aDelta[pos][dim] = gauss(rng) * 0.4;
        }
        // FFN delta: also sparse on the subset, but distinct
        for (const dim of subset) {
          fDelta[pos][dim] = gauss(rng) * 0.5;
        }
      }
      // Add an extra "interesting" delta to position 4 (the last token, "o")
      // in block 3 to mimic late-layer output specialization.
      if (l === 2) {
        for (let dim = 0; dim < D; dim++) {
          fDelta[4][dim] += gauss(rng) * 0.7;
        }
      }
      attnDeltas.push(aDelta);
      ffnDeltas.push(fDelta);
    }

    // Assemble the stream snapshots: h_0, after-attn-1, after-ffn-1, ..., after-ffn-N.
    const streams: number[][][] = [];
    streams.push(h0.map((row) => [...row]));
    let cur = h0.map((row) => [...row]);
    for (let l = 0; l < N; l++) {
      const afterAttn = cur.map((row, pos) =>
        row.map((v, dim) => v + attnDeltas[l][pos][dim])
      );
      streams.push(afterAttn.map((row) => [...row]));
      const afterFfn = afterAttn.map((row, pos) =>
        row.map((v, dim) => v + ffnDeltas[l][pos][dim])
      );
      streams.push(afterFfn.map((row) => [...row]));
      cur = afterFfn;
    }

    return { streams, attnDeltas, ffnDeltas };
  }

  const state = buildState();
  const streams = state.streams; // length 2N+1 = 9
  const STATE_LABELS: { name: string; kind: 'init' | 'attn' | 'ffn'; block?: number }[] = [
    { name: 'h₀ (embed)', kind: 'init' },
    ...Array.from({ length: N }, (_, l) => [
      { name: `after attn ${l + 1}`, kind: 'attn' as const, block: l + 1 },
      { name: `after ffn ${l + 1}`, kind: 'ffn' as const, block: l + 1 },
    ]).flat(),
  ];

  let selectedRow = $state(STATE_LABELS.length - 1); // top of stack
  let selectedCol = $state(0);

  const selectedVector = $derived(streams[selectedRow][selectedCol]);
  const vecScale = $derived(
    Math.max(0.01, ...selectedVector.map((v) => Math.abs(v)))
  );
  const cellMagnitudes = $derived.by(() => {
    const mags: number[][] = [];
    for (let row = 0; row < streams.length; row++) {
      const rowMags: number[] = [];
      for (let col = 0; col < T; col++) {
        let m = 0;
        for (let dim = 0; dim < D; dim++) m += streams[row][col][dim] ** 2;
        rowMags.push(Math.sqrt(m));
      }
      mags.push(rowMags);
    }
    return mags;
  });
  const cellMax = $derived(Math.max(0.01, ...cellMagnitudes.flat()));

  function selectCell(row: number, col: number) {
    selectedRow = row;
    selectedCol = col;
  }
</script>

<div class="widget">
  <div class="legend-row">
    <span class="title">residual stream — click any cell</span>
    <span class="hint">rows: states inside the stack · columns: token positions</span>
  </div>

  <div class="grid">
    <div class="empty"></div>
    {#each TOKENS as tok, col}
      <div class="col-label">
        <span class="col-pos">pos {col}</span>
        <span class="col-tok">"{tok}"</span>
      </div>
    {/each}

    {#each STATE_LABELS as label, row}
      <div class="row-label" data-kind={label.kind}>{label.name}</div>
      {#each TOKENS as _, col}
        {@const mag = cellMagnitudes[row][col]}
        <button
          type="button"
          class="cell"
          class:selected={selectedRow === row && selectedCol === col}
          style="--mag:{mag / cellMax}"
          onclick={() => selectCell(row, col)}
          aria-label={`${label.name} at position ${col}, magnitude ${mag.toFixed(2)}`}
        ></button>
      {/each}
    {/each}
  </div>

  <div class="vector-pane">
    <div class="vector-header">
      <span class="vector-title">
        <em>x</em><sub>{selectedRow === 0 ? '0' : `${Math.ceil(selectedRow / 2)}${selectedRow % 2 === 1 ? 'a' : 'b'}`}</sub>
        at position <strong>{selectedCol}</strong> ("<em>{TOKENS[selectedCol]}</em>")
      </span>
      <span class="vector-state">{STATE_LABELS[selectedRow].name}</span>
    </div>
    <div class="vector-bar">
      {#each selectedVector as v, dim}
        <div
          class="vec-cell"
          class:pos={v >= 0}
          class:neg={v < 0}
          style="--w:{Math.abs(v) / vecScale}"
          title="dim {dim}: {v.toFixed(3)}"
        ></div>
      {/each}
    </div>
    <div class="vector-axis">
      <span>dim 0</span>
      <span>dim {D - 1}</span>
    </div>
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

  .legend-row {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .title {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
  }
  .hint {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--site-fg-muted);
  }

  .grid {
    display: grid;
    grid-template-columns: 9rem repeat(5, 1fr);
    gap: 0.25rem;
    align-items: center;
  }
  .empty {
    /* corner */
  }
  .col-label {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.1rem;
  }
  .col-pos {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--site-fg-muted);
  }
  .col-tok {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 700;
    color: var(--site-fg);
  }
  .row-label {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
    text-align: right;
    padding-right: 0.5rem;
    border-right: 2px solid var(--site-fg-muted);
  }
  .row-label[data-kind='init'] {
    color: var(--ink-orange);
    border-right-color: var(--ink-orange);
    font-weight: 700;
  }
  .row-label[data-kind='attn'] {
    border-right-color: var(--ink-sea);
  }
  .row-label[data-kind='ffn'] {
    border-right-color: var(--ink-coral);
  }

  .cell {
    appearance: none;
    border: 1px solid color-mix(in srgb, var(--site-fg) 8%, transparent);
    background: color-mix(in srgb, var(--ink-sea) calc(var(--mag) * 100%), transparent);
    border-radius: 6px;
    height: 30px;
    cursor: pointer;
    transition: background 220ms ease, transform 160ms ease, box-shadow 160ms ease;
  }
  .cell:hover {
    transform: translateY(-1px);
  }
  .cell.selected {
    box-shadow: 0 0 0 2px var(--ink-sun);
    border-color: var(--ink-sun);
  }

  .vector-pane {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding: 0.7rem 0.85rem;
    border-radius: 12px;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-left: 3px solid var(--ink-sun);
  }
  .vector-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.5rem;
  }
  .vector-title {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--site-fg);
  }
  .vector-title em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }
  .vector-title strong {
    color: var(--ink-sun);
    font-weight: 700;
  }
  .vector-state {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }

  .vector-bar {
    display: grid;
    grid-template-columns: repeat(16, 1fr);
    gap: 2px;
    height: 36px;
    align-items: center;
  }
  .vec-cell {
    height: 100%;
    border-radius: 4px;
    border: 1px solid color-mix(in srgb, var(--site-fg) 6%, transparent);
    transition: background 220ms ease;
  }
  .vec-cell.pos {
    background: color-mix(in srgb, var(--ink-sea) calc(var(--w) * 100%), transparent);
  }
  .vec-cell.neg {
    background: color-mix(in srgb, var(--ink-coral) calc(var(--w) * 100%), transparent);
  }
  .vector-axis {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--site-fg-muted);
  }
</style>
