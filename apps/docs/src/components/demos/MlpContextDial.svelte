<script lang="ts">
  interface Props {
    /** Initial context window length. */
    initialK?: number;
    /** Embedding dimension. */
    d?: number;
    /** Hidden layer size. */
    hidden?: number;
    /** Vocabulary size. */
    V?: number;
  }

  let {
    initialK = 3,
    d = 10,
    hidden = 200,
    V = 27,
  }: Props = $props();

  const VOCAB: string[] = ['.', ...'abcdefghijklmnopqrstuvwxyz'.split('')];
  const display = (c: string) => (c === '.' ? '·' : c);
  const MAX_K = 8;

  let k: number = $state(initialK);
  // Initial context — last `k` chars of "olivia." (so the slots have meaningful starting state).
  const seedContext: string[] = ['o', 'l', 'i', 'v', 'i', 'a', '·', 'e'];
  let context: string[] = $state(seedContext.slice(0, MAX_K));
  let editingSlot: number | null = $state(null);

  const visibleContext = $derived(context.slice(MAX_K - k));

  // Parameter counts (Bengio-style: C, W₁ that consumes k·d concat'd embeddings,
  // hidden bias, output W₂ + bias). We omit the direct skip-connection from
  // the original paper for clarity.
  const paramsC = $derived(V * d);
  const paramsW1 = $derived(hidden * (k * d));
  const paramsB1 = $derived(hidden);
  const paramsW2 = $derived(V * hidden);
  const paramsB2 = $derived(V);
  const totalParams = $derived(paramsC + paramsW1 + paramsB1 + paramsW2 + paramsB2);

  // For comparison: a pure n-gram count table for window k would have V^k cells.
  const ngramCells = $derived(Math.pow(V, k + 1)); // (k context tokens) → predicting the (k+1)th
  const ngramReadable = $derived.by(() => {
    const n = ngramCells;
    if (n < 1e6) return n.toLocaleString();
    if (n < 1e9) return (n / 1e6).toFixed(1) + 'M';
    if (n < 1e12) return (n / 1e9).toFixed(1) + 'B';
    return (n / 1e12).toFixed(1) + 'T';
  });

  function setSlot(slotIdx: number, ch: string): void {
    const absoluteIdx = MAX_K - k + slotIdx;
    const next = [...context];
    next[absoluteIdx] = ch;
    context = next;
    editingSlot = null;
  }

  function cycleK(delta: number): void {
    const nk = Math.max(1, Math.min(MAX_K, k + delta));
    k = nk;
  }

  // Color the param-count meter from green (cheap) → coral (expensive)
  // on a log scale across the visible range.
  const paramT = $derived.by(() => {
    const minP = V * d + hidden * (1 * d) + hidden + V * hidden + V; // k=1
    const maxP = V * d + hidden * (MAX_K * d) + hidden + V * hidden + V; // k=8
    return (totalParams - minP) / (maxP - minP);
  });
  const paramColor = $derived(
    `color-mix(in srgb, var(--ink-coral) ${Math.round(paramT * 100)}%, var(--cta) ${Math.round((1 - paramT) * 100)}%)`,
  );
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">context size k</span>
      <span class="meta-val">{k}</span>
    </div>
    <div class="meta">
      <span class="meta-key">embedding dim d</span>
      <span class="meta-val">{d}</span>
    </div>
    <div class="meta">
      <span class="meta-key">hidden h</span>
      <span class="meta-val">{hidden}</span>
    </div>
    <div class="meta meta-total">
      <span class="meta-key">total params</span>
      <span class="meta-val" style="color:{paramColor};">
        {totalParams.toLocaleString()}
      </span>
    </div>
  </header>

  <div class="diagram">
    <!-- Context slots -->
    <div class="layer layer-tokens">
      <div class="layer-label">context</div>
      <div class="slots">
        {#each visibleContext as ch, i (i)}
          <div class="slot-wrap">
            <button
              type="button"
              class="slot"
              onclick={() => (editingSlot = editingSlot === i ? null : i)}
              aria-label={`Slot ${i + 1}: ${display(ch)}`}
              aria-expanded={editingSlot === i}
            >{display(ch)}</button>
            {#if editingSlot === i}
              <div class="slot-pop" role="listbox">
                {#each VOCAB as v}
                  <button
                    type="button"
                    class="slot-pop-btn"
                    class:active={v === ch}
                    onclick={() => setSlot(i, v)}
                  >{display(v)}</button>
                {/each}
              </div>
            {/if}
          </div>
        {/each}
        {#if k < MAX_K}
          <button type="button" class="slot slot-add" onclick={() => cycleK(1)} aria-label="Add a context slot">+</button>
        {/if}
        {#if k > 1}
          <button type="button" class="slot slot-remove" onclick={() => cycleK(-1)} aria-label="Remove a context slot">−</button>
        {/if}
      </div>
    </div>

    <!-- Embeddings (one per slot) -->
    <div class="layer layer-emb">
      <div class="layer-label">embeddings · C[token]</div>
      <div class="emb-row">
        {#each visibleContext as ch, i (i)}
          <div class="emb-block" title={`embedding of ${display(ch)} (d = ${d})`}>
            {#each Array(d) as _, b}
              <div class="emb-bar" style="height:{(40 + 28 * Math.sin((b + i * 3) * 0.7))}%"></div>
            {/each}
          </div>
        {/each}
      </div>
    </div>

    <!-- Concat → Hidden -->
    <div class="layer layer-hidden">
      <div class="layer-label">
        concat → W₁ · x + b₁ → tanh
        <span class="layer-shape">(W₁ shape: <strong>{hidden} × {k * d}</strong> = {paramsW1.toLocaleString()})</span>
      </div>
      <div class="hidden-bar">
        <div class="hidden-fill" style="width:{Math.min(100, paramT * 100 + 12)}%;"></div>
        <span class="hidden-label">hidden ({hidden})</span>
      </div>
    </div>

    <!-- Output softmax -->
    <div class="layer layer-out">
      <div class="layer-label">W₂ · h + b₂ → softmax</div>
      <div class="out-row">
        {#each Array(V) as _, j}
          <div
            class="out-bar"
            style="height:{(20 + 60 * Math.exp(-Math.pow(j - 13, 2) / 60))}%"
            title={`P(${display(VOCAB[j])} | context)`}
          ></div>
        {/each}
      </div>
      <div class="out-axis">
        <span>·</span><span>m</span><span>z</span>
      </div>
    </div>
  </div>

  <div class="param-breakdown">
    <div class="pb-row"><span>C (embedding lookup)</span><strong>V × d = {V}×{d} = {paramsC.toLocaleString()}</strong></div>
    <div class="pb-row pb-grow"><span>W₁ (hidden weights)</span><strong>h × (k·d) = {hidden}×{k * d} = {paramsW1.toLocaleString()}</strong></div>
    <div class="pb-row"><span>W₂ (output projection)</span><strong>V × h = {V}×{hidden} = {paramsW2.toLocaleString()}</strong></div>
    <div class="pb-row pb-total">
      <span>total</span><strong>{totalParams.toLocaleString()}</strong>
    </div>
  </div>

  <div class="callout">
    <div class="callout-key">vs n-gram count table</div>
    <div class="callout-val">
      A pure (k+1)-gram table would need <strong>{V}<sup>{k + 1}</sup></strong>
      cells = <strong>{ngramReadable}</strong> entries. The MLP gets the same
      context window for far less storage — and generalizes.
    </div>
  </div>

  <p class="caption">
    Pull the context window from k = 1 to k = 8 and watch W₁'s parameter
    count grow linearly. Click any slot to change its character. The pure
    n-gram table for the same window grows <em>exponentially</em>, blowing
    past memory by k = 5; the MLP stays manageable because all the heavy
    lifting fits in one weight matrix that scales with k, not k·V.
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 50%, transparent);
  }
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .meta-total .meta-val { font-size: 1rem; }

  .diagram {
    background: var(--demo-stage);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
  }
  .layer {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .layer-label {
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--site-fg-muted);
    display: flex; flex-wrap: wrap; gap: 0.35rem 0.6rem; align-items: baseline;
  }
  .layer-shape {
    color: var(--site-fg);
    font-size: 0.7rem;
  }
  .layer-shape strong { color: var(--ink-red); font-weight: 600; }

  .slots {
    display: flex; gap: 0.35rem; flex-wrap: wrap; align-items: center;
  }
  .slot-wrap { position: relative; }
  .slot {
    width: 36px; height: 36px;
    display: inline-flex; align-items: center; justify-content: center;
    border: 1.5px solid var(--ink-red);
    background: color-mix(in srgb, var(--ink-red) 8%, var(--demo-stage));
    color: var(--ink-red);
    border-radius: 8px;
    font-family: var(--font-mono);
    font-size: 1rem; font-weight: 700;
    cursor: pointer;
    transition: transform 120ms ease, background 160ms ease;
  }
  .slot:hover { transform: translateY(-1px); background: color-mix(in srgb, var(--ink-red) 16%, var(--demo-stage)); }
  .slot-add, .slot-remove {
    border-style: dashed;
    border-color: color-mix(in srgb, var(--ink-red) 50%, transparent);
    color: color-mix(in srgb, var(--ink-red) 80%, transparent);
    background: transparent;
  }
  .slot-pop {
    position: absolute; top: 100%; left: 0; margin-top: 4px; z-index: 5;
    display: grid; grid-template-columns: repeat(7, 1fr);
    gap: 2px; padding: 4px;
    background: var(--demo-card);
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    border-radius: 8px;
    width: 200px;
    box-shadow: 0 8px 24px -8px rgba(0,0,0,0.3);
  }
  .slot-pop-btn {
    width: 24px; height: 24px;
    display: inline-flex; align-items: center; justify-content: center;
    border: none;
    background: transparent;
    color: var(--site-fg);
    border-radius: 4px;
    font-family: var(--font-mono); font-size: 0.85rem;
    cursor: pointer;
  }
  .slot-pop-btn:hover { background: color-mix(in srgb, var(--ink-red) 16%, transparent); }
  .slot-pop-btn.active { background: var(--ink-red); color: #fdfdfc; }

  .emb-row { display: flex; gap: 0.35rem; align-items: flex-end; }
  .emb-block {
    display: flex;
    align-items: flex-end;
    gap: 1px;
    width: 36px; height: 32px;
    background: color-mix(in srgb, var(--ink-sea) 6%, transparent);
    border-radius: 4px;
    padding: 2px;
  }
  .emb-bar {
    flex: 1 1 0;
    background: var(--ink-sea);
    border-radius: 1px;
    min-height: 2px;
  }

  .hidden-bar {
    position: relative;
    width: 100%; height: 22px;
    background: color-mix(in srgb, var(--site-fg) 6%, transparent);
    border-radius: 6px;
    overflow: hidden;
  }
  .hidden-fill {
    height: 100%;
    background: linear-gradient(90deg,
      color-mix(in srgb, var(--cta) 80%, transparent),
      color-mix(in srgb, var(--ink-coral) 80%, transparent)
    );
    transition: width 240ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .hidden-label {
    position: absolute; left: 8px; top: 50%; transform: translateY(-50%);
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--site-fg);
    text-shadow: 0 0 4px var(--demo-stage);
  }

  .out-row {
    display: flex; align-items: flex-end; gap: 1px;
    height: 50px;
  }
  .out-bar {
    flex: 1 1 0;
    background: var(--ink-red);
    border-radius: 1px 1px 0 0;
    min-height: 2px;
  }
  .out-axis {
    display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: 0.68rem; color: var(--site-fg-muted);
  }

  .param-breakdown {
    display: flex; flex-direction: column; gap: 0.2rem;
    padding: 0.55rem 0.75rem;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-radius: 8px;
  }
  .pb-row {
    display: flex; justify-content: space-between; align-items: baseline;
    font-family: var(--font-mono); font-size: 0.82rem;
    color: var(--site-fg-muted);
  }
  .pb-row strong { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .pb-row.pb-grow strong { color: var(--ink-red); }
  .pb-row.pb-total {
    padding-top: 0.25rem;
    border-top: 1px dashed color-mix(in srgb, var(--site-fg) 22%, transparent);
    color: var(--site-fg);
    font-weight: 600;
  }

  .callout {
    display: flex; flex-direction: column; gap: 0.2rem;
    padding: 0.55rem 0.75rem;
    background: color-mix(in srgb, var(--ink-coral) 8%, transparent);
    border-left: 3px solid var(--ink-coral);
    border-radius: 0 8px 8px 0;
  }
  .callout-key {
    font-family: var(--font-mono); font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--ink-coral); font-weight: 600;
  }
  .callout-val { font-size: 0.88rem; line-height: 1.5; }
  .callout-val strong {
    font-variant-numeric: tabular-nums; color: var(--ink-coral); font-weight: 600;
  }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg); font-style: normal;
    font-family: var(--font-mono); font-size: 0.85em;
  }

  @media (prefers-reduced-motion: reduce) { .hidden-fill { transition: none; } }
</style>
