<script lang="ts">
  interface Props {
    /** Initial corpus of words. Empty array means "the user types to fill it." */
    corpus?: string[];
    /** Allow the learner to add words via the input box. */
    editable?: boolean;
    /** Show a bar-chart of P(·|i) for the focused row. */
    showDistribution?: boolean;
    /** Letter to focus on load (e.g. "a"). "." is the start/end token. */
    focusOnLoad?: string | null;
    /** Hint copy under the heatmap. */
    hint?: string;
  }

  let {
    corpus = [],
    editable = true,
    showDistribution = true,
    focusOnLoad = null,
    hint = 'click a row label on the left to inspect P(next | row).',
  }: Props = $props();

  // Vocabulary: index 0 = '.' (start/end), 1..26 = a..z.
  const VOCAB: string[] = ['.', ...'abcdefghijklmnopqrstuvwxyz'.split('')];
  const V = VOCAB.length; // 27
  const INDEX: Record<string, number> = Object.fromEntries(
    VOCAB.map((c, i) => [c, i]),
  );

  let words: string[] = $state([...corpus]);
  let focused: number | null = $state(
    focusOnLoad != null ? (INDEX[focusOnLoad] ?? null) : null,
  );
  let hovered: { i: number; j: number } | null = $state(null);
  let typed: string = $state('');
  let lastError: string = $state('');

  const counts: number[][] = $derived.by(() => {
    const N: number[][] = Array.from({ length: V }, () => Array(V).fill(0));
    for (const w of words) {
      const cleaned = w.toLowerCase().replace(/[^a-z]/g, '');
      if (!cleaned) continue;
      const seq = '.' + cleaned + '.';
      for (let t = 0; t < seq.length - 1; t++) {
        const i = INDEX[seq[t]];
        const j = INDEX[seq[t + 1]];
        if (i != null && j != null) N[i][j] += 1;
      }
    }
    return N;
  });

  const rowSums: number[] = $derived(counts.map((r) => r.reduce((a, b) => a + b, 0)));
  const totalCount: number = $derived(rowSums.reduce((a, b) => a + b, 0));
  const maxCount: number = $derived.by(() => {
    let m = 0;
    for (const row of counts) for (const v of row) if (v > m) m = v;
    return m;
  });

  function colorFor(value: number): string {
    if (value === 0 || maxCount === 0) return 'var(--cell-empty)';
    // Sequential lightness ramp from cream → ink-red.
    const t = Math.sqrt(value / maxCount); // sqrt to lift small values visually
    // mix from cream to red via the design system color-mix.
    return `color-mix(in srgb, var(--ink-red) ${Math.round(t * 100)}%, var(--cell-empty))`;
  }

  function textColorFor(value: number): string {
    if (value === 0 || maxCount === 0) return 'transparent';
    return value / maxCount > 0.45 ? 'var(--cream-on-red)' : 'var(--site-fg)';
  }

  function pickRow(i: number) {
    focused = focused === i ? null : i;
  }

  function addWord() {
    const cleaned = typed.toLowerCase().replace(/[^a-z]/g, '');
    if (!cleaned) {
      lastError = 'Use letters a–z only.';
      return;
    }
    if (cleaned.length > 24) {
      lastError = 'That is too long for a name.';
      return;
    }
    words = [...words, cleaned];
    typed = '';
    lastError = '';
  }

  function removeWord(idx: number) {
    words = words.filter((_, i) => i !== idx);
  }

  function reset() {
    words = [...corpus];
    typed = '';
    lastError = '';
    focused = null;
    hovered = null;
  }

  // Distribution bar chart for the focused row.
  const distribution: { char: string; count: number; prob: number }[] = $derived.by(() => {
    if (focused == null) return [];
    const sum = rowSums[focused];
    return counts[focused].map((c, j) => ({
      char: VOCAB[j],
      count: c,
      prob: sum > 0 ? c / sum : 0,
    }));
  });

  const distMax: number = $derived.by(() => {
    let m = 0;
    for (const d of distribution) if (d.prob > m) m = d.prob;
    return m;
  });

  // Inspector — show details for the hovered or focused-row hovered cell.
  const inspectedCell = $derived(
    hovered ?? (focused != null ? null : null),
  );
  const inspector = $derived.by(() => {
    if (hovered == null) return null;
    const { i, j } = hovered;
    const sum = rowSums[i];
    return {
      from: VOCAB[i],
      to: VOCAB[j],
      n: counts[i][j],
      p: sum > 0 ? counts[i][j] / sum : 0,
      sum,
    };
  });

  const display = (c: string) => (c === '.' ? '·' : c);
  const fmtProb = (p: number) =>
    p === 0 ? '0' : p < 0.001 ? p.toExponential(1) : p.toFixed(3);
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">corpus</span>
      <span class="meta-val">{words.length} word{words.length === 1 ? '' : 's'}</span>
    </div>
    <div class="meta">
      <span class="meta-key">bigrams counted</span>
      <span class="meta-val">{totalCount}</span>
    </div>
    {#if focused != null}
      <div class="meta meta-focus">
        <span class="meta-key">row</span>
        <span class="meta-val">{display(VOCAB[focused])}</span>
        <button
          type="button"
          class="meta-clear"
          onclick={() => (focused = null)}
          aria-label="Clear row focus"
        >×</button>
      </div>
    {/if}
  </header>

  <div class="grid-wrap" role="figure" aria-label="Bigram count matrix; rows are first character, columns are second character.">
    <!-- column header: second character -->
    <div class="col-headers" aria-hidden="true">
      <div class="corner" />
      {#each VOCAB as c, j}
        <div
          class="col-label"
          class:active={focused != null && counts[focused][j] > 0}
        >{display(c)}</div>
      {/each}
    </div>

    {#each VOCAB as rowChar, i}
      <div class="row" class:row-focused={focused === i}>
        <button
          type="button"
          class="row-label"
          class:row-label-active={focused === i}
          class:row-label-empty={rowSums[i] === 0}
          onclick={() => pickRow(i)}
          aria-pressed={focused === i}
          aria-label={`Row ${display(rowChar)}; ${rowSums[i]} bigrams.`}
        >{display(rowChar)}</button>
        {#each VOCAB as _colChar, j}
          <button
            type="button"
            class="cell"
            class:cell-zero={counts[i][j] === 0}
            class:cell-hover={hovered?.i === i && hovered?.j === j}
            style="background:{colorFor(counts[i][j])};color:{textColorFor(counts[i][j])};"
            onmouseenter={() => (hovered = { i, j })}
            onmouseleave={() => (hovered = null)}
            onfocus={() => (hovered = { i, j })}
            onblur={() => (hovered = null)}
            tabindex={counts[i][j] > 0 ? 0 : -1}
            aria-label={`Bigram ${display(rowChar)} then ${display(VOCAB[j])}, count ${counts[i][j]}.`}
          >{counts[i][j] || ''}</button>
        {/each}
      </div>
    {/each}
  </div>

  <div class="inspector" aria-live="polite">
    {#if inspector}
      <span class="ins-pair">
        <code>{display(inspector.from)}</code>
        <span class="arrow">→</span>
        <code>{display(inspector.to)}</code>
      </span>
      <span class="ins-stat">
        N = <strong>{inspector.n}</strong>
      </span>
      <span class="ins-stat">
        P({display(inspector.to)} | {display(inspector.from)}) =
        <strong>{fmtProb(inspector.p)}</strong>
        {#if inspector.sum > 0}
          <span class="muted">({inspector.n}/{inspector.sum})</span>
        {/if}
      </span>
    {:else if focused != null}
      <span class="ins-hint">Row <code>{display(VOCAB[focused])}</code> is focused — hover any cell to read its conditional probability. Distribution below.</span>
    {:else}
      <span class="ins-hint">{hint}</span>
    {/if}
  </div>

  {#if showDistribution && focused != null && rowSums[focused] > 0}
    <section class="dist" aria-label={`Conditional distribution P of next character given ${display(VOCAB[focused])}.`}>
      <header class="dist-head">
        <span class="dist-title">P( · | <code>{display(VOCAB[focused])}</code> )</span>
        <span class="dist-sum">row sum = 1.000</span>
      </header>
      <div class="dist-bars">
        {#each distribution as d (d.char)}
          {#if d.prob > 0}
            <div class="dist-bar" title={`${display(VOCAB[focused])} → ${display(d.char)}: ${(d.prob * 100).toFixed(1)}%`}>
              <div class="dist-fill" style="height:{(d.prob / distMax) * 100}%;"></div>
              <span class="dist-char">{display(d.char)}</span>
              <span class="dist-prob">{(d.prob * 100).toFixed(0)}%</span>
            </div>
          {/if}
        {/each}
      </div>
    </section>
  {:else if showDistribution && focused != null}
    <section class="dist dist-empty">
      <span class="ins-hint">No bigrams starting with <code>{display(VOCAB[focused])}</code> in this corpus.</span>
    </section>
  {/if}

  {#if editable}
    <form
      class="add-form"
      onsubmit={(e) => {
        e.preventDefault();
        addWord();
      }}
    >
      <label class="add-label">
        <span class="add-prompt">add a word</span>
        <input
          type="text"
          class="add-input"
          bind:value={typed}
          placeholder="emma"
          maxlength="24"
          autocomplete="off"
          spellcheck="false"
          aria-label="Type a word to add to the corpus"
        />
      </label>
      <button type="submit" class="add-btn">Add</button>
      <button type="button" class="reset-btn" onclick={reset} aria-label="Reset corpus">Reset</button>
    </form>
    {#if lastError}
      <p class="err">{lastError}</p>
    {/if}
  {/if}

  {#if words.length > 0}
    <ul class="chips" aria-label="Words in the corpus">
      {#each words as w, idx (idx)}
        <li class="chip">
          <span class="chip-text">{w}</span>
          {#if editable}
            <button
              type="button"
              class="chip-x"
              onclick={() => removeWord(idx)}
              aria-label={`Remove ${w}`}
            >×</button>
          {/if}
        </li>
      {/each}
    </ul>
  {:else}
    <p class="empty">Corpus is empty. Add a word above and watch the matrix fill in.</p>
  {/if}
</div>

<style>
  .widget {
    --cell-size: clamp(16px, 2.4vw, 22px);
    --cell-empty: color-mix(in srgb, var(--site-fg) 5%, transparent);
    --cream-on-red: var(--on-color-fg);
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(0.85rem, 2vw, 1.4rem);
    color: var(--site-fg);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 50%, transparent);
  }

  .head {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.1rem;
    align-items: baseline;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
  .meta {
    display: inline-flex;
    align-items: baseline;
    gap: 0.4rem;
  }
  .meta-key {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
  }
  .meta-val {
    color: var(--site-fg);
    font-variant-numeric: tabular-nums;
    font-weight: 600;
  }
  .meta-focus {
    background: color-mix(in srgb, var(--ink-red) 10%, transparent);
    color: var(--ink-red);
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
  }
  .meta-focus .meta-val {
    color: var(--ink-red);
  }
  .meta-clear {
    border: none;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    padding: 0 0.15rem;
  }
  .meta-clear:hover {
    opacity: 0.7;
  }

  .grid-wrap {
    display: flex;
    flex-direction: column;
    background: var(--demo-stage);
    border-radius: 12px;
    padding: 0.55rem;
    overflow-x: auto;
    overflow-y: hidden;
    user-select: none;
    -webkit-user-select: none;
    touch-action: pan-x;
  }

  .col-headers,
  .row {
    display: grid;
    grid-template-columns: var(--cell-size) repeat(27, var(--cell-size));
    gap: 1px;
  }
  .row + .row {
    margin-top: 1px;
  }

  .corner {
    width: var(--cell-size);
    height: var(--cell-size);
  }
  .col-label,
  .row-label {
    width: var(--cell-size);
    height: var(--cell-size);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.7rem;
    color: var(--site-fg-muted);
    background: transparent;
    border: none;
    padding: 0;
  }
  .col-label.active {
    color: var(--ink-red);
    font-weight: 700;
  }
  .row-label {
    cursor: pointer;
    transition: color 160ms ease, background 160ms ease;
    border-radius: 4px;
  }
  .row-label:hover:not(.row-label-empty) {
    color: var(--ink-red);
  }
  .row-label-active {
    color: var(--ink-red);
    background: color-mix(in srgb, var(--ink-red) 12%, transparent);
    font-weight: 700;
  }
  .row-label-empty {
    cursor: default;
    opacity: 0.4;
  }

  .cell {
    width: var(--cell-size);
    height: var(--cell-size);
    border: none;
    padding: 0;
    margin: 0;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--site-fg);
    line-height: 1;
    border-radius: 2px;
    cursor: pointer;
    transition: transform 120ms ease, box-shadow 120ms ease;
  }
  .cell.cell-zero {
    cursor: default;
  }
  .cell.cell-hover,
  .cell:focus-visible {
    transform: scale(1.18);
    z-index: 2;
    outline: none;
    box-shadow: 0 0 0 2px var(--ink-sea);
  }
  .row.row-focused .cell:not(.cell-zero) {
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--ink-red) 40%, transparent);
  }

  .inspector {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.4rem 1rem;
    min-height: 1.6rem;
    padding: 0.45rem 0.7rem;
    border-radius: 10px;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.85rem;
    color: var(--site-fg);
  }
  .ins-pair {
    display: inline-flex;
    align-items: baseline;
    gap: 0.35rem;
  }
  .ins-pair code,
  .inspector code {
    background: var(--site-fg);
    color: var(--demo-card);
    border-radius: 4px;
    padding: 0.05rem 0.35rem;
    font-size: 0.85rem;
  }
  .arrow {
    color: var(--site-fg-muted);
  }
  .ins-stat strong {
    font-variant-numeric: tabular-nums;
  }
  .ins-stat .muted {
    color: var(--site-fg-muted);
    margin-left: 0.25rem;
    font-size: 0.78rem;
  }
  .ins-hint {
    color: var(--site-fg-muted);
    font-family: var(--font-body, Inter, system-ui, sans-serif);
    font-size: 0.85rem;
  }

  .dist {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.6rem 0.7rem 0.7rem;
    border-radius: 12px;
    background: color-mix(in srgb, var(--ink-red) 4%, transparent);
    border: 1px solid color-mix(in srgb, var(--ink-red) 18%, transparent);
  }
  .dist-empty {
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-color: color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
  .dist-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.8rem;
  }
  .dist-title {
    color: var(--site-fg);
    font-weight: 600;
  }
  .dist-title code {
    background: var(--ink-red);
    color: var(--cream-on-red);
    border-radius: 4px;
    padding: 0.05rem 0.4rem;
    font-size: 0.85rem;
  }
  .dist-sum {
    color: var(--site-fg-muted);
    font-variant-numeric: tabular-nums;
  }
  .dist-bars {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    min-height: 110px;
    padding-top: 4px;
  }
  .dist-bar {
    flex: 1 1 0;
    min-width: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    gap: 2px;
    height: 110px;
    position: relative;
  }
  .dist-fill {
    width: 100%;
    background: var(--ink-red);
    border-radius: 3px 3px 0 0;
    min-height: 2px;
    transition: height 220ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .dist-char {
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.7rem;
    color: var(--site-fg);
  }
  .dist-prob {
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.62rem;
    color: var(--site-fg-muted);
    font-variant-numeric: tabular-nums;
  }

  .add-form {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: flex-end;
    padding-top: 0.2rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
  .add-label {
    flex: 1 1 180px;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .add-prompt {
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--site-fg-muted);
  }
  .add-input {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    border-radius: 8px;
    padding: 0.45rem 0.65rem;
    background: var(--demo-stage);
    color: var(--site-fg);
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.95rem;
  }
  .add-input:focus-visible {
    outline: 2px solid var(--ink-sea);
    outline-offset: 1px;
  }
  .add-btn,
  .reset-btn {
    border: none;
    border-radius: 999px;
    padding: 0.5rem 1rem;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.85rem;
    transition: transform 120ms ease, background 160ms ease;
  }
  .add-btn {
    background: var(--ink-red);
    color: var(--cream-on-red);
  }
  .add-btn:hover {
    transform: translateY(-1px);
    background: color-mix(in srgb, var(--ink-red) 88%, black);
  }
  .reset-btn {
    background: transparent;
    color: var(--site-fg-muted);
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
  }
  .reset-btn:hover {
    color: var(--site-fg);
    border-color: var(--site-fg);
  }
  .err {
    margin: 0;
    color: var(--ink-coral);
    font-size: 0.8rem;
  }

  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: color-mix(in srgb, var(--ink-sea) 10%, transparent);
    color: var(--site-fg);
    padding: 0.18rem 0.45rem 0.18rem 0.65rem;
    border-radius: 999px;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.78rem;
  }
  .chip-x {
    border: none;
    background: transparent;
    color: var(--site-fg-muted);
    font-size: 0.95rem;
    line-height: 1;
    cursor: pointer;
    padding: 0 0.2rem;
  }
  .chip-x:hover {
    color: var(--ink-coral);
  }
  .empty {
    margin: 0;
    color: var(--site-fg-muted);
    font-style: italic;
    font-size: 0.85rem;
  }

  @media (prefers-reduced-motion: reduce) {
    .cell,
    .dist-fill,
    .add-btn {
      transition: none;
    }
  }
</style>
