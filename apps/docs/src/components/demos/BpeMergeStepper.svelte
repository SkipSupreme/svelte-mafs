<script lang="ts">
  // BpeMergeStepper — step through BPE on the canonical Hugging Face / Sennrich
  // toy corpus (E1 in the M17 research brief). Expose the pair-frequency histogram
  // as the mathematical object the algorithm consults; let the learner advance one
  // merge at a time and rewind.

  type Word = { surface: string; tokens: string[]; count: number };

  // The pedagogical corpus: hug:10, pug:5, pun:12, bun:4, hugs:5.
  function fresh(): Word[] {
    return [
      { surface: 'hug',  tokens: ['h', 'u', 'g'],       count: 10 },
      { surface: 'pug',  tokens: ['p', 'u', 'g'],       count: 5  },
      { surface: 'pun',  tokens: ['p', 'u', 'n'],       count: 12 },
      { surface: 'bun',  tokens: ['b', 'u', 'n'],       count: 4  },
      { surface: 'hugs', tokens: ['h', 'u', 'g', 's'],  count: 5  },
    ];
  }

  let words: Word[] = $state(fresh());
  let merges: Array<{ a: string; b: string; merged: string; count: number }> = $state([]);

  // Compute pair counts from the current token sequences.
  const pairCounts: Array<[string, string, number]> = $derived.by(() => {
    const m = new Map<string, number>();
    for (const w of words) {
      for (let i = 0; i < w.tokens.length - 1; i++) {
        const key = w.tokens[i] + '' + w.tokens[i + 1];
        m.set(key, (m.get(key) ?? 0) + w.count);
      }
    }
    const arr = [...m.entries()].map(([k, c]) => {
      const [a, b] = k.split('');
      return [a, b, c] as [string, string, number];
    });
    arr.sort((x, y) => y[2] - x[2]);
    return arr;
  });

  const winning = $derived(pairCounts[0] ?? null);

  function mergeOnce() {
    if (!winning) return;
    const [a, b, count] = winning;
    const merged = a + b;
    words = words.map((w) => {
      const out: string[] = [];
      let i = 0;
      while (i < w.tokens.length) {
        if (i < w.tokens.length - 1 && w.tokens[i] === a && w.tokens[i + 1] === b) {
          out.push(merged);
          i += 2;
        } else {
          out.push(w.tokens[i]);
          i += 1;
        }
      }
      return { ...w, tokens: out };
    });
    merges = [...merges, { a, b, merged, count }];
  }

  function reset() {
    words = fresh();
    merges = [];
  }

  // Tokenize a fresh word against the current merge list. Used for the
  // "encode 'hugs'" check at the bottom — answers Step 5 of Lesson 17.2.
  let probe: string = $state('hugs');
  const probeTokens: string[] = $derived.by(() => {
    let toks: string[] = [...probe];
    for (const { a, b, merged } of merges) {
      const out: string[] = [];
      let i = 0;
      while (i < toks.length) {
        if (i < toks.length - 1 && toks[i] === a && toks[i + 1] === b) {
          out.push(merged);
          i += 2;
        } else {
          out.push(toks[i]);
          i += 1;
        }
      }
      toks = out;
    }
    return toks;
  });

  // Bar geometry for the pair-frequency histogram.
  const maxCount = $derived(Math.max(1, ...pairCounts.map((p) => p[2])));
  function barH(c: number): number {
    return 8 + (c / maxCount) * 64;
  }
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">step</span>
      <span class="meta-val">{merges.length}</span>
    </div>
    <div class="meta">
      <span class="meta-key">winning pair</span>
      <span class="meta-val">
        {#if winning}
          <code>({winning[0]}, {winning[1]})</code> = {winning[2]}
        {:else}
          —
        {/if}
      </span>
    </div>
    <div class="meta">
      <span class="meta-key">vocab</span>
      <span class="meta-val">{7 + merges.length}</span>
    </div>
  </header>

  <div class="grid">
    <section class="words">
      <h4>corpus</h4>
      <ul>
        {#each words as w}
          <li>
            <span class="word-count">×{w.count}</span>
            <span class="word-tokens">
              {#each w.tokens as t, i}
                <code>{t}</code>{#if i < w.tokens.length - 1}<span class="dot">·</span>{/if}
              {/each}
            </span>
            <span class="word-surface">"{w.surface}"</span>
          </li>
        {/each}
      </ul>
    </section>

    <section class="pairs">
      <h4>pair frequencies</h4>
      <div class="bars">
        {#each pairCounts as [a, b, c], idx}
          {@const win = idx === 0}
          <div class="bar-col">
            <div class="bar-num">{c}</div>
            <div class="bar" style="height: {barH(c)}px; background: {win ? 'var(--ink-red)' : 'color-mix(in srgb, var(--site-fg) 22%, transparent)'};"></div>
            <div class="bar-label" class:win><code>{a}</code><code>{b}</code></div>
          </div>
        {/each}
      </div>
    </section>
  </div>

  <div class="controls">
    <button type="button" class="btn-primary" onclick={mergeOnce} disabled={!winning}>
      ▶ merge once
    </button>
    <button type="button" class="btn-ghost" onclick={reset} disabled={merges.length === 0}>
      reset
    </button>
  </div>

  {#if merges.length > 0}
    <section class="merges">
      <h4>merge list (the tokenizer)</h4>
      <ol>
        {#each merges as m, i}
          <li>
            <span class="merge-i">{i + 1}</span>
            <code>{m.a}</code> + <code>{m.b}</code> →
            <code class="merged">{m.merged}</code>
            <span class="merge-c">(count = {m.count})</span>
          </li>
        {/each}
      </ol>
    </section>
  {/if}

  <section class="probe">
    <h4>tokenize a word with the current merge list</h4>
    <div class="probe-row">
      <input
        type="text"
        class="probe-input"
        bind:value={probe}
        spellcheck="false"
        aria-label="Word to tokenize"
      />
      <span class="arrow">→</span>
      <span class="probe-tokens">
        {#each probeTokens as t, i (i)}
          <code class="probe-token">{t}</code>
        {/each}
        <span class="probe-count">{probeTokens.length} token{probeTokens.length === 1 ? '' : 's'}</span>
      </span>
    </div>
  </section>
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-red) 40%, transparent);
  }
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }

  h4 {
    margin: 0 0 0.4rem;
    font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--site-fg-muted); font-family: var(--font-mono);
  }

  .grid {
    display: grid; grid-template-columns: 1fr 1.4fr; gap: 1rem;
  }
  @media (max-width: 640px) {
    .grid { grid-template-columns: 1fr; }
  }

  .words ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; }
  .words li {
    display: flex; align-items: center; gap: 0.5rem;
    font-family: var(--font-mono); font-size: 0.85rem;
  }
  .word-count {
    color: var(--site-fg-muted); width: 2.4rem; text-align: right; font-size: 0.78rem;
  }
  .word-tokens { display: inline-flex; gap: 0.05rem; align-items: center; }
  .word-tokens code {
    background: var(--demo-stage); padding: 0.1rem 0.35rem; border-radius: 3px;
  }
  .dot { color: var(--site-fg-muted); padding: 0 0.18rem; font-size: 0.75rem; }
  .word-surface { color: var(--site-fg-muted); font-size: 0.78rem; margin-left: auto; }

  .pairs .bars {
    display: flex; gap: 0.4rem; align-items: flex-end;
    padding: 0.4rem;
    background: var(--demo-stage); border-radius: 10px;
    overflow-x: auto;
    min-height: 110px;
  }
  .bar-col {
    display: flex; flex-direction: column; align-items: center; gap: 0.2rem;
    min-width: 38px;
  }
  .bar-num {
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--site-fg-muted);
    font-variant-numeric: tabular-nums;
  }
  .bar { width: 18px; border-radius: 3px 3px 0 0; transition: height 220ms ease, background 220ms ease; }
  .bar-label {
    font-family: var(--font-mono); font-size: 0.72rem;
    display: flex; gap: 0;
  }
  .bar-label code {
    background: var(--demo-card); padding: 0.05rem 0.18rem; border-radius: 2px;
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
  }
  .bar-label.win code {
    border-color: var(--ink-red); color: var(--ink-red); font-weight: 700;
  }

  .controls { display: flex; gap: 0.5rem; }
  .btn-primary, .btn-ghost {
    border-radius: 999px; padding: 0.32rem 0.95rem;
    font-size: 0.85rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, border-color 160ms ease, transform 120ms ease;
  }
  .btn-primary {
    background: var(--ink-red); color: var(--on-color-fg); border: 1px solid var(--ink-red);
  }
  .btn-primary:hover:not(:disabled) { transform: translateY(-1px); }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-ghost {
    background: transparent; color: var(--site-fg-muted);
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
  }
  .btn-ghost:hover:not(:disabled) { color: var(--site-fg); border-color: var(--site-fg); }

  .merges ol {
    list-style: none; padding: 0; margin: 0;
    font-family: var(--font-mono); font-size: 0.82rem;
    display: flex; flex-direction: column; gap: 0.18rem;
  }
  .merges li { display: flex; align-items: center; gap: 0.35rem; }
  .merge-i { color: var(--site-fg-muted); width: 1.4rem; text-align: right; font-size: 0.72rem; }
  .merges code {
    background: var(--demo-stage); padding: 0.05rem 0.3rem; border-radius: 3px;
  }
  .merges code.merged {
    background: color-mix(in srgb, var(--ink-red) 22%, transparent);
    color: var(--site-fg); font-weight: 600;
  }
  .merge-c { color: var(--site-fg-muted); font-size: 0.74rem; margin-left: auto; }

  .probe-row {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
  }
  .probe-input {
    background: var(--input-bg);
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    border-radius: 8px; padding: 0.35rem 0.55rem;
    font-family: var(--font-mono); font-size: 0.9rem; color: var(--site-fg);
    width: 9rem;
  }
  .probe-input:focus { outline: 2px solid var(--ink-red); outline-offset: 1px; }
  .arrow { color: var(--site-fg-muted); font-family: var(--font-mono); }
  .probe-tokens { display: inline-flex; gap: 0.2rem; align-items: center; flex-wrap: wrap; }
  .probe-token {
    background: color-mix(in srgb, var(--ink-red) 18%, transparent);
    border-bottom: 2px solid var(--ink-red);
    padding: 0.1rem 0.35rem; border-radius: 3px;
    font-family: var(--font-mono); font-size: 0.9rem;
  }
  .probe-count {
    color: var(--site-fg-muted); font-family: var(--font-mono); font-size: 0.78rem;
    margin-left: 0.5rem;
  }
</style>
