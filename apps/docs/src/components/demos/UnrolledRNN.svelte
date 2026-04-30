<script lang="ts">
  interface Props {
    /** Hidden dimension. Kept small (4) so h_t fits as a 4-bar vector. */
    d?: number;
    /** Default input sequence (length determines unrolled depth). */
    initialInput?: string;
  }

  let { d = 4, initialInput = 'rita' }: Props = $props();

  const VOCAB: string[] = ['.', ...'abcdefghijklmnopqrstuvwxyz'.split('')];
  const V = VOCAB.length;
  const display = (c: string) => (c === '.' ? '·' : c);

  let mode: 'rolled' | 'unrolled' = $state('rolled');
  let inputStr: string = $state(initialInput);
  let cursor: number = $state(0);
  let showBackward: boolean = $state(false);

  // Hidden state is a vector. We render it as `d` bars so the learner can
  // see it being overwritten step by step.
  // Fixed pseudo-random weights for visual stability.
  function fixedRand(seed: number): () => number {
    let s = seed | 0 || 1;
    return () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return (s / 0x7fffffff) * 2 - 1;
    };
  }
  const r1 = fixedRand(42);
  const Wx: number[][] = Array.from({ length: V }, () =>
    Array.from({ length: d }, () => 0.6 * r1()),
  );
  const Wh: number[][] = Array.from({ length: d }, () =>
    Array.from({ length: d }, () => 0.4 * r1()),
  );
  const b: number[] = Array.from({ length: d }, () => 0.1 * r1());

  const tokens: number[] = $derived(
    [...inputStr.toLowerCase().replace(/[^a-z]/g, '')]
      .map((c) => VOCAB.indexOf(c))
      .filter((i) => i >= 0),
  );
  const T = $derived(tokens.length);

  // Forward pass: compute h_t for every timestep (h_0 = 0).
  const states: number[][] = $derived.by(() => {
    const out: number[][] = [Array(d).fill(0)];
    for (let t = 0; t < T; t++) {
      const tok = tokens[t];
      const next = new Array<number>(d).fill(0);
      for (let j = 0; j < d; j++) {
        let s = b[j];
        s += Wx[tok][j]; // W_x · onehot(tok) = column tok of W_x
        for (let k = 0; k < d; k++) s += Wh[k][j] * out[t][k];
        next[j] = Math.tanh(s);
      }
      out.push(next);
    }
    return out;
  });

  // Output logits at each timestep — a softmax over V from a small Wy.
  const r2 = fixedRand(99);
  const Wy: number[][] = Array.from({ length: d }, () =>
    Array.from({ length: V }, () => 0.4 * r2()),
  );
  function nextProbs(h: number[]): number[] {
    const logits = new Array<number>(V).fill(0);
    for (let j = 0; j < V; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += h[k] * Wy[k][j];
      logits[j] = s;
    }
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }
  const cursorProbs = $derived(
    cursor >= 0 && cursor < states.length ? nextProbs(states[cursor]) : [],
  );
  const argmaxIdx = $derived.by(() => {
    if (!cursorProbs.length) return -1;
    let m = -Infinity, idx = 0;
    for (let i = 0; i < cursorProbs.length; i++) if (cursorProbs[i] > m) { m = cursorProbs[i]; idx = i; }
    return idx;
  });

  function setInput(s: string): void {
    inputStr = s.slice(0, 8);
    cursor = Math.min(cursor, tokens.length);
  }
  function unroll(): void { mode = 'unrolled'; cursor = T; }
  function reroll(): void { mode = 'rolled'; cursor = 0; }

  // Bar height for h_t entries (range [-1, 1]).
  function hBarHeight(v: number): string {
    return `${(Math.abs(v) * 100).toFixed(0)}%`;
  }
  function hBarColor(v: number): string {
    return v >= 0 ? 'var(--ink-sea)' : 'var(--ink-coral)';
  }

  // Total params (W_x + W_h + b + W_y).
  const paramsWx = V * d;
  const paramsWh = d * d;
  const paramsB = d;
  const paramsWy = d * V;
  const totalParams = paramsWx + paramsWh + paramsB + paramsWy;
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">hidden dim d</span>
      <span class="meta-val">{d}</span>
    </div>
    <div class="meta">
      <span class="meta-key">timesteps T</span>
      <span class="meta-val">{T}</span>
    </div>
    <div class="meta">
      <span class="meta-key">total params</span>
      <span class="meta-val">{totalParams}</span>
    </div>
    <div class="meta">
      <span class="meta-key">cursor</span>
      <span class="meta-val">t = {cursor}</span>
    </div>
  </header>

  <div class="seg">
    <button
      type="button"
      class="seg-btn"
      class:active={mode === 'rolled'}
      onclick={reroll}
    >Rolled (one cell)</button>
    <button
      type="button"
      class="seg-btn"
      class:active={mode === 'unrolled'}
      onclick={unroll}
    >Unrolled ({T} copies)</button>
  </div>

  {#if mode === 'rolled'}
    <div class="rolled">
      <div class="cell">
        <div class="cell-label">RNN cell</div>
        <div class="cell-eqn">h<sub>t</sub> = tanh(W<sub>x</sub> x<sub>t</sub> + W<sub>h</sub> h<sub>t−1</sub> + b)</div>
        <div class="cell-loop" aria-hidden="true">↺</div>
      </div>
      <p class="rolled-explainer">
        One cell. One <em>W<sub>x</sub></em>, one <em>W<sub>h</sub></em>, one
        <em>b</em>. The output of step <em>t</em> feeds back as the input
        of step <em>t + 1</em>. This is the model. Press <strong>Unroll</strong>
        to see what happens when you run it through {T} timesteps.
      </p>
    </div>
  {:else}
    <div class="unrolled">
      <div class="chain" role="group" aria-label="Unrolled RNN graph.">
        {#each states as h, t}
          {@const isCursor = cursor === t}
          {@const isAfter = showBackward && t < cursor}
          <div class="step" class:current={isCursor} class:in-backward={isAfter}>
            <!-- Token going IN -->
            <div class="step-token">
              {#if t < T}
                <div class="tok">{display(VOCAB[tokens[t]])}</div>
                <div class="tok-label">x<sub>{t + 1}</sub></div>
              {:else}
                <div class="tok tok-empty">—</div>
              {/if}
            </div>
            <!-- The cell itself -->
            <button
              type="button"
              class="ucell"
              class:current={isCursor}
              onclick={() => (cursor = t)}
              aria-label={`Inspect timestep ${t}`}
            >
              {#if t === 0}
                <span class="ucell-label">h<sub>0</sub></span>
              {:else}
                <span class="ucell-label">cell</span>
              {/if}
            </button>
            <!-- Hidden state as bars -->
            <div class="h-bars" title={`h_${t} = (${h.map((v) => v.toFixed(2)).join(', ')})`}>
              {#each h as v, i}
                <div
                  class="hb"
                  class:hb-pos={v >= 0}
                  class:hb-neg={v < 0}
                  style="height:{hBarHeight(v)};background:{hBarColor(v)};"
                ></div>
              {/each}
            </div>
            <div class="h-label">h<sub>{t}</sub></div>
            <!-- Forward arrow -->
            {#if t < states.length - 1}
              <div class="arrow arrow-fwd" aria-hidden="true">→</div>
            {/if}
            <!-- Backward arrow overlay -->
            {#if showBackward && t > 0 && t <= cursor}
              <div class="arrow arrow-bwd" aria-hidden="true">←</div>
            {/if}
          </div>
        {/each}
      </div>

      <div class="scrub">
        <label class="slider">
          <span class="slider-label">scrub timestep: <strong>t = {cursor}</strong></span>
          <input type="range" min="0" max={states.length - 1} step="1" bind:value={cursor} aria-label="Timestep cursor" />
        </label>
        <div class="ctl-row">
          <button
            type="button"
            class="btn btn-toggle"
            class:active={showBackward}
            onclick={() => (showBackward = !showBackward)}
          >{showBackward ? 'backward pass: ON' : 'backward pass: OFF'}</button>
        </div>
      </div>

      <div class="cursor-readout">
        <div class="cr-block">
          <div class="cr-key">h<sub>{cursor}</sub></div>
          <div class="cr-val">
            ({states[cursor]?.map((v) => v.toFixed(2)).join(', ') ?? ''})
          </div>
        </div>
        {#if cursor > 0 && cursor <= T}
          <div class="cr-block">
            <div class="cr-key">predicted next from h<sub>{cursor}</sub></div>
            <div class="cr-val">
              <span class="argmax-pill">{display(VOCAB[argmaxIdx])}</span>
              <span class="muted">P = {cursorProbs[argmaxIdx]?.toFixed(3)}</span>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <div class="input-row">
    <label class="input-label">
      <span>input sequence</span>
      <input
        type="text"
        class="input-text"
        value={inputStr}
        oninput={(e) => setInput((e.target as HTMLInputElement).value)}
        maxlength="8"
        autocomplete="off"
      />
    </label>
    <span class="input-hint">letters a–z, max 8</span>
  </div>

  <p class="caption">
    Same <em>W<sub>x</sub></em>, <em>W<sub>h</sub></em>, <em>b</em> at every
    timestep — that is <strong>weight sharing</strong>. h<sub>t</sub> is just
    a vector ({d} numbers, drawn as {d} bars). It gets overwritten each step.
    Toggle <em>backward pass</em> to see how gradients flow from the cursor
    timestep all the way back to the input — that's BPTT, and it's just
    backprop on the unrolled graph.
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 50%, transparent);
  }
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }

  .seg {
    align-self: flex-start;
    display: inline-flex; gap: 0;
    background: color-mix(in srgb, var(--site-fg) 8%, transparent);
    border-radius: 999px; padding: 3px;
  }
  .seg-btn {
    border: none; background: transparent; color: var(--site-fg-muted);
    padding: 0.32rem 0.85rem; border-radius: 999px; cursor: pointer;
    font-size: 0.83rem; font-weight: 600;
    transition: background 160ms ease, color 160ms ease;
  }
  .seg-btn.active { background: var(--ink-sea); color: var(--on-color-fg); }

  .rolled {
    display: flex; flex-direction: column;
    align-items: center; gap: 1rem;
    padding: 1.4rem 1rem;
    background: var(--demo-stage); border-radius: 12px;
  }
  .cell {
    position: relative;
    width: 200px; height: 110px;
    border: 2px solid var(--ink-sea);
    border-radius: 12px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 0.45rem;
    background: color-mix(in srgb, var(--ink-sea) 6%, var(--demo-card));
  }
  .cell-label {
    font-family: var(--font-mono); font-size: 0.78rem; font-weight: 700;
    color: var(--ink-sea); text-transform: uppercase; letter-spacing: 0.08em;
  }
  .cell-eqn {
    font-family: var(--font-mono); font-size: 0.78rem;
    color: var(--site-fg);
  }
  .cell-loop {
    position: absolute; right: -22px; top: 50%; transform: translateY(-50%);
    width: 30px; height: 30px;
    border: 2px solid var(--ink-sea);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: var(--ink-sea); font-size: 1rem;
    background: var(--demo-stage);
  }
  .rolled-explainer {
    margin: 0.5rem 0 0;
    font-size: 0.88rem; color: var(--site-fg-muted);
    text-align: center; max-width: 36ch; line-height: 1.55;
  }
  .rolled-explainer em { color: var(--site-fg); font-style: normal; font-family: var(--font-mono); font-size: 0.85em; }
  .rolled-explainer strong { color: var(--ink-sea); }

  .unrolled {
    display: flex; flex-direction: column; gap: 0.7rem;
  }
  .chain {
    background: var(--demo-stage);
    border-radius: 12px;
    padding: 0.8rem 0.5rem;
    overflow-x: auto;
    display: flex;
    align-items: stretch;
    gap: 0;
  }
  .step {
    display: grid;
    grid-template-rows: auto auto auto auto auto;
    align-items: center;
    justify-items: center;
    gap: 0.25rem;
    padding: 0 0.55rem;
    position: relative;
    flex-shrink: 0;
  }
  .step-token {
    display: flex; flex-direction: column; align-items: center; gap: 1px;
    margin-bottom: 0.2rem;
  }
  .tok {
    width: 28px; height: 28px;
    display: inline-flex; align-items: center; justify-content: center;
    background: var(--ink-red); color: var(--on-color-fg);
    border-radius: 6px;
    font-family: var(--font-mono); font-weight: 700; font-size: 0.95rem;
  }
  .tok-empty { background: transparent; color: var(--site-fg-muted); border: 1px dashed currentColor; }
  .tok-label {
    font-family: var(--font-mono); font-size: 0.6rem; color: var(--site-fg-muted);
  }
  .ucell {
    width: 56px; height: 38px;
    border: 1.5px solid var(--ink-sea);
    background: color-mix(in srgb, var(--ink-sea) 6%, var(--demo-card));
    color: var(--ink-sea);
    border-radius: 8px;
    cursor: pointer;
    font-family: var(--font-mono); font-size: 0.7rem;
    transition: transform 120ms ease, background 160ms ease;
    padding: 0;
  }
  .ucell:hover { transform: translateY(-1px); }
  .ucell.current {
    border-width: 2.5px;
    background: var(--ink-sea); color: var(--on-color-fg);
  }
  .ucell-label sub { font-size: 0.7em; }

  .h-bars {
    display: flex; align-items: flex-end;
    gap: 1.5px;
    width: 56px; height: 26px;
    border-bottom: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
  }
  .hb {
    flex: 1 1 0;
    min-height: 1px;
    border-radius: 1.5px 1.5px 0 0;
    transition: height 200ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .h-label { font-family: var(--font-mono); font-size: 0.65rem; color: var(--site-fg-muted); }

  .step .arrow {
    position: absolute;
    top: 90px; /* aligned with cell midpoint */
    color: var(--site-fg-muted);
    font-size: 1.2rem;
    line-height: 1;
    pointer-events: none;
  }
  .arrow-fwd { right: -8px; }
  .arrow-bwd { left: -8px; color: var(--ink-coral); top: 110px; }

  .step.current .h-label { color: var(--ink-sea); font-weight: 700; }
  .step.in-backward .ucell { border-color: var(--ink-coral); }

  .scrub {
    display: flex; flex-direction: column; gap: 0.4rem;
  }
  .slider { display: flex; flex-direction: column; gap: 0.2rem; }
  .slider-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted); }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input[type='range'] { width: 100%; accent-color: var(--ink-sea); }
  .ctl-row { display: flex; gap: 0.4rem; }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.32rem 0.85rem;
    font-size: 0.82rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover { transform: translateY(-1px); border-color: var(--site-fg); }
  .btn-toggle.active { background: var(--ink-coral); color: var(--on-color-fg); border-color: var(--ink-coral); }

  .cursor-readout {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem;
    padding: 0.55rem 0.75rem;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-radius: 8px;
  }
  .cr-block { display: flex; flex-direction: column; gap: 0.15rem; }
  .cr-key { font-family: var(--font-mono); font-size: 0.7rem; color: var(--site-fg-muted); text-transform: uppercase; letter-spacing: 0.08em; }
  .cr-val { font-family: var(--font-mono); font-size: 0.85rem; color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .argmax-pill {
    background: var(--ink-red); color: var(--on-color-fg);
    padding: 0 0.4rem; border-radius: 4px;
    margin-right: 0.35rem;
  }
  .cr-val .muted { color: var(--site-fg-muted); }

  .input-row {
    display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: flex-end;
    padding-top: 0.4rem;
    border-top: 1px dashed color-mix(in srgb, var(--site-fg) 14%, transparent);
  }
  .input-label {
    display: flex; flex-direction: column; gap: 0.2rem;
    flex: 1 1 200px;
  }
  .input-label > span {
    font-family: var(--font-mono); font-size: 0.7rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--site-fg-muted);
  }
  .input-text {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    border-radius: 8px;
    padding: 0.4rem 0.65rem;
    background: var(--demo-stage); color: var(--site-fg);
    font-family: var(--font-mono); font-size: 0.95rem;
  }
  .input-text:focus-visible { outline: 2px solid var(--ink-sea); outline-offset: 1px; }
  .input-hint { font-size: 0.78rem; color: var(--site-fg-muted); }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg); font-style: normal;
    font-family: var(--font-mono); font-size: 0.85em;
  }
  .caption strong { color: var(--ink-sea); }

  @media (prefers-reduced-motion: reduce) { .hb { transition: none; } }
</style>
