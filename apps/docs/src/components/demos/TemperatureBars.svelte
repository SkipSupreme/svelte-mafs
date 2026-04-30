<script lang="ts">
  import { onDestroy } from 'svelte';

  interface Props {
    /** Initial logits (length 27 by default — same vocab as the bigram lessons). */
    initialLogits?: number[];
    initialTemperature?: number;
    initialTopK?: number;
  }

  // Default logits: shaped after a realistic "what comes after a vowel" row —
  // common consonants (n, r, l, s, t) get higher logits, rarer letters get
  // lower, and the start/end token · sits in the middle.
  function defaultLogits(): number[] {
    const VOCAB = ['.', ...'abcdefghijklmnopqrstuvwxyz'.split('')];
    // Hand-tuned approximation of a real bigram row's log-probabilities.
    const heat: Record<string, number> = {
      '.': 1.4,
      a: 0.6, b: -0.4, c: 0.0, d: 0.7, e: 0.5, f: -0.5, g: -0.2, h: 0.1,
      i: 0.4, j: -1.2, k: -0.4, l: 1.1, m: 0.5, n: 1.6, o: 0.2, p: 0.0,
      q: -2.0, r: 1.5, s: 1.2, t: 1.3, u: -0.2, v: -0.4, w: -0.4, x: -1.6,
      y: 0.3, z: -1.2,
    };
    return VOCAB.map((c) => heat[c] ?? 0);
  }

  let {
    initialLogits = defaultLogits(),
    initialTemperature = 1,
    initialTopK = 27,
  }: Props = $props();

  const VOCAB: string[] = ['.', ...'abcdefghijklmnopqrstuvwxyz'.split('')];
  const V = VOCAB.length;
  const display = (c: string) => (c === '.' ? '·' : c);

  let logits: number[] = $state([...initialLogits]);
  let temperature: number = $state(initialTemperature);
  let topK: number = $state(initialTopK);
  let sampleCounts: number[] = $state(Array(V).fill(0));
  let nSamples: number = $state(0);
  let auto: boolean = $state(false);
  let rafId: number | null = null;

  // Stable softmax with temperature applied.
  const probs: number[] = $derived.by(() => {
    const scaled = logits.map((l) => l / temperature);
    const m = Math.max(...scaled);
    const exps = scaled.map((l) => Math.exp(l - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  });

  // Top-k masked + renormalized distribution.
  const topkProbs: number[] = $derived.by(() => {
    if (topK >= V) return probs;
    const idxByProb = probs
      .map((p, i) => [p, i] as [number, number])
      .sort((a, b) => b[0] - a[0]);
    const keep = new Set<number>(idxByProb.slice(0, topK).map(([, i]) => i));
    const masked = probs.map((p, i) => (keep.has(i) ? p : 0));
    const s = masked.reduce((a, b) => a + b, 0);
    return s > 0 ? masked.map((p) => p / s) : masked;
  });

  // The argmax — used to highlight that argmax is invariant under temperature.
  const argmax = $derived.by(() => {
    let m = -Infinity, idx = 0;
    for (let i = 0; i < V; i++) if (probs[i] > m) { m = probs[i]; idx = i; }
    return idx;
  });

  function sampleOnce(): void {
    const u = Math.random();
    let acc = 0;
    let pick = topkProbs.length - 1;
    for (let j = 0; j < V; j++) {
      acc += topkProbs[j];
      if (u <= acc) { pick = j; break; }
    }
    sampleCounts = sampleCounts.map((c, i) => (i === pick ? c + 1 : c));
    nSamples += 1;
  }
  function sampleN(n: number): void {
    const next = [...sampleCounts];
    for (let k = 0; k < n; k++) {
      const u = Math.random();
      let acc = 0;
      let pick = topkProbs.length - 1;
      for (let j = 0; j < V; j++) {
        acc += topkProbs[j];
        if (u <= acc) { pick = j; break; }
      }
      next[pick] += 1;
    }
    sampleCounts = next;
    nSamples += n;
  }
  function startAuto(): void {
    if (auto) return;
    auto = true;
    const tick = () => {
      if (!auto) return;
      sampleN(20);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }
  function stopAuto(): void {
    auto = false;
    if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
  }
  function resetSamples(): void {
    sampleCounts = Array(V).fill(0);
    nSamples = 0;
  }
  function resetAll(): void {
    stopAuto();
    logits = [...initialLogits];
    temperature = 1;
    topK = V;
    resetSamples();
  }
  onDestroy(() => stopAuto());

  // === SVG geometry ===
  const W = 460;
  const H = 220;
  const padL = 22;
  const padR = 12;
  const padT = 16;
  const padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const barGap = 1;
  const barW = (plotW - barGap * (V - 1)) / V;

  // Bar height scales by max prob in the un-tempered scenario so the
  // y-axis stays stable as the user drags T.
  const yScale = $derived.by(() => {
    let m = 0;
    for (const p of topkProbs) if (p > m) m = p;
    return Math.max(m, 0.4);
  });
  function barX(i: number): number {
    return padL + i * (barW + barGap);
  }
  function barTop(p: number): number {
    return padT + plotH - (p / yScale) * plotH;
  }
  function barH(p: number): number {
    return (p / yScale) * plotH;
  }

  // Empirical histogram (relative frequencies) for overlay below.
  const empirical: number[] = $derived(
    nSamples > 0 ? sampleCounts.map((c) => c / nSamples) : Array(V).fill(0),
  );

  // === Drag-to-edit logits ===
  let dragging: number | null = null;

  function handlePointerDown(e: PointerEvent, i: number) {
    dragging = i;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    handleDrag(e, i);
  }
  function handlePointerMove(e: PointerEvent, i: number) {
    if (dragging !== i) return;
    handleDrag(e, i);
  }
  function handlePointerUp(e: PointerEvent, i: number) {
    if (dragging === i) {
      try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch {}
    }
    dragging = null;
  }
  function handleDrag(e: PointerEvent, i: number) {
    const svg = (e.currentTarget as Element).ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    // viewBox is 0 0 W H; client coord → svg coord
    const yClient = e.clientY - rect.top;
    const yLocal = (yClient / rect.height) * H;
    const yInPlot = Math.max(0, Math.min(plotH, yLocal - padT));
    // Map to a probability target, then to a logit.
    // We hold all other logits fixed and adjust this one to *want* a
    // certain probability — solve softmax for the corresponding logit.
    const targetProb = Math.max(0.001, Math.min(0.99, 1 - yInPlot / plotH));
    // Compute denominator from other logits (stable form).
    const otherScaled = logits.map((l, k) => (k === i ? -Infinity : l / temperature));
    const m = Math.max(...otherScaled.filter(Number.isFinite));
    let sumOther = 0;
    for (let k = 0; k < V; k++) if (k !== i) sumOther += Math.exp(logits[k] / temperature - m);
    // p_i = exp(L_i/T - m) / (sum_other + exp(L_i/T - m))
    // → exp(L_i/T - m) = p · sumOther / (1 - p)
    const ratio = (targetProb * sumOther) / (1 - targetProb);
    const newLogit = (Math.log(ratio) + m) * temperature;
    const next = [...logits];
    next[i] = Math.max(-6, Math.min(6, newLogit));
    logits = next;
    resetSamples();
  }

  const fmt3 = (n: number) => (n < 0.001 ? n.toExponential(1) : n.toFixed(3));
  const argmaxLabel = $derived(display(VOCAB[argmax]));
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">argmax</span>
      <span class="meta-val argmax">{argmaxLabel}</span>
    </div>
    <div class="meta">
      <span class="meta-key">P(argmax)</span>
      <span class="meta-val">{fmt3(probs[argmax])}</span>
    </div>
    <div class="meta">
      <span class="meta-key">T</span>
      <span class="meta-val">{temperature.toFixed(2)}</span>
    </div>
    {#if topK < V}
      <div class="meta">
        <span class="meta-key">top-k</span>
        <span class="meta-val">{topK}</span>
      </div>
    {/if}
    <div class="meta">
      <span class="meta-key">samples</span>
      <span class="meta-val">{nSamples}</span>
    </div>
  </header>

  <div class="plot-wrap">
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label="Editable categorical distribution over 27 characters.">
      <rect x="0" y="0" width={W} height={H} fill="var(--demo-stage)" rx="8" />

      <!-- y gridlines -->
      {#each [0.25, 0.5, 0.75, 1.0] as y}
        {#if y * yScale <= yScale + 0.001 && y <= 1}
          <line
            x1={padL} x2={W - padR}
            y1={padT + plotH - (y * yScale / yScale) * plotH}
            y2={padT + plotH - (y * yScale / yScale) * plotH}
            stroke="color-mix(in srgb, var(--site-fg) 8%, transparent)"
            stroke-width="0.5"
          />
        {/if}
      {/each}

      <!-- bars + drag handles -->
      {#each VOCAB as char, i}
        {@const isTopK = topkProbs[i] > 0}
        {@const baseP = topkProbs[i]}
        <g>
          <!-- bar (probability) -->
          <rect
            x={barX(i)}
            y={barTop(baseP)}
            width={barW}
            height={barH(baseP)}
            fill={isTopK ? 'var(--ink-red)' : 'color-mix(in srgb, var(--ink-red) 12%, transparent)'}
            opacity={isTopK ? 1 : 0.3}
            rx="1.5"
          />
          <!-- empirical -->
          {#if isTopK && empirical[i] > 0}
            <rect
              x={barX(i)}
              y={barTop(empirical[i])}
              width={barW}
              height="2"
              fill="var(--ink-coral)"
              opacity="0.95"
            />
          {/if}
          <!-- drag handle: invisible large area at top of bar for easy grabbing -->
          <rect
            x={barX(i) - 1}
            y={Math.max(padT, barTop(baseP) - 12)}
            width={barW + 2}
            height={Math.min(plotH, barH(baseP) + 14)}
            fill="transparent"
            style="cursor: ns-resize; touch-action: none;"
            onpointerdown={(e) => handlePointerDown(e, i)}
            onpointermove={(e) => handlePointerMove(e, i)}
            onpointerup={(e) => handlePointerUp(e, i)}
            onpointercancel={(e) => handlePointerUp(e, i)}
            aria-label={`Drag to set probability of ${display(char)}`}
          />
          <!-- handle dot at top of bar -->
          <circle
            cx={barX(i) + barW / 2}
            cy={barTop(baseP)}
            r="2.5"
            fill={isTopK ? 'var(--ink-red)' : 'color-mix(in srgb, var(--ink-red) 30%, transparent)'}
            stroke="var(--demo-stage)"
            stroke-width="1"
            pointer-events="none"
          />
          <!-- label -->
          <text
            x={barX(i) + barW / 2}
            y={H - 12}
            text-anchor="middle"
            font-size="9"
            font-family="var(--font-mono)"
            fill={i === argmax ? 'var(--ink-red)' : 'var(--site-fg-muted)'}
            font-weight={i === argmax ? '700' : '400'}
          >{display(char)}</text>
        </g>
      {/each}

      <!-- legend -->
      <g transform={`translate(${W - padR - 165}, 12)`}>
        <rect x="0" y="0" width="10" height="6" fill="var(--ink-red)" rx="1" />
        <text x="14" y="6" font-size="9" fill="var(--site-fg)" font-family="var(--font-mono)">P (drag bar tops)</text>
        <rect x="100" y="2" width="10" height="2" fill="var(--ink-coral)" />
        <text x="114" y="6" font-size="9" fill="var(--site-fg)" font-family="var(--font-mono)">empirical</text>
      </g>
    </svg>
  </div>

  <div class="controls">
    <label class="slider">
      <span class="slider-label">temperature T = <strong>{temperature.toFixed(2)}</strong></span>
      <input type="range" min="0.1" max="3" step="0.05" bind:value={temperature} aria-label="Temperature" />
    </label>
    <label class="slider">
      <span class="slider-label">top-k = <strong>{topK}</strong>{topK === V ? ' (off)' : ''}</span>
      <input type="range" min="1" max={V} step="1" bind:value={topK} aria-label="Top-k cutoff" />
    </label>
    <div class="ctl-row">
      <button type="button" class="btn btn-primary" onclick={() => sampleN(1)} disabled={auto}>Sample × 1</button>
      <button type="button" class="btn" onclick={() => sampleN(100)} disabled={auto}>Sample × 100</button>
      <button
        type="button"
        class="btn btn-toggle"
        class:active={auto}
        onclick={() => (auto ? stopAuto() : startAuto())}
      >{auto ? 'Stop' : 'Auto'}</button>
      <button type="button" class="btn btn-ghost" onclick={resetSamples}>Reset samples</button>
      <button type="button" class="btn btn-ghost" onclick={resetAll}>Reset all</button>
    </div>
  </div>

  <p class="caption">
    Drag any bar's top to set its probability directly. The temperature
    slider stretches or squeezes the whole distribution — but notice the
    argmax (highlighted character) doesn't change for any T &gt; 0. Top-k
    censors all but the largest k bars before sampling. Empirical samples
    appear as coral ticks above each bar; with enough samples they lock
    onto the bar tops.
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
  .meta-val.argmax {
    background: var(--ink-red); color: var(--on-color-fg);
    padding: 0 0.45rem; border-radius: 4px;
    font-family: var(--font-mono);
  }

  .plot-wrap {
    width: 100%; overflow: hidden; border-radius: 12px;
    user-select: none; -webkit-user-select: none;
  }
  .plot-wrap svg { width: 100%; height: auto; display: block; }

  .controls { display: flex; flex-direction: column; gap: 0.55rem; }
  .ctl-row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.32rem 0.85rem;
    font-size: 0.83rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover:not(:disabled) { transform: translateY(-1px); border-color: var(--site-fg); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: var(--ink-red); color: var(--on-color-fg); border-color: var(--ink-red); }
  .btn-toggle.active { background: var(--cta); color: var(--cta-fg); border-color: var(--cta); }
  .btn-ghost { color: var(--site-fg-muted); font-weight: 500; }

  .slider { display: flex; flex-direction: column; gap: 0.2rem; }
  .slider-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted); }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input[type='range'] { width: 100%; accent-color: var(--ink-red); }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
</style>
