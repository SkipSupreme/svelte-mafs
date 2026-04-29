<script lang="ts">
  // Pattern-recognition exercise: 6 unlabeled loss curves, learner picks
  // the diagnosis. Curves are synthetic but shaped after the canonical
  // pathologies named in the lesson.

  type Diagnosis =
    | 'clean'
    | 'overfit'
    | 'plateau'
    | 'diverge'
    | 'dead';

  interface Case {
    train: number[];
    val: number[];
    answer: Diagnosis;
    why: string;
    /** Step at which divergence happens, for the spike case. */
    spikeAt?: number;
  }

  const STEPS = 60;
  const DIAGNOSES: { key: Diagnosis; label: string; short: string }[] = [
    { key: 'clean', label: 'Clean fit', short: 'Both curves decay; gap stays small.' },
    { key: 'overfit', label: 'Overfitting fork', short: 'Train keeps dropping; val turns and climbs.' },
    { key: 'plateau', label: 'Plateau / underfit', short: 'Both curves stop dropping high.' },
    { key: 'diverge', label: 'Divergence spike', short: 'Was fine, then loss exploded.' },
    { key: 'dead', label: 'Dead curve', short: 'Loss never moves from initial value.' },
  ];

  function noise(seed: number): () => number {
    let s = seed | 0 || 1;
    return () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return (s / 0x7fffffff) * 2 - 1;
    };
  }

  function clean(seed: number): Case {
    const r = noise(seed);
    const train = Array.from({ length: STEPS }, (_, t) => 0.4 + 2.6 * Math.exp(-t / 14) + 0.04 * r());
    const val = Array.from({ length: STEPS }, (_, t) => 0.55 + 2.6 * Math.exp(-t / 14) + 0.07 * r());
    return { train, val, answer: 'clean', why: 'Both curves decay together; the gap stays small. Ship it.' };
  }

  function overfit(seed: number): Case {
    const r = noise(seed);
    const train = Array.from({ length: STEPS }, (_, t) => 0.15 + 2.8 * Math.exp(-t / 12) + 0.03 * r());
    const val = Array.from({ length: STEPS }, (_, t) => {
      const decay = 0.6 + 2.4 * Math.exp(-t / 9);
      const climb = t > 18 ? 0.025 * (t - 18) : 0;
      return decay + climb + 0.05 * r();
    });
    return { train, val, answer: 'overfit', why: 'Train keeps falling but val turns up around step 20 — model is memorizing past the point of generalization.' };
  }

  function plateau(seed: number): Case {
    const r = noise(seed);
    const train = Array.from({ length: STEPS }, (_, t) => 2.4 + 0.55 * Math.exp(-t / 18) + 0.04 * r());
    const val = Array.from({ length: STEPS }, (_, t) => 2.55 + 0.55 * Math.exp(-t / 18) + 0.06 * r());
    return { train, val, answer: 'plateau', why: 'Both curves bottom out around 2.4 — the model never learned much. Underfitting or an optimization problem.' };
  }

  function diverge(seed: number): Case {
    const r = noise(seed);
    const spikeAt = 32;
    const train = Array.from({ length: STEPS }, (_, t) => {
      if (t < spikeAt) return 1.1 + 1.9 * Math.exp(-t / 12) + 0.04 * r();
      return 1.1 + 1.9 * Math.exp(-spikeAt / 12) + Math.pow(1.55, t - spikeAt) * 0.4;
    });
    const val = Array.from({ length: STEPS }, (_, t) => {
      if (t < spikeAt) return 1.25 + 1.9 * Math.exp(-t / 12) + 0.06 * r();
      return 1.25 + 1.9 * Math.exp(-spikeAt / 12) + Math.pow(1.55, t - spikeAt) * 0.4;
    });
    return { train, val, answer: 'diverge', why: `A clean run, then loss explodes at step ~${spikeAt}. Learning rate too high, missing warmup, or an exploded gradient.`, spikeAt };
  }

  function dead(seed: number): Case {
    const r = noise(seed);
    const baseline = 3.3;
    const train = Array.from({ length: STEPS }, () => baseline + 0.04 * r());
    const val = Array.from({ length: STEPS }, () => baseline + 0.05 * r());
    return { train, val, answer: 'dead', why: 'Loss never drops from −log(1/27). The network is not learning at all — almost always bad init or the wrong loss.' };
  }

  // 6 cases — every diagnosis appears at least once, plus one extra
  // overfit (the most common real-world pathology).
  const cases: Case[] = [
    clean(11),
    overfit(23),
    diverge(7),
    plateau(31),
    dead(53),
    overfit(89),
  ];

  let idx: number = $state(0);
  let picked: Diagnosis | null = $state(null);
  let revealed: number[] = $state([]); // case indices solved correctly

  const current = $derived(cases[idx]);
  const correct = $derived(picked != null && picked === current.answer);
  const showAnswer = $derived(picked != null);
  const nextIdx = $derived((idx + 1) % cases.length);

  function pick(d: Diagnosis) {
    if (showAnswer) return;
    picked = d;
    if (d === current.answer && !revealed.includes(idx)) {
      revealed = [...revealed, idx];
    }
  }

  function next() {
    idx = nextIdx;
    picked = null;
  }

  // SVG layout
  const W = 460;
  const H = 200;
  const padL = 36;
  const padR = 12;
  const padT = 14;
  const padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Visualization clamps loss to [0, ~5] so divergence reads as "off the top"
  // rather than crashing the y-axis.
  const Y_MAX = 5;

  function pathFor(arr: number[], spikeAt?: number): string {
    const parts: string[] = [];
    arr.forEach((v, i) => {
      const x = padL + (i / (STEPS - 1)) * plotW;
      const clamped = Math.min(v, Y_MAX);
      const y = padT + plotH - (clamped / Y_MAX) * plotH;
      parts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`);
      if (spikeAt != null && i === spikeAt && v > Y_MAX * 0.6) {
        // mark the spike with a tick
      }
    });
    return parts.join(' ');
  }

  const trainPath = $derived(pathFor(current.train));
  const valPath = $derived(pathFor(current.val, current.spikeAt));

  function gridY(yVal: number): number {
    return padT + plotH - (yVal / Y_MAX) * plotH;
  }
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">case</span>
      <span class="meta-val">{idx + 1} / {cases.length}</span>
    </div>
    <div class="meta">
      <span class="meta-key">solved</span>
      <span class="meta-val" class:strong={revealed.length === cases.length}>
        {revealed.length} / {cases.length}
      </span>
    </div>
  </header>

  <div class="plot-wrap">
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Train and validation loss curves over training steps."
    >
      <!-- background -->
      <rect x="0" y="0" width={W} height={H} fill="var(--demo-stage)" rx="8" />

      <!-- y gridlines -->
      {#each [1, 2, 3, 4] as y}
        <line
          x1={padL}
          x2={W - padR}
          y1={gridY(y)}
          y2={gridY(y)}
          stroke="color-mix(in srgb, var(--site-fg) 10%, transparent)"
          stroke-width="0.5"
        />
        <text
          x={padL - 6}
          y={gridY(y) + 3}
          text-anchor="end"
          font-size="9"
          fill="var(--site-fg-muted)"
          font-family="var(--font-mono)"
        >{y}</text>
      {/each}

      <!-- spike marker -->
      {#if current.spikeAt != null}
        <line
          x1={padL + (current.spikeAt / (STEPS - 1)) * plotW}
          x2={padL + (current.spikeAt / (STEPS - 1)) * plotW}
          y1={padT}
          y2={padT + plotH}
          stroke="color-mix(in srgb, var(--ink-coral) 25%, transparent)"
          stroke-width="1"
          stroke-dasharray="3 3"
          opacity={showAnswer ? 1 : 0}
        />
      {/if}

      <!-- val curve -->
      <path d={valPath} fill="none" stroke="var(--ink-sea)" stroke-width="1.8" stroke-linejoin="round" />
      <!-- train curve -->
      <path d={trainPath} fill="none" stroke="var(--ink-red)" stroke-width="1.8" stroke-linejoin="round" />

      <!-- x-axis label -->
      <text
        x={W / 2}
        y={H - 8}
        text-anchor="middle"
        font-size="9"
        fill="var(--site-fg-muted)"
        font-family="var(--font-mono)"
      >training steps →</text>

      <!-- legend -->
      <g transform={`translate(${W - padR - 90}, ${padT + 6})`}>
        <line x1="0" x2="14" y1="4" y2="4" stroke="var(--ink-red)" stroke-width="2" />
        <text x="18" y="7" font-size="9" fill="var(--site-fg)" font-family="var(--font-mono)">train</text>
        <line x1="44" x2="58" y1="4" y2="4" stroke="var(--ink-sea)" stroke-width="2" />
        <text x="62" y="7" font-size="9" fill="var(--site-fg)" font-family="var(--font-mono)">val</text>
      </g>
    </svg>
  </div>

  <div class="choices" role="group" aria-label="Diagnosis choices">
    {#each DIAGNOSES as d}
      <button
        type="button"
        class="choice"
        class:correct={showAnswer && d.key === current.answer}
        class:wrong={showAnswer && picked === d.key && d.key !== current.answer}
        class:disabled={showAnswer && picked !== d.key && d.key !== current.answer}
        onclick={() => pick(d.key)}
        disabled={showAnswer}
      >
        <span class="choice-label">{d.label}</span>
        <span class="choice-short">{d.short}</span>
      </button>
    {/each}
  </div>

  <div class="feedback" aria-live="polite" class:show={showAnswer}>
    {#if showAnswer}
      <p class="fb-line">
        <strong class:right={correct} class:wrong={!correct}>
          {correct ? 'Correct.' : 'Not quite.'}
        </strong>
        {current.why}
      </p>
      <button type="button" class="next-btn" onclick={next}>
        Next case →
      </button>
    {:else}
      <p class="fb-line muted">Pick the diagnosis that matches the curves above.</p>
    {/if}
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
    padding: clamp(0.85rem, 2vw, 1.4rem);
    color: var(--site-fg);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 50%, transparent);
  }
  .head {
    display: flex;
    flex-wrap: wrap;
    gap: 1.1rem;
    font-family: var(--font-mono, ui-monospace, "SF Mono", Menlo, monospace);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key {
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem;
  }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .meta-val.strong { color: var(--cta); }

  .plot-wrap {
    width: 100%;
    overflow: hidden;
    border-radius: 12px;
  }
  .plot-wrap svg { width: 100%; height: auto; display: block; }

  .choices {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.5rem;
  }
  .choice {
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
    padding: 0.65rem 0.85rem;
    border-radius: 10px;
    border: 1px solid color-mix(in srgb, var(--site-fg) 16%, transparent);
    background: transparent;
    color: var(--site-fg);
    cursor: pointer;
    text-align: left;
    transition: transform 120ms ease, border-color 160ms ease, background 160ms ease;
  }
  .choice:hover:not(:disabled) {
    transform: translateY(-1px);
    border-color: var(--site-fg);
  }
  .choice-label {
    font-weight: 600;
    font-size: 0.92rem;
  }
  .choice-short {
    font-size: 0.78rem;
    color: var(--site-fg-muted);
    line-height: 1.4;
  }
  .choice.correct {
    border-color: var(--cta);
    background: color-mix(in srgb, var(--cta) 10%, transparent);
  }
  .choice.correct .choice-label { color: var(--cta); }
  .choice.wrong {
    border-color: var(--ink-coral);
    background: color-mix(in srgb, var(--ink-coral) 10%, transparent);
  }
  .choice.wrong .choice-label { color: var(--ink-coral); }
  .choice.disabled { opacity: 0.4; }

  .feedback {
    min-height: 2.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1rem;
    align-items: baseline;
    justify-content: space-between;
  }
  .fb-line {
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
    flex: 1 1 320px;
  }
  .fb-line.muted { color: var(--site-fg-muted); }
  .fb-line strong { margin-right: 0.35rem; }
  .fb-line strong.right { color: var(--cta); }
  .fb-line strong.wrong { color: var(--ink-coral); }
  .next-btn {
    border: none;
    background: var(--ink-red);
    color: #fdfdfc;
    border-radius: 999px;
    padding: 0.45rem 1rem;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.85rem;
    transition: background 160ms ease, transform 120ms ease;
  }
  .next-btn:hover {
    background: color-mix(in srgb, var(--ink-red) 88%, black);
    transform: translateY(-1px);
  }

  @media (prefers-reduced-motion: reduce) {
    .choice, .next-btn { transition: none; }
  }
</style>
