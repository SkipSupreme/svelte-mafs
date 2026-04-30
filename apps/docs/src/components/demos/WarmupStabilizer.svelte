<script lang="ts">
  interface Props {
    /** Total simulated training steps. */
    totalSteps?: number;
    initialMode?: 'pre-ln' | 'post-ln';
    initialWarmup?: number;
  }

  let {
    totalSteps = 100,
    initialMode = 'post-ln',
    initialWarmup = 0,
  }: Props = $props();

  let mode: 'pre-ln' | 'post-ln' = $state(initialMode);
  let warmup: number = $state(initialWarmup);

  // Conceptual model of the early-training instability:
  // - Linear warmup gives effective learning rate η_t / η_max = min(t/T_w, 1).
  // - Post-LN has an "instability amplifier" that decays as exp(-t/τ).
  // - The danger of a step is η_t/η_max × amplifier; if the worst danger
  //   exceeds a threshold, the run diverges starting a few steps later.
  // Pre-LN has no amplifier.
  const TAU = 12;
  const DANGER_THRESHOLD = 0.5;

  type Sim = { losses: number[]; diverges: boolean; divergeAt: number | null; danger: number };

  function simulate(m: 'pre-ln' | 'post-ln', w: number, T: number): Sim {
    let danger = 0;
    let dangerStep = 0;
    if (m === 'post-ln') {
      for (let t = 0; t < T; t++) {
        const eta = w === 0 ? 1 : Math.min(1, t / w);
        const instab = Math.exp(-t / TAU);
        const d = eta * instab;
        if (d > danger) { danger = d; dangerStep = t; }
      }
    }
    const diverges = m === 'post-ln' && danger > DANGER_THRESHOLD;
    const divergeAt = diverges ? Math.min(T - 1, dangerStep + 4) : null;

    const losses: number[] = [];
    for (let t = 0; t < T; t++) {
      // Base: smooth exponential decay 3.3 → 0.3 with light periodic wiggle.
      const noise = (Math.sin(t * 0.7) + Math.cos(t * 1.4)) * 0.04;
      const eta = w === 0 ? 1 : Math.min(1, t / w);
      // Warmup also slows convergence; folded in for visual honesty.
      const effectiveT = t * (0.4 + 0.6 * eta);
      const base = 0.3 + 3.0 * Math.exp(-effectiveT / 22) + noise;

      if (divergeAt != null && t >= divergeAt) {
        const explode = Math.pow(1.45, t - divergeAt);
        losses.push(Math.min(50, base + explode));
      } else {
        losses.push(base);
      }
    }
    return { losses, diverges, divergeAt, danger };
  }

  const sim: Sim = $derived(simulate(mode, warmup, totalSteps));

  // SVG layout
  const W = 460;
  const H = 200;
  const padL = 32;
  const padR = 12;
  const padT = 14;
  const padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const Y_MAX = 6;

  const path = $derived.by(() => {
    const T = sim.losses.length;
    const parts: string[] = [];
    for (let i = 0; i < T; i++) {
      const v = Math.min(sim.losses[i], Y_MAX);
      const x = padL + (i / (T - 1)) * plotW;
      const y = padT + plotH - (v / Y_MAX) * plotH;
      parts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`);
    }
    return parts.join(' ');
  });

  const warmupX = $derived(padL + (Math.min(warmup, totalSteps - 1) / (totalSteps - 1)) * plotW);
  const divergeX = $derived(
    sim.divergeAt != null
      ? padL + (sim.divergeAt / (totalSteps - 1)) * plotW
      : null,
  );

  function gridY(yVal: number): number {
    return padT + plotH - (yVal / Y_MAX) * plotH;
  }

  const finalLoss = $derived(sim.losses[sim.losses.length - 1]);
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">final loss</span>
      <span class="meta-val" class:bad={finalLoss > 5}>
        {finalLoss > 49 ? '∞ (NaN)' : finalLoss.toFixed(2)}
      </span>
    </div>
    <div class="meta">
      <span class="meta-key">survived?</span>
      <span class="meta-val" class:bad={sim.diverges} class:good={!sim.diverges}>
        {sim.diverges ? 'no — diverged' : 'yes'}
      </span>
    </div>
  </header>

  <div class="plot-wrap">
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label="Loss curve over training steps.">
      <rect x="0" y="0" width={W} height={H} fill="var(--demo-stage)" rx="8" />

      <!-- y gridlines -->
      {#each [1, 2, 3, 4, 5] as y}
        <line x1={padL} x2={W - padR} y1={gridY(y)} y2={gridY(y)}
          stroke="color-mix(in srgb, var(--site-fg) 10%, transparent)" stroke-width="0.5" />
        <text x={padL - 6} y={gridY(y) + 3} text-anchor="end" font-size="9" fill="var(--site-fg-muted)" font-family="var(--font-mono)">{y}</text>
      {/each}

      <!-- warmup region shading -->
      {#if warmup > 0}
        <rect
          x={padL}
          y={padT}
          width={warmupX - padL}
          height={plotH}
          fill="color-mix(in srgb, var(--ink-sun) 14%, transparent)"
        />
        <text
          x={(padL + warmupX) / 2}
          y={padT + 12}
          text-anchor="middle"
          font-size="9"
          fill="var(--ink-sun)"
          font-family="var(--font-mono)"
        >warmup</text>
      {/if}

      <!-- divergence marker -->
      {#if divergeX != null}
        <line
          x1={divergeX} x2={divergeX} y1={padT} y2={padT + plotH}
          stroke="var(--ink-coral)" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"
        />
        <text x={divergeX + 4} y={padT + 12} font-size="9" fill="var(--ink-coral)" font-family="var(--font-mono)">
          divergence
        </text>
      {/if}

      <!-- loss curve -->
      <path d={path} fill="none" stroke={sim.diverges ? 'var(--ink-coral)' : 'var(--ink-red)'} stroke-width="1.8" stroke-linejoin="round" />

      <!-- x-axis label -->
      <text x={W / 2} y={H - 8} text-anchor="middle" font-size="9" fill="var(--site-fg-muted)" font-family="var(--font-mono)">training steps →</text>
    </svg>
  </div>

  <div class="controls">
    <div class="ctl-row">
      <div class="seg">
        <button
          type="button"
          class="seg-btn"
          class:active={mode === 'post-ln'}
          onclick={() => (mode = 'post-ln')}
        >Post-LN</button>
        <button
          type="button"
          class="seg-btn"
          class:active={mode === 'pre-ln'}
          onclick={() => (mode = 'pre-ln')}
        >Pre-LN</button>
      </div>
    </div>
    <label class="slider">
      <span class="slider-label">linear warmup steps = <strong>{warmup}</strong></span>
      <input type="range" min="0" max="60" step="1" bind:value={warmup} aria-label="Warmup steps" />
    </label>
  </div>

  <p class="caption">
    Linear warmup ramps the learning rate from 0 to its peak over the first
    <em>T</em><sub>w</sub> steps. Post-LN's early instability — Adam's
    second-moment estimator hasn't stabilized yet — is what diverges when
    the first step is too aggressive. Warmup buys time. Pre-LN doesn't
    need any: the residual stream stays bounded by construction.
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-sun) 50%, transparent);
  }
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .meta-val.bad { color: var(--ink-coral); }
  .meta-val.good { color: var(--cta); }

  .plot-wrap { width: 100%; overflow: hidden; border-radius: 12px; }
  .plot-wrap svg { width: 100%; height: auto; display: block; }

  .controls { display: flex; flex-direction: column; gap: 0.55rem; }
  .ctl-row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
  .seg {
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
  .seg-btn.active { background: var(--ink-red); color: var(--on-color-fg); }

  .slider { display: flex; flex-direction: column; gap: 0.2rem; }
  .slider-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted); }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input[type='range'] { width: 100%; accent-color: var(--ink-sun); }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg); font-style: normal;
    font-family: var(--font-mono); font-size: 0.85em;
  }
</style>
