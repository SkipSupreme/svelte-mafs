<script lang="ts">
  interface Props {
    /** Number of stacked sub-blocks. */
    depth?: number;
    /** Initial state of the residual shortcut. */
    initialResidual?: boolean;
    /** Initial activation function. */
    initialActivation?: 'tanh' | 'relu';
  }

  let {
    depth = 12,
    initialResidual = true,
    initialActivation = 'tanh',
  }: Props = $props();

  let residual: boolean = $state(initialResidual);
  let activation: 'tanh' | 'relu' = $state(initialActivation);

  // Per-layer multiplicative factor on gradient magnitude during backprop.
  // Numbers are pedagogical — chosen so the lesson's "ten to twenty orders
  // of magnitude" claim shows up at depth ~12 with no residual + tanh.
  function factor(): number {
    if (residual) {
      // Identity path dominates: ∂h_i/∂h_{i-1} ≈ I + small. Gradient roughly preserved.
      return activation === 'tanh' ? 1.02 : 1.05;
    }
    // No identity path: ∂h_i/∂h_{i-1} = W · diag(act'(z)). Saturation kills tanh harder.
    return activation === 'tanh' ? 0.20 : 0.55;
  }

  // Gradient magnitude at the input of layer k (1-indexed from the top, where
  // the loss is). At the top (loss layer), magnitude = 1 by convention.
  // At the bottom (input), magnitude = factor^depth.
  const grads = $derived.by(() => {
    const f = factor();
    const out: number[] = [];
    for (let i = 0; i < depth; i++) {
      // Layer i counted from the top (i=0 is closest to loss; i=depth-1 is input layer).
      out.push(Math.pow(f, i + 1));
    }
    return out;
  });

  // Convert gradient magnitude to a log10 bar fraction. Range [10^-25, 10^2].
  const LOG_MIN = -25;
  const LOG_MAX = 2;
  function barWidth(g: number): string {
    if (g <= 0) return '0%';
    const lg = Math.log10(g);
    const t = (lg - LOG_MIN) / (LOG_MAX - LOG_MIN);
    return `${Math.max(0, Math.min(100, t * 100)).toFixed(2)}%`;
  }
  function fmtSci(g: number): string {
    if (g <= 0) return '0';
    if (g >= 0.01 && g <= 100) return g.toFixed(3);
    return g.toExponential(1);
  }

  const inputGrad = $derived(grads[grads.length - 1]);
  const ratio = $derived(grads[0] / grads[grads.length - 1]);
  const ordersOfMag = $derived(Math.log10(ratio));

  const dead = $derived(inputGrad < 1e-6);
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">depth</span>
      <span class="meta-val">{depth}</span>
    </div>
    <div class="meta">
      <span class="meta-key">∂L/∂x at layer 1</span>
      <span class="meta-val" class:dead>{fmtSci(inputGrad)}</span>
    </div>
    <div class="meta">
      <span class="meta-key">orders of magnitude lost</span>
      <span class="meta-val" class:dead={ordersOfMag > 6}>
        {ordersOfMag >= 0 ? `−${ordersOfMag.toFixed(1)}` : `+${(-ordersOfMag).toFixed(1)}`}
      </span>
    </div>
  </header>

  <div class="stack" role="figure" aria-label="Gradient magnitude at each layer of a deep network, drawn in log scale.">
    <div class="stack-axis" aria-hidden="true">
      <span>10⁻²⁵</span>
      <span>10⁻²⁰</span>
      <span>10⁻¹⁵</span>
      <span>10⁻¹⁰</span>
      <span>10⁻⁵</span>
      <span>10⁰</span>
    </div>
    {#each grads as g, i}
      {@const layerNum = depth - i}
      <div class="row" class:loss-side={i === 0} class:input-side={i === depth - 1}>
        <span class="row-label">
          {#if i === 0}
            ↑ loss
          {:else if i === depth - 1}
            ↓ input
          {:else}
            layer {layerNum}
          {/if}
        </span>
        <div class="bar-track">
          <div
            class="bar"
            class:dead={g < 1e-6}
            class:hot={g > 10}
            style="width:{barWidth(g)};"
          ></div>
          <span class="bar-num">{fmtSci(g)}</span>
        </div>
      </div>
    {/each}
  </div>

  <div class="controls">
    <div class="ctl-row">
      <button
        type="button"
        class="btn btn-toggle"
        class:active={residual}
        onclick={() => (residual = !residual)}
      >
        {residual ? 'residuals: ON' : 'residuals: OFF'}
      </button>
      <div class="seg">
        <button
          type="button"
          class="seg-btn"
          class:active={activation === 'tanh'}
          onclick={() => (activation = 'tanh')}
        >tanh</button>
        <button
          type="button"
          class="seg-btn"
          class:active={activation === 'relu'}
          onclick={() => (activation = 'relu')}
        >ReLU</button>
      </div>
    </div>
  </div>

  <p class="caption">
    Each row is one layer; bar width is its gradient magnitude on a log₁₀
    axis. Backprop multiplies by ∂h<sub>i</sub>/∂h<sub>i−1</sub> at each
    step. Without residuals, that Jacobian shrinks the gradient at every
    layer (worse for tanh than ReLU) — by the time you reach the input,
    the signal is gone. Flip residuals ON and ∂h<sub>i</sub>/∂h<sub>i−1</sub>
    becomes <em>I + ∂F/∂x</em> ≈ <em>I</em>, and the gradient cruises
    through unchanged. The identity term in the Jacobian is the entire
    point.
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
  .meta-val.dead { color: var(--ink-coral); }

  .stack {
    background: var(--demo-stage);
    border-radius: 12px;
    padding: 0.6rem 0.85rem 0.85rem;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .stack-axis {
    display: flex;
    justify-content: space-between;
    padding: 0 0 0.45rem 5.5rem;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--site-fg-muted);
  }

  .row {
    display: grid;
    grid-template-columns: 5rem 1fr;
    gap: 0.55rem;
    align-items: center;
  }
  .row-label {
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--site-fg-muted);
    text-align: right;
  }
  .row.loss-side .row-label { color: var(--ink-sea); font-weight: 600; }
  .row.input-side .row-label { color: var(--ink-coral); font-weight: 600; }

  .bar-track {
    position: relative;
    height: 16px;
    background: color-mix(in srgb, var(--site-fg) 5%, transparent);
    border-radius: 4px;
    overflow: hidden;
  }
  .bar {
    height: 100%;
    background: linear-gradient(
      90deg,
      color-mix(in srgb, var(--ink-red) 60%, transparent),
      var(--ink-red)
    );
    border-radius: 4px;
    transition: width 220ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .bar.dead {
    background: linear-gradient(
      90deg,
      color-mix(in srgb, var(--ink-coral) 30%, transparent),
      var(--ink-coral)
    );
  }
  .bar.hot {
    background: linear-gradient(
      90deg,
      color-mix(in srgb, var(--ink-sun) 60%, transparent),
      var(--ink-sun)
    );
  }
  .bar-num {
    position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
    font-family: var(--font-mono); font-size: 0.68rem;
    color: var(--site-fg);
    text-shadow: 0 0 2px var(--demo-stage);
    font-variant-numeric: tabular-nums;
  }

  .controls { display: flex; flex-direction: column; gap: 0.4rem; }
  .ctl-row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.35rem 0.85rem;
    font-size: 0.83rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover { transform: translateY(-1px); border-color: var(--site-fg); }
  .btn-toggle.active { background: var(--cta); color: var(--cta-fg); border-color: var(--cta); }

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

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg); font-style: normal;
    font-family: var(--font-mono); font-size: 0.85em;
  }

  @media (prefers-reduced-motion: reduce) { .bar { transition: none; } }
</style>
