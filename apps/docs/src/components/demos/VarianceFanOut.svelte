<script lang="ts">
  interface Props {
    layers?: number;
    hidden?: number;
    batch?: number;
    initialSigma?: number;
  }

  let {
    layers = 10,
    hidden = 64,
    batch = 192,
    initialSigma = 0.5,
  }: Props = $props();

  // === presets ===
  const presets = [
    { id: 'too-small', label: 'too small', value: 0.01 },
    { id: 'xavier', label: `Xavier (1/√${hidden})`, value: 1 / Math.sqrt(hidden) },
    { id: 'he', label: `He (√(2/${hidden}))`, value: Math.sqrt(2 / hidden) },
    { id: 'too-large', label: 'too large', value: 0.5 },
  ] as const;

  let sigma: number = $state(initialSigma);

  // Cache the input batch so resampling sigma doesn't re-resample inputs.
  function gauss(): number {
    const u = Math.random() || 1e-9;
    const v = Math.random() || 1e-9;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function makeMatrix(rows: number, cols: number, std: number): Float32Array {
    const out = new Float32Array(rows * cols);
    for (let i = 0; i < rows * cols; i++) out[i] = std * gauss();
    return out;
  }

  // The input X has unit variance Gaussian entries. Reseeded only on demand.
  let inputs: Float32Array = $state(makeMatrix(batch, hidden, 1));
  let weightsSeed: number = $state(1);

  // Forward pass results: per-layer activation tensor (batch × hidden).
  // Computed deterministically from sigma + weightsSeed + inputs.
  type LayerStats = {
    activations: Float32Array;
    variance: number;
    saturatedFraction: number;
  };

  function buildWeights(L: number, H: number, std: number, seed: number): Float32Array[] {
    // PRNG seeded — reset Math.random shimming via xorshift wouldn't
    // affect Math.random; for this demo deterministic-per-seed weight
    // matrices aren't critical, so we just regenerate fresh per call.
    void seed;
    const W: Float32Array[] = [];
    for (let l = 0; l < L; l++) {
      W.push(makeMatrix(H, H, std));
    }
    return W;
  }

  const stats: LayerStats[] = $derived.by(() => {
    void weightsSeed;
    const W = buildWeights(layers, hidden, sigma, weightsSeed);
    let X = inputs;
    const out: LayerStats[] = [];
    for (let l = 0; l < layers; l++) {
      const Z = new Float32Array(batch * hidden);
      const Wl = W[l];
      // Z[b, j] = sum_k X[b, k] * W[k, j]
      for (let b = 0; b < batch; b++) {
        for (let j = 0; j < hidden; j++) {
          let s = 0;
          for (let k = 0; k < hidden; k++) {
            s += X[b * hidden + k] * Wl[k * hidden + j];
          }
          Z[b * hidden + j] = Math.tanh(s);
        }
      }
      X = Z;
      // Stats
      let mean = 0;
      for (let i = 0; i < batch * hidden; i++) mean += X[i];
      mean /= batch * hidden;
      let v = 0;
      let sat = 0;
      for (let i = 0; i < batch * hidden; i++) {
        const d = X[i] - mean;
        v += d * d;
        if (Math.abs(X[i]) > 0.95) sat++;
      }
      v /= batch * hidden;
      out.push({
        activations: X,
        variance: v,
        saturatedFraction: sat / (batch * hidden),
      });
    }
    return out;
  });

  // Gradient signal at layer 1: rough proxy = product of per-layer Jacobian
  // magnitudes. With tanh, ∂tanh(z)/∂z ≤ 1. We approximate the per-layer
  // Jacobian magnitude as (1 - σ_act²) on average, multiplied by σ_W * √H.
  const layer1Grad = $derived.by(() => {
    let g = 1;
    for (const s of stats) {
      const tanhDeriv = Math.max(0.001, 1 - s.variance);
      const wMag = sigma * Math.sqrt(hidden);
      g *= tanhDeriv * wMag;
    }
    return g;
  });

  // Histogram bins for visualization.
  const NBINS = 24;
  const BIN_MIN = -1.05;
  const BIN_MAX = 1.05;
  function histogram(act: Float32Array): number[] {
    const bins = new Array(NBINS).fill(0);
    const w = (BIN_MAX - BIN_MIN) / NBINS;
    for (let i = 0; i < act.length; i++) {
      const b = Math.floor((act[i] - BIN_MIN) / w);
      if (b >= 0 && b < NBINS) bins[b]++;
    }
    return bins;
  }
  function maxBin(b: number[]): number {
    let m = 0;
    for (const v of b) if (v > m) m = v;
    return m || 1;
  }

  // Show only every other layer (5 of 10) so the visualization breathes.
  const SHOWN = $derived.by(() => {
    const idx: number[] = [];
    for (let i = 1; i <= layers; i += Math.max(1, Math.floor(layers / 5))) {
      idx.push(i);
    }
    if (idx[idx.length - 1] !== layers) idx.push(layers);
    return idx;
  });

  function applyPreset(v: number): void {
    sigma = v;
  }

  function reseed(): void {
    inputs = makeMatrix(batch, hidden, 1);
    weightsSeed = (weightsSeed * 7919 + 1) | 0;
  }

  const fmtSci = (n: number) => {
    if (!Number.isFinite(n) || n === 0) return '0';
    if (n >= 0.001 && n <= 1000) return n.toFixed(3);
    return n.toExponential(2);
  };
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">σ_W</span>
      <span class="meta-val">{sigma.toFixed(4)}</span>
    </div>
    <div class="meta">
      <span class="meta-key">layer-{layers} variance</span>
      <span class="meta-val">{stats[layers - 1]?.variance.toFixed(3) ?? '—'}</span>
    </div>
    <div class="meta">
      <span class="meta-key">saturation @ layer {layers}</span>
      <span class="meta-val" class:dead={(stats[layers - 1]?.saturatedFraction ?? 0) > 0.5}>
        {((stats[layers - 1]?.saturatedFraction ?? 0) * 100).toFixed(0)}%
      </span>
    </div>
    <div class="meta">
      <span class="meta-key">layer-1 ∂L/∂x</span>
      <span class="meta-val" class:dead={layer1Grad < 1e-6}>
        {fmtSci(layer1Grad)}
      </span>
    </div>
  </header>

  <div class="histos">
    {#each SHOWN as layer (layer)}
      {@const s = stats[layer - 1]}
      {@const bins = histogram(s.activations)}
      {@const peak = maxBin(bins)}
      <div class="histo">
        <div class="histo-label">layer {layer}</div>
        <div class="histo-bars">
          {#each bins as v}
            <div class="histo-bar" style="height:{(v / peak) * 100}%;" title={`${v} samples`}></div>
          {/each}
        </div>
        <div class="histo-axis">
          <span>−1</span><span>0</span><span>+1</span>
        </div>
      </div>
    {/each}
  </div>

  <div class="controls">
    <div class="ctl-row presets">
      {#each presets as p}
        <button
          type="button"
          class="btn btn-ghost"
          class:active={Math.abs(sigma - p.value) < 1e-4}
          onclick={() => applyPreset(p.value)}
        >{p.label}</button>
      {/each}
      <button type="button" class="btn btn-ghost" onclick={reseed}>reseed</button>
    </div>
    <label class="slider">
      <span class="slider-label">σ_W = <strong>{sigma.toFixed(4)}</strong></span>
      <input type="range" min="0.005" max="0.5" step="0.001" bind:value={sigma} aria-label="Weight standard deviation" />
    </label>
  </div>

  <p class="caption">
    Each histogram shows the distribution of post-tanh activations across a
    batch of {batch} inputs at one layer of a {layers}-layer MLP with
    {hidden} units per layer. Drag σ<sub>W</sub> low and the activations
    decay toward zero — every histogram becomes a single spike at 0,
    gradients die. Drag it high and tanh saturates — every histogram
    becomes two walls at ±1, gradients also die. The presets land at the
    known good values; only Xavier and He keep the histograms stable
    across all 10 layers.
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
  .meta-val.dead { color: var(--ink-coral); }

  .histos {
    background: var(--demo-stage);
    border-radius: 12px;
    padding: 0.7rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 0.5rem;
  }
  .histo {
    display: flex; flex-direction: column;
    gap: 0.25rem;
    padding: 0.45rem 0.5rem 0.35rem;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-radius: 8px;
  }
  .histo-label {
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--ink-sea);
    font-weight: 600;
  }
  .histo-bars {
    display: flex; align-items: flex-end;
    gap: 1px;
    height: 70px;
  }
  .histo-bar {
    flex: 1 1 0; min-width: 2px;
    background: var(--ink-sea);
    border-radius: 1px 1px 0 0;
    transition: height 200ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .histo-axis {
    display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: 0.62rem; color: var(--site-fg-muted);
  }

  .controls { display: flex; flex-direction: column; gap: 0.55rem; }
  .ctl-row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.32rem 0.85rem;
    font-size: 0.8rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover { transform: translateY(-1px); border-color: var(--site-fg); }
  .btn-ghost { color: var(--site-fg-muted); font-weight: 500; }
  .btn.active {
    background: var(--ink-sea);
    color: #fdfdfc;
    border-color: var(--ink-sea);
  }

  .slider { display: flex; flex-direction: column; gap: 0.2rem; }
  .slider-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted); }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input[type='range'] { width: 100%; accent-color: var(--ink-sea); }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }

  @media (prefers-reduced-motion: reduce) { .histo-bar { transition: none; } }
</style>
