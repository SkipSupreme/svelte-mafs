<script lang="ts">
  import { Mafs, Coordinates, Plot, Polygon, Circle, Point, Line } from 'svelte-mafs';

  interface Props {
    /** Data minimum (where unregularized loss is zero). */
    dataMin?: [number, number];
    initialLambda?: number;
    initialPenalty?: 'l1' | 'l2';
  }

  let {
    dataMin = [1.6, 0.5],
    initialLambda = 0,
    initialPenalty = 'l2',
  }: Props = $props();

  let lambda: number = $state(initialLambda);
  let penalty: 'l1' | 'l2' = $state(initialPenalty);

  // === MAP solutions ===
  // Data loss: 1/2 ||w − w_data||^2.
  // L2 reg: λ/2 ||w||^2 → ∂/∂w gives (w − w_data) + λw = 0 → w_map = w_data / (1+λ).
  // L1 reg: λ ||w||_1 → soft-thresholding per-coord: w_i = sign(d_i) · max(0, |d_i| − λ).
  const map = $derived.by<[number, number]>(() => {
    if (penalty === 'l2') {
      return [dataMin[0] / (1 + lambda), dataMin[1] / (1 + lambda)];
    }
    const sft = (d: number) =>
      Math.sign(d) * Math.max(0, Math.abs(d) - lambda);
    return [sft(dataMin[0]), sft(dataMin[1])];
  });

  // The prior-norm level set passing through the MAP point. For visual
  // intuition: the smallest "prior ball" that still touches the data loss
  // contour through the MAP is exactly the ball of radius ||w_map||_p.
  const radiusL2 = $derived(Math.hypot(map[0], map[1]));
  const radiusL1 = $derived(Math.abs(map[0]) + Math.abs(map[1]));

  // L1 diamond polygon: |x|+|y| = r → corners at (±r, 0), (0, ±r).
  const diamond = $derived.by<[number, number][]>(() => {
    const r = radiusL1;
    return [
      [r, 0],
      [0, r],
      [-r, 0],
      [0, -r],
    ];
  });

  const dataLossAtMap = $derived(
    0.5 * ((map[0] - dataMin[0]) ** 2 + (map[1] - dataMin[1]) ** 2),
  );
  const priorLossAtMap = $derived(
    penalty === 'l2'
      ? (lambda / 2) * (map[0] ** 2 + map[1] ** 2)
      : lambda * (Math.abs(map[0]) + Math.abs(map[1])),
  );
  const totalLossAtMap = $derived(dataLossAtMap + priorLossAtMap);

  const snappedX = $derived(penalty === 'l1' && Math.abs(map[0]) < 1e-6);
  const snappedY = $derived(penalty === 'l1' && Math.abs(map[1]) < 1e-6);

  const fmt = (n: number, d = 2) =>
    Math.abs(n) < 1e-9 ? '0' : n.toFixed(d);
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">data minimum</span>
      <span class="meta-val">({fmt(dataMin[0])}, {fmt(dataMin[1])})</span>
    </div>
    <div class="meta map-meta">
      <span class="meta-key">MAP w*</span>
      <span class="meta-val">
        (<span class:snap={snappedX}>{fmt(map[0])}</span>,
        <span class:snap={snappedY}>{fmt(map[1])}</span>)
      </span>
    </div>
    <div class="meta">
      <span class="meta-key">total loss</span>
      <span class="meta-val">{fmt(totalLossAtMap, 3)}</span>
    </div>
  </header>

  <div class="stage">
    <Mafs width={460} height={360} viewBox={{ x: [-2.3, 2.6], y: [-1.6, 1.9] }}>
      <Coordinates.Cartesian
        xAxis={{ labels: (n) => n.toString() }}
        yAxis={{ labels: (n) => n.toString() }}
      />
      <!-- Origin marker -->
      <Point x={0} y={0} color="var(--site-fg-muted)" opacity={0.6} />

      <!-- Prior level set passing through MAP -->
      {#if penalty === 'l2' && radiusL2 > 0.01}
        <Circle
          center={[0, 0]}
          radius={radiusL2}
          color="var(--ink-red)"
          fillOpacity={0.07}
          strokeOpacity={0.9}
          strokeStyle="dashed"
          weight={1.5}
        />
      {:else if penalty === 'l1' && radiusL1 > 0.01}
        <Polygon
          points={diamond}
          color="var(--ink-red)"
          fillOpacity={0.07}
          strokeOpacity={0.9}
          strokeStyle="dashed"
          weight={1.5}
        />
      {/if}

      <!-- Data loss contour passing through MAP (also a circle, centered at data minimum) -->
      {#if dataLossAtMap > 0.01}
        <Circle
          center={dataMin}
          radius={Math.sqrt(2 * dataLossAtMap)}
          color="var(--ink-sea)"
          fillOpacity={0}
          strokeOpacity={0.5}
          weight={1}
        />
      {/if}

      <!-- Data minimum (blue) -->
      <Point x={dataMin[0]} y={dataMin[1]} color="var(--ink-sea)" />
      <!-- MAP solution (coral) -->
      <Point x={map[0]} y={map[1]} color="var(--ink-coral)" />

      <!-- Connector from origin to MAP -->
      {#if radiusL2 > 0.01}
        <Line.Segment
          point1={[0, 0]}
          point2={map}
          color="var(--ink-coral)"
          opacity={0.4}
          weight={1.2}
        />
      {/if}
      <!-- Trajectory from data-min to MAP -->
      <Line.Segment
        point1={dataMin}
        point2={map}
        color="var(--site-fg-muted)"
        opacity={0.45}
        weight={1}
      />
    </Mafs>
  </div>

  <div class="controls">
    <div class="ctl-row">
      <div class="seg">
        <button
          type="button"
          class="seg-btn"
          class:active={penalty === 'l2'}
          onclick={() => (penalty = 'l2')}
        >L2 (Gaussian prior)</button>
        <button
          type="button"
          class="seg-btn"
          class:active={penalty === 'l1'}
          onclick={() => (penalty = 'l1')}
        >L1 (Laplace prior)</button>
      </div>
    </div>
    <label class="slider">
      <span class="slider-label">λ = <strong>{lambda.toFixed(2)}</strong></span>
      <input type="range" min="0" max="3" step="0.05" bind:value={lambda} aria-label="Regularization strength" />
    </label>
    {#if penalty === 'l1' && (snappedX || snappedY)}
      <p class="snap-note">
        <strong>Snap.</strong>
        {#if snappedX && snappedY}
          Both coordinates of <em>w*</em> are exactly zero. Sparsity in action.
        {:else if snappedX}
          <em>w*<sub>1</sub></em> snapped to zero. The smaller coordinate of the data minimum was the first to die.
        {:else}
          <em>w*<sub>2</sub></em> snapped to zero. The smaller coordinate was 0.5; once λ &gt; 0.5, the soft-threshold killed it.
        {/if}
      </p>
    {/if}
  </div>

  <p class="caption">
    Blue dot is the data minimum (where the unregularized loss is zero).
    Coral dot is the MAP solution. Dashed shape is the prior level set passing
    through MAP — a <em>circle</em> for the Gaussian prior, a <em>diamond</em>
    for the Laplace prior. Crank λ on L1 and you'll watch the coral dot slide
    toward the nearest axis and stick. That sticking is sparsity, and the
    diamond's corners are why it happens.
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
      0 24px 48px -28px color-mix(in srgb, var(--ink-coral) 50%, transparent);
  }
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .meta-val .snap { color: var(--ink-coral); }

  .stage {
    width: 100%; background: var(--demo-stage); border-radius: 12px;
    overflow: hidden; user-select: none; -webkit-user-select: none; touch-action: none;
  }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

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
  .seg-btn.active {
    background: var(--ink-red); color: var(--on-color-fg);
  }

  .slider { display: flex; flex-direction: column; gap: 0.2rem; }
  .slider-label { font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted); }
  .slider-label strong { color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .slider input[type='range'] { width: 100%; accent-color: var(--ink-red); }

  .snap-note {
    margin: 0;
    padding: 0.45rem 0.7rem;
    border-radius: 8px;
    background: color-mix(in srgb, var(--ink-coral) 10%, transparent);
    color: var(--site-fg);
    font-size: 0.85rem;
  }
  .snap-note strong { color: var(--ink-coral); margin-right: 0.35rem; }
  .snap-note em {
    color: var(--site-fg);
    font-style: normal;
    font-family: var(--font-mono);
    font-size: 0.85em;
  }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg); font-style: normal;
    font-family: var(--font-mono); font-size: 0.85em;
  }
</style>
