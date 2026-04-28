<script lang="ts">
  import { onDestroy } from 'svelte';
  import { Mafs, Coordinates, Circle, Ellipse, Plot, Point, Line, MovablePoint, Vector } from 'svelte-mafs';

  /**
   * LossLandscapeNavigator — mandatory keystone widget for Module 10.
   *
   * Drag the START point. Pick a landscape. Pick an optimizer. Hit Run.
   * Watch the trajectory descend (or fail to, for educational reasons).
   *
   * v1 scope: 3 landscapes (bowl, ravine, saddle), 2 optimizers (SGD, momentum).
   * Adam + RMSProp + Rosenbrock are queued for v2.
   */

  type Landscape = 'bowl' | 'ravine' | 'saddle';
  type Optimizer = 'sgd' | 'momentum';

  interface LandscapeDef {
    name: string;
    f: (x: number, y: number) => number;
    grad: (x: number, y: number) => [number, number];
    levels: number[];
    drawLevel: 'circle' | 'ellipse' | 'hyperbola';
    ellipseRadii?: (c: number) => [number, number];
  }

  const LANDSCAPES: Record<Landscape, LandscapeDef> = {
    bowl: {
      name: 'convex bowl · f = x² + y²',
      f: (x, y) => x * x + y * y,
      grad: (x, y) => [2 * x, 2 * y],
      levels: [0.25, 1, 2.25, 4, 6.25],
      drawLevel: 'circle',
    },
    ravine: {
      name: 'narrow ravine · f = 0.5x² + 10y²',
      f: (x, y) => 0.5 * x * x + 10 * y * y,
      grad: (x, y) => [x, 20 * y],
      levels: [0.5, 2, 4.5, 8, 12.5],
      drawLevel: 'ellipse',
      ellipseRadii: (c) => [Math.sqrt(2 * c), Math.sqrt(c / 10)],
    },
    saddle: {
      name: 'saddle · f = x² − y²',
      f: (x, y) => x * x - y * y,
      grad: (x, y) => [2 * x, -2 * y],
      levels: [-2, -0.5, 0.5, 2],
      drawLevel: 'hyperbola',
    },
  };

  // --- state ---
  let landscape = $state<Landscape>('ravine');
  let optimizer = $state<Optimizer>('sgd');
  let lr = $state(0.05);
  let beta = $state(0.9);

  // Start point (MovablePoint). Default varies by landscape.
  let startX = $state(1.8);
  let startY = $state(0.5);

  // Current optimizer state.
  let trajectory = $state<[number, number][]>([]);
  let velX = 0;
  let velY = 0;
  let isRunning = $state(false);
  let timerId: number | null = null;
  let stepCount = $state(0);

  const MAX_STEPS = 300;
  const STEP_INTERVAL_MS = 40;

  const ldef = $derived(LANDSCAPES[landscape]);
  const current = $derived<[number, number]>(
    trajectory.length > 0 ? trajectory[trajectory.length - 1] : [startX, startY],
  );
  const curGrad = $derived(ldef.grad(current[0], current[1]));
  const curLoss = $derived(ldef.f(current[0], current[1]));
  const curGradMag = $derived(Math.hypot(curGrad[0], curGrad[1]));

  function resetState() {
    trajectory = [];
    velX = 0;
    velY = 0;
    stepCount = 0;
  }

  function stepOnce() {
    const [x, y] = current;
    const [gx, gy] = ldef.grad(x, y);

    let nx: number, ny: number;
    if (optimizer === 'sgd') {
      nx = x - lr * gx;
      ny = y - lr * gy;
    } else {
      // momentum: v ← β v + g ; w ← w − η v
      velX = beta * velX + gx;
      velY = beta * velY + gy;
      nx = x - lr * velX;
      ny = y - lr * velY;
    }

    // Bound check — stop if we've flown off the viewport.
    if (Math.abs(nx) > 6 || Math.abs(ny) > 4) {
      stop();
      return;
    }
    trajectory = [...trajectory, [nx, ny]];
    stepCount++;
    if (stepCount >= MAX_STEPS) stop();
  }

  function run() {
    if (isRunning) return;
    if (trajectory.length === 0) trajectory = [[startX, startY]];
    isRunning = true;
    timerId = window.setInterval(stepOnce, STEP_INTERVAL_MS);
  }

  function stop() {
    isRunning = false;
    if (timerId !== null) {
      clearInterval(timerId);
      timerId = null;
    }
  }

  function resetAll() {
    stop();
    resetState();
  }

  function singleStep() {
    if (isRunning) return;
    if (trajectory.length === 0) trajectory = [[startX, startY]];
    stepOnce();
  }

  // When user drags the start point OR switches landscape/optimizer, reset.
  $effect(() => {
    // Read startX, startY, landscape, optimizer — re-run effect on change.
    startX; startY; landscape; optimizer;
    resetAll();
  });

  onDestroy(stop);

  // Hyperbola parametrization for saddle: x² − y² = c.
  function hyperbolaBranch(c: number, branch: 1 | -1) {
    if (c > 0) {
      // Opens left/right. x = ±√(c + y²), y ∈ [-2, 2]
      return (t: number): [number, number] => [branch * Math.sqrt(c + t * t), t];
    } else {
      // Opens up/down. y = ±√(−c + x²), x ∈ [-3, 3]
      return (t: number): [number, number] => [t, branch * Math.sqrt(-c + t * t)];
    }
  }

  const fmt = (n: number, d = 2) => (Number.isFinite(n) ? (n >= 0 ? '+' : '') + n.toFixed(d) : '—');
</script>

<div class="widget">
  <header class="header">
    <div class="selectors">
      <label>
        <span>landscape</span>
        <select bind:value={landscape}>
          <option value="bowl">convex bowl</option>
          <option value="ravine">narrow ravine</option>
          <option value="saddle">saddle</option>
        </select>
      </label>
      <label>
        <span>optimizer</span>
        <select bind:value={optimizer}>
          <option value="sgd">SGD</option>
          <option value="momentum">SGD + momentum</option>
        </select>
      </label>
    </div>
    <p class="landscape-formula">{ldef.name}</p>
  </header>

  <div class="stage">
    <Mafs width={620} height={440} viewBox={{ x: [-3, 3], y: [-2, 2] }}>
      <Coordinates.Cartesian />

      <!-- Level curves for the selected landscape. -->
      {#if ldef.drawLevel === 'circle'}
        {#each ldef.levels as c}
          <Circle
            center={[0, 0]}
            radius={Math.sqrt(c)}
            color="var(--ink-red)"
            weight={1.2}
            opacity={0.32}
          />
        {/each}
      {:else if ldef.drawLevel === 'ellipse'}
        {#each ldef.levels as c}
          {@const radii = ldef.ellipseRadii!(c)}
          <Ellipse
            center={[0, 0]}
            radius={radii}
            color="var(--ink-red)"
            weight={1.2}
            opacity={0.32}
          />
        {/each}
      {:else if ldef.drawLevel === 'hyperbola'}
        {#each ldef.levels as c}
          {@const range = c > 0 ? ([-1.8, 1.8] as [number, number]) : ([-2.8, 2.8] as [number, number])}
          <Plot.Parametric
            xy={hyperbolaBranch(c, 1)}
            t={range}
            color="var(--ink-red)"
            weight={1.2}
            opacity={0.32}
          />
          <Plot.Parametric
            xy={hyperbolaBranch(c, -1)}
            t={range}
            color="var(--ink-red)"
            weight={1.2}
            opacity={0.32}
          />
        {/each}
      {/if}

      <!-- Trajectory as a chain of Line.Segments. -->
      {#each trajectory as pt, i}
        {#if i > 0}
          <Line.Segment
            point1={trajectory[i - 1]}
            point2={pt}
            color="var(--ink-coral)"
            weight={1.75}
            opacity={0.85}
          />
        {/if}
      {/each}

      <!-- Gradient arrow at the current point (scaled down to fit). -->
      {#if curGradMag > 0.02}
        {@const scale = Math.min(0.25, 0.8 / curGradMag)}
        <Vector
          tail={current}
          tip={[current[0] - curGrad[0] * scale, current[1] - curGrad[1] * scale]}
          color="var(--ink-sea)"
          weight={2}
          opacity={0.8}
        />
      {/if}

      <!-- Start point (draggable). -->
      <MovablePoint bind:x={startX} bind:y={startY} color="var(--ink-red)" />

      <!-- Current (coral) trajectory head, distinct from start. -->
      {#if trajectory.length > 0}
        <Point x={current[0]} y={current[1]} color="var(--ink-coral)" />
      {/if}
    </Mafs>
  </div>

  <div class="controls">
    <div class="slider-row">
      <label class="slider">
        <span class="label">
          <em>η</em> (learning rate)
          <span class="val">{lr.toFixed(3)}</span>
        </span>
        <input type="range" min="0.005" max="0.3" step="0.005" bind:value={lr} />
      </label>
      {#if optimizer === 'momentum'}
        <label class="slider">
          <span class="label">
            <em>β</em> (momentum)
            <span class="val">{beta.toFixed(2)}</span>
          </span>
          <input type="range" min="0" max="0.99" step="0.01" bind:value={beta} />
        </label>
      {/if}
    </div>
    <div class="buttons">
      {#if !isRunning}
        <button type="button" class="primary" onclick={run}>▶ Run</button>
        <button type="button" onclick={singleStep}>Step</button>
      {:else}
        <button type="button" class="primary stop" onclick={stop}>⏸ Stop</button>
      {/if}
      <button type="button" onclick={resetAll}>Reset</button>
    </div>
  </div>

  <dl class="readout">
    <div><dt>step</dt><dd>{stepCount}</dd></div>
    <div><dt>x, y</dt><dd>({fmt(current[0])}, {fmt(current[1])})</dd></div>
    <div><dt>L(x, y)</dt><dd>{fmt(curLoss, 3)}</dd></div>
    <div><dt>‖∇L‖</dt><dd>{fmt(curGradMag, 3)}</dd></div>
  </dl>
</div>

<style>
  .widget {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-coral) 45%, transparent);
  }

  .header {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .selectors {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .selectors label {
    display: inline-flex;
    flex-direction: column;
    gap: 0.25rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
  .selectors label span {
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  select {
    padding: 0.4rem 0.6rem;
    font-family: var(--font-body);
    font-size: 0.9rem;
    border: 1px solid var(--site-border);
    border-radius: 8px;
    background: var(--demo-card);
    color: var(--site-fg);
  }
  .landscape-formula {
    margin: 0;
    font-family: var(--font-mono);
    font-size: 0.86rem;
    color: var(--site-fg);
  }

  .stage {
    width: 100%;
    background: var(--demo-stage);
    border-radius: 12px;
    overflow: hidden;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
  }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

  .controls {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: center;
    padding-top: 0.4rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
  }
  .slider-row {
    flex: 1 1 16rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .slider {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.8rem;
    align-items: center;
  }
  .slider .label {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    color: var(--site-fg);
    min-width: 10rem;
  }
  .slider .label em {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 1.1em;
    color: var(--ink-coral);
  }
  .slider .val {
    margin-left: 0.6rem;
    color: var(--site-fg-muted);
    font-weight: 600;
  }
  input[type="range"] {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 6px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--site-fg) 18%, transparent);
    outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 20px; height: 20px; border-radius: 999px;
    background: var(--ink-coral);
    border: 2px solid var(--demo-card);
    cursor: grab;
  }
  input[type="range"]::-moz-range-thumb {
    width: 20px; height: 20px; border-radius: 999px;
    background: var(--ink-coral);
    border: 2px solid var(--demo-card);
    cursor: grab;
  }

  .buttons { display: flex; gap: 0.4rem; flex-wrap: wrap; }
  .buttons button {
    padding: 0.5rem 0.95rem;
    font: inherit;
    font-size: 0.88rem;
    font-weight: 600;
    border-radius: 999px;
    border: 1px solid var(--site-border);
    background: var(--demo-card);
    color: var(--site-fg);
    cursor: pointer;
  }
  .buttons button:hover { border-color: var(--ink-coral); color: var(--ink-coral); }
  .buttons .primary {
    background: var(--cta);
    border-color: var(--cta);
    color: var(--cta-fg);
  }
  .buttons .primary:hover {
    background: var(--cta-hover);
    border-color: var(--cta-hover);
    color: var(--cta-fg);
  }
  .buttons .primary.stop {
    background: var(--ink-coral);
    border-color: var(--ink-coral);
    color: var(--band-dark-fg);
  }

  .readout {
    margin: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 0.5rem 1rem;
    padding: 0.55rem 0.75rem;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-radius: 10px;
    font-family: var(--font-mono);
  }
  .readout > div { display: flex; flex-direction: column; gap: 0.1rem; }
  .readout dt {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--site-fg-muted);
  }
  .readout dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
    color: var(--site-fg);
    font-weight: 600;
  }
</style>
