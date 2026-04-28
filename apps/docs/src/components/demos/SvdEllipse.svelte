<script lang="ts">
  import { Mafs, Coordinates, Plot, Vector, Point } from 'svelte-mafs';

  /**
   * SvdEllipse — every matrix is rotate-stretch-rotate. Pick singular
   * values σ₁, σ₂ and two rotations Vᵀ and U; the widget composes
   * A = U Σ Vᵀ and shows the unit circle progressively transformed.
   *
   * Animation slider t ∈ [0, 3]:
   *   0–1 : apply Vᵀ (rotation, no shape change)
   *   1–2 : apply Σ (axis-aligned stretch, becomes ellipse)
   *   2–3 : apply U (final rotation)
   *
   * Drop σ₂ to zero — ellipse collapses to a line. Best rank-1 approx.
   */

  let sigma1 = $state(1.8);
  let sigma2 = $state(0.7);
  let thetaV = $state(-0.4);
  let thetaU = $state(0.6);
  let t = $state(2.6);

  // Rotation matrix R(θ) applied as [[c, -s], [s, c]] · v.
  function rot(angle: number, v: [number, number]): [number, number] {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [c * v[0] - s * v[1], s * v[0] + c * v[1]];
  }

  // The transformation applied at progress t (0–3) to a unit-circle point.
  function transform(theta0: number): [number, number] {
    let p: [number, number] = [Math.cos(theta0), Math.sin(theta0)];
    // Stage 1: apply Vᵀ. Vᵀ rotates by -thetaV.
    if (t <= 0) return p;
    const s1 = Math.min(1, t);
    p = rot(-thetaV * s1, p);
    if (t <= 1) return p;
    // Stage 2: stretch along x and y.
    const s2 = Math.min(1, t - 1);
    const stretchX = 1 + (sigma1 - 1) * s2;
    const stretchY = 1 + (sigma2 - 1) * s2;
    p = [p[0] * stretchX, p[1] * stretchY];
    if (t <= 2) return p;
    // Stage 3: apply U.
    const s3 = Math.min(1, t - 2);
    p = rot(thetaU * s3, p);
    return p;
  }

  // Curve as parametric.
  const curve = (theta: number) => transform(theta);

  // Stage label.
  const stage = $derived(
    t < 1 ? '1: Vᵀ rotation' : t < 2 ? '2: Σ stretch' : '3: U rotation',
  );

  // Compute the singular vectors as final-image basis arrows.
  const u1 = $derived(rot(thetaU, [sigma1, 0]));
  const u2 = $derived(rot(thetaU, [0, sigma2]));
</script>

<div class="widget">
  <header class="header">
    <p class="title">unit circle → <em>A</em> = <em>U</em>Σ<em>V</em>ᵀ</p>
    <span class="stage-badge">stage {stage}</span>
  </header>

  <div class="stage">
    <Mafs width={500} height={360} viewBox={{ x: [-3, 3], y: [-2.4, 2.4] }}>
      <Coordinates.Cartesian />

      <!-- Original unit circle, faint reference. -->
      <Plot.Parametric xy={(theta) => [Math.cos(theta), Math.sin(theta)]} t={[0, 2 * Math.PI]} color="var(--site-fg-muted)" weight={1} opacity={0.35} />

      <!-- Currently-transformed shape. -->
      <Plot.Parametric xy={curve} t={[0, 2 * Math.PI]} color="var(--ink-red)" weight={2.5} />

      <!-- Final singular vectors at t = 3 (visible only in stage 3+). -->
      {#if t >= 2}
        <Vector tail={[0, 0]} tip={u1} color="var(--ink-coral)" weight={2.25} opacity={Math.min(1, (t - 2) * 1.5 + 0.3)} />
        <Vector tail={[0, 0]} tip={u2} color="var(--ink-sea)" weight={2.25} opacity={Math.min(1, (t - 2) * 1.5 + 0.3)} />
      {/if}

      <Point x={0} y={0} color="var(--ink-red)" />
    </Mafs>
  </div>

  <div class="controls">
    <label class="full">
      <span>animate <em>t</em>: {t.toFixed(2)}</span>
      <input type="range" min="0" max="3" step="0.02" bind:value={t} aria-label="Animation progress" />
    </label>
    <label>
      <span>σ₁ = {sigma1.toFixed(2)}</span>
      <input type="range" min="0.5" max="2.5" step="0.05" bind:value={sigma1} aria-label="Sigma 1" />
    </label>
    <label>
      <span>σ₂ = {sigma2.toFixed(2)}</span>
      <input type="range" min="0" max="1.8" step="0.02" bind:value={sigma2} aria-label="Sigma 2" />
    </label>
    <label>
      <span>θ<sub>V</sub> = {(thetaV * 180 / Math.PI).toFixed(0)}°</span>
      <input type="range" min={-Math.PI} max={Math.PI} step="0.02" bind:value={thetaV} aria-label="V rotation" />
    </label>
    <label>
      <span>θ<sub>U</sub> = {(thetaU * 180 / Math.PI).toFixed(0)}°</span>
      <input type="range" min={-Math.PI} max={Math.PI} step="0.02" bind:value={thetaU} aria-label="U rotation" />
    </label>
  </div>

  <p class="note">
    Push σ₂ to zero. The ellipse collapses to a line — that's the best rank-1 approximation of <em>A</em>. Every matrix factors this way: rotate, stretch, rotate.
  </p>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: .85rem;
    background: var(--demo-card); border: 1px solid var(--demo-card-border);
    border-radius: 20px; padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-coral) 45%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: .5rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .title em { font-style: italic; }
  .stage-badge { padding: .25rem .65rem; border-radius: 999px; font-family: var(--font-mono); font-size: .78rem; background: color-mix(in srgb, var(--ink-coral) 18%, transparent); color: var(--ink-coral); }

  .stage { width: 100%; background: var(--demo-stage); border-radius: 12px; padding: .35rem; touch-action: none; }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

  .controls { display: grid; grid-template-columns: 1fr 1fr; gap: .45rem .8rem; padding-top: .5rem; border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent); }
  @media (max-width: 600px) { .controls { grid-template-columns: 1fr; } }
  .controls label { display: grid; grid-template-columns: 7rem 1fr; gap: .6rem; align-items: center; font-family: var(--font-mono); font-size: .82rem; }
  .controls label.full { grid-column: 1 / -1; grid-template-columns: 9rem 1fr; }
  .controls em { font-style: italic; font-family: var(--font-display); }
  input[type="range"] { -webkit-appearance: none; appearance: none; width: 100%; height: 6px; border-radius: 999px; background: color-mix(in srgb, var(--site-fg) 18%, transparent); outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 20px; height: 20px; border-radius: 999px; background: var(--ink-coral); border: 2px solid var(--demo-card); cursor: grab; }
  input[type="range"]::-moz-range-thumb { width: 20px; height: 20px; border-radius: 999px; background: var(--ink-coral); border: 2px solid var(--demo-card); cursor: grab; }

  .note { margin: 0; padding: .55rem .85rem; background: color-mix(in srgb, var(--site-fg) 4%, transparent); border-radius: 10px; font-family: var(--font-body); font-size: .82rem; color: var(--site-fg); }
  .note em { font-style: italic; }
</style>
