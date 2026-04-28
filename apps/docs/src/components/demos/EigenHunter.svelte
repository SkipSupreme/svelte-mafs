<script lang="ts">
  import { Mafs, Coordinates, Vector, MovablePoint, Line } from 'svelte-mafs';

  /**
   * EigenHunter — drag a probe vector v. The widget computes Av and shows it
   * as a ghost arrow. When v and Av point along the same line (within a
   * small tolerance), the widget cheers and reports λ = (Av · v̂) / |v|.
   *
   * Fixed matrix A = [[2, 1], [0, 3]] — eigenvalues 2 (along [1, 0]) and
   * 3 (along [1, 1]). Two findable lines, neither obvious by guessing.
   */

  const A = [
    [2, 1],
    [0, 3],
  ] as const;

  let vx = $state(1.4);
  let vy = $state(0.7);

  const Avx = $derived(A[0][0] * vx + A[0][1] * vy);
  const Avy = $derived(A[1][0] * vx + A[1][1] * vy);

  const magV = $derived(Math.hypot(vx, vy));
  const magAv = $derived(Math.hypot(Avx, Avy));

  // Cosine of angle between v and Av — for "are they collinear?"
  const cosAngle = $derived(
    magV > 0 && magAv > 0
      ? (vx * Avx + vy * Avy) / (magV * magAv)
      : 0,
  );

  // Collinear within tolerance — same OR opposite direction.
  const isEigen = $derived(Math.abs(Math.abs(cosAngle) - 1) < 0.005);

  // Signed eigenvalue ratio = projection of Av onto v / |v|² × |v| = (Av·v) / |v|².
  // Equivalently sign(cosAngle) * |Av| / |v|.
  const lambda = $derived(
    magV > 0 ? Math.sign(cosAngle) * (magAv / magV) : 0,
  );
</script>

<div class="widget">
  <header class="header">
    <p class="title">
      <em>A</em> = <span class="mat">[[2, 1], [0, 3]]</span> · drag <em>v</em>
    </p>
    <p class="hint">find a direction where <em>A</em><em>v</em> stays on the same line as <em>v</em></p>
  </header>

  <div class="stage">
    <Mafs width={560} height={380} viewBox={{ x: [-3, 3], y: [-2.4, 2.4] }}>
      <Coordinates.Cartesian />

      {#if isEigen}
        <!-- The eigenline through origin and v, drawn faintly when locked. -->
        <Line.Segment
          point1={[-3 * vx / magV, -3 * vy / magV]}
          point2={[3 * vx / magV, 3 * vy / magV]}
          color="var(--cta-hover)"
          opacity={0.35}
          weight={1.5}
        />
      {/if}

      <Vector tail={[0, 0]} tip={[Avx, Avy]} color="var(--ink-coral)" weight={2.5} opacity={0.85} />
      <Vector tail={[0, 0]} tip={[vx, vy]} color="var(--ink-red)" weight={2.75} />
      <MovablePoint bind:x={vx} bind:y={vy} color="var(--ink-red)" label="Probe vector v tip" />
    </Mafs>
  </div>

  <dl class="readout" aria-live="polite">
    <div class="row">
      <dt><span class="dot violet"></span><em>v</em></dt>
      <dd>({vx.toFixed(2)}, {vy.toFixed(2)})</dd>
    </div>
    <div class="row">
      <dt><span class="dot coral"></span><em>A</em><em>v</em></dt>
      <dd>({Avx.toFixed(2)}, {Avy.toFixed(2)})</dd>
    </div>
    <div class="row hero" class:hero-on={isEigen}>
      <dt>{isEigen ? 'eigenvector found' : 'cos∠(v, Av)'}</dt>
      <dd>
        {#if isEigen}
          <strong>λ = {lambda.toFixed(2)}</strong>
        {:else}
          {cosAngle.toFixed(3)}
        {/if}
      </dd>
    </div>
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
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-red) 45%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: .6rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .title em { font-style: italic; }
  .mat { font-family: var(--font-mono); font-size: .92rem; color: var(--site-fg-muted); }
  .hint { margin: 0; font-size: .82rem; color: var(--site-fg-muted); }
  .hint em { font-style: italic; font-family: var(--font-display); }

  .stage { width: 100%; background: var(--demo-stage); border-radius: 12px; overflow: hidden; user-select: none; touch-action: none; }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

  .readout { margin: 0; display: grid; gap: .35rem; padding: .55rem .85rem; background: color-mix(in srgb, var(--site-fg) 4%, transparent); border-radius: 10px; font-family: var(--font-mono); }
  .readout .row { display: flex; justify-content: space-between; align-items: baseline; gap: .8rem; font-size: .92rem; }
  .readout dt { color: var(--site-fg-muted); display: inline-flex; gap: .4rem; align-items: center; }
  .readout dt em { font-style: italic; font-family: var(--font-display); font-weight: 600; }
  .readout dd { margin: 0; font-variant-numeric: tabular-nums; color: var(--site-fg); }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 999px; }
  .dot.violet { background: var(--ink-red); }
  .dot.coral { background: var(--ink-coral); }
  .hero { padding-top: .35rem; border-top: 1px dashed color-mix(in srgb, var(--site-fg) 18%, transparent); margin-top: .15rem; transition: color 200ms ease; }
  .hero strong { color: var(--cta-hover); font-size: 1.1rem; }
  .hero-on dt { color: var(--cta-hover); font-weight: 600; }
</style>
