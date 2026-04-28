<script lang="ts">
  import { Mafs, Coordinates, MovablePoint, Vector } from 'svelte-mafs';

  interface Props {
    /** Show |u|, |v|, cos θ alongside the headline u·v. Off for the
        opening "feel the number" hook; on for the geometric-formula step. */
    showDecomposition?: boolean;
  }

  let { showDecomposition = false }: Props = $props();

  let ux = $state(3);
  let uy = $state(1);
  let vx = $state(2);
  let vy = $state(2);

  const dot = $derived(ux * vx + uy * vy);
  const magU = $derived(Math.hypot(ux, uy));
  const magV = $derived(Math.hypot(vx, vy));
  const cosTheta = $derived(
    magU > 0 && magV > 0 ? dot / (magU * magV) : 0,
  );
  // clamp before acos — float error can push cos slightly past ±1.
  const thetaDeg = $derived(
    magU > 0 && magV > 0
      ? (Math.acos(Math.max(-1, Math.min(1, cosTheta))) * 180) / Math.PI
      : 0,
  );

  const sign = $derived(
    Math.abs(dot) < 0.005 ? 'zero' : dot > 0 ? 'pos' : 'neg',
  );

  const fmt = (n: number) => {
    if (Math.abs(n) < 0.005) return '0.00';
    return (n > 0 ? '+' : '') + n.toFixed(2);
  };
</script>

<div class="widget">
  <div class="stage">
    <Mafs width={560} height={360} viewBox={{ x: [-4, 4], y: [-3, 3] }}>
      <Coordinates.Cartesian />
      <Vector tail={[0, 0]} tip={[ux, uy]} color="var(--ink-red)" weight={2.75} />
      <Vector tail={[0, 0]} tip={[vx, vy]} color="var(--ink-sea)" weight={2.75} />
      <MovablePoint bind:x={ux} bind:y={uy} color="var(--ink-red)" label="Vector u tip" />
      <MovablePoint bind:x={vx} bind:y={vy} color="var(--ink-sea)" label="Vector v tip" />
    </Mafs>
  </div>

  <div class="readout" aria-live="polite">
    <div class="hero hero-{sign}">
      <span class="hero-label"><em>u</em> · <em>v</em></span>
      <span class="hero-value">{fmt(dot)}</span>
    </div>

    {#if showDecomposition}
      <dl class="decomp">
        <div class="row">
          <dt>|<em>u</em>|</dt>
          <dd>{magU.toFixed(2)}</dd>
        </div>
        <div class="row">
          <dt>|<em>v</em>|</dt>
          <dd>{magV.toFixed(2)}</dd>
        </div>
        <div class="row">
          <dt>cos θ</dt>
          <dd>{cosTheta.toFixed(3)}</dd>
        </div>
        <div class="row">
          <dt>θ</dt>
          <dd>{thetaDeg.toFixed(1)}°</dd>
        </div>
        <div class="row check">
          <dt>|<em>u</em>| |<em>v</em>| cos θ</dt>
          <dd>{fmt(magU * magV * cosTheta)}</dd>
        </div>
      </dl>
    {/if}

    <p class="hint">drag either arrow's dot — watch the number</p>
  </div>
</div>

<style>
  .widget {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 55%, transparent);
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
  .stage :global(svg) {
    display: block;
    width: 100%;
    height: auto;
    max-width: 100%;
  }

  .readout {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    color: var(--site-fg);
  }

  /* Hero readout — the "alignment number" the lesson is teaching. */
  .hero {
    display: grid;
    grid-template-columns: auto 1fr;
    align-items: baseline;
    gap: 1rem;
    padding: 0.5rem 0.85rem;
    border-radius: 12px;
    border-left: 4px solid var(--site-fg-muted);
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    transition: border-color 220ms ease, background 220ms ease;
  }
  .hero-label {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .hero-label em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
    text-transform: none;
    letter-spacing: 0;
    font-size: 1rem;
  }
  .hero-value {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    justify-self: end;
    transition: color 220ms ease;
  }
  .hero-pos {
    border-left-color: var(--cta-hover);
    background: color-mix(in srgb, var(--cta-hover) 8%, transparent);
  }
  .hero-pos .hero-value { color: var(--cta-hover); }
  .hero-neg {
    border-left-color: var(--ink-coral);
    background: color-mix(in srgb, var(--ink-coral) 8%, transparent);
  }
  .hero-neg .hero-value { color: var(--ink-coral); }
  .hero-zero .hero-value { color: var(--site-fg-muted); }

  /* Decomposition table — only when showDecomposition is on. */
  .decomp {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.25rem 1rem;
    margin: 0;
    padding: 0.4rem 0.85rem 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.9rem;
  }
  .decomp .row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.6rem;
  }
  .decomp dt {
    color: var(--site-fg-muted);
  }
  .decomp dt em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }
  .decomp dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
  }
  .decomp .row.check {
    grid-column: 1 / -1;
    padding-top: 0.35rem;
    border-top: 1px dashed color-mix(in srgb, var(--site-fg) 22%, transparent);
    margin-top: 0.15rem;
  }
  .decomp .row.check dt,
  .decomp .row.check dd {
    color: var(--site-fg);
    font-weight: 600;
  }

  .hint {
    margin: 0;
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--site-fg-muted);
  }
</style>
