<!--
  Temporary Stream 5 preview. Stream 7 Wave B replaces this with a per-
  component examples set (one route each for Coordinates, Plot.OfX, etc.).
-->
<script lang="ts">
  import { Mafs, Coordinates, Plot } from "svelte-mafs";
</script>

<div class="stream5-grid">
  <figure>
    <Mafs width={360} height={270} viewBox={{ x: [-2 * Math.PI, 2 * Math.PI], y: [-1.5, 1.5] }}>
      <Coordinates.Cartesian />
      <Plot.OfX y={(x: number) => Math.sin(x)} />
    </Mafs>
    <figcaption>Plot.OfX — <code>y = sin(x)</code></figcaption>
  </figure>

  <figure>
    <Mafs width={360} height={270} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
      <Coordinates.Cartesian />
      <Plot.Parametric
        xy={(t: number) => [3 * Math.cos(t), 2 * Math.sin(t)]}
        t={[0, 2 * Math.PI]}
        color="var(--site-accent, #d946ef)"
      />
    </Mafs>
    <figcaption>Plot.Parametric — ellipse <code>(3 cos t, 2 sin t)</code></figcaption>
  </figure>

  <figure>
    <Mafs width={360} height={270} viewBox={{ x: [-3, 3], y: [-3, 3] }}>
      <Coordinates.Cartesian />
      <Plot.VectorField xy={(x: number, y: number) => [-y, x]} step={0.5} />
    </Mafs>
    <figcaption>Plot.VectorField — rotational <code>(-y, x)</code></figcaption>
  </figure>

  <figure>
    <Mafs width={360} height={270} viewBox={{ x: [-3, 3], y: [-3, 3] }}>
      <Coordinates.Cartesian />
      <Plot.Inequality y={(x: number) => x * x - 2} color="#10b981" opacity={0.25} />
      <Plot.OfX y={(x: number) => x * x - 2} color="#10b981" />
    </Mafs>
    <figcaption>Plot.Inequality — <code>y &gt; x² − 2</code></figcaption>
  </figure>
</div>

<style>
  .stream5-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
    margin: 1.5rem 0;
  }
  figure {
    margin: 0;
    padding: 0.75rem;
    border: 1px solid var(--site-border);
    border-radius: var(--radius-md);
    background: var(--site-surface);
  }
  figure :global(svg) {
    display: block;
    max-width: 100%;
    height: auto;
    background: var(--site-bg);
    border-radius: var(--radius-sm);
  }
  figcaption {
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: var(--site-fg-muted);
  }
  figcaption :global(code) {
    font-family: var(--font-mono, ui-monospace, monospace);
  }
</style>
