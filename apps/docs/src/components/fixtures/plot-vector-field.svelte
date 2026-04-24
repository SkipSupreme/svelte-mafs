<script lang="ts">
  import { Mafs, Coordinates, Plot } from "svelte-mafs";

  // Rotational field (-y, x) — sampling is deterministic on a grid so
  // visual regression stays stable run-to-run.
</script>

<Mafs width={360} height={360} viewBox={{ x: [-3, 3], y: [-3, 3] }}>
  <Coordinates.Cartesian />
  <Plot.VectorField xy={(x, y) => [-y, x]} step={0.5} />
</Mafs>
