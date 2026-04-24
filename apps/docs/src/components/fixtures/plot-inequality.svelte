<script lang="ts">
  import { Mafs, Coordinates, Plot } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-3, 3], y: [-3, 3] }}>
  <Coordinates.Cartesian />
  <Plot.Inequality y={(x) => x * x - 2} direction="above" color="#10b981" opacity={0.3} />
  <Plot.OfX y={(x) => x * x - 2} color="#10b981" />
</Mafs>
