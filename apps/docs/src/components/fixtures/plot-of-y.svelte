<script lang="ts">
  import { Mafs, Coordinates, Plot } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Plot.OfY x={(y) => y * y - 2} />
  <Plot.OfY x={(y) => Math.cos(y) * 2} color="#2563eb" />
</Mafs>
