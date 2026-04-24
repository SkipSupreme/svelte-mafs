<script lang="ts">
  import { Mafs, Coordinates, Plot } from "svelte-mafs";
</script>

<!--
  Three plots in a row: well-behaved sin, pathological tan (asymptotes),
  and 1/x (singularity at 0). Adaptive sampling should handle all three
  without visual artifacts.
-->
<div style="display:flex;gap:8px;flex-wrap:wrap">
  <Mafs width={240} height={240} viewBox={{ x: [-6, 6], y: [-1.5, 1.5] }}>
    <Coordinates.Cartesian />
    <Plot.OfX y={(x) => Math.sin(x)} />
  </Mafs>

  <Mafs width={240} height={240} viewBox={{ x: [-3, 3], y: [-5, 5] }}>
    <Coordinates.Cartesian />
    <Plot.OfX y={(x) => Math.tan(x)} color="#dc2626" />
  </Mafs>

  <Mafs width={240} height={240} viewBox={{ x: [-3, 3], y: [-5, 5] }}>
    <Coordinates.Cartesian />
    <Plot.OfX y={(x) => 1 / x} color="#9333ea" />
  </Mafs>
</div>
