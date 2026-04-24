<script lang="ts">
  import { Mafs, Coordinates, Plot } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-4, 4], y: [-4, 4] }}>
  <Coordinates.Cartesian />
  <Plot.Parametric
    xy={(t) => [3 * Math.cos(t), 2 * Math.sin(t)]}
    t={[0, 2 * Math.PI]}
  />
  <Plot.Parametric
    xy={(t) => [Math.cos(3 * t) * 2, Math.sin(2 * t) * 2]}
    t={[0, 2 * Math.PI]}
    color="#ea580c"
  />
</Mafs>
