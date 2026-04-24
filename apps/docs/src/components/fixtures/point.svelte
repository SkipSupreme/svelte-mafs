<script lang="ts">
  import { Mafs, Coordinates, Point } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Point x={2} y={1} />
  <Point x={-3} y={-2} color="#2563eb" />
  <Point x={0} y={3} color="#dc2626" />
</Mafs>
