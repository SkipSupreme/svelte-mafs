<script lang="ts">
  import { Mafs, Coordinates, Line } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Line.ThroughPoints point1={[-2, -1]} point2={[2, 1]} />
  <Line.ThroughPoints point1={[0, -4]} point2={[1, 0]} color="#9333ea" />
</Mafs>
