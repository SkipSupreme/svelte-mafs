<script lang="ts">
  import { Mafs, Coordinates, Line } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Line.Segment point1={[-3, -2]} point2={[3, 2]} />
  <Line.Segment point1={[-2, 3]} point2={[2, -3]} color="#16a34a" />
</Mafs>
