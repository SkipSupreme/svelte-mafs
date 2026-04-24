<script lang="ts">
  import { Mafs, Coordinates, Circle } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Circle center={[0, 0]} radius={3} />
  <Circle center={[2, 2]} radius={1} color="#2563eb" />
</Mafs>
