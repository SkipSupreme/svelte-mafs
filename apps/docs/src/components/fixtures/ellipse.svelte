<script lang="ts">
  import { Mafs, Coordinates, Ellipse } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Ellipse center={[0, 0]} radius={[3, 1.5]} />
  <Ellipse center={[1, 1]} radius={[2, 1]} angle={Math.PI / 6} color="#ea580c" />
</Mafs>
