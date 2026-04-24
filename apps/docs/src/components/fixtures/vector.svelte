<script lang="ts">
  import { Mafs, Coordinates, Vector } from "svelte-mafs";
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Vector tail={[0, 0]} tip={[3, 2]} />
  <Vector tail={[0, 0]} tip={[-2, 3]} color="#16a34a" />
  <Vector tail={[1, -1]} tip={[4, -3]} color="#9333ea" />
</Mafs>
