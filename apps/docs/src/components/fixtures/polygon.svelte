<script lang="ts">
  import { Mafs, Coordinates, Polygon } from "svelte-mafs";

  const hexagon: [number, number][] = Array.from({ length: 6 }, (_, i) => {
    const theta = (i / 6) * Math.PI * 2;
    return [3 * Math.cos(theta), 3 * Math.sin(theta)];
  });
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Polygon points={hexagon} fillColor="#60a5fa" fillOpacity={0.3} />
</Mafs>
