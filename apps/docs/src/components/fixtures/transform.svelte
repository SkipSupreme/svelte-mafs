<script lang="ts">
  import { Mafs, Coordinates, Polygon, Transform, Matrix } from "svelte-mafs";

  const triangle: [number, number][] = [
    [0, 2],
    [-1, 0],
    [1, 0],
  ];

  // compose applies matrices left-to-right: first rotate 30°, then scale 1.5x.
  const composed = Matrix.compose(Matrix.rotate(Math.PI / 6), Matrix.scale(1.5));
</script>

<Mafs width={360} height={360} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <!-- baseline, no transform -->
  <Polygon points={triangle} color="#9ca3af" />
  <!-- rotated 45° -->
  <Transform matrix={Matrix.rotate(Math.PI / 4)}>
    <Polygon points={triangle} color="#2563eb" />
  </Transform>
  <!-- composed: rotate 30° then scale 1.5x -->
  <Transform matrix={composed}>
    <Polygon points={triangle} color="#dc2626" />
  </Transform>
</Mafs>
