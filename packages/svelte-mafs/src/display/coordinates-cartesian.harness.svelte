<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import CoordinatesCartesian from "./coordinates-cartesian.svelte";
  import type { UserViewBox } from "../context/coordinate-context.js";

  interface Props {
    width?: number;
    height?: number;
    viewBox?: UserViewBox;
    xAxis?: boolean;
    yAxis?: boolean;
    grid?: boolean;
    subdivisions?: number;
  }

  const {
    width = 400,
    height = 300,
    viewBox = { x: [-5, 5], y: [-5, 5] },
    xAxis,
    yAxis,
    grid,
    subdivisions,
  }: Props = $props();
</script>

<Mafs {width} {height} {viewBox}>
  <CoordinatesCartesian {xAxis} {yAxis} {grid} {subdivisions} />
</Mafs>
