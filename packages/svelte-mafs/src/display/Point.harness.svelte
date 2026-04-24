<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import type { UserViewBox } from "../context/coordinate-context.js";
  import Point from "./Point.svelte";

  type Props = {
    x: number;
    y: number;
    color?: string;
    opacity?: number;
    width?: number;
    height?: number;
    viewBox?: UserViewBox;
  };

  const {
    x,
    y,
    color,
    opacity,
    width = 400,
    height = 300,
    viewBox = { x: [-5, 5], y: [-5, 5] },
  }: Props = $props();
</script>

<Mafs {width} {height} {viewBox}>
  <Point {x} {y} {color} {opacity} />
</Mafs>
