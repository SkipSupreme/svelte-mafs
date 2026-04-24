<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import type { UserViewBox } from "../context/coordinate-context.js";
  import type { Vec2 } from "../vec.js";
  import Polygon from "./Polygon.svelte";

  type Props = {
    points: readonly Vec2[];
    color?: string;
    opacity?: number;
    weight?: number;
    fillColor?: string;
    fillOpacity?: number;
    width?: number;
    height?: number;
    viewBox?: UserViewBox;
  };

  const {
    points,
    color,
    opacity,
    weight,
    fillColor,
    fillOpacity,
    width = 400,
    height = 300,
    viewBox = { x: [-5, 5], y: [-5, 5] },
  }: Props = $props();
</script>

<Mafs {width} {height} {viewBox}>
  <Polygon
    {points}
    {color}
    {opacity}
    {weight}
    {fillColor}
    {fillOpacity}
  />
</Mafs>
