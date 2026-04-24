<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import type { UserViewBox } from "../context/coordinate-context.js";
  import type { Vec2 } from "../vec.js";
  import Ellipse from "./Ellipse.svelte";

  type Props = {
    center: Vec2;
    radius: Vec2;
    angle?: number;
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
    center,
    radius,
    angle,
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
  <Ellipse
    {center}
    {radius}
    {angle}
    {color}
    {opacity}
    {weight}
    {fillColor}
    {fillOpacity}
  />
</Mafs>
