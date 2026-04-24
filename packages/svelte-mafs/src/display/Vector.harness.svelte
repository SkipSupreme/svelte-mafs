<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import type { UserViewBox } from "../context/coordinate-context.js";
  import type { Vec2 } from "../vec.js";
  import Vector from "./Vector.svelte";

  type Props = {
    tail: Vec2;
    tip: Vec2;
    color?: string;
    opacity?: number;
    weight?: number;
    width?: number;
    height?: number;
    viewBox?: UserViewBox;
  };

  const {
    tail,
    tip,
    color,
    opacity,
    weight,
    width = 400,
    height = 300,
    viewBox = { x: [-5, 5], y: [-5, 5] },
  }: Props = $props();
</script>

<Mafs {width} {height} {viewBox}>
  <Vector {tail} {tip} {color} {opacity} {weight} />
</Mafs>
