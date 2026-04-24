<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import CoordinatesCartesian from "./coordinates-cartesian.svelte";
  import type { UserViewBox } from "../context/coordinate-context.js";
  import type { ComponentProps } from "svelte";

  type CCProps = ComponentProps<typeof CoordinatesCartesian>;

  interface Props extends CCProps {
    width?: number;
    height?: number;
    viewBox?: UserViewBox;
  }

  const {
    width = 400,
    height = 300,
    viewBox = { x: [-5, 5], y: [-5, 5] },
    ...coords
  }: Props = $props();
</script>

<Mafs {width} {height} {viewBox}>
  <CoordinatesCartesian {...coords} />
</Mafs>
