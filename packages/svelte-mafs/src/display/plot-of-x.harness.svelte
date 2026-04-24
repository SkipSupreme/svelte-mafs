<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import PlotOfX from "./plot-of-x.svelte";
  import type { UserViewBox } from "../context/coordinate-context.js";
  import type { ComponentProps } from "svelte";

  interface Props {
    width?: number;
    height?: number;
    viewBox?: UserViewBox;
    plot: ComponentProps<typeof PlotOfX>;
  }

  const {
    width = 400,
    height = 300,
    viewBox = { x: [-5, 5], y: [-5, 5] },
    plot,
  }: Props = $props();
</script>

<Mafs {width} {height} {viewBox}>
  <PlotOfX {...plot} />
</Mafs>
