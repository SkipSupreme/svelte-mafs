<script lang="ts">
  import Mafs from "./Mafs.svelte";
  import Child from "./Mafs.probe-child.svelte";
  import type { MafsProps } from "./mafs-props.js";
  import type { CoordContext } from "../context/coordinate-context.js";

  type Props = Omit<MafsProps, "children"> & {
    onRead: (ctx: CoordContext) => void;
  };

  const { onRead, ...mafsProps }: Props = $props();
</script>

<Mafs {...mafsProps}>
  <Child {onRead} />
</Mafs>
