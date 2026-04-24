<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import Transform from "./Transform.svelte";
  import type { Mat3 } from "./matrix.js";

  type Props = {
    matrix?: Mat3;
  };

  const { matrix }: Props = $props();
</script>

<Mafs width={400} height={400} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Transform {matrix}>
    <circle cx="0" cy="0" r="1" />
  </Transform>
</Mafs>
