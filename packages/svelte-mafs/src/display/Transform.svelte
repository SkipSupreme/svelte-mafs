<script lang="ts">
  import type { Snippet } from "svelte";
  import { identity, toMatrixString, type Mat3 } from "./matrix.js";

  type Props = {
    /**
     * Affine matrix to apply. Defaults to identity (no-op). Build with the
     * helpers re-exported alongside this component:
     *
     *   `<Transform matrix={compose(translate(3, 0), rotate(Math.PI/2))}>`
     */
    matrix?: Mat3;
    children?: Snippet;
  };

  const { matrix = identity, children }: Props = $props();

  const transform = $derived(toMatrixString(matrix));
</script>

<g transform={transform}>
  {@render children?.()}
</g>
