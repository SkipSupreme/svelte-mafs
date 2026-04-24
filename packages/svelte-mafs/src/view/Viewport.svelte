<script lang="ts">
  import type { Snippet } from "svelte";
  import {
    normalizeViewBox,
    type UserViewBox,
  } from "../context/coordinate-context.js";
  import { setPaneContext } from "../context/pane-context.js";

  type Props = {
    viewBox: UserViewBox;
    children?: Snippet;
  };

  const { viewBox: propViewBox, children }: Props = $props();

  // Stub: the full nested-viewport feature (its own <svg>/<clipPath>, remapped
  // user-to-px transforms, sub-tree isolation) lands in a later stream. For
  // now, Viewport just publishes a pane override so future sub-graphs have a
  // place to read their local viewBox from. Children render inline.
  //
  // setPaneContext can only run once during init, so we publish an object
  // whose viewBox is a getter over a $derived — prop changes flow through
  // without needing to re-publish.
  let paneViewBox = $derived(normalizeViewBox(propViewBox));
  setPaneContext({
    get viewBox() {
      return paneViewBox;
    },
  });
</script>

{@render children?.()}
