<script lang="ts">
  import {
    getCoordContext,
    type CoordContext,
  } from "../context/coordinate-context.js";

  type Props = {
    onRead: (ctx: CoordContext) => void;
  };

  const { onRead }: Props = $props();

  // Intentional: fire once at component init with the context snapshot.
  // svelte-ignore state_referenced_locally
  onRead(getCoordContext());
</script>

<span data-probe="coord-ctx"></span>
