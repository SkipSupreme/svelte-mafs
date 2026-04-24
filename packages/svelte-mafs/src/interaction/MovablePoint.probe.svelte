<script lang="ts">
  import Mafs from "../view/Mafs.svelte";
  import MovablePoint from "./MovablePoint.svelte";
  import type { Constraint } from "./constraints.js";

  type Props = {
    initialX: number;
    initialY: number;
    constrain?: Constraint;
    step?: number;
    onChange?: (x: number, y: number) => void;
  };

  const { initialX, initialY, constrain, step, onChange }: Props = $props();

  // svelte-ignore state_referenced_locally
  let x = $state(initialX);
  // svelte-ignore state_referenced_locally
  let y = $state(initialY);

  // Fires once on mount with the initial pair, then again for every update —
  // tests can assert lastCall's args.
  $effect(() => {
    onChange?.(x, y);
  });
</script>

<Mafs width={400} height={400} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <MovablePoint bind:x bind:y {constrain} {step} />
</Mafs>
