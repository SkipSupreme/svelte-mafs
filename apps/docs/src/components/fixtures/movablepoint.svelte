<script lang="ts">
  import { Mafs, Coordinates, MovablePoint } from "svelte-mafs";

  let x = $state(1);
  let y = $state(2);

  // Hydration marker — tests wait for this before interacting so they
  // don't race the Svelte $effect that attaches `use:drag` to the slider.
  $effect(() => {
    (window as unknown as { __fixtureHydrated: boolean }).__fixtureHydrated = true;
  });
</script>

<!--
  Interactive fixture for the MovablePoint drag/keyboard e2e spec.
  The readout below mirrors the bound state so tests can assert point
  position without reading SVG attrs.
-->
<Mafs width={400} height={400} viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <MovablePoint bind:x bind:y />
</Mafs>

<output data-testid="movable-point-readout" data-x={x} data-y={y}>
  x: {x.toFixed(4)} · y: {y.toFixed(4)}
</output>

<style>
  output {
    display: block;
    margin-top: 8px;
    font-family: ui-monospace, monospace;
    font-size: 13px;
  }
</style>
