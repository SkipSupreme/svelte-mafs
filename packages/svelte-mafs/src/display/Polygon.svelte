<script lang="ts">
  import type { Vec2 } from "../vec.js";

  interface Props {
    points: readonly Vec2[];
    color?: string;
    opacity?: number;
    weight?: number;
    fillColor?: string;
    fillOpacity?: number;
    svg?: Record<string, string | number>;
  }

  const {
    points,
    color = "var(--mafs-fg, #333)",
    opacity = 1,
    weight = 2,
    fillColor,
    fillOpacity,
    svg,
  }: Props = $props();

  // SVG's `points` attr is space-separated "x,y" pairs. Either "x y" or "x,y"
  // works; we use "x,y x,y ..." for readability in devtools.
  const pointsAttr = $derived(
    points.map(([x, y]) => `${x},${y}`).join(" "),
  );
  const fill = $derived(fillColor ?? "none");
  const fOp = $derived(fillOpacity ?? (fillColor ? 0.15 : 1));
</script>

{#if points.length > 0}
  <polygon
    points={pointsAttr}
    stroke={color}
    stroke-opacity={opacity}
    stroke-width={weight}
    {fill}
    fill-opacity={fOp}
    vector-effect="non-scaling-stroke"
    {...svg}
  />
{/if}
