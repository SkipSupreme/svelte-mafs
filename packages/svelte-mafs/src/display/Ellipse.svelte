<script lang="ts">
  import type { Vec2 } from "../vec.js";

  interface Props {
    center: Vec2;
    /** [rx, ry] in user-space units. */
    radius: Vec2;
    /** Rotation in radians, counter-clockwise (math convention). */
    angle?: number;
    color?: string;
    opacity?: number;
    weight?: number;
    fillColor?: string;
    fillOpacity?: number;
    svg?: Record<string, string | number>;
  }

  const {
    center,
    radius,
    angle = 0,
    color = "var(--mafs-fg, #333)",
    opacity = 1,
    weight = 2,
    fillColor,
    fillOpacity,
    svg,
  }: Props = $props();

  // SVG rotate(α) is in degrees. Inside the parent <g scale(1,-1)>, SVG-CW
  // and math-CCW coincide, so we pass the same numeric magnitude.
  const deg = $derived((angle * 180) / Math.PI);
  const fill = $derived(fillColor ?? "none");
  const fOp = $derived(fillOpacity ?? (fillColor ? 0.15 : 1));
</script>

<ellipse
  cx={center[0]}
  cy={center[1]}
  rx={radius[0]}
  ry={radius[1]}
  transform="rotate({deg} {center[0]} {center[1]})"
  stroke={color}
  stroke-opacity={opacity}
  stroke-width={weight}
  {fill}
  fill-opacity={fOp}
  vector-effect="non-scaling-stroke"
  {...svg}
/>
