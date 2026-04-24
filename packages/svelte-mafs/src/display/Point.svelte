<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";

  interface Props {
    x: number;
    y: number;
    /** Fill color. Defaults to `var(--mafs-fg)` once Stream 6 ships theme.css. */
    color?: string;
    /** Fill opacity, 0–1. */
    opacity?: number;
    /** Escape hatch for arbitrary SVG attributes on the underlying <circle>. */
    svg?: Record<string, string | number>;
  }

  const {
    x,
    y,
    color = "var(--mafs-fg, #333)",
    opacity = 1,
    svg,
  }: Props = $props();

  const ctx = getCoordContext();

  // Target pixel radius for the dot. Uniform preserveAspectRatio means both
  // axes end up at the same effective px/unit scale, so `widthPx / xSpan`
  // is representative. See math.ts nearestPowerOfTen / mapRange for the
  // general tick-inference pattern; here we just need one scalar.
  const RADIUS_PX = 6;
  const r = $derived.by(() => {
    const xSpan = ctx.viewBox.xMax - ctx.viewBox.xMin;
    const ySpan = ctx.viewBox.yMax - ctx.viewBox.yMin;
    // `min` matches `preserveAspectRatio: xMidYMid meet` — whichever axis
    // is tighter sets the effective scale.
    const scale = Math.min(ctx.widthPx / xSpan, ctx.heightPx / ySpan);
    return scale > 0 ? RADIUS_PX / scale : 0;
  });
</script>

<circle
  cx={x}
  cy={y}
  {r}
  fill={color}
  fill-opacity={opacity}
  {...svg}
/>
