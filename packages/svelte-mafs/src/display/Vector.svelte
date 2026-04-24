<script module lang="ts">
  // Monotonic counter for unique arrowhead marker IDs. Shared across all
  // <Vector> instances in the page — first render gets -1, next -2, etc.
  // Not cryptographically unique, just collision-free within a process.
  let counter = 0;
  const nextId = () => `svelte-mafs-arrow-${++counter}`;
</script>

<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";
  import type { Vec2 } from "../vec.js";

  interface Props {
    tail: Vec2;
    tip: Vec2;
    color?: string;
    opacity?: number;
    weight?: number;
    svg?: Record<string, string | number>;
  }

  const {
    tail,
    tip,
    color = "var(--mafs-fg, #333)",
    opacity = 1,
    weight = 2,
    svg,
  }: Props = $props();

  const ctx = getCoordContext();

  // Arrowhead size target in screen pixels. `userSpaceOnUse` requires the
  // marker dimensions in user-space, so divide by the effective px/unit scale
  // (min of x/y, matching preserveAspectRatio: xMidYMid meet).
  const ARROW_PX = 10;
  const arrowSize = $derived.by(() => {
    const xSpan = ctx.viewBox.xMax - ctx.viewBox.xMin;
    const ySpan = ctx.viewBox.yMax - ctx.viewBox.yMin;
    const scale = Math.min(ctx.widthPx / xSpan, ctx.heightPx / ySpan);
    return scale > 0 ? ARROW_PX / scale : 0;
  });

  const markerId = nextId();
  const degenerate = $derived(tail[0] === tip[0] && tail[1] === tip[1]);
</script>

{#if !degenerate}
  <defs>
    <marker
      id={markerId}
      viewBox="0 0 10 10"
      refX="10"
      refY="5"
      markerWidth={arrowSize}
      markerHeight={arrowSize}
      markerUnits="userSpaceOnUse"
      orient="auto"
    >
      <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
    </marker>
  </defs>
  <line
    x1={tail[0]}
    y1={tail[1]}
    x2={tip[0]}
    y2={tip[1]}
    stroke={color}
    stroke-opacity={opacity}
    stroke-width={weight}
    vector-effect="non-scaling-stroke"
    marker-end="url(#{markerId})"
    {...svg}
  />
{/if}
