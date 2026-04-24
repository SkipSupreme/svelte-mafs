<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";
  import {
    buildPath,
    dashArrayFor,
    sampleParametric,
    type StrokeStyle,
  } from "./_plot-utils.js";

  export interface Props {
    /** Parametric function f(t) = [x, y]. */
    xy: (t: number) => readonly [number, number];
    /** t-range to sample over. No sensible viewport default — required. */
    t: readonly [number, number];
    color?: string;
    opacity?: number;
    weight?: number;
    style?: StrokeStyle;
    minSamplingDepth?: number;
    maxSamplingDepth?: number;
  }

  let {
    xy,
    t,
    color = "var(--mafs-blue, #3B82F6)",
    opacity = 1,
    weight = 2,
    style = "solid",
    minSamplingDepth = 4,
    maxSamplingDepth = 14,
  }: Props = $props();

  const ctx = getCoordContext();
  const vb = $derived(ctx.viewBox);

  // Tolerance in output (x, y) space — scale against the smaller visible
  // span so chord error is perceptually uniform on non-square viewBoxes.
  const tolerance = $derived(
    Math.min(vb.xMax - vb.xMin, vb.yMax - vb.yMin) / 500,
  );

  const samples = $derived(
    sampleParametric(xy, {
      domain: t,
      tolerance,
      minDepth: minSamplingDepth,
      maxDepth: maxSamplingDepth,
    }),
  );

  const d = $derived(buildPath(samples));
  const dash = $derived(dashArrayFor(style));
</script>

<path
  data-mafs-plot="parametric"
  {d}
  fill="none"
  stroke={color}
  stroke-opacity={opacity}
  stroke-width={weight}
  vector-effect="non-scaling-stroke"
  stroke-linejoin="round"
  stroke-linecap="round"
  stroke-dasharray={dash ?? undefined}
/>
