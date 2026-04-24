<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";
  import { sample } from "../sampling.js";
  import {
    buildPath,
    dashArrayFor,
    defaultTolerance,
    type StrokeStyle,
  } from "./_plot-utils.js";

  export interface Props {
    y: (x: number) => number;
    /** Restrict sampling to this x-range. Defaults to the visible viewport. */
    domain?: readonly [number, number];
    color?: string;
    opacity?: number;
    /** Stroke width in CSS pixels (non-scaling). */
    weight?: number;
    style?: StrokeStyle;
    /** Force this many unconditional subdivisions before the flatness check. */
    minSamplingDepth?: number;
    /** Recursion cap to keep pathological functions terminating. */
    maxSamplingDepth?: number;
  }

  let {
    y,
    domain,
    color = "var(--mafs-blue, #3B82F6)",
    opacity = 1,
    weight = 2,
    style = "solid",
    minSamplingDepth = 4,
    maxSamplingDepth = 14,
  }: Props = $props();

  const ctx = getCoordContext();
  const vb = $derived(ctx.viewBox);

  const effectiveDomain = $derived(
    domain ?? ([vb.xMin, vb.xMax] as readonly [number, number]),
  );

  const samples = $derived(
    sample(y, {
      domain: effectiveDomain,
      tolerance: defaultTolerance(vb.yMax - vb.yMin),
      minDepth: minSamplingDepth,
      maxDepth: maxSamplingDepth,
    }),
  );

  const d = $derived(buildPath(samples));
  const dash = $derived(dashArrayFor(style));
</script>

<path
  data-mafs-plot="x"
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
