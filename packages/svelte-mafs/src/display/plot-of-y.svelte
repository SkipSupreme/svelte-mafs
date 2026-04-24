<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";
  import { sample, type Sample } from "../sampling.js";
  import {
    buildPath,
    dashArrayFor,
    defaultTolerance,
    type StrokeStyle,
  } from "./_plot-utils.js";

  export interface Props {
    x: (y: number) => number;
    /** Restrict sampling to this y-range. Defaults to the visible viewport. */
    domain?: readonly [number, number];
    color?: string;
    opacity?: number;
    weight?: number;
    style?: StrokeStyle;
    minSamplingDepth?: number;
    maxSamplingDepth?: number;
  }

  let {
    x,
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
    domain ?? ([vb.yMin, vb.yMax] as readonly [number, number]),
  );

  // Sampler returns [input, f(input)] pairs. For OfY, input is y and output is x,
  // so we swap to [x, y] before emitting SVG path commands.
  const samples = $derived(
    sample(x, {
      domain: effectiveDomain,
      // Tolerance is in output units — here that's x. Use the x-span.
      tolerance: defaultTolerance(vb.xMax - vb.xMin),
      minDepth: minSamplingDepth,
      maxDepth: maxSamplingDepth,
    }),
  );

  const swapped = $derived<readonly Sample[]>(
    samples.map(([inY, outX]) => [outX, inY] as const),
  );

  const d = $derived(buildPath(swapped));
  const dash = $derived(dashArrayFor(style));
</script>

<path
  data-mafs-plot="y"
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
