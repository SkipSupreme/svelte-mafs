<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";
  import { sample } from "../sampling.js";
  import { defaultTolerance } from "./_plot-utils.js";

  export interface Props {
    /** The boundary curve. Region is {(x, y) : y > f(x)} or y < f(x). */
    y: (x: number) => number;
    /**
     * Which half-plane to fill. v1 ships the single-curve form only;
     * between-two-curves lands in a follow-up.
     */
    direction?: "above" | "below";
    domain?: readonly [number, number];
    color?: string;
    opacity?: number;
    minSamplingDepth?: number;
    maxSamplingDepth?: number;
  }

  let {
    y,
    direction = "above",
    domain,
    color = "var(--mafs-blue, #3B82F6)",
    opacity = 0.25,
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

  // v1 limitation: NaN/divergent curve points are dropped rather than
  // sub-dividing the fill into multiple polygons. This renders cleanly
  // for smooth f and acceptably for isolated discontinuities.
  const finite = $derived(
    samples.filter(([sx, sy]) => Number.isFinite(sx) && Number.isFinite(sy)),
  );

  const closeY = $derived(direction === "above" ? vb.yMax : vb.yMin);

  const d = $derived.by(() => {
    if (finite.length < 2) return "";
    const parts: string[] = [];
    const [x0, y0] = finite[0]!;
    parts.push(`M ${x0} ${y0}`);
    for (let i = 1; i < finite.length; i++) {
      const [x, yv] = finite[i]!;
      parts.push(`L ${x} ${yv}`);
    }
    const [xLast] = finite.at(-1)!;
    parts.push(`L ${xLast} ${closeY}`);
    parts.push(`L ${x0} ${closeY}`);
    parts.push("Z");
    return parts.join(" ");
  });
</script>

<path
  data-mafs-plot="inequality"
  data-mafs-direction={direction}
  {d}
  fill={color}
  fill-opacity={opacity}
  stroke="none"
/>
