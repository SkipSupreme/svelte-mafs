<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";

  export interface Props {
    /** Vector field f(x, y) = [dx, dy] in user-space units. */
    xy: (x: number, y: number) => readonly [number, number];
    /** Grid spacing in user-space units. */
    step?: number;
    /** Restrict x-range. Defaults to visible viewport. */
    xDomain?: readonly [number, number];
    /** Restrict y-range. Defaults to visible viewport. */
    yDomain?: readonly [number, number];
    color?: string;
    opacity?: number;
    /** Arrow shaft width in CSS pixels (non-scaling). */
    weight?: number;
    /**
     * Fraction of `step` used as the upper bound on drawn arrow length.
     * 0.9 keeps neighboring arrows from visually overlapping.
     */
    normalizedLength?: number;
  }

  let {
    xy,
    step = 1,
    xDomain,
    yDomain,
    color = "var(--mafs-blue, #3B82F6)",
    opacity = 1,
    weight = 1.5,
    normalizedLength = 0.9,
  }: Props = $props();

  const ctx = getCoordContext();
  const vb = $derived(ctx.viewBox);

  const xRange = $derived(
    xDomain ?? ([vb.xMin, vb.xMax] as readonly [number, number]),
  );
  const yRange = $derived(
    yDomain ?? ([vb.yMin, vb.yMax] as readonly [number, number]),
  );

  interface Arrow {
    key: string;
    x: number;
    y: number;
    tipX: number;
    tipY: number;
    head: string;
  }

  const EPS = 1e-9;

  const arrows = $derived.by((): Arrow[] => {
    if (step <= 0) return [];
    const [x0, x1] = xRange;
    const [y0, y1] = yRange;

    // Snap grid to integer multiples of step so the pattern stays stable
    // across pan/zoom instead of sliding with the viewport edges.
    const xStart = Math.ceil(x0 / step) * step;
    const yStart = Math.ceil(y0 / step) * step;

    interface Raw {
      x: number;
      y: number;
      dx: number;
      dy: number;
      mag: number;
    }
    const raws: Raw[] = [];
    let maxMag = 0;
    for (let x = xStart; x <= x1 + EPS; x += step) {
      for (let y = yStart; y <= y1 + EPS; y += step) {
        const v = xy(x, y);
        const dx = v[0];
        const dy = v[1];
        if (!Number.isFinite(dx) || !Number.isFinite(dy)) continue;
        const mag = Math.hypot(dx, dy);
        if (mag === 0) continue;
        raws.push({ x, y, dx, dy, mag });
        if (mag > maxMag) maxMag = mag;
      }
    }
    if (maxMag === 0) return [];

    const maxDrawn = step * normalizedLength;
    const scale = maxDrawn / maxMag;
    // Arrowhead size relative to the drawn arrow length.
    const headScale = 0.3;

    return raws.map(({ x, y, dx, dy, mag }) => {
      const sx = dx * scale;
      const sy = dy * scale;
      const tipX = x + sx;
      const tipY = y + sy;
      const len = mag * scale;
      const h = Math.min(len * headScale, maxDrawn * 0.4);
      const nx = sx / len;
      const ny = sy / len;
      // Back-center sits `h` user-units behind the tip along the shaft.
      const backX = tipX - h * nx;
      const backY = tipY - h * ny;
      // Perpendicular to (nx, ny).
      const px = -ny;
      const py = nx;
      const leftX = backX + (h / 2) * px;
      const leftY = backY + (h / 2) * py;
      const rightX = backX - (h / 2) * px;
      const rightY = backY - (h / 2) * py;
      const head = `${leftX},${leftY} ${tipX},${tipY} ${rightX},${rightY}`;
      return {
        key: `${x},${y}`,
        x,
        y,
        // Shaft stops at the back-center so the head doesn't sit on top of a stroke.
        tipX: backX,
        tipY: backY,
        head,
      };
    });
  });
</script>

<g
  data-mafs-plot="vector-field"
  stroke={color}
  fill={color}
  stroke-opacity={opacity}
  fill-opacity={opacity}
>
  {#each arrows as a (a.key)}
    <line
      data-mafs-arrow="shaft"
      x1={a.x}
      y1={a.y}
      x2={a.tipX}
      y2={a.tipY}
      stroke-width={weight}
      vector-effect="non-scaling-stroke"
      stroke-linecap="round"
    />
    <polygon data-mafs-arrow="head" points={a.head} stroke="none" />
  {/each}
</g>
