<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";
  import { inferLabels } from "../math.js";

  export interface Props {
    xAxis?: boolean;
    yAxis?: boolean;
    grid?: boolean;
    subdivisions?: number;
  }

  let {
    xAxis = true,
    yAxis = true,
    grid = true,
    // `subdivisions` is reserved for future minor-tick support; accepted now
    // so callers can pass it without re-plumbing when Stream 6 wires it up.
    subdivisions: _subdivisions = 1,
  }: Props = $props();

  const ctx = getCoordContext();
  const vb = $derived(ctx.viewBox);

  // User-units-per-pixel on each axis. Used to counter-scale text so
  // `font-size="12"` renders as ~12 CSS pixels regardless of viewBox span.
  const sx = $derived((vb.xMax - vb.xMin) / Math.max(ctx.widthPx, 1));
  const sy = $derived((vb.yMax - vb.yMin) / Math.max(ctx.heightPx, 1));

  const xTicks = $derived(inferLabels(vb.xMin, vb.xMax));
  const yTicks = $derived(inferLabels(vb.yMin, vb.yMax));

  const xLabels = $derived(xTicks.filter((t) => t !== 0));
  const yLabels = $derived(yTicks.filter((t) => t !== 0));
</script>

<g data-mafs-coordinates>
  {#if grid}
    <g
      data-mafs-grid
      stroke="var(--mafs-grid-color, #e5e5e5)"
      stroke-width="1"
      vector-effect="non-scaling-stroke"
      fill="none"
    >
      {#each xTicks as x (`gx-${x}`)}
        <line x1={x} y1={vb.yMin} x2={x} y2={vb.yMax} />
      {/each}
      {#each yTicks as y (`gy-${y}`)}
        <line x1={vb.xMin} y1={y} x2={vb.xMax} y2={y} />
      {/each}
    </g>
  {/if}

  {#if xAxis}
    <line
      data-mafs-axis="x"
      x1={vb.xMin}
      y1={0}
      x2={vb.xMax}
      y2={0}
      stroke="var(--mafs-fg, #333)"
      stroke-width="1"
      vector-effect="non-scaling-stroke"
    />
    <g data-mafs-labels="x">
      {#each xLabels as x (`lx-${x}`)}
        <g transform="translate({x}, 0) scale({sx}, {-sy})">
          <text
            y="16"
            text-anchor="middle"
            fill="var(--mafs-fg, #333)"
            font-size="12"
            font-family="var(--mafs-font, system-ui, sans-serif)"
          >{x}</text>
        </g>
      {/each}
    </g>
  {/if}

  {#if yAxis}
    <line
      data-mafs-axis="y"
      x1={0}
      y1={vb.yMin}
      x2={0}
      y2={vb.yMax}
      stroke="var(--mafs-fg, #333)"
      stroke-width="1"
      vector-effect="non-scaling-stroke"
    />
    <g data-mafs-labels="y">
      {#each yLabels as y (`ly-${y}`)}
        <g transform="translate(0, {y}) scale({sx}, {-sy})">
          <text
            x="-6"
            dy="4"
            text-anchor="end"
            fill="var(--mafs-fg, #333)"
            font-size="12"
            font-family="var(--mafs-font, system-ui, sans-serif)"
          >{y}</text>
        </g>
      {/each}
    </g>
  {/if}
</g>
