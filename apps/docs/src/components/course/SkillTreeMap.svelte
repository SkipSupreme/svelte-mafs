<script lang="ts">
  import { onMount } from 'svelte';

  interface ModuleNode {
    id: string;
    order: number;
    title: string;
    shortTitle: string;
    summary: string;
    status: 'planned' | 'drafting' | 'shipped';
    endgameConnection: string | null;
    arc: string;
    arcLabel: string;
    arcColor: string;
    arcXIdx: number;
    arcOrder: number;
    arcSize: number;
    prereqs: string[];
    lessonCount: number;
    href: string;
    keystone: boolean;
  }
  interface Arc {
    id: string;
    color: string;
    xIdx: number;
    label: string;
  }

  let { modules, arcs }: { modules: ModuleNode[]; arcs: Arc[] } = $props();

  // --- Layout: arcs flow left-to-right; modules stack vertically within
  // each arc, centered on the canvas y-axis. Keystones are slightly offset
  // to an outer rail so their larger radius doesn't crowd neighbors. ---
  const ARC_SPACING = 260;
  const NODE_SPACING = 180;          // bumped from 140 to fit short-title labels below each circle
  const CANVAS_PAD_X = 140;
  const CANVAS_PAD_Y = 120;
  const CENTER_Y = 540;              // recentered for taller canvas

  type PositionedNode = ModuleNode & { x: number; y: number; r: number };
  const nodes: PositionedNode[] = modules.map((m) => {
    const x = CANVAS_PAD_X + m.arcXIdx * ARC_SPACING;
    const yOffset = (m.arcOrder - (m.arcSize - 1) / 2) * NODE_SPACING;
    // tiny left/right nudge so columns feel organic, not gridded
    const nudge = m.arcOrder % 2 === 0 ? -18 : 18;
    const y = CENTER_Y + yOffset;
    return {
      ...m,
      x: x + nudge,
      y,
      r: m.keystone ? 38 : 26,
    };
  });

  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  const edges = nodes.flatMap((n) =>
    n.prereqs
      .map((p) => nodeById.get(p))
      .filter((p): p is PositionedNode => !!p)
      .map((src) => ({
        id: `${src.id}->${n.id}`,
        src,
        dst: n,
      })),
  );

  // canvas dimensions
  const maxArcIdx = Math.max(...arcs.map((a) => a.xIdx), 0);
  const CANVAS_W = CANVAS_PAD_X * 2 + maxArcIdx * ARC_SPACING;
  const maxArcSize = Math.max(...modules.map((m) => m.arcSize), 1);
  const CANVAS_H = CENTER_Y * 2 + CANVAS_PAD_Y + NODE_SPACING * 0.2;

  // --- Viewport: pan + zoom. ---
  let tx = $state(0);
  let ty = $state(0);
  let scale = $state(1);
  const MIN_SCALE = 0.4;
  const MAX_SCALE = 2.5;

  let svgEl: SVGSVGElement | undefined = $state();
  let hovered: PositionedNode | null = $state(null);
  let dragging = false;
  let dragStart = { x: 0, y: 0, tx: 0, ty: 0 };

  function onPointerDown(e: PointerEvent) {
    if ((e.target as Element)?.closest('a')) return; // let links click through
    dragging = true;
    dragStart = { x: e.clientX, y: e.clientY, tx, ty };
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  }
  function onPointerMove(e: PointerEvent) {
    if (!dragging) return;
    tx = dragStart.tx + (e.clientX - dragStart.x);
    ty = dragStart.ty + (e.clientY - dragStart.y);
  }
  function onPointerUp(e: PointerEvent) {
    dragging = false;
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
  }
  function onWheel(e: WheelEvent) {
    if (!svgEl) return;
    e.preventDefault();
    const rect = svgEl.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    const next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
    // zoom toward cursor: keep the world-point under cursor fixed
    const worldX = (cx - tx) / scale;
    const worldY = (cy - ty) / scale;
    scale = next;
    tx = cx - worldX * scale;
    ty = cy - worldY * scale;
  }
  function zoomBy(factor: number) {
    if (!svgEl) return;
    const rect = svgEl.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale * factor));
    const worldX = (cx - tx) / scale;
    const worldY = (cy - ty) / scale;
    scale = next;
    tx = cx - worldX * scale;
    ty = cy - worldY * scale;
  }
  function reset() {
    if (!svgEl) return;
    const rect = svgEl.getBoundingClientRect();
    const widthFit = rect.width / (CANVAS_W + 40);
    const heightFit = rect.height / (CANVAS_H + 40);
    // Floor at 0.7 so nodes + labels stay readable on narrow viewports; user
    // can still pan to see parts of the tree that don't fit.
    const s = Math.max(0.7, Math.min(widthFit, heightFit, 1));
    scale = s;
    tx = (rect.width - CANVAS_W * s) / 2;
    ty = (rect.height - CANVAS_H * s) / 2;
  }

  onMount(() => {
    reset();
    const onResize = () => reset();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  });

  function edgePath(src: PositionedNode, dst: PositionedNode): string {
    const dx = dst.x - src.x;
    const dy = dst.y - src.y;
    const cx1 = src.x + dx * 0.5;
    const cy1 = src.y;
    const cx2 = dst.x - dx * 0.5;
    const cy2 = dst.y;
    return `M ${src.x} ${src.y} C ${cx1} ${cy1} ${cx2} ${cy2} ${dst.x} ${dst.y}`;
  }

  // Tooltip position tracks the hovered node in SCREEN space. Flip
  // above/below so the tooltip doesn't clip when the node is near the
  // top of the canvas.
  const tooltipPos = $derived.by(() => {
    if (!hovered) return null;
    const screenY = hovered.y * scale + ty;
    const canvasH = svgEl?.getBoundingClientRect().height ?? 700;
    const showAbove = screenY > canvasH * 0.42;
    return {
      left: hovered.x * scale + tx,
      top: showAbove
        ? screenY - hovered.r * scale - 14
        : screenY + hovered.r * scale + 14,
      placement: showAbove ? 'above' : 'below',
    };
  });
</script>

<div class="map-wrap">
  <div class="toolbar">
    <div class="legend">
      {#each arcs as arc}
        <span class="legend-item">
          <span class="legend-dot" style:background={arc.color}></span>
          {arc.label}
        </span>
      {/each}
    </div>
    <div class="controls">
      <button onclick={() => zoomBy(1.25)} aria-label="Zoom in">+</button>
      <button onclick={() => zoomBy(1 / 1.25)} aria-label="Zoom out">−</button>
      <button onclick={reset} aria-label="Reset view">Reset</button>
    </div>
  </div>

  <div class="canvas" role="region" aria-label="Course skill tree">
    <svg
      bind:this={svgEl}
      class="tree-svg"
      class:dragging
      onpointerdown={onPointerDown}
      onpointermove={onPointerMove}
      onpointerup={onPointerUp}
      onpointercancel={onPointerUp}
      onwheel={onWheel}
    >
      <defs>
        <marker
          id="arrowhead"
          markerWidth="8"
          markerHeight="8"
          refX="7"
          refY="4"
          orient="auto"
        >
          <path d="M0 0 L8 4 L0 8 Z" fill="var(--site-fg-muted)" opacity="0.55" />
        </marker>
      </defs>
      <g transform={`translate(${tx} ${ty}) scale(${scale})`}>
        <!-- edges -->
        {#each edges as edge}
          <path
            d={edgePath(edge.src, edge.dst)}
            class="edge"
            class:edge-active={hovered && (hovered.id === edge.src.id || hovered.id === edge.dst.id)}
            marker-end="url(#arrowhead)"
            fill="none"
          />
        {/each}

        <!-- nodes -->
        {#each nodes as n}
          <a href={n.href} class="node-link" aria-label={`Module ${n.order}: ${n.title}`}>
            <g
              class={`node status-${n.status}`}
              class:keystone={n.keystone}
              class:hovered={hovered?.id === n.id}
              onpointerenter={() => (hovered = n)}
              onpointerleave={() => { if (hovered?.id === n.id) hovered = null; }}
              onfocus={() => (hovered = n)}
              onblur={() => { if (hovered?.id === n.id) hovered = null; }}
            >
              <circle
                cx={n.x}
                cy={n.y}
                r={n.r + 4}
                class="node-halo"
                style:stroke={n.arcColor}
              />
              <circle
                cx={n.x}
                cy={n.y}
                r={n.r}
                class="node-body"
                style:--arc-color={n.arcColor}
              />
              <text
                x={n.x}
                y={n.y + 1}
                class="node-label"
                text-anchor="middle"
                dominant-baseline="middle"
              >M{n.order}</text>
              <text
                x={n.x}
                y={n.y + n.r + 20}
                class="node-short-title"
                class:keystone-title={n.keystone}
                text-anchor="middle"
              >{n.shortTitle}</text>
              {#if n.keystone}
                <text
                  x={n.x}
                  y={n.y - n.r - 8}
                  text-anchor="middle"
                  class="node-keystone-label"
                >KEYSTONE</text>
              {/if}
            </g>
          </a>
        {/each}
      </g>
    </svg>

    {#if hovered && tooltipPos}
      <aside
        class="tooltip"
        class:tt-below={tooltipPos.placement === 'below'}
        style:left={`${tooltipPos.left}px`}
        style:top={`${tooltipPos.top}px`}
        aria-hidden="true"
      >
        <header class="tt-head">
          <span class="tt-index">M{hovered.order}</span>
          <span class={`tt-status status-${hovered.status}`}>{hovered.status}</span>
        </header>
        <h3 class="tt-title">{hovered.title}</h3>
        <p class="tt-summary">{hovered.summary}</p>
        {#if hovered.endgameConnection}
          <p class="tt-endgame">
            <span class="tt-kicker">⚡</span>
            {hovered.endgameConnection}
          </p>
        {/if}
        <footer class="tt-meta">
          <span>{hovered.arcLabel}</span>
          {#if hovered.lessonCount > 0}
            <span>·</span><span>{hovered.lessonCount} lesson{hovered.lessonCount === 1 ? '' : 's'}</span>
          {/if}
        </footer>
      </aside>
    {/if}

    <footer class="help" aria-hidden="true">
      drag to pan · scroll to zoom · click any node to open the module
    </footer>
  </div>
</div>

<style>
  .map-wrap {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding: 0 1.25rem;
    max-width: var(--max-hero);
    margin: 0 auto;
    width: 100%;
  }
  .legend {
    display: flex;
    gap: 0.9rem;
    flex-wrap: wrap;
    font-size: 0.82rem;
    color: var(--site-fg-muted);
    font-family: var(--font-mono);
  }
  .legend-item { display: inline-flex; align-items: center; gap: 0.35rem; }
  .legend-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 999px;
  }
  .controls { display: flex; gap: 0.35rem; }
  .controls button {
    padding: 0.4rem 0.75rem;
    font: inherit;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid var(--site-border);
    border-radius: 999px;
    background: var(--demo-card);
    color: var(--site-fg);
    cursor: pointer;
    min-width: 2.2rem;
  }
  .controls button:hover { border-color: var(--ink-red); color: var(--ink-red); }

  .canvas {
    position: relative;
    width: 100%;
    height: clamp(520px, 72vh, 860px);
    background: var(--demo-stage);
    border-top: 1px solid var(--site-border);
    border-bottom: 1px solid var(--site-border);
    overflow: hidden;
    user-select: none;
    -webkit-user-select: none;
  }
  .tree-svg {
    width: 100%;
    height: 100%;
    cursor: grab;
    touch-action: none;
    display: block;
  }
  .tree-svg.dragging { cursor: grabbing; }

  .edge {
    stroke: color-mix(in srgb, var(--site-fg) 25%, transparent);
    stroke-width: 1.5;
    transition: stroke 160ms ease, stroke-width 160ms ease, opacity 160ms ease;
    opacity: 0.75;
  }
  .edge-active {
    stroke: var(--ink-red);
    stroke-width: 2.25;
    opacity: 1;
  }

  .node-link { cursor: pointer; }
  .node { transition: transform 180ms ease; transform-origin: center; transform-box: fill-box; }
  .node.hovered { transform: scale(1.08); }

  .node-halo {
    fill: none;
    stroke-width: 1.25;
    opacity: 0.4;
    stroke-dasharray: 4 3;
  }
  .node.keystone .node-halo {
    opacity: 0.85;
    stroke-width: 1.75;
    stroke-dasharray: none;
  }
  .node.status-shipped .node-halo {
    opacity: 0.95;
    stroke-width: 2;
    stroke-dasharray: none;
  }

  .node-body {
    fill: var(--demo-card);
    stroke: var(--arc-color);
    stroke-width: 2.5;
    transition: fill 180ms ease, stroke-width 180ms ease;
  }
  .node.status-shipped .node-body {
    fill: color-mix(in srgb, var(--cta) 35%, var(--demo-card));
    stroke: var(--cta-hover);
  }
  .node.status-drafting .node-body {
    fill: color-mix(in srgb, var(--ink-sun) 30%, var(--demo-card));
    stroke: color-mix(in srgb, var(--ink-sun) 85%, var(--site-fg));
  }
  .node.status-planned .node-body {
    fill: var(--demo-card);
    stroke-dasharray: 4 3;
    stroke-opacity: 0.6;
  }
  .node.keystone .node-body {
    stroke-width: 3.5;
  }
  .node.hovered .node-body {
    stroke-width: 4;
  }

  .node-label {
    font-family: var(--font-mono);
    font-size: 14px;
    font-weight: 700;
    fill: var(--site-fg);
    pointer-events: none;
  }
  .node.status-planned .node-label {
    fill: var(--site-fg-muted);
  }
  .node.keystone .node-label {
    font-size: 16px;
  }
  .node-keystone-label {
    font-family: var(--font-mono);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.18em;
    fill: var(--arc-color);
    pointer-events: none;
    opacity: 0.85;
  }
  .node-short-title {
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 600;
    fill: var(--site-fg);
    pointer-events: none;
    letter-spacing: -0.005em;
  }
  .node-short-title.keystone-title {
    font-size: 17px;
    font-weight: 700;
  }
  .node.status-planned .node-short-title {
    fill: var(--site-fg-muted);
    font-weight: 500;
  }
  .node.hovered .node-short-title {
    fill: var(--ink-red);
  }

  .tooltip {
    position: absolute;
    transform: translate(-50%, -100%);
    width: 300px;
    max-width: calc(100vw - 2.5rem);
    padding: 0.85rem 1rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 14px;
    box-shadow: 0 20px 44px -18px rgba(0,0,0,0.35);
    pointer-events: none;
    z-index: 5;
  }
  .tooltip.tt-below { transform: translate(-50%, 0); }
  .tt-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin: 0 0 0.35rem;
  }
  .tt-index {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--ink-red);
    font-weight: 700;
    letter-spacing: 0.08em;
  }
  .tt-status {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
  }
  .tt-status.status-shipped  { background: color-mix(in srgb, var(--cta) 20%, transparent); color: var(--cta-hover); }
  .tt-status.status-drafting { background: color-mix(in srgb, var(--ink-sun) 20%, transparent); color: color-mix(in srgb, var(--ink-sun) 80%, var(--site-fg)); }
  .tt-status.status-planned  { background: color-mix(in srgb, var(--site-fg) 10%, transparent); color: var(--site-fg-muted); }
  .tt-title {
    font-family: var(--font-display);
    font-weight: 650;
    font-size: 1.05rem;
    line-height: 1.2;
    margin: 0 0 0.4rem;
  }
  .tt-summary {
    font-size: 0.86rem;
    line-height: 1.5;
    color: var(--site-fg-muted);
    margin: 0 0 0.55rem;
  }
  .tt-endgame {
    margin: 0 0 0.5rem;
    padding: 0.45rem 0.6rem;
    background: color-mix(in srgb, var(--ink-sun) 14%, transparent);
    border-left: 3px solid var(--ink-sun);
    border-radius: 0 8px 8px 0;
    font-family: var(--font-display);
    font-style: italic;
    font-size: 0.85rem;
    line-height: 1.35;
  }
  .tt-kicker { font-style: normal; margin-right: 0.25rem; }
  .tt-meta {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--site-fg-muted);
  }

  .help {
    position: absolute;
    left: 0; right: 0; bottom: 0.5rem;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--site-fg-muted);
    opacity: 0.75;
    pointer-events: none;
  }
</style>
