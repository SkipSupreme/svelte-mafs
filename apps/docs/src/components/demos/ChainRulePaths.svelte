<script lang="ts">
  /**
   * ChainRulePaths — a small DAG of partials. The graph is:
   *
   *      x       y
   *      ├──g₁──┤
   *      │      │
   *      u      v
   *       \    /
   *        \  /
   *         z
   *
   * Click an edge to highlight its partial; click "show ∂z/∂x" to highlight
   * both paths from x to z and assemble the chain-rule sum.
   *
   * Pure SVG — no Mafs needed because the geometry is positional, not
   * coordinate-system-y.
   */

  type EdgeId = 'xu' | 'yu' | 'xv' | 'yv' | 'uz' | 'vz';
  const edges: { id: EdgeId; from: [number, number]; to: [number, number]; label: string }[] = [
    { id: 'xu', from: [80, 60], to: [180, 180], label: '∂u/∂x' },
    { id: 'yu', from: [320, 60], to: [180, 180], label: '∂u/∂y' },
    { id: 'xv', from: [80, 60], to: [380, 180], label: '∂v/∂x' },
    { id: 'yv', from: [320, 60], to: [380, 180], label: '∂v/∂y' },
    { id: 'uz', from: [180, 180], to: [280, 300], label: '∂z/∂u' },
    { id: 'vz', from: [380, 180], to: [280, 300], label: '∂z/∂v' },
  ];

  const nodes = [
    { id: 'x', pos: [80, 60], label: 'x' },
    { id: 'y', pos: [320, 60], label: 'y' },
    { id: 'u', pos: [180, 180], label: 'u' },
    { id: 'v', pos: [380, 180], label: 'v' },
    { id: 'z', pos: [280, 300], label: 'z' },
  ] as const;

  let active = $state<Set<EdgeId>>(new Set());
  let pathLabel = $state<string | null>(null);

  function toggle(id: EdgeId) {
    const next = new Set(active);
    if (next.has(id)) next.delete(id); else next.add(id);
    active = next;
    pathLabel = null;
  }

  function showPathFor(input: 'x' | 'y') {
    // Two paths from input to z; one through u, one through v.
    const a: EdgeId = input === 'x' ? 'xu' : 'yu';
    const b: EdgeId = input === 'x' ? 'xv' : 'yv';
    active = new Set([a, 'uz', b, 'vz']);
    pathLabel = input === 'x'
      ? '∂z/∂x = (∂z/∂u)(∂u/∂x) + (∂z/∂v)(∂v/∂x)'
      : '∂z/∂y = (∂z/∂u)(∂u/∂y) + (∂z/∂v)(∂v/∂y)';
  }

  function clearAll() {
    active = new Set();
    pathLabel = null;
  }
</script>

<div class="widget">
  <header class="header">
    <p class="title">click edges, or pick a path</p>
    <div class="actions">
      <button type="button" onclick={() => showPathFor('x')}>show ∂z/∂x</button>
      <button type="button" onclick={() => showPathFor('y')}>show ∂z/∂y</button>
      <button type="button" class="ghost" onclick={clearAll}>clear</button>
    </div>
  </header>

  <div class="stage">
    <svg viewBox="0 0 460 360" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Chain-rule DAG: x and y feed u and v, both feed z">
      {#each edges as e}
        <g class="edge" class:on={active.has(e.id)}>
          <line
            x1={e.from[0]} y1={e.from[1]}
            x2={e.to[0]} y2={e.to[1]}
            class="edge-line"
            onclick={() => toggle(e.id)}
            role="button"
            tabindex="0"
            onkeydown={(ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); toggle(e.id); } }}
          />
          <text
            x={(e.from[0] + e.to[0]) / 2 + 8}
            y={(e.from[1] + e.to[1]) / 2 - 4}
            class="edge-label"
          >{e.label}</text>
        </g>
      {/each}

      {#each nodes as n}
        <g class="node node-{n.id}">
          <circle cx={n.pos[0]} cy={n.pos[1]} r={22} />
          <text x={n.pos[0]} y={n.pos[1] + 6} text-anchor="middle">{n.label}</text>
        </g>
      {/each}
    </svg>
  </div>

  <div class="readout" aria-live="polite">
    {#if pathLabel}
      <p class="formula">{pathLabel}</p>
      <p class="caption">sum over paths · multiply along edges</p>
    {:else if active.size === 0}
      <p class="caption">click an edge to highlight a partial, or use a button to assemble a chain-rule sum.</p>
    {:else}
      <p class="caption">{active.size} edge{active.size === 1 ? '' : 's'} highlighted.</p>
    {/if}
  </div>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: .85rem;
    background: var(--demo-card); border: 1px solid var(--demo-card-border);
    border-radius: 20px; padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow: 0 1px 0 rgba(0,0,0,.04), 0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 50%, transparent);
  }
  .header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .6rem; margin: 0; }
  .title { margin: 0; font-family: var(--font-display); font-weight: 600; font-size: 1rem; color: var(--site-fg); }
  .actions { display: flex; gap: .35rem; flex-wrap: wrap; }
  .actions button {
    padding: .35rem .7rem; font-family: var(--font-mono); font-size: .8rem;
    border: 1px solid var(--site-border); border-radius: 999px;
    background: var(--demo-card); color: var(--site-fg); cursor: pointer;
  }
  .actions button:hover { border-color: var(--ink-sea); color: var(--ink-sea); }
  .actions button.ghost { color: var(--site-fg-muted); }

  .stage { background: var(--demo-stage); border-radius: 12px; padding: .5rem; }
  .stage svg { display: block; width: 100%; height: auto; max-height: 380px; }

  .edge-line {
    stroke: color-mix(in srgb, var(--site-fg) 30%, transparent);
    stroke-width: 2;
    fill: none;
    cursor: pointer;
    transition: stroke 200ms ease, stroke-width 200ms ease;
  }
  .edge-line:hover { stroke: var(--ink-sea); }
  .edge-line:focus { outline: 2px solid var(--ink-sea); outline-offset: 2px; }
  .edge.on .edge-line { stroke: var(--ink-coral); stroke-width: 3.5; }

  .edge-label {
    font-family: var(--font-mono);
    font-size: 12px;
    fill: var(--site-fg-muted);
    pointer-events: none;
  }
  .edge.on .edge-label { fill: var(--ink-coral); font-weight: 700; }

  .node circle {
    fill: var(--demo-card);
    stroke: var(--ink-red);
    stroke-width: 2;
  }
  .node text {
    font-family: var(--font-display);
    font-style: italic;
    font-size: 18px;
    font-weight: 600;
    fill: var(--ink-red);
  }
  .node-z circle { stroke: var(--ink-coral); }
  .node-z text { fill: var(--ink-coral); }

  .readout {
    padding: .55rem .85rem;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-radius: 10px;
    min-height: 3rem;
  }
  .formula {
    margin: 0;
    font-family: var(--font-mono);
    font-size: .98rem;
    color: var(--ink-coral);
    font-weight: 600;
  }
  .caption {
    margin: .25rem 0 0;
    font-family: var(--font-body);
    font-size: .8rem;
    color: var(--site-fg-muted);
  }
</style>
