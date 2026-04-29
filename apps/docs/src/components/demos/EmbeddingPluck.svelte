<script lang="ts">
  import { Mafs, Coordinates, MovablePoint, Point } from 'svelte-mafs';

  interface Props {
    /** Initial selected character. */
    initialPick?: string;
  }

  let { initialPick = 'a' }: Props = $props();

  const VOCAB: string[] = ['.', ...'abcdefghijklmnopqrstuvwxyz'.split('')];
  const display = (c: string) => (c === '.' ? '·' : c);

  // Ideal embedding positions — pre-arranged into rough clusters that
  // reflect bigram co-occurrence (vowels group, common consonants group,
  // rare letters scatter at the edges). The simulated loss is the mean
  // squared deviation from these ideals; "training" the embedding means
  // pulling each character toward its ideal.
  const idealMap: Record<string, [number, number]> = {
    '.': [0.0, 1.4],
    a: [-1.4, 0.5], e: [-1.6, -0.3], i: [-1.0, 0.9], o: [-1.7, 0.2], u: [-1.2, -0.6], y: [-0.9, -0.9],
    n: [0.6, 0.4], r: [0.9, 0.0], s: [0.4, -0.5], t: [0.7, 0.6], l: [0.2, 0.3], d: [0.5, -0.2], m: [0.3, 0.7],
    h: [1.1, 0.4], c: [0.3, -0.8], b: [0.1, 1.0], f: [-0.5, -1.2], g: [0.6, -0.6],
    p: [1.2, -0.4], k: [1.4, 0.7], v: [-0.3, -0.4], w: [-0.6, 1.1], j: [1.6, -0.9],
    q: [2.1, -1.3], x: [-2.0, 1.2], z: [1.9, 1.1],
  };

  function makeInitialEmbeddings(): Record<string, [number, number]> {
    // Start at ideal + tiny noise so the layout looks "trained" but the
    // user can still see drift if they pluck a character far away.
    const out: Record<string, [number, number]> = {};
    for (const c of VOCAB) {
      const [ix, iy] = idealMap[c] ?? [0, 0];
      out[c] = [ix + (Math.random() - 0.5) * 0.06, iy + (Math.random() - 0.5) * 0.06];
    }
    return out;
  }

  let positions: Record<string, [number, number]> = $state(makeInitialEmbeddings());
  let selected: string = $state(initialPick);

  function loss(): number {
    let s = 0;
    for (const c of VOCAB) {
      const [px, py] = positions[c];
      const [ix, iy] = idealMap[c] ?? [0, 0];
      s += (px - ix) ** 2 + (py - iy) ** 2;
    }
    return s / VOCAB.length;
  }
  const currentLoss = $derived.by(() => {
    void positions; // re-run on positions change
    return loss();
  });

  // Two-way bind for the selected point so dragging updates state.
  const selX = $derived(positions[selected]?.[0] ?? 0);
  const selY = $derived(positions[selected]?.[1] ?? 0);
  function setSelectedX(x: number) {
    positions = { ...positions, [selected]: [x, positions[selected][1]] };
  }
  function setSelectedY(y: number) {
    positions = { ...positions, [selected]: [positions[selected][0], y] };
  }

  function pick(c: string) {
    selected = c;
  }

  function snapToIdeal(): void {
    const next = { ...positions };
    const [ix, iy] = idealMap[selected];
    next[selected] = [ix, iy];
    positions = next;
  }
  function reset(): void {
    positions = makeInitialEmbeddings();
  }
  function shuffle(): void {
    const next: Record<string, [number, number]> = {};
    for (const c of VOCAB) {
      next[c] = [(Math.random() - 0.5) * 4, (Math.random() - 0.5) * 3];
    }
    positions = next;
  }

  // Cluster a char belongs to (for color coding).
  const VOWELS = new Set(['a', 'e', 'i', 'o', 'u', 'y']);
  const COMMON = new Set(['n', 'r', 's', 't', 'l', 'd', 'm', 'h', 'c']);
  function colorFor(c: string): string {
    if (c === '.') return 'var(--ink-sun)';
    if (VOWELS.has(c)) return 'var(--ink-red)';
    if (COMMON.has(c)) return 'var(--ink-sea)';
    return 'var(--site-fg-muted)';
  }

  const fmt = (n: number) => n.toFixed(3);
  const isFar = $derived.by(() => {
    const [px, py] = positions[selected];
    const [ix, iy] = idealMap[selected];
    return Math.hypot(px - ix, py - iy) > 0.5;
  });
</script>

<div class="widget">
  <header class="head">
    <div class="meta">
      <span class="meta-key">selected</span>
      <span class="meta-val sel">{display(selected)}</span>
    </div>
    <div class="meta">
      <span class="meta-key">C[{display(selected)}]</span>
      <span class="meta-val">({fmt(selX)}, {fmt(selY)})</span>
    </div>
    <div class="meta meta-loss" class:bad={isFar}>
      <span class="meta-key">simulated loss</span>
      <span class="meta-val">{fmt(currentLoss)}</span>
    </div>
  </header>

  <div class="stage">
    <Mafs width={460} height={340} viewBox={{ x: [-2.7, 2.7], y: [-1.9, 1.9] }}>
      <Coordinates.Cartesian
        xAxis={{ labels: () => '' }}
        yAxis={{ labels: () => '' }}
      />

      <!-- All other characters as static labeled points -->
      {#each VOCAB as c}
        {#if c !== selected}
          <Point x={positions[c][0]} y={positions[c][1]} color={colorFor(c)} opacity={0.8} />
        {/if}
      {/each}

      <!-- Selected char as a movable point -->
      <MovablePoint
        bind:x={positions[selected][0]}
        bind:y={positions[selected][1]}
        color={colorFor(selected)}
        label={`Embedding of ${display(selected)}`}
      />
    </Mafs>

    <!-- Overlay labels, positioned via CSS calc on the SVG-coord system.
         We absolutely position by transforming to a percentage of the viewBox. -->
    <div class="labels" aria-hidden="true">
      {#each VOCAB as c}
        {@const [x, y] = positions[c]}
        {@const left = ((x - -2.7) / (2.7 - -2.7)) * 100}
        {@const top = ((1.9 - y) / (1.9 - -1.9)) * 100}
        <span
          class="lbl"
          class:active={c === selected}
          style="left:{left}%; top:{top}%; color:{colorFor(c)};"
        >{display(c)}</span>
      {/each}
    </div>
  </div>

  <div class="picker">
    <span class="picker-label">click a token to pluck →</span>
    <div class="picker-grid">
      {#each VOCAB as c}
        <button
          type="button"
          class="pick-btn"
          class:active={c === selected}
          style={c === selected ? `border-color:${colorFor(c)};color:${colorFor(c)};` : ''}
          onclick={() => pick(c)}
          aria-pressed={c === selected}
        >{display(c)}</button>
      {/each}
    </div>
  </div>

  <div class="controls">
    <button type="button" class="btn" onclick={snapToIdeal}>Snap {display(selected)} to its trained position</button>
    <button type="button" class="btn btn-ghost" onclick={shuffle}>Shuffle (untrained init)</button>
    <button type="button" class="btn btn-ghost" onclick={reset}>Reset</button>
  </div>

  <p class="caption">
    Each dot is a row of the embedding matrix <em>C</em>. Vowels (red) cluster
    together because the model learned that their following-character
    distributions are similar; common consonants (blue) cluster too. Pluck
    one and drag it across the plane — you're physically editing
    <em>C[i]</em>, and the simulated loss reacts. The point of an embedding
    isn't anything mystical: it's a vector of learned parameters that
    happens to encode semantic structure.
  </p>
</div>

<style>
  .widget {
    display: flex; flex-direction: column; gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(0.85rem, 2vw, 1.4rem);
    color: var(--site-fg);
    box-shadow:
      0 1px 0 rgba(0,0,0,0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 50%, transparent);
  }
  .head {
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.1rem;
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--site-fg-muted);
  }
  .meta { display: inline-flex; gap: 0.4rem; align-items: baseline; }
  .meta-key { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.7rem; }
  .meta-val { color: var(--site-fg); font-variant-numeric: tabular-nums; font-weight: 600; }
  .meta-val.sel {
    background: var(--site-fg); color: var(--demo-card);
    padding: 0 0.4rem; border-radius: 4px;
  }
  .meta-loss.bad .meta-val { color: var(--ink-coral); }

  .stage {
    position: relative;
    width: 100%; background: var(--demo-stage); border-radius: 12px;
    overflow: hidden; user-select: none; -webkit-user-select: none; touch-action: none;
  }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }

  .labels {
    position: absolute; inset: 0;
    pointer-events: none;
  }
  .lbl {
    position: absolute;
    transform: translate(-50%, -120%);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 600;
    text-shadow:
      -1px -1px 0 var(--demo-stage),
      1px -1px 0 var(--demo-stage),
      -1px 1px 0 var(--demo-stage),
      1px 1px 0 var(--demo-stage);
  }
  .lbl.active { font-size: 0.85rem; transform: translate(-50%, -130%); }

  .picker { display: flex; flex-direction: column; gap: 0.35rem; }
  .picker-label {
    font-family: var(--font-mono); font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--site-fg-muted);
  }
  .picker-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(28px, 1fr));
    gap: 3px;
  }
  .pick-btn {
    width: 100%; aspect-ratio: 1;
    display: inline-flex; align-items: center; justify-content: center;
    border: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
    background: transparent;
    color: var(--site-fg);
    border-radius: 4px;
    font-family: var(--font-mono); font-size: 0.78rem; font-weight: 600;
    cursor: pointer;
  }
  .pick-btn:hover { border-color: var(--site-fg); }
  .pick-btn.active { border-width: 2px; font-weight: 700; }

  .controls { display: flex; flex-wrap: wrap; gap: 0.4rem; }
  .btn {
    border: 1px solid color-mix(in srgb, var(--site-fg) 18%, transparent);
    background: transparent; color: var(--site-fg);
    border-radius: 999px; padding: 0.35rem 0.85rem;
    font-size: 0.83rem; font-weight: 600; cursor: pointer;
    transition: background 160ms ease, transform 120ms ease, border-color 160ms ease;
  }
  .btn:hover { transform: translateY(-1px); border-color: var(--site-fg); }
  .btn-ghost { color: var(--site-fg-muted); font-weight: 500; }

  .caption {
    margin: 0; font-size: 0.85rem; color: var(--site-fg-muted); line-height: 1.55;
  }
  .caption em {
    color: var(--site-fg); font-style: normal;
    font-family: var(--font-mono); font-size: 0.85em;
  }
</style>
