<script lang="ts">
  import { Mafs, Coordinates, Vector, Circle, Text } from 'svelte-mafs';

  // Fixed query and key (un-rotated). Visual values — far enough from the
  // origin that rotation traces a clear arc, not symmetric so direction
  // matters.
  const q0 = [1.4, 0.3] as const;
  const k0 = [1.0, 1.0] as const;

  // Frequency: 45° per position step. In a real transformer this would be
  // 1/10000^(2i/d) and tiny; we exaggerate for visibility.
  const THETA = Math.PI / 4;
  const POS_MAX = 8;

  let m = $state(1);
  let n = $state(3);

  function rotate([x, y]: readonly [number, number], a: number) {
    const c = Math.cos(a);
    const s = Math.sin(a);
    return [c * x - s * y, s * x + c * y] as const;
  }

  const qm = $derived(rotate(q0, m * THETA));
  const kn = $derived(rotate(k0, n * THETA));

  const dotOriginal = $derived(q0[0] * k0[0] + q0[1] * k0[1]);
  const dotRotated = $derived(qm[0] * kn[0] + qm[1] * kn[1]);

  // The relative-only property: q_m · k_n depends only on (n - m).
  // Fix a reference offset = (n - m) at the same value but with different
  // absolute positions, and show the same number comes out.
  const offset = $derived(n - m);
  const refDot = $derived.by(() => {
    // Use absolute positions (0, offset). Rotated q_0 is just q_0.
    const ref_qm = rotate(q0, 0);
    const ref_kn = rotate(k0, offset * THETA);
    return ref_qm[0] * ref_kn[0] + ref_qm[1] * ref_kn[1];
  });

  function shiftBoth(d: number) {
    const newM = m + d;
    const newN = n + d;
    if (newM >= 0 && newM <= POS_MAX && newN >= 0 && newN <= POS_MAX) {
      m = newM;
      n = newN;
    }
  }

  const fmt = (n: number) => (Math.abs(n) < 0.005 ? '0.00' : n.toFixed(3));
</script>

<div class="widget">
  <div class="controls">
    <label class="ctrl">
      <span class="ctrl-label">m (query position)</span>
      <input
        type="range"
        min="0"
        max={POS_MAX}
        step="1"
        bind:value={m}
        aria-label="query position m"
      />
      <output class="ctrl-val">{m}</output>
    </label>
    <label class="ctrl">
      <span class="ctrl-label">n (key position)</span>
      <input
        type="range"
        min="0"
        max={POS_MAX}
        step="1"
        bind:value={n}
        aria-label="key position n"
      />
      <output class="ctrl-val">{n}</output>
    </label>
    <div class="shift">
      <span class="ctrl-label">shift both</span>
      <button type="button" class="seg-btn" onclick={() => shiftBoth(-1)} disabled={m <= 0 || n <= 0}>−1</button>
      <button type="button" class="seg-btn" onclick={() => shiftBoth(1)} disabled={m >= POS_MAX || n >= POS_MAX}>+1</button>
    </div>
  </div>

  <div class="stage">
    <Mafs width={560} height={360} viewBox={{ x: [-2, 2], y: [-2, 2] }}>
      <Coordinates.Cartesian />

      <Circle center={[0, 0]} radius={Math.hypot(q0[0], q0[1])} color="var(--ink-red)" weight={1} fillOpacity={0} strokeOpacity={0.25} />
      <Circle center={[0, 0]} radius={Math.hypot(k0[0], k0[1])} color="var(--ink-sea)" weight={1} fillOpacity={0} strokeOpacity={0.25} />

      <Vector tail={[0, 0]} tip={[q0[0], q0[1]]} color="var(--ink-red)" weight={1.5} opacity={0.35} />
      <Vector tail={[0, 0]} tip={[k0[0], k0[1]]} color="var(--ink-sea)" weight={1.5} opacity={0.35} />

      <Vector tail={[0, 0]} tip={[qm[0], qm[1]]} color="var(--ink-red)" weight={3} />
      <Vector tail={[0, 0]} tip={[kn[0], kn[1]]} color="var(--ink-sea)" weight={3} />

      <Text x={qm[0] * 1.18} y={qm[1] * 1.18} color="var(--ink-red)" size={14} latex="q_m" />
      <Text x={kn[0] * 1.22} y={kn[1] * 1.22} color="var(--ink-sea)" size={14} latex="k_n" />
    </Mafs>
  </div>

  <div class="readout">
    <dl class="dots">
      <div class="row dim">
        <dt><em>q</em>·<em>k</em> <span class="muted">(unrotated)</span></dt>
        <dd>{fmt(dotOriginal)}</dd>
      </div>
      <div class="row hot">
        <dt><em>q</em><sub>m</sub>·<em>k</em><sub>n</sub> <span class="muted">(rotated)</span></dt>
        <dd>{fmt(dotRotated)}</dd>
      </div>
      <div class="row check">
        <dt>same offset reference (m=0, n={offset})</dt>
        <dd>{fmt(refDot)}</dd>
      </div>
    </dl>

    <div class="verdict" data-state={Math.abs(dotRotated - refDot) < 1e-9 ? 'ok' : 'broken'}>
      <span class="verdict-icon">{Math.abs(dotRotated - refDot) < 1e-9 ? '✓' : '✗'}</span>
      <span class="verdict-text">
        rotated dot product depends only on <em>n − m</em> = {offset}
      </span>
    </div>
  </div>
</div>

<style>
  .widget {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    background: var(--demo-card);
    border: 1px solid var(--demo-card-border);
    border-radius: 20px;
    padding: clamp(1rem, 2vw, 1.5rem);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04),
      0 24px 48px -28px color-mix(in srgb, var(--ink-sea) 55%, transparent);
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.85rem;
  }
  .ctrl {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.85rem;
  }
  .ctrl-label {
    color: var(--site-fg-muted);
    font-weight: 600;
  }
  .ctrl input[type='range'] {
    width: 140px;
    accent-color: var(--ink-sea);
  }
  .ctrl-val {
    min-width: 1.5rem;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    color: var(--site-fg);
  }
  .shift {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
  .seg-btn {
    appearance: none;
    border: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
    background: transparent;
    color: var(--site-fg-muted);
    padding: 0.3rem 0.6rem;
    border-radius: 8px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    cursor: pointer;
  }
  .seg-btn:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }
  .seg-btn:not(:disabled):hover {
    background: color-mix(in srgb, var(--site-fg) 5%, transparent);
  }

  .stage {
    width: 100%;
    background: var(--demo-stage);
    border-radius: 12px;
    overflow: hidden;
    user-select: none;
    -webkit-user-select: none;
    touch-action: none;
  }
  .stage :global(svg) {
    display: block;
    width: 100%;
    height: auto;
    max-width: 100%;
  }

  .readout {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 14%, transparent);
  }
  .dots {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin: 0;
    font-family: var(--font-mono);
    font-size: 0.9rem;
  }
  .dots .row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.4rem 0.7rem;
    border-radius: 10px;
    border-left: 3px solid var(--site-fg-muted);
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
  }
  .dots .row.dim {
    opacity: 0.7;
  }
  .dots .row.hot {
    border-left-color: var(--ink-sea);
    background: color-mix(in srgb, var(--ink-sea) 8%, transparent);
  }
  .dots .row.check {
    border-left-color: var(--ink-sun);
    background: color-mix(in srgb, var(--ink-sun) 12%, transparent);
  }
  .dots dt em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }
  .dots dt .muted {
    color: var(--site-fg-muted);
    font-size: 0.8rem;
    margin-left: 0.3rem;
  }
  .dots dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    color: var(--site-fg);
  }

  .verdict {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.8rem;
    border-radius: 12px;
    border-left: 3px solid var(--site-fg-muted);
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    font-family: var(--font-mono);
    font-size: 0.85rem;
  }
  .verdict[data-state='ok'] {
    border-left-color: var(--cta-hover);
    background: color-mix(in srgb, var(--cta-hover) 10%, transparent);
  }
  .verdict[data-state='ok'] .verdict-icon {
    color: var(--cta-hover);
  }
  .verdict-icon {
    font-weight: 700;
    width: 1.2rem;
    text-align: center;
  }
  .verdict em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }
</style>
