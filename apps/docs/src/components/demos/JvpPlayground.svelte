<script lang="ts">
  import { Mafs, Coordinates, Vector, MovablePoint, Point } from 'svelte-mafs';

  /**
   * JvpPlayground — visualize forward (JVP) and reverse (VJP) mode autodiff
   * on a small vector-valued function F.
   *
   *   F(x, y) = (x² + y,  x·y)
   *   J(x, y) = [[2x, 1], [y, x]]
   *
   * Forward (JVP): user drags the *input tangent* dv. We compute J·dv and
   *   draw it as the output tangent.
   * Reverse (VJP): user drags the *output cotangent* w. We compute Jᵀ·w
   *   and draw it as the input cotangent.
   *
   * Same J. Different direction of action. The whole point of the lesson.
   */

  type Mode = 'forward' | 'reverse';
  let mode = $state<Mode>('forward');

  // Base point — fixed for visual stability. Pick one with non-trivial J.
  const baseX = 1;
  const baseY = 0.8;
  const Fx = baseX * baseX + baseY;
  const Fy = baseX * baseY;
  const J = [
    [2 * baseX, 1],
    [baseY, baseX],
  ];

  // The draggable handle, interpreted differently per mode.
  let hx = $state(0.6);
  let hy = $state(0.4);

  // Forward: input tangent dv → output tangent J·dv.
  const outFwdX = $derived(J[0][0] * hx + J[0][1] * hy);
  const outFwdY = $derived(J[1][0] * hx + J[1][1] * hy);

  // Reverse: output cotangent w → input cotangent Jᵀ·w.
  // (Jᵀ has rows = J's columns.)
  const inRevX = $derived(J[0][0] * hx + J[1][0] * hy);
  const inRevY = $derived(J[0][1] * hx + J[1][1] * hy);

  // What's drawn in each panel depends on mode.
  const inputTangent = $derived<[number, number]>(
    mode === 'forward' ? [hx, hy] : [inRevX, inRevY],
  );
  const outputTangent = $derived<[number, number]>(
    mode === 'forward' ? [outFwdX, outFwdY] : [hx, hy],
  );
</script>

<div class="widget">
  <header class="header">
    <p class="title"><em>F</em>(<em>x</em>,<em>y</em>) = (<em>x</em>²+<em>y</em>, <em>xy</em>) at <em>p</em> = (1, 0.8)</p>
    <div class="modes">
      <button type="button" class:active={mode === 'forward'} onclick={() => (mode = 'forward')}>forward (JVP)</button>
      <button type="button" class:active={mode === 'reverse'} onclick={() => (mode = 'reverse')}>reverse (VJP)</button>
    </div>
  </header>

  <div class="grid">
    <div class="panel">
      <p class="panel-label">input space {mode === 'forward' ? '(drag d v)' : '(computed: Jᵀ w)'}</p>
      <div class="stage">
        <Mafs width={300} height={280} viewBox={{ x: [-0.5, 2.2], y: [-1, 2] }}>
          <Coordinates.Cartesian />
          <Point x={baseX} y={baseY} color="var(--ink-red)" />
          <Vector
            tail={[baseX, baseY]}
            tip={[baseX + inputTangent[0], baseY + inputTangent[1]]}
            color={mode === 'forward' ? 'var(--ink-coral)' : 'var(--ink-sea)'}
            weight={2.5}
          />
          {#if mode === 'forward'}
            <MovablePoint bind:x={hx} bind:y={hy} color="var(--ink-coral)" label="Input tangent dv" />
          {/if}
        </Mafs>
      </div>
      <p class="formula">d<em>v</em> = ({inputTangent[0].toFixed(2)}, {inputTangent[1].toFixed(2)})</p>
    </div>

    <div class="panel">
      <p class="panel-label">output space {mode === 'reverse' ? '(drag w)' : '(computed: J · d v)'}</p>
      <div class="stage">
        <Mafs width={300} height={280} viewBox={{ x: [0, 4], y: [-1, 2] }}>
          <Coordinates.Cartesian />
          <Point x={Fx} y={Fy} color="var(--ink-red)" />
          <Vector
            tail={[Fx, Fy]}
            tip={[Fx + outputTangent[0], Fy + outputTangent[1]]}
            color={mode === 'forward' ? 'var(--ink-sea)' : 'var(--ink-coral)'}
            weight={2.5}
          />
          {#if mode === 'reverse'}
            <MovablePoint bind:x={hx} bind:y={hy} color="var(--ink-coral)" label="Output cotangent w" />
          {/if}
        </Mafs>
      </div>
      <p class="formula"><em>w</em> = ({outputTangent[0].toFixed(2)}, {outputTangent[1].toFixed(2)})</p>
    </div>
  </div>

  <p class="note">
    <strong>{mode === 'forward' ? 'forward mode' : 'reverse mode'}:</strong>
    {#if mode === 'forward'}
      drag the coral arrow on the left. The output tangent updates as <em>J</em> · d<em>v</em>.
    {:else}
      drag the coral arrow on the right. The input cotangent appears on the left as <em>J</em>ᵀ · <em>w</em>.
    {/if}
    Same Jacobian, opposite direction. Reverse mode is what <code>loss.backward()</code> does.
  </p>
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
  .title em { font-style: italic; }
  .modes { display: flex; gap: .35rem; }
  .modes button { padding: .35rem .75rem; font-family: var(--font-mono); font-size: .8rem; border: 1px solid var(--site-border); border-radius: 999px; background: var(--demo-card); color: var(--site-fg); cursor: pointer; }
  .modes button:hover { border-color: var(--ink-sea); color: var(--ink-sea); }
  .modes button.active { background: var(--ink-sea); color: var(--cta-fg); border-color: var(--ink-sea); }

  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
  .panel { display: flex; flex-direction: column; gap: .35rem; }
  .panel-label { margin: 0; font-family: var(--font-mono); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: var(--site-fg-muted); }
  .stage { width: 100%; background: var(--demo-stage); border-radius: 12px; padding: .35rem; touch-action: none; }
  .stage :global(svg) { display: block; width: 100%; height: auto; max-width: 100%; }
  .formula { margin: 0; font-family: var(--font-mono); font-size: .85rem; color: var(--site-fg); font-variant-numeric: tabular-nums; }
  .formula em { font-style: italic; font-family: var(--font-display); }

  .note { margin: 0; padding: .55rem .85rem; background: color-mix(in srgb, var(--site-fg) 4%, transparent); border-radius: 10px; font-family: var(--font-body); font-size: .82rem; color: var(--site-fg); }
  .note em { font-style: italic; }
  .note strong { color: var(--ink-sea); }
  .note code { font-family: var(--font-mono); font-size: .8rem; background: color-mix(in srgb, var(--ink-sea) 14%, transparent); padding: .05rem .35rem; border-radius: 4px; }
</style>
