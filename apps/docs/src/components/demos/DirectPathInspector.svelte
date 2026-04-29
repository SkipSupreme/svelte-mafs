<script lang="ts">
  // The "blocks as additive corrections to a bigram floor" demo. Scripted on
  // a tiny char vocab (V=8) at d=4 with N=4 blocks. The user drags k from 0
  // to N. At k=0 the prediction is exactly softmax(W_E[t] · W_E^T) — the
  // bigram model. Each subsequent block adds a delta that pushes the
  // residual stream toward the right next-token direction.

  const VOCAB = ['a', 'e', 'h', 'l', 'o', 'r', 't', '.'];
  const V = VOCAB.length;
  const D = 4;
  const N = 4;
  const INPUT_IDX = 2; // 'h'
  const TARGET_IDX = 1; // 'e' — the right next-token after 'h' for "hello"

  function mulberry32(a: number) {
    return () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng: () => number) {
    const u = 1 - rng();
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Build a deterministic W_E.
  const rng = mulberry32(42);
  const W_E: number[][] = Array.from({ length: V }, () =>
    Array.from({ length: D }, () => gauss(rng) * 0.6)
  );

  // Per-block deltas: each block's contribution to the residual stream at
  // the current input position. Hand-shaped so adding them sharpens the
  // model's prediction toward the target ('e') and slightly biases 'l'.
  const blockDeltas: number[][] = (() => {
    const eDir = W_E[TARGET_IDX];
    const lDir = W_E[3]; // 'l'
    const norm = (v: number[]) => Math.sqrt(v.reduce((a, b) => a + b * b, 0));
    const eNorm = norm(eDir);
    const lNorm = norm(lDir);
    const eUnit = eDir.map((x) => x / eNorm);
    const lUnit = lDir.map((x) => x / lNorm);
    const scale = 0.65;
    return [
      // block 1: small bias toward e
      eUnit.map((v) => v * scale * 0.4),
      // block 2: more bias toward e
      eUnit.map((v) => v * scale * 0.6),
      // block 3: stronger e
      eUnit.map((v) => v * scale * 0.7),
      // block 4: e + a bit of l (knowing 'hel' continues as 'l')
      eUnit.map((v, i) => v * scale * 0.5 + lUnit[i] * scale * 0.4),
    ];
  })();

  let k = $state(0);

  function dot(a: number[], b: number[]): number {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }
  function softmax(logits: number[]): number[] {
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const z = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / z);
  }

  const residual = $derived.by(() => {
    const r = [...W_E[INPUT_IDX]];
    for (let i = 0; i < k; i++) {
      for (let d = 0; d < D; d++) r[d] += blockDeltas[i][d];
    }
    return r;
  });

  const logits = $derived(W_E.map((row) => dot(residual, row)));
  const probs = $derived(softmax(logits));
  const argmaxIdx = $derived(probs.indexOf(Math.max(...probs)));
  const targetProb = $derived(probs[TARGET_IDX]);
</script>

<div class="widget">
  <div class="header">
    <div class="setup">
      <span class="label">input token</span>
      <span class="token-pill">"<em>{VOCAB[INPUT_IDX]}</em>"</span>
      <span class="muted">predict next</span>
    </div>
    <div class="setup">
      <span class="label">block budget</span>
      <input
        type="range"
        min="0"
        max={N}
        step="1"
        bind:value={k}
        aria-label="number of active blocks"
      />
      <span class="kval">k = {k} / {N}</span>
    </div>
  </div>

  <div class="status">
    {#if k === 0}
      <strong>k = 0:</strong> identity-only — every block replaced with the identity map. The output is exactly the bigram floor: softmax(W<sub>E</sub>[<em>{VOCAB[INPUT_IDX]}</em>] · W<sub>E</sub><sup>⊤</sup>).
    {:else}
      <strong>k = {k}:</strong> first {k} block{k > 1 ? 's' : ''} active. Each one has added a delta to the residual stream.
    {/if}
  </div>

  <div class="vocab">
    {#each VOCAB as char, i}
      <div class="vrow" class:argmax={i === argmaxIdx} class:target={i === TARGET_IDX}>
        <span class="vchar">{char === '.' ? '·' : char}</span>
        <div class="vbar">
          <div class="vfill" style="width:{probs[i] * 100}%"></div>
        </div>
        <span class="vprob">{(probs[i] * 100).toFixed(1)}%</span>
      </div>
    {/each}
  </div>

  <div class="footer">
    <span class="footer-key target">★ target</span>
    <span class="footer-key argmax">▼ argmax</span>
    <div class="target-meter">
      P("<em>{VOCAB[TARGET_IDX]}</em>" | "<em>{VOCAB[INPUT_IDX]}</em>"):
      <strong>{(targetProb * 100).toFixed(1)}%</strong>
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

  .header {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem 1.4rem;
    align-items: center;
  }
  .setup {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.85rem;
  }
  .label {
    color: var(--site-fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem;
    font-weight: 700;
  }
  .token-pill {
    padding: 0.2rem 0.5rem;
    background: color-mix(in srgb, var(--ink-sun) 18%, transparent);
    border: 1px solid var(--ink-sun);
    border-radius: 8px;
    color: var(--site-fg);
  }
  .token-pill em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 700;
  }
  .muted {
    color: var(--site-fg-muted);
    font-size: 0.78rem;
  }
  .setup input[type='range'] {
    width: 220px;
    accent-color: var(--ink-sea);
  }
  .kval {
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    color: var(--site-fg);
    min-width: 5rem;
  }

  .status {
    padding: 0.55rem 0.85rem;
    border-radius: 10px;
    background: color-mix(in srgb, var(--site-fg) 4%, transparent);
    border-left: 3px solid var(--ink-sea);
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: var(--site-fg);
  }
  .status strong {
    font-weight: 700;
  }
  .status em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }

  .vocab {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }
  .vrow {
    display: grid;
    grid-template-columns: 2.5rem 1fr 4rem;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0.5rem;
    border-radius: 8px;
    transition: background 220ms ease;
  }
  .vrow.target {
    background: color-mix(in srgb, var(--ink-sun) 10%, transparent);
  }
  .vrow.argmax .vchar {
    color: var(--ink-sea);
    font-weight: 700;
  }
  .vchar {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
    font-size: 1.1rem;
    text-align: center;
    color: var(--site-fg);
  }
  .vbar {
    height: 14px;
    background: color-mix(in srgb, var(--site-fg) 6%, transparent);
    border-radius: 7px;
    overflow: hidden;
  }
  .vfill {
    height: 100%;
    background: var(--ink-sea);
    border-radius: 7px;
    transition: width 220ms ease;
  }
  .vrow.argmax .vfill {
    background: var(--ink-sea);
  }
  .vrow.target .vfill {
    background: var(--ink-sun);
  }
  .vprob {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    text-align: right;
    color: var(--site-fg-muted);
    font-size: 0.85rem;
  }

  .footer {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem 1rem;
    padding-top: 0.5rem;
    border-top: 1px solid color-mix(in srgb, var(--site-fg) 12%, transparent);
    font-family: var(--font-mono);
    font-size: 0.78rem;
  }
  .footer-key {
    color: var(--site-fg-muted);
  }
  .target-meter {
    margin-left: auto;
    color: var(--site-fg);
  }
  .target-meter em {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 600;
  }
  .target-meter strong {
    color: var(--ink-sun);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }
</style>
