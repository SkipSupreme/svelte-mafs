<script lang="ts">
  interface Token {
    name: string;
    label: string;
    value: string;
    hint?: string;
  }

  const initial: Token[] = [
    { name: '--mafs-fg',         label: 'Foreground',  value: '#1a1a1a', hint: 'text + axes' },
    { name: '--mafs-bg',         label: 'Background',  value: '#fdfdfc', hint: 'canvas' },
    { name: '--mafs-line-color', label: 'Line',        value: '#1a1a1a', hint: 'axis lines' },
    { name: '--mafs-grid-color', label: 'Grid',        value: '#e4e4e0', hint: 'grid ticks' },
    { name: '--mafs-blue',       label: 'Blue',        value: '#2563eb' },
    { name: '--mafs-red',        label: 'Red',         value: '#dc2626' },
    { name: '--mafs-green',      label: 'Green',       value: '#16a34a' },
    { name: '--mafs-yellow',     label: 'Yellow',      value: '#ca8a04' },
  ];

  let tokens = $state<Token[]>(initial.map((t) => ({ ...t })));

  const styleDecl = $derived(tokens.map((t) => `${t.name}: ${t.value};`).join('\n  '));
  const cssSnippet = $derived(`:root {\n  ${styleDecl}\n}`);
  const inlineStyle = $derived(tokens.map((t) => `${t.name}: ${t.value};`).join(' '));

  let copied = $state(false);

  async function copy(): Promise<void> {
    try {
      await navigator.clipboard.writeText(cssSnippet);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch {
      copied = false;
    }
  }

  function reset(): void {
    tokens = initial.map((t) => ({ ...t }));
  }
</script>

<div class="playground">
  <div class="controls" role="group" aria-label="Theme tokens">
    <div class="token-grid">
      {#each tokens as token, i (token.name)}
        <label class="token-row">
          <span class="token-label">
            <span class="token-name"><code>{token.name}</code></span>
            {#if token.hint}<span class="token-hint">{token.hint}</span>{/if}
          </span>
          <span class="token-inputs">
            <input
              type="color"
              aria-label={`${token.label} color picker`}
              bind:value={tokens[i].value}
            />
            <input
              type="text"
              aria-label={`${token.label} hex value`}
              spellcheck="false"
              bind:value={tokens[i].value}
            />
          </span>
        </label>
      {/each}
    </div>
    <div class="actions">
      <button type="button" class="action" on:click={reset}>Reset</button>
      <button type="button" class="action action-primary" on:click={copy}>
        {copied ? 'Copied!' : 'Copy CSS'}
      </button>
    </div>
  </div>

  <div class="preview-column">
    <div class="preview" style={inlineStyle} aria-label="Theme preview (decorative)">
      <svg viewBox="0 0 240 160" preserveAspectRatio="xMidYMid meet">
        <rect x="0" y="0" width="240" height="160" fill="var(--mafs-bg)" />
        <g class="grid">
          {#each Array.from({ length: 7 }) as _, i}
            <line x1={30 + i * 30} y1="10" x2={30 + i * 30} y2="150" />
          {/each}
          {#each Array.from({ length: 5 }) as _, i}
            <line x1="10" y1={20 + i * 30} x2="230" y2={20 + i * 30} />
          {/each}
        </g>
        <line class="axis" x1="10" y1="80" x2="230" y2="80" />
        <line class="axis" x1="120" y1="10" x2="120" y2="150" />
        <path
          class="curve-blue"
          d="M10,80 C40,20 70,20 100,80 S160,140 190,80 S230,40 230,60"
          fill="none"
        />
        <path
          class="curve-red"
          d="M10,120 Q60,60 120,100 T230,60"
          fill="none"
        />
        <circle class="dot-green" cx="120" cy="100" r="5" />
        <circle class="dot-yellow" cx="80" cy="50" r="5" />
      </svg>
    </div>

    <pre class="snippet" aria-label="Generated CSS">{cssSnippet}</pre>
  </div>
</div>

<style>
  .playground {
    display: grid;
    gap: 1.5rem;
    grid-template-columns: minmax(0, 1fr);
    margin-top: 1.5rem;
  }
  @media (min-width: 860px) {
    .playground {
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }
  }

  .token-grid {
    display: grid;
    gap: 0.5rem;
  }
  .token-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.55rem 0.75rem;
    border: 1px solid var(--site-border);
    border-radius: var(--radius-md);
    background: var(--site-surface);
  }
  .token-label { display: flex; flex-direction: column; min-width: 0; }
  .token-name { font-size: 0.9rem; }
  .token-hint { font-size: 0.72rem; color: var(--site-fg-muted); }
  .token-inputs { display: flex; gap: 0.4rem; align-items: center; }
  .token-inputs input[type='color'] {
    width: 2rem;
    height: 2rem;
    padding: 0;
    border: 1px solid var(--site-border);
    border-radius: var(--radius-sm);
    background: transparent;
    cursor: pointer;
  }
  .token-inputs input[type='text'] {
    width: 9ch;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    padding: 0.35rem 0.5rem;
    border: 1px solid var(--site-border);
    border-radius: var(--radius-sm);
    background: var(--site-bg);
    color: var(--site-fg);
  }

  .actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    justify-content: flex-end;
  }
  .action {
    padding: 0.45rem 0.9rem;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--site-fg);
    border: 1px solid var(--site-border);
    cursor: pointer;
    font-size: 0.9rem;
    transition: background-color 120ms ease;
  }
  .action:hover { background: var(--site-surface); }
  .action-primary {
    background: var(--site-accent);
    color: var(--site-accent-fg);
    border-color: var(--site-accent);
  }

  .preview-column { display: flex; flex-direction: column; gap: 1rem; }
  .preview {
    border: 1px solid var(--site-border);
    border-radius: var(--radius-lg);
    padding: 0.75rem;
    overflow: hidden;
  }
  .preview svg { display: block; width: 100%; height: auto; }
  .grid line { stroke: var(--mafs-grid-color); stroke-width: 0.5; }
  .axis { stroke: var(--mafs-line-color); stroke-width: 1; }
  .curve-blue { stroke: var(--mafs-blue); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .curve-red { stroke: var(--mafs-red); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .dot-green { fill: var(--mafs-green); }
  .dot-yellow { fill: var(--mafs-yellow); }

  .snippet {
    margin: 0;
    padding: 0.9rem 1rem;
    font-size: 0.8rem;
    background: var(--site-code-bg);
    color: var(--site-code-fg);
    border: 1px solid var(--site-border);
    border-radius: var(--radius-md);
    overflow-x: auto;
    white-space: pre;
  }
</style>
