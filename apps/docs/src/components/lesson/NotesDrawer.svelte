<script lang="ts">
  let { lessonSlug } = $props<{ lessonSlug: string }>();

  let open = $state(false);
  let body = $state('');
  let status = $state<'idle' | 'loading' | 'saving' | 'saved' | 'error'>('idle');
  let lastSavedAt = $state<Date | null>(null);
  let saveTimer: ReturnType<typeof setTimeout> | undefined;

  function getCsrf(): string {
    const m =
      document.cookie.match(/(?:^|;\s*)__Secure-tinker\.csrf_token=([^;]+)/) ??
      document.cookie.match(/(?:^|;\s*)tinker\.csrf_token=([^;]+)/);
    return m?.[1] ? decodeURIComponent(m[1]) : '';
  }

  async function load() {
    status = 'loading';
    try {
      const res = await fetch('/api/notes/' + encodeURIComponent(lessonSlug), {
        credentials: 'same-origin',
      });
      if (!res.ok) throw new Error('load failed');
      const data = (await res.json()) as { body: string; updatedAt: string | null };
      body = data.body;
      lastSavedAt = data.updatedAt ? new Date(data.updatedAt) : null;
      status = 'idle';
    } catch {
      status = 'error';
    }
  }

  async function save() {
    status = 'saving';
    try {
      const res = await fetch('/api/notes/' + encodeURIComponent(lessonSlug), {
        method: 'PUT',
        credentials: 'same-origin',
        headers: {
          'content-type': 'application/json',
          'x-tinker-csrf': getCsrf(),
        },
        body: JSON.stringify({ body }),
      });
      if (!res.ok) throw new Error('save failed');
      const data = (await res.json()) as { updatedAt: string };
      lastSavedAt = new Date(data.updatedAt);
      status = 'saved';
    } catch {
      status = 'error';
    }
  }

  function onInput(ev: Event) {
    body = (ev.currentTarget as HTMLTextAreaElement).value;
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(save, 1500);
  }

  function toggle() {
    open = !open;
    if (open && status === 'idle' && !lastSavedAt) load();
  }
</script>

<button class="notes-toggle" type="button" onclick={toggle} aria-expanded={open}>
  📝 Notes
</button>

{#if open}
  <aside class="notes-drawer" role="complementary" aria-label="Notes">
    <header>
      <h3>Notes for this lesson</h3>
      <button type="button" onclick={() => (open = false)} aria-label="Close notes">×</button>
    </header>
    <textarea
      placeholder="Your private notes for this lesson…"
      value={body}
      oninput={onInput}
      disabled={status === 'loading'}
    ></textarea>
    <footer>
      {#if status === 'loading'}
        <span class="status">Loading…</span>
      {:else if status === 'saving'}
        <span class="status">Saving…</span>
      {:else if status === 'saved' && lastSavedAt}
        <span class="status">Saved {lastSavedAt.toLocaleTimeString()}</span>
      {:else if status === 'error'}
        <span class="status err">Couldn't save. <button onclick={save}>Retry</button></span>
      {:else if lastSavedAt}
        <span class="status">Last saved {lastSavedAt.toLocaleString()}</span>
      {/if}
    </footer>
  </aside>
{/if}

<style>
  .notes-toggle {
    background: var(--site-surface);
    border: 1px solid var(--site-border);
    border-radius: var(--radius-md);
    padding: 0.4rem 0.75rem;
    cursor: pointer;
    font: inherit;
    color: var(--site-fg);
  }
  .notes-toggle:hover { background: color-mix(in srgb, var(--site-surface) 70%, var(--site-bg)); }
  .notes-drawer {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(420px, 100vw);
    background: var(--site-bg);
    border-left: 1px solid var(--site-border);
    z-index: 50;
    display: flex;
    flex-direction: column;
    box-shadow: -8px 0 24px rgba(0,0,0,0.06);
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem;
    border-bottom: 1px solid var(--site-border);
  }
  header h3 { margin: 0; font-size: 1rem; }
  header button {
    background: none;
    border: none;
    font-size: 1.5rem;
    line-height: 1;
    cursor: pointer;
    color: var(--site-fg-muted);
  }
  textarea {
    flex: 1;
    width: 100%;
    border: none;
    outline: none;
    padding: 1rem;
    background: var(--site-bg);
    color: var(--site-fg);
    font-family: var(--font-body);
    font-size: 0.95rem;
    resize: none;
  }
  footer {
    padding: 0.6rem 1rem;
    border-top: 1px solid var(--site-border);
    color: var(--site-fg-muted);
    font-size: 0.8rem;
  }
  .status.err { color: #b91c1c; }
  .status button {
    background: none;
    border: none;
    color: var(--site-fg);
    text-decoration: underline;
    cursor: pointer;
    padding: 0;
    font: inherit;
  }
</style>
