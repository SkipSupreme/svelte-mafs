<script lang="ts">
  import { Spring, prefersReducedMotion } from 'svelte/motion';
  import { onMount } from 'svelte';
  import { TINKER_EVENT } from '../../lib/events';
  import { LS_KEY } from '../../lib/storage-keys';

  const THROTTLE_MS = 120_000; // 2 min global

  let visible = $state(false);
  let hint = $state('');
  const slide = new Spring(120, { stiffness: 0.14, damping: 0.6 });

  function show(h: string) {
    let last = 0;
    try { last = Number(localStorage.getItem(LS_KEY.stuckLastShown) ?? '0'); } catch {}
    if (Date.now() - last < THROTTLE_MS) return;
    hint = h;
    visible = true;
    if (prefersReducedMotion.current) {
      slide.set(0, { duration: 0 });
    } else {
      slide.target = 0;
    }
    try { localStorage.setItem(LS_KEY.stuckLastShown, String(Date.now())); } catch {}
    window.dispatchEvent(
      new CustomEvent(TINKER_EVENT.announce, { detail: { message: `Hint: ${h}` } }),
    );
  }

  function hide() {
    if (prefersReducedMotion.current) {
      slide.set(120, { duration: 0 });
    } else {
      slide.target = 120;
    }
    setTimeout(() => (visible = false), 280);
  }

  onMount(() => {
    const onStuck = (e: Event) => {
      const detail = (e as CustomEvent<{ hint: string }>).detail;
      if (detail?.hint) show(detail.hint);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') hide(); };
    window.addEventListener(TINKER_EVENT.stuck, onStuck);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener(TINKER_EVENT.stuck, onStuck);
      window.removeEventListener('keydown', onKey);
    };
  });
</script>

{#if visible}
  <button
    class="stuck"
    style="transform: translateY({slide.current}%)"
    onclick={hide}
    aria-label="Dismiss hint"
  >
    <img src="/logo-mark.png" alt="" width="48" height="53" />
    <span class="bubble">{hint}</span>
  </button>
{/if}

<style>
  .stuck {
    position: fixed;
    right: 1.5rem;
    bottom: 1.5rem;
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--band-cream);
    border: 1px solid color-mix(in srgb, var(--ink-red) 24%, transparent);
    border-radius: 999px;
    box-shadow: 0 1px 0 rgba(0,0,0,0.04), 0 18px 36px -20px color-mix(in srgb, var(--ink-red) 40%, transparent);
    cursor: pointer;
    z-index: 50;
    font-family: inherit;
    color: var(--site-fg);
  }
  .bubble {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.95rem;
    color: var(--site-fg);
    max-width: 28ch;
    text-align: left;
  }
  img { display: block; }
</style>
