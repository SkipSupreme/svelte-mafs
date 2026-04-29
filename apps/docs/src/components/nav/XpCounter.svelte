<script lang="ts">
  import { Tween, prefersReducedMotion } from 'svelte/motion';
  import { cubicOut } from 'svelte/easing';
  import { fly, fade } from 'svelte/transition';
  import type { XpEventDetail } from '../../lib/xp';
  import { TINKER_EVENT } from '../../lib/events';

  let { initial = 0 }: { initial?: number } = $props();
  const display = new Tween(initial, { duration: 900, easing: cubicOut });
  const shown = $derived(Math.round(display.current));
  let floaters = $state<{ id: number; amount: number }[]>([]);
  let nextId = 1;

  $effect(() => {
    const onXp = (e: Event) => {
      const detail = (e as CustomEvent<XpEventDetail>).detail;
      if (!detail) return;
      if (prefersReducedMotion.current) {
        display.set(detail.total, { duration: 0 });
      } else {
        display.target = detail.total;
      }
      floaters = [...floaters, { id: nextId++, amount: detail.amount }];
    };
    window.addEventListener(TINKER_EVENT.xp, onXp);
    return () => window.removeEventListener(TINKER_EVENT.xp, onXp);
  });

  function remove(id: number) {
    floaters = floaters.filter((f) => f.id !== id);
  }
</script>

<span class="xp-counter" aria-label={`XP ${shown}`}>
  <span class="xp-num">XP {shown.toLocaleString()}</span>
  <span class="xp-floaters" aria-hidden="true">
    {#each floaters as f (f.id)}
      {#if prefersReducedMotion.current}
        <span
          class="xp-float"
          in:fade={{ duration: 700 }}
          out:fade={{ duration: 700 }}
          onintroend={() => setTimeout(() => remove(f.id), 700)}
        >+{f.amount}</span>
      {:else}
        <span
          class="xp-float"
          in:fly={{ y: -28, duration: 700, easing: cubicOut }}
          out:fade={{ duration: 700 }}
          onintroend={() => setTimeout(() => remove(f.id), 700)}
        >+{f.amount}</span>
      {/if}
    {/each}
  </span>
  <span class="sr-only">{shown}</span>
</span>

<style>
  .xp-counter {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-family: 'Space Grotesk', system-ui, sans-serif;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--ink-sun);
  }
  .xp-floaters {
    position: absolute;
    inset: 0;
    pointer-events: none;
  }
  .xp-float {
    position: absolute;
    top: 0;
    right: -0.4ch;
    color: var(--ink-sun);
    font-weight: 700;
    font-size: 0.85em;
  }
  .sr-only {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
  }
</style>
