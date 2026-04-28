<script lang="ts">
  import type { StreakEventDetail } from '../../lib/xp';

  let { streak: initial = 0 }: { streak?: number } = $props();
  let streak = $state(initial);
  let flicker = $state(false);

  $effect(() => {
    const onBump = (e: Event) => {
      const detail = (e as CustomEvent<StreakEventDetail>).detail;
      if (!detail) return;
      streak = detail.streak;
      flicker = true;
      setTimeout(() => (flicker = false), 800);
      window.dispatchEvent(
        new CustomEvent('tinker:announce', {
          detail: { message: `Streak now ${detail.streak} days.` },
        }),
      );
    };
    window.addEventListener('tinker:streak', onBump);
    return () => window.removeEventListener('tinker:streak', onBump);
  });
</script>

{#if streak > 0}
  <span class="streak" class:flicker>
    <span class="num">{streak}</span>
    <span class="flame" aria-hidden="true">🔥</span>
    <span class="sr-only">{streak} day streak</span>
  </span>
{/if}

<style>
  .streak {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-family: 'Space Grotesk', system-ui, sans-serif;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--ink-sun);
  }
  .flame { display: inline-block; transform-origin: center bottom; }
  .flicker .flame { animation: flame-flicker 800ms ease-out 1; }
  .sr-only {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
  }
</style>
