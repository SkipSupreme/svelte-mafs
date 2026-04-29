<script lang="ts">
  /**
   * Lesson- and module-complete celebration. Spec in DESIGN.md §Motion →
   * Celebration sequences and §Mascot.
   *
   *   level="lesson"  → Tinker hops twice, ~600ms, small confetti, chime.
   *   level="module"  → Graduation cap drops, full-width confetti burst,
   *                     anthem audio (caller plays).
   *
   * This component renders its own mascot image, separate from the hero
   * mascot in `<Tinker />`. Mount it inside the lesson-complete card so
   * the celebration is local to where the user just earned it.
   */
  import { onMount } from 'svelte';
  import { burst } from '../../lib/confetti';
  import { TINKER_EVENT } from '../../lib/events';

  interface Props {
    /** Default level when triggered via the window event without a detail. */
    level?: 'lesson' | 'module';
    size?: number | string;
    /** Auto-fire celebration on mount. Default false — usually triggered via window event. */
    autoplay?: boolean;
    src?: string;
  }

  let {
    level: defaultLevel = 'lesson',
    size = 140,
    autoplay = false,
    src = '/logo-mark.png',
  }: Props = $props();

  let root: HTMLDivElement | undefined = $state();
  let hopping = $state(false);
  let activeLevel = $state<'lesson' | 'module'>(defaultLevel);
  let reducedMotion = $state(false);

  const sizeCss = $derived(typeof size === 'number' ? `${size}px` : size);

  function play(level: 'lesson' | 'module' = defaultLevel) {
    activeLevel = level;
    hopping = true;
    if (root) {
      const count = level === 'module' ? 22 : 10;
      const spread = level === 'module' ? 1.5 : 1;
      burst(root, { count, spread });
    }
    const duration = level === 'module' ? 1100 : 720;
    setTimeout(() => (hopping = false), duration + 50);
  }

  onMount(() => {
    try {
      reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch {
      /* ignore */
    }
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ level?: 'lesson' | 'module' }>).detail;
      play(detail?.level ?? defaultLevel);
    };
    window.addEventListener(TINKER_EVENT.celebrate, handler);
    if (autoplay) {
      requestAnimationFrame(() => play(defaultLevel));
    }
    return () => window.removeEventListener(TINKER_EVENT.celebrate, handler);
  });
</script>

<div
  bind:this={root}
  class="hop"
  class:hop--lesson={activeLevel === 'lesson'}
  class:hop--module={activeLevel === 'module'}
  class:hop--hopping={hopping}
  class:hop--reduced={reducedMotion}
  style="--hop-size: {sizeCss};"
  aria-hidden="true"
>
  {#if activeLevel === 'module'}
    <span class="grad-cap" aria-hidden="true">
      <span class="grad-cap-top"></span>
      <span class="grad-cap-band"></span>
      <span class="grad-cap-tassel"></span>
    </span>
  {/if}
  <img {src} alt="" class="hop-img" width="432" height="477" draggable="false" />
</div>

<style>
  .hop {
    --hop-size: 140px;
    position: relative;
    display: inline-flex;
    align-items: flex-end;
    justify-content: center;
    width: var(--hop-size);
    aspect-ratio: 432 / 477;
    height: auto;
    pointer-events: none;
  }

  .hop-img {
    position: relative;
    z-index: 1;
    width: 100%;
    height: 100%;
    /* /logo-mark.png is the apple, pre-cropped — contain just shows it whole. */
    object-fit: contain;
    object-position: center;
    transform: rotate(-3deg);
    filter: drop-shadow(0 12px 22px rgba(230, 57, 106, 0.32));
    animation: hop-idle 4.6s ease-in-out infinite;
  }

  /* Lesson-complete: hop twice, ~600ms. */
  .hop--lesson.hop--hopping .hop-img {
    animation: hop-twice 720ms cubic-bezier(0.2, 0.8, 0.2, 1) 1;
  }

  @keyframes hop-twice {
    0%   { transform: rotate(-3deg) translateY(0)    scale(1); }
    20%  { transform: rotate(-6deg) translateY(-22px) scale(1.04); }
    40%  { transform: rotate(-3deg) translateY(0)    scale(1); }
    60%  { transform: rotate(-1deg) translateY(-18px) scale(1.03); }
    100% { transform: rotate(-3deg) translateY(0)    scale(1); }
  }

  @keyframes hop-idle {
    0%, 100% { translate: 0 0; }
    50%      { translate: 0 -4px; }
  }

  /* Module-complete: longer choreography, grad cap drops onto head. */
  .hop--module.hop--hopping .hop-img {
    animation: hop-flourish 1100ms cubic-bezier(0.2, 0.8, 0.2, 1) 1;
  }

  @keyframes hop-flourish {
    0%   { transform: rotate(-3deg) translateY(0)    scale(1); }
    25%  { transform: rotate(-8deg) translateY(-32px) scale(1.06); }
    50%  { transform: rotate(0deg)  translateY(-2px)  scale(1.02); }
    70%  { transform: rotate(-4deg) translateY(-22px) scale(1.04); }
    100% { transform: rotate(-3deg) translateY(0)    scale(1); }
  }

  /* Graduation cap. Drops onto Tinker's head from above. */
  .grad-cap {
    position: absolute;
    top: -18%;
    left: 38%;
    width: 38%;
    height: 22%;
    transform: translateY(-150%);
    z-index: 2;
    opacity: 0;
  }
  .hop--module.hop--hopping .grad-cap {
    animation: cap-drop 900ms cubic-bezier(0.34, 1.4, 0.64, 1) 280ms forwards;
  }

  .grad-cap-top {
    position: absolute;
    inset: 0 0 35% 0;
    background: var(--site-fg);
    border-radius: 4px;
    transform: skewX(-8deg);
  }
  .grad-cap-band {
    position: absolute;
    inset: 65% 8% 5% 8%;
    background: var(--site-fg);
    border-radius: 2px;
  }
  .grad-cap-tassel {
    position: absolute;
    top: 20%;
    right: 14%;
    width: 6px;
    height: 60%;
    background: var(--ink-sun);
    border-radius: 2px;
  }

  @keyframes cap-drop {
    0%   { opacity: 0; transform: translateY(-160%) rotate(-12deg); }
    70%  { opacity: 1; transform: translateY(8%)    rotate(4deg); }
    100% { opacity: 1; transform: translateY(0)     rotate(0deg); }
  }

  /* Reduced motion: everything stills. The user still gets confetti (which
     also no-ops under reduced motion via burst()) and the chime sound. */
  .hop--reduced .hop-img,
  .hop--reduced.hop--hopping .hop-img {
    animation: none;
    transform: rotate(-3deg);
  }
  .hop--reduced .grad-cap,
  .hop--reduced.hop--hopping .grad-cap {
    animation: none;
    opacity: 1;
    transform: translateY(0);
  }
</style>
