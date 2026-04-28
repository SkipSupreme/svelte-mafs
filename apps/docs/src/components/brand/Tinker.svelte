<script lang="ts">
  /**
   * Tinker the Apple — the live mascot. Spec in DESIGN.md §Mascot.
   *
   * Idle behavior: always-on bob (±2px) + cursor-aware tilt. Click triggers
   * a bounce, a tick sound, and a math-symbol burst. Every 10th click is a
   * bigger milestone burst. Reduced motion freezes ambient animation; the
   * click bounce still fires because it's a deliberate user response.
   */
  import { onMount } from 'svelte';
  import { play } from '../../lib/sound';
  import { burst } from '../../lib/confetti';

  interface Props {
    /** Mascot width. Number = px; string = any CSS length, e.g. "clamp(180px, 38vw, 380px)". */
    size?: number | string;
    /** Add the soft red glow behind the apple. */
    glow?: boolean;
    /** Apply a slight rotation. Mascot is most charming around -3deg. */
    tilt?: number;
    /** Optional alt text override. Default per DESIGN.md voice. */
    alt?: string;
    /** Source image. Defaults to the approved Tinker logo. */
    src?: string;
  }

  let {
    size = 320,
    glow = true,
    tilt = -3,
    alt = 'Tinker the apple, the mascot of Tinker',
    src = '/logo-mark.png',
  }: Props = $props();

  const sizeCss = $derived(typeof size === 'number' ? `${size}px` : size);
  // Natural intrinsic ratio of /logo-mark.png — the apple cropped out of
  // the full lockup. 432, 477 (slight portrait, ~0.89:1). Asset IS the
  // apple, so no aspect-ratio crop hack is needed.
  const NATURAL_W = 432;
  const NATURAL_H = 477;

  let root: HTMLDivElement | undefined = $state();
  let img: HTMLImageElement | undefined = $state();
  let bouncing = $state(false);
  let petCount = $state(0);
  let cursorTilt = $state(0);

  const PET_LS_KEY = 'tinker:pet-count';
  const PET_THRESHOLD = 10;
  const HOVER_TILT_RANGE_DEG = 3;

  let reducedMotion = $state(false);

  onMount(() => {
    try {
      reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch {
      /* ignore */
    }
    try {
      petCount = Number.parseInt(localStorage.getItem(PET_LS_KEY) ?? '0', 10) || 0;
    } catch {
      /* ignore */
    }
  });

  function onPointerMove(e: PointerEvent) {
    if (reducedMotion || !root) return;
    const r = root.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const offset = (e.clientX - cx) / (r.width / 2); // -1 left, +1 right
    cursorTilt = Math.max(-1, Math.min(1, offset)) * HOVER_TILT_RANGE_DEG;
  }

  function onPointerLeave() {
    cursorTilt = 0;
  }

  function persistPet(count: number) {
    try {
      localStorage.setItem(PET_LS_KEY, String(count));
    } catch {
      /* ignore */
    }
  }

  function onClick(e: MouseEvent) {
    bouncing = true;
    setTimeout(() => (bouncing = false), 240);

    play('tick');

    const target = (e.currentTarget as HTMLElement) ?? root;
    if (target) {
      petCount += 1;
      persistPet(petCount);
      const isMilestone = petCount > 0 && petCount % PET_THRESHOLD === 0;
      burst(target, {
        count: isMilestone ? 28 : 10,
        spread: isMilestone ? 1.6 : 1.1,
      });
    }
  }
</script>

<div
  bind:this={root}
  class="tinker"
  class:tinker--bouncing={bouncing}
  class:tinker--reduced={reducedMotion}
  style="--tinker-size: {sizeCss}; --tinker-tilt: {tilt}deg; --tinker-cursor-tilt: {cursorTilt}deg;"
  onpointermove={onPointerMove}
  onpointerleave={onPointerLeave}
  role="button"
  tabindex="0"
  aria-label="Tinker the apple. Click to play."
  onclick={onClick}
  onkeydown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onClick(e as unknown as MouseEvent);
    }
  }}
>
  {#if glow}
    <span class="tinker-glow" aria-hidden="true"></span>
  {/if}
  <img
    bind:this={img}
    {src}
    {alt}
    class="tinker-img"
    width={NATURAL_W}
    height={NATURAL_H}
    draggable="false"
  />
</div>

<style>
  .tinker {
    --tinker-size: 320px;
    --tinker-tilt: -3deg;
    --tinker-cursor-tilt: 0deg;
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: var(--tinker-size);
    /* /logo-mark.png is the apple, pre-cropped (432, 477). Container
       matches the asset aspect so the apple fills the box with zero
       letterbox and zero hidden overflow. */
    aspect-ratio: 432 / 477;
    height: auto;
    cursor: pointer;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
    border-radius: 12px;
    /* Bigger than the visible apple so click target feels generous. */
    padding: 0.5rem;
  }

  .tinker:focus-visible {
    outline: 3px solid var(--ink-red);
    outline-offset: 4px;
    border-radius: 999px;
  }

  .tinker-img {
    position: relative;
    z-index: 1;
    width: 100%;
    height: 100%;
    /* Asset IS the apple — no crop / no overflow. contain preserves the
       full apple at any container size; cover would do the same here
       because container aspect matches asset aspect, but contain is
       safer if the asset gets re-exported with different padding. */
    object-fit: contain;
    object-position: center;
    transform: rotate(calc(var(--tinker-tilt) + var(--tinker-cursor-tilt))) translateY(0);
    /* Drop-shadow tinted with apple red so the mascot reads as 3D-on-warm. */
    filter: drop-shadow(0 18px 32px rgba(230, 57, 106, 0.32));
    animation: tinker-bob 4.6s ease-in-out infinite;
    transition: transform 320ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }

  .tinker-glow {
    position: absolute;
    inset: -10%;
    border-radius: 50%;
    background: radial-gradient(
      circle,
      color-mix(in srgb, var(--ink-red) 38%, transparent) 0%,
      color-mix(in srgb, var(--ink-red) 12%, transparent) 45%,
      transparent 70%
    );
    filter: blur(18px);
    z-index: 0;
    pointer-events: none;
  }

  @keyframes tinker-bob {
    0%, 100% { translate: 0 0; }
    50%      { translate: 0 -6px; }
  }

  /* Click bounce — fires regardless of reduced-motion since it's a
     deliberate response to user input. Brief enough that vestibular-
     sensitive users won't be bothered. */
  .tinker--bouncing .tinker-img {
    animation:
      tinker-bob 4.6s ease-in-out infinite,
      tinker-bounce 240ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }

  @keyframes tinker-bounce {
    0%   { scale: 1; }
    35%  { scale: 1.06; }
    100% { scale: 1; }
  }

  /* Reduced motion: no bob, no cursor tilt. Click bounce stays. */
  .tinker--reduced .tinker-img {
    animation: none;
    transform: rotate(var(--tinker-tilt));
  }
  .tinker--reduced.tinker--bouncing .tinker-img {
    animation: tinker-bounce 240ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }
</style>
