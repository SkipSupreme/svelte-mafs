<script lang="ts">
  /**
   * Tinker the Apple — the live mascot. Spec in DESIGN.md §Mascot.
   *
   * Idle behavior: always-on bob (±6px) + cursor-aware tilt. Every 6–18s
   * Tinker also self-initiates a tiny wiggle or a hop so the mascot reads
   * as alive instead of merely animated. Idle actions pause while the user
   * is touching the page, while sleeping, or under reduced motion — and
   * are silent (no jump sound), since they aren't user-driven.
   *
   * Click triggers a bounce, the cartoon-jump sound, and a math-symbol
   * burst. Every 10th click is a bigger milestone burst. Reduced motion
   * freezes ambient animation; the click bounce still fires because it's
   * a deliberate user response.
   */
  import { onMount } from 'svelte';
  import { prefersReducedMotion } from 'svelte/motion';
  import { play } from '../../lib/sound';
  import { burst } from '../../lib/confetti';
  import { eggs } from '../../lib/easterEggs.svelte';

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
    /** Server-passed: render the birthday party hat. */
    birthday?: boolean;
    /** Server-passed: render one tiny grad-cap badge per completed course (cap at 5). */
    completedCourses?: number;
  }

  let {
    size = 320,
    glow = true,
    tilt = -3,
    alt = 'Tinker the apple, the mascot of Tinker',
    src = '/logo-mark.png',
    birthday = false,
    completedCourses = 0,
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
  let sleeping = $state(false);
  let idleAction = $state<'none' | 'wiggle' | 'hop' | 'lean'>('none');

  const PET_LS_KEY = 'tinker:pet-count';
  const PET_THRESHOLD = 10;
  const HOVER_TILT_RANGE_DEG = 3;
  const SLEEP_MS = 60_000;
  // Random window for the next self-initiated idle action.
  const IDLE_MIN_MS = 6_000;
  const IDLE_MAX_MS = 18_000;
  let sleepTimer: number | null = null;
  let idleTimer: number | null = null;
  let idleClearTimer: number | null = null;

  const reducedMotion = $derived(prefersReducedMotion.current);

  function resetSleep() {
    sleeping = false;
    if (sleepTimer) clearTimeout(sleepTimer);
    sleepTimer = window.setTimeout(() => {
      sleeping = true;
    }, SLEEP_MS);
    // Push idle action back too — Tinker shouldn't twitch while the user
    // is touching the page.
    scheduleIdleAction();
  }

  function scheduleIdleAction() {
    if (idleTimer) clearTimeout(idleTimer);
    if (idleClearTimer) clearTimeout(idleClearTimer);
    if (typeof window === 'undefined') return;
    if (reducedMotion) return;
    const delay = IDLE_MIN_MS + Math.random() * (IDLE_MAX_MS - IDLE_MIN_MS);
    idleTimer = window.setTimeout(fireIdleAction, delay);
  }

  function fireIdleAction() {
    if (sleeping || reducedMotion || bouncing) {
      scheduleIdleAction();
      return;
    }
    // Weighted pick: small wiggle most often, hop occasionally, lean rarely.
    const r = Math.random();
    const action: 'wiggle' | 'hop' | 'lean' =
      r < 0.55 ? 'wiggle' : r < 0.85 ? 'hop' : 'lean';
    idleAction = action;
    const duration = action === 'lean' ? 1600 : action === 'hop' ? 720 : 520;
    idleClearTimer = window.setTimeout(() => {
      idleAction = 'none';
      scheduleIdleAction();
    }, duration);
  }

  $effect(() => {
    resetSleep();
    scheduleIdleAction();
    const evts = ['pointermove', 'keydown', 'scroll'] as const;
    for (const ev of evts) document.addEventListener(ev, resetSleep, { passive: true });
    return () => {
      if (sleepTimer) clearTimeout(sleepTimer);
      if (idleTimer) clearTimeout(idleTimer);
      if (idleClearTimer) clearTimeout(idleClearTimer);
      for (const ev of evts) document.removeEventListener(ev, resetSleep);
    };
  });

  onMount(() => {
    try {
      petCount = Number.parseInt(localStorage.getItem(PET_LS_KEY) ?? '0', 10) || 0;
    } catch {
      /* ignore */
    }
    eggs.init();
  });

  $effect(() => {
    // bouncePulse is reactive; reading it here subscribes the effect.
    if (eggs.bouncePulse === 0) return; // skip initial 0
    bouncing = true;
    setTimeout(() => (bouncing = false), 240);
    if (root) burst(root, { count: 8 });
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

    play('jump');

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
  class:tinker--sleeping={sleeping && !reducedMotion}
  class:tinker--wiggle={idleAction === 'wiggle'}
  class:tinker--hop={idleAction === 'hop'}
  class:tinker--lean={idleAction === 'lean'}
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
  <svg class="tinker-sparkles" viewBox="0 0 432 477" aria-hidden="true">
    <g class="sp sp-1" style="--c: var(--ink-orange);">
      <path d="M0,-10 L2,-2 L10,0 L2,2 L0,10 L-2,2 L-10,0 L-2,-2 Z" transform="translate(110, 96)" />
    </g>
    <g class="sp sp-2" style="--c: var(--ink-teal);">
      <path d="M0,-8 L2,-2 L8,0 L2,2 L0,8 L-2,2 L-8,0 L-2,-2 Z" transform="translate(196, 70)" />
    </g>
    <g class="sp sp-3" style="--c: var(--ink-red);">
      <path d="M0,-9 L2,-2 L9,0 L2,2 L0,9 L-2,2 L-9,0 L-2,-2 Z" transform="translate(286, 100)" />
    </g>
  </svg>
  {#if eggs.sunglasses}
    <svg class="acc acc-sunglasses" viewBox="0 0 432 477" aria-hidden="true">
      <g transform="translate(140, 200)">
        <rect x="0"  y="0" width="60" height="36" rx="14" fill="#1a1a1a" />
        <rect x="92" y="0" width="60" height="36" rx="14" fill="#1a1a1a" />
        <rect x="58" y="14" width="36" height="6"            fill="#1a1a1a" />
      </g>
    </svg>
  {/if}

  {#if birthday}
    <svg class="acc acc-hat" viewBox="0 0 432 477" aria-hidden="true">
      <polygon points="180,40 250,40 215,-30" fill="var(--ink-pink)" />
      <circle cx="215" cy="-30" r="8" fill="var(--ink-orange)" />
    </svg>
  {/if}

  {#each Array.from({ length: Math.min(completedCourses, 5) }) as _, i (i)}
    <svg class="acc acc-grad" viewBox="0 0 432 477" aria-hidden="true">
      <g transform="translate({90 + i * 50}, 380)">
        <rect x="0"  y="6"  width="34" height="6" fill="#1a1a1a" />
        <polygon points="-4,6 38,6 17,-4" fill="#1a1a1a" />
      </g>
    </svg>
  {/each}
  {#if sleeping && !reducedMotion}
    <span class="tinker-z" aria-hidden="true">z</span>
  {/if}
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

  /* === Self-initiated idle behaviors ===
     Layered on top of the perpetual bob via the `rotate` and `scale`
     properties (independent of `transform`/`translate`, which are already
     used by the static tilt and the bob). Silent — no sound on these,
     since they aren't user-driven. */

  /* Wiggle — quick "what was that" head shake. Most common idle action. */
  .tinker--wiggle .tinker-img {
    animation:
      tinker-bob 4.6s ease-in-out infinite,
      tinker-wiggle 520ms ease-in-out;
  }
  @keyframes tinker-wiggle {
    0%, 100% { rotate: 0deg; }
    20%      { rotate: -6deg; }
    50%      { rotate: 5deg; }
    80%      { rotate: -3deg; }
  }

  /* Hop — vertical jump with squash/stretch. The bob keeps running underneath
     (it uses `translate`); the hop adds an additive translateY via a
     wrapper-style scale + raise. Bigger than the click bounce. */
  .tinker--hop .tinker-img {
    animation:
      tinker-bob 4.6s ease-in-out infinite,
      tinker-hop 720ms cubic-bezier(0.34, 1.4, 0.64, 1);
  }
  @keyframes tinker-hop {
    0%   { scale: 1; rotate: 0deg; }
    20%  { scale: 1.08 0.94; rotate: 0deg; }
    45%  { scale: 0.96 1.06; rotate: -3deg; }
    70%  { scale: 1.02 0.98; rotate: 2deg; }
    100% { scale: 1; rotate: 0deg; }
  }

  /* Lean — slow look-around, side to side. Rare. */
  .tinker--lean .tinker-img {
    animation:
      tinker-bob 4.6s ease-in-out infinite,
      tinker-lean 1600ms ease-in-out;
  }
  @keyframes tinker-lean {
    0%, 100% { rotate: 0deg; }
    35%      { rotate: -7deg; }
    65%      { rotate: 6deg; }
  }

  /* Reduced motion: no bob, no cursor tilt, no idle actions. Click
     bounce stays — it's a deliberate user response. */
  .tinker--reduced .tinker-img {
    animation: none;
    transform: rotate(var(--tinker-tilt));
  }
  .tinker--reduced.tinker--bouncing .tinker-img {
    animation: tinker-bounce 240ms cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .tinker--reduced.tinker--wiggle .tinker-img,
  .tinker--reduced.tinker--hop .tinker-img,
  .tinker--reduced.tinker--lean .tinker-img {
    animation: none;
    transform: rotate(var(--tinker-tilt));
  }

  .tinker-sparkles {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 2;
  }

  .acc {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 4;
  }
  .tinker-sparkles .sp {
    fill: var(--c);
    opacity: 0;
    animation: sparkle-flicker 4.6s ease-in-out infinite;
  }
  .tinker-sparkles .sp-1 { animation-delay: 0s; }
  .tinker-sparkles .sp-2 { animation-delay: 1.5s; }
  .tinker-sparkles .sp-3 { animation-delay: 3s; }
  .tinker--reduced .tinker-sparkles .sp { animation: none; opacity: 0.6; }

  .tinker-z {
    position: absolute;
    top: -8%;
    right: 14%;
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-weight: 600;
    font-size: 1.6rem;
    color: var(--site-fg-muted);
    animation: tinker-z-rise 2.4s ease-out infinite;
    z-index: 3;
  }
  .tinker--sleeping .tinker-img {
    animation-duration: 7s;
  }
</style>
