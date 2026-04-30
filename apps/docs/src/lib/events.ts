/**
 * Window event names used across Tinker's progress + celebration pipeline.
 * Centralized so a future rename can't slip through grep.
 *
 * Payload shapes are documented inline next to the canonical emitter:
 *   - tinker:xp        → see XpEventDetail in lib/xp.ts
 *   - tinker:streak    → see StreakEventDetail in lib/xp.ts
 *   - tinker:announce  → { message: string }       (lib/celebrate.ts)
 *   - tinker:celebrate → { level: 'step' | 'lesson' | 'module' }
 *   - tinker:stuck     → { hint: string }          (layouts/Lesson.astro)
 *
 * Hero-region events (homepage-only, scoped to the hero region's parent —
 * NOT window-level). Used by the reactive apple to watch what the visitor
 * does to the hero widget. See HeroFocusDetail / HeroDragDetail /
 * HeroThresholdDetail below.
 */
export const TINKER_EVENT = {
  xp: 'tinker:xp',
  streak: 'tinker:streak',
  announce: 'tinker:announce',
  celebrate: 'tinker:celebrate',
  stuck: 'tinker:stuck',
} as const;

export type TinkerEventName = (typeof TINKER_EVENT)[keyof typeof TINKER_EVENT];

/**
 * Hero-region event names. Scoped DOM events bubbled on the hero region's
 * parent. Tinker.svelte attaches a listener (with a small replay buffer for
 * events that fire before hydration completes — Astro islands hydrate
 * independently).
 *
 * Coordinates in HeroFocusDetail / HeroDragDetail are normalized to the
 * hero region's bounding box: (0,0) is top-left, (1,1) is bottom-right.
 * This keeps the apple's reaction layout-independent.
 */
export const TINKER_HERO_EVENT = {
  focus: 'tinker:hero:focus',
  drag: 'tinker:hero:drag',
  threshold: 'tinker:hero:threshold',
  success: 'tinker:hero:success',
  idle: 'tinker:hero:idle',
} as const;

export type TinkerHeroEventName = (typeof TINKER_HERO_EVENT)[keyof typeof TINKER_HERO_EVENT];

/** `tinker:hero:focus` — visitor's pointer / drag handle position changed. */
export interface HeroFocusDetail {
  /** Normalized x in [0, 1] relative to hero region bounds. */
  x: number;
  /** Normalized y in [0, 1] relative to hero region bounds. */
  y: number;
  /** Optional semantic region the visitor is hovering over (e.g. a cluster name). */
  region?: string;
}

/** `tinker:hero:drag` — visitor is actively dragging an object in the widget. */
export interface HeroDragDetail {
  /** Normalized x in [0, 1] relative to hero region bounds. */
  x: number;
  /** Normalized y in [0, 1] relative to hero region bounds. */
  y: number;
  /** Optional id of the dragged element (e.g. a letter in EmbeddingPluck). */
  handle?: string;
  /** Drag phase. `start` and `end` fire once; `move` fires on every input. */
  phase: 'start' | 'move' | 'end';
}

/** `tinker:hero:threshold` — meaningful state change worth a face shift. */
export interface HeroThresholdDetail {
  /** Short string identifying which threshold crossed (e.g. 'cluster-entered'). */
  kind: string;
  /** Optional human-readable label for debugging / telemetry. */
  label?: string;
}

/** `tinker:hero:success` — visitor reached a milestone worth celebrating. */
export interface HeroSuccessDetail {
  /** What was achieved (e.g. 'first-drag', 'all-vowels-clustered'). */
  milestone: string;
}

/**
 * Helper for emitting hero events from a widget. Dispatches a CustomEvent
 * with `bubbles: true`, `composed: false` (no need to cross shadow DOM).
 * The widget passes its closest hero-region ancestor; the listener on
 * Tinker.svelte (or its parent) catches it.
 */
export function emitHeroEvent<T>(target: Element, name: TinkerHeroEventName, detail: T): void {
  target.dispatchEvent(new CustomEvent(name, { detail, bubbles: true, composed: false }));
}
