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
 */
export const TINKER_EVENT = {
  xp: 'tinker:xp',
  streak: 'tinker:streak',
  announce: 'tinker:announce',
  celebrate: 'tinker:celebrate',
  stuck: 'tinker:stuck',
} as const;

export type TinkerEventName = (typeof TINKER_EVENT)[keyof typeof TINKER_EVENT];
