/**
 * localStorage keys used by Tinker. Centralized so a future rename can't
 * slip through grep.
 *
 * These values are stable user data — never change a string here without
 * shipping a migration that copies the old key forward.
 */
export const LS_KEY = {
  xp: 'tinker:xp',
  streak: 'tinker:streak',
  streakMax: 'tinker:streak-max',
  streakLastDay: 'tinker:streak-last-day',
  soundMuted: 'tinker:sound-muted',
  stuckLastShown: 'tinker:stuck-last-shown',
} as const;

export const lessonUnlockedKey = (slug: string) => `tinker:lesson:${slug}:unlocked`;
export const lessonCompletedKey = (slug: string) => `tinker:lesson:${slug}:completed`;
