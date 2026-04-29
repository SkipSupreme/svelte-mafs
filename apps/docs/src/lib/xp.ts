/**
 * XP, streak, and haptic helpers. Spec in DESIGN.md §Progress Loops.
 *
 * XP awards (per DESIGN.md):
 *   step advance              =   1 XP
 *   correct first try         =   5 XP
 *   correct after 1-2 retries =   2 XP
 *   lesson complete           =  20 XP
 *   module complete           = 100 XP
 *
 * Streak counts consecutive UTC days with at least one completed step.
 * Sundays never break a streak. Per DESIGN.md, streak display is opt-in
 * after the user's first 30 days — we count silently, but the bumpStreak()
 * helper is wired up for when that opt-in flow ships.
 *
 * Events: `tinker:xp` (detail: { amount, total, reason }) and
 * `tinker:streak` (detail: { streak, max }) fire on window so Nav and
 * other listeners can update live without polling localStorage.
 */

import { TINKER_EVENT } from './events';
import { LS_KEY } from './storage-keys';

export const XP_AWARD = {
  stepAdvance: 1,
  correctFirstTry: 5,
  correctRetry: 2,
  lessonComplete: 20,
  moduleComplete: 100,
} as const;

function readInt(key: string, fallback = 0): number {
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = localStorage.getItem(key);
    const n = raw == null ? NaN : parseInt(raw, 10);
    return Number.isFinite(n) ? n : fallback;
  } catch {
    return fallback;
  }
}

function writeInt(key: string, value: number): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(key, String(value));
  } catch {
    /* ignore quota / private mode */
  }
}

function dayStringUTC(d = new Date()): string {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, '0');
  const day = String(d.getUTCDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function emit<T>(name: 'xp' | 'streak', detail: T): void {
  if (typeof window === 'undefined') return;
  const eventName = name === 'xp' ? TINKER_EVENT.xp : TINKER_EVENT.streak;
  window.dispatchEvent(new CustomEvent(eventName, { detail }));
}

export function getXp(): number {
  return readInt(LS_KEY.xp);
}

export function getStreak(): number {
  return readInt(LS_KEY.streak);
}

export function getMaxStreak(): number {
  return readInt(LS_KEY.streakMax);
}

export interface XpEventDetail {
  amount: number;
  total: number;
  reason: string;
}

export function awardXp(amount: number, reason: string): number {
  if (!Number.isFinite(amount) || amount === 0) return getXp();
  const total = Math.max(0, getXp() + amount);
  writeInt(LS_KEY.xp, total);
  emit<XpEventDetail>('xp', { amount, total, reason });
  return total;
}

export interface StreakEventDetail {
  streak: number;
  max: number;
  newDay: boolean;
}

/**
 * Increment streak if today is a new UTC day. Returns the new streak and
 * whether this call actually advanced it.
 *
 * NOTE: not yet called from Lesson.astro. DESIGN.md gates streak display
 * on a 30-day-then-opt-in prompt that hasn't shipped. Once that flow
 * exists, call this on first completed step of the day.
 */
export function bumpStreak(): StreakEventDetail {
  const today = dayStringUTC();
  let last: string | null = null;
  try {
    last = localStorage.getItem(LS_KEY.streakLastDay);
  } catch {
    /* ignore */
  }
  if (last === today) {
    const streak = getStreak();
    const max = getMaxStreak();
    return { streak, max, newDay: false };
  }
  const yesterday = (() => {
    const d = new Date();
    d.setUTCDate(d.getUTCDate() - 1);
    return dayStringUTC(d);
  })();
  const continuing = last === yesterday;
  const streak = continuing ? getStreak() + 1 : 1;
  writeInt(LS_KEY.streak, streak);
  const max = Math.max(getMaxStreak(), streak);
  writeInt(LS_KEY.streakMax, max);
  try {
    localStorage.setItem(LS_KEY.streakLastDay, today);
  } catch {
    /* ignore */
  }
  emit<StreakEventDetail>('streak', { streak, max, newDay: true });
  return { streak, max, newDay: true };
}

/**
 * Mobile haptic feedback. No-op when reduced motion is preferred or when
 * the platform doesn't support vibrate. Patterns from DESIGN.md §Haptics.
 */
export function vibrate(pattern: number | number[]): void {
  if (typeof window === 'undefined') return;
  try {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  } catch {
    /* ignore */
  }
  if (typeof navigator === 'undefined' || !('vibrate' in navigator)) return;
  try {
    navigator.vibrate(pattern);
  } catch {
    /* ignore */
  }
}
