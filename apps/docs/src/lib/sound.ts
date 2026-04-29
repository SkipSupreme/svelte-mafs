/**
 * Tinker sound palette.
 *
 * Was: 4 hand-rolled Web Audio sine tones. The synth pings sounded like a
 * 1980s alarm clock. Replaced with curated CC0 freesound mp3s in
 * `apps/docs/public/`. Same `play()` API; call sites unchanged.
 *
 * Mute toggle persists to localStorage (`tinker:sound-muted`). Default is
 * unmuted, EXCEPT when the user prefers reduced motion — then mute is on
 * by default per DESIGN.md §Sound rules.
 *
 * If audio fails to load or play (older browsers, autoplay block before
 * the first user gesture), each call is a silent no-op. We never throw.
 *
 * To swap a sound: drop a new mp3 into `apps/docs/public/` and edit the
 * PALETTE entry. Volume is 0..1; tweak per-sound to balance loudness
 * across the four events. The other 5 mp3s in /public/ are unused at
 * the moment but available for future events (woosh for transitions,
 * cartoon-jump for the mascot, etc).
 */

import { LS_KEY } from './storage-keys';

export type SoundName = 'tick' | 'ding' | 'chime' | 'anthem' | 'jump';

type SoundDef = { src: string; volume: number };

/**
 * Event → file mapping. Tones progress up the celebration ladder so
 * finishing a lesson sounds like finishing.
 */
const PALETTE: Record<SoundName, SoundDef> = {
  // Step advance, widget snap, continue click. Soft, frequent, doesn't
  // get annoying.
  tick: {
    src: '/humordome-soft-ui-pop-light-minimal-click-451232.mp3',
    volume: 0.45,
  },
  // Correct answer feedback. Literally named "ui correct button" upstream.
  ding: {
    src: '/freesound_community-ui_correct_button2-103167.mp3',
    volume: 0.55,
  },
  // Lesson complete. A reflective chime.
  chime: {
    src: '/freesound_community-melancholy-ui-chime-47804.mp3',
    volume: 0.6,
  },
  // Module complete. The biggest pay-off.
  anthem: {
    src: '/freesound_community-success-1-6297.mp3',
    volume: 0.6,
  },
  // Mascot bounce. Plays when the user clicks Tinker the apple.
  jump: {
    src: '/freesound_community-cartoon-jump-6462.mp3',
    volume: 0.5,
  },
};

let mutedCached: boolean | null = null;
const audios: Partial<Record<SoundName, HTMLAudioElement>> = {};

function readMuted(): boolean {
  if (mutedCached !== null) return mutedCached;
  if (typeof window === 'undefined') return true;
  try {
    const stored = localStorage.getItem(LS_KEY.soundMuted);
    if (stored !== null) {
      mutedCached = stored === '1';
      return mutedCached;
    }
  } catch {
    /* ignore */
  }
  // No explicit preference: respect prefers-reduced-motion.
  try {
    mutedCached = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  } catch {
    mutedCached = false;
  }
  return mutedCached;
}

export function isMuted(): boolean {
  return readMuted();
}

export function setMuted(value: boolean): void {
  mutedCached = value;
  try {
    localStorage.setItem(LS_KEY.soundMuted, value ? '1' : '0');
  } catch {
    /* ignore */
  }
}

export function toggleMuted(): boolean {
  const next = !readMuted();
  setMuted(next);
  return next;
}

function getAudio(name: SoundName): HTMLAudioElement | null {
  if (typeof window === 'undefined') return null;
  const cached = audios[name];
  if (cached) return cached;
  const def = PALETTE[name];
  try {
    const a = new Audio(def.src);
    a.volume = def.volume;
    a.preload = 'auto';
    audios[name] = a;
    return a;
  } catch {
    return null;
  }
}

export function play(name: SoundName): void {
  if (readMuted()) return;
  const a = getAudio(name);
  if (!a) return;
  // Reset to start so rapid back-to-back plays restart instead of being
  // ignored. Browsers may reject play() before the first user gesture;
  // silently swallow that — every Tinker trigger is itself a user gesture
  // so subsequent calls work.
  try {
    a.currentTime = 0;
    void a.play().catch(() => {});
  } catch {
    /* ignore */
  }
}
