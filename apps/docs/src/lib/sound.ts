/**
 * Tinker sound palette — Web Audio sine tones with attack/decay envelopes.
 * Spec lives in DESIGN.md §Sound. Tones progress up the scale so finishing
 * a lesson sounds like finishing.
 *
 * Mute toggle persists to localStorage (`tinker:sound-muted`). Default is
 * unmuted, EXCEPT when the user prefers reduced motion — then mute is on
 * by default per DESIGN.md §Sound rules.
 *
 * If AudioContext fails to construct (older Safari, locked-down env), every
 * call is a silent no-op. We never throw from this module.
 */

export type SoundName = 'tick' | 'ding' | 'chime' | 'anthem';

const LS_MUTED = 'tinker:sound-muted';

let ctx: AudioContext | null = null;
let mutedCached: boolean | null = null;

function getCtx(): AudioContext | null {
  if (ctx) return ctx;
  if (typeof window === 'undefined') return null;
  try {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    ctx = new Ctor();
    return ctx;
  } catch {
    return null;
  }
}

function readMuted(): boolean {
  if (mutedCached !== null) return mutedCached;
  if (typeof window === 'undefined') return true;
  try {
    const stored = localStorage.getItem(LS_MUTED);
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
    localStorage.setItem(LS_MUTED, value ? '1' : '0');
  } catch {
    /* ignore */
  }
}

export function toggleMuted(): boolean {
  const next = !readMuted();
  setMuted(next);
  return next;
}

function tone(freq: number, duration: number, startOffset = 0, gain = 0.18): void {
  const c = getCtx();
  if (!c) return;
  const osc = c.createOscillator();
  const env = c.createGain();
  osc.type = 'sine';
  osc.frequency.value = freq;
  const t = c.currentTime + startOffset;
  env.gain.setValueAtTime(0, t);
  env.gain.linearRampToValueAtTime(gain, t + 0.005);
  env.gain.exponentialRampToValueAtTime(0.001, t + duration);
  osc.connect(env).connect(c.destination);
  osc.start(t);
  osc.stop(t + duration + 0.05);
}

export function play(name: SoundName): void {
  if (readMuted()) return;
  const c = getCtx();
  if (!c) return;
  // Browsers may suspend the context before the first user gesture. The
  // first call to play() is itself triggered by a click/keydown, so resume
  // is safe here.
  if (c.state === 'suspended') {
    void c.resume().catch(() => {});
  }
  switch (name) {
    case 'tick':
      tone(880, 0.06, 0, 0.14);
      return;
    case 'ding':
      tone(880, 0.12, 0, 0.16);
      tone(1320, 0.12, 0, 0.12);
      return;
    case 'chime':
      tone(660, 0.18, 0.0, 0.15);
      tone(880, 0.18, 0.13, 0.15);
      tone(1320, 0.22, 0.26, 0.13);
      return;
    case 'anthem':
      tone(523, 0.28, 0.0, 0.16);
      tone(659, 0.28, 0.22, 0.16);
      tone(784, 0.28, 0.44, 0.16);
      tone(1047, 0.36, 0.66, 0.14);
      return;
  }
}
