/**
 * lib/celebrate.ts — orchestrator for earned-state celebrations.
 *
 * One async function per moment. Tuning the feel of any celebration
 * is a one-line edit here, not a sweep across StepCheck.astro,
 * Lesson.astro, and inline event listeners.
 *
 * Spec: docs/plans/2026-04-27-alive-layer-design.md §3, §4.
 */

import { Spring, Tween, prefersReducedMotion } from 'svelte/motion';
import { cubicOut } from 'svelte/easing';
import { play } from './sound';
import { burst } from './confetti';
import { awardXp, vibrate, XP_AWARD } from './xp';

const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export function announce(message: string) {
  if (typeof window === 'undefined') return;
  window.dispatchEvent(
    new CustomEvent('tinker:announce', { detail: { message } }),
  );
}

// Module-scoped Spring/Tween instances — one per surface so multiple
// concurrent celebrations don't fight each other.
export const progressSpring = new Spring(0, { stiffness: 0.12, damping: 0.4 });
export const scoreTween = new Tween(0, { duration: 900, easing: cubicOut });

export async function celebrateCorrect(checkBtn: HTMLElement) {
  play('ding');
  burst(checkBtn, { count: 6 });
  vibrate(20);
  awardXp(XP_AWARD.correctFirstTry, 'correct-first-try');
  checkBtn.classList.add('halo-pulse');
  await wait(320);
  checkBtn.classList.remove('halo-pulse');
  announce('Correct.');
}

export async function celebrateWrong(input: HTMLElement) {
  input.classList.add('wrong-shake');
  await wait(240);
  input.classList.remove('wrong-shake');
  announce('Try again. Read the hint.');
}

export async function celebrateLesson(card: HTMLElement) {
  if (prefersReducedMotion.current) {
    progressSpring.set(1, { instant: true });
  } else {
    progressSpring.set(1);
  }
  await wait(120);
  burst(card, { count: 14 });
  scoreTween.target = scoreTween.current + XP_AWARD.lessonComplete;
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent('tinker:celebrate', { detail: { level: 'lesson' } }),
    );
  }
  play('chime');
  vibrate(50);
  awardXp(XP_AWARD.lessonComplete, 'lesson-complete');
  card.classList.add('sweep-green');
  await wait(640);
  card.classList.remove('sweep-green');
  announce('Lesson complete. Plus 20 XP.');
}

export async function celebrateModule(
  card: HTMLElement,
  nodeId: string,
  nextModuleName?: string,
) {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent('tinker:celebrate', { detail: { level: 'module' } }),
    );
  }
  await wait(180);
  burst(card, { count: 28, spread: 1.6 });
  scoreTween.target = scoreTween.current + XP_AWARD.moduleComplete;
  play('anthem');
  vibrate([50, 100, 50]);
  awardXp(XP_AWARD.moduleComplete, 'module-complete');
  // TODO(alive-layer): glowMasteredNode(nodeId) once the map node component lands.
  void nodeId;
  const tail = nextModuleName ? ` ${nextModuleName} unlocked.` : '';
  announce(`Module complete. Plus 100 XP.${tail}`);
}
