/**
 * Math-symbol confetti burst with sparkles. Tinker is math-as-play, so
 * the celebration is the alphabet of math itself plus a few stars for
 * pure delight: π ∫ ∂ ∇ ∞ ∑ √ Δ x + = ★ ✦ ✧ …
 *
 * The five-color palette pulls every confetti hue defined in DESIGN.md
 * (teal / pink / orange / sun / red) so a single burst already feels
 * like a fistful of joy. Sizes range from a small 16px to a chunky 42px
 * so the spread reads as varied, not uniform.
 *
 * Usage:
 *   import { burst } from '../lib/confetti';
 *   burst(buttonElement);                  // burst from element center
 *   burst({ x: 100, y: 200 });             // burst from absolute viewport coords
 *   burst(el, { count: 22, spread: 1.5 }); // bigger, wider burst
 *
 * Symbols are appended to a single fixed-position layer on document.body
 * and self-clean after their animation. No-op under prefers-reduced-motion.
 */

const COLORS = [
  'var(--ink-teal)',
  'var(--ink-pink)',
  'var(--ink-orange)',
  'var(--ink-sun)',
  'var(--ink-red)',
];

// Math symbols + stars/sparkles. Stars get drawn via Fraunces too — they
// render fine as Unicode geometric glyphs in any serif fallback.
const SYMBOLS = [
  // Calculus & analysis
  'π', '∫', '∂', '∇', '∞', '∑', '√', 'Δ', 'θ', 'λ', 'α', 'β',
  // Operators & relations
  '+', '−', '×', '÷', '=', '≈', '≠',
  // Variables
  'x', 'y', 'ƒ',
  // Set theory & arrows
  '∈', '⊂', '∩', '∪', '→', '↔',
  // Pure delight
  '★', '✦', '✧', '✺', '❋', '✿',
];

// Render some symbols a notch bigger to break the visual rhythm. These
// are "headline" symbols — they read at a glance.
const HEADLINE = new Set(['π', '∫', '∞', '∑', 'Δ', '★', '✦', '✺', '❋']);

const LAYER_ID = 'tinker-confetti-layer';

export interface BurstOptions {
  /** Number of symbols. Default 12 (a "ding" burst). */
  count?: number;
  /** Spread multiplier on travel distance. Default 1.0. */
  spread?: number;
  /** Animation duration in ms. Default ~960 jittered. */
  duration?: number;
}

export type BurstTarget = HTMLElement | { x: number; y: number };

function ensureLayer(): HTMLElement | null {
  if (typeof document === 'undefined') return null;
  let layer = document.getElementById(LAYER_ID);
  if (layer) return layer;
  layer = document.createElement('div');
  layer.id = LAYER_ID;
  layer.setAttribute('aria-hidden', 'true');
  layer.style.cssText =
    'position:fixed;inset:0;pointer-events:none;z-index:99;overflow:visible;';
  document.body.appendChild(layer);
  return layer;
}

function originOf(target: BurstTarget): { x: number; y: number } {
  if (target instanceof HTMLElement) {
    const r = target.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }
  return { x: target.x, y: target.y };
}

function pick<T>(arr: readonly T[], i: number): T {
  // Jittered index so a small burst still feels random, not striped.
  const j = (i + Math.floor(Math.random() * arr.length)) % arr.length;
  return arr[j];
}

export function burst(target: BurstTarget, options: BurstOptions = {}): void {
  if (typeof window === 'undefined') return;
  try {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  } catch {
    /* ignore */
  }
  const layer = ensureLayer();
  if (!layer) return;

  const count = options.count ?? 12;
  const spread = options.spread ?? 1;
  const baseDuration = options.duration ?? 1500;
  const { x, y } = originOf(target);

  for (let i = 0; i < count; i++) {
    const sym = document.createElement('span');
    const color = pick(COLORS, i);
    const symbol = pick(SYMBOLS, i);
    // Headline symbols render beefier — 28-42px. Regular symbols 18-30px.
    // The result is a deliberate size mix instead of uniform-stripe boring.
    const isHeadline = HEADLINE.has(symbol);
    const fontSize = isHeadline ? 28 + Math.random() * 14 : 18 + Math.random() * 12;
    const fontWeight = isHeadline ? 700 : 600;
    sym.textContent = symbol;
    sym.style.cssText =
      `position:absolute;left:${x}px;top:${y}px;` +
      `font-family: var(--font-display, 'Fraunces'), serif;` +
      `font-style: italic; font-weight: ${fontWeight};` +
      `font-size: ${fontSize}px; line-height: 1;` +
      `color: ${color};` +
      // Stronger glow than dots had — feels alive, not flat.
      `text-shadow: 0 0 12px color-mix(in srgb, ${color} 55%, transparent),` +
      ` 0 2px 4px color-mix(in srgb, ${color} 35%, transparent);` +
      `pointer-events:none; will-change: transform, opacity;` +
      `transform: translate(-50%, -50%); user-select: none;`;
    layer.appendChild(sym);

    // Three-phase trajectory: BURST outward + up, HANG for a moment,
    // then FALL with gravity. Per-keyframe easing gives the right physics:
    // ease-out for the launch (fast → slow), ease-in for the fall
    // (slow → fast, just like real gravity).
    const angle = (Math.PI * 2 * i) / count + (Math.random() * 0.6 - 0.3);
    const horizontal = (110 + Math.random() * 90) * spread;
    const lift = (50 + Math.random() * 90) * spread;
    const peakX = Math.cos(angle) * horizontal;
    const peakY = Math.sin(angle) * horizontal - lift;
    // Where it ends up after gravity: drift slightly sideways + fall a lot.
    const fallDrift = (Math.random() - 0.5) * 80;
    const fallDistance = 320 + Math.random() * 260;
    const fallX = peakX + fallDrift;
    const fallY = peakY + fallDistance;
    const rot = Math.random() * 720 - 360;
    const rotPeak = rot * 0.45;
    const duration = baseDuration + Math.random() * 500;

    const animation = sym.animate(
      [
        {
          transform: 'translate(-50%, -50%) scale(0.4) rotate(0deg)',
          opacity: 1,
          easing: 'cubic-bezier(0.18, 0.7, 0.3, 1)',
        },
        // Peak — outward and up. Burst is mostly done by 30%.
        {
          offset: 0.3,
          transform: `translate(calc(-50% + ${peakX}px), calc(-50% + ${peakY}px)) scale(1.1) rotate(${rotPeak}deg)`,
          opacity: 1,
          easing: 'cubic-bezier(0.5, 0, 0.7, 0.45)',
        },
        // Hang for a beat — symbol mostly stationary, slight drift.
        {
          offset: 0.45,
          transform: `translate(calc(-50% + ${peakX * 1.03}px), calc(-50% + ${peakY * 0.92}px)) scale(1) rotate(${rotPeak * 1.15}deg)`,
          opacity: 1,
          easing: 'cubic-bezier(0.55, 0, 1, 0.5)',
        },
        // Fall — gravity accelerates it down and slightly off-screen.
        {
          offset: 1,
          transform: `translate(calc(-50% + ${fallX}px), calc(-50% + ${fallY}px)) scale(0.85) rotate(${rot}deg)`,
          opacity: 0,
        },
      ],
      { duration, fill: 'forwards' },
    );
    animation.onfinish = () => sym.remove();
    // Safety net in case onfinish doesn't fire (e.g. tab backgrounded).
    setTimeout(() => sym.remove(), duration + 300);
  }
}
