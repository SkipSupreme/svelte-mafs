/**
 * 2D affine matrix helpers for <Transform>.
 *
 * Originally named `Transform.svelte.ts` per the plan's namespace
 * convention, but TypeScript's "Bundler" resolver prefers a same-basename
 * `.ts` over Svelte's `*.svelte` wildcard shim — which makes
 * `import X from "./Transform.svelte"` resolve here instead of to the
 * component. The rename sidesteps that ambiguity; the module is still
 * co-located with Transform.svelte.
 *
 * We store a 3x3 affine matrix as the 6-tuple SVG consumes directly:
 *
 *     [ a c e ]
 *     [ b d f ]
 *     [ 0 0 1 ]
 *
 * which maps a point (x, y) to (a*x + c*y + e, b*x + d*y + f). Packing the
 * math into a single tuple means `compose(a, b, c)` just folds via matrix
 * multiply, and serialization is a one-line `matrix(a, b, c, d, e, f)`.
 *
 * Rotations follow the math-y-up convention (positive θ is
 * counter-clockwise). That matches what a user sees inside <Mafs>, whose
 * root <g scale(1, -1)> has already flipped screen-y to math-y.
 */

export type Mat3 = readonly [
  a: number,
  b: number,
  c: number,
  d: number,
  e: number,
  f: number,
];

export const identity: Mat3 = [1, 0, 0, 1, 0, 0];

export const translate = (tx: number, ty: number): Mat3 => [1, 0, 0, 1, tx, ty];

export const scale = (sx: number, sy: number = sx): Mat3 => [sx, 0, 0, sy, 0, 0];

export const rotate = (theta: number): Mat3 => {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c, s, -s, c, 0, 0];
};

/**
 * Compose matrices left-to-right in application order. `compose(A, B)`
 * produces A then B — equivalently, the SVG transform `<g transform="A B">`
 * (read left-to-right: outermost first, innermost last).
 *
 * Concretely: for a point p, `compose(A, B)(p) = B(A(p))`.
 */
export const compose = (...matrices: readonly Mat3[]): Mat3 => {
  if (matrices.length === 0) return identity;
  return matrices.reduce((acc, m) => multiply(acc, m));
};

const multiply = (m: Mat3, n: Mat3): Mat3 => {
  const [a, b, c, d, e, f] = m;
  const [A, B, C, D, E, F] = n;
  // Compose so that (m then n) is equivalent to n ∘ m (apply m first, then n).
  // SVG's nested <g transform="m"><g transform="n"> would apply m outermost
  // (first) — our `compose(m, n)` mirrors that reading order.
  return [
    A * a + C * b,
    B * a + D * b,
    A * c + C * d,
    B * c + D * d,
    A * e + C * f + E,
    B * e + D * f + F,
  ];
};

/**
 * Format a matrix as SVG's `matrix(a, b, c, d, e, f)`. Near-zero values
 * collapse to `0` so `rotate(π/2)` produces a clean `matrix(0, 1, -1, 0, 0,
 * 0)` instead of `matrix(6.12e-17, 1, -1, 6.12e-17, 0, 0)` — the former
 * snapshots cleanly and renders identically.
 */
export const toMatrixString = (m: Mat3): string => {
  const parts = m.map(fmt).join(", ");
  return `matrix(${parts})`;
};

const EPSILON = 1e-12;
const fmt = (n: number): string => {
  if (Math.abs(n) < EPSILON) return "0";
  // Use default toString for stable exact integers, fall back to 12 sig figs
  // for irrationals (enough precision for any realistic viewBox scale).
  const s = n.toString();
  return s;
};

