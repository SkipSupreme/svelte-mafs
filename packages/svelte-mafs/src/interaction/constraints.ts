import type { Vec2 } from "../vec.js";

/**
 * A Constraint takes an unconstrained user position and returns the allowed
 * position. `<MovablePoint constrain={...}>` pipes every drag/keyboard
 * update through this function — snap-to-grid, stick-to-line, bound-to-box
 * are all just different Constraint factories.
 *
 * Constraints are pure functions (no context, no state) so they compose
 * trivially: `const both = (p) => snapToGrid(0.5)(clamp(bounds)(p));`
 */
export type Constraint = (pos: Vec2) => Vec2;

/**
 * Snap to the nearest multiple of `step`. Accepts a scalar for a uniform
 * grid or a [dx, dy] tuple for anisotropic grids.
 */
export const snapToGrid = (step: number | readonly [number, number]): Constraint => {
  const [sx, sy] = typeof step === "number" ? [step, step] : [step[0], step[1]];
  if (sx === 0 || sy === 0) {
    // A zero step would divide-by-zero in the round step; fail loudly so
    // typo'd props surface at construction, not at drag time.
    throw new Error("snapToGrid: step must be non-zero");
  }
  return ([x, y]) => [Math.round(x / sx) * sx, Math.round(y / sy) * sy];
};

/**
 * Project onto the infinite line through `a` and `b`. If a === b (degenerate
 * line, no direction) returns `a`.
 */
export const snapToLine = (a: Vec2, b: Vec2): Constraint => {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) {
    // Degenerate line — nothing to project onto. Collapse to the anchor so
    // the constrained point is at least well-defined.
    return () => [a[0], a[1]];
  }
  return ([px, py]) => {
    const t = ((px - a[0]) * dx + (py - a[1]) * dy) / lenSq;
    return [a[0] + t * dx, a[1] + t * dy];
  };
};

/**
 * Snap to the nearest sampled point on a parametric curve.
 *
 * Samples the curve at `samples + 1` evenly-spaced `t` values and returns
 * the sample nearest to the input point. Accuracy is bounded by sample
 * density — consumers who need sub-sample precision should do a local
 * refinement pass after this, or raise `samples`.
 */
export const snapToCurve = (
  f: (t: number) => Vec2,
  [tMin, tMax]: readonly [number, number],
  samples = 100,
): Constraint => {
  return (p) => {
    let best = f(tMin);
    let bestSq = distSq(p, best);
    const steps = Math.max(1, samples);
    for (let i = 1; i <= steps; i++) {
      const t = tMin + ((tMax - tMin) * i) / steps;
      const q = f(t);
      const d = distSq(p, q);
      if (d < bestSq) {
        bestSq = d;
        best = q;
      }
    }
    return [best[0], best[1]];
  };
};

/**
 * Clamp a point to an axis-aligned box. Axes are independent — omit `x` or
 * `y` to leave that axis untouched. This is the constraint-factory clamp,
 * distinct from the scalar `math.clamp(value, min, max)` that lives in
 * math.ts.
 */
export const clamp = (bounds: {
  x?: readonly [number, number];
  y?: readonly [number, number];
}): Constraint => {
  return ([x, y]) => [
    bounds.x ? clampScalar(x, bounds.x[0], bounds.x[1]) : x,
    bounds.y ? clampScalar(y, bounds.y[0], bounds.y[1]) : y,
  ];
};

const clampScalar = (v: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, v));

const distSq = (a: Vec2, b: Vec2): number => {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return dx * dx + dy * dy;
};
