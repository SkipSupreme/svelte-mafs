// Public entry point for svelte-mafs.
//
// Each stream appends its own re-exports below. Keep this file additive —
// captain serializes merges so multiple PRs can touch it safely.

export const VERSION = "0.0.0";

// Stream 1 — pure math core
export type { Vec2 } from "./vec.js";
export { add, sub, scale, dot, mag, normalize, rotate, lerp } from "./vec.js";

export {
  clamp,
  round,
  mapRange,
  nearestPowerOfTen,
  snapAngleToDegrees,
  inferLabels,
} from "./math.js";

export type { Sample, SampleOptions } from "./sampling.js";
export { sample } from "./sampling.js";

// Stream 2 — coordinate context + root view
export { default as Mafs } from "./view/Mafs.svelte";
export type { CoordContext } from "./context/coordinate-context.js";

// Stream 3 — gesture actions (use:drag, use:panZoom)
export type { DragOptions, PanZoomOptions } from "./gestures/index.js";
export { drag, panZoom } from "./gestures/index.js";

// Stream 4 — static display primitives
export {
  Point,
  Line,
  Circle,
  Ellipse,
  Polygon,
  Vector,
} from "./display/index.js";

// Stream 5 — Coordinates + Plot namespaces
export { Coordinates, Plot } from "./display/index.js";

// Stream 6 — interaction + display + theme
export { Text, Transform, Matrix } from "./display/index.js";
export {
  MovablePoint,
  snapToGrid,
  snapToLine,
  snapToCurve,
  type Constraint,
} from "./interaction/index.js";
// Re-exported as `clampToBox` to avoid shadowing math.ts's scalar
// `clamp(value, min, max)`. Shape here is `(bounds) => (p) => p'` — a
// constraint factory, not a scalar op. Captain: deviates from the
// literal plan text (`clamp`) due to the name collision.
export { clamp as clampToBox } from "./interaction/index.js";
// Theme tokens — `colors.blue` resolves to `var(--mafs-blue)` so
// components can pass it straight into `fill=` / `stroke=`.
export { colors, fonts, theme } from "./theme.js";
