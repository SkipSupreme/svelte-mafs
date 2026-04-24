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

// Stream 3 — gesture actions (use:drag, use:panZoom)
export type { DragOptions, PanZoomOptions } from "./gestures/index.js";
export { drag, panZoom } from "./gestures/index.js";
