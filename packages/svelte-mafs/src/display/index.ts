// display/ barrel.
//
// Append-only — captain serializes merges across streams. Add new streams'
// exports below, never reorder or remove existing lines.

// Stream 4 — static display primitives
export { default as Point } from "./Point.svelte";
export * as Line from "./Line.svelte.js";
export { default as Circle } from "./Circle.svelte";
export { default as Ellipse } from "./Ellipse.svelte";
export { default as Polygon } from "./Polygon.svelte";
export { default as Vector } from "./Vector.svelte";

// Stream 5 — Coordinates + Plot namespaces
export { Coordinates } from "./Coordinates.svelte.js";
export { Plot } from "./Plot.svelte.js";
