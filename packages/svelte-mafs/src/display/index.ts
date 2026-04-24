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

// Stream 6 — text + affine transforms
export { default as Text } from "./Text.svelte";
export { default as Transform } from "./Transform.svelte";
// Matrix helpers as a namespace so `Matrix.rotate(...)` can coexist with
// vec.ts's `rotate`, `scale`, `translate` at the top-level entrypoint.
// Sourced from matrix.ts rather than Transform.svelte.ts to avoid the
// TypeScript bundler-resolution conflict on `./Transform.svelte`.
export * as Matrix from "./matrix.js";
