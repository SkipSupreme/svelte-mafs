import { describe, expect, it } from "vitest";
import type { Vec2 } from "../vec.js";
import { clamp, snapToCurve, snapToGrid, snapToLine } from "./constraints.js";

describe("snapToGrid", () => {
  it("snaps to a uniform step", () => {
    const snap = snapToGrid(1);
    expect(snap([0.3, 0.7])).toEqual([0, 1]);
    expect(snap([1.4, -1.6])).toEqual([1, -2]);
  });

  it("accepts per-axis steps", () => {
    const snap = snapToGrid([0.5, 1]);
    expect(snap([0.3, 0.7])).toEqual([0.5, 1]);
    expect(snap([0.24, -1.4])).toEqual([0, -1]);
  });

  it("is idempotent (snapping a snapped point changes nothing)", () => {
    const snap = snapToGrid(0.25);
    const a = snap([0.37, -0.83]);
    const b = snap(a);
    expect(b).toEqual(a);
  });

  it("throws on a zero step (would divide by zero)", () => {
    expect(() => snapToGrid(0)).toThrow();
    expect(() => snapToGrid([1, 0])).toThrow();
  });
});

describe("snapToLine", () => {
  it("projects onto a horizontal line (y = 0)", () => {
    const snap = snapToLine([-1, 0], [1, 0]);
    const [x, y] = snap([0.5, 3]);
    expect(x).toBeCloseTo(0.5, 10);
    expect(y).toBeCloseTo(0, 10);
  });

  it("projects onto a diagonal line (y = x)", () => {
    const snap = snapToLine([0, 0], [1, 1]);
    const [x, y] = snap([2, 0]);
    // Projection of (2, 0) onto line y=x is (1, 1).
    expect(x).toBeCloseTo(1, 10);
    expect(y).toBeCloseTo(1, 10);
  });

  it("is idempotent", () => {
    const snap = snapToLine([0, 0], [2, 1]);
    const p = snap([3, 5]);
    expect(snap(p)).toEqual(p);
  });

  it("returns the anchor point when the line is degenerate (a === b)", () => {
    const snap = snapToLine([1, 2], [1, 2]);
    expect(snap([5, 9])).toEqual([1, 2]);
  });
});

describe("snapToCurve", () => {
  it("snaps onto the unit circle parametrized by angle", () => {
    const circle = (t: number): Vec2 => [Math.cos(t), Math.sin(t)];
    const snap = snapToCurve(circle, [0, 2 * Math.PI], 360);
    const [x, y] = snap([2, 0]);
    expect(x).toBeCloseTo(1, 2);
    expect(y).toBeCloseTo(0, 2);
  });

  it("picks the closest sample, not a random one", () => {
    // Straight line from (0, 0) to (10, 0), sampled. Query point (3.2, 5)
    // should snap to sample closest to x=3.2.
    const line = (t: number): Vec2 => [t, 0];
    const snap = snapToCurve(line, [0, 10], 100);
    const [x, y] = snap([3.2, 5]);
    expect(x).toBeCloseTo(3.2, 1);
    expect(y).toBe(0);
  });

  it("is stable with a single-sample range", () => {
    const curve = (_t: number): Vec2 => [0, 0];
    const snap = snapToCurve(curve, [0, 1], 2);
    expect(snap([5, 5])).toEqual([0, 0]);
  });
});

describe("clamp (bounds)", () => {
  it("clamps both axes when both bounds given", () => {
    const c = clamp({ x: [-1, 1], y: [-2, 2] });
    expect(c([5, -5])).toEqual([1, -2]);
    expect(c([0.5, 0.5])).toEqual([0.5, 0.5]);
    expect(c([-10, 10])).toEqual([-1, 2]);
  });

  it("only clamps the axes supplied", () => {
    const c = clamp({ x: [0, 10] });
    expect(c([-5, 999])).toEqual([0, 999]);
    expect(c([42, -42])).toEqual([10, -42]);
  });

  it("is a no-op when no axes are supplied", () => {
    const c = clamp({});
    expect(c([1, 2])).toEqual([1, 2]);
  });

  it("is idempotent", () => {
    const c = clamp({ x: [-1, 1], y: [-1, 1] });
    const once = c([3, -3]);
    expect(c(once)).toEqual(once);
  });
});
