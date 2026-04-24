import { describe, expect, it } from "vitest";
import type { Sample } from "../sampling.js";
import {
  buildPath,
  dashArrayFor,
  defaultTolerance,
  sampleParametric,
} from "./_plot-utils.js";

describe("buildPath", () => {
  it("empty sample list → empty string", () => {
    expect(buildPath([])).toBe("");
  });

  it("single sample → one M command, no L", () => {
    expect(buildPath([[1, 2]])).toBe("M 1 2");
  });

  it("three samples → M + 2 L", () => {
    const s: Sample[] = [
      [0, 0],
      [1, 1],
      [2, 4],
    ];
    expect(buildPath(s)).toBe("M 0 0 L 1 1 L 2 4");
  });

  it("NaN y breaks the stroke into sub-paths with a fresh M", () => {
    const s: Sample[] = [
      [0, 0],
      [1, 1],
      [1.5, Number.NaN],
      [2, 4],
      [3, 9],
    ];
    expect(buildPath(s)).toBe("M 0 0 L 1 1 M 2 4 L 3 9");
  });

  it("leading NaN is skipped; first finite point is the M", () => {
    const s: Sample[] = [
      [0, Number.NaN],
      [1, 1],
      [2, 4],
    ];
    expect(buildPath(s)).toBe("M 1 1 L 2 4");
  });

  it("infinite y is treated identically to NaN (stroke break)", () => {
    const s: Sample[] = [
      [0, 0],
      [1, Number.POSITIVE_INFINITY],
      [2, 2],
    ];
    expect(buildPath(s)).toBe("M 0 0 M 2 2");
  });
});

describe("defaultTolerance", () => {
  it("scales with y-span", () => {
    expect(defaultTolerance(10)).toBe(10 / 500);
    expect(defaultTolerance(1000)).toBe(2);
  });
});

describe("dashArrayFor", () => {
  it('solid → null', () => {
    expect(dashArrayFor("solid")).toBeNull();
  });
  it("dashed → '6 6'", () => {
    expect(dashArrayFor("dashed")).toBe("6 6");
  });
  it("dotted → '2 4'", () => {
    expect(dashArrayFor("dotted")).toBe("2 4");
  });
});

describe("sampleParametric", () => {
  it("tMin == tMax → single point", () => {
    const out = sampleParametric((t) => [t, t * t], {
      domain: [2, 2],
      tolerance: 0.01,
      maxDepth: 10,
    });
    expect(out).toEqual([[2, 4]]);
  });

  it("straight-line parametric requires only a handful of samples", () => {
    const out = sampleParametric((t) => [t, 2 * t], {
      domain: [0, 1],
      tolerance: 0.01,
      maxDepth: 10,
      minDepth: 0,
    });
    // minDepth=0 + linear output → flatness check fires immediately → 3 samples.
    expect(out.length).toBe(3);
    expect(out[0]).toEqual([0, 0]);
    expect(out.at(-1)).toEqual([1, 2]);
  });

  it("circle r=1 produces many samples and every sample is ~unit distance from origin", () => {
    const out = sampleParametric(
      (t) => [Math.cos(t), Math.sin(t)],
      { domain: [0, 2 * Math.PI], tolerance: 0.01, maxDepth: 14 },
    );
    expect(out.length).toBeGreaterThan(32);
    for (const [x, y] of out) {
      expect(Math.hypot(x, y)).toBeCloseTo(1, 2);
    }
  });

  it("reversed domain is normalized (tMin > tMax)", () => {
    const out = sampleParametric((t) => [t, t], {
      domain: [5, -5],
      tolerance: 0.01,
      maxDepth: 10,
    });
    expect(out[0]).toEqual([-5, -5]);
    expect(out.at(-1)).toEqual([5, 5]);
  });

  it("non-finite output inserts a [NaN, NaN] gap marker", () => {
    const out = sampleParametric(
      (t) => [t, 1 / t],
      { domain: [-1, 1], tolerance: 0.01, maxDepth: 12 },
    );
    const hasGap = out.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y));
    expect(hasGap).toBe(true);
  });

  it("respects maxDepth on pathological oscillation", () => {
    // sin(1/t) oscillates infinitely near 0; maxDepth must still terminate.
    const out = sampleParametric(
      (t) => [t, Math.sin(1 / t)],
      { domain: [1e-9, 1], tolerance: 0.001, maxDepth: 6 },
    );
    expect(out.length).toBeLessThan(1000);
  });
});
