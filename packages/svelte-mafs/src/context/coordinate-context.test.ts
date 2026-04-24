import { describe, expect, it } from "vitest";
import {
  makeCoordContext,
  normalizeViewBox,
  type NormalizedViewBox,
  type UserViewBox,
} from "./coordinate-context.js";

const vb = (xMin: number, xMax: number, yMin: number, yMax: number): NormalizedViewBox => ({
  xMin,
  xMax,
  yMin,
  yMax,
});

describe("normalizeViewBox", () => {
  it("converts prop shape {x:[a,b], y:[c,d]} to {xMin,xMax,yMin,yMax}", () => {
    const input: UserViewBox = { x: [-5, 5], y: [-3, 7] };
    expect(normalizeViewBox(input)).toEqual({ xMin: -5, xMax: 5, yMin: -3, yMax: 7 });
  });

  it("handles reversed x range by swapping so xMin < xMax", () => {
    const input: UserViewBox = { x: [5, -5], y: [0, 10] };
    const out = normalizeViewBox(input);
    expect(out.xMin).toBe(-5);
    expect(out.xMax).toBe(5);
  });

  it("handles reversed y range by swapping so yMin < yMax", () => {
    const input: UserViewBox = { x: [0, 10], y: [10, -10] };
    const out = normalizeViewBox(input);
    expect(out.yMin).toBe(-10);
    expect(out.yMax).toBe(10);
  });
});

describe("makeCoordContext", () => {
  describe("userToPx — square viewBox", () => {
    // x:[-5,5], y:[-5,5], 400x300 SVG → origin maps to center (200,150)
    const ctx = makeCoordContext(vb(-5, 5, -5, 5), 400, 300);

    it("origin → center pixel", () => {
      expect(ctx.userToPx([0, 0])).toEqual([200, 150]);
    });

    it("bottom-left user corner → bottom-left px corner", () => {
      // user (-5, -5) → left edge x, bottom y → (0, 300)
      expect(ctx.userToPx([-5, -5])).toEqual([0, 300]);
    });

    it("top-right user corner → top-right px corner", () => {
      // user (5, 5) → right edge, top y → (400, 0)
      expect(ctx.userToPx([5, 5])).toEqual([400, 0]);
    });

    it("y-axis is flipped: increasing user y = decreasing px y", () => {
      const [, y0] = ctx.userToPx([0, 0]);
      const [, y1] = ctx.userToPx([0, 1]);
      expect(y1).toBeLessThan(y0);
    });
  });

  describe("userToPx — wide viewBox (landscape)", () => {
    // x:[-10,10] (20 wide), y:[-1,1] (2 tall), 800x200 SVG
    const ctx = makeCoordContext(vb(-10, 10, -1, 1), 800, 200);

    it("origin → center pixel", () => {
      expect(ctx.userToPx([0, 0])).toEqual([400, 100]);
    });

    it("x=5 → three-quarter width", () => {
      const [px] = ctx.userToPx([5, 0]);
      expect(px).toBe(600);
    });

    it("y=0.5 → upper quarter px", () => {
      const [, py] = ctx.userToPx([0, 0.5]);
      expect(py).toBe(50);
    });
  });

  describe("userToPx — tall viewBox (portrait)", () => {
    // x:[-1,1], y:[-10,10], 200x800
    const ctx = makeCoordContext(vb(-1, 1, -10, 10), 200, 800);

    it("origin → center pixel", () => {
      expect(ctx.userToPx([0, 0])).toEqual([100, 400]);
    });

    it("y=5 → upper quarter px", () => {
      const [, py] = ctx.userToPx([0, 5]);
      expect(py).toBe(200);
    });
  });

  describe("userToPx — negative-only range", () => {
    // x:[-20,-10], y:[5,15]
    const ctx = makeCoordContext(vb(-20, -10, 5, 15), 400, 300);

    it("xMin maps to px 0", () => {
      const [px] = ctx.userToPx([-20, 10]);
      expect(px).toBe(0);
    });

    it("xMax maps to px width", () => {
      const [px] = ctx.userToPx([-10, 10]);
      expect(px).toBe(400);
    });

    it("yMin (user) maps to bottom px (= height)", () => {
      const [, py] = ctx.userToPx([-15, 5]);
      expect(py).toBe(300);
    });

    it("yMax (user) maps to top px (= 0)", () => {
      const [, py] = ctx.userToPx([-15, 15]);
      expect(py).toBe(0);
    });

    it("midpoint of user range maps to center px", () => {
      expect(ctx.userToPx([-15, 10])).toEqual([200, 150]);
    });
  });

  describe("userToPx — asymmetric y range crossing zero", () => {
    // y:[-2,10] exposes the "-yMax" viewBox trick — if there were a sign bug,
    // a symmetric test wouldn't catch it.
    const ctx = makeCoordContext(vb(-5, 5, -2, 10), 400, 300);

    it("y=10 (top) → py=0", () => {
      const [, py] = ctx.userToPx([0, 10]);
      expect(py).toBe(0);
    });

    it("y=-2 (bottom) → py=height", () => {
      const [, py] = ctx.userToPx([0, -2]);
      expect(py).toBe(300);
    });

    it("y=0 → not center, but 10/12 of the way down", () => {
      // from top (y=10) to y=0 spans 10 of 12 user-units → py = 10/12 * 300 = 250
      const [, py] = ctx.userToPx([0, 0]);
      expect(py).toBe(250);
    });
  });

  describe("pxToUser — exact inverse of userToPx", () => {
    it("round-trips origin on square viewBox", () => {
      const ctx = makeCoordContext(vb(-5, 5, -5, 5), 400, 300);
      expect(ctx.pxToUser([200, 150])).toEqual([0, 0]);
    });

    it("round-trips corners", () => {
      const ctx = makeCoordContext(vb(-5, 5, -5, 5), 400, 300);
      expect(ctx.pxToUser([0, 0])).toEqual([-5, 5]);
      expect(ctx.pxToUser([400, 300])).toEqual([5, -5]);
    });

    it("round-trips arbitrary points on wide viewBox", () => {
      const ctx = makeCoordContext(vb(-10, 10, -1, 1), 800, 200);
      const samples: Array<[number, number]> = [
        [0, 0],
        [3.7, 0.42],
        [-8.1, -0.99],
        [10, 1],
        [-10, -1],
      ];
      for (const user of samples) {
        const [ux, uy] = ctx.pxToUser(ctx.userToPx(user));
        expect(ux).toBeCloseTo(user[0], 10);
        expect(uy).toBeCloseTo(user[1], 10);
      }
    });

    it("round-trips on negative-only range", () => {
      const ctx = makeCoordContext(vb(-20, -10, 5, 15), 400, 300);
      const samples: Array<[number, number]> = [
        [-15, 10],
        [-20, 5],
        [-10, 15],
        [-12.34, 6.78],
      ];
      for (const user of samples) {
        const [ux, uy] = ctx.pxToUser(ctx.userToPx(user));
        expect(ux).toBeCloseTo(user[0], 10);
        expect(uy).toBeCloseTo(user[1], 10);
      }
    });
  });

  describe("context shape", () => {
    const ctx = makeCoordContext(vb(-5, 5, -2, 10), 400, 300);

    it("exposes the normalized viewBox", () => {
      expect(ctx.viewBox).toEqual({ xMin: -5, xMax: 5, yMin: -2, yMax: 10 });
    });

    it("exposes widthPx and heightPx", () => {
      expect(ctx.widthPx).toBe(400);
      expect(ctx.heightPx).toBe(300);
    });
  });

  describe("degenerate dims", () => {
    it("zero-width viewBox does not NaN userToPx x (returns xMin)", () => {
      const ctx = makeCoordContext(vb(3, 3, 0, 10), 400, 300);
      const [px] = ctx.userToPx([3, 5]);
      expect(Number.isFinite(px)).toBe(true);
    });

    it("zero-height viewBox does not NaN userToPx y", () => {
      const ctx = makeCoordContext(vb(0, 10, 5, 5), 400, 300);
      const [, py] = ctx.userToPx([5, 5]);
      expect(Number.isFinite(py)).toBe(true);
    });
  });
});
