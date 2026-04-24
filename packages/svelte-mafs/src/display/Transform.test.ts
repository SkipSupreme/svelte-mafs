import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import {
  compose,
  identity,
  rotate,
  scale,
  toMatrixString,
  translate,
} from "./matrix.js";
import TransformProbe from "./Transform.probe.svelte";

describe("matrix helpers", () => {
  it("identity produces the no-op matrix", () => {
    expect(identity).toEqual([1, 0, 0, 1, 0, 0]);
    expect(toMatrixString(identity)).toBe("matrix(1, 0, 0, 1, 0, 0)");
  });

  it("translate(3, 4) offsets without scaling or rotating", () => {
    expect(translate(3, 4)).toEqual([1, 0, 0, 1, 3, 4]);
  });

  it("scale(2, 3) stretches axes independently", () => {
    expect(scale(2, 3)).toEqual([2, 0, 0, 3, 0, 0]);
  });

  it("scale(2) is uniform when only one arg is given", () => {
    expect(scale(2)).toEqual([2, 0, 0, 2, 0, 0]);
  });

  it("rotate(π/2) produces matrix(0, 1, -1, 0, 0, 0)", () => {
    // Plan-mandated assertion: the classic 90° rotation. Near-zero cos
    // values must collapse to 0 in the string form.
    expect(toMatrixString(rotate(Math.PI / 2))).toBe(
      "matrix(0, 1, -1, 0, 0, 0)",
    );
  });

  it("rotate(π) produces matrix(-1, 0, 0, -1, 0, 0) (up to epsilon)", () => {
    expect(toMatrixString(rotate(Math.PI))).toBe("matrix(-1, 0, 0, -1, 0, 0)");
  });

  it("compose() with zero args returns identity", () => {
    expect(compose()).toEqual(identity);
  });

  it("compose(A) with one arg returns A", () => {
    const A = rotate(Math.PI / 4);
    expect(compose(A)).toEqual(A);
  });

  it("compose(translate, rotate) applies translate first then rotate", () => {
    // Apply to the origin: translate moves (0,0)→(3,0), then rotate(π/2)
    // sends that to (0, 3).
    const M = compose(translate(3, 0), rotate(Math.PI / 2));
    const applied = applyTo(M, [0, 0]);
    expect(applied[0]).toBeCloseTo(0, 10);
    expect(applied[1]).toBeCloseTo(3, 10);
  });

  it("compose is associative on sample data", () => {
    const A = translate(1, 2);
    const B = rotate(Math.PI / 3);
    const C = scale(2, 3);
    const left = compose(compose(A, B), C);
    const right = compose(A, compose(B, C));
    // noUncheckedIndexedAccess types tuple indices as `T | undefined`, so
    // use `.forEach` over the first matrix to get non-nullable entries.
    left.forEach((value, i) => {
      expect(value).toBeCloseTo(right[i]!, 10);
    });
  });
});

describe("<Transform>", () => {
  it("renders a <g> with matrix() transform string from the matrix prop", () => {
    const { container } = render(TransformProbe, {
      props: { matrix: rotate(Math.PI / 2) },
    });
    const inner = container.querySelector("g > g");
    expect(inner).toBeTruthy();
    expect(inner!.getAttribute("transform")).toBe("matrix(0, 1, -1, 0, 0, 0)");
  });

  it("defaults to identity when no matrix is supplied", () => {
    const { container } = render(TransformProbe, { props: {} });
    const inner = container.querySelector("g > g");
    expect(inner!.getAttribute("transform")).toBe("matrix(1, 0, 0, 1, 0, 0)");
  });
});

/**
 * Apply a 2D affine matrix to a point (test-only helper so tests read
 * naturally without poking at the internal multiply).
 */
const applyTo = (
  [a, b, c, d, e, f]: readonly [number, number, number, number, number, number],
  [x, y]: [number, number],
): [number, number] => [a * x + c * y + e, b * x + d * y + f];
