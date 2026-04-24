import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./plot-of-x.harness.svelte";
import type { ComponentProps } from "svelte";
import type PlotOfX from "./plot-of-x.svelte";

type PlotProps = ComponentProps<typeof PlotOfX>;

const mount = (plot: PlotProps, mafs: Partial<{ width: number; height: number; viewBox: { x: readonly [number, number]; y: readonly [number, number] } }> = {}) => {
  const { container } = render(Harness, { props: { plot, ...mafs } });
  const path = container.querySelector<SVGPathElement>('path[data-mafs-plot="x"]');
  if (!path) throw new Error('<path data-mafs-plot="x"> not rendered');
  return path;
};

const countMoveTo = (d: string) => (d.match(/M /g) ?? []).length;
const countLineTo = (d: string) => (d.match(/L /g) ?? []).length;

describe("<Plot.OfX>", () => {
  describe("path geometry", () => {
    it("linear f produces a path starting at [xMin, f(xMin)] and ending at [xMax, f(xMax)]", () => {
      const path = mount({ y: (x: number) => x });
      const d = path.getAttribute("d")!;
      expect(d.startsWith("M -5 -5")).toBe(true);
      // Last command must land at x=5, y=5 (line y=x at domain edges).
      expect(d.endsWith("L 5 5")).toBe(true);
    });

    it("linear f has exactly one M (no stroke breaks)", () => {
      const path = mount({ y: (x: number) => 2 * x });
      expect(countMoveTo(path.getAttribute("d")!)).toBe(1);
    });

    it("sinusoid produces strictly more segments than a line over the same range", () => {
      const line = mount({ y: (x: number) => x });
      const sinu = mount({ y: (x: number) => Math.sin(x) });
      expect(countLineTo(sinu.getAttribute("d")!)).toBeGreaterThan(
        countLineTo(line.getAttribute("d")!),
      );
    });

    it("honors a custom domain narrower than the viewport", () => {
      const path = mount({ y: (x: number) => x, domain: [0, 1] as const });
      const d = path.getAttribute("d")!;
      expect(d.startsWith("M 0 0")).toBe(true);
      expect(d.endsWith("L 1 1")).toBe(true);
    });

    it("breaks the stroke into multiple sub-paths when f emits non-finite values", () => {
      // f(x) = 1/x near 0: sampler emits NaN-markers around the singularity.
      const path = mount({ y: (x: number) => 1 / x }, { viewBox: { x: [-2, 2], y: [-10, 10] } });
      expect(countMoveTo(path.getAttribute("d")!)).toBeGreaterThanOrEqual(2);
    });
  });

  describe("styling", () => {
    it("fills the stroke with `color` and leaves fill=none", () => {
      const path = mount({ y: () => 0, color: "#ff0000" });
      expect(path.getAttribute("stroke")).toBe("#ff0000");
      expect(path.getAttribute("fill")).toBe("none");
    });

    it("applies `opacity` as stroke-opacity", () => {
      const path = mount({ y: () => 0, opacity: 0.5 });
      expect(path.getAttribute("stroke-opacity")).toBe("0.5");
    });

    it("applies `weight` as stroke-width", () => {
      const path = mount({ y: () => 0, weight: 4 });
      expect(path.getAttribute("stroke-width")).toBe("4");
    });

    it("dashed style sets stroke-dasharray", () => {
      const path = mount({ y: () => 0, style: "dashed" });
      expect(path.getAttribute("stroke-dasharray")).toBe("6 6");
    });

    it("dotted style sets stroke-dasharray", () => {
      const path = mount({ y: () => 0, style: "dotted" });
      expect(path.getAttribute("stroke-dasharray")).toBe("2 4");
    });

    it("solid style omits stroke-dasharray", () => {
      const path = mount({ y: () => 0, style: "solid" });
      expect(path.getAttribute("stroke-dasharray")).toBeNull();
    });

    it("uses vector-effect=non-scaling-stroke so weight stays pixel-true under viewBox scaling", () => {
      const path = mount({ y: () => 0 });
      expect(path.getAttribute("vector-effect")).toBe("non-scaling-stroke");
    });
  });
});
