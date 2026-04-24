import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./plot-of-y.harness.svelte";
import type { ComponentProps } from "svelte";
import type PlotOfY from "./plot-of-y.svelte";

type PlotProps = ComponentProps<typeof PlotOfY>;

const mount = (plot: PlotProps, mafs: Partial<{ width: number; height: number; viewBox: { x: readonly [number, number]; y: readonly [number, number] } }> = {}) => {
  const { container } = render(Harness, { props: { plot, ...mafs } });
  const path = container.querySelector<SVGPathElement>('path[data-mafs-plot="y"]');
  if (!path) throw new Error('<path data-mafs-plot="y"> not rendered');
  return path;
};

const countMoveTo = (d: string) => (d.match(/M /g) ?? []).length;
const countLineTo = (d: string) => (d.match(/L /g) ?? []).length;

describe("<Plot.OfY>", () => {
  describe("path geometry", () => {
    it("swaps axes: for x(y)=y the path goes [yMin,yMin]→[yMax,yMax] as [x,y] pairs", () => {
      // f(y) = y is the line y=x but parameterized on y. Start/end x should equal y.
      const path = mount({ x: (y) => y });
      const d = path.getAttribute("d")!;
      expect(d.startsWith("M -5 -5")).toBe(true);
      expect(d.endsWith("L 5 5")).toBe(true);
    });

    it("x(y)=2 is a vertical line — every point has x=2, y walks [yMin..yMax]", () => {
      const path = mount({ x: () => 2 });
      const d = path.getAttribute("d")!;
      expect(d.startsWith("M 2 -5")).toBe(true);
      expect(d.endsWith("L 2 5")).toBe(true);
    });

    it("sinusoid over y produces more segments than a constant", () => {
      const flat = mount({ x: () => 0 });
      const sinu = mount({ x: (y) => Math.sin(y) });
      expect(countLineTo(sinu.getAttribute("d")!)).toBeGreaterThan(
        countLineTo(flat.getAttribute("d")!),
      );
    });

    it("honors a custom y-domain narrower than the viewport", () => {
      const path = mount({ x: (y) => y, domain: [0, 2] as const });
      const d = path.getAttribute("d")!;
      expect(d.startsWith("M 0 0")).toBe(true);
      expect(d.endsWith("L 2 2")).toBe(true);
    });

    it("breaks the stroke at non-finite outputs", () => {
      const path = mount({ x: (y) => 1 / y }, { viewBox: { x: [-10, 10], y: [-2, 2] } });
      expect(countMoveTo(path.getAttribute("d")!)).toBeGreaterThanOrEqual(2);
    });
  });

  describe("styling", () => {
    it("color + opacity + weight + dashed → corresponding SVG stroke attrs", () => {
      const path = mount({
        x: () => 0,
        color: "#abcdef",
        opacity: 0.3,
        weight: 3,
        style: "dashed",
      });
      expect(path.getAttribute("stroke")).toBe("#abcdef");
      expect(path.getAttribute("stroke-opacity")).toBe("0.3");
      expect(path.getAttribute("stroke-width")).toBe("3");
      expect(path.getAttribute("stroke-dasharray")).toBe("6 6");
      expect(path.getAttribute("fill")).toBe("none");
    });
  });
});
