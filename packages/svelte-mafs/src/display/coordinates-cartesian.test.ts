import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import type { ComponentProps } from "svelte";
import { inferLabels } from "../math.js";
import Harness from "./coordinates-cartesian.harness.svelte";

type HarnessProps = ComponentProps<typeof Harness>;

const mount = (props: Partial<HarnessProps> = {}) => {
  const { container } = render(Harness, { props: props as HarnessProps });
  const svg = container.querySelector("svg[data-mafs-root]");
  if (!svg) throw new Error("<Mafs> root not rendered");
  return svg as SVGSVGElement;
};

describe("<Coordinates.Cartesian>", () => {
  describe("axes", () => {
    it("renders an x-axis line spanning [xMin, xMax] at y=0 by default", () => {
      const svg = mount({ viewBox: { x: [-5, 5], y: [-5, 5] } });
      const xAxis = svg.querySelector('[data-mafs-axis="x"]');
      expect(xAxis).not.toBeNull();
      expect(xAxis!.getAttribute("x1")).toBe("-5");
      expect(xAxis!.getAttribute("x2")).toBe("5");
      expect(xAxis!.getAttribute("y1")).toBe("0");
      expect(xAxis!.getAttribute("y2")).toBe("0");
    });

    it("renders a y-axis line spanning [yMin, yMax] at x=0 by default", () => {
      const svg = mount({ viewBox: { x: [-5, 5], y: [-2, 10] } });
      const yAxis = svg.querySelector('[data-mafs-axis="y"]');
      expect(yAxis).not.toBeNull();
      expect(yAxis!.getAttribute("x1")).toBe("0");
      expect(yAxis!.getAttribute("x2")).toBe("0");
      expect(yAxis!.getAttribute("y1")).toBe("-2");
      expect(yAxis!.getAttribute("y2")).toBe("10");
    });

    it("omits the x-axis when xAxis=false", () => {
      const svg = mount({ xAxis: false });
      expect(svg.querySelector('[data-mafs-axis="x"]')).toBeNull();
    });

    it("omits the y-axis when yAxis=false", () => {
      const svg = mount({ yAxis: false });
      expect(svg.querySelector('[data-mafs-axis="y"]')).toBeNull();
    });

    it("applies vector-effect=non-scaling-stroke on axis lines", () => {
      const svg = mount();
      const xAxis = svg.querySelector('[data-mafs-axis="x"]');
      expect(xAxis!.getAttribute("vector-effect")).toBe("non-scaling-stroke");
    });
  });

  describe("grid", () => {
    it("emits one <line> per x-tick plus one per y-tick inside [data-mafs-grid]", () => {
      const vb = { x: [-5, 5] as const, y: [-5, 5] as const };
      const svg = mount({ viewBox: vb });
      const grid = svg.querySelector("[data-mafs-grid]");
      expect(grid).not.toBeNull();
      const expected = inferLabels(-5, 5).length + inferLabels(-5, 5).length;
      expect(grid!.querySelectorAll("line").length).toBe(expected);
    });

    it("omits the grid when grid=false", () => {
      const svg = mount({ grid: false });
      expect(svg.querySelector("[data-mafs-grid]")).toBeNull();
    });

    it("asymmetric viewBox uses the asymmetric tick set", () => {
      // Catches a bug where we reuse xTicks for y.
      const vb = { x: [0, 100] as const, y: [-1, 1] as const };
      const svg = mount({ viewBox: vb });
      const grid = svg.querySelector("[data-mafs-grid]")!;
      const xCount = inferLabels(0, 100).length;
      const yCount = inferLabels(-1, 1).length;
      expect(grid.querySelectorAll("line").length).toBe(xCount + yCount);
    });
  });

  describe("labels", () => {
    it("emits one x-label per nonzero x-tick", () => {
      const svg = mount({ viewBox: { x: [-5, 5], y: [-5, 5] } });
      const labels = svg.querySelectorAll('[data-mafs-labels="x"] text');
      const ticks = inferLabels(-5, 5).filter((t) => t !== 0);
      expect(labels.length).toBe(ticks.length);
      // Every tick value should appear as some label's text content.
      const texts = Array.from(labels).map((l) => l.textContent);
      for (const t of ticks) expect(texts).toContain(String(t));
    });

    it("wraps each label in a <g> that counter-flips y with scale(1,-1)", () => {
      // Counter-flip is required because the Mafs root applies <g scale(1,-1)>.
      // Without the inner flip, text renders mirrored.
      const svg = mount();
      const firstLabelGroup = svg.querySelector('[data-mafs-labels="x"] g');
      expect(firstLabelGroup).not.toBeNull();
      const transform = firstLabelGroup!.getAttribute("transform") ?? "";
      // Accept scale(1, -1) or any scale(sx, -sy) that counter-flips y.
      expect(transform).toMatch(/scale\([^)]*-/);
    });

    it("omits labels when the axis is disabled", () => {
      const svg = mount({ xAxis: false });
      expect(svg.querySelector('[data-mafs-labels="x"]')).toBeNull();
    });
  });

  describe("defaults", () => {
    it("renders xAxis + yAxis + grid by default", () => {
      const svg = mount();
      expect(svg.querySelector('[data-mafs-axis="x"]')).not.toBeNull();
      expect(svg.querySelector('[data-mafs-axis="y"]')).not.toBeNull();
      expect(svg.querySelector("[data-mafs-grid]")).not.toBeNull();
    });
  });
});
