import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import type { CoordContext } from "../context/coordinate-context.js";
import type { MafsProps } from "./mafs-props.js";
import Mafs from "./Mafs.svelte";
import MafsProbe from "./Mafs.probe.svelte";

const mountWithProbe = (props: Omit<MafsProps, "children">): CoordContext => {
  let captured: CoordContext | undefined;
  render(MafsProbe, {
    props: {
      ...props,
      onRead: (ctx: CoordContext) => {
        captured = ctx;
      },
    },
  });
  if (!captured) {
    throw new Error("probe child never ran — context was not published");
  }
  return captured;
};

describe("<Mafs>", () => {
  describe("published coordinate context", () => {
    it("userToPx([0,0]) returns the center pixel for a symmetric viewBox", () => {
      const ctx = mountWithProbe({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [-5, 5] },
      });
      expect(ctx.userToPx([0, 0])).toEqual([200, 150]);
    });

    it("pxToUser([200,150]) round-trips back to [0,0]", () => {
      const ctx = mountWithProbe({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [-5, 5] },
      });
      expect(ctx.pxToUser([200, 150])).toEqual([0, 0]);
    });

    it("exposes widthPx and heightPx matching the numeric props", () => {
      const ctx = mountWithProbe({
        width: 640,
        height: 480,
        viewBox: { x: [-1, 1], y: [-1, 1] },
      });
      expect(ctx.widthPx).toBe(640);
      expect(ctx.heightPx).toBe(480);
    });

    it("exposes normalized viewBox (min/max ordered)", () => {
      const ctx = mountWithProbe({
        width: 400,
        height: 300,
        viewBox: { x: [5, -5], y: [10, -10] },
      });
      expect(ctx.viewBox).toEqual({ xMin: -5, xMax: 5, yMin: -10, yMax: 10 });
    });

    it("maps correctly across an asymmetric y range", () => {
      // y:[-2,10], height 300 → y=10 → py=0, y=-2 → py=300, y=0 → py=250.
      // Catches a -yMax vs yMin sign bug that symmetric tests miss.
      const ctx = mountWithProbe({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [-2, 10] },
      });
      expect(ctx.userToPx([0, 10])[1]).toBe(0);
      expect(ctx.userToPx([0, -2])[1]).toBe(300);
      expect(ctx.userToPx([0, 0])[1]).toBe(250);
    });

    it('falls back to a finite widthPx/heightPx when width/height are "auto"', () => {
      const ctx = mountWithProbe({
        width: "auto",
        height: "auto",
        viewBox: { x: [0, 1], y: [0, 1] },
      });
      expect(Number.isFinite(ctx.widthPx)).toBe(true);
      expect(Number.isFinite(ctx.heightPx)).toBe(true);
    });
  });

  describe("SVG markup", () => {
    const getSvg = (props: Omit<MafsProps, "children">) => {
      const { container } = render(Mafs, { props });
      const svg = container.querySelector("svg[data-mafs-root]");
      if (!svg) throw new Error("root <svg> not rendered");
      return svg as SVGSVGElement;
    };

    it("emits a <svg> root with viewBox in [xMin, -yMax, xSpan, ySpan] form", () => {
      const svg = getSvg({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [-5, 5] },
      });
      expect(svg.getAttribute("viewBox")).toBe("-5 -5 10 10");
    });

    it("uses -yMax (not yMin) for asymmetric y ranges — catches a sign bug", () => {
      const svg = getSvg({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [0, 10] },
      });
      expect(svg.getAttribute("viewBox")).toBe("-5 -10 10 10");
    });

    it("emits width and height attrs on the <svg>", () => {
      const svg = getSvg({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [-5, 5] },
      });
      expect(svg.getAttribute("width")).toBe("400");
      expect(svg.getAttribute("height")).toBe("300");
    });

    it("wraps children in an inner <g transform='scale(1, -1)'>", () => {
      const svg = getSvg({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [-5, 5] },
      });
      const g = svg.querySelector("g");
      expect(g?.getAttribute("transform")).toBe("scale(1, -1)");
    });

    it('maps preserveAspectRatio=true to "xMidYMid meet"', () => {
      const svg = getSvg({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [-5, 5] },
        preserveAspectRatio: true,
      });
      expect(svg.getAttribute("preserveAspectRatio")).toBe("xMidYMid meet");
    });

    it('maps preserveAspectRatio=false to "none"', () => {
      const svg = getSvg({
        width: 400,
        height: 300,
        viewBox: { x: [-5, 5], y: [-5, 5] },
        preserveAspectRatio: false,
      });
      expect(svg.getAttribute("preserveAspectRatio")).toBe("none");
    });
  });
});
