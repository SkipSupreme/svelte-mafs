import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./Polygon.harness.svelte";

const getPolygon = (
  props: Parameters<typeof render<typeof Harness>>[1]["props"],
) => {
  const { container } = render(Harness, { props });
  const p = container.querySelector("polygon");
  if (!p) throw new Error("no <polygon> rendered");
  return p;
};

describe("<Polygon>", () => {
  it("renders a <polygon> with user-space points in the 'points' attr", () => {
    const p = getPolygon({
      points: [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ],
    });
    const attr = p.getAttribute("points");
    // Accept either comma-separated or space-separated formatting.
    const coords = attr!
      .trim()
      .split(/[\s,]+/)
      .map(Number);
    expect(coords).toEqual([0, 0, 1, 0, 1, 1, 0, 1]);
  });

  it("stroke-only by default (transparent fill)", () => {
    const p = getPolygon({
      points: [
        [0, 0],
        [1, 0],
        [0.5, 1],
      ],
    });
    expect(p.getAttribute("fill")).toBe("none");
    expect(p.getAttribute("stroke")).toBeTruthy();
  });

  it("accepts fillColor and default translucent fill-opacity", () => {
    const p = getPolygon({
      points: [
        [0, 0],
        [1, 0],
        [0.5, 1],
      ],
      fillColor: "green",
    });
    expect(p.getAttribute("fill")).toBe("green");
    const fo = Number(p.getAttribute("fill-opacity"));
    expect(fo).toBeGreaterThan(0);
    expect(fo).toBeLessThanOrEqual(1);
  });

  it("accepts explicit fillOpacity", () => {
    const p = getPolygon({
      points: [
        [0, 0],
        [1, 0],
        [0.5, 1],
      ],
      fillColor: "green",
      fillOpacity: 0.9,
    });
    expect(p.getAttribute("fill-opacity")).toBe("0.9");
  });

  it("uses non-scaling-stroke", () => {
    const p = getPolygon({
      points: [
        [0, 0],
        [1, 0],
        [0.5, 1],
      ],
    });
    expect(p.getAttribute("vector-effect")).toBe("non-scaling-stroke");
  });

  it("supports negative and fractional coordinates", () => {
    const p = getPolygon({
      points: [
        [-1.5, 0.5],
        [2, -1.25],
      ],
    });
    const coords = p
      .getAttribute("points")!
      .trim()
      .split(/[\s,]+/)
      .map(Number);
    expect(coords).toEqual([-1.5, 0.5, 2, -1.25]);
  });

  it("renders nothing for an empty points array", () => {
    const { container } = render(Harness, { props: { points: [] } });
    expect(container.querySelector("polygon")).toBeNull();
  });
});
