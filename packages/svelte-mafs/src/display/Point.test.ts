import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./Point.harness.svelte";

const getCircle = (props: Parameters<typeof render<typeof Harness>>[1]["props"]) => {
  const { container } = render(Harness, { props });
  const circle = container.querySelector("circle");
  if (!circle) throw new Error("no <circle> rendered");
  return circle;
};

describe("<Point>", () => {
  it("writes user-space x,y directly to cx,cy (SVG viewBox handles the scale)", () => {
    const circle = getCircle({ x: 1, y: 2 });
    expect(circle.getAttribute("cx")).toBe("1");
    expect(circle.getAttribute("cy")).toBe("2");
  });

  it("handles negative coordinates", () => {
    const circle = getCircle({ x: -3, y: -4 });
    expect(circle.getAttribute("cx")).toBe("-3");
    expect(circle.getAttribute("cy")).toBe("-4");
  });

  it("handles fractional coordinates", () => {
    const circle = getCircle({ x: 0.5, y: -1.25 });
    expect(circle.getAttribute("cx")).toBe("0.5");
    expect(circle.getAttribute("cy")).toBe("-1.25");
  });

  it("default fill falls back to a concrete color until theme lands", () => {
    const circle = getCircle({ x: 0, y: 0 });
    // Stream 6 swaps this to var(--mafs-fg). Hardcoded fallback until then.
    expect(circle.getAttribute("fill")).toBeTruthy();
    expect(circle.getAttribute("fill")).not.toBe("");
  });

  it("accepts a custom color prop", () => {
    const circle = getCircle({ x: 0, y: 0, color: "red" });
    expect(circle.getAttribute("fill")).toBe("red");
  });

  it("default opacity is 1", () => {
    const circle = getCircle({ x: 0, y: 0 });
    // fill-opacity omitted or 1 — either is valid.
    const op = circle.getAttribute("fill-opacity");
    if (op !== null) expect(op).toBe("1");
  });

  it("accepts a custom opacity prop", () => {
    const circle = getCircle({ x: 0, y: 0, opacity: 0.5 });
    expect(circle.getAttribute("fill-opacity")).toBe("0.5");
  });

  it("computes a scale-aware radius so the dot looks consistent across viewBox sizes", () => {
    // On a 400x300 canvas with a 10-unit xSpan, one user unit = 40 px.
    // A target ~6px radius should yield ~0.15 user units.
    const circle = getCircle({ x: 0, y: 0 });
    const r = Number(circle.getAttribute("r"));
    expect(r).toBeGreaterThan(0);
    // Sanity: for these dims the radius should be small (well under 1 user unit).
    expect(r).toBeLessThan(1);
  });

  it("radius scales down when the user viewBox is zoomed out (more user units per pixel)", () => {
    const small = Number(
      getCircle({
        x: 0,
        y: 0,
        viewBox: { x: [-5, 5], y: [-5, 5] }, // 10 units across
      }).getAttribute("r"),
    );
    const large = Number(
      getCircle({
        x: 0,
        y: 0,
        viewBox: { x: [-20, 20], y: [-20, 20] }, // 40 units across
      }).getAttribute("r"),
    );
    // 4x zoom out → radius in user units is 4x bigger (same visual pixel size).
    expect(large).toBeCloseTo(small * 4, 3);
  });
});
