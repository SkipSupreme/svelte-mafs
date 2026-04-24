import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./Ellipse.harness.svelte";

const getEllipse = (
  props: Parameters<typeof render<typeof Harness>>[1]["props"],
) => {
  const { container } = render(Harness, { props });
  const el = container.querySelector("ellipse");
  if (!el) throw new Error("no <ellipse> rendered");
  return el;
};

describe("<Ellipse>", () => {
  it("renders cx/cy/rx/ry in user-space", () => {
    const el = getEllipse({ center: [1, 2], radius: [2, 1] });
    expect(el.getAttribute("cx")).toBe("1");
    expect(el.getAttribute("cy")).toBe("2");
    expect(el.getAttribute("rx")).toBe("2");
    expect(el.getAttribute("ry")).toBe("1");
  });

  it("has an identity rotation when angle is 0 or omitted", () => {
    const el = getEllipse({ center: [0, 0], radius: [2, 1] });
    const t = el.getAttribute("transform");
    if (t !== null) {
      expect(t).toMatch(/rotate\(0 /);
    }
  });

  it("emits rotate(deg cx cy) with angle converted from radians to degrees", () => {
    const el = getEllipse({
      center: [1, 2],
      radius: [2, 1],
      angle: Math.PI / 2,
    });
    const t = el.getAttribute("transform");
    expect(t).toBeTruthy();
    const parts = t!.match(/rotate\(([^\s]+)\s+([^\s]+)\s+([^)]+)\)/);
    expect(parts).not.toBeNull();
    expect(Number(parts![1])).toBeCloseTo(90);
    expect(Number(parts![2])).toBeCloseTo(1);
    expect(Number(parts![3])).toBeCloseTo(2);
  });

  it("stroke-only by default; fill='none'", () => {
    const el = getEllipse({ center: [0, 0], radius: [2, 1] });
    expect(el.getAttribute("fill")).toBe("none");
    expect(el.getAttribute("stroke")).toBeTruthy();
  });

  it("uses non-scaling-stroke", () => {
    const el = getEllipse({ center: [0, 0], radius: [2, 1] });
    expect(el.getAttribute("vector-effect")).toBe("non-scaling-stroke");
  });

  it("accepts fillColor and a default translucent fillOpacity", () => {
    const el = getEllipse({
      center: [0, 0],
      radius: [2, 1],
      fillColor: "tomato",
    });
    expect(el.getAttribute("fill")).toBe("tomato");
    const fo = Number(el.getAttribute("fill-opacity"));
    expect(fo).toBeGreaterThan(0);
    expect(fo).toBeLessThanOrEqual(1);
  });
});
