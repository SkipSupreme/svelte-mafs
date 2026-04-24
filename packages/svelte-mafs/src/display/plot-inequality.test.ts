import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./plot-inequality.harness.svelte";
import type { ComponentProps } from "svelte";
import type PlotInequality from "./plot-inequality.svelte";

type PlotProps = ComponentProps<typeof PlotInequality>;

const mount = (plot: PlotProps) => {
  const { container } = render(Harness, { props: { plot } });
  const path = container.querySelector<SVGPathElement>(
    'path[data-mafs-plot="inequality"]',
  );
  if (!path) throw new Error('<path data-mafs-plot="inequality"> not rendered');
  return path;
};

describe("<Plot.Inequality>", () => {
  it("defaults to direction=above and closes the region at yMax", () => {
    const path = mount({ y: () => 0 });
    const d = path.getAttribute("d")!;
    expect(path.getAttribute("data-mafs-direction")).toBe("above");
    // Path must end with a close to yMax on both corners.
    expect(d).toMatch(/L 5 5\s+L -5 5\s+Z$/);
  });

  it("direction=below closes at yMin", () => {
    const path = mount({ y: () => 0, direction: "below" });
    const d = path.getAttribute("d")!;
    expect(path.getAttribute("data-mafs-direction")).toBe("below");
    expect(d).toMatch(/L 5 -5\s+L -5 -5\s+Z$/);
  });

  it("starts the path at [xMin, f(xMin)] and ends the curve portion at [xMax, f(xMax)]", () => {
    // f(x) = x/2: at x=-5 → y=-2.5; at x=5 → y=2.5.
    const path = mount({ y: (x: number) => x / 2 });
    const d = path.getAttribute("d")!;
    expect(d.startsWith("M -5 -2.5")).toBe(true);
    // Last curve point before closure: "L 5 2.5 L 5 5 L -5 5 Z"
    expect(d).toMatch(/L 5 2\.5\s+L 5 5\s+L -5 5\s+Z$/);
  });

  it("fill uses color + fill-opacity; stroke is none", () => {
    const path = mount({ y: () => 0, color: "#ff00aa", opacity: 0.4 });
    expect(path.getAttribute("fill")).toBe("#ff00aa");
    expect(path.getAttribute("fill-opacity")).toBe("0.4");
    expect(path.getAttribute("stroke")).toBe("none");
  });

  it("honors a narrower domain", () => {
    const path = mount({ y: () => 1, domain: [0, 2] as const });
    const d = path.getAttribute("d")!;
    expect(d.startsWith("M 0 1")).toBe(true);
    // Closure back to x=0 at yMax.
    expect(d).toMatch(/L 2 5\s+L 0 5\s+Z$/);
  });

  it("closes the polygon with a Z command", () => {
    const path = mount({ y: () => 0 });
    expect(path.getAttribute("d")).toMatch(/\sZ$/);
  });
});
