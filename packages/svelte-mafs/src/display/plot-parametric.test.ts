import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./plot-parametric.harness.svelte";
import type { ComponentProps } from "svelte";
import type PlotParametric from "./plot-parametric.svelte";

type PlotProps = ComponentProps<typeof PlotParametric>;

const mount = (plot: PlotProps) => {
  const { container } = render(Harness, { props: { plot } });
  const path = container.querySelector<SVGPathElement>(
    'path[data-mafs-plot="parametric"]',
  );
  if (!path) throw new Error('<path data-mafs-plot="parametric"> not rendered');
  return path;
};

const parsePoints = (d: string): Array<[number, number]> => {
  const tokens = d.split(/\s+/);
  const pts: Array<[number, number]> = [];
  for (let i = 0; i < tokens.length; i += 3) {
    const cmd = tokens[i];
    if (cmd !== "M" && cmd !== "L") continue;
    pts.push([Number(tokens[i + 1]), Number(tokens[i + 2])]);
  }
  return pts;
};

describe("<Plot.Parametric>", () => {
  it("renders an empty-ish path when domain is a single point", () => {
    const path = mount({ xy: (t: number) => [t, t], t: [1, 1] });
    expect(path.getAttribute("d")).toBe("M 1 1");
  });

  it("linear parametric (t, 2t) goes from endpoint to endpoint", () => {
    const path = mount({ xy: (t: number) => [t, 2 * t], t: [0, 2] });
    const d = path.getAttribute("d")!;
    expect(d.startsWith("M 0 0")).toBe(true);
    expect(d.endsWith("L 2 4")).toBe(true);
  });

  it("unit circle sampled over [0, 2π] stays on the unit circle", () => {
    const path = mount({
      xy: (t: number) => [Math.cos(t), Math.sin(t)],
      t: [0, 2 * Math.PI],
    });
    const pts = parsePoints(path.getAttribute("d")!);
    expect(pts.length).toBeGreaterThan(32);
    for (const [x, y] of pts) {
      expect(Math.hypot(x, y)).toBeCloseTo(1, 2);
    }
  });

  it("spiral path: xy=(t cos t, t sin t) endpoints match f(tMin)/f(tMax)", () => {
    const xy = (t: number) =>
      [t * Math.cos(t), t * Math.sin(t)] as const;
    const path = mount({ xy, t: [0, 4] });
    const pts = parsePoints(path.getAttribute("d")!);
    const [xStart, yStart] = xy(0);
    const [xEnd, yEnd] = xy(4);
    expect(pts[0]?.[0]).toBeCloseTo(xStart, 5);
    expect(pts[0]?.[1]).toBeCloseTo(yStart, 5);
    expect(pts.at(-1)?.[0]).toBeCloseTo(xEnd, 5);
    expect(pts.at(-1)?.[1]).toBeCloseTo(yEnd, 5);
  });

  it("non-finite output breaks the stroke", () => {
    const path = mount({
      xy: (t: number) => [t, 1 / t],
      t: [-1, 1],
    });
    const d = path.getAttribute("d")!;
    const moves = (d.match(/M /g) ?? []).length;
    expect(moves).toBeGreaterThanOrEqual(2);
  });

  it("styling props propagate", () => {
    const path = mount({
      xy: (t: number) => [t, t],
      t: [0, 1],
      color: "#112233",
      weight: 5,
      opacity: 0.7,
      style: "dotted",
    });
    expect(path.getAttribute("stroke")).toBe("#112233");
    expect(path.getAttribute("stroke-width")).toBe("5");
    expect(path.getAttribute("stroke-opacity")).toBe("0.7");
    expect(path.getAttribute("stroke-dasharray")).toBe("2 4");
  });
});
