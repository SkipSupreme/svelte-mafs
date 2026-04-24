import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./plot-vector-field.harness.svelte";
import type { ComponentProps } from "svelte";
import type PlotVectorField from "./plot-vector-field.svelte";

type PlotProps = ComponentProps<typeof PlotVectorField>;

const mount = (plot: PlotProps) => {
  const { container } = render(Harness, { props: { plot } });
  const group = container.querySelector<SVGGElement>(
    'g[data-mafs-plot="vector-field"]',
  );
  if (!group) throw new Error('<g data-mafs-plot="vector-field"> not rendered');
  return group;
};

describe("<Plot.VectorField>", () => {
  it("renders one shaft + one head per grid point with a non-zero vector", () => {
    // Constant vector field; 11×11 grid on [-5,5]×[-5,5] at step=1.
    const group = mount({ xy: () => [1, 0], step: 1 });
    const shafts = group.querySelectorAll('[data-mafs-arrow="shaft"]');
    const heads = group.querySelectorAll('[data-mafs-arrow="head"]');
    expect(shafts.length).toBe(heads.length);
    expect(shafts.length).toBe(11 * 11);
  });

  it("step=2 produces a sparser grid", () => {
    const fine = mount({ xy: () => [1, 0], step: 1 });
    const coarse = mount({ xy: () => [1, 0], step: 2 });
    expect(coarse.querySelectorAll('[data-mafs-arrow="shaft"]').length).toBeLessThan(
      fine.querySelectorAll('[data-mafs-arrow="shaft"]').length,
    );
  });

  it("zero vector field renders no arrows", () => {
    const group = mount({ xy: () => [0, 0] });
    expect(group.querySelectorAll('[data-mafs-arrow="shaft"]').length).toBe(0);
  });

  it("non-finite vectors are skipped (no shaft for that grid point)", () => {
    const group = mount({
      xy: (x, y) => (x === 0 && y === 0 ? [Number.NaN, Number.NaN] : [1, 0]),
    });
    // 121 total - 1 skipped at origin.
    expect(group.querySelectorAll('[data-mafs-arrow="shaft"]').length).toBe(120);
  });

  it("auto-normalizes so the longest drawn arrow fits inside step×normalizedLength", () => {
    // Single huge vector at (0, 0); elsewhere unit.
    const group = mount({
      xy: (x, y) => (x === 0 && y === 0 ? [1000, 0] : [1, 0]),
      step: 1,
      normalizedLength: 0.8,
    });
    const shafts = Array.from(
      group.querySelectorAll<SVGLineElement>('[data-mafs-arrow="shaft"]'),
    );
    for (const s of shafts) {
      const x1 = Number(s.getAttribute("x1"));
      const y1 = Number(s.getAttribute("y1"));
      const x2 = Number(s.getAttribute("x2"));
      const y2 = Number(s.getAttribute("y2"));
      // Shaft ends at head-base, which is ~70% of drawn length (30% head).
      // So shaft length ≤ 0.8 * step.
      expect(Math.hypot(x2 - x1, y2 - y1)).toBeLessThanOrEqual(0.8 + 1e-9);
    }
  });

  it("custom color + opacity propagate to stroke/fill attrs on the group", () => {
    const group = mount({
      xy: () => [1, 0],
      color: "#123456",
      opacity: 0.6,
    });
    expect(group.getAttribute("stroke")).toBe("#123456");
    expect(group.getAttribute("fill")).toBe("#123456");
    expect(group.getAttribute("stroke-opacity")).toBe("0.6");
    expect(group.getAttribute("fill-opacity")).toBe("0.6");
  });

  it("weight sets stroke-width on shafts", () => {
    const group = mount({ xy: () => [1, 0], weight: 3 });
    const shaft = group.querySelector('[data-mafs-arrow="shaft"]');
    expect(shaft?.getAttribute("stroke-width")).toBe("3");
  });

  it("restricted xDomain/yDomain shrinks the sampled grid", () => {
    const full = mount({ xy: () => [1, 0] });
    const restricted = mount({
      xy: () => [1, 0],
      xDomain: [0, 2],
      yDomain: [0, 2],
    });
    expect(
      restricted.querySelectorAll('[data-mafs-arrow="shaft"]').length,
    ).toBeLessThan(
      full.querySelectorAll('[data-mafs-arrow="shaft"]').length,
    );
  });

  it("step=0 renders nothing (guard against infinite loops)", () => {
    const group = mount({ xy: () => [1, 0], step: 0 });
    expect(group.querySelectorAll('[data-mafs-arrow="shaft"]').length).toBe(0);
  });
});
