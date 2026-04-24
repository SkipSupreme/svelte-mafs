import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import Harness from "./Vector.harness.svelte";

const mount = (props: Parameters<typeof render<typeof Harness>>[1]["props"]) => {
  const { container } = render(Harness, { props });
  const line = container.querySelector("line");
  const marker = container.querySelector("marker");
  const defs = container.querySelector("defs");
  return { container, line, marker, defs };
};

describe("<Vector>", () => {
  it("renders a <line> from tail to tip in user space", () => {
    const { line } = mount({ tail: [0, 0], tip: [3, 4] });
    expect(line).not.toBeNull();
    expect(line!.getAttribute("x1")).toBe("0");
    expect(line!.getAttribute("y1")).toBe("0");
    expect(line!.getAttribute("x2")).toBe("3");
    expect(line!.getAttribute("y2")).toBe("4");
  });

  it("emits a <defs><marker> with an arrowhead path", () => {
    const { marker, defs } = mount({ tail: [0, 0], tip: [3, 4] });
    expect(defs).not.toBeNull();
    expect(marker).not.toBeNull();
    const arrowShape =
      marker!.querySelector("path") ||
      marker!.querySelector("polygon");
    expect(arrowShape).not.toBeNull();
  });

  it("line's marker-end attribute references the emitted marker id", () => {
    const { line, marker } = mount({ tail: [0, 0], tip: [1, 1] });
    const id = marker!.getAttribute("id");
    expect(id).toBeTruthy();
    expect(line!.getAttribute("marker-end")).toBe(`url(#${id})`);
  });

  it("generates a unique marker id per instance to avoid collisions", () => {
    const { marker: m1 } = mount({ tail: [0, 0], tip: [1, 1] });
    const { marker: m2 } = mount({ tail: [0, 0], tip: [1, 1] });
    expect(m1!.getAttribute("id")).not.toBe(m2!.getAttribute("id"));
  });

  it("propagates color to both stroke and arrowhead fill", () => {
    const { line, marker } = mount({
      tail: [0, 0],
      tip: [1, 1],
      color: "magenta",
    });
    expect(line!.getAttribute("stroke")).toBe("magenta");
    const arrowShape =
      marker!.querySelector("path") ||
      marker!.querySelector("polygon");
    expect(arrowShape!.getAttribute("fill")).toBe("magenta");
  });

  it("line uses non-scaling-stroke", () => {
    const { line } = mount({ tail: [0, 0], tip: [1, 1] });
    expect(line!.getAttribute("vector-effect")).toBe("non-scaling-stroke");
  });

  it("accepts custom weight and opacity", () => {
    const { line } = mount({
      tail: [0, 0],
      tip: [1, 1],
      weight: 3,
      opacity: 0.4,
    });
    expect(line!.getAttribute("stroke-width")).toBe("3");
    expect(line!.getAttribute("stroke-opacity")).toBe("0.4");
  });

  it("marker uses orient='auto' so it follows the line direction", () => {
    const { marker } = mount({ tail: [0, 0], tip: [1, 1] });
    expect(marker!.getAttribute("orient")).toBe("auto");
  });

  it("renders nothing for a zero-length vector (tail == tip)", () => {
    const { line } = mount({ tail: [1, 1], tip: [1, 1] });
    expect(line).toBeNull();
  });
});
