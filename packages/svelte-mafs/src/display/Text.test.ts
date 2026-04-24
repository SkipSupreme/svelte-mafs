import { render } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import TextProbe from "./Text.probe.svelte";

describe("<Text>", () => {
  it("renders a <foreignObject> at the given user coordinate", () => {
    const { container } = render(TextProbe, {
      props: { x: 1, y: 2, latex: "x^2" },
    });

    const fo = container.querySelector("foreignObject");
    expect(fo).toBeTruthy();
    expect(fo!.getAttribute("x")).toBe("1");
    expect(fo!.getAttribute("y")).toBe("2");
  });

  it("injects KaTeX markup (contains .katex)", () => {
    // Plan-mandated assertion: the mounted component must have KaTeX's
    // rendered root class somewhere inside the foreignObject so consumers
    // can style/target the math markup with `.katex` rules.
    const { container } = render(TextProbe, {
      props: { x: 0, y: 0, latex: "\\frac{1}{2}" },
    });

    const fo = container.querySelector("foreignObject");
    expect(fo).toBeTruthy();
    expect(fo!.querySelector(".katex")).toBeTruthy();
  });

  it("counter-flips the y-axis on the inner HTML so text reads upright", () => {
    // Mafs's root <g> applies scale(1,-1); Text must apply a matching
    // scale(1,-1) so glyphs aren't upside-down. Any CSS transform string
    // that contains "scale(1, -1)" (whitespace-flexible) satisfies this.
    const { container } = render(TextProbe, {
      props: { x: 0, y: 0, latex: "a" },
    });

    const inner = container.querySelector("foreignObject > div");
    expect(inner).toBeTruthy();
    const style = inner!.getAttribute("style") ?? "";
    expect(style).toMatch(/scale\(\s*1\s*,\s*-1\s*\)/);
  });

  it("applies the color prop as CSS color on the inner div", () => {
    const { container } = render(TextProbe, {
      props: { x: 0, y: 0, latex: "x", color: "#ff00ff" },
    });
    const inner = container.querySelector("foreignObject > div") as HTMLElement;
    // jsdom normalizes hex to rgb(); accept either surface form as long as
    // the color magenta is present on the element.
    const style = inner.getAttribute("style") ?? "";
    expect(style).toMatch(/color:\s*(#ff00ff|rgb\(\s*255\s*,\s*0\s*,\s*255\s*\))/);
  });

  it("gracefully degrades on invalid LaTeX (no throw)", () => {
    // throwOnError: false in the component's call to renderToString means
    // malformed input becomes red error text rather than crashing the mount.
    expect(() =>
      render(TextProbe, { props: { x: 0, y: 0, latex: "\\unknownmacro" } }),
    ).not.toThrow();
  });
});
