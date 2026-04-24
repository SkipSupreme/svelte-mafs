import { render } from "@testing-library/svelte";
import { flushSync } from "svelte";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { clamp, snapToGrid, type Constraint } from "./constraints.js";
import MovablePointProbe from "./MovablePoint.probe.svelte";

// Svelte 5 $effect runs in a microtask; flushSync() forces pending effects
// to run so tests can observe the post-update state synchronously.
const after = (el: Element, event: Event) => {
  el.dispatchEvent(event);
  flushSync();
};

type ProbeProps = {
  initialX: number;
  initialY: number;
  constrain?: Constraint;
  step?: number;
  onChange?: (x: number, y: number) => void;
};

beforeAll(() => {
  // Stream 3's drag test establishes the same polyfill — jsdom 25 doesn't
  // expose pointer-capture methods on Element.prototype, but the drag
  // action calls them defensively. Replicate here so the action mounts
  // cleanly inside MovablePoint.
  const proto = Element.prototype as unknown as {
    setPointerCapture?: (id: number) => void;
    releasePointerCapture?: (id: number) => void;
    hasPointerCapture?: (id: number) => boolean;
  };
  if (!proto.setPointerCapture) proto.setPointerCapture = () => {};
  if (!proto.releasePointerCapture) proto.releasePointerCapture = () => {};
  if (!proto.hasPointerCapture) proto.hasPointerCapture = () => false;
});

beforeEach(() => {
  document.body.replaceChildren();
});

const mount = (props: ProbeProps) => {
  const onChange = vi.fn();
  const { container } = render(MovablePointProbe, {
    props: { onChange, ...props },
  });
  const circle = container.querySelector(".mafs-movable-point") as SVGCircleElement;
  if (!circle) throw new Error("MovablePoint circle did not mount");
  return { circle, onChange };
};

/** Construct a pointer-like event matching Stream 3's test helpers. */
const pe = (
  type: string,
  init: { pointerId?: number; clientX?: number; clientY?: number } = {},
): Event => {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, {
    pointerId: init.pointerId ?? 1,
    clientX: init.clientX ?? 0,
    clientY: init.clientY ?? 0,
    button: 0,
    buttons: 1,
  });
  return e;
};

const ke = (key: string, shift = false): KeyboardEvent =>
  new KeyboardEvent("keydown", { key, shiftKey: shift, bubbles: true, cancelable: true });

describe("<MovablePoint>", () => {
  describe("render", () => {
    it("mounts a circle at (x, y) in user coords", () => {
      const { circle } = mount({ initialX: 1, initialY: 2 });
      expect(circle.getAttribute("cx")).toBe("1");
      expect(circle.getAttribute("cy")).toBe("2");
    });

    it("sets the plan-mandated ARIA attributes", () => {
      const { circle } = mount({ initialX: 0, initialY: 0 });
      expect(circle.getAttribute("role")).toBe("slider");
      expect(circle.getAttribute("tabindex")).toBe("0");
      expect(circle.getAttribute("aria-label")).toBe("Movable point");
      expect(circle.getAttribute("aria-valuenow-x")).toBe("0");
      expect(circle.getAttribute("aria-valuenow-y")).toBe("0");
    });

    it("provides aria-valuetext for screen readers (since aria-valuenow-x/y are non-standard)", () => {
      const { circle } = mount({ initialX: 1.5, initialY: -2 });
      expect(circle.getAttribute("aria-valuetext")).toMatch(/x:\s*1\.5.*y:\s*-2/);
    });
  });

  describe("keyboard", () => {
    it("ArrowRight increments x by step", () => {
      const { circle, onChange } = mount({ initialX: 0, initialY: 0 });
      after(circle, ke("ArrowRight"));
      const last = onChange.mock.calls.at(-1)!;
      expect(last[0]).toBeCloseTo(0.1, 10);
      expect(last[1]).toBe(0);
    });

    it("ArrowLeft decrements x by step", () => {
      const { circle, onChange } = mount({ initialX: 0, initialY: 0 });
      after(circle, ke("ArrowLeft"));
      expect(onChange.mock.calls.at(-1)![0]).toBeCloseTo(-0.1, 10);
    });

    it("ArrowUp increments y (math y-up)", () => {
      const { circle, onChange } = mount({ initialX: 0, initialY: 0 });
      after(circle, ke("ArrowUp"));
      expect(onChange.mock.calls.at(-1)![1]).toBeCloseTo(0.1, 10);
    });

    it("ArrowDown decrements y", () => {
      const { circle, onChange } = mount({ initialX: 0, initialY: 0 });
      after(circle, ke("ArrowDown"));
      expect(onChange.mock.calls.at(-1)![1]).toBeCloseTo(-0.1, 10);
    });

    it("Shift+ArrowRight moves by 10× the step", () => {
      const { circle, onChange } = mount({ initialX: 0, initialY: 0 });
      after(circle, ke("ArrowRight", true));
      expect(onChange.mock.calls.at(-1)![0]).toBeCloseTo(1.0, 10);
    });

    it("ignores non-arrow keys", () => {
      const { circle, onChange } = mount({ initialX: 0, initialY: 0 });
      const before = onChange.mock.calls.length;
      after(circle, ke("Enter"));
      after(circle, ke("a"));
      expect(onChange.mock.calls.length).toBe(before);
    });

    it("pipes keyboard updates through constrain", () => {
      const { circle, onChange } = mount({
        initialX: 0,
        initialY: 0,
        constrain: snapToGrid(0.5),
        step: 0.1, // would land at 0.1, but snap pulls it back to 0
      });
      after(circle, ke("ArrowRight"));
      // ArrowRight takes (0, 0) → (0.1, 0); snap to 0.5 grid → (0, 0).
      const last = onChange.mock.calls.at(-1)!;
      expect(last[0]).toBe(0);
      expect(last[1]).toBe(0);
    });

    it("preventDefault is called so the page doesn't scroll", () => {
      const { circle } = mount({ initialX: 0, initialY: 0 });
      const event = ke("ArrowRight");
      circle.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(true);
    });
  });

  describe("drag", () => {
    it("updates x, y on pointerdown → pointermove", () => {
      // viewBox x:[-5,5], y:[-5,5] at 400×400px means pxToUser([200,200]) = [0,0]
      // and pxToUser([240, 160]) = [1, 1].
      const { circle, onChange } = mount({ initialX: 0, initialY: 0 });
      after(circle, pe("pointerdown", { pointerId: 1, clientX: 200, clientY: 200 }));
      after(circle, pe("pointermove", { pointerId: 1, clientX: 240, clientY: 160 }));
      const last = onChange.mock.calls.at(-1)!;
      expect(last[0]).toBeCloseTo(1, 6);
      expect(last[1]).toBeCloseTo(1, 6);
    });

    it("pipes drag updates through constrain (clamp bounds)", () => {
      const { circle, onChange } = mount({
        initialX: 0,
        initialY: 0,
        constrain: clamp({ x: [-1, 1], y: [-1, 1] }),
      });
      // Drag far out of bounds — should clamp to (1, -1) since pxToUser
      // at the lower-right takes us to (large +x, large -y) and the
      // constraint pins to the box corner.
      after(circle, pe("pointerdown", { pointerId: 2, clientX: 200, clientY: 200 }));
      after(circle, pe("pointermove", { pointerId: 2, clientX: 400, clientY: 400 }));
      const last = onChange.mock.calls.at(-1)!;
      expect(last[0]).toBe(1);
      expect(last[1]).toBe(-1);
    });
  });
});
