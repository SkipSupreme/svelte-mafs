import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import type { Vec2 } from "../vec.js";
import { drag, type DragOptions } from "./drag.js";

beforeAll(() => {
  // jsdom 25 doesn't ship pointer capture on Element.prototype.
  const proto = Element.prototype as unknown as {
    setPointerCapture?: (id: number) => void;
    releasePointerCapture?: (id: number) => void;
    hasPointerCapture?: (id: number) => boolean;
  };
  if (!proto.setPointerCapture) proto.setPointerCapture = () => {};
  if (!proto.releasePointerCapture) proto.releasePointerCapture = () => {};
  if (!proto.hasPointerCapture) proto.hasPointerCapture = () => false;
});

function pe(
  type: string,
  init: {
    pointerId?: number;
    clientX?: number;
    clientY?: number;
    button?: number;
  } = {},
): Event {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, {
    pointerId: init.pointerId ?? 1,
    clientX: init.clientX ?? 0,
    clientY: init.clientY ?? 0,
    button: init.button ?? 0,
    buttons: 1,
  });
  return e;
}

const identity = (px: Vec2): Vec2 => [px[0], px[1]];
const affine = (px: Vec2): Vec2 => [(px[0] - 100) / 10, (px[1] - 100) / 10];

function mount(options: DragOptions) {
  const node = document.createElement("div");
  document.body.appendChild(node);
  const lifecycle = drag(node, options);
  return { node, lifecycle };
}

describe("drag action", () => {
  beforeEach(() => {
    document.body.replaceChildren();
  });

  it("calls onDragStart with pxToUser(clientXY) on pointerdown", () => {
    const onDragStart = vi.fn();
    const { node } = mount({ pxToUser: affine, onDragStart });
    node.dispatchEvent(pe("pointerdown", { pointerId: 2, clientX: 150, clientY: 200 }));
    expect(onDragStart).toHaveBeenCalledTimes(1);
    expect(onDragStart).toHaveBeenCalledWith([5, 10]);
  });

  it("calls onDrag on pointermove with active pointer id", () => {
    const onDrag = vi.fn();
    const { node } = mount({ pxToUser: affine, onDrag });
    node.dispatchEvent(pe("pointerdown", { pointerId: 3, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 3, clientX: 130, clientY: 160 }));
    expect(onDrag).toHaveBeenCalledTimes(1);
    expect(onDrag).toHaveBeenCalledWith([3, 6]);
  });

  it("ignores pointermove from a different pointerId", () => {
    const onDrag = vi.fn();
    const { node } = mount({ pxToUser: identity, onDrag });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 99, clientX: 50, clientY: 50 }));
    expect(onDrag).not.toHaveBeenCalled();
  });

  it("ignores pointermove before any pointerdown", () => {
    const onDrag = vi.fn();
    const { node } = mount({ pxToUser: identity, onDrag });
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 10, clientY: 10 }));
    expect(onDrag).not.toHaveBeenCalled();
  });

  it("emits onDragEnd on pointerup and stops further onDrag calls", () => {
    const onDrag = vi.fn();
    const onDragEnd = vi.fn();
    const { node } = mount({ pxToUser: identity, onDrag, onDragEnd });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 5, clientY: 5 }));
    node.dispatchEvent(pe("pointerup", { pointerId: 1, clientX: 5, clientY: 5 }));
    expect(onDragEnd).toHaveBeenCalledTimes(1);
    onDrag.mockClear();
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 9, clientY: 9 }));
    expect(onDrag).not.toHaveBeenCalled();
  });

  it("emits onDragEnd on pointercancel", () => {
    const onDragEnd = vi.fn();
    const { node } = mount({ pxToUser: identity, onDragEnd });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointercancel", { pointerId: 1 }));
    expect(onDragEnd).toHaveBeenCalledTimes(1);
  });

  it("ignores a second pointerdown while already active", () => {
    const onDragStart = vi.fn();
    const onDrag = vi.fn();
    const { node } = mount({ pxToUser: identity, onDragStart, onDrag });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointerdown", { pointerId: 2, clientX: 50, clientY: 50 }));
    expect(onDragStart).toHaveBeenCalledTimes(1);
    node.dispatchEvent(pe("pointermove", { pointerId: 2, clientX: 60, clientY: 60 }));
    expect(onDrag).not.toHaveBeenCalled();
  });

  it("calls setPointerCapture on pointerdown", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    const spy = vi.spyOn(node, "setPointerCapture");
    drag(node, { pxToUser: identity });
    node.dispatchEvent(pe("pointerdown", { pointerId: 7, clientX: 0, clientY: 0 }));
    expect(spy).toHaveBeenCalledWith(7);
  });

  it("swallows errors from releasePointerCapture on pointerup", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    drag(node, { pxToUser: identity });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    const spy = vi
      .spyOn(node, "releasePointerCapture")
      .mockImplementation(() => {
        throw new DOMException("InvalidStateError");
      });
    expect(() =>
      node.dispatchEvent(pe("pointerup", { pointerId: 1, clientX: 0, clientY: 0 })),
    ).not.toThrow();
    expect(spy).toHaveBeenCalledWith(1);
  });

  it("update() swaps options: new callbacks hit, old ones don't", () => {
    const firstDrag = vi.fn();
    const secondDrag = vi.fn();
    const { node, lifecycle } = mount({ pxToUser: identity, onDrag: firstDrag });
    lifecycle!.update?.({ pxToUser: identity, onDrag: secondDrag });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 10, clientY: 10 }));
    expect(firstDrag).not.toHaveBeenCalled();
    expect(secondDrag).toHaveBeenCalledTimes(1);
    expect(secondDrag).toHaveBeenCalledWith([10, 10]);
  });

  it("update() with a new pxToUser applies to subsequent events", () => {
    const onDrag = vi.fn();
    const { node, lifecycle } = mount({ pxToUser: identity, onDrag });
    lifecycle!.update?.({ pxToUser: affine, onDrag });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 150, clientY: 200 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 130, clientY: 160 }));
    expect(onDrag).toHaveBeenCalledWith([3, 6]);
  });

  it("destroy() removes all listeners — no callbacks fire afterwards", () => {
    const onDragStart = vi.fn();
    const onDrag = vi.fn();
    const onDragEnd = vi.fn();
    const { node, lifecycle } = mount({
      pxToUser: identity,
      onDragStart,
      onDrag,
      onDragEnd,
    });
    lifecycle!.destroy?.();
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 5, clientY: 5 }));
    node.dispatchEvent(pe("pointerup", { pointerId: 1, clientX: 5, clientY: 5 }));
    expect(onDragStart).not.toHaveBeenCalled();
    expect(onDrag).not.toHaveBeenCalled();
    expect(onDragEnd).not.toHaveBeenCalled();
  });

  it("destroy() during an active drag releases capture", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    const lifecycle = drag(node, { pxToUser: identity });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    const releaseSpy = vi.spyOn(node, "releasePointerCapture");
    lifecycle.destroy?.();
    expect(releaseSpy).toHaveBeenCalledWith(1);
  });

  it("destroy() swallows releasePointerCapture errors (e.g. page unmount mid-drag)", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    const lifecycle = drag(node, { pxToUser: identity });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    vi.spyOn(node, "releasePointerCapture").mockImplementation(() => {
      throw new DOMException("InvalidStateError");
    });
    expect(() => lifecycle.destroy?.()).not.toThrow();
  });

  it("swallows setPointerCapture errors on pointerdown", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    vi.spyOn(node, "setPointerCapture").mockImplementation(() => {
      throw new DOMException("InvalidPointerId");
    });
    const onDragStart = vi.fn();
    drag(node, { pxToUser: identity, onDragStart });
    expect(() =>
      node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 10, clientY: 20 })),
    ).not.toThrow();
    expect(onDragStart).toHaveBeenCalledWith([10, 20]);
  });

  it("works on SVG elements", () => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    document.body.appendChild(svg);
    const onDragStart = vi.fn();
    drag(svg as unknown as HTMLElement, { pxToUser: identity, onDragStart });
    svg.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 42, clientY: 24 }));
    expect(onDragStart).toHaveBeenCalledWith([42, 24]);
  });
});
