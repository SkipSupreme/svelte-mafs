import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import type { Vec2 } from "../vec.js";
import { panZoom, type PanZoomOptions } from "./pan-zoom.js";

beforeAll(() => {
  const proto = Element.prototype as unknown as {
    setPointerCapture?: (id: number) => void;
    releasePointerCapture?: (id: number) => void;
  };
  if (!proto.setPointerCapture) proto.setPointerCapture = () => {};
  if (!proto.releasePointerCapture) proto.releasePointerCapture = () => {};
});

function pe(
  type: string,
  init: {
    pointerId?: number;
    clientX?: number;
    clientY?: number;
  } = {},
): Event {
  const e = new Event(type, { bubbles: true, cancelable: true });
  Object.assign(e, {
    pointerId: init.pointerId ?? 1,
    clientX: init.clientX ?? 0,
    clientY: init.clientY ?? 0,
    buttons: 1,
  });
  return e;
}

function we(init: {
  deltaY?: number;
  deltaX?: number;
  ctrlKey?: boolean;
  metaKey?: boolean;
  clientX?: number;
  clientY?: number;
} = {}): Event {
  const e = new Event("wheel", { bubbles: true, cancelable: true });
  Object.assign(e, {
    deltaY: init.deltaY ?? 0,
    deltaX: init.deltaX ?? 0,
    ctrlKey: init.ctrlKey ?? false,
    metaKey: init.metaKey ?? false,
    clientX: init.clientX ?? 0,
    clientY: init.clientY ?? 0,
  });
  return e;
}

const identity = (px: Vec2): Vec2 => [px[0], px[1]];
// user = (px - 100) / 10  →  linear so px-delta of (+10,+20) = user-delta of (+1,+2)
const affine = (px: Vec2): Vec2 => [(px[0] - 100) / 10, (px[1] - 100) / 10];

function mount(options: PanZoomOptions) {
  const node = document.createElement("div");
  document.body.appendChild(node);
  const lifecycle = panZoom(node, options);
  return { node, lifecycle };
}

describe("panZoom action — wheel zoom", () => {
  beforeEach(() => {
    document.body.replaceChildren();
  });

  it("emits onZoom with factor > 1 for scroll-up (negative deltaY) + ctrlKey", () => {
    const onZoom = vi.fn();
    const { node } = mount({ pxToUser: identity, onZoom });
    node.dispatchEvent(
      we({ deltaY: -100, ctrlKey: true, clientX: 50, clientY: 60 }),
    );
    expect(onZoom).toHaveBeenCalledTimes(1);
    const [factor, centerPx] = onZoom.mock.calls[0]!;
    expect(factor).toBeGreaterThan(1);
    expect(centerPx).toEqual([50, 60]);
  });

  it("emits onZoom with factor < 1 for scroll-down (positive deltaY)", () => {
    const onZoom = vi.fn();
    const { node } = mount({ pxToUser: identity, onZoom });
    node.dispatchEvent(we({ deltaY: 100, ctrlKey: true }));
    expect(onZoom).toHaveBeenCalledTimes(1);
    expect(onZoom.mock.calls[0]![0]).toBeLessThan(1);
  });

  it("metaKey also activates zoom", () => {
    const onZoom = vi.fn();
    const { node } = mount({ pxToUser: identity, onZoom });
    node.dispatchEvent(we({ deltaY: -50, metaKey: true }));
    expect(onZoom).toHaveBeenCalledTimes(1);
  });

  it("plain wheel (no modifier) does NOT emit onZoom (lets page scroll)", () => {
    const onZoom = vi.fn();
    const { node } = mount({ pxToUser: identity, onZoom });
    node.dispatchEvent(we({ deltaY: -100 }));
    expect(onZoom).not.toHaveBeenCalled();
  });

  it("calls preventDefault when zoom fires (stops browser zoom)", () => {
    const { node } = mount({ pxToUser: identity, onZoom: vi.fn() });
    const ev = we({ deltaY: -100, ctrlKey: true });
    node.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
  });

  it("does NOT preventDefault when wheel is ignored (no modifier)", () => {
    const { node } = mount({ pxToUser: identity, onZoom: vi.fn() });
    const ev = we({ deltaY: -100 });
    node.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(false);
  });

  it("enabled.zoom=false suppresses onZoom even with modifier", () => {
    const onZoom = vi.fn();
    const { node } = mount({
      pxToUser: identity,
      onZoom,
      enabled: { zoom: false },
    });
    node.dispatchEvent(we({ deltaY: -100, ctrlKey: true }));
    expect(onZoom).not.toHaveBeenCalled();
  });

  it("wheelSensitivity scales the factor", () => {
    const onZoomA = vi.fn();
    const onZoomB = vi.fn();
    const { node: nodeA } = mount({
      pxToUser: identity,
      onZoom: onZoomA,
      wheelSensitivity: 0.001,
    });
    const { node: nodeB } = mount({
      pxToUser: identity,
      onZoom: onZoomB,
      wheelSensitivity: 0.01,
    });
    nodeA.dispatchEvent(we({ deltaY: -100, ctrlKey: true }));
    nodeB.dispatchEvent(we({ deltaY: -100, ctrlKey: true }));
    const factorA = onZoomA.mock.calls[0]![0];
    const factorB = onZoomB.mock.calls[0]![0];
    expect(factorB).toBeGreaterThan(factorA);
  });
});

describe("panZoom action — pan via pointer drag", () => {
  beforeEach(() => {
    document.body.replaceChildren();
  });

  it("emits onPan with user-space delta between pointermove steps", () => {
    const onPan = vi.fn();
    const { node } = mount({ pxToUser: affine, onPan });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 110, clientY: 120 }));
    expect(onPan).toHaveBeenCalledTimes(1);
    expect(onPan).toHaveBeenCalledWith([1, 2]);
  });

  it("emits incremental deltas across successive moves (not cumulative)", () => {
    const onPan = vi.fn();
    const { node } = mount({ pxToUser: affine, onPan });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 110, clientY: 100 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 130, clientY: 100 }));
    expect(onPan).toHaveBeenCalledTimes(2);
    expect(onPan.mock.calls[0]![0]).toEqual([1, 0]);
    expect(onPan.mock.calls[1]![0]).toEqual([2, 0]);
  });

  it("pointermove without prior pointerdown does nothing", () => {
    const onPan = vi.fn();
    const { node } = mount({ pxToUser: identity, onPan });
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 50, clientY: 50 }));
    expect(onPan).not.toHaveBeenCalled();
  });

  it("pointerup ends pan; further moves do not emit", () => {
    const onPan = vi.fn();
    const { node } = mount({ pxToUser: affine, onPan });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 110, clientY: 100 }));
    node.dispatchEvent(pe("pointerup", { pointerId: 1, clientX: 110, clientY: 100 }));
    onPan.mockClear();
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 150, clientY: 100 }));
    expect(onPan).not.toHaveBeenCalled();
  });

  it("pointercancel ends pan", () => {
    const onPan = vi.fn();
    const { node } = mount({ pxToUser: affine, onPan });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointercancel", { pointerId: 1 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 130, clientY: 100 }));
    expect(onPan).not.toHaveBeenCalled();
  });

  it("ignores secondary pointers while one is active", () => {
    const onPan = vi.fn();
    const { node } = mount({ pxToUser: affine, onPan });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointerdown", { pointerId: 2, clientX: 200, clientY: 200 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 2, clientX: 220, clientY: 220 }));
    expect(onPan).not.toHaveBeenCalled();
  });

  it("enabled.pan=false suppresses pan handling (no capture, no onPan)", () => {
    const onPan = vi.fn();
    const node = document.createElement("div");
    document.body.appendChild(node);
    const captureSpy = vi.spyOn(node, "setPointerCapture");
    panZoom(node, { pxToUser: identity, onPan, enabled: { pan: false } });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 5, clientY: 5 }));
    expect(captureSpy).not.toHaveBeenCalled();
    expect(onPan).not.toHaveBeenCalled();
  });

  it("captures pointer on pan start and releases on end", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    panZoom(node, { pxToUser: identity, onPan: vi.fn() });
    const capture = vi.spyOn(node, "setPointerCapture");
    const release = vi.spyOn(node, "releasePointerCapture");
    node.dispatchEvent(pe("pointerdown", { pointerId: 5, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointerup", { pointerId: 5, clientX: 0, clientY: 0 }));
    expect(capture).toHaveBeenCalledWith(5);
    expect(release).toHaveBeenCalledWith(5);
  });
});

describe("panZoom action — lifecycle", () => {
  beforeEach(() => {
    document.body.replaceChildren();
  });

  it("update() swaps callbacks: new ones hit, old ones don't", () => {
    const firstPan = vi.fn();
    const secondPan = vi.fn();
    const { node, lifecycle } = mount({ pxToUser: affine, onPan: firstPan });
    lifecycle.update?.({ pxToUser: affine, onPan: secondPan });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 110, clientY: 100 }));
    expect(firstPan).not.toHaveBeenCalled();
    expect(secondPan).toHaveBeenCalledWith([1, 0]);
  });

  it("update() can re-enable pan mid-session", () => {
    const onPan = vi.fn();
    const { node, lifecycle } = mount({
      pxToUser: affine,
      onPan,
      enabled: { pan: false },
    });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 110, clientY: 100 }));
    expect(onPan).not.toHaveBeenCalled();
    lifecycle.update?.({ pxToUser: affine, onPan, enabled: { pan: true } });
    node.dispatchEvent(pe("pointerdown", { pointerId: 2, clientX: 100, clientY: 100 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 2, clientX: 110, clientY: 100 }));
    expect(onPan).toHaveBeenCalledWith([1, 0]);
  });

  it("destroy() removes all listeners", () => {
    const onPan = vi.fn();
    const onZoom = vi.fn();
    const { node, lifecycle } = mount({ pxToUser: identity, onPan, onZoom });
    lifecycle.destroy?.();
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 5, clientY: 5 }));
    node.dispatchEvent(we({ deltaY: -100, ctrlKey: true }));
    expect(onPan).not.toHaveBeenCalled();
    expect(onZoom).not.toHaveBeenCalled();
  });

  it("destroy() during active pan releases capture", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    const lifecycle = panZoom(node, { pxToUser: identity, onPan: vi.fn() });
    node.dispatchEvent(pe("pointerdown", { pointerId: 9, clientX: 0, clientY: 0 }));
    const release = vi.spyOn(node, "releasePointerCapture");
    lifecycle.destroy?.();
    expect(release).toHaveBeenCalledWith(9);
  });

  it("destroy() swallows releasePointerCapture errors (page unmount mid-pan)", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    const lifecycle = panZoom(node, { pxToUser: identity, onPan: vi.fn() });
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
    const onPan = vi.fn();
    panZoom(node, { pxToUser: affine, onPan });
    expect(() =>
      node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 100, clientY: 100 })),
    ).not.toThrow();
    node.dispatchEvent(pe("pointermove", { pointerId: 1, clientX: 110, clientY: 100 }));
    expect(onPan).toHaveBeenCalledWith([1, 0]);
  });

  it("swallows releasePointerCapture errors", () => {
    const node = document.createElement("div");
    document.body.appendChild(node);
    panZoom(node, { pxToUser: identity, onPan: vi.fn() });
    node.dispatchEvent(pe("pointerdown", { pointerId: 1, clientX: 0, clientY: 0 }));
    vi.spyOn(node, "releasePointerCapture").mockImplementation(() => {
      throw new DOMException("InvalidStateError");
    });
    expect(() =>
      node.dispatchEvent(pe("pointerup", { pointerId: 1, clientX: 0, clientY: 0 })),
    ).not.toThrow();
  });
});
