import type { ActionReturn } from "svelte/action";
import type { Vec2 } from "../vec.js";

export interface PanZoomOptions {
  pxToUser: (px: Vec2) => Vec2;
  onPan?: (deltaUser: Vec2) => void;
  onZoom?: (factor: number, centerPx: Vec2) => void;
  enabled?: { pan?: boolean; zoom?: boolean };
  /** e^(-deltaY * sensitivity) — default 0.002 gives ~1.22× per 100px scroll. */
  wheelSensitivity?: number;
}

const DEFAULT_SENSITIVITY = 0.002;

type PointerLike = { pointerId: number; clientX: number; clientY: number };
type WheelLike = {
  deltaY: number;
  ctrlKey: boolean;
  metaKey: boolean;
  clientX: number;
  clientY: number;
  preventDefault: () => void;
};

// TODO: pinch-zoom via multi-touch (track 2 pointers, emit onZoom from distance
// ratio). Skipped in unit tests — jsdom can't drive coordinated multi-pointer
// gestures. Add a Playwright fixture spec under tests/e2e when that harness
// lands (Stream 8).

export function panZoom(
  node: HTMLElement | SVGElement,
  initial: PanZoomOptions,
): ActionReturn<PanZoomOptions> {
  let opts: PanZoomOptions = initial;
  let activePointerId: number | null = null;
  let lastPx: Vec2 | null = null;

  const panEnabled = () => opts.enabled?.pan ?? true;
  const zoomEnabled = () => opts.enabled?.zoom ?? true;

  const onWheel = (raw: Event) => {
    if (!zoomEnabled() || !opts.onZoom) return;
    const e = raw as unknown as WheelLike;
    if (!(e.ctrlKey || e.metaKey)) return;
    const sensitivity = opts.wheelSensitivity ?? DEFAULT_SENSITIVITY;
    const factor = Math.exp(-e.deltaY * sensitivity);
    e.preventDefault();
    opts.onZoom(factor, [e.clientX, e.clientY]);
  };

  const onPointerDown = (raw: Event) => {
    if (!panEnabled()) return;
    if (activePointerId !== null) return;
    const e = raw as unknown as PointerLike;
    activePointerId = e.pointerId;
    lastPx = [e.clientX, e.clientY];
    try {
      node.setPointerCapture(e.pointerId);
    } catch {
      /* noop */
    }
  };

  const onPointerMove = (raw: Event) => {
    const e = raw as unknown as PointerLike;
    if (e.pointerId !== activePointerId || !lastPx) return;
    const curr: Vec2 = [e.clientX, e.clientY];
    const prevUser = opts.pxToUser(lastPx);
    const currUser = opts.pxToUser(curr);
    lastPx = curr;
    opts.onPan?.([currUser[0] - prevUser[0], currUser[1] - prevUser[1]]);
  };

  const endPan = (raw: Event) => {
    const e = raw as unknown as PointerLike;
    if (e.pointerId !== activePointerId) return;
    try {
      node.releasePointerCapture(e.pointerId);
    } catch {
      /* noop */
    }
    activePointerId = null;
    lastPx = null;
  };

  node.addEventListener("wheel", onWheel, { passive: false });
  node.addEventListener("pointerdown", onPointerDown);
  node.addEventListener("pointermove", onPointerMove);
  node.addEventListener("pointerup", endPan);
  node.addEventListener("pointercancel", endPan);

  return {
    update(next: PanZoomOptions) {
      opts = next;
    },
    destroy() {
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("pointerdown", onPointerDown);
      node.removeEventListener("pointermove", onPointerMove);
      node.removeEventListener("pointerup", endPan);
      node.removeEventListener("pointercancel", endPan);
      if (activePointerId !== null) {
        try {
          node.releasePointerCapture(activePointerId);
        } catch {
          /* noop */
        }
        activePointerId = null;
        lastPx = null;
      }
    },
  };
}
