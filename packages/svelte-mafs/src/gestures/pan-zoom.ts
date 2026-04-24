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

export function panZoom(
  node: HTMLElement | SVGElement,
  initial: PanZoomOptions,
): ActionReturn<PanZoomOptions> {
  let opts: PanZoomOptions = initial;

  // Insertion-ordered so the first two fingers consistently drive pinch math
  // even if a third pointer joins and then leaves.
  const pointers = new Map<number, Vec2>();
  let panPointerId: number | null = null;
  let lastPanPx: Vec2 | null = null;
  let lastPinchDistance: number | null = null;

  const panEnabled = () => opts.enabled?.pan ?? true;
  const zoomEnabled = () => opts.enabled?.zoom ?? true;

  const distance = (a: Vec2, b: Vec2): number => {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
  };

  const midpoint = (a: Vec2, b: Vec2): Vec2 => [
    (a[0] + b[0]) / 2,
    (a[1] + b[1]) / 2,
  ];

  const firstTwo = (): [Vec2, Vec2] => {
    const iter = pointers.values();
    const a = iter.next().value as Vec2;
    const b = iter.next().value as Vec2;
    return [a, b];
  };

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
    const e = raw as unknown as PointerLike;
    const pos: Vec2 = [e.clientX, e.clientY];
    pointers.set(e.pointerId, pos);

    // Second pointer arrives → enter pinch mode. Any in-flight pan is abandoned
    // (there's no "pan end" signal by design — onPan just stops firing until
    // all fingers lift). Capture this pointer so slides outside the node still
    // deliver moves.
    if (pointers.size === 2 && zoomEnabled()) {
      panPointerId = null;
      lastPanPx = null;
      const [a, b] = firstTwo();
      lastPinchDistance = distance(a, b);
      try {
        node.setPointerCapture(e.pointerId);
      } catch {
        /* noop */
      }
      return;
    }

    // First pointer → start pan (only capture if pan is enabled, otherwise we
    // swallow pointer events the page may want).
    if (pointers.size === 1 && panEnabled() && panPointerId === null) {
      panPointerId = e.pointerId;
      lastPanPx = pos;
      try {
        node.setPointerCapture(e.pointerId);
      } catch {
        /* noop */
      }
    }
  };

  const onPointerMove = (raw: Event) => {
    const e = raw as unknown as PointerLike;
    if (!pointers.has(e.pointerId)) return;
    const pos: Vec2 = [e.clientX, e.clientY];
    pointers.set(e.pointerId, pos);

    // Pinch mode wins over pan. Factor = new-distance / old-distance; the
    // center is the midpoint of the two fingers in page-pixel space.
    if (pointers.size >= 2 && zoomEnabled() && lastPinchDistance !== null && opts.onZoom) {
      const [a, b] = firstTwo();
      const nextDistance = distance(a, b);
      if (lastPinchDistance > 0 && nextDistance > 0) {
        opts.onZoom(nextDistance / lastPinchDistance, midpoint(a, b));
      }
      lastPinchDistance = nextDistance;
      return;
    }

    if (e.pointerId === panPointerId && lastPanPx) {
      const prevUser = opts.pxToUser(lastPanPx);
      const currUser = opts.pxToUser(pos);
      lastPanPx = pos;
      opts.onPan?.([currUser[0] - prevUser[0], currUser[1] - prevUser[1]]);
    }
  };

  const endPointer = (raw: Event) => {
    const e = raw as unknown as PointerLike;
    if (!pointers.has(e.pointerId)) return;
    pointers.delete(e.pointerId);

    try {
      node.releasePointerCapture(e.pointerId);
    } catch {
      /* noop */
    }

    // Dropping below 2 pointers exits pinch. We intentionally do NOT auto-
    // resume pan with the remaining finger — mid-gesture mode switches are
    // surprising. User lifts and re-presses to pan again.
    if (pointers.size < 2) {
      lastPinchDistance = null;
    }

    if (e.pointerId === panPointerId) {
      panPointerId = null;
      lastPanPx = null;
    }
  };

  node.addEventListener("wheel", onWheel, { passive: false });
  node.addEventListener("pointerdown", onPointerDown);
  node.addEventListener("pointermove", onPointerMove);
  node.addEventListener("pointerup", endPointer);
  node.addEventListener("pointercancel", endPointer);

  return {
    update(next: PanZoomOptions) {
      opts = next;
    },
    destroy() {
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("pointerdown", onPointerDown);
      node.removeEventListener("pointermove", onPointerMove);
      node.removeEventListener("pointerup", endPointer);
      node.removeEventListener("pointercancel", endPointer);
      for (const id of pointers.keys()) {
        try {
          node.releasePointerCapture(id);
        } catch {
          /* noop */
        }
      }
      pointers.clear();
      panPointerId = null;
      lastPanPx = null;
      lastPinchDistance = null;
    },
  };
}
