import type { ActionReturn } from "svelte/action";
import type { Vec2 } from "../vec.js";

export interface DragOptions {
  pxToUser: (px: Vec2) => Vec2;
  onDragStart?: (userPos: Vec2) => void;
  onDrag?: (userPos: Vec2) => void;
  onDragEnd?: () => void;
}

type PointerLike = {
  pointerId: number;
  clientX: number;
  clientY: number;
};

export function drag(
  node: HTMLElement | SVGElement,
  initial: DragOptions,
): ActionReturn<DragOptions> {
  let opts: DragOptions = initial;
  let activePointerId: number | null = null;

  const toUser = (e: PointerLike): Vec2 => opts.pxToUser([e.clientX, e.clientY]);

  const onPointerDown = (raw: Event) => {
    const e = raw as unknown as PointerLike;
    if (activePointerId !== null) return;
    activePointerId = e.pointerId;
    try {
      node.setPointerCapture(e.pointerId);
    } catch {
      // Safari can throw if the pointer isn't active yet; safe to ignore.
    }
    opts.onDragStart?.(toUser(e));
  };

  const onPointerMove = (raw: Event) => {
    const e = raw as unknown as PointerLike;
    if (e.pointerId !== activePointerId) return;
    opts.onDrag?.(toUser(e));
  };

  const endDrag = (raw: Event) => {
    const e = raw as unknown as PointerLike;
    if (e.pointerId !== activePointerId) return;
    try {
      node.releasePointerCapture(e.pointerId);
    } catch {
      // Capture may already have been lost on some browsers.
    }
    activePointerId = null;
    opts.onDragEnd?.();
  };

  node.addEventListener("pointerdown", onPointerDown);
  node.addEventListener("pointermove", onPointerMove);
  node.addEventListener("pointerup", endDrag);
  node.addEventListener("pointercancel", endDrag);

  return {
    update(next: DragOptions) {
      opts = next;
    },
    destroy() {
      node.removeEventListener("pointerdown", onPointerDown);
      node.removeEventListener("pointermove", onPointerMove);
      node.removeEventListener("pointerup", endDrag);
      node.removeEventListener("pointercancel", endDrag);
      if (activePointerId !== null) {
        try {
          node.releasePointerCapture(activePointerId);
        } catch {
          /* noop */
        }
        activePointerId = null;
      }
    },
  };
};
