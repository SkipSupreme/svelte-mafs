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

  // Pointer events carry page-absolute clientX/Y, but consumers (e.g.
  // Mafs's coord context) provide a pxToUser that expects coords local to
  // the drag's reference element — for an SVG descendant, the enclosing
  // <svg>; for a plain HTML node, the node itself. Without this
  // translation, MovablePoint dragging was off by the SVG's page offset
  // whenever Mafs wasn't rendered at (0, 0); jsdom masked the bug because
  // its layout engine reports all rects as zero-sized / zero-offset.
  const toUser = (e: PointerLike): Vec2 => {
    const refEl =
      "ownerSVGElement" in node && node.ownerSVGElement
        ? node.ownerSVGElement
        : (node as HTMLElement | SVGElement);
    const rect = refEl.getBoundingClientRect();
    return opts.pxToUser([e.clientX - rect.left, e.clientY - rect.top]);
  };

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
