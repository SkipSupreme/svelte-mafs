<script lang="ts">
  import { panZoom } from "svelte-mafs";

  // Captures every onZoom call on window so the e2e spec can read them
  // back via page.evaluate() / CDPSession after dispatching touches.
  type ZoomCall = { factor: number; centerPx: [number, number] };
  type PanCall = { deltaUser: [number, number] };

  const zoomCalls: ZoomCall[] = [];
  const panCalls: PanCall[] = [];

  if (typeof window !== "undefined") {
    (window as unknown as { __zoomCalls: ZoomCall[] }).__zoomCalls = zoomCalls;
    (window as unknown as { __panCalls: PanCall[] }).__panCalls = panCalls;
  }

  // Hydration marker — tests wait for this before dispatching pointer
  // events so the panZoom action is guaranteed attached.
  $effect(() => {
    (window as unknown as { __fixtureHydrated: boolean }).__fixtureHydrated = true;
  });

  // pxToUser: identity — the spec works in page-pixel space, which is
  // what panZoom's onZoom already reports for centerPx.
  const pxToUser = (px: [number, number]): [number, number] => px;

  const onPan = (deltaUser: [number, number]) => {
    panCalls.push({ deltaUser });
  };

  const onZoom = (factor: number, centerPx: [number, number]) => {
    zoomCalls.push({ factor, centerPx });
  };
</script>

<div
  use:panZoom={{ pxToUser, onPan, onZoom }}
  data-testid="pan-zoom-target"
  class="target"
>
  pan / zoom target
</div>

<style>
  .target {
    width: 400px;
    height: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f3f4f6;
    border: 2px dashed #9ca3af;
    color: #374151;
    font-size: 14px;
    touch-action: none; /* let pointer events fire without browser gestures */
    user-select: none;
  }
</style>
