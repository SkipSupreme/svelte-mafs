<script lang="ts">
  import {
    makeCoordContext,
    normalizeViewBox,
    setCoordContext,
    type CoordContext,
  } from "../context/coordinate-context.js";
  import type { MafsProps } from "./mafs-props.js";

  const {
    width,
    height,
    viewBox: propViewBox,
    preserveAspectRatio = true,
    // pan / zoom are accepted but inert in this stream. Gesture actions land
    // in Stream 3; the integration that mutates `vb` lands in Stream 6.
    pan: _pan = false,
    zoom: _zoom = false,
    children,
  }: MafsProps = $props();

  // ResizeObserver-driven "auto" sizing lands post-Phase-2. For now, fall back
  // to a concrete pixel value so `userToPx` is still well-defined.
  const AUTO_PX_FALLBACK = 500;

  let widthPx = $derived(typeof width === "number" ? width : AUTO_PX_FALLBACK);
  let heightPx = $derived(typeof height === "number" ? height : AUTO_PX_FALLBACK);

  // Prop-reactive for now. When Stream 6 wires pan/zoom gestures, this will
  // refactor to `$state` so gesture handlers can mutate it directly.
  let vb = $derived(normalizeViewBox(propViewBox));

  let reactive = $derived(makeCoordContext(vb, widthPx, heightPx));

  // Getter-based context: children see the latest transforms if vb later
  // mutates (without needing to re-publish context, which can only happen
  // during parent init).
  const ctx: CoordContext = {
    get userToPx() {
      return reactive.userToPx;
    },
    get pxToUser() {
      return reactive.pxToUser;
    },
    get viewBox() {
      return reactive.viewBox;
    },
    get widthPx() {
      return reactive.widthPx;
    },
    get heightPx() {
      return reactive.heightPx;
    },
  };
  setCoordContext(ctx);

  // The inner <g> flips y so children render with y-up math convention; that
  // means the SVG viewBox must span [-yMax, -yMin] on its y-axis, not
  // [yMin, yMax].
  const svgViewBox = $derived(
    `${vb.xMin} ${-vb.yMax} ${vb.xMax - vb.xMin} ${vb.yMax - vb.yMin}`,
  );
  const preserve = $derived(preserveAspectRatio ? "xMidYMid meet" : "none");
</script>

<svg
  viewBox={svgViewBox}
  width={widthPx}
  height={heightPx}
  preserveAspectRatio={preserve}
  role="img"
  style="aspect-ratio: {widthPx} / {heightPx}; overflow: visible;"
  data-mafs-root
>
  <g transform="scale(1, -1)">
    {@render children?.()}
  </g>
</svg>
