<script lang="ts">
  import katex from "katex";
  import { getCoordContext } from "../context/coordinate-context.js";
  import { colors } from "../theme.js";

  type Props = {
    x: number;
    y: number;
    latex: string;
    size?: number;
    color?: string;
  };

  const {
    x,
    y,
    latex,
    size = 14,
    color = colors.fg,
  }: Props = $props();

  const ctx = getCoordContext();

  // KaTeX outputs static HTML — no DOM needed, so this is safe in SSR and
  // tests. throwOnError: false turns parse failures into visible red markup
  // instead of crashing the mount.
  const html = $derived(
    katex.renderToString(latex, { throwOnError: false, output: "html" }),
  );

  // HTML inside <foreignObject> lives in the SVG's user-unit coordinate
  // system, so a raw `font-size: 14px` becomes 14 *user units* — huge when
  // the viewBox spans a few units. Convert pixel intent back into user
  // units using the current px-per-user ratio.
  const pxPerUnit = $derived(
    (ctx.viewBox.xMax - ctx.viewBox.xMin) === 0
      ? 1
      : ctx.widthPx / (ctx.viewBox.xMax - ctx.viewBox.xMin),
  );
  const fontUserUnits = $derived(size / pxPerUnit);
</script>

<!--
  `width=1 height=1` with `overflow: visible` gives the inner HTML a valid
  layout box while letting glyphs render outside it. The inner <div> applies
  `scale(1, -1)` to cancel Mafs's outer y-flip, and `translate(-50%, -50%)`
  to center the rendered math on (x, y).
-->
<foreignObject
  x={x}
  y={y}
  width="1"
  height="1"
  style="overflow: visible;"
>
  <div
    xmlns="http://www.w3.org/1999/xhtml"
    class="mafs-text"
    style:transform="scale(1, -1) translate(-50%, -50%)"
    style:transform-origin="0 0"
    style:position="absolute"
    style:color={color}
    style:font-size={`${fontUserUnits}px`}
    style:line-height="1"
    style:white-space="nowrap"
  >
    {@html html}
  </div>
</foreignObject>
