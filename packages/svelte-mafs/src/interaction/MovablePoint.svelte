<script lang="ts">
  import { getCoordContext } from "../context/coordinate-context.js";
  import { drag } from "../gestures/drag.js";
  import { colors } from "../theme.js";
  import type { Vec2 } from "../vec.js";
  import type { Constraint } from "./constraints.js";

  type Props = {
    x: number;
    y: number;
    color?: string;
    /** Optional constraint piped through both drag and keyboard updates. */
    constrain?: Constraint;
    /** User-unit step for arrow keys. Shift multiplies by 10. */
    step?: number;
    /** Circle radius in user units. Defaults to a hit-target-friendly size. */
    radius?: number;
    /** Accessible label for screen readers. */
    label?: string;
  };

  let {
    x = $bindable(),
    y = $bindable(),
    color = colors.pink,
    constrain,
    step = 0.1,
    radius = 0.2,
    label = "Movable point",
  }: Props = $props();

  const ctx = getCoordContext();

  // Single funnel for every x/y update so constrain() runs consistently
  // whether the source is a drag, a keystroke, or anything added later.
  const setPos = (p: Vec2) => {
    const [nx, ny] = constrain ? constrain(p) : p;
    x = nx;
    y = ny;
  };

  const handleKeydown = (e: KeyboardEvent) => {
    const mul = e.shiftKey ? 10 : 1;
    let dx = 0;
    let dy = 0;
    switch (e.key) {
      case "ArrowLeft":
        dx = -step * mul;
        break;
      case "ArrowRight":
        dx = step * mul;
        break;
      // ArrowUp increments y because Mafs is math-y-up (the root <g
      // scale(1,-1)> flipped screen-y), so visually "up" is +y.
      case "ArrowUp":
        dy = step * mul;
        break;
      case "ArrowDown":
        dy = -step * mul;
        break;
      default:
        return;
    }
    e.preventDefault();
    setPos([x + dx, y + dy]);
  };
</script>

<!--
  aria-valuenow-x / aria-valuenow-y are plan-mandated non-standard attrs
  (ARIA only defines a single aria-valuenow). We emit both per-axis for
  tests + introspection, AND a real aria-valuenow (bound to x) plus a
  human aria-valuetext so actual screen readers get something meaningful.
  svelte-ignore suppresses the a11y lints for the non-standard per-axis
  attrs — they're intentional, not typos.
-->
<!-- svelte-ignore a11y_unknown_aria_attribute -->
<circle
  cx={x}
  cy={y}
  r={radius}
  fill={color}
  class="mafs-movable-point"
  role="slider"
  tabindex={0}
  aria-label={label}
  aria-valuenow={x}
  aria-valuenow-x={x}
  aria-valuenow-y={y}
  aria-valuetext={`x: ${x}, y: ${y}`}
  onkeydown={handleKeydown}
  use:drag={{
    pxToUser: ctx.pxToUser,
    onDrag: (p) => setPos(p),
  }}
/>

<style>
  .mafs-movable-point {
    cursor: grab;
    outline: none;
    /* Small translucent halo on the fill itself so the hit target visually
       matches the logical one. */
    fill-opacity: 0.9;
  }

  .mafs-movable-point:focus,
  .mafs-movable-point:focus-visible {
    outline: 2px solid var(--mafs-blue, #3b82f6);
    outline-offset: 2px;
  }

  .mafs-movable-point:active {
    cursor: grabbing;
  }
</style>
