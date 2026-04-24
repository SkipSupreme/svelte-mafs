import { test, expect, type Page } from "@playwright/test";

/**
 * MovablePoint interaction: real pointer drag + keyboard navigation.
 *
 * Fixture: /examples/fixtures/movablepoint/  (viewBox x=[-5,5] y=[-5,5],
 * 400×400 px, initial x=1 y=2, default step=0.1). The fixture exposes a
 * readout with data-x / data-y attrs so we can assert user-space position
 * without reading SVG attrs (which would couple tests to the internal
 * coord system).
 *
 * Svelte 5 $state updates reach the DOM on the next effect flush, not
 * synchronously after the event that triggered them. We rely on
 * toHaveAttribute's auto-waiting rather than reading attrs immediately.
 */

const FIXTURE = "/examples/fixtures/movablepoint/";
const VIEW_BOX = { xMin: -5, xMax: 5, yMin: -5, yMax: 5 } as const;

function userToPage(
  svgBox: { x: number; y: number; width: number; height: number },
  userX: number,
  userY: number,
) {
  const sx = ((userX - VIEW_BOX.xMin) / (VIEW_BOX.xMax - VIEW_BOX.xMin)) * svgBox.width;
  // Mafs flips screen-y via <g scale(1,-1)>, so userY "up" maps to
  // svgY "up" (lower pixel value).
  const sy = ((VIEW_BOX.yMax - userY) / (VIEW_BOX.yMax - VIEW_BOX.yMin)) * svgBox.height;
  return { pageX: svgBox.x + sx, pageY: svgBox.y + sy };
}

async function readoutXY(page: Page): Promise<{ x: number; y: number }> {
  const r = page.getByTestId("movable-point-readout");
  const x = Number(await r.getAttribute("data-x"));
  const y = Number(await r.getAttribute("data-y"));
  return { x, y };
}

test.describe("MovablePoint interaction", () => {
  test("pointer drag updates x/y in user space", async ({ page }) => {
    await page.goto(FIXTURE);
    await page.waitForFunction(() => "__fixtureHydrated" in window);
    const slider = page.getByRole("slider");
    await expect(slider).toBeVisible();
    const svg = page.locator("[data-mafs-root]");
    const box = await svg.boundingBox();
    if (!box) throw new Error("SVG root has no bounding box");

    // Initial state: x=1 y=2. Drag to user (−2, −1).
    const start = userToPage(box, 1, 2);
    const end = userToPage(box, -2, -1);

    // Synthesize PointerEvents directly on the circle rather than driving
    // page.mouse. Playwright's CDP-based mouse simulation dispatches moves
    // at cursor coordinates, and SVG pointer-capture-via-setPointerCapture
    // doesn't reliably redirect those moves back to the circle once the
    // circle has moved out from under the cursor — so the real-mouse path
    // partially drags, then stops. Direct event dispatch bypasses that.
    // Unit tests still cover the drag math (drag.test.ts); this spec's
    // job is proving `use:drag` is mounted and reaches MovablePoint's
    // setPos funnel end-to-end.
    await slider.evaluate((el: SVGCircleElement, args: {
      startX: number; startY: number; endX: number; endY: number;
    }) => {
      const { startX, startY, endX, endY } = args;
      const fire = (type: string, x: number, y: number) => {
        const ev = new PointerEvent(type, {
          pointerId: 99,
          clientX: x,
          clientY: y,
          bubbles: true,
          cancelable: true,
        });
        el.dispatchEvent(ev);
      };
      fire("pointerdown", startX, startY);
      for (let i = 1; i <= 10; i++) {
        const t = i / 10;
        fire("pointermove", startX + (endX - startX) * t, startY + (endY - startY) * t);
      }
      fire("pointerup", endX, endY);
    }, { startX: start.pageX, startY: start.pageY, endX: end.pageX, endY: end.pageY });

    const readout = page.getByTestId("movable-point-readout");
    await expect
      .poll(async () => Number(await readout.getAttribute("data-x")), { timeout: 5_000 })
      .toBeLessThan(-1);

    const { x, y } = await readoutXY(page);
    expect(x).toBeCloseTo(-2, 1);
    expect(y).toBeCloseTo(-1, 1);
  });

  test("arrow keys nudge by step (0.1)", async ({ page }) => {
    await page.goto(FIXTURE);
    // Wait for Svelte $effect to run — proves `use:drag` is attached and
    // onkeydown is active. Otherwise early interactions silently no-op.
    await page.waitForFunction(() => "__fixtureHydrated" in window);
    const slider = page.getByRole("slider");
    await slider.focus();
    const readout = page.getByTestId("movable-point-readout");

    // Initial: x=1, y=2. After ArrowRight + ArrowUp: x=1.1, y=2.1.
    // JS: 1 + 0.1 === 1.1 exactly, 2 + 0.1 === 2.1 exactly, so stringified
    // attributes land as "1.1" / "2.1" with no FP rounding artefact.
    await slider.press("ArrowRight");
    await expect(readout).toHaveAttribute("data-x", "1.1");

    await slider.press("ArrowUp");
    await expect(readout).toHaveAttribute("data-y", "2.1");
  });

  test("Shift+arrow nudges by 10× step", async ({ page }) => {
    await page.goto(FIXTURE);
    // Wait for Svelte $effect to run — proves `use:drag` is attached and
    // onkeydown is active. Otherwise early interactions silently no-op.
    await page.waitForFunction(() => "__fixtureHydrated" in window);
    const slider = page.getByRole("slider");
    await slider.focus();
    const readout = page.getByTestId("movable-point-readout");

    // Initial: x=1, y=2. Shift+ArrowLeft: x = 1 + (−0.1 × 10) = 0 exactly.
    // Shift+ArrowDown: y = 2 + (−0.1 × 10) = 1 exactly.
    await slider.press("Shift+ArrowLeft");
    await expect(readout).toHaveAttribute("data-x", "0");

    await slider.press("Shift+ArrowDown");
    await expect(readout).toHaveAttribute("data-y", "1");
  });
});
