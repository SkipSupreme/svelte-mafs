import { test, expect } from "@playwright/test";

/**
 * Pinch-zoom e2e. The fixture at /examples/fixtures/pan-zoom/ attaches the
 * `use:panZoom` action to a 400×400 div and parks every onZoom / onPan
 * invocation on `window.__zoomCalls` / `window.__panCalls` so the spec can
 * read them back after a coordinated pointer sequence.
 *
 * We inject PointerEvents directly via page.evaluate rather than driving
 * the browser's touchscreen. The panZoom action listens for `pointerdown /
 * pointermove / pointerup`, not `touchstart/*`, and synthetic PointerEvents
 * exercise exactly the path we care about — regardless of whether the
 * project has `hasTouch: true`. That makes this spec portable across
 * desktop chromium and mobile emulation projects.
 */

const FIXTURE = "/examples/fixtures/pan-zoom/";

type ZoomCall = { factor: number; centerPx: [number, number] };

// Dispatch a sequence of PointerEvents at a target in the page. The
// `steps` per move provide intermediate pointermove frames so factor=
// new/old stays stable (a single jump produces one huge ratio; several
// steps produce a stream of small ratios multiplying to the expected
// spread, which is more realistic and easier to assert).
async function dispatchPinch(
  page: import("@playwright/test").Page,
  args: {
    origin: [number, number];
    startA: [number, number];
    startB: [number, number];
    endA: [number, number];
    endB: [number, number];
    steps?: number;
  },
) {
  await page.evaluate(
    ({ origin, startA, startB, endA, endB, steps }) => {
      const target = document.querySelector('[data-testid="pan-zoom-target"]');
      if (!(target instanceof HTMLElement)) throw new Error("no target");
      const rect = target.getBoundingClientRect();
      const ox = rect.left + origin[0];
      const oy = rect.top + origin[1];

      const fire = (type: string, id: number, localX: number, localY: number) => {
        const ev = new PointerEvent(type, {
          pointerId: id,
          clientX: ox + localX,
          clientY: oy + localY,
          bubbles: true,
          cancelable: true,
          pointerType: "touch",
        });
        target.dispatchEvent(ev);
      };

      fire("pointerdown", 1, startA[0], startA[1]);
      fire("pointerdown", 2, startB[0], startB[1]);
      for (let i = 1; i <= steps; i++) {
        const t = i / steps;
        const ax = startA[0] + (endA[0] - startA[0]) * t;
        const ay = startA[1] + (endA[1] - startA[1]) * t;
        const bx = startB[0] + (endB[0] - startB[0]) * t;
        const by = startB[1] + (endB[1] - startB[1]) * t;
        fire("pointermove", 1, ax, ay);
        fire("pointermove", 2, bx, by);
      }
      fire("pointerup", 1, endA[0], endA[1]);
      fire("pointerup", 2, endB[0], endB[1]);
    },
    { ...args, steps: args.steps ?? 5 },
  );
}

async function readZoomCalls(page: import("@playwright/test").Page): Promise<ZoomCall[]> {
  return page.evaluate(
    () => (window as unknown as { __zoomCalls: ZoomCall[] }).__zoomCalls ?? [],
  );
}

test.describe("panZoom pinch gesture", () => {
  test("fingers spreading outward emits onZoom factors > 1", async ({ page }) => {
    await page.goto(FIXTURE);
    await page.waitForFunction(() => "__fixtureHydrated" in window && "__zoomCalls" in window);
    // Fingers start 100px apart, end 300px apart → aggregate factor = 3.
    await dispatchPinch(page, {
      origin: [0, 200],
      startA: [150, 0],
      startB: [250, 0],
      endA: [50, 0],
      endB: [350, 0],
    });
    const calls = await readZoomCalls(page);
    expect(calls.length).toBeGreaterThan(0);
    for (const { factor } of calls) expect(factor).toBeGreaterThan(1);
    const aggregate = calls.reduce((acc, { factor }) => acc * factor, 1);
    expect(aggregate).toBeCloseTo(3, 1);
  });

  test("fingers pinching inward emits onZoom factors < 1", async ({ page }) => {
    await page.goto(FIXTURE);
    await page.waitForFunction(() => "__fixtureHydrated" in window && "__zoomCalls" in window);
    // Start 300px apart, end 100px apart → aggregate factor = 1/3.
    await dispatchPinch(page, {
      origin: [0, 200],
      startA: [50, 0],
      startB: [350, 0],
      endA: [150, 0],
      endB: [250, 0],
    });
    const calls = await readZoomCalls(page);
    expect(calls.length).toBeGreaterThan(0);
    for (const { factor } of calls) expect(factor).toBeLessThan(1);
    const aggregate = calls.reduce((acc, { factor }) => acc * factor, 1);
    expect(aggregate).toBeCloseTo(1 / 3, 1);
  });

  test("onZoom centerPx tracks midpoint as fingers move", async ({ page }) => {
    await page.goto(FIXTURE);
    await page.waitForFunction(() => "__fixtureHydrated" in window && "__zoomCalls" in window);
    await dispatchPinch(page, {
      origin: [0, 200],
      startA: [100, 0],
      startB: [300, 0],
      endA: [50, 50],
      endB: [350, -50],
      steps: 1,
    });
    const calls = await readZoomCalls(page);
    expect(calls.length).toBeGreaterThan(0);
    // Final call: pointerA at end (50, 50), pointerB at end (350, -50).
    // Midpoint = (200, 0) in local space → (origin.x + 200, origin.y + 0)
    // in page space. Fixture has padding so we can't hard-code absolute
    // page coords, but midpoint X should equal midpoint-of-starts'
    // averaged endpoints. Assert the relative relationship.
    const target = await page.locator('[data-testid="pan-zoom-target"]').boundingBox();
    if (!target) throw new Error("no target box");
    const last = calls[calls.length - 1]!;
    // dispatchPinch origin = [0, 200] → every event is fired at
    // (target.x + localX, target.y + 200 + localY). endA=(50,50),
    // endB=(350,−50), midpoint local = (200, 200) in page-target space →
    // (target.x+200, target.y+200) in page-absolute.
    expect(last.centerPx[0]).toBeCloseTo(target.x + 200, 0);
    expect(last.centerPx[1]).toBeCloseTo(target.y + 200, 0);
  });
});
