import { test, expect } from "@playwright/test";

/**
 * Visual regression for display primitives, plots, text, and transforms.
 * One baseline PNG per fixture × Playwright project lives in
 * `__screenshots__/`; regenerate with `--update-snapshots` after an
 * intentional visual change.
 *
 * Screenshots target the `[data-fixture]` wrapper, not the full viewport —
 * keeps tests immune to changes in layout / scrollbars / nav outside the
 * fixture body.
 */

// Fixtures are (slug, maxDiffPixelRatio). null ratio = inherit the global
// 0.02 threshold from playwright.config.ts. Looser thresholds below are for
// fixtures that render non-deterministically in ways visual regression
// shouldn't trip on:
//   - adaptive sampling near asymptotes can land on slightly different
//     sample points across browser engines (plot-of-x tan / 1/x)
//   - KaTeX renders via browser-default fonts here (no CSS import), and
//     default font metrics vary by OS
const FIXTURES: ReadonlyArray<readonly [slug: string, threshold: number | null]> = [
  ["point", null],
  ["line-segment", null],
  ["line-through-points", null],
  ["circle", null],
  ["ellipse", null],
  ["polygon", null],
  ["vector", null],
  ["coordinates-cartesian", null],
  ["plot-of-x", 0.05],
  ["plot-of-y", null],
  ["plot-parametric", null],
  ["plot-vector-field", null],
  ["plot-inequality", null],
  ["text", 0.15],
  ["transform", null],
];

for (const [slug, threshold] of FIXTURES) {
  test(`fixture ${slug} renders deterministically`, async ({ page }) => {
    await page.goto(`/examples/fixtures/${slug}/`);
    const region = page.locator(`[data-fixture="${slug}"]`);
    await expect(region).toBeVisible();
    const opts = threshold === null ? undefined : { maxDiffPixelRatio: threshold };
    await expect(region).toHaveScreenshot(`${slug}.png`, opts);
  });
}

test("fixture movablepoint renders deterministically at initial state", async ({ page }) => {
  // MovablePoint is interactive, but its initial render is static. Drag /
  // keyboard behavior lives in specs/interaction/movable-point.spec.ts.
  await page.goto("/examples/fixtures/movablepoint/");
  const region = page.locator('[data-fixture="movablepoint"]');
  await expect(region).toBeVisible();
  // Wait for Svelte hydration so the focusable slider halo is in its final
  // state before we screenshot.
  await page.waitForFunction(() => {
    const el = document.querySelector('[role="slider"]');
    return el !== null;
  });
  await expect(region).toHaveScreenshot("movablepoint.png");
});
