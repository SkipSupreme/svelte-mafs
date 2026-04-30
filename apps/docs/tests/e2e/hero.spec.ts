import { expect, test } from "@playwright/test";

/**
 * Hero smoke + reactive-apple integration test (T-1 of /plan-eng-review).
 *
 * The differentiator: visitor lands on / and the apple visibly reacts to
 * what they do to the hero widget. This test asserts the wiring works
 * end-to-end in a real browser — not just that events fire, but that the
 * apple's data-tinker-state attribute changes in response.
 */

test.describe("homepage hero", () => {
  test("apple, widget, hero region all hydrated", async ({ page }) => {
    await page.goto("/");

    await expect(page.locator("[data-hero-region]")).toBeVisible();
    await expect(page.locator(".tinker")).toBeVisible();
    await expect(page.locator(".widget-hero")).toBeVisible();

    // 27 letters in the embedding constellation.
    await expect(page.locator(".widget-hero .lbl")).toHaveCount(27);
  });

  test("apple state defaults to idle", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".tinker")).toHaveAttribute("data-tinker-state", "idle");
  });

  test("hero CTAs render above the fold", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("a.btn.primary")).toContainText("Sign up");
    await expect(page.locator("a.btn.secondary")).toContainText("Explore the course");
  });

  test("COMING SOON cards are gone from upper fold", async ({ page }) => {
    await page.goto("/");
    // No tile-soon class anywhere on the page.
    await expect(page.locator(".tile-soon")).toHaveCount(0);
  });

  test("apple reacts to widget hover (focus event)", async ({ page }) => {
    await page.goto("/");
    const apple = page.locator(".tinker");
    await expect(apple).toHaveAttribute("data-tinker-state", "idle");

    // Move pointer over the widget stage to trigger tinker:hero:focus.
    const stage = page.locator(".widget-hero .stage");
    await stage.hover({ position: { x: 200, y: 150 } });

    // State should flip to 'focus' within a short window.
    await expect(apple).toHaveAttribute("data-tinker-state", /focus|drag|threshold/, { timeout: 2000 });
  });
});
