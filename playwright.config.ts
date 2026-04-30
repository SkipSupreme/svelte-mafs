import { defineConfig, devices } from "@playwright/test";

/**
 * Repo-level Playwright config (TENSION-2 of /plan-eng-review).
 *
 * Two projects under one config — tests stay package-local but the config
 * is shared:
 *   - `svelte-mafs` runs the primitive harness specs in
 *     packages/svelte-mafs/tests/e2e/
 *   - `docs` runs the apps/docs E2E specs (homepage hero, widget-lab,
 *     reactive-apple integration) in apps/docs/tests/e2e/
 *
 * Both projects share the same webServer — `pnpm -F docs dev` serves both
 * package surfaces (svelte-mafs primitives are imported into docs pages).
 *
 * The existing packages/svelte-mafs/playwright.config.ts remains as a
 * back-compat surface for `pnpm -F svelte-mafs test:e2e`. The repo-level
 * config is preferred for new work.
 */
const PORT = 4321;
const BASE_URL = `http://localhost:${PORT}`;

export default defineConfig({
  // Don't set a top-level testDir — let each project specify its own.
  snapshotPathTemplate: "{testDir}/__screenshots__/{projectName}/{arg}{ext}",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI
    ? [["github"], ["html", { open: "never", outputFolder: "playwright-report" }]]
    : [["list"], ["html", { open: "on-failure", outputFolder: "playwright-report" }]],

  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: process.env.CI ? "retain-on-failure" : "off",
  },

  expect: {
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.02,
      animations: "disabled",
    },
  },

  projects: [
    {
      name: "svelte-mafs-chromium",
      testDir: "./packages/svelte-mafs/tests/e2e",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "svelte-mafs-iphone",
      testDir: "./packages/svelte-mafs/tests/e2e",
      use: { ...devices["iPhone 14"] },
    },
    {
      name: "docs-chromium",
      testDir: "./apps/docs/tests/e2e",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "docs-tablet",
      testDir: "./apps/docs/tests/e2e",
      use: { ...devices["iPad Mini"] },
    },
    {
      name: "docs-iphone",
      testDir: "./apps/docs/tests/e2e",
      use: { ...devices["iPhone 14"] },
    },
  ],

  webServer: {
    command: "pnpm -F docs dev --port 4321 --host 127.0.0.1",
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
