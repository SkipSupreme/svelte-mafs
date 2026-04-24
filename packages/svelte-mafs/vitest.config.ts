import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { svelteTesting } from "@testing-library/svelte/vite";

export default defineConfig({
  // svelteTesting() adds `browser` to resolve.conditions so svelte/mount is
  // available, wires noExternal for @testing-library/svelte-core, and
  // registers the auto-cleanup beforeEach. Stream-1 follow-up: unblocks
  // .svelte component tests in every downstream stream.
  plugins: [svelte({ hot: false }), svelteTesting()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
    // Playwright e2e specs share the .spec.ts extension — exclude the
    // tests/e2e tree so vitest doesn't try to load them in jsdom and
    // trip on the `@playwright/test` import.
    exclude: [
      "**/node_modules/**",
      "**/dist/**",
      "**/.svelte-kit/**",
      "tests/e2e/**",
    ],
    coverage: {
      provider: "v8",
      reporter: ["text", "html", "lcov"],
      include: ["src/**/*.{ts,svelte}"],
      exclude: [
        "src/**/*.test.ts",
        "src/test-setup.ts",
        "src/index.ts",
      ],
    },
  },
});
