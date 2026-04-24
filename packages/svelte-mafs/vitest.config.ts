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
