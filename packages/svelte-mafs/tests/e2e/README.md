# E2E + visual regression tests

These specs run against the **Astro 6 docs site** (`apps/docs`) on `http://localhost:4321`. Playwright boots that server automatically via `webServer` in `playwright.config.ts`.

## Layout

```
tests/e2e/
├── README.md             — you are here
├── fixtures/             — small reusable Astro/Svelte fixture pages added by component streams (S3, S6) when jsdom can't faithfully test a gesture or a11y flow
├── specs/                — visual-regression + drag/keyboard specs added by Stream 8 once docs example pages exist
└── __screenshots__/      — committed baseline PNGs (one per spec × project combo). Diff threshold: 2% pixel ratio (see playwright.config.ts).
```

## Running locally

```bash
# one-time browser install
pnpm exec playwright install --with-deps chromium webkit

# run all specs (will boot the docs dev server if not already running)
pnpm -F svelte-mafs test:e2e

# update screenshot baselines after an intentional visual change
pnpm -F svelte-mafs test:e2e --update-snapshots
```

## Updating baselines (the right way)

1. Make the visual change in the lib.
2. Run e2e — failing diffs are uploaded to `playwright-report/` as side-by-side PNGs.
3. Eyeball the diffs. **If the new pixels are correct**, regenerate baselines:
   ```bash
   pnpm -F svelte-mafs test:e2e --update-snapshots
   ```
4. Commit the updated `__screenshots__/*.png` in the same PR as the visual change.

Never `--update-snapshots` "just to make CI green" without inspecting the diffs first — that defeats the entire point of visual regression.

## Adding a fixture (component streams)

Stream 3 (drag) and Stream 6 (MovablePoint a11y) may need fixture pages that the docs site itself doesn't host. Put them under `tests/e2e/fixtures/<slug>/` as either:

- a tiny standalone HTML file (no framework) loaded via `page.setContent(...)` in the spec, or
- a route added to `apps/docs/src/pages/_test/<slug>.astro` (these live under `_test/` so they don't ship publicly, gated by `import.meta.env.DEV`).

Coordinate with Stream 8 before adding anything — file ownership matrix in `docs/plans/2026-04-23-svelte-mafs-streams.md` Appendix A.
