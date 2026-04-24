# Contributing to svelte-mafs

Thank you for considering a contribution. The bar for this library is high: it ships type-safe, accessible, well-tested Svelte 5 components for math education tooling. The conventions below exist to keep that bar where it is.

## Development setup

```bash
# clone and bootstrap
git clone https://github.com/SkipSupreme/svelte-mafs.git
cd svelte-mafs
pnpm install

# verify baseline
pnpm -F svelte-mafs typecheck
pnpm -F svelte-mafs test
```

Required tooling:

- **Node** 20.11+ (CI matrix: 20 + 22)
- **pnpm** 9 (declared in `packageManager`; corepack will pick it up)
- **Playwright browsers** (one-time): `pnpm -F svelte-mafs exec playwright install --with-deps chromium`

## Working in a git worktree (recommended for parallel work)

Long-running feature branches and parallel agent sessions belong in their own worktree, not in your main checkout. This is how the initial 8-stream parallel build was organized — see [`docs/plans/2026-04-23-svelte-mafs-streams.md`](./docs/plans/2026-04-23-svelte-mafs-streams.md) for the full convention.

```bash
# from your main checkout
git fetch origin
git worktree add ../svelte-mafs-worktrees/feat-my-feature -b feat/my-feature origin/main

cd ../svelte-mafs-worktrees/feat-my-feature
pnpm install   # each worktree gets its own node_modules — no cross-talk
```

When the work is done and merged: `git worktree remove ../svelte-mafs-worktrees/feat-my-feature`.

## Test-Driven Development

**Write the test first.** Then write the implementation. This is non-negotiable for new components and for bug fixes.

```bash
# watch mode while you iterate
pnpm -F svelte-mafs test:watch

# one-shot run with coverage
pnpm -F svelte-mafs exec vitest run --coverage
```

Coverage thresholds:

| Layer | Target |
|-------|--------|
| Pure math (`vec.ts`, `math.ts`, `sampling.ts`) | ≥95% lines |
| Components (`*.svelte`) | ≥80% lines |
| Gestures (`drag.ts`, `pan-zoom.ts`) | ≥80% with at least one e2e fixture covering pointer-capture paths |

If a regression slips in, the bug-fix PR's first commit is **a failing test reproducing the bug**. The second commit is the fix. CI will not let you skip the first.

## Commit conventions

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <imperative summary>

<body, wrapping at ~72 cols, explaining why if non-obvious>
```

Types: `feat` | `fix` | `docs` | `chore` | `refactor` | `test` | `perf` | `build` | `ci`.

Scope is the package or module touched: `svelte-mafs`, `docs`, `ci`, `display`, `gestures`, etc.

Examples:

```
feat(svelte-mafs): add Plot.Inequality with solid fill region
fix(svelte-mafs): drag callback received pixel coords, expected user coords
docs: clarify keyboard shortcuts in MovablePoint a11y section
chore(ci): bump pnpm to 10 in CI matrix
```

**One commit per component / per logical step.** "WIP fixes" and "address review" commits should be squashed before merge — the merged history reads as the project's documentation.

## Adding a new component

1. **Plan it.** Skim the existing component closest to yours (`Point` for static, `MovablePoint` for interactive). Match the prop-naming patterns and JSDoc style.
2. **Write the test first.** Place it next to the source: `display/MyThing.svelte` + `display/MyThing.test.ts`. Render inside a known `<Mafs>` viewBox and assert SVG attributes.
3. **Implement minimally.** The component should be ~50–150 lines. If it grows past 200, consider whether it should be split.
4. **Wire the export.** Add to `src/display/index.ts` (or appropriate sub-index) AND to `src/index.ts`. The append-only convention exists because parallel branches re-export here too — keep your additions on their own line.
5. **Add an example page.** When `apps/docs` is in place, create `apps/docs/src/pages/examples/<your-thing>.mdx` with a live demo + copy-paste snippet.
6. **Add visual-regression baselines.** Once your example page exists, the e2e suite will screenshot it. See [Visual baselines](#visual-baselines) below.
7. **Add a changeset.** `pnpm changeset` → pick `minor` for new components, write a one-line summary. Commit the generated `.changeset/*.md`.

## Visual baselines

Visual-regression tests use Playwright's `toHaveScreenshot()` against PNGs committed in `packages/svelte-mafs/tests/e2e/__screenshots__/`. The diff threshold is **2% pixel ratio** (set in `playwright.config.ts`).

```bash
# run e2e (will boot the docs dev server if not already running)
pnpm -F svelte-mafs test:e2e

# update baselines after an INTENTIONAL visual change
pnpm -F svelte-mafs test:e2e --update-snapshots
```

**Inspect the diffs before regenerating.** When a visual test fails, Playwright drops side-by-side actual/expected/diff PNGs in `playwright-report/`. Open them, confirm the new pixels are correct, *then* run `--update-snapshots` and commit the new baselines in the same PR as the visual change.

If you `--update-snapshots` blindly to make CI green, expect a code review comment.

## Releases

Releases use [Changesets](https://github.com/changesets/changesets). Every PR that changes the lib's surface needs a changeset:

```bash
pnpm changeset
# pick svelte-mafs, pick patch/minor/major, write a 1-line summary
git add .changeset/*.md && git commit -m "chore: changeset for <thing>"
```

When PRs land on `main`, the Release workflow either:

1. Opens (or updates) a single **Version Packages** PR that consumes pending changesets, bumps versions, and regenerates `CHANGELOG.md`.
2. If that Version Packages PR is what just merged, runs `pnpm changeset publish` (which builds the lib and pushes to npm with [provenance attestation](https://docs.npmjs.com/generating-provenance-statements)).

You don't run version/publish yourself.

## Known issues / gotchas

- **`pnpm -F svelte-mafs build` currently fails** in the lib because `svelte-package` defaults to `src/lib/` (SvelteKit convention) but sources sit at `src/`. Fix: change `package.json` script to `svelte-package -i src -o dist` (and add an exclude pattern for `*.test.ts`). Tracked: this gates `pnpm publish` for v0.1.0 and is a one-line fix from the lib package's owner.
- **`pnpm lint` is not yet wired** — there's an `eslint src` script but no eslint config. CI's lint job is feature-gated and skips when no eslint config exists. Adding the config is a captain-level chore.
- **`pnpm -F docs ...` will fail** until Stream 7 lands the Astro docs app. CI's e2e job is feature-gated on `apps/docs/package.json` existing.

## Code of Conduct

Be kind. We're here to ship math software for educators. Reports of unacceptable conduct go to joshhunterduvar@gmail.com.

## License

By contributing, you agree your contributions are licensed under the [MIT License](./LICENSE).
