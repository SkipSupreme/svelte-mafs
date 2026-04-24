# Contributing to Tinker

Tinker is closed to external contributions while we build toward launch. These notes are for the internal build team. If that's you, read on.

The bar is high: Tinker ships type-safe, accessible, well-tested Svelte 5 components for math education tooling. The conventions below exist to keep that bar where it is.

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

## Svelte 5 gotchas you'll hit

### The `.svelte.ts` + `.svelte` resolver trap

Svelte 5 lets you colocate rune-aware helpers in a `*.svelte.ts` file next to a component. If that helper module shares a basename with a component (e.g. `Transform.svelte.ts` + `Transform.svelte`), **TypeScript's "Bundler" resolver silently prefers the `.ts`** — so `import X from "./Transform.svelte"` can resolve to the helper instead of the component, with no error at compile time.

**Rule:** put rune-aware helpers in a file with a *distinct* basename from any component it lives next to. The codebase currently does this as `matrix.ts` (not `Transform.svelte.ts`) and `Line.svelte.ts` (namespace-only, no `Line.svelte` component). If you add a new rune helper, check for a basename collision first.

### `flushSync()` for `$effect` in jsdom tests

Svelte 5's `$effect` callbacks are scheduled, not synchronous. Unit tests that mount a component and assert against its DOM will see stale state unless effects have run. The pattern the test suite standardises on:

```ts
import { flushSync } from "svelte";

test("some reactive thing", () => {
  const { container } = render(Comp, { props: { value: 1 } });
  flushSync();                          // force pending $effects to run
  expect(container.querySelector("..."));
});
```

When rune state updates in response to a user event (e.g. a pointer drag that mutates `$state`), call `flushSync()` between the event dispatch and the assertion. See `src/interaction/MovablePoint.test.ts` for real examples.

### Cross-file coordinate expectations

`ctx.pxToUser` inside `<Mafs>` expects coordinates local to the SVG root — not page-absolute `clientX/clientY`. The `use:drag` action handles that translation internally via `node.ownerSVGElement.getBoundingClientRect()`; if you write a new gesture action that consumes pointer events, do the same. Unit tests don't catch the page-offset issue because jsdom reports all rects as zero.

## Releases / changesets (internal-only)

This repo contains a `.changeset/` directory as an internal changelog tool. **There is no npm publish pipeline for Tinker — it's a product, not a library.** When you land work that changes user-visible behavior, you can still run `pnpm changeset` to write a one-line note; Changesets will aggregate those into `CHANGELOG.md` when we cut internal version tags. But no external release artifact is produced.

The `.github/workflows/release.yml` workflow from the original library scaffold was removed when we pivoted to product framing.

## Known issues / gotchas

- **`pnpm lint` is not yet wired** — there's an `eslint src` script but no eslint config. CI's lint job is feature-gated and skips when no eslint config exists. Adding the config is a captain-level chore.

## Code of Conduct

Be kind. We're here to ship math software for educators. Reports of unacceptable conduct go to joshhunterduvar@gmail.com.

## License

By contributing, you agree your contributions are licensed under the [MIT License](./LICENSE).
