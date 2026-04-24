# Svelte Mafs — 8-Stream Parallel Build

**Merge captain:** Claude session in `/Users/joshhd/Documents/tinker` (the one that wrote the plan).
**Master plan:** `docs/plans/2026-04-23-svelte-mafs.md` — every stream MUST read this first.
**Repo:** to be initialized by Stream 1 at `/Users/joshhd/Documents/tinker`.

---

## Workflow (read this before opening any stream)

Each stream is a separate Claude Code session working in its own **git worktree** on its own branch. Branches target `main`. Merge captain (this session) merges PRs, resolves conflicts, and keeps `main` clean. Streams never push to `main` directly and never rebase each other's branches.

**Branch naming:** `stream/N-<slug>` (e.g. `stream/1-scaffold`, `stream/4-primitives`).

**Worktree paths:** `/Users/joshhd/Documents/tinker-worktrees/stream-N-<slug>/`. The `superpowers:using-git-worktrees` skill handles creation — every stream prompt below tells Claude to invoke it.

**Commit convention:** Conventional Commits. Prefix with stream number so captain can grep: `feat(svelte-mafs,s4): add Point component`.

**Definition of done per stream:**
1. All tasks for stream complete.
2. `pnpm -F svelte-mafs typecheck` green.
3. `pnpm -F svelte-mafs test` green.
4. Coverage unchanged-or-up on affected files.
5. Branch pushed, PR opened to `main` with a checklist.
6. Captain review + merge.

**File-ownership rule (hard):** Each stream ONLY touches paths in its own allotment (see Appendix A). If you need to change a file owned by another stream, post in the PR description and wait for captain. The one exception is `packages/svelte-mafs/src/index.ts` — every stream adds its re-exports there at the end, captain merges in sequence.

**Dependency gates (hard):**
- Streams 2–8 wait for Stream 1 to tag `v0.0.0-scaffold` (scaffold + math core landed on `main`).
- Streams 4, 5 wait for Stream 2 (`Mafs` context published).
- Stream 6 waits for Streams 2 and 3 (needs context + drag action).
- Stream 7 (docs) can start after Stream 1 but most pages block on components landing.
- Stream 8 (CI/release) mostly parallel to Stream 1, but release job waits for everything.

`★ Insight ─────────────────────────────────────`
- **Why worktrees, not just branches**: worktrees let 8 agents each have their own working directory with their own `node_modules`, dev server, and test watcher without stepping on each other. Switching branches in a single checkout would trash parallel dev servers and pnpm lockfile races.
- **Why strict file ownership over trusting merge**: Svelte component files are small enough that two agents editing the same file almost always ends in a semantic conflict git can't detect (both add an import to `index.ts`, both add a test to `math.test.ts`). The index re-export file is the one unavoidable shared touch-point — we serialize it through captain.
`─────────────────────────────────────────────────`

---

## Captain Protocol (this session)

Before streams start:
1. `cd /Users/joshhd/Documents/tinker && git init -b main && git commit --allow-empty -m "chore: init repo"` (so streams have something to branch from).
2. Drop the master plan + this prompts file into `main`, commit, done.
3. Announce readiness; streams open.

While streams run:
- Monitor PRs in order 1 → 8.
- Stream 1 merge first (gates others). Tag `v0.0.0-scaffold` on `main` after merge.
- Stream 2, 3, 7, 8 can merge in parallel (no file overlap with each other).
- Stream 4, 5 merge after 2.
- Stream 6 merges after 2 + 3.
- Resolve any `src/index.ts` conflicts manually (trivial: append each stream's exports).
- Run full test suite on `main` after each merge.

After all merges:
- Stream 8 cuts `v0.1.0` via Changesets.
- Captain writes release notes, announces.

---

# The 8 Stream Prompts

Each section below is a **copy-pasteable prompt** for a fresh Claude Code session. Do NOT edit them before pasting — they're self-contained. After pasting, the new session should use `superpowers:using-git-worktrees` and `superpowers:executing-plans` without further nudging.

---

## Stream 1 — Scaffold + Math Core

```
You are Stream 1 of an 8-stream parallel build of `svelte-mafs`, a Svelte 5 port of the React library Mafs (https://mafs.dev). You gate every other stream — land fast.

READ FIRST:
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs.md (master plan, full context)
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs-streams.md (this workflow, Appendix A = file ownership)

SETUP:
Invoke the superpowers:using-git-worktrees skill to create worktree at
  /Users/joshhd/Documents/tinker-worktrees/stream-1-scaffold
on branch `stream/1-scaffold` based on `main`.

SCOPE (you own these paths exclusively):
- /  (root package.json, pnpm-workspace.yaml, .gitignore, tsconfig.base.json)
- packages/svelte-mafs/package.json
- packages/svelte-mafs/tsconfig.json
- packages/svelte-mafs/vite.config.ts
- packages/svelte-mafs/vitest.config.ts
- packages/svelte-mafs/src/index.ts (initial skeleton — other streams append)
- packages/svelte-mafs/src/test-setup.ts
- packages/svelte-mafs/src/vec.ts, vec.test.ts
- packages/svelte-mafs/src/math.ts, math.test.ts
- packages/svelte-mafs/src/sampling.ts, sampling.test.ts

DO NOT TOUCH anything else. Specifically: no context/, no view/, no display/, no gestures/, no interaction/, no apps/.

TASKS (execute in order, TDD, one commit per task):
1. Phase 0 Task 0.1 — repo init, pnpm workspace, tsconfig.base.json
2. Phase 0 Task 0.2 — scaffold packages/svelte-mafs with all config files
3. Phase 0 Task 0.3 — scaffold apps/docs with `pnpm create svelte@latest docs` (skeleton, TypeScript). Add `"svelte-mafs": "workspace:*"` to its deps. Commit. Stream 7 takes ownership from here.
4. Phase 1 Task 1.1 — vec.ts (add, sub, scale, dot, mag, normalize, rotate, lerp) — tests first
5. Phase 1 Task 1.2 — math.ts (clamp, round, mapRange, nearestPowerOfTen, snapAngleToDegrees, inferLabels) — tests first, one commit per function
6. Phase 1 Task 1.3 — sampling.ts (adaptive midpoint subdivision) — tests with pathological fns (sin(1/x), tan, step)

DONE CRITERIA:
- `pnpm install` succeeds in root
- `pnpm -F svelte-mafs typecheck` green
- `pnpm -F svelte-mafs test` green with ≥95% coverage on vec/math/sampling
- Branch pushed: `git push -u origin stream/1-scaffold`
- PR opened to `main` with title "Stream 1: scaffold + math core"
- PR description lists exported symbols + coverage %

COMMIT PREFIX: `feat(svelte-mafs,s1):` or `chore(svelte-mafs,s1):`

Invoke superpowers:executing-plans. Begin.
```

---

## Stream 2 — Context + Root View

```
You are Stream 2 of an 8-stream parallel build of `svelte-mafs`. Stream 1 must land before you start.

READ FIRST:
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs.md (master plan)
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs-streams.md (workflow)

GATE: Before anything else, `git fetch origin && git log origin/main --oneline | grep v0.0.0-scaffold || echo "WAIT — Stream 1 not merged"`. If not merged, stop and tell the user.

SETUP:
Invoke superpowers:using-git-worktrees to create worktree at
  /Users/joshhd/Documents/tinker-worktrees/stream-2-context
on branch `stream/2-context` based on latest `main`.

SCOPE (exclusive ownership):
- packages/svelte-mafs/src/context/coordinate-context.ts + .test.ts
- packages/svelte-mafs/src/context/pane-context.ts + .test.ts
- packages/svelte-mafs/src/view/Mafs.svelte + Mafs.test.ts
- packages/svelte-mafs/src/view/Viewport.svelte
- packages/svelte-mafs/src/view/index.ts (re-exports for view/)

NO OTHER PATHS. Especially: no gestures/, no display/.

TASKS (TDD, one commit per task):
1. coordinate-context.ts — CoordContext type, setCoordContext/getCoordContext helpers. Tests for round-trip userToPx/pxToUser with varied viewBoxes (square, wide, tall, negative ranges).
2. pane-context.ts — nested pane overrides (stub for now, full use is later). Minimal impl + types.
3. Mafs.svelte — props: {width, height, viewBox, preserveAspectRatio, pan, zoom}. Uses $state for viewBox, $derived for userToPx/pxToUser. Renders <svg viewBox> with inner <g transform="scale(1,-1)">. Publishes context.
   - Component test: mount with known dims, use a test-only child that reads context, assert userToPx([0,0]) returns correct center pixel.
4. Viewport.svelte — placeholder for nested viewports (used by future pane features). Stub + types.

INTEGRATION:
- Add re-exports to packages/svelte-mafs/src/index.ts:
  `export { Mafs } from "./view/Mafs.svelte";`
  `export type { CoordContext } from "./context/coordinate-context.js";`
- If Stream 1's index.ts has changed since your branch point, rebase on main to get it.

DONE CRITERIA:
- `pnpm -F svelte-mafs typecheck` green
- `pnpm -F svelte-mafs test` green, Mafs mount test passes
- PR opened: "Stream 2: coordinate context + Mafs root"

COMMIT PREFIX: `feat(svelte-mafs,s2):`

Invoke superpowers:executing-plans. Begin.
```

---

## Stream 3 — Gestures (pan-zoom + drag)

```
You are Stream 3 of an 8-stream parallel build of `svelte-mafs`. Stream 1 must land before you start. You can run fully in parallel with Stream 2 (different files).

READ FIRST:
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs.md
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs-streams.md

GATE: `git fetch origin && git log origin/main --oneline | grep v0.0.0-scaffold` must succeed.

SETUP:
Invoke superpowers:using-git-worktrees for
  /Users/joshhd/Documents/tinker-worktrees/stream-3-gestures
on branch `stream/3-gestures` based on latest `main`.

SCOPE:
- packages/svelte-mafs/src/gestures/drag.ts + drag.test.ts
- packages/svelte-mafs/src/gestures/pan-zoom.ts + pan-zoom.test.ts
- packages/svelte-mafs/src/gestures/index.ts

NO OTHER PATHS.

DESIGN CONSTRAINTS:
- Svelte 5 actions (use: syntax). Signature: `(node, options) => { update?, destroy? }`.
- Pointer Events only (no separate mouse/touch). Use `setPointerCapture` on pointerdown.
- Both actions accept a `pxToUser` function as option — you do NOT import from context/. Consumer wires this.
  Rationale: keeps gestures zero-dep on context module, makes them trivially unit-testable.
- Drag: emits onDragStart(userPos), onDrag(userPos), onDragEnd().
- PanZoom: emits onPan(deltaUser), onZoom(factor, centerPx). Wheel zoom with ctrl/meta modifier; pinch via pointerup multi-touch (defer if hard).

TASKS:
1. drag.ts — action + tests. Use happy-dom or jsdom with manual PointerEvent dispatch; if flaky, write Playwright fixture spec instead (document the decision in commit).
2. pan-zoom.ts — action + tests. Test wheel zoom, drag-pan, pinch-zoom (or skip pinch with a TODO comment if jsdom can't).
3. gestures/index.ts — named re-exports.

INTEGRATION:
- Add to src/index.ts: `export { drag, panZoom } from "./gestures/index.js";`

DONE CRITERIA:
- Types green, unit tests green.
- If you fell back to Playwright for any action, include the fixture page under `tests/e2e/fixtures/` and document it.
- PR opened: "Stream 3: drag + panZoom actions"

COMMIT PREFIX: `feat(svelte-mafs,s3):`

Invoke superpowers:executing-plans. Begin.
```

---

## Stream 4 — Static Display Primitives

```
You are Stream 4 of an 8-stream parallel build of `svelte-mafs`. Streams 1 AND 2 must land before you start.

READ FIRST:
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs.md (see Phase 3 esp.)
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs-streams.md

GATE: Verify Stream 2's Mafs component is on main: `git fetch origin && test -f packages/svelte-mafs/src/view/Mafs.svelte && echo OK`.

SETUP:
Invoke superpowers:using-git-worktrees for
  /Users/joshhd/Documents/tinker-worktrees/stream-4-primitives
on branch `stream/4-primitives` based on latest `main`.

SCOPE (exclusive):
- packages/svelte-mafs/src/display/Point.svelte + .test.ts
- packages/svelte-mafs/src/display/Line.svelte.ts (namespace module)
- packages/svelte-mafs/src/display/line-segment.svelte + .test.ts
- packages/svelte-mafs/src/display/line-through-points.svelte + .test.ts
- packages/svelte-mafs/src/display/line-parallel.svelte + .test.ts (optional v1)
- packages/svelte-mafs/src/display/line-perpendicular.svelte + .test.ts (optional v1)
- packages/svelte-mafs/src/display/Circle.svelte + .test.ts
- packages/svelte-mafs/src/display/Ellipse.svelte + .test.ts
- packages/svelte-mafs/src/display/Polygon.svelte + .test.ts
- packages/svelte-mafs/src/display/Vector.svelte + .test.ts

STAY OUT OF: Coordinates.svelte, Plot*, Text.svelte, Transform.svelte — those belong to Streams 5 and 6.

TASKS (one commit per component, TDD):
1. Point — <circle> at transformed coords. Props: x, y, color?, opacity?, svg?.
2. Line namespace module (display/Line.svelte.ts): exports Segment, ThroughPoints, Parallel, Perpendicular as component re-exports.
3. line-segment — two endpoints. Props: point1, point2, color?, opacity?, weight?.
4. line-through-points — infinite line through two points, clipped to viewBox. Uses math.ts.
5. Circle — center + radius (in user space, so radius=1 means 1 unit).
6. Ellipse — center + [rx, ry] + rotation.
7. Polygon — points array, closed path.
8. Vector — line + SVG marker arrowhead. Use a single shared <defs><marker> per Mafs root (inject via context? — simplest: inline per Vector, optimize later).

DESIGN NOTES:
- Every component starts with `const ctx = getCoordContext();` and `let px = $derived(ctx.userToPx([x, y]));`.
- All colors default to CSS custom properties (`var(--mafs-fg)`, `var(--mafs-blue)`) which Stream 6 defines. Until then, hardcode fallbacks (`#333`).
- Test pattern: render inside <Mafs width={400} height={300} viewBox={{x:[-5,5],y:[-5,5]}}> and assert SVG attributes on the rendered element.

INTEGRATION:
- Create display/index.ts with re-exports.
- Append to src/index.ts: `export { Point, Line, Circle, Ellipse, Polygon, Vector } from "./display/index.js";`

DONE CRITERIA:
- Types green. All component tests green. Coverage ≥80% on display/.
- PR opened: "Stream 4: static display primitives (Point, Line, Circle, Ellipse, Polygon, Vector)"

COMMIT PREFIX: `feat(svelte-mafs,s4):`

Invoke superpowers:executing-plans. Begin.
```

---

## Stream 5 — Coordinates + Plot namespace

```
You are Stream 5 of an 8-stream parallel build of `svelte-mafs`. Streams 1 AND 2 must land before you start. Stream 4 running in parallel is fine (non-overlapping files).

READ FIRST:
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs.md (Phases 3.9 + 4)
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs-streams.md

GATE: Verify Stream 1's sampling.ts and Stream 2's Mafs.svelte are on main.

SETUP:
Invoke superpowers:using-git-worktrees for
  /Users/joshhd/Documents/tinker-worktrees/stream-5-plot
on branch `stream/5-plot` based on latest `main`.

SCOPE (exclusive):
- packages/svelte-mafs/src/display/Coordinates.svelte.ts (namespace)
- packages/svelte-mafs/src/display/coordinates-cartesian.svelte + .test.ts
- packages/svelte-mafs/src/display/Plot.svelte.ts (namespace)
- packages/svelte-mafs/src/display/plot-of-x.svelte + .test.ts
- packages/svelte-mafs/src/display/plot-of-y.svelte + .test.ts
- packages/svelte-mafs/src/display/plot-parametric.svelte + .test.ts
- packages/svelte-mafs/src/display/plot-inequality.svelte + .test.ts
- packages/svelte-mafs/src/display/plot-vector-field.svelte + .test.ts

STAY OUT OF: all files Stream 4 owns (Point, Line, Circle, etc.).

TASKS:
1. Coordinates.Cartesian — axes + grid + labels. Uses math.ts inferLabels to pick tick positions. Props: xAxis?, yAxis?, grid?, subdivisions?. For labels use <text> with manual sign-flip for now (Text.svelte with KaTeX is Stream 6).
2. Plot.OfX — samples `f: (x:number)=>number` via sampling.sample over viewport x-range, produces SVG path. Test asserts <path d> has expected segment count for a straight line and a sinusoid.
3. Plot.OfY — mirror of OfX with axes swapped. Factor shared code into a helper.
4. Plot.Parametric — (t) => [x,y], samples on t in [tMin, tMax].
5. Plot.Inequality — fill region between two curves / above-below a curve. Harder, defer to end; ship minimal "y > f(x)" version with solid fill.
6. Plot.VectorField — grid of <Vector>s computed from f: (x,y) => [dx, dy]. Auto-normalize lengths to avoid clipping.

DESIGN NOTES:
- All Plot components accept `weight`, `color`, `opacity`, `style` (solid | dashed | dotted).
- Path strings: prefer `M x y L x y L x y` format (not `H`/`V`) — simpler, tests cleaner.
- Namespace export pattern (Svelte 5): `display/Plot.svelte.ts` exports `{ OfX, OfY, Parametric, Inequality, VectorField }` as default object.

INTEGRATION:
- Append to src/index.ts: `export { Coordinates, Plot } from "./display/index.js";`
- Coordinate with Stream 4 on display/index.ts additions (captain merges).

DONE CRITERIA:
- Types green. Tests green. Coverage ≥80% on plot files.
- Visual check: add a temporary examples page under apps/docs showing Coordinates + Plot.OfX of sin(x), screenshot in PR.
- PR opened: "Stream 5: Coordinates + Plot namespace"

COMMIT PREFIX: `feat(svelte-mafs,s5):`

Invoke superpowers:executing-plans. Begin.
```

---

## Stream 6 — Interaction + Theme + Text

```
You are Stream 6 of an 8-stream parallel build of `svelte-mafs`. Streams 1, 2, AND 3 must land before you start.

READ FIRST:
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs.md (Phases 5 + 6)
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs-streams.md

GATE: Verify drag action is on main: `git fetch origin && test -f packages/svelte-mafs/src/gestures/drag.ts && echo OK`.

SETUP:
Invoke superpowers:using-git-worktrees for
  /Users/joshhd/Documents/tinker-worktrees/stream-6-interaction
on branch `stream/6-interaction` based on latest `main`.

SCOPE (exclusive):
- packages/svelte-mafs/src/interaction/MovablePoint.svelte + .test.ts
- packages/svelte-mafs/src/interaction/constraints.ts + .test.ts (snapToGrid, snapToLine, snapToCurve, clamp)
- packages/svelte-mafs/src/interaction/index.ts
- packages/svelte-mafs/src/display/Text.svelte + .test.ts
- packages/svelte-mafs/src/display/Transform.svelte + .test.ts (+ Transform.svelte.ts namespace module with translate/rotate/scale helpers)
- packages/svelte-mafs/src/theme.ts
- packages/svelte-mafs/src/styles/core.css
- packages/svelte-mafs/src/styles/font.css

STAY OUT OF: other display files (Point/Line/etc. = Stream 4; Plot/Coordinates = Stream 5).

TASKS:
1. theme.ts + core.css — CSS custom properties (--mafs-fg, --mafs-bg, --mafs-line-color, --mafs-grid-color, --mafs-blue, --mafs-red, --mafs-green, --mafs-yellow, --mafs-orange, --mafs-purple, --mafs-pink). Light + dark variants under `:where([data-theme="dark"]) .mafs-root`. Write a CSS snapshot test.
2. font.css — bundled numeric font (start with system-ui mono fallback; note follow-up to ship a real bundled font).
3. Text.svelte — KaTeX renderToString, inject in <foreignObject>. Counter-flip y with inner `<div style="transform:scale(1,-1)">`. Props: x, y, latex, size?, color?. Test: assert foreignObject contains `.katex` class after mount.
4. Transform.svelte (+ namespace) — <g transform="matrix(...)">. Namespace exports translate/rotate/scale components that compose onto affine matrix. Test: assert matrix() string is correct for rotate(π/2).
5. constraints.ts — pure functions. Tests drive each.
6. MovablePoint.svelte — props: {x, y} bindable via $bindable(), color?, constrain?. Uses use:drag action from Stream 3. A11y: role="slider", aria-valuenow-x, aria-valuenow-y, tabindex=0, arrow keys (shift for 10×), focus ring. Test: drag fixture via Playwright + keyboard test via jsdom.

INTEGRATION:
- Append to src/index.ts: `export { MovablePoint, snapToGrid, snapToLine, snapToCurve, clamp } from "./interaction/index.js"; export { Text, Transform } from "./display/index.js";`
- Export CSS: packages/svelte-mafs/package.json "exports" maps already point at dist/styles/. Ensure Vite build copies src/styles/ to dist/styles/.

DONE CRITERIA:
- Types green. Tests green including keyboard a11y test.
- PR opened: "Stream 6: MovablePoint + Text + Transform + theme"

COMMIT PREFIX: `feat(svelte-mafs,s6):`

Invoke superpowers:executing-plans. Begin.
```

---

## Stream 7 — Docs Site (Astro 6 + @astrojs/svelte on Cloudflare Pages)

```
You are Stream 7 of an 8-stream parallel build of `svelte-mafs`. Stream 1 must land before you start. Your work has two waves: (A) site chassis can ship immediately after Stream 1; (B) example pages backfill as Streams 4–6 merge.

STACK CHOICE: Astro 6 with @astrojs/svelte integration. Not SvelteKit. Reason: docs sites are content-first (MDX, syntax highlighting, static output) and Astro's island architecture means each interactive Mafs demo is its own hydrated bundle — non-interactive pages ship zero JS. Use `client:visible` or `client:idle` on Svelte components.

READ FIRST:
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs.md (Phase 8)
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs-streams.md

GATE: Stream 1 merged and v0.0.0-scaffold tagged. Note: Stream 1's initial `apps/docs` scaffold was torn down by captain before Stream 7 started — you create the Astro project fresh.

SETUP:
Invoke superpowers:using-git-worktrees for
  /Users/joshhd/Documents/tinker-worktrees/stream-7-docs
on branch `stream/7-docs` based on latest `main`.

SCOPE (exclusive):
- apps/docs/** (everything under — create from scratch)
- docs/examples/*.md (content source — raw text examples for the gallery)

STAY OUT OF: packages/svelte-mafs/**. If you need a feature the lib doesn't expose, open an issue and tag the owning stream. Never edit lib source to unblock docs.

TASKS (Wave A — independent):
1. Scaffold Astro 6 + Svelte:
   ```
   cd apps && pnpm create astro@latest docs -- --template minimal --typescript strict --no-install --no-git
   cd docs && pnpm astro add svelte && pnpm astro add cloudflare && pnpm astro add mdx
   ```
   Add `"svelte-mafs": "workspace:*"` to `apps/docs/package.json`. Pin Svelte to ^5 matching the lib's peerDep.
2. Layout (`src/layouts/Base.astro`): top nav (Home / Getting Started / API / Examples / GitHub), dark-mode toggle persisted to localStorage, responsive. Layout is pure Astro; dark-mode toggle is a tiny `<script>` block, no hydration needed.
3. Home page (`src/pages/index.astro`): hero headline + 2–3 Mafs demos rendered as Svelte islands with `client:visible`.
4. Getting Started page (MDX): install (`pnpm add svelte-mafs`), first `<Mafs>` snippet, theming, a11y notes. Shiki via `@astrojs/mdx` for syntax highlighting.
5. API reference pages (MDX, one per exported component): auto-generated via a small docgen script that reads `packages/svelte-mafs/dist/*.d.ts` + JSDoc blocks from `.svelte` files. Write the script under `apps/docs/scripts/docgen.ts`, run in `prebuild`.
6. Theme playground (`src/pages/playground.astro` with a Svelte island): CSS custom property editor with live preview.
7. Deploy config: `@astrojs/cloudflare` adapter, output `apps/docs/dist/`. Document Cloudflare Pages build command `pnpm -F docs build` in README.

TASKS (Wave B — backfill as components land):
8. As each primitive/plot/interactive component merges to main, add an examples page at `apps/docs/src/pages/examples/<slug>.mdx` with a live interactive Svelte island (`client:visible`) + copy-paste code block.
9. Port Mafs homepage examples (mafs.dev): coordinates grid, parametric curves, vector field, draggable points exercise, bezier curve editor. One MDX file per.

DESIGN NOTES:
- Shiki is built into @astrojs/mdx — use it for syntax highlighting. No runtime highlighter.
- "Copy code" button on every snippet (tiny inline script, no framework).
- Dark mode default matches OS pref; toggle persists to localStorage.
- Every interactive demo: `client:visible` to defer hydration until scrolled into view.
- Ensure every example is keyboard-accessible (draft `apps/docs/A11Y.md`).

DONE CRITERIA (Wave A):
- Site runs locally: `pnpm -F docs dev`
- Builds clean: `pnpm -F docs build` → `apps/docs/dist/`
- Deploys to a preview Cloudflare Pages project (user may need to provide Pages project name/token — ask before attempting deploy)
- PR opened: "Stream 7: Astro 6 docs site chassis"

DONE CRITERIA (Wave B, per-component):
- Incremental PRs like "Stream 7: example page for Plot.OfX" — merged after the underlying component is on main.

COMMIT PREFIX: `feat(docs,s7):`

Invoke superpowers:executing-plans. Begin with Wave A.
```

---

## Stream 8 — E2E + CI + Release

```
You are Stream 8 of an 8-stream parallel build of `svelte-mafs`. Stream 1 must land before you start. Most of your work runs in parallel with everyone else; the release job waits for everything.

READ FIRST:
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs.md (Phases 7 + 9)
- /Users/joshhd/Documents/tinker/docs/plans/2026-04-23-svelte-mafs-streams.md

GATE: v0.0.0-scaffold tag on main.

SETUP:
Invoke superpowers:using-git-worktrees for
  /Users/joshhd/Documents/tinker-worktrees/stream-8-release
on branch `stream/8-release` based on latest `main`.

SCOPE (exclusive):
- .github/workflows/ci.yml
- .github/workflows/release.yml
- .changeset/** (Changesets config)
- packages/svelte-mafs/playwright.config.ts
- packages/svelte-mafs/tests/e2e/** (EXCEPT fixtures added by other streams — coordinate)
- README.md
- CONTRIBUTING.md
- LICENSE
- NOTICE
- Root scripts for release (scripts/*.ts)

DO NOT TOUCH src/ of the lib or apps/docs/ source (add e2e specs that *import* from docs dev server but don't edit its source).

TASKS:
1. Playwright config — chromium + iphone-14 projects, webServer pinned to Astro docs dev server (`pnpm -F docs dev`, default port 4321). Baseline threshold 2%.
2. CI workflow — matrix Node 20 + 22. Jobs: lint, typecheck, test:unit, test:e2e (headless chromium), build. Cache pnpm store. Post coverage to GitHub Pages or codecov.
3. Changesets — `pnpm changeset init`, configure baseBranch: main, access: public.
4. Release workflow — on push to main after PR merge, `changesets/action@v1` creates a "Version Packages" PR or publishes to npm.
5. README — hero example (paste-able), install, features, credits (MIT Mafs prominently), links to docs.
6. CONTRIBUTING — TDD expectation, commit conventions, how to add a component, how to run/update visual baselines, how to work in a worktree.
7. LICENSE (MIT) + NOTICE (Mafs attribution + KaTeX attribution).
8. Visual regression baseline workflow — once apps/docs has example pages (from Stream 7 Wave B), add Playwright specs that screenshot each page and compare against committed PNGs in tests/e2e/__screenshots__/. Add CI step that fails on diff >2%, comments on PR with visual diff.
9. Publish dry-run script — scripts/release-check.ts verifies dist/ contents, types ship, no secret leaks.
10. **Pinch-zoom e2e (handoff from Stream 3)**: Stream 3 stayed in jsdom for drag + pan + wheel-zoom unit tests, but deferred multi-pointer pinch-zoom to your harness (see TODO comment in `packages/svelte-mafs/src/gestures/pan-zoom.ts` ~line 25). Add a Playwright fixture that dispatches two coordinated pointer sequences to verify pinch triggers `onZoom` with the correct factor and center.

DONE CRITERIA:
- CI passes on `main` after every stream merge
- README renders correctly on GitHub
- `pnpm -F svelte-mafs publish --dry-run` produces a valid package
- PR opened in two parts: "Stream 8a: CI + release infra" (early) and "Stream 8b: visual regression + v0.1.0 cut" (after all streams land)

COMMIT PREFIX: `chore(ci,s8):` or `docs(s8):`

Invoke superpowers:executing-plans. Begin.
```

---

# Appendix A — File Ownership Matrix

Canonical list for conflict detection. If your stream wants to touch a path not in its row, STOP and post to captain.

| Path | Owner | Read-by |
|------|-------|---------|
| `/package.json`, `pnpm-workspace.yaml`, `.gitignore`, `tsconfig.base.json` | S1 | all |
| `packages/svelte-mafs/package.json` | S1 (add deps via PR to S1 branch if open, else captain merges) | all |
| `packages/svelte-mafs/tsconfig.json`, `vite.config.ts` | S1 | all |
| `packages/svelte-mafs/vitest.config.ts` | **first-mover, captain merges** (S2 adds `svelteTesting()` plugin; later streams may extend) | all component streams |
| `packages/svelte-mafs/src/index.ts` | **shared — append-only, captain serializes** | all |
| `packages/svelte-mafs/src/test-setup.ts` | S1 | all |
| `packages/svelte-mafs/src/{vec,math,sampling}.{ts,test.ts}` | S1 | S5 reads sampling |
| `packages/svelte-mafs/src/context/**` | S2 | S4, S5, S6 |
| `packages/svelte-mafs/src/view/**` | S2 | all display streams |
| `packages/svelte-mafs/src/gestures/**` | S3 | S6 |
| `packages/svelte-mafs/src/display/{Point,Line*,Circle,Ellipse,Polygon,Vector}.svelte` | S4 | — |
| `packages/svelte-mafs/src/display/{Coordinates,Plot}*.svelte` | S5 | — |
| `packages/svelte-mafs/src/display/{Text,Transform}*.svelte` | S6 | — |
| `packages/svelte-mafs/src/display/index.ts` | **shared — append-only, captain serializes** | — |
| `packages/svelte-mafs/src/interaction/**` | S6 | — |
| `packages/svelte-mafs/src/theme.ts`, `src/styles/**` | S6 | — |
| `apps/docs/**` | S1 scaffolds, S7 owns thereafter | — |
| `packages/svelte-mafs/tests/e2e/**` | S8 (streams may add fixture pages under `fixtures/` with coordination) | — |
| `.github/**`, `.changeset/**`, `README.md`, `CONTRIBUTING.md`, `LICENSE`, `NOTICE` | S8 | — |

---

# Appendix B — Merge Order Playbook (for captain)

```
1. S1 PR      → review, merge, tag v0.0.0-scaffold, push tag
   Unblocks: all
2. S2 PR and S3 PR and S7 Wave A PR and S8a PR
   → review in parallel, merge independently (no file overlap)
   Unblocks: S4, S5, S6
3. S4 PR and S5 PR
   → may conflict on display/index.ts (append-only, resolve trivially)
   → may conflict on src/index.ts (same)
   → merge one, rebase the other
4. S6 PR
   → may conflict on display/index.ts + src/index.ts + theme consumption by S4/S5 (they hardcoded colors, switch to var())
   → captain either opens a mini-PR to swap hardcoded colors, or does it in the merge commit
5. S7 Wave B PRs (example pages)
   → trickle in as components merge
6. S8b PR — visual regression baselines + v0.1.0 release
   → cut the tag, publish
```

Conflict resolution cheatsheet for `src/index.ts` merges:
- Conflict marker on re-exports is always `<<<<<<< HEAD \n export { A } from "./a.js" \n ======= \n export { B } from "./b.js" \n >>>>>>>`. Resolution: keep both. Never discard.

---

# Appendix C — Pre-flight checklist (captain runs once before dispatching streams)

```bash
cd /Users/joshhd/Documents/tinker

# 1. init repo, commit plan + streams file
git init -b main
git add docs/plans/
git commit -m "docs: master plan + stream briefs for svelte-mafs"

# 2. set up remote (Stream 1 will push to this)
# — user: create empty GitHub repo `svelte-mafs` (or whatever you pick) and set remote
# git remote add origin git@github.com:<you>/svelte-mafs.git
# git push -u origin main

# 3. confirm gh CLI works (for PR review)
gh auth status

# 4. open 8 terminals / Claude Code sessions, paste each prompt from this doc
```
