# Tinker

[![CI](https://github.com/SkipSupreme/svelte-mafs/actions/workflows/ci.yml/badge.svg)](https://github.com/SkipSupreme/svelte-mafs/actions/workflows/ci.yml)

> **Learn math, physics, and computer science the way 3Blue1Brown wishes Brilliant worked.** Rigorous, visual, spaced-repetition-backed learning for adults who actually want to understand — not just answer questions correctly.

**Site:** [learntinker.com](https://learntinker.com/) · **Status:** in development, waitlist open · **Target:** 2026

---

## Status

Tinker is a product in active build. The public site is a waitlist placeholder. This repository is not yet accepting external contributions — the interaction surface, content model, and business decisions are still fluid. Watch / star if you want to follow along.

## What's in this repo

Monorepo, pnpm workspaces:

```
packages/svelte-mafs/   — Svelte 5 math-visualization widget engine that
                          powers every interactive diagram, plot, and
                          draggable control in Tinker. Embeds inside Tinker
                          lessons; not distributed standalone.
apps/docs/              — The Astro app deployed to learntinker.com. Today
                          that's the waitlist landing; as lessons land it
                          grows into the product.
```

The widget engine (`svelte-mafs`) is architected as a port of [Mafs](https://mafs.dev) by [Steven Petryk](https://github.com/stevenpetryk) — the same API ergonomics, rebuilt on native Svelte 5 runes. Component surface:

| Category | Components |
|----------|------------|
| Root | `Mafs` |
| Axes | `Coordinates.Cartesian` |
| Plots | `Plot.OfX`, `Plot.OfY`, `Plot.Parametric`, `Plot.Inequality`, `Plot.VectorField` |
| Primitives | `Point`, `Line.Segment`, `Line.ThroughPoints`, `Vector`, `Circle`, `Ellipse`, `Polygon` |
| Text | `Text` (KaTeX) |
| Transforms | `Transform` + `Matrix.{identity, translate, rotate, scale, compose}` |
| Interactive | `MovablePoint` + `snapToGrid`, `snapToLine`, `snapToCurve`, `clampToBox` |
| Gestures | `use:drag`, `use:panZoom` (wheel + pinch) |

## Running it locally

```bash
pnpm install
pnpm -F svelte-mafs build        # widget engine → dist/
pnpm -F docs dev                 # site on http://localhost:4321
pnpm -F svelte-mafs test         # 362 unit tests
pnpm -F svelte-mafs test:e2e     # visual regression + interaction specs
```

The plan that's driving the build is in [`docs/plans/2026-04-23-svelte-mafs.md`](./docs/plans/2026-04-23-svelte-mafs.md). The 8-stream parallel captain/stream workflow is documented in [`docs/plans/2026-04-23-svelte-mafs-streams.md`](./docs/plans/2026-04-23-svelte-mafs-streams.md).

## Credits

The widget engine's API shape, component naming, and architectural conventions follow [**Mafs**](https://mafs.dev) by Steven Petryk, used under the MIT License (see [`NOTICE`](./NOTICE) for full attribution). Where we diverge from Mafs, it's to make the API idiomatic Svelte 5 rather than React-via-compat — the math model and viewport semantics are unchanged.

LaTeX rendering inside `<Text>` is powered by [KaTeX](https://katex.org) (MIT, Khan Academy and contributors).

## License

[MIT](./LICENSE) © Josh Hunter-Duvar and Tinker contributors.
