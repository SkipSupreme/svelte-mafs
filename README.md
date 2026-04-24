# svelte-mafs

[![CI](https://github.com/SkipSupreme/svelte-mafs/actions/workflows/ci.yml/badge.svg)](https://github.com/SkipSupreme/svelte-mafs/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/svelte-mafs.svg)](https://www.npmjs.com/package/svelte-mafs)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![bundle size](https://img.shields.io/bundlephobia/minzip/svelte-mafs?label=gzip)](https://bundlephobia.com/package/svelte-mafs)

> **Svelte 5 components for interactive math visualization.** Plot functions, draw geometry, drag points around, render LaTeX labels. A faithful port of [Mafs](https://mafs.dev) by [Steven Petryk](https://github.com/stevenpetryk) — with the same API ergonomics, native Svelte 5 runes, and zero React.

> ⚠️ **Status: pre-1.0, in active development.** v0.1.0 lands when the full primitive + plot + interaction surface is ready. Pin exact versions until then.

## What it looks like

```svelte
<script lang="ts">
  import { Mafs, Coordinates, Plot, MovablePoint } from "svelte-mafs";
  import "svelte-mafs/core.css";

  let a = $state(1);
  let b = $state(0);
</script>

<Mafs viewBox={{ x: [-5, 5], y: [-5, 5] }}>
  <Coordinates.Cartesian />
  <Plot.OfX y={(x) => a * x + b} color="var(--mafs-blue)" />
  <MovablePoint bind:x={a} bind:y={b} />
</Mafs>
```

That's it. Drag the point and the line follows.

## Install

```bash
pnpm add svelte-mafs
# peers (you almost certainly already have svelte)
pnpm add -D svelte
```

Import the stylesheet once at your app root:

```ts
import "svelte-mafs/core.css";
```

For LaTeX labels via `<Text>`, also import the KaTeX stylesheet from the `katex` package (already a transitive dep).

## Why use it

- **Svelte 5 native** — built around `$state`, `$derived`, `$effect`, `$bindable()`. No store boilerplate, no React-via-compat.
- **Tree-shakable, ESM-only** — `sideEffects: false` (except CSS). Importing `<Mafs>` + `<Plot.OfX>` lands under 20 KB gzipped before KaTeX.
- **Math-first coordinates** — write `<Point x={0} y={3}>` and y-up means up. The root flip is invisible to you.
- **Adaptive sampling** — function plots subdivide where curvature is high, so `sin(1/x)` near zero doesn't alias.
- **Real a11y** — `<MovablePoint>` is keyboard-draggable (arrow keys move 1 unit, shift+arrow moves 10), focusable, and announces position via `aria-live`.
- **Pinned theming** — CSS custom properties (`--mafs-fg`, `--mafs-blue`, `--mafs-grid-color`, …). Override at any ancestor; light/dark via `data-theme`.

## What's in the box (v0.1.0 target)

| Category | Components |
|----------|------------|
| Root | `Mafs` |
| Axes | `Coordinates.Cartesian` |
| Plots | `Plot.OfX`, `Plot.OfY`, `Plot.Parametric`, `Plot.Inequality`, `Plot.VectorField` |
| Primitives | `Point`, `Line.Segment`, `Line.ThroughPoints`, `Vector`, `Circle`, `Ellipse`, `Polygon` |
| Text | `Text` (KaTeX) |
| Transforms | `Transform`, `Transform.translate`, `Transform.rotate`, `Transform.scale` |
| Interactive | `MovablePoint` + `snapToGrid`, `snapToLine`, `snapToCurve`, `clamp` constraints |
| Gestures | `use:drag`, `use:panZoom` actions |

Out of scope for v1: polar coordinates, 3D, animation hooks. Tracked in [`docs/plans/2026-04-23-svelte-mafs.md`](./docs/plans/2026-04-23-svelte-mafs.md).

## Theming

```css
.my-app {
  --mafs-fg: #1a1a1a;
  --mafs-bg: #fafafa;
  --mafs-grid-color: #e0e0e0;
  --mafs-blue: #2b6cb0;
  /* etc */
}

.my-app[data-theme="dark"] {
  --mafs-fg: #f5f5f5;
  --mafs-bg: #0e0e0e;
  --mafs-grid-color: #2a2a2a;
}
```

## SSR

Components are SSR-safe — they emit semantic SVG markup on the server and hydrate without remounting. Pan/zoom/drag actions are no-ops until hydration completes.

## Documentation

Full docs and the example gallery live at **[svelte-mafs.dev](https://svelte-mafs.dev)** (Astro-built, deployed to Cloudflare Pages — TBD until Stream 7 lands the site).

In the meantime, [Mafs's documentation](https://mafs.dev) is a near-1:1 reference for the API. Where we diverge, the differences are documented in the per-component JSDoc.

## Credits

This library is a port of [**Mafs**](https://mafs.dev) by [**Steven Petryk**](https://github.com/stevenpetryk), used and extended under the MIT License. The component naming, prop shapes, and architectural conventions follow Mafs closely so its docs and examples remain useful for svelte-mafs users.

If you build something with svelte-mafs, please consider also crediting Mafs. Full attribution: [`NOTICE`](./NOTICE).

LaTeX rendering is powered by [KaTeX](https://katex.org) (MIT, by Khan Academy and contributors).

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) — TDD-first, one commit per component, visual-regression baselines committed. Worktree-friendly setup for parallel work.

## License

[MIT](./LICENSE) © Josh Hunter-Duvar and svelte-mafs contributors.
