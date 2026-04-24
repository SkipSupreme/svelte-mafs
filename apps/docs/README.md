# svelte-mafs docs site

Astro 6 + `@astrojs/svelte` + MDX. Deploys to Cloudflare Pages.

## Run locally

From the monorepo root:

```bash
pnpm install
pnpm -F docs dev          # http://localhost:4321
pnpm -F docs build        # prebuild runs docgen, then astro build
pnpm -F docs preview      # wrangler preview against the built bundle
```

Node `>=22.12` required (matches Astro 6 and the Cloudflare adapter).

## Project layout

```
apps/docs/
├── scripts/
│   └── docgen.ts                  # parses lib exports → MDX
├── src/
│   ├── layouts/Base.astro         # <html> shell + Nav + footer
│   ├── components/
│   │   ├── Nav.astro              # sticky top bar
│   │   ├── ThemeToggle.astro      # inline-script dark-mode flip
│   │   ├── DemoPlaceholder.astro  # decorative preview card
│   │   └── ThemePlayground.svelte # hydrated island
│   ├── pages/
│   │   ├── index.astro            # hero + previews + features
│   │   ├── getting-started.mdx    # install, first snippet, a11y
│   │   ├── examples.astro         # stub; Wave B backfills
│   │   ├── playground.astro       # wraps ThemePlayground
│   │   └── api/
│   │       ├── index.astro        # TOC (reads api-manifest.json)
│   │       └── {vec,math,...}.mdx # auto-generated per module
│   ├── data/api-manifest.json     # written by docgen
│   └── styles/global.css          # design tokens + resets
└── astro.config.mjs               # svelte + mdx + cloudflare
```

## Auto-generated API pages

`pnpm -F docs docgen` reads `packages/svelte-mafs/src/index.ts`, follows
each re-export to its source (`.ts` or, in Wave B, `.svelte`), extracts
JSDoc + signatures, and writes one MDX page per module to
`src/pages/api/`. A manifest lands at `src/data/api-manifest.json` for
the index page to read.

The script runs as `prebuild` so CI and local builds are always
consistent with whatever the lib currently exports. The generated MDX
is committed so fresh clones can run `dev` without a prebuild step.

When a new component PR merges to `main`, run `pnpm -F docs docgen &&
git add src/pages/api src/data/api-manifest.json && git commit` to
refresh the reference.

## Deploying to Cloudflare Pages

The site builds to **`apps/docs/dist/client/`** — that's the publish
directory to point Pages at. The `@astrojs/cloudflare` adapter also
emits a server bundle at `dist/server/` for any page that opts into
SSR with `export const prerender = false`, but the current chassis
is 100 % prerendered.

### Pages (Git integration)

1. Create a Cloudflare Pages project tied to this repo.
2. **Build command**: `pnpm -F docs build`
3. **Build output directory**: `apps/docs/dist/client`
4. **Root directory**: *(leave blank — the monorepo root)*
5. Environment variables:
   - `NODE_VERSION=22`

### Workers / Pages with functions (if SSR is ever enabled)

The `apps/docs/wrangler.jsonc` file configures the Cloudflare Worker.
From `apps/docs/`:

```bash
pnpm -F docs build
pnpm -F docs wrangler deploy
```

If you enable SSR routes, set the `nodejs_compat` compatibility flag
on the Worker — Svelte's server renderer imports `node:async_hooks`,
which needs it. Pure static deploys (the current state) don't need
the flag.

## Commands

| Command                     | What it does                                     |
| :-------------------------- | :----------------------------------------------- |
| `pnpm -F docs dev`          | Dev server on `localhost:4321`                   |
| `pnpm -F docs build`        | `prebuild` → `docgen` → `astro build`            |
| `pnpm -F docs docgen`       | Regenerate API pages from lib source             |
| `pnpm -F docs preview`      | Preview the built site via Wrangler              |
| `pnpm -F docs astro check`  | Type-check `.astro` / `.svelte` / `.ts`          |

## Contributing pages

- **Astro pages** (`src/pages/*.astro`) use the `Base` layout via
  `<Base title="..." prose>…</Base>`.
- **MDX pages** (`src/pages/*.mdx`) set the layout in front-matter:
  `layout: ../layouts/Base.astro`, with `title`, `description`, and
  optional `prose: true` for the reading-width column.
- **Svelte islands** should default to `client:visible` unless a
  component needs to be interactive above the fold (then `client:load`).
- **A11Y**: every interactive page must satisfy `A11Y.md` before
  merging.
