# Changesets

This directory holds [Changesets](https://github.com/changesets/changesets) — small Markdown files describing version bumps that haven't shipped yet.

## TL;DR for contributors

When your PR changes anything users see, add a changeset:

```bash
pnpm changeset
```

Pick the affected package(s), pick `patch` / `minor` / `major`, and write a one-line summary. Commit the generated `.changeset/<random-name>.md` with your PR.

## What happens on merge

When a PR with one or more changesets lands on `main`, the **Release** workflow either:

1. Opens (or updates) a single **Version Packages** PR that consolidates pending changesets into the next version + CHANGELOG entry, **or**
2. If that Version Packages PR is what was just merged, runs `pnpm changeset publish` to push to npm.

You don't run `pnpm changeset version` or `publish` yourself — the workflow does it.

## Bump rules of thumb

- **patch** — bug fixes, internal refactors, perf
- **minor** — new components, new props, anything additive
- **major** — removed/renamed exports, prop signature breaks, runtime behavior changes that existing consumers can't opt out of

The `apps/docs` package is in `ignore` and never publishes — don't bother adding changesets for it.
