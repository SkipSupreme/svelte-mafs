# CLAUDE instructions for Tinker

## Design System

**Read `DESIGN.md` before making any visual or UI decision.**

All typography, color, spacing, motion, sound, haptic, and progress-loop tokens are locked in that file. Do not deviate without explicit user approval.

Specifics:
- CSS variables are defined in `apps/docs/src/styles/global.css`. Use `var(--token)`, never hex.
- Any new token: add to `DESIGN.md` first, then `global.css`, then use it.
- Any deviation from `DESIGN.md` is a bug. Flag it in code review.
- If a widget needs a color or font that isn't in `DESIGN.md`, stop and propose the addition.

## Project Layout

- `apps/docs/` — Astro 6 + Svelte 5 site deployed to learntinker.com via Cloudflare Workers
- `packages/svelte-mafs/` — interactive math-viz widget engine inside Tinker (not a separate library)
- `docs/plans/` — strategy docs and research prompt packs
- `docs/research/` — Deep Research output for each module before conversion to MDX lessons

## Deployment

- Build: `pnpm -F docs build` from repo root
- Deploy: `pnpm dlx wrangler@latest deploy` from `apps/docs/`
- Production: https://learntinker.com

## Skill routing

When the user's request matches an available gstack skill, invoke it using the Skill tool as your FIRST action. Do not answer directly, do not use other tools first.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke `office-hours`
- Bugs, errors, "why is this broken" → invoke `investigate`
- Ship, deploy, push, create PR → invoke `ship`
- QA, test the site, find bugs → invoke `qa`
- Code review, check my diff → invoke `review`
- Update docs after shipping → invoke `document-release`
- Design system, brand, mood → invoke `design-consultation`
- Visual audit, mobile polish, design bugs → invoke `design-review`
- Architecture review → invoke `plan-eng-review`
- Save progress, checkpoint, resume → invoke `checkpoint`
- Code quality, health check → invoke `health`
