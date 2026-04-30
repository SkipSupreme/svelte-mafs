# TODOS

## Tinker Premium Polish Plan — Phase-1-conditional follow-ups

### Bounding-box label-collision detection in Playwright

**What:** Add bounding-box label-collision detection to the visual regression suite as a stronger enforcement of the "no covered numbers" Quality Bar invariant.

**Why:** Visual regression catches "pixels changed" but may miss subtle 3-px label overlaps that fall under the snapshot fuzz threshold. If Phase 1 baselines surface this gap, this is the fix.

**Pros:**
- Stronger guarantee that `no covered numbers` actually holds across all 86+ widgets.
- Independent of pixel-diff sensitivity — works at the DOM level.

**Cons:**
- Real implementation cost (bounding-box logic for SVG + DOM, exception handling for legitimate adjacency).
- May be unnecessary if Phase 1's plain visual regression turns out to catch enough.

**Context:** Surfaced during /plan-eng-review on 2026-04-30 (TODO-1). The Quality Bar invariant: numbers never overlap data marks, gridlines, or other labels. Phase 1 enforces via Playwright snapshots; this TODO is the belt-and-suspenders version that gets evaluated AFTER Phase 1 produces real evidence.

**Depends on / blocked by:** Phase 1 baselines complete. Cannot be evaluated until we see what plain visual regression actually catches in the wild.

**Decision criterion:** Re-open after Phase 1 ships. If audit found ≥3 subtle overlap bugs that visual regression missed, build this. If 0-1, close with note "regression suite caught everything."

### Stylelint Phase 3 cleanup — remaining hex literals

**What:** 11 remaining `color-no-hex` violations in component files after the bulk `#fdfdfc → var(--on-color-fg)` sweep landed.

**Why:** Lint surfaced them; per-file judgment needed for the replacement.

**Findings (run `pnpm lint:css` to refresh):**
- `#fff` / `#ffffff` — alias of `#fdfdfc`. Replace with `var(--on-color-fg)`.
- `#111`, `#666` — generic dark / muted text. Replace with `var(--site-fg)` / `var(--site-fg-muted)` after verifying theme behaviour.
- `#b91c1c` — error red. Replace with `var(--site-error)`.
- `#e8e8e3` — light border. Replace with `var(--site-border)` or `var(--demo-card-border)`.
- `#1a1402` — accent text used in sunglasses easter egg. Either add a token or stylelint-disable with a TODO.

**Pros:** Closes the lint cleanly, lint becomes truly green-or-blocking in CI.
**Cons:** Each replacement needs a per-file check that the token's theme behaviour is what the component wants (some intent here is "always-this-color regardless of theme", same case as `--on-color-fg`).
**Context:** Surfaced by /plan-eng-review CQ-1 stylelint setup (2026-04-30). The `#fdfdfc` sweep was mechanical; these last 11 need eyes.
**Depends on / blocked by:** None. Can be done at any time.
