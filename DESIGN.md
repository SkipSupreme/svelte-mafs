# Design System — Tinker

*Locked by `/design-consultation` on 2026-04-24. Positioning pivot on 2026-04-24 after logo approval.*

> Every visual, UI, typography, color, spacing, motion, sound, and haptic decision in this repo MUST check against this file. If code deviates from this file, it's a bug.

---

## Product Context

- **What this is:** Tinker is a **general-purpose interactive learning platform.** Any subject, any age, same playful tone. Live at learntinker.com.
- **Who it's for:** Everyone who wants to *tinker* with ideas. The founder learns first (Math-for-ML is the v1 course), then Oliver (the founder's son) gets tracks built for him, then everyone else. Kids and adults, same product, same voice.
- **Starting content:** Math-for-ML (Derivatives → Gradients → Linear Algebra → Optimization → Neural Nets → Backprop → Attention → Train a GPT). Next tracks will be authored based on what the founder wants to learn and what Oliver is ready for. Platform is content-agnostic.
- **Space:** Ed-tech. Adjacent to Duolingo (play-first), Brilliant (interactive, intermediate), Khan Academy (K–12 through intro college), Math Academy (rigor), 3Blue1Brown (visual intuition), Mathigon (playground), Beast Academy (kids-first rigor). We take the play-first energy, the widget rigor, and the respect-the-learner voice, whether the learner is 9 or 39.
- **Project type:** Web app built with Astro 6 + Svelte 5 + MDX + a custom svelte-mafs interactive-math widget engine. Deploys to Cloudflare Workers. Capacitor wrap for iOS planned.
- **Core belief:** adults can play. Kids can learn hard things. "Big boys and big girls" is the register. No corner of the product talks down to the learner.

## Positioning

**"Tinker — Play. Learn. Grow."**

Playful and inviting on the outside. Rigorous on the inside. A 9-year-old learning fractions and a 34-year-old learning backpropagation should both feel welcome and both feel like the product respects them. The tone is warm and a little silly. The content doesn't cheat the math.

---

## Mascot — Tinker the Apple

A flat cartoon apple with a cream face and a gentle smile. Apple-red body, two leaves on top (one teal, one pink), three sparkle marks above (orange, teal, red). Deliberately *not* an owl — Duolingo owns the owl-as-learning-mascot register; the apple is the older "teacher's apple" symbol with no current category-defining brand on it. Logo lockup pairs the apple with the "Tinker!" wordmark in Nunito 800 + the "PLAY. LEARN. GROW." subtitle in three-color caps.

### Where Tinker appears

- **Logo and wordmark** (all marketing surfaces, app icon, favicon, browser tab).
- **Landing hero** (greeting the visitor).
- **Loading states** (apple gently bobbing, optional book or sparkle).
- **Empty states** (apple pointing to what to do next, or sleeping with a "z").
- **Celebration moments:** after a lesson, the apple does a quick double-hop. After a module, a graduation cap drops onto its leaves. After a whole course, sparkles burst around it.
- **404 page** (apple with a magnifying glass, lost).
- **Social sharing cards.**

### Where Tinker does NOT appear

- **Inside lesson bodies.** The moment the learner is doing math, the content takes over. Tinker doesn't hover in the margins of a chain-rule derivation. Distracting.
- **Inside widgets.** Math objects are the visual language of the widgets themselves (draggable points, vectors, tangent lines). The apple doesn't stand on the coordinate plane.
- **Next to paywall / consent / legal copy.** Anywhere seriousness is required. Tinker is for the fun parts.

### Expressions planned

Default smile, wide-eyed surprise, sleeping (loading / idle), graduation cap (module complete), magnifying glass (404 or search), sparkle burst (easter egg / course complete). Expressions ride on the same base apple silhouette — accessories layer on top, the apple itself doesn't morph.

### Interactivity — Tinker is alive

Tinker is **never a static PNG**. The mascot has idle behavior, reacts to the user, and hides delight. This is a first-class product feature, not decoration. Implementation lives in `components/brand/Tinker.svelte` — interactive Svelte component driving idle, hover, and click states.

**Idle behavior (always on):**
- **Idle bob** — ±6px vertical drift, 4.6s sine loop. Almost subliminal.
- **Sparkle pulse** — the three sparkle marks above the apple shift opacity slightly on a slow loop. Reads as "alive but calm."

**Reactive behavior:**
- **Cursor tilt** — when the cursor is over the hero region, the apple rotates ±3° toward the cursor's horizontal position. Subtle, feels like the mascot is paying attention.
- **Hover** — slight scale-up (~2%) on the mascot, with a transition.
- **Click** (any tap on Tinker):
  1. Quick bounce (scale 1.0 → 1.06 → 1.0 in 240ms with spring easing).
  2. Math-symbol burst of 5 particles (π, ∫, ∂, ∞, Δ, ∑, …) in teal / pink / orange.
  3. `tick` audio.
  4. Increments an invisible "pet counter" stored in localStorage. Every 10th click triggers a 18-particle milestone burst.

**Proactive hints (planned, not yet built):**
- When a learner is stuck on a `StepCheck` for 30+ seconds without submitting, Tinker pokes in from a corner with a speech-bubble hint drawn from the step's `hint` prop.
- Dismissible by clicking anywhere or pressing Escape. Max one per step, max one per 2 minutes globally. No nagging.

**Celebration choreography:**
| Moment | Animation |
|---|---|
| Correct StepCheck | Math-symbol burst at the Check button + `ding` audio. |
| Lesson complete | Apple does a double-hop (~720ms), green sweep across the lesson-complete card, full math-symbol burst, `chime` audio. |
| Module complete | Graduation cap drops onto the apple's leaves (~1100ms), bigger math-symbol burst, `anthem` audio. |
| Course complete | Full-screen math-symbol burst, sparkle accents, sustained chime. |

**Easter eggs (some shipped, some planned):**
- **Pet counter** ✅ — every 10th click in a session triggers a milestone math-symbol burst (18 symbols, 1.4× spread).
- **Konami code** (↑ ↑ ↓ ↓ ← → ← → B A) — apple wears tiny sunglasses for 10 seconds and bobs to a silent beat. *Planned.*
- **Type "tinker"** anywhere (captured keyboard input when not focused on an input) — apple does a quick bounce and a teal sparkle. *Planned.*
- **Birthday hat** — if the user has a birthdate in profile, that day the apple wears a party hat site-wide. *Planned.*
- **Milestone badges** appear on the apple after course completions: tiny graduation-cap icon for each completed course. *Planned.*

All interactivity respects `prefers-reduced-motion: reduce` — bobbing and cursor-tilt are disabled. The click bounce still fires because it's a deliberate user response, not ambient motion. The math-symbol confetti is a no-op under reduced motion.

### Rules

- Tinker is always rendered from the approved source asset (`/logo.png`). No re-drawing, no filter swaps, no AI regens that drift the character.
- Tinker keeps the original apple-red body, cream face, and teal/pink leaves on every surface. The canvas palette adjusts around the mascot, not the other way around.
- Tinker does NOT appear inside lesson bodies or widgets. See "Where Tinker does NOT appear" above.
- The mascot is **not** an owl. If a future asset drifts toward owl-shaped silhouettes, that is a bug — the apple is the brand.

---

## Aesthetic Direction

**Rigorous Delight — play-first skin, rigorous core.**

Marketing, onboarding, celebration, navigation, empty states: **playful**. Rounded geometry, warm typography, math-symbol confetti, sound, haptic bursts. Tinker the apple doing things. The confetti palette (teal, orange, pink) as accents on marketing surfaces.

Inside lessons, inside widgets, inside the knowledge graph: **rigorous**. Clean typography, Fraunces serif for display moments, Inter for prose, math-widget-as-illustration, no cartoon elements, no hand-drawn decoration. The content is the content.

The two registers share the same palette and type system, so the transition never feels jarring — it's the same product, dialed up or dialed down depending on what the user is doing.

**Decoration level:** intentional. Ambient math-object motifs on lesson surfaces. Confetti-dot motifs on marketing surfaces. Mascot takes the decorative weight on hero and celebration surfaces.

---

## Color

Tokens live in `apps/docs/src/styles/global.css`. Use the CSS variable, never the hex directly.

### Surface bands (alternating for rhythm)

| Var | Hex | Role |
|---|---|---|
| `--band-onyx` | `#0b0d12` | Dark hero, closing CTA, premium sections. |
| `--band-white` | `#fdfdfc` | Primary light surface. |
| `--band-cream` | `#faf6ef` | Warm-light section. |
| `--band-lavender` | `#ecebfb` | Cool-light section. Pairs well with sea + pink accents. |
| `--band-mint` | `#e6f3e8` | Success / approval section. |
| `--band-sky` | `#e6eff8` | Reflective / who-it's-for section. |
| `--band-butter` | `#fbf2dc` | Quote / pull-out section. |

Dark-theme variants for each band are defined in `:root[data-theme="dark"]` in the same file.

### Ink accents

| Var | Light hex | Dark hex | Role |
|---|---|---|---|
| `--ink-red` | `#e6396a` | `#ff7095` | **Primary brand color.** Apple body, display italic accent words, primary math ink. Replaced `--ink-violet` in the 2026-04-24 owl→apple pivot. |
| `--ink-coral` | `#ff6a4d` | `#ff8a74` | Interactive elements (drag handles, tangent lines). |
| `--ink-sea` | `#2a9fd6` | `#5db4e0` | Secondary math (axes, grids), info state. |
| `--ink-sun` | `#f2b400` | `#f5c745` | XP, streak flame, warmth accents. |
| `--ink-teal` | `#4ecdc4` | `#6de0d7` | Confetti accent. Celebration moments. One of the apple's leaves. |
| `--ink-pink` | `#ff4f9e` | `#ff7bbd` | Confetti accent. The other apple leaf. Playful emphasis. |
| `--ink-orange` | `#ff9f43` | `#ffb873` | Confetti accent. Energy beats. |

### CTA (always green, always the one green)

| Var | Hex | Role |
|---|---|---|
| `--cta` | `#22c55e` | Primary CTA pill. Completion fill. "Ding!" visual pulse. |
| `--cta-hover` | `#16a34a` | Hover on CTA. |
| `--cta-fg` | `#0a1a10` | Text on green CTA. |

### Text

| Var | Light | Dark | Role |
|---|---|---|---|
| `--site-fg` | `#17181a` | `#ededec` | Body text. |
| `--site-fg-muted` | `#585a60` | `#9a9a98` | Lede, metadata, hints. |

### Rules

- **Red (`--ink-red`) is the primary brand color.** Mascot body, any single-accent-only moment, display italic accent words. **Purple/violet is explicitly retired** — it's the AI slop color.
- **Confetti palette (teal + orange + pink) is for celebration.** Never use more than two confetti colors together in a single element — three becomes carnival.
- **One CTA color per page.** Green is the only button color that promises action.
- **Sun stays in XP/streak territory.** Don't spread it.
- **Dark mode is a redesign, not an inversion.** Hand-tuned block already in `global.css`. Do not auto-invert.

---

## Typography

Four faces. Google Fonts loaded in `Base.astro`. If a new face is needed, add to that `<link>`, update this file, done.

| Role | Family | Usage |
|---|---|---|
| Wordmark | **Nunito** (700, 800, 900 + italic) | The "Tinker!" wordmark. Next to (or under) the apple. Rounded, heavy, warm. |
| Display | **Fraunces** (500, 600, 700, 800, italic) | Section headlines, lesson titles, big display moments. `text-wrap: balance` always. |
| Body | **Inter** (400, 500, 600, 700) | Lede, prose, lesson body, UI. |
| Mono / data | `ui-monospace`, `"SF Mono"`, Menlo | Readouts, step counters, code samples. |
| Gamification | **Space Grotesk** (500, 600, 700) | XP counters, streak numbers, badge text. Rounded-sans register that softens Fraunces. |

**Scale** (clamp-based, fluid):

| Token | Value |
|---|---|
| Display huge | `clamp(3rem, 9vw, 6.5rem)` |
| Display | `clamp(2.1rem, 5.5vw, 4.25rem)` |
| Display manifesto | `clamp(1.75rem, 4.5vw, 3.5rem)` |
| H2 step title | `clamp(1.5rem, 3vw, 2rem)` |
| Lede | `clamp(1.05rem, 1.6vw, 1.3rem)` |
| Body | `1rem` (16px base) |
| Small / meta | `0.82rem` |
| Mono code | `0.9rem` |
| Label / eyebrow | `0.75rem` uppercase, `0.14em` letter-spacing |

**Rules:**
- Wordmark uses Nunito 800, tight letter-spacing, one exclamation point at end for marketing lockups. Lowercase-t-with-dot style matches the logo asset.
- All `.display` headlines use `text-wrap: balance`.
- Italic on display is ALWAYS paired with a color accent (usually `--ink-red`). Never italic in plain body color.
- Numbers that matter use tabular-nums: `font-variant-numeric: tabular-nums`.

---

## Spacing

Base unit: `4px`. Comfortable density.

| Token | px |
|---|---|
| `2xs` | 2 |
| `xs` | 4 |
| `sm` | 8 |
| `md` | 16 |
| `lg` | 24 |
| `xl` | 32 |
| `2xl` | 48 |
| `3xl` | 64 |
| Band padding | `clamp(3rem, 7vw, 5.5rem)` vertical, `1.25rem` horizontal |
| Hero band padding | `clamp(3rem, 8vw, 6.5rem)` vertical |
| Band padding <380px | `clamp(2rem, 6vw, 3rem) 1rem` |

---

## Layout

- **Approach:** hybrid. Grid-disciplined inside widgets and lessons. Creative-editorial asymmetry in marketing bands.
- **Max content width:**
  - `--max-prose: 68ch`
  - `--max-shell: 1120px`
  - `--max-hero: 1200px`
- **Border radius:**
  - `--radius-sm: 4px`
  - `--radius-md: 8px`
  - `--radius-lg: 12px`
  - Widget cards: `20px`
  - `--radius-pill: 999px`
- **Shadow recipe for elevated cards:**
  `0 1px 0 rgba(0,0,0,0.04), 0 24px 48px -28px color-mix(in srgb, <accent> 50%, transparent)`
  where `<accent>` is the dominant ink of the component.

---

## Motion

- **Easing:**
  - Enter: `cubic-bezier(0.2, 0.8, 0.2, 1)` (spring-ish)
  - Exit: `cubic-bezier(0.4, 0, 1, 1)` (ease-in, quick)
  - Move / slider: `ease-in-out`
- **Duration:** micro 120-180ms · short 220-320ms · medium 450-700ms · long 900-1500ms.
- **Mascot animation:** Tinker blinks once per 4-8 seconds on hover surfaces. On celebration, Tinker does a 500ms hop. Respect `prefers-reduced-motion: reduce` — Tinker goes static.
- **No parallax. No scroll-jack.**

### Celebration sequences

| Event | Visual | Audio | Haptic |
|---|---|---|---|
| Step advance | Soft slide-in, green ✓ flash | `tick` | `10ms` |
| Correct answer | Widget jiggle, green halo pulse, 6-particle confetti in teal/orange/pink | `ding` | `20ms` |
| Lesson complete | Full-width green sweep, Tinker does a hop, XP ticker animates, streak pulses | `chime` | `50ms` |
| Module complete | Full-screen confetti burst (teal/orange/pink), Tinker puts on graduation cap, XP burst | `anthem` | `[50, 100, 50]` pattern |

All celebrations respect `prefers-reduced-motion`. Audio mute-by-default only if `prefers-reduced-motion: reduce`; otherwise on.

---

## Sound

Four tones, all Web Audio API sine waves with attack/decay envelopes. Implementation in `apps/docs/src/lib/sound.ts`.

| Token | Frequency | Duration | Role |
|---|---|---|---|
| `tick` | 880 Hz | 60ms | Step advance, widget snap, continue click. |
| `ding` | 880 + 1320 Hz chord | 120ms | Correct answer. |
| `chime` | 660 → 880 → 1320 Hz | 400ms | Lesson complete. |
| `anthem` | 523 → 659 → 784 → 1047 Hz | 1000ms | Module complete. |

Tones progress up the scale. Finishing a lesson *sounds* like finishing.

Rules:
- Default on. Settings toggle to mute, persisted to localStorage.
- Sound only on earned state changes. Never on page load or navigation.
- If `AudioContext` fails: fail silent, never throw.
- Respect `(prefers-reduced-motion: reduce)` — if so, default to off.

---

## Haptics

Mobile only (Capacitor native or PWA via `navigator.vibrate()`).

| Event | Pattern |
|---|---|
| Correct answer | `10ms` |
| Lesson complete | `50ms` |
| Module complete | `[50, 100, 50]` |

No haptics on step advance, no haptics on hover.

---

## Progress Loops — XP, Streak, Mastery

Duolingo-grade completion loops, minus the streak-shaming.

### XP

- Step advance: 1 XP
- Correct StepCheck first try: 5 XP
- Correct StepCheck 2nd-3rd try: 2 XP
- Lesson complete: 20 XP
- Module complete: 100 XP

Display: always visible in nav as `XP 2,340`, Space Grotesk with small upward-arrow glyph. On gain, `+N XP` floater near the event, ticker in nav, `tick` sound.

No XP cap. No leaderboards in v1.

### Streak

- Counted as consecutive UTC days with at least one completed step.
- Display: `14 🔥` (mascot-flame illustration is fine for marketing surfaces, emoji is fine for app chrome).
- Sundays never break a streak.
- Opt-in only. Off by default for the first 30 days of a user's lifetime; one-time prompt after that.
- Never send guilt push notifications. Ever.
- Max-streak saved separately as an all-time badge.

### Mastery (Skill-tree glow)

Module node transitions: `planned` (dashed, muted) → `reachable` (solid outline, subtle pulse) → `drafting` (butter fill, currently working) → `shipped`/`mastered` (green fill, solid label, optional glow).

Mastery defined as FSRS retention ≥ 0.75 once FSRS ships (Phase 3 of master plan). Until then, mastered = every lesson in module completed.

No-guilt rules: no red badges, no "falling behind" notifications, Sundays free.

---

## Decoration

Two decoration registers depending on surface:

**Marketing / hero / celebration surfaces:** Tinker the apple + floating math-symbol motifs (π, ∫, ∂, ∞, Δ, ∑ in teal/orange/pink) + big display typography. Rounded, warm, alive.

**Lesson / widget / knowledge-graph surfaces:** ambient math objects at 0.15-0.35 opacity. Vector arrows, coordinate points, grid fragments, integral symbols, derivative notation. No mascot inside widgets.

Rules:
- No icon circles. No stock-illustration ed-tech tropes (character with laptop, light-bulb in hand).
- Max 5 floating decorations per hero. More is noise.
- `prefers-reduced-motion: reduce` → decorations freeze.

---

## Components (implementation targets)

Tokens and patterns must match this file. Deviations are bugs.

### Existing components to audit

- `Nav.astro` — replace SVG brand mark with `<img src="/logo.png">` and `Tinker!` wordmark in Nunito 800. Add XP counter + streak + avatar to right side.
- `pages/index.astro` — dark hero band (`--band-onyx`) with the apple mascot, Fraunces headline, floating math-symbol accents, green CTA, live math-widget card.
- `pages/courses/[course]/index.astro` — pastel arc bands, unchanged structurally. No mascot inside the course page.
- `components/course/SkillTreeMap.svelte` — node glow states for reachable / drafting / mastered; no mascot inside.
- `layouts/Lesson.astro` — celebration sequence on lesson complete (Tinker hops, sound, confetti, XP ticker). No mascot while the user is mid-lesson.
- `components/lesson/StepCheck.astro` — ding + confetti burst on correct answer.
- `components/demos/*.svelte` — widgets use `--demo-card` / `--demo-stage` / `--ink-*` tokens; no mascot inside.

### Components — current state

- `public/logo.png`, `public/logo-mark.png` — approved logo assets. ✅ shipped.
- `components/brand/Tinker.svelte` — interactive apple mascot. Idle bob, cursor-aware tilt, click-bounce, click math-symbol burst, pet counter, reduced-motion safe. ✅ shipped. Renamed from the original `.astro` plan because state-machine behavior requires reactivity.
- `components/celebrate/TinkerHop.svelte` — lesson- and module-complete hop animation, listens for `tinker:celebrate` window events with `{ level: 'lesson' | 'module' }`. Module level adds a graduation cap drop. ✅ shipped.
- `lib/sound.ts` — Web Audio sine palette (`tick` / `ding` / `chime` / `anthem`) with attack/decay envelopes, mute toggle persisted to localStorage, default-mute when reduced motion. ✅ shipped.
- `lib/xp.ts` — XP awards (`stepAdvance` / `correctFirstTry` / `correctRetry` / `lessonComplete` / `moduleComplete`), event emitters (`tinker:xp`, `tinker:streak`), `bumpStreak()`, `vibrate()` helper. ✅ shipped (streak-bumping wired but not called — see Decisions Log).
- `lib/confetti.ts` — `burst(target, opts)` particle helper using teal/pink/orange/violet, Web Animations API, self-cleaning layer, reduced-motion no-op. ✅ shipped.
- `components/nav/XpCounter.svelte` — XP readout with live `+N` floater and scale pulse. **Inlined into `Nav.astro`** rather than extracted as its own Svelte component. The Nav script listens for `tinker:xp` events directly. Promote to a standalone Svelte component if it grows beyond what fits in Nav.
- `components/nav/StreakFlame.svelte` — streak chip. **Inlined into `Nav.astro`** for the same reason. Display gated on `streak > 0`. Bump logic awaits the 30-day opt-in flow.
- `components/celebrate/ConfettiBurst.svelte` — replaced by `lib/confetti.ts` (a plain JS module). Functions, not components, are the right shape since callers fire from non-Svelte script contexts (the Lesson layout's TS script tag).

---

## Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-24 | Design system created via `/design-consultation` | First real design pass. Approved hero variant: `variant-A.png`. User feedback: "Definite A". |
| 2026-04-24 | Dark hero band added (`--band-onyx #0b0d12`) | White hero looked like every other ed-tech landing. Dark hero with violet/coral ink creates premium read. |
| 2026-04-24 | Space Grotesk added | Rounded-sans register for XP/streak. Softens Fraunces + Inter where fun needs to show. |
| 2026-04-24 | Sound palette locked | User explicitly asked for "nice sounds when you complete something". Web Audio, zero bundle cost. |
| 2026-04-24 | XP + streak + mastery-glow progress loop | Duolingo completion loops without streak-shaming. Sundays always free. |
| **2026-04-24 (pivot)** | **Positioning: general learning platform, play-first, all ages** | **User dropped a cartoon owl logo and clarified the vision: "I've always wanted this as a general learning app. Starting with subjects I want to learn. Eventually tracks for my son Oliver. Adults can play too! And learn too! We're big boys now!" Tinker is a platform, not a math-for-ML-specific product. Play-first voice across all ages. Math-for-ML is track #1, not the whole thing.** |
| 2026-04-24 (pivot) | Cartoon mascot added: Tinker the Owl | Violet body, cream face, winking, confetti bursting. Lives on marketing surfaces, not inside lessons. Replaces "math objects are the mascot" rule on outer surfaces; math objects retain that role inside widgets. |
| 2026-04-24 (pivot) | Palette extended: teal, pink, orange | Confetti colors from the logo become first-class ink tokens. Used on celebration surfaces. |
| 2026-04-24 (pivot) | Wordmark face: Nunito 800 | Matches logo typography. Rounded, heavy, warm. |
| 2026-04-24 (build) | Implementation pass: alive layer | Built `lib/sound.ts`, `lib/xp.ts`, `lib/confetti.ts`, `components/brand/Tinker.svelte`, `components/celebrate/TinkerHop.svelte`. Wired `Lesson.astro` to play tick on advance, ding+confetti+XP on correct answer, chime+big confetti+TinkerHop on lesson complete. Nav listens for `tinker:xp` events and shows a `+N` floater + scale pulse. Static hero owl in `index.astro` replaced with the interactive `<Tinker />` (idle bob, hover-tilt, click-bounce, click confetti, pet counter). |
| 2026-04-24 (build) | Mascot interactive components are `.svelte`, not `.astro` | The Components section originally listed `Tinker.astro` and `TinkerHop.astro`. Interactive state machines (idle bob, click bounce, event listeners) need reactive runtime, so both are Svelte components mounted with `client:visible` (hero) or `client:idle` (lesson celebration). The Mascot — Interactivity section already implied `.svelte`; this clarifies the Components section to match. |
| 2026-04-24 (build) | Streak counts but does not display yet | `bumpStreak()` is implemented in `lib/xp.ts` but is NOT yet called from `Lesson.astro`. DESIGN.md §Streak gates display on a 30-day-then-opt-in prompt that hasn't shipped. Will wire on first completed step of the day once that opt-in flow exists. |
| **2026-04-24 (pivot v2)** | **Mascot: owl → apple. Primary color: violet → red.** | User: "I've flattened it out because the owl looked too much like Duolingo. AI uses purple way too much, it's the AI slop color." New asset is a flat cartoon apple in `public/logo.png` (1024×1536 with apple + Tinker! wordmark + PLAY/LEARN/GROW subtitle). `--ink-red` (#e6396a) replaces `--ink-violet` everywhere. The mascot is **deliberately not an owl** — Duolingo owns that register; the apple is the older "teacher's apple" symbol. |
| 2026-04-24 (pivot v2) | Confetti: dots → math symbols | User: "make the confetti like math symbols and shit". `lib/confetti.ts` now bursts π / ∫ / ∂ / ∇ / ∞ / ∑ / √ / Δ / x / + / = / etc. in Fraunces italic, palette teal/pink/orange (no purple). Hero static decorations are also six floating math symbols above the apple. Reinforces the "math is play" identity. |
| 2026-04-24 (pivot v2) | Owl-specific Mascot interactions retired | Eye tracking, blink, head tilt, "settles to nap" assumed owl physiology (eyes, pupils, head/body separation). The flat apple has none of those. Replaced with idle bob, sparkle pulse, cursor-aware tilt of the whole apple, click bounce + math-symbol burst. Mascot Interactivity section in this file rewritten accordingly. |
