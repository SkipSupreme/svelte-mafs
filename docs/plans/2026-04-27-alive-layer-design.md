# Alive Layer — Design

*Validated 2026-04-27 via `superpowers:brainstorming`. Backs the "give the learning app some life" pass against the shipped v2 mascot infrastructure (`lib/sound.ts`, `lib/confetti.ts`, `lib/xp.ts`, `Tinker.svelte`, `TinkerHop.svelte`).*

> Every visual, motion, audio, and haptic decision in this doc is checked against `DESIGN.md`. Where the two disagree, `DESIGN.md` wins; flag and resolve before deviating.

---

## 1. Overview, scope, non-goals

A coordinated "alive layer" pass across five surfaces, all backed by the same shared primitive set: keep custom `lib/confetti.ts` (math symbols) and `lib/sound.ts` (sine palette), add Svelte 5 `Spring`/`Tween` runes from `svelte/motion`, add `tw-animate-css` for ambient loops, use the new reactive `prefersReducedMotion` from `svelte/motion`. Five surfaces, one design language, one shared primitive set.

**Surfaces.**

1. **Mid-lesson microcelebrations** — StepCheck correct/wrong choreography, Step advance, widget jiggle. Mascot stays out.
2. **Lesson + module complete sequences** — multi-stage `async/await` chains (Spring fill → Tween XP tick → TinkerHop → confetti → chime).
3. **Mascot backlog burndown** — Konami code sunglasses, "type tinker" anywhere, proactive hint pokes, milestone badges, birthday hat.
4. **Hero/landing ambient life** — floating math-symbol decorations drift, sparkle marks pulse, mascot gets a sleepy yawn after long inactivity.
5. **XP counter + streak chip polish** — Tween-driven counter (`Math.round(display.current)`), smoother +N floater, streak flame flicker via `tw-animate-css`.

**Non-goals (locked).**

- No swap of `lib/confetti.ts` or `lib/sound.ts` — math symbols + sine waves are brand (`DESIGN.md` Decisions Log, 2026-04-24 pivot v2).
- No Rive, GSAP, Motion, anime.js, or `canvas-confetti`.
- No Tinker inside lesson bodies or widgets (`DESIGN.md` §Where Tinker does NOT appear).
- No streak guilt, no leaderboards, no full-screen red.
- No new color tokens; `--ink-red` stays primary, no purple, no introductions.
- No paid asset generation. No timeline estimates anywhere in the doc.

**Adds to `package.json`.** One: `tw-animate-css`. Everything else is already on the box.

**Adds to `global.css`.** Six new keyframes (sparkle-flicker, tinker-z-rise, flame-flicker, sweep-green, halo-pulse, wrong-shake). No new tokens.

---

## 2. Stack additions and where each primitive is used

The whole design boils down to four primitives mapped to four motion archetypes.

| Primitive | Source | Used for |
|---|---|---|
| `Tween` | `svelte/motion` | **Numeric reveals.** XP counter ticking, score reveal, lesson-progress percent. Read `Math.round(t.current)` in template; set `t.target = N` to animate. |
| `Spring` | `svelte/motion` | **Physical-feeling overshoots.** Progress bar fills (`stiffness: 0.12`, `damping: 0.4` for the Duolingo bounce), widget-jiggle scale on correct, hint-poke entry. |
| `prefersReducedMotion.current` | `svelte/motion` (reactive, replaces manual `matchMedia`) | **Single source of truth for reduced motion.** Migrate `Tinker.svelte:54-59` to read this rune; the local `reducedMotion` `$state` and `onMount` block both go away. |
| `tw-animate-css` classes | new dep | **Ambient loops.** `animate-pulse` on sparkle marks, custom keyframes for streak-flame flicker, sleepy-yawn, sparkle-shimmer. CSS-only, zero JS. |
| `lib/confetti.ts` | existing | **Earned-state visual rewards.** Stays. |
| `lib/sound.ts` | existing | **Earned-state audio rewards.** Stays. |
| `lib/xp.ts` | existing | **State + event bus.** Stays. Already emits `tinker:xp` and `tinker:streak` window events. |

**Two new shared modules.**

- `lib/celebrate.ts` — pure-function orchestrator. Exports `celebrateLesson()`, `celebrateModule()`, `celebrateCorrect(target)`, `celebrateWrong(target)`. Wraps the existing `play()`, `burst()`, XP, and `tinker:celebrate` event-fire into one async-chained sequence per moment. This is what `Lesson.astro` and `StepCheck.astro` will call instead of inline calls.
- `lib/easterEggs.svelte.ts` — global keyboard listener (Konami, "type tinker"), a `$state`-backed `mascotMode = 'normal' | 'sunglasses' | 'birthday'`, and a `MascotEffects` slot the apple reads. Initialized once via `$effect.root` from `Tinker.svelte`.

**Removed.** `Tinker.svelte:54-59` (`matchMedia` `onMount`) — replaced by the rune. The local `reducedMotion` state stays as a `$derived` from the rune for the CSS class binding.

---

## 3. Mid-lesson microcelebrations

The rigorous core, but alive at every interaction. Mascot stays out (`DESIGN.md` §Where Tinker does NOT appear). Motion lives on the math objects themselves.

**Correct StepCheck answer.** Currently `StepCheck.astro` fires inline `play('ding')`, `burst(...)`, and XP. Move that orchestration into `celebrateCorrect(target)` in `lib/celebrate.ts` and chain:

```ts
async function celebrateCorrect(checkBtn: HTMLElement) {
  jiggleSpring.set(1);              // Spring scale 1 → 1.06 → 1 on widget
  play('ding');
  burst(checkBtn, { count: 6 });    // existing math-symbol burst
  vibrate(20);                      // existing
  awardXp('correctFirstTry');       // emits tinker:xp; Nav handles +N float
  await haloPulse(checkBtn);        // CSS green halo overshoot, 320ms
  announce('Correct.');
}
```

`jiggleSpring` is a `Spring(1, { stiffness: 0.18, damping: 0.55 })` mounted on the widget root. Step exposes `--widget-scale: {jiggleSpring.current}` to the wrapping `<div>`. The halo is a `tw-animate-css`-driven `::after` pseudo-element with `animate-ping`-style scale + opacity decay.

**Wrong StepCheck answer.** New behavior. `DESIGN.md` §A11y: "localized shake or border-color shift, not a full-screen red." Implement as `wrongShake` CSS keyframe (10px horizontal, 6 frames, 240ms) on the input + a 320ms border color shift to `--ink-coral` (NOT red — coral is the interactive token, red is the brand mascot color). No haptic, no sound — wrong answers are quiet by design. Aria-live announces "Try again. Read the hint."

**Step advance.** Currently `Step.astro` with `EndgameCallback.astro` plays tick. Add a Svelte `transition:slide` (existing `svelte/transition`) with `easing: cubicOut, duration: 280` on the new step, plus a green ✓ flash via `tw-animate-css` `animate-flash` on the prev-step's number badge. The `tick` sound and 10ms haptic stay where they are.

**Widget interactions.** New shared `useDragSpring()` factory exported from `packages/svelte-mafs` — wraps `MovablePoint.svelte` drag in a `Spring` with low stiffness for snap-back when constraint-clamped. Optional, opt-in per widget.

**Aria-live mirror.** Single `<SrAnnouncer>` mounted in `Lesson.astro` listens for `tinker:announce` events; correct/wrong/advance push messages.

---

## 4. Lesson + module complete sequences

Today these moments fire each piece independently inside `Lesson.astro` script tag. The design tightens them into one orchestrator each, in `lib/celebrate.ts`, chained with `async/await` so timing is explicit and tunable in one place.

**`celebrateLesson()` — `DESIGN.md` §Motion lesson-complete row.**

```ts
export async function celebrateLesson(card: HTMLElement) {
  progressSpring.set(1);                       // Spring overshoot (.12 stiff / .4 damp)
  await wait(120);
  burst(card, { count: 14 });                  // math-symbol burst
  scoreTween.target = scoreTween.current + 20; // XP ticks from current → +20 over 900ms
  window.dispatchEvent(new CustomEvent('tinker:celebrate', { detail: { level: 'lesson' } }));
  play('chime');
  vibrate(50);
  await sweepGreen(card);                      // CSS keyframe gradient sweep, 640ms
  announce(`Lesson complete. Plus 20 XP.`);    // aria-live
}
```

`progressSpring` and `scoreTween` are module-scoped instances; the lesson-progress bar and XP counter `$derived` from their `.current`. `sweepGreen` adds and removes a class — no JS animation.

**`celebrateModule()` — `DESIGN.md` §Motion module-complete row.**

```ts
export async function celebrateModule(card: HTMLElement, nodeId: string, nextModuleName?: string) {
  window.dispatchEvent(new CustomEvent('tinker:celebrate', { detail: { level: 'module' } }));
  await wait(180);
  burst(card, { count: 28, spread: 1.6 });     // bigger burst, wider spread
  scoreTween.target = scoreTween.current + 100;
  play('anthem');
  vibrate([50, 100, 50]);
  await glowMasteredNode(nodeId);              // SkillTreeMap node → green fill + glow
  const tail = nextModuleName ? ` ${nextModuleName} unlocked.` : '';
  announce(`Module complete. Plus 100 XP.${tail}`);
}
```

The `glowMasteredNode` call mutates the skill-tree node from `drafting` → `mastered` per `DESIGN.md` §Mastery, with a `tw-animate-css`-driven one-time pulse on the new state.

**Reduced-motion fork (one branch in each function).** When `prefersReducedMotion.current` is true: `progressSpring.set(1, { instant: true })`, `scoreTween` set instant, `burst()` is already a no-op, `TinkerHop` is gated already. Sound + haptic still fire — they're deliberate user-earned responses, not ambient.

**Why one orchestrator per moment.** Today, tweaking the lesson-complete feel means editing `Lesson.astro`, `StepCheck.astro`, and the inline event listeners. After this pass, it's one function. Tuning is a one-line edit.

---

## 5. Mascot backlog burndown

`DESIGN.md` §Mascot lists five *planned* moments. All concentrate in one new module + one new component, plus prop additions to the existing `Tinker.svelte`.

**`lib/easterEggs.svelte.ts` (new).** Singleton class with `$state`-backed flags, global keyboard listener wired in `$effect.root`:

```ts
class Eggs {
  sunglasses = $state(false);   // Konami 10s window
  bouncePulse = $state(0);      // increments → Tinker reads as derived effect
  init() {
    $effect.root(() => {
      window.addEventListener('keydown', this.#onKey);
      return () => window.removeEventListener('keydown', this.#onKey);
    });
  }
}
export const eggs = new Eggs();
```

`#onKey` keeps a rolling buffer of last 10 keys. Match against `↑↑↓↓←→←→ba` → `sunglasses = true`, `setTimeout` 10s to flip back. Last 6 keys match `tinker` (only when `document.activeElement` isn't an input/textarea/contenteditable) → `bouncePulse++`. `Tinker.svelte` adds a `$effect` reading `eggs.bouncePulse` to fire the click-bounce + teal `burst({ palette: ['teal'] })`.

**Mascot accessory layer.** Add three optional props to `Tinker.svelte`: `sunglasses` (bound to `eggs.sunglasses`), `birthday: boolean` (server-passed from `session.birthdate` vs UTC today), `completedCourses: number` (server-passed). Each renders an absolutely-positioned SVG accessory over the apple — sunglasses across the eye-region, party hat on the leaves, one tiny grad-cap badge per completed course at fixed offset positions. PNG sprites in `public/mascot/` so no inline SVG bloat. Layered on `<div class="tinker-accessories">` siblings to `<img class="tinker-img">`.

**Proactive hint poke.** New `components/lesson/StuckHint.svelte`. `StepCheck.astro` starts a 30s timer on mount, clears on submit. On expiry, dispatches `tinker:stuck` with `{ hint }`. `StuckHint` is mounted once in `Lesson.astro` (`client:idle`), listens, throttles via `localStorage` last-shown timestamp (2 min global). Renders bottom-right corner: `/logo-mark.png` thumbnail + speech bubble with the hint string, `Spring`-driven slide-in from off-canvas. Dismiss via click anywhere or Escape. Reduced motion → instant fade, no slide.

**Birthday + milestone data.** Astro page reads `better-auth` session, computes `birthday: dob && sameUtcDay(dob, now)` and `completedCourses: profile.completedCourseIds.length`, passes to `<Tinker>` as props. SSR-safe — no JS needed for either.

---

## 6. Hero / landing ambient life

`DESIGN.md` §Decoration calls for "alive but calm" — max 5 floating decorations per hero, sparkle pulse on the apple, no carnival. Three additions, all CSS-only.

**Floating math symbols on `index.astro`.** New small Svelte component `components/brand/FloatingMath.svelte`, mounted `client:visible` inside the dark hero band. Renders 5 absolutely-positioned `<span>`s — `π ∫ ∂ Δ ∑` — in Fraunces italic, palette `--ink-teal | --ink-pink | --ink-orange` (no purple, no red — red is reserved for the apple). Each has two CSS animations layered: `drift-y` (4.6–7.2s sine bob, ±14px) and `drift-x` (8–11s horizontal sway, ±8px), both `ease-in-out infinite` with staggered `animation-delay` so the field never pulses in unison. Opacity 0.18–0.35 per `DESIGN.md` §Decoration ambient range. Positioned via `inset` rules — not random JS — so SSR is identical to hydration. `prefers-reduced-motion: reduce` → freeze (`animation-play-state: paused`).

**Sparkle pulse polish around the apple.** The three sparkles are baked into `/logo-mark.png` (bitmap, can't pulse the alpha). Solution: overlay three SVG 8-point stars at known offsets matching the baked sparkles, in the same teal/orange/red trio, each with a new `sparkle-flicker` keyframe (opacity 0.0 → 1.0 → 0.0, 4.6s, staggered delays 0s/1.5s/3s). Adds `<svg class="tinker-sparkles">` as a sibling to `<img class="tinker-img">` inside `Tinker.svelte`. The baked PNG sparkles stay visible underneath as the floor — the overlay modulates apparent brightness.

**Sleepy yawn after long inactivity.** New idle state for `Tinker.svelte`. A document-level `pointermove`/`keydown`/`scroll` listener resets a 60s timer; on expiry, set `sleeping = true`. Visual: a `<span class="tinker-z">` containing a Fraunces-italic "z" appears above the apple, drifting up and fading via `tinker-z-rise` keyframe (translateY -28px, opacity 1 → 0, 2.4s, infinite). The bob slows to 7s. Any input → `sleeping = false`. Reduced motion → no z, no slow-bob change; sleeping is just a visual freeze (apple unchanged). `DESIGN.md` §Mascot already lists "sleeping (loading / idle)" as a planned expression — this is the implementation.

**Two new global.css keyframes.** `sparkle-flicker`, `tinker-z-rise`. The `drift-x`/`drift-y` keyframes live local to `FloatingMath.svelte` since they're not reused.

---

## 7. XP counter + streak chip polish

`DESIGN.md` §Components — current state: "Promote to a standalone Svelte component if it grows beyond what fits in Nav." This is that moment.

**`components/nav/XpCounter.svelte` (promoted from Nav inline).**

```ts
import { Tween, prefersReducedMotion } from 'svelte/motion';
import { cubicOut } from 'svelte/easing';

let { initial = 0 }: { initial: number } = $props();
const display = new Tween(initial, { duration: 900, easing: cubicOut });
const shown = $derived(Math.round(display.current));
let floaters = $state<{ id: number; delta: number }[]>([]);

$effect(() => {
  const onXp = (e: CustomEvent<{ delta: number; total: number }>) => {
    display.target = e.detail.total;
    floaters = [...floaters, { id: Date.now(), delta: e.detail.delta }];
  };
  window.addEventListener('tinker:xp', onXp as EventListener);
  return () => window.removeEventListener('tinker:xp', onXp as EventListener);
});
```

Template renders `XP {shown.toLocaleString()}` in Space Grotesk + `font-variant-numeric: tabular-nums` per `DESIGN.md` §Typography rule. Each floater renders via `{#each}` with `transition:fly={{ y: -28, duration: 700, easing: cubicOut }}` + `transition:fade={{ duration: 700 }}` (combined), in `--ink-sun` (XP color per `DESIGN.md`). Outro completes → `floaters = floaters.filter(f => f.id !== id)`. Reduced-motion fork: `display.set(value, { instant: true })`, floater becomes pure fade.

A11y: aria-live `<span class="sr-only">{shown}</span>` announces only the resting value (skip every intermediate `Math.round`).

**`components/nav/StreakFlame.svelte` (promoted, gated).**

Reads `streak` from a server-passed prop (`session.streak`). Hidden when `streak === 0` (`DESIGN.md` §Streak: opt-in only, off by default for first 30 days). Listens for `tinker:streak` events; on bump, fires a one-shot `flame-flicker` keyframe class for 800ms (custom keyframe in global.css: `filter: brightness()` and `scale` micro-pulse). The 🔥 emoji stays the visual; the flicker reads as life. Aria-live announces "Streak now {n} days" on bump only.

**Nav.astro changes.** Strip the inline XP/streak markup and the script-tag event listeners. Mount `<XpCounter client:load initial={session.xp} />` and `<StreakFlame client:load streak={session.streak} />` instead. The `client:load` matches `DESIGN.md` §Hydration table (persistent chrome, must work first paint).

**Why this matters.** Today's manual scale pulse + immediate number swap reads as "data updated." Tween + fly-floater reads as "you earned this." Same data, completely different perceived feel — the smallest surface in the design but probably the most-seen.

---

## 8. Architecture summary

**`package.json` delta.** One line:

```jsonc
"tw-animate-css": "^1"
```

Spring/Tween/`prefersReducedMotion` are already on the box (`svelte/motion`).

**`global.css` additions.** Six new keyframes (no new tokens):

- `sparkle-flicker` — 4.6s opacity pulse for sparkle overlay
- `tinker-z-rise` — sleepy "z" float-up
- `flame-flicker` — one-shot 800ms streak bump shimmer
- `sweep-green` — 640ms left-to-right gradient on lesson-complete card
- `halo-pulse` — green halo on correct StepCheck
- `wrong-shake` — 240ms 6-frame horizontal shake on wrong answer

`drift-x`/`drift-y` (FloatingMath) and `tinker-yawn` (sleeping bob slowdown) live local to their components since they're not reused.

**File map.**

```
NEW
  apps/docs/src/lib/celebrate.ts                    ← orchestrators (lesson, module, correct, wrong)
  apps/docs/src/lib/easterEggs.svelte.ts            ← Konami, "tinker", state singleton
  apps/docs/src/components/lesson/StuckHint.svelte
  apps/docs/src/components/lesson/SrAnnouncer.svelte
  apps/docs/src/components/brand/FloatingMath.svelte
  apps/docs/src/components/nav/XpCounter.svelte     ← promoted from Nav inline
  apps/docs/src/components/nav/StreakFlame.svelte   ← promoted from Nav inline
  public/mascot/{sunglasses,party-hat,grad-cap}.png

MODIFIED
  apps/docs/src/components/brand/Tinker.svelte      ← accessories, sparkle overlay, sleeping, eggs subscription, prefersReducedMotion rune
  apps/docs/src/components/lesson/StepCheck.astro   ← call celebrateCorrect/celebrateWrong, start 30s stuck timer
  apps/docs/src/layouts/Lesson.astro                ← call celebrateLesson/Module, mount StuckHint + SrAnnouncer
  apps/docs/src/components/Nav.astro                ← strip inline, mount XpCounter + StreakFlame
  apps/docs/src/pages/index.astro                   ← mount FloatingMath in hero band
  apps/docs/src/styles/global.css                   ← six new keyframes
  apps/docs/package.json                            ← +tw-animate-css

REMOVED
  Inline XP/streak markup + listener script in Nav.astro
```

**Build order.** Phased, no calendar dates. Each step is independently shippable and reversible.

1. Add dep, wire global.css keyframes.
2. Build `lib/celebrate.ts` + `SrAnnouncer.svelte`.
3. Promote XpCounter + StreakFlame, swap into Nav.
4. Wire StepCheck to celebrateCorrect/celebrateWrong + stuck timer.
5. Wire Lesson endgame to celebrateLesson/celebrateModule.
6. Build FloatingMath, mount in hero.
7. Tinker accessory layer + sparkle overlay + sleeping state + `prefersReducedMotion` rune migration.
8. Build `easterEggs.svelte.ts`, wire Konami + "tinker", build StuckHint.
9. Reduced-motion audit across every surface.
10. Aria-live audit across every surface.
11. Deploy and verify against `learntinker.com` (not dev server).

---

## 9. Accessibility, reduced motion, testing

**Reduced-motion policy, surface by surface.** The rule: ambient stops, deliberate user-response stays.

| Surface | Ambient (off) | Deliberate (on) |
|---|---|---|
| Mascot | Bob, cursor-tilt, sparkle flicker, sleepy yawn, "z" rise | Click bounce, click math-burst (existing rule) |
| FloatingMath | All drift-x/y | n/a |
| Mid-lesson | Spring widget jiggle, halo overshoot | Sound, haptic, XP fly-floater becomes pure fade |
| Lesson/module complete | Spring overshoot, sweep-green, Tween XP tick | Sound, haptic, confetti is already a no-op |
| StuckHint | Slide-in from corner | Replaced with instant fade |
| XpCounter | Tween transition | Replaced with `display.set(value, { instant: true })` |
| Streak | Flame-flicker keyframe | n/a |

Single source of truth: `prefersReducedMotion.current` from `svelte/motion`. Every component reads from this — no per-component matchMedia. The reactive rune means OS toggles flip behavior live without reload.

**Aria-live mirror.** One `<SrAnnouncer>` mounted in `Lesson.astro`, listens for `tinker:announce` events. Every celebration calls `announce(msg)`:

- StepCheck correct: `"Correct."` (delta + total announced separately by XpCounter)
- StepCheck wrong: `"Try again. Read the hint."`
- Lesson complete: `"Lesson complete. Plus 20 XP."`
- Module complete: `"Module complete. Plus 100 XP. {nextModuleName} unlocked."`
- StuckHint: `"Hint: {hint}"`
- XpCounter: announces *resting* value only — never every Math.round frame
- Streak: `"Streak now {n} days"` on bump only

**Mascot a11y.** Tinker stays `role="button" tabindex="0"` because it's an interactive easter-egg trigger (not purely decorative — it does something on click). Accessory overlays (sunglasses, party hat, grad caps) wrapped in `aria-hidden="true"` — they convey no state-changing affordance.

**Photosensitive (WCAG 2.3.1, Level A).** Confetti particles are well below the 25%-of-viewport threshold. No full-screen red — wrong answers use `--ink-coral` localized border, not a full-bleed flash. Streak flame flicker is one-shot 800ms — well under 3 flashes/sec. No animation pairs saturated red with white.

**Testing.** Deploy to `learntinker.com` and verify there, not against `astro dev`. Manual checklist per surface: reduced-motion toggle on (System Settings → Display), confetti renders nothing; sound mute persists across reload; aria-live messages read by VoiceOver; iOS hardware mute silences sine palette (expected behavior); StuckHint throttle holds (no second poke within 2 min).

---

## 10. Open questions

Not blocking; resolve during implementation.

1. **Mascot accessory sprites.** `sunglasses.png`, `party-hat.png`, `grad-cap.png` — handmade, sourced from open-licensed art, or hand-drawn SVG inline? Cheapest is hand-drawn SVG to keep `DESIGN.md`'s "always the approved asset" rule honest (overlays don't replace the apple). No paid generation regardless.
2. **StuckHint copy.** The hint string comes from each step's `hint` prop, but most existing Math-for-ML steps don't have hints authored yet. Soft-fail when `hint` is empty — don't poke without a payload.
3. **Streak opt-in flow.** `DESIGN.md` gates display on a 30-day-then-prompt flow that hasn't shipped. `bumpStreak()` already runs in `lib/xp.ts`. Display stays gated `streak > 0` for now; the opt-in prompt is out of scope for this pass.
4. **Module-complete `nextModuleName`.** The aria-live announce string references an "unlocked" name. Source from the course manifest in `src/content/`. If unavailable, drop the second sentence — graceful degrade.

---

## Decisions snapshot

| Decision | Reason |
|---|---|
| Keep `lib/confetti.ts` math-symbol burst, do NOT swap to `canvas-confetti` | Math-symbol confetti is brand identity, locked in `DESIGN.md` Decisions Log 2026-04-24 pivot v2 ("make the confetti like math symbols and shit"). |
| Keep `lib/sound.ts` Web Audio sine palette, do NOT swap to Howler + Kenney | Tonal progression (660→880→1320 Hz) literally encodes "finishing." Zero bundle. Brand-load-bearing. |
| Add `Spring`/`Tween`/`prefersReducedMotion` from `svelte/motion` | Pure upside, no brand collision. Powers XP counter, progress bars, widget jiggle, hint slide-in. |
| Add `tw-animate-css` for ambient classes | Zero-JS pulse/wiggle/shimmer surfaces. Replaces `tailwindcss-animate` (deprecated 2025-03). |
| Migrate `Tinker.svelte` from manual `matchMedia` to the new `prefersReducedMotion` rune | Reactive rune flips live on OS toggle without reload; one source of truth across all components. |
| Promote `XpCounter` and `StreakFlame` from Nav-inline to standalone Svelte components | Tween/transition needs Svelte component context. Trigger met per `DESIGN.md` §Components: "promote when it grows beyond what fits in Nav." |
| Orchestrate celebrations in `lib/celebrate.ts` rather than inline in `Lesson.astro`/`StepCheck.astro` | Tuning the lesson-complete feel becomes a one-line edit in one file. |
