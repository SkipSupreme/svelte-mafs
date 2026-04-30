# M18 — Capstone: Implementation Plan

Source of truth: `docs/research/m18-capstone.md` (Deep Research, 2026-04-29).
Module manifest: `apps/docs/src/content/modules/m18-capstone.md` (status: planned → shipped at end of slice 6).

## 0. The architectural calls (locked)

The research surfaced three decisions; user signed off on all three.

- **Architecture:** `n_layer=4, n_head=4, d_model=64, d_ff=256, T=64, vocab=65, tied unembedding, no biases, dropout=0.0` → **196{,}608 trainable parameters** by the M16 12d² rule (`12 · 64² · 4 = 196{,}608`). Plus `8{,}320` for the embedding (tied with unembed) and `4{,}096` for learned positional → **~209k total**, the honest "200k-parameter" number.
- **Runtime:** **Hand-written WGSL kernels.** ~26 kernels (13 forward, 13 backward+optimizer). The "you wrote it" pedagogical claim only lands if we actually wrote it. Reference: `0hq/WebGPT` (MIT) for kernel shapes.
- **Sequencing:** Engine-first. Three internal-only slices land working WGSL training before any lesson page goes public; three public slices ship the three lessons.

Trade accepted: the engine is bigger than any prior module's widget set. The pedagogy demands it.

## 1. Lessons (final, 3)

| # | Title | Slug | ~Min | Widgets |
|---|---|---|---|---|
| 18.1 | Push the button | `push-the-button` | 50 | runnerPanel, liveSampleStream, seedScrubber, lossCurvePathologyZoo |
| 18.2 | Now make it talk | `now-make-it-talk` | 30 | samplerKnobsPlayground |
| 18.3 | The credits roll | `the-credits-roll` | 18 | creditsRollPanel, liveSampleStream (frozen variant) |

Total: ~98 minutes. Capstone lessons are *experiences*, not new theory — most of the wall time is the learner watching their own model train.

## 2. Engine — WGSL kernel inventory

13 forward kernels, 13 backward + optimizer kernels. All `f32`, all portable, no `shader-f16`. Workgroup size 64 default; 16×16 for tiled matmul.

**Forward (10 ops × 4 blocks + 3 boundary = 43 dispatches/step):**
1. `embeddingGather` — token-id + position-id → `[B·T, d]` row
2. `layerNorm` — one workgroup per row, 2-pass reduce, `[B·T, d]`
3. `qkvMatmul` — fused `[B·T, d] × [d, 3d]`, tile 16×16
4. `causalSdpa` — `softmax((QK^T + mask)/√d_k) V`, one workgroup per `(b, head, t)`
5. `attnOutMatmul` — `[B·T, d] × [d, d]`
6. `residualAdd` — elementwise
7. `ffnMatmul1` — `[B·T, d] × [d, 4d]`
8. `gelu` — elementwise on `[B·T, 4d]`
9. `ffnMatmul2` — `[B·T, 4d] × [4d, d]`
10. `unembedding` — `[B·T, d] × [d, V]` (tied with embed; same buffer)

**Backward (every forward op needs one) + optimizer:**
11. `softmaxCrossEntropyBwd` — `dlogits[i] = (p[i] − onehot(y)[i]) / (B·T)` (the line the credits roll calls out)
12. `matmulBwd` — generic, used 4× in attention + 2× in FFN + once in unembedding (`dA = dC·B^T`, `dB = A^T·dC`)
13. `causalSdpaBwd` — softmax-Jacobian-via-`p ⊙ (dy − Σ p·dy)` then through the V and (Q,K) paths
14. `lnBwd` — 3-term in-shader form (research §5)
15. `geluBwd` — elementwise (use the smooth `0.5(1+tanh(...))` derivative, not the GELU-approx hack)
16. `residualAddBwd` — copy of gradient (collapsed into the next op's read)
17. `embeddingBwd` — scatter-add into the tied `wte` buffer (atomicAdd for the float32-emulated case, or per-token serial accumulate)
18. `gradClipReduce` + `gradClipScale` — global `‖g‖₂` reduction + elementwise rescale
19. `adamwStep` — one elementwise pass: read `(θ, g, m, v, t, lr, β₁, β₂, ε, λ)`, write `(θ', m', v')`

Engine TS layer (~250 lines): `Tensor` = `{buffer: GPUBuffer, shape, dtype:'f32'}`; `Parameter` registry; `Optimizer` holds `(m, v, t)` per param; `forward(model, x) → logits`; `backward(logits, y) → param.grad`; `step(optimizer)` runs grad-clip then adamw.

## 3. Widgets (5 unique, all custom for M18)

| Widget | Lessons | Implementation note |
|---|---|---|
| `runnerPanel` | 18.1 | Centerpiece. Start / Pause / Reset; seed input; 4 hyperparameter sliders (lr, batch, dropout, n_layer). Live: dual loss curve (Canvas, circular buffer), iters/sec via `performance.now()`, current iter, current LR, "Compiling shaders…" boot state, "Paused (tab in background)" badge wired to `Page Visibility API`. |
| `liveSampleStream` | 18.1, 18.3 | Scrolling `<pre>` log: `iter 0: …`, every K iters appends a fresh sample from the current weights. Annotates the iter-band with the expected qualitative phase (research §10 trap 4). 18.3 variant runs against frozen weights, no append timer. |
| `seedScrubber` | 18.1 | Twin runs side-by-side. Type a seed string → both panes start identical. Edit one character → curves diverge after iter ~3. The "byte-identical" claim made tangible. |
| `lossCurvePathologyZoo` | 18.1 | Six preset buttons that re-train under known pathologies (lr=10, no warmup, dropout=0.9, no zero_grad, overfit on 1k chars, NaN-injection at iter 5). Lays your live curve next to the labelled gallery from M13. |
| `samplerKnobsPlayground` | 18.2 | Frozen checkpoint loaded once. Sliders (τ, top-k, top-p), prompt textbox, regenerate button. Live next-token bar histogram for the cursor position with bars greying out under each truncation pass. |

`creditsRollPanel` is a one-shot Astro page rather than a widget — see slice 6.

The M16 `ParamBudgetPie` and M17 `LossCurveDoctor` are NOT reused: M18 wants live curves on a real running engine, not pre-recorded traces. Different widget contract.

## 4. Build order — six slices

Each slice = one or more commits, build + deploy + smoke-verify, push.

- **Slice 1 (this turn)** — Plan committed. Research already saved.
- **Slice 2 — Engine forward.** All 10 forward kernels + Engine TS layer. Smoke test: instantiate the model from a fixed seed, run one forward pass on a hand-crafted batch, assert logit shape and softmax sums to 1 within tolerance. Lives at `apps/docs/src/lib/m18/engine/`. No public route.
- **Slice 3 — Engine backward + training loop.** All 13 backward+optimizer kernels. Smoke test: train the actual `n=4, d=64` model on tiny-shakespeare-char headless (Node + `@webgpu/dawn` or Bun's WebGPU) for 2{,}000 iters, assert val NLL drops below 1.7. This is the gate — if this doesn't converge, nothing else ships.
- **Slice 4 — Determinism + save/load.** sfc32 + cyrb128 threaded through (data-loader, dropout-mask, weight-init, sampler). `.bin` format with 512-byte JSON header + Float32Array tail. Twin-run determinism test: two CLI runs with the same seed string produce byte-identical weight `.bin` files.
- **Slice 5 — Lesson 18.1.** Build `runnerPanel` + `seedScrubber` + `liveSampleStream` + `lossCurvePathologyZoo`. Wire to engine. Ship `/lessons/push-the-button/`. First publicly visible M18 commit.
- **Slice 6 — Lesson 18.2 + 18.3 + module shipped.** Train the reference checkpoint once on the slowest dev machine, ship the `.bin` as a static asset (~840 KB). Build `samplerKnobsPlayground`, ship `/lessons/now-make-it-talk/`. Build `creditsRollPanel` reading `apps/docs/src/lib/m18/engine/*` directly so the credits roll is the actual code. Ship `/lessons/the-credits-roll/`. Flip module manifest `status: planned → shipped`. Update m18 manifest's summary to reflect the locked architecture (n_layer=4, d_model=64, ~200k true).

Per memory rule: every slice deploys to learntinker.com and verifies before commit.

## 5. Pedagogical decisions (locked)

- **Endgame callback:** research candidate A — *"This is the entire course. You started in M5… There is no module after this one. There is the model, the artifact, and the next thing you decide to learn."* In all three lesson MDX files; in the module manifest.
- **3 lessons.** No further splits. The capstone *is* the experience of running the trainer; padding it with prose lessons would dilute the moment.
- **Reference checkpoint shipped with the page.** Lesson 18.2 must work even if the learner skipped 18.1 — the playground loads the canonical `.bin` if no local checkpoint exists. The "use your own checkpoint" affordance is additive.
- **No fp16 path.** `f32` only. Mention `shader-f16` in research-trap prose; do not implement.
- **No mobile path in v1.** WebGPU on iOS Safari is shipping but spotty (April 2026); detect, message gracefully, link to a desktop browser. Never silently fall back to CPU.
- **No service-worker corpus caching in v1.** The 1 MB tiny-shakespeare loads each visit; caching is an optimization for after the lessons ship.

## 6. Voice & cut rules (applied)

- "Capstone celebration" is a *register*, not a slop license. No emojis in lesson prose. The mascot stays in the chrome (per `DESIGN.md`).
- Lesson 18.1 step 1 frames the model honestly as 200k params vs ChatGPT's 1.7T — research trap 6, applied verbatim.
- The runtime decision is one sentence in step 1 of 18.1 ("we wrote ~400 lines of WGSL ourselves"), not a lesson step.
- Live samples annotated with their expected band ("iter 0 — uniform across 65 chars (this is correct)") so the gibberish-before-iter-200 phase is framed, not feared.
- The credits roll lists every module by number and one-word concept; resists the urge to editorialize.

## 7. What this plan does NOT do

- Does not ship a multi-checkpoint system (only one save slot, plus the reference checkpoint).
- Does not implement KV-caching for the in-tab sampler — inference is fast enough at this scale that a fresh forward pass per token is fine, and re-implementing KV-cache correctness in WGSL costs more than the inference time it saves at d=64.
- Does not include mech-interp tooling. The "where to next?" callout in 18.3 *links to* a future track but doesn't seed any of it.
- Does not stub MDX files. Lesson files land when their engine slice and widgets land.

## 8. Done definition

All three lessons live on `learntinker.com`, all five widgets implemented as Svelte components, the engine compiles 26 WGSL kernels and trains the reference model to val NLL < 1.7 in under 5 minutes on an M-series Mac, twin-seed determinism passes byte-identically, the m18 manifest is `status: shipped` with the corrected `n_layer=4, d_model=64` summary, CI green, the course's final lesson endpoint matches the endgame callback to the character.
