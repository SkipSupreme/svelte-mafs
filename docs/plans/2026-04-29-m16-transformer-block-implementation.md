# M16 — The Transformer Block: Implementation Plan

Source of truth: `docs/research/m16-transformer-block.md` (Deep Research, 2026-04-29).
Module manifest: `apps/docs/src/content/modules/m16-transformer-block.md` (status: drafting).

## 0. The big architectural call

The research asks for several widgets that read like "tiny pre-trained model running in the browser" — `residualStreamScope`, `directPathInspector`, `ablationLab`. Building a real char-level GPT in WebGPU/WASM is M18 Capstone territory and out of scope for this module.

**Decision: scripted activations, not a real model.** Each widget uses pre-computed deterministic vectors, hand-tuned to demonstrate the pedagogical point (per-block deltas with visible structure, the bigram floor at N=0, ablation that visibly worsens NLL on a fixed sequence). The pedagogy lands on *composition rules* — sum-of-deltas, where LN goes, how params scale — not on model fidelity. Revisit when M18 ships and we have a real tiny GPT.

Trade-off: the user can't ablate "their own" model and watch real generations. They can manipulate a deterministic walkthrough that shows the principle. Same trade we made in M15's `AttentionMatrixHeatmap` (hand-tuned scores).

## 1. Lessons (final, 5)

| # | Title | Slug | ~Min | Widgets |
|---|---|---|---|---|
| 16.1 | One block, top to bottom | `one-block-top-to-bottom` | 15 | residualStreamScope (read-only), paramBudgetPie |
| 16.2 | Why pre-LN | `why-pre-ln` | 14 | lnPlacementGrad, buildTheBlock |
| 16.3 | The residual stream as the object | `residual-stream-as-the-object` | 18 | residualStreamScope, directPathInspector |
| 16.4 | Stacking N and the full forward pass | `stacking-n-and-the-full-forward-pass` | 16 | ablationLab |
| 16.5 | Counting the cost | `counting-the-cost` | 12 | paramBudgetPie |

Total: ~75 minutes. Matches research's estimate exactly.

## 2. Widgets (6 → 4 unique components)

| Widget | Lessons | Implementation note |
|---|---|---|
| `paramBudgetPie` | 16.1, 16.5 | Pure parametric: drag d_model / N / V / FFN ratio / tying toggle, render a sunburst slicing total params into {tied embedding, attention QKVO, FFN, LN, positional, LM head}. GPT-2-small / medium / GPT-3 presets. |
| `residualStreamScope` | 16.1, 16.3, 16.4 | Renders a 2D grid (T positions × N+1 layers). Click a cell to see the residual-stream vector at that position/depth as a horizontal bar of d cells. Decomposition toggle: raw stream / per-block-Δ contributions / direct path. Hand-scripted activations for a tiny T=5, d=32, N=4 walkthrough. |
| `lnPlacementGrad` | 16.2 | Two superimposed gradient-norm-vs-depth curves (pre-LN, post-LN), depth slider 1–48. Use Xiong 2020's analytical bounds (Θ(d/√L · ln L) vs Θ(d · ln L)) directly — no training simulation needed for the pedagogical point. Mark depth at which post-LN exceeds a "warmup needed" threshold. |
| `buildTheBlock` | 16.2 | Drag-drop palette of {LN, MHA, FFN, residual-add, dropout} into a block schematic. DAG validator labels result as ✓ pre-LN / ✓ post-LN / ✗ broken (with the specific structural failure named: "trunk passes through LN", "missing residual on second sub-layer", etc.). |
| `directPathInspector` | 16.3 | Block-budget slider 0..N. Shows model output with first k blocks active, rest replaced by identity. At k=0 the prediction is the bigram floor `softmax(W_E[t] · W_E^T)`. Displays current top-5 next-token guesses. Scripted activations on a small char-level vocab. |
| `ablationLab` | 16.4 | Stack diagram, click any sub-layer (head / FFN / LN) to ablate (zero its delta). NLL-on-shakespeare-passage updates from a pre-recorded table of (ablation set → NLL) values. Per-token loss heatmap. |

`residualStreamScope` and `paramBudgetPie` are reused; build them first.

## 3. Build order

1. **15.5 → 15.6** in M15 used the `paramBudgetPie` reuse pattern (same widget, two lessons). Same here:
   - **Slice 1 (16.1)**: build `paramBudgetPie` and `residualStreamScope` (read-only mode); ship lesson 16.1.
   - **Slice 2 (16.2)**: build `lnPlacementGrad` and `buildTheBlock`; ship lesson 16.2.
   - **Slice 3 (16.3)**: extend `residualStreamScope` to exploration mode (decomposition toggle); build `directPathInspector`; ship lesson 16.3.
   - **Slice 4 (16.4)**: build `ablationLab`; ship lesson 16.4.
   - **Slice 5 (16.5)**: ship lesson 16.5 (reuses `paramBudgetPie`); flip module status to shipped.

Each slice = build/deploy/commit/push on main. Per-slice commits, two if widget + lesson are conceptually distinct (research/plan vs lesson, like the M15 first slice).

## 4. Pedagogical decisions baked in

Locked from the research's recommendations and my prior plan-decisions pattern:
- **Endgame callback**: research option #1 — *"Tokens at the bottom. Attention and MLPs in the middle. Unembedding at the top. Bigram NLL on the outside. Stack N, train with Adam, warm up the LR. That is GPT — there is nothing else."* Already in the module manifest.
- **5 lessons, no further splits.** Research's decomposition is already balanced (avg 12 steps, 2 widgets max per lesson).
- **Pre-LN as default**, post-LN as the "original 2017 variant" recap. Match the modern world.
- **Weight tying assumed throughout.** Note the untied case in problem-bank only.
- **No dropout in widget visuals.** Mention placement in prose; don't render it as a separate visual.

## 5. Voice & cut rules (applied)

- Cut "from m13" / "from m15" callbacks down to single-sentence pointers; do not re-explain the primitive.
- No "Hook:" framings on step 1. Open with the problem statement directly.
- No "Aside:" labels. Asides become normal paragraphs.
- The Anthropic residual-stream framing is the *primary* lens for 16.3, not a side gloss. Lead with it.

## 6. What this plan does NOT do

- Does not build a real char-level GPT. (See §0.)
- Does not stub MDX files. Lessons land when their widget ships.
- Does not include a separate skill-tree update commit — module manifest's `status: drafting → shipped` flip ships in slice 5.

## 7. Done definition

All five lessons live on `learntinker.com`, all six widget specs realized as four unique Svelte components, m16 manifest status flipped to `shipped`, CI green, two commits per slice (widgets + lesson) merged to main.
