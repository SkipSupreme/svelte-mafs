# Deep-Research Prompt Pack — Tinker Course Content

> **Purpose:** Batch-commission the raw pedagogical material for every unwritten module. Fire one prompt per module into Deep Research (ChatGPT, Claude, Gemini, Perplexity — any). Paste the output into this repo; Claude Code converts it into MDX lessons + Svelte widgets.

## Workflow

1. **Fire** one per-module prompt into Deep Research.
2. **Save** the response to `docs/research/m{N}-{slug}.md` verbatim.
3. **Queue** a Claude Code session: "Convert `docs/research/m7-linear-algebra.md` into the 4–6 MDX lessons it describes, following our existing Step / StepCheck / EndgameCallback patterns."
4. I generate lesson drafts + widget specs. Human edits for voice.
5. Ship the module.

Per-module work units: one research fire, one conversion pass, one editing pass, one deploy. Independent per module — multiple can run in parallel (one research tool per module, Claude Code converts them as results land).

---

## What Deep Research can't do

- **Our voice.** Research outputs read like surveys. We need tight, opinionated, hungover-adult-friendly prose. I do this pass.
- **Our widgets.** Research describes interactions; it can't build a Svelte+Mafs component that drags a vector and updates a dot product. That's me + the svelte-mafs library.
- **The narrative spine.** "This IS `loss.backward()`." "This IS attention." The endgame callback threaded through every module is ours to enforce.
- **License gatekeeping.** Research will happily cite Khan Academy (CC BY-NC-SA) or Wikipedia (CC BY-SA). We reject both for commercial content. The prompt below locks this down.

---

## Master template

Paste this template into Deep Research, then replace every `{{...}}` token with the per-module block below. Everything else stays verbatim.

```
# Research brief: Tinker Module {{ORDER}} — {{TITLE}}

## Role

You are a world-class mathematics educator and interactive-pedagogy researcher. You have read Bret Victor's "Learnable Programming," Nicky Case's essays on explorable explanations, the full Distill.pub archive, every 3Blue1Brown Essence-of-X series, and Andrej Karpathy's "Neural Networks: Zero to Hero." You understand the difference between a "boring slider" (a labeled number whose only purpose is to wiggle) and a direct-manipulation widget where the learner is grabbing the mathematical object itself.

## Context

I am building an interactive online course called Tinker: "Machine Learning, Backpropagation, and AI — The Math." It takes adult technical learners (think: software engineers who took calc a decade ago and want to actually understand ML from the math up) from pre-algebra to training a tiny character-level transformer in the browser via WebGPU.

The course has 18 modules across 5 arcs. Every module must pay back to the transformer endpoint via a 1–2 sentence "Where this shows up in the transformer" callback.

Stack: Astro 6 + Svelte 5 + MDX + a custom Svelte+Mafs widget library + KaTeX for math. Each lesson is 8–15 gated steps; some steps are prose-with-widget, some are StepCheck (numeric-answer gate before advance). The closest thing to a model of what a "good" lesson looks like is 3Blue1Brown's visual rigor married to Karpathy's build-something-from-scratch concreteness.

## Module under research

- **Module order:** {{ORDER}}
- **Module title:** {{TITLE}}
- **Arc:** {{ARC}}
- **Assumed prior knowledge (already taught earlier in the course):** {{PRIOR}}
- **Must-cover concepts:** {{CONCEPTS}}
- **Immediate next module:** {{NEXT_MODULE}}
- **Endgame callback (the one-liner at the bottom of every lesson in this module):** {{ENDGAME}}

## Deliverables — return ONE response with exactly these 10 sections, in order. No preamble.

### 1. Concept dependency graph
10–25 specific concepts this module must cover, topologically sorted.
For each: concept id (kebab-case), one-sentence definition, direct prerequisites (from this module or earlier — use module ids like `m5-calculus`).

### 2. Canonical worked examples
6–10 worked examples that are the "default teaching examples" for these concepts across MIT OCW, Stanford CS courses, Karpathy's videos, 3B1B, and Mathematics for Machine Learning (Deisenroth/Faisal/Ong).
For each: problem statement (LaTeX-ready), step-by-step solution, the pedagogical point it makes, the single most common student mistake on this example.

### 3. Common misconceptions
8–12 specific wrong mental models adult learners arrive with for this topic.
For each: the misconception in one sentence, why it's natural to fall for, and the specific counterexample, reframe, or visualization that kills it.

### 4. Interactive widget suggestions
4–7 widget ideas specific to this module's concepts.
For each:
- **Name** (camelCase, ≤25 chars)
- **What the user manipulates directly** (drag what, scrub what, click what)
- **What updates live** (specific readouts, visual changes)
- **What concept it makes tangible** (the pedagogical moment)
- **Why it beats a slider + static figure** (the Bret Victor / Nicky Case test)
- **Prior art** — 1–3 existing widgets on the web that do something similar (URLs if known)

Reject any idea that reduces to "drag a slider, watch an unlabeled number change." Every widget must let the learner grab a mathematical object the lesson has already named.

### 5. Key formulas
Every formula a student must recognize and/or reproduce.
Return as LaTeX source strings ready to drop into MDX (e.g. `\frac{\partial f}{\partial x}`). No commentary per formula — just the list, grouped by concept.

### 6. Lesson decomposition
Propose 3–6 lessons this module should split into. For each lesson:
- **Title** (conversational, not textbook — e.g. "What is a derivative?" not "1.1 Introduction to differentiation")
- **Summary** (one sentence, learner-facing)
- **Step outline** (8–15 gated steps, each a title + one-line description; note which steps are StepCheck with the expected scalar answer)
- **Widgets it uses** (refs to section 4)
- **Estimated minutes**

### 7. Problem bank
20 assessment problems ranging novice → advanced.
For each:
- Statement (LaTeX-ready)
- Expected answer (scalar, symbolic, or multi-part)
- Difficulty tier: novice / intermediate / advanced
- Cognitive type: computation / interpretation / construction / debugging / proof-scaffold
- Concept tags (from section 1)

### 8. Endgame callback — refined
Propose 1–3 candidate one-liners for the module's endgame callback. I supplied a starter; give me better options.

### 9. Sources (licensing-aware)
5–10 best external resources for this module.
For each: title, author, URL, medium (textbook / paper / video / blog / widget), license if known, and what specifically to use it for (worked examples / pedagogy reference / visualization inspiration).

**CRITICAL LICENSE RULES:**
- We are a **paid commercial product**.
- **DO NOT** recommend CC BY-NC, CC BY-NC-SA, CC BY-SA, or unlicensed content for **incorporation or adaptation**.
- Khan Academy content: CC BY-NC-SA → reference-only, never adapt.
- Wikipedia / Wikibooks: CC BY-SA → reference-only.
- MIT OCW: CC BY-NC-SA → reference-only.
- **OK to adapt:** CC BY, public domain, US federal government, OpenStax books that are CC BY (verify per-book — some OpenStax are NC-SA).
- Every source must be tagged **[ADAPT]** (we can reuse) or **[REFERENCE-ONLY]** (inspiration, don't embed).

### 10. Pedagogical traps
4–6 specific things that commonly go wrong when teaching this topic to adult self-learners.
For each: the trap, why it happens, and the concrete lesson-design mitigation.

## Style constraints

- Be opinionated. If two standard approaches exist, recommend one and say why.
- Target an adult technical audience (software engineers, engineers, scientists). High-school math is a distant memory but not a wall.
- Prefer construction and manipulation over multiple choice.
- Every widget must pass the "could this be a slider with a number on it?" test. If the answer is yes, reject it.
- Markdown-only. No tables requiring complex rendering.
```

---

## Per-module parameter blocks

Fire one at a time. Paste the master template above, then replace `{{...}}` tokens with the block below.

### M6 — Multivariable Calculus — Partial Derivatives, Gradients, Jacobians

```
ORDER: 6
TITLE: Multivariable Calculus — Partial Derivatives, Gradients, Jacobians
ARC: Arc 1 — Prerequisite Math
MINUTES: 150
PRIOR: Single-variable derivatives, chain rule, power rule (m5-calculus). Vectors as lists of numbers and dot products (m7-linear-algebra).
CONCEPTS: functions of multiple variables, partial derivatives, gradient, directional derivative, level sets, Jacobian, chain rule for composed multivariable functions, Hessian (intuition only)
NEXT_MODULE: m7-linear-algebra (linear algebra — vectors, matrices as transformations, matrix multiplication)
ENDGAME: loss.backward() computes the gradient of loss with respect to every parameter. That's the Jacobian-vector product you're about to implement by hand.
```

### M7 — Linear Algebra

```
ORDER: 7
TITLE: Linear Algebra
ARC: Arc 1 — Prerequisite Math
MINUTES: 240
PRIOR: Algebra and function composition (m2-algebra). Can be taken in parallel with multivariable calculus (m6).
CONCEPTS: vectors as arrows and lists, vector addition and scaling, dot product (as similarity and projection), matrices as linear transformations of space, matrix-vector multiplication as linear combination of columns, matrix-matrix multiplication as composition, determinant geometrically (signed area/volume), eigenvalues and eigenvectors, singular value decomposition (intuition: best low-rank approximation), change of basis
NEXT_MODULE: m8-probability and m10-optimization (both depend on linear algebra)
ENDGAME: A transformer is x @ W_q, x @ W_k, x @ W_v, then dot products through softmax, then more matrix multiplies. The attention matrix is a dot-product similarity matrix. Linear algebra is 80% of the transformer.
```

### M10 — Optimization

```
ORDER: 10
TITLE: Optimization
ARC: Arc 2 — ML Foundations
MINUTES: 150
PRIOR: Single-variable derivatives (m5), gradients (m6), linear algebra (m7).
CONCEPTS: minimizing a scalar function, gradient descent, learning rate as step size, batch vs stochastic vs mini-batch gradient descent, momentum, RMSProp, Adam, learning-rate schedules (warmup, cosine decay), loss-landscape pathologies (saddle points, local minima, cliffs), convergence criteria
NEXT_MODULE: m11-neural-networks (applying optimization to learn MLP parameters)
ENDGAME: Adam is the optimizer you'll actually use to train the transformer. The LR warmup schedule, the betas, the epsilon — all of it has reasons, and you'll understand them.
```

### M11 — Neural Network Fundamentals

```
ORDER: 11
TITLE: Neural Network Fundamentals
ARC: Arc 2 — ML Foundations
MINUTES: 150
PRIOR: Linear algebra (m7), optimization (m10). Light exposure to the chain rule (m5).
CONCEPTS: perceptron, linear layer with bias, activation functions (sigmoid, tanh, ReLU, GELU), single-layer limitations (the XOR problem), multilayer perceptrons (MLPs), forward pass as a stack of (matmul + nonlinearity), universal approximation theorem (intuition only, no proof), decision boundaries
NEXT_MODULE: m12-backpropagation (how MLPs actually learn their weights)
ENDGAME: The transformer's feed-forward block is literally an MLP. Everything you learn here plugs directly into the transformer architecture two modules from now.
```

### M12 — Backpropagation from Scratch (THE KEYSTONE)

```
ORDER: 12
TITLE: Backpropagation from Scratch
ARC: Arc 2 — ML Foundations
MINUTES: 240
PRIOR: Chain rule (m5-calculus), gradients (m6-multivariable), linear algebra (m7), MLPs (m11-neural-networks).
CONCEPTS: computational graph (nodes = values, edges = ops), forward pass as graph evaluation, local derivatives at each node, reverse-mode automatic differentiation, chain rule applied across the graph, building micrograd scalar-by-scalar (Karpathy-style), vectorizing from scalars to tensors, gradient checking (numerical vs analytical), common backprop bugs (missing .zero_grad(), in-place ops, detached tensors)
NEXT_MODULE: m13-training-dynamics (stabilizing the training of deep nets — BatchNorm, dropout, residuals)
ENDGAME: You just built micrograd. Every deep-learning framework on earth is a fancier version of what you now understand end-to-end. loss.backward() is no longer magic — it's your code.
```

---

## Widget research template (fire per interactive widget)

Some widgets in the design doc are their own pedagogical research project. Use this shorter template for those.

```
# Widget research: {{WIDGET_NAME}}

## Role
(same role block as the module template)

## The widget
I am building an interactive Svelte + Mafs widget called {{WIDGET_NAME}} for Module {{ORDER}} of Tinker. Its job is to make the following pedagogical moment tangible: {{MOMENT}}.

## Deliverables — return these 6 sections, nothing else

### 1. Prior art
5–8 existing interactive widgets on the web that do something similar. For each: URL, what it lets the user manipulate, what it does well, what it does badly, whether it's free to clone (MIT/CC BY) or reference-only.

### 2. Design variants
4–6 different ways this widget could be designed. For each:
- **Interaction surface** — what the learner drags, clicks, scrubs
- **Live updates** — what visual state changes in response
- **Failure mode** — the "boring slider" degenerate case this variant risks
- **Strength** — what this variant teaches that others don't

### 3. Pedagogical pitfalls
Specific things that go wrong when learners use widgets of this kind. Include the Bret Victor criticism ("adjust mystery numbers") and Nicky Case's "sandbox without scaffolding" failure mode where applicable.

### 4. Accessibility
Keyboard navigation plan, screen-reader labels, color-blind-safe alternatives, motion-reduction fallback.

### 5. Recommended variant
Pick ONE. Say why. Include a rough storyboard: initial state → first interaction → aha moment → final state.

### 6. Parametrization
Exact numerical defaults — viewBox ranges, slider min/max, default parameter values — that produce the intended pedagogical moment on load.
```

---

## Batch queue (priority order)

Fire in this order. MVP spine first — everything in Tier 1 blocks the capstone.

**Tier 1 — MVP spine:**
1. M6 Multivariable Calculus
2. M7 Linear Algebra
3. M10 Optimization
4. M11 Neural Network Fundamentals
5. M12 Backpropagation (keystone)

**Tier 2 — fill out the arcs:**
6. M13 Training Dynamics
7. M14 Sequence Modeling (bigrams + RNNs)
8. M15 Attention
9. M16 Transformer Block

**Tier 3 — capstone + completeness:**
10. M17 Tokenization & Sampling
11. M18 Capstone — Train Your Transformer
12. M8 Probability & Statistics
13. M9 Information Theory

**Deferred** (safety-floor modules — route around with a diagnostic instead of teaching linearly):
- M0, M1, M2, M3, M4

---

## Post-research workflow

When Deep Research returns, save to `docs/research/m{N}-{slug}.md` and start a Claude Code session with:

```
Read docs/research/m7-linear-algebra.md. Convert sections 6 (lesson decomposition) and 7 (problem bank) into MDX lessons under apps/docs/src/content/lessons/, following the existing Step / StepCheck / EndgameCallback pattern used in what-is-a-derivative.mdx and the-chain-rule.mdx. For each lesson, propose any new Svelte widgets we need and write stubs for them (I'll fill in the svelte-mafs integration). Update the m7-linear-algebra.md module file's status from `planned` to `drafting`. Deploy.
```

Expect 3–6 lessons per module and 1–3 new widgets per module. Per-module cycle: one research fire → one conversion session → one editing pass → deploy.
