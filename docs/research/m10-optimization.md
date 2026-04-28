# Module 10 — Optimization: Research Brief

*Tinker: Machine Learning, Backpropagation, and AI — The Math*

> Deep-research output, saved verbatim. 5 MDX lessons converted from §6 and problem bank from §7 live under `apps/docs/src/content/lessons/`. Keystone widget `LossLandscapeNavigator.svelte` built per §4.1.

---

## 1. Concept dependency graph

1. **`scalar-loss-function`** — A smooth function `L: ℝⁿ → ℝ` that assigns a single real number ("cost") to a parameter vector. Prereqs: `m6-multivariable`.
2. **`gradient-as-steepest-ascent`** — The vector `∇L(w)` whose direction is the locally steepest ascent of `L` and whose magnitude is that slope. Prereqs: `m6-multivariable`, `scalar-loss-function`.
3. **`descent-direction`** — Any direction `d` with `d·∇L < 0`; `-∇L` is the canonical choice. Prereqs: `gradient-as-steepest-ascent`, `m7-linear-algebra` (dot product).
4. **`gd-update-rule`** — The iteration `w_{t+1} = w_t - η∇L(w_t)`; one step against the gradient scaled by learning rate `η`. Prereqs: `descent-direction`.
5. **`learning-rate-as-distance`** — `η‖∇L‖` is the *length* of the step on the parameter space, not just a number. Prereqs: `gd-update-rule`, `m7-linear-algebra` (norms).
6. **`step-size-regimes`** — Three behaviors as `η` grows: slow monotone convergence → oscillation around minimum → divergence. Prereqs: `learning-rate-as-distance`.
7. **`condition-number-and-anisotropy`** — Ratio `κ = λ_max/λ_min` of Hessian eigenvalues governs whether GD zig-zags. Prereqs: `m7-linear-algebra` (eigenvalues), `gd-update-rule`.
8. **`stochastic-gradient-estimator`** — A cheap unbiased estimate `ĝ = ∇L_B(w)` from a mini-batch `B` substituted for the full-data gradient. Prereqs: `gd-update-rule`.
9. **`batch-vs-sgd-vs-minibatch`** — Bias–variance tradeoff: full batch = zero variance but expensive; SGD = high variance, cheap; mini-batch is the sweet spot. Prereqs: `stochastic-gradient-estimator`.
10. **`noise-ball`** — With fixed `η`, SGD does not converge to a point but to a neighborhood whose radius ∝ `η·σ_g`. Prereqs: `stochastic-gradient-estimator`, `step-size-regimes`.
11. **`exponential-moving-average`** — Scalar recursion `ŷ_t = βŷ_{t-1} + (1-β)x_t`; a low-pass filter with effective window `≈ 1/(1-β)`. Prereqs: `m5-calculus` (recursions).
12. **`momentum-update`** — `v_t = βv_{t-1} + ∇L; w_{t+1} = w_t - ηv_t`; adds inertia that cancels orthogonal oscillation and accumulates along persistent directions. Prereqs: `exponential-moving-average`, `gd-update-rule`.
13. **`rmsprop-update`** — Per-coordinate adaptive step: divide update by `√(EMA of g²) + ε`. Prereqs: `exponential-moving-average`.
14. **`bias-correction`** — EMAs initialized at zero are biased toward zero early; dividing by `1-β^t` undoes the warmup transient. Prereqs: `exponential-moving-average`.
15. **`adam-update`** — Momentum on `g`, RMSProp on `g²`, both bias-corrected. Prereqs: `momentum-update`, `rmsprop-update`, `bias-correction`.
16. **`adamw-decoupled-decay`** — Weight decay applied to the weights, not to the gradient. Prereqs: `adam-update`.
17. **`lr-schedule-warmup`** — Linearly ramp `η` from near-zero to peak over the first `T_w` steps. Prereqs: `bias-correction`, `step-size-regimes`.
18. **`lr-schedule-cosine`** — `η_t = η_min + ½(η_max-η_min)(1+cos(π·t/T))`. Prereqs: `lr-schedule-warmup`.
19. **`lr-schedule-step`** — Piecewise-constant `η`, divided by 10 at hand-picked milestones. Prereqs: `lr-schedule-warmup`.
20. **`local-minimum`** — A point where `∇L=0` and Hessian is positive semidefinite. Prereqs: `m6-multivariable`.
21. **`saddle-point`** — `∇L=0` with indefinite Hessian. Combinatorially dominant in high-d. Prereqs: `m7-linear-algebra` (eigenvalues).
22. **`cliff-and-ravine-pathologies`** — Regions where gradient magnitude changes by orders of magnitude across short distances. Prereqs: `condition-number-and-anisotropy`.
23. **`gradient-clipping`** — Project `g` onto a ball of radius `c` when `‖g‖>c`. Prereqs: `cliff-and-ravine-pathologies`.
24. **`convergence-criterion`** — Stop when `‖∇L‖ < ε`, loss plateaus, or held-out metric stops improving. Prereqs: `gradient-as-steepest-ascent`.
25. **`overparameterized-regime`** — When `#params ≫ #data`, many zero-loss minima exist. Prereqs: `local-minimum`.

## 8. Endgame callback (selected — Candidate A)

> "Adam, you'll actually meet in Module 11, is assembled from parts you now own: SGD, an EMA of past gradients (momentum), an EMA of past squared gradients (RMSProp), and a bias correction to fix the cold start. The LR warmup you'll see is there to keep that cold start from blowing up the first few steps."

## 9. Sources (licensing summary)

**[ADAPT]** allowed:
- Distill *Why Momentum Really Works* (Goh 2017) — CC BY 4.0
- nanoGPT repo — MIT (adapt code + configure_optimizers walkthrough)
- nn-zero-to-hero notebooks — MIT (videos reference-only)
- PyTorch docs — BSD-3-Clause (doc adaptation allowed)

**[REFERENCE-ONLY]**:
- Adam paper (Kingma & Ba 2014) — arXiv license
- AdamW paper (Loshchilov & Hutter 2017) — arXiv license
- Dauphin et al. 2014 (saddle points) — arXiv license
- Wilson et al. 2017 (Adam's marginal value) — arXiv license
- Boyd & Vandenberghe *Convex Optimization* — Cambridge copyright
- Goodfellow/Bengio/Courville *Deep Learning* — MIT Press copyright
- Ruder blog — no explicit open license
- 3Blue1Brown videos — CC BY-NC-SA

Widgets re-derived from scratch in Svelte+Mafs. Prose original.

## 10. Trap mitigations carried into lesson design

1. **Adam built from pieces, not a monolith.** SGD → + momentum → + RMSProp → + bias correction. Full Adam box appears only after every piece is built.
2. **"Saddles dominate" earned, not sloganed.** Lesson 10.5 gives the combinatorial argument (at `d=1000`, local minima are `2^{-1000}` rare) before invoking the Dauphin result.
3. **LR as step distance.** Introduced in Lesson 10.1 with the draggable ruler; every later "shrink the LR" is "shrink the ruler."
4. **Convergence ≠ global minimum.** Named explicitly in 10.2 (noise ball) and 10.5 (overparameterized regime).
5. **Optimizer depends on landscape.** Every optimizer lesson ends with a race in the navigator. Wilson et al. counterexample in 10.4.
6. **Bias correction as scaling, not mystery.** `adamInternalsInspector` shows uncorrected bars tiny, corrected bars unit-scale from step 1.

---

**The 5 lessons built from §6:**
1. `minimize-a-function.mdx` — What does it mean to minimize a function?
2. `batches-and-noise.mdx` — Batches, stochastic, and the noise ball
3. `momentum.mdx` — Momentum: gradient descent with a memory
4. `rmsprop-and-adam.mdx` — Per-parameter steps: RMSProp, and then Adam
5. `schedules-and-pathologies.mdx` — Schedules, pathologies, and what nanoGPT actually does

**The widget built live (§4.1):**
- `LossLandscapeNavigator.svelte` — 3 landscapes (bowl, ravine, saddle), 2 optimizers (SGD, SGD+momentum) for v1. Adam + RMSProp + rosenbrock queued for follow-up.

**Widgets flagged as placeholders:**
- `oneDStepDial` (§4.2)
- `adamInternalsInspector` (§4.3)
- `scheduleDesigner` (§4.4)
- `saddleEscapeComparator` (§4.5)
- `gradientVsNoiseBall` (§4.6)
