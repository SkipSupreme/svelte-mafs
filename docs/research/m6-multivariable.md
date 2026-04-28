# Module 6 Research Brief — Multivariable Calculus: Partial Derivatives, Gradients, Jacobians

> Deep-research output, saved verbatim. Conversion log in `../plans/2026-04-24-research-prompts.md`. See the `m6-calculus` lesson files under `apps/docs/src/content/lessons/` for what we built from it.

## 1. Concept dependency graph

Topologically sorted. Every concept lists direct prerequisites from this module (mX.Y) or earlier (`m5-calculus`, `m0-notation`). Vectors are treated as *tuples of numbers* only; matrix fluency is explicitly deferred to `m7-linear-algebra`.

1. **`multivar-function`** — A rule that eats a tuple of numbers $(x_1,\dots,x_n)$ and returns a single number. Prereqs: `m5-calculus` (notion of function).
2. **`graph-of-two-var`** — The surface $z=f(x,y)$ in 3-space, plus the "heightmap" mental model. Prereqs: `multivar-function`.
3. **`contour-level-set`** — The set $\{(x,y) : f(x,y)=c\}$; a horizontal slice of the graph, a.k.a. a level curve. Prereqs: `graph-of-two-var`.
4. **`slice-function`** — Freezing all but one input yields a single-variable function whose ordinary derivative is meaningful. Prereqs: `multivar-function`, `m5-calculus`.
5. **`partial-derivative`** — $\partial f/\partial x_i$: the derivative of the slice obtained by holding all other inputs constant. Prereqs: `slice-function`.
6. **`partial-notation`** — The three notations ($\partial f/\partial x$, $f_x$, $D_x f$) refer to the same object; subscript index form $\partial_i f$. Prereqs: `partial-derivative`.
7. **`second-partials-clairaut`** — Mixed partials $f_{xy}$ and $f_{yx}$ agree when continuous (Clairaut/Schwarz). Prereqs: `partial-derivative`.
8. **`tangent-plane-linearization`** — Near $(a,b)$, $f(x,y)\approx f(a,b)+f_x(a,b)(x-a)+f_y(a,b)(y-b)$. Prereqs: `partial-derivative`, `m5-calculus` (linear approximation).
9. **`gradient-vector`** — The tuple $\nabla f=(\partial f/\partial x_1,\dots,\partial f/\partial x_n)$ bundling all partials. Prereqs: `partial-derivative`, `m0-notation` (tuples).
10. **`dot-product-refresher`** — $\mathbf{u}\cdot\mathbf{v}=\sum u_i v_i$ and the $|\mathbf{u}||\mathbf{v}|\cos\theta$ form. Prereqs: just-taught vectors.
11. **`directional-derivative`** — Rate of change along unit vector $\mathbf{u}$: $D_\mathbf{u}f(\mathbf{p})=\nabla f(\mathbf{p})\cdot\mathbf{u}$. Prereqs: `gradient-vector`, `dot-product-refresher`.
12. **`steepest-ascent`** — The gradient points in the direction of fastest increase; $-\nabla f$ is steepest descent; magnitude $|\nabla f|$ is the slope in that direction. Prereqs: `directional-derivative`.
13. **`gradient-perp-level`** — $\nabla f$ is perpendicular to the level set through $\mathbf{p}$ (follows from the chain rule applied to any curve $\mathbf{r}(t)$ on the level set). Prereqs: `gradient-vector`, `contour-level-set`, `multivar-chain-rule` (or preview of).
14. **`critical-point`** — A point where $\nabla f=\mathbf{0}$; candidates for local min/max/saddle. Prereqs: `gradient-vector`.
15. **`hessian-intuition`** — The table of second partials; positive-definite ⇒ bowl (min), negative-definite ⇒ cap (max), mixed-sign eigenvalues ⇒ saddle. Intuition only — no matrix mechanics. Prereqs: `second-partials-clairaut`, `critical-point`.
16. **`vector-valued-function`** — A function $\mathbf{F}:\mathbb{R}^n\to\mathbb{R}^m$, i.e., a tuple of $m$ scalar functions each taking $n$ inputs. Prereqs: `multivar-function`.
17. **`jacobian`** — The $m\times n$ grid of partial derivatives $J_{ij}=\partial F_i/\partial x_j$; row $i$ is the gradient of output $i$. Prereqs: `vector-valued-function`, `gradient-vector`.
18. **`multivar-chain-rule`** — If $z=f(\mathbf{y})$ and $\mathbf{y}=\mathbf{g}(\mathbf{x})$, then $\partial z/\partial x_j=\sum_i (\partial f/\partial y_i)(\partial y_i/\partial x_j)$; compactly, Jacobians multiply. Prereqs: `jacobian`, `m5-calculus` (single-var chain rule).
19. **`jacobian-vector-product`** — $J\mathbf{v}$ (forward mode: push a tangent through); $\mathbf{v}^\top J$ (reverse mode: pull a cotangent back). The operation that backprop is actually doing. Prereqs: `jacobian`, `multivar-chain-rule`.
20. **`loss-as-scalar-function`** — A neural net loss is $L:\mathbb{R}^P\to\mathbb{R}$ with $P$ = number of parameters; its gradient is a $P$-tuple and `.backward()` fills it in. Prereqs: `gradient-vector`, `jacobian-vector-product`.

## 8. Endgame callback — selected

> "You just built, by hand, the exact object that `loss.backward()` will quietly fill in for every parameter of a 50,000-parameter transformer. It's a vector–Jacobian product. There's no extra magic — there's just a lot of it."

## 9. Sources (licensing summary)

- **[REFERENCE-ONLY]** MIT OCW 18.02, Mathematics for Machine Learning Ch. 5, CS231n derivatives handout, Khan Academy videos, APEX Calculus, OpenStax Calculus Vol 3, 3Blue1Brown.
- **[ADAPT]** `karpathy/micrograd` (MIT license), Distill.pub (CC BY 4.0) — inspiration only, no direct embed.

## 10. Trap mitigations carried into lesson design

1. **Gradient ≠ derivative tangent.** Contour-view dominant, not surface.
2. **Gradient lives in input space.** Draw it there, always.
3. **Matrix-calculus notation deferred.** Tuples and explicit partials through Lesson 6.4.
4. **Numerator layout pledge.** Rows = output, columns = input.
5. **Chain rule as graph, not formula.** `chainRulePaths` before the sum formula.
6. **Vectors refresher required.** One explicit step on dot product / unit vector in Lesson 6.2.

---

(Full brief — sections 2, 3, 4, 5, 6, 7 — available from the Deep Research output log.
Conversion of §6 decomposition → MDX lessons and §7 problem bank → StepChecks is committed in this same push.)
