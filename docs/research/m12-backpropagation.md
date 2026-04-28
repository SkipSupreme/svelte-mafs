# Module 12 — Backpropagation from Scratch — Research Brief

## 1. Concept dependency graph

1. **computational-graph** — A directed acyclic graph whose nodes are intermediate values (including inputs, parameters, and the final scalar loss) and whose edges represent elementary operations that produced the child from its parents. *Prereqs:* m5-calculus, m6-multivariable, m11-neural-networks.
2. **forward-pass-as-graph-eval** — Evaluating the graph from leaf inputs toward the root by topologically walking each node and computing `node.data = op(parents.data)`. *Prereqs:* computational-graph.
3. **local-derivative** — The partial derivative of a node's output value with respect to each of its immediate parents, expressible from the op alone (e.g., for `z = x*y`, ∂z/∂x = y). *Prereqs:* m5-calculus, computational-graph.
4. **chain-rule-on-a-graph** — The multivariable chain rule reinterpreted as: "the derivative of the loss w.r.t. any node equals the sum over outgoing edges of (local derivae) × (loss-gradient of the downstream node)." *Prereqs:* m6-multivariable, local-derivative.
5. **upstream-vs-local-gradient** — The convention that at each node, `grad = upstream_grad * local_grad`, where "upstream" is dL/d(this node) and "local" comes from the op. Fixes the direction of information flow. *Prereqs:* chain-rule-on-a-graph.
6. **gradient-accumulation** — When a node has more than one outgoing edge (used multiple times), its total gradient is the *sum* over those paths, implemented in code as `self.grad += ...`. *Prereqs:* chain-rule-on-a-graph.
7. **reverse-mode-autodiff** — An algorithm that computes `dL/d*` for every node in one backward sweep by factoring the "sum over paths" so each edge is touched exactly once, starting from `dL/dL = 1` and moving parents-ward. *Prereqs:* chain-rule-on-a-graph, gradient-accumulation.
8. **forward-mode-autodiff** — Dual algorithm that propagates dual numbers forward from one input, computing one column of the Jacobian per sweep. Used as a contraeqs:* local-derivative.
9. **why-reverse-mode-for-dl** — For functions R^n → R^1 (i.e., loss functions), reverse mode costs O(forward pass) once, while forward mode costs O(n × forward pass). Neural-net losses have one output and millions of inputs, so reverse mode wins by a factor of n. *Prereqs:* reverse-mode-autodiff, forward-mode-autodiff.
10. **value-class** — The micrograd data structure: a Python object wrapping `.data` (scalar), `.grad` (scalar, initialized 0), `._prev` (parent Values), `._op` (string tag), and a closure `._backward` that knows how to distribute `self.grad` to its parents. *Prereqs:* computational-graph, local-derivative.
11. **per-op-backward-closures** — For each overloaded op (`__add__`, `__mul__`, `__pow__`, `tanh`, `exp`, `relu`), defining a `_backward` that reads `out.grad` and `+=` the correct local-derivative-scaled contribution into each parent's `.grad`. *Prereqs:* value-class, upstream-vs-local-gradient, gradient-accumulation.
12. **topological-sort** — Orderin so every node appears *after* all its children (for the reverse sweep: all its dependents). Built by DFS with a visited set; without it, diamond-shaped graphs give wrong gradients because a node's `_backward` runs before its downstream `.grad` is fully accumulated. *Prereqs:* computational-graph.
13. **backward-pass-implementation** — The top-level `loss.backward()`: build topo, set `self.grad = 1.0`, iterate topo in reverse calling each node's `_backward`. *Prereqs:* topological-sort, per-op-backward-closures.
14. **parameter-update-and-zero-grad** — Using computed gradients to update parameters via SGD/Adam, then explicitly resetting `.grad = 0` before the next forward pass, because `+=` would otherwise accumulate across iterations. *Prereqs:* backward-pass-implementation, m10-optimization.
15. **scalar-to-tensor-vectorization** — Repackaging an entire layer's scalars into a single ndarray op (e.g., `Y = X @ W + b`) for speed. The same chain rule applies, but now gradients are matrices. *Prereqs:* rd-pass-implementation, m7-linear-algebra.
16. **matmul-backward** — For `Y = X W`: `dL/dX = (dL/dY) W^T` and `dL/dW = X^T (dL/dY)`. Derivable by summing the scalar rule over batch and output indices. *Prereqs:* scalar-to-tensor-vectorization, m7-linear-algebra.
17. **elementwise-nonlinearity-backward** — For `Y = f(X)` applied elementwise: `dL/dX = dL/dY ⊙ f'(X)`. Includes `tanh' = 1 - tanh^2`, `sigmoid' = σ(1-σ)`, `relu' = [x>0]`. *Prereqs:* scalar-to-tensor-vectorization, m11-neural-networks.
18. **broadcasting-gradient** — When a tensor is broadcast in the forward pass (e.g., bias vector added to every row), its backward gradient must be summed along the broadcast axes so the shape matches the original. *Prereqs:* scalar-to-tensor-vectorization, gradient-accumulation.
19. **softmax-ce-fused-backward** — For `L = CE(softmax(z), y)` with one-hot `y`: `dL/dz = p - y`. The Jacobian of softmax times the gradient of CE collapses algebraically. *Prereqs:* matmul-backward, m11-neural-networks.
20. *ecking** — Verify analytical gradients against the centered finite-difference approximation `(f(θ+ε) − f(θ−ε)) / (2ε)` using relative error `||g_ana − g_num|| / (||g_ana|| + ||g_num||)`; expect < 1e-7 with ε≈1e-5. *Prereqs:* backward-pass-implementation.
21. **common-backprop-bugs** — Forgetting `.zero_grad()`, in-place ops corrupting saved-for-backward buffers, `detach()`/`.data` silently severing the graph, assigning (`=`) instead of accumulating (`+=`) in `_backward`, swapping `X` and `W` in matmul backward, broadcasting without un-broadcasting. *Prereqs:* backward-pass-implementation, broadcasting-gradient.
22. **autograd-as-tape** — Framing: modern libraries (PyTorch, JAX) do not do symbolic differentiation; they record a "tape" of ops during the forward pass and replay it in reverse, exactly like micrograd but with tensors. *Prereqs:* reverse-mode-autodiff, value-class.

## 2. Canonical worked examples

**Example 1 — The three-variable scalar graph (Karpathy + CS231n canonical).* (a+b) \cdot c$. Evaluate at $a=2, b=-3, c=10$.
Forward: $e = a+b = -1$; $f = e \cdot c = -10$.
Backward: seed $\frac{\partial f}{\partial f}=1$; at $f=e\cdot c$ → $\frac{\partial f}{\partial e}=c=10$, $\frac{\partial f}{\partial c}=e=-1$; at $e=a+b$ → $\frac{\partial f}{\partial a}=\frac{\partial f}{\partial e}\cdot 1 = 10$, $\frac{\partial f}{\partial b}=10$.
*Pedagogical point:* every `+` node copies the incoming gradient to both parents; every `*` node swaps and scales. Gives the learner the two irreducible local-derivative rules. [Source](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf)
*Most common mistake:* students forget that "local derivative times upstream" applies to *each parent* and just compute one; or they plug in $a$'s value instead of $c$'s into $\partial f/\partial a$.

**Example 2 — Karpathy's one-neuron derivation.**
Problem: $o = \tanh(w_1 x_1 + w_2 x_2 + b)$. Inputs $x_1=2, x_2=0, w_1=-3, w_2=1, b=6.8813735870195432$ (Karpathy's chosen constants so $o \approx 0..
Forward: $x_1 w_1 = -6$; $x_2 w_2 = 0$; $n = x_1w_1 + x_2w_2 + b \approx 0.8814$; $o=\tanh(n)\approx 0.7071$.
Backward: $\frac{\partial o}{\partial o}=1$; $\frac{\partial o}{\partial n} = 1-\tanh^2(n) = 1 - 0.5 = 0.5$; $\frac{\partial o}{\partial b}=0.5$; $\frac{\partial o}{\partial (x_1 w_1)} = 0.5$; $\frac{\partial o}{\partial w_1} = 0.5 \cdot x_1 = 1.0$; $\frac{\partial o}{\partial x_1} = 0.5 \cdot w_1 = -1.5$; similarly $\frac{\partial o}{\partial w_2}= 0$, $\frac{\partial o}{\partial x_2}=0.5$.
*Pedagogical point:* this is *the* canonical micrograd example — it composes `*`, `+`, and a nonlinearity and teaches "gradient flows backward through the neuron." [Source](https://github.com/karpathy/micrograd)
*Most common mistake:* computing $\partial o/\partial w_2 = 0.5 \cdot x_2 = 0$ and thinking "the gradient is broken" — it's correct, it's just that dead inputs produce dead weights this step.

**Example 3 — A node used twice (the `+=` demo).**
Problem: $f(a) = a \cdot a$ (equivalently $b = a+a$,  \cdot a$ if you want a richer graph). At $a=3$.
Forward: $f = 9$.
Backward via the graph: the `*` node sends gradient $a$ to *each* operand, but both operands are the same `Value` object. Correct: $\frac{\partial f}{\partial a} = 2a = 6$, achieved only if `_backward` does `a.grad += out.grad * a.data` *twice* (accumulating), not assigning.
*Pedagogical point:* this is the single step where `=` vs `+=` in `_backward` becomes a bug. Karpathy dedicates an entire chapter of the video to "fixing a backprop bug when one node is used multiple times." [Source](https://github.com/karpathy/micrograd)
*Most common mistake:* writing `a.grad = out.grad * b.data` in `__mul__`, which overwrites the first contribution; the final gradient is 3 instead of 6.

**Example 4 — Matrix-multiply backward (CS231n / Eli Bendersky canonical).**
Problem: $Y = X W$ with $X \in \mathbb{R}^{N\times D}$, $W \in \mathbb{R}^{D\times M}$. Given $\frac{\partial L}{\partial Y} \in \mathbb{R}^{N\times M}$, derive $\frac{\partial L}{\partial X and $\frac{\partial L}{\partial W}$.
Derivation: write $Y_{ij} = \sum_k X_{ik} W_{kj}$, so $\frac{\partial L}{\partial X_{ik}} = \sum_j \frac{\partial L}{\partial Y_{ij}} W_{kj}$, which is $\frac{\partial L}{\partial Y} W^\top$; similarly $\frac{\partial L}{\partial W_{kj}} = \sum_i X_{ik}\frac{\partial L}{\partial Y_{ij}} = X^\top \frac{\partial L}{\partial Y}$.
Result: $\boxed{\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top}$, $\boxed{\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}}$. [Source](https://cs231n.stanford.edu/handouts/linear-backprop.pdf) [Source](https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/)
*Pedagogical point:* learners don't need to derive this from Kronecker products — they just check shapes. "The only matrix product that produces the right shape is the right one." This is the gateway from scalar autodiff to tensor autodiff.
*Most common mistake:* swapping the operand order and writing $\frac{\partial L}{\ptial W} = \frac{\partial L}{\partial Y} X^\top$, which has the wrong shape but looks plausible.

**Example 5 — Softmax + cross-entropy collapse to $p - y$.**
Problem: logits $z \in \mathbb{R}^K$, softmax $p_i = e^{z_i}/\sum_k e^{z_k}$, one-hot target $y$, loss $L = -\sum_i y_i \log p_i$. Derive $\frac{\partial L}{\partial z}$.
Derivation: $\frac{\partial L}{\partial p_i} = -y_i/p_i$; $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$; chain & sum: $\frac{\partial L}{\partial z_j} = -\sum_i (y_i/p_i) \cdot p_i(\delta_{ij}-p_j) = -y_j + p_j \sum_i y_i = p_j - y_j$ (since $\sum_i y_i = 1$).
Result: $\boxed{\frac{\partial L}{\partial z} = p - y}$. [Source](https://parasdahal.com/softmax-crossentropy/) [Source](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
*Pedagogical point:* an ugly Jacobian collapses to the "prediction error." This is *why* cross-entropy is paired with softmax in every modern classifier, and why the transformer's output head is trained with exactly ts gradient.
*Most common mistake:* computing the Jacobian of softmax and the gradient of log separately and multiplying matrices — correct but painful, and a production bug trap (you want a single fused op for numerical stability via log-sum-exp).

**Example 6 — Broadcasting a bias vector.**
Problem: forward is $Y = XW + b$ with $X\in\mathbb{R}^{N\times M}$, $b\in\mathbb{R}^{M}$ broadcast across the batch. Given $\frac{\partial L}{\partial Y}\in\mathbb{R}^{N\times M}$, find $\frac{\partial L}{\partial b}$.
Answer: $\frac{\partial L}{\partial b} = \sum_{n=1}^N \frac{\partial L}{\partial Y_{n,\cdot}} \in \mathbb{R}^M$; i.e., `db = dY.sum(axis=0)`.
*Pedagogical point:* "broadcasting in the forward pass = reduce-sum in the backward pass" — a universal rule ([Source](https://gumran.github.io/mgp.html)). Every tensor autodiff framework implements an "unbroadcast" helper that sums over axes that were introduced or expanded.
*Most common mistake:* returning a gradient with the same shape as $Y$ — PyTorch witly let this into a parameter with a non-broadcasting shape and produce a shape error one layer later, confusing the learner about which layer is broken.

**Example 7 — Diamond graph.**
Problem: $a = 2$; $b = a+1$; $c = a \cdot 3$; $d = b \cdot c$. Forward: $b=3$, $c=6$, $d=18$. Backward.
Backward: $\frac{\partial d}{\partial d}=1$; $\frac{\partial d}{\partial b}=c=6$, $\frac{\partial d}{\partial c}=b=3$. Now $a$ has two paths: via $b$ contributes $6 \cdot 1 = 6$; via $c$ contributes $3 \cdot 3 = 9$. $\frac{\partial d}{\partial a} = 15$.
*Pedagogical point:* without topological sort, you might pop `a` off the stack before both paths have contributed. Without `+=`, you'd get 6 or 9 instead of 15.
*Most common mistake:* running `a._backward()` from `b` before `c`'s gradient has been computed, then overwriting with the `c` contribution.

**Example 8 — Numerical gradient check on the one-neuron net.**
Problem: verify Example 2's analytical `w1.grad = 1.0` numerically. Compute $\tilde g = (f(w_1+\varepsilon)(w_1-\varepsilon))/(2\varepsilon)$ with $\varepsilon = 10^{-5}$. Expect relative error < $10^{-7}$. [Source](http://deeplearning.stanford.edu/tutorial/supervised/DebuggingGradientChecking/)
*Pedagogical point:* gives the learner a *test harness* for every future gradient they write. Mirrors micrograd's own `test_engine.py` which checks every Value op against PyTorch. [Source](https://github.com/karpathy/micrograd)
*Most common mistake:* using the one-sided finite difference `(f(w+ε)-f(w))/ε`, which has O(ε) truncation error and won't agree to 7 digits; or using $\varepsilon = 10^{-10}$, where floating-point cancellation wrecks the numerator.

## 3. Common misconceptions

1. **"Backprop is just the chain rule."** *Why natural:* the derivation literally is the multivariable chain rule. *Why broken:* the chain rule tells you *what* each derivative is; backprop is the *algorithm* for computing all of them in one sweep without recomputing shared subexpressions. *Kill it with:* Olah's "sum over paths" combinatal-explosion picture — 9 paths vs. 6 edges — and the explicit claim: reverse-mode is the chain rule *plus* dynamic programming over a graph. [Source](https://colah.github.io/posts/2015-08-Backprop/)
2. **"The backward pass undoes the forward pass."** *Why natural:* the arrows literally reverse. *Why broken:* the backward pass doesn't unwind the computation — it *accumulates new values* (gradients) at each node using cached forward values. Killing it: show the `Value` object has *both* `.data` and `.grad`; running `.backward()` leaves `.data` untouched.
3. **"PyTorch does symbolic differentiation."** *Why natural:* it feels magic, like Mathematica. *Why broken:* PyTorch runs your Python forward code, *records a tape* of ops on Tensors, and replays it in reverse — no symbolic formula is ever built. *Kill it with:* the `autograd-as-tape` framing; show that Python `if`/`while` works inside `torch.no_grad`'s opposite because the tape just records whatever branch ran. [Source](https://en.wikipedia.org/wiktic_differentiation)
4. **"You need to derive gradients by hand for every layer."** *Why natural:* textbook derivations do exactly this. *Why broken:* if you implement `_backward` for each *primitive* op, autodiff composes them for free. *Kill it with:* build micrograd's 6 primitive backwards (+, *, **, tanh, exp, relu) and show that a 2-layer MLP's gradient falls out with zero additional derivation.
5. **"Gradients flow through any function."** *Why natural:* everything is smooth in calc class. *Why broken:* `torch.argmax`, discrete sampling, `int()` casts, comparison masks — these have zero gradient everywhere they're defined. *Kill it with:* a widget that tries to backprop through `argmax` and shows `.grad` is 0 on the parameter controlling the argmax; contrast with the "straight-through estimator" hack used in practice.
6. **"In-place ops are fine because the values still match."** *Why natural:* `x += 1` and `x = x + 1` look equivalent. *Why broken:* the autograd tape stores a *reference* to the tens it'll need in `_backward` (e.g., `tanh` stores its output to compute $1-t^2$); an in-place modification corrupts that saved buffer. *Kill it with:* an explicit before/after widget where in-place modification of a tanh's output produces a visibly wrong gradient on the weight.
7. **"`.zero_grad()` is a PyTorch quirk; the math shouldn't require it."** *Why natural:* you don't sum gradients across iterations in the derivation on paper. *Why broken:* the same `+=` that makes diamond graphs work is what forces you to explicitly zero between iterations. *Kill it with:* a training-loop widget with a toggle for `.zero_grad()`; show the loss diverging when it's off because gradient magnitudes grow linearly with step number. [Source](https://whatilearnedtoday.net/2024/09/23/intro-to-llms-andrej-karpathys-micrograd/)
8. **"Gradients and derivatives are the same thing."** *Why natural:* in 1D they literally are. *Why broken:* a gradient is a vector/tensor of partial derivatives; for autodiff purposes, `.grad` on a node is specifically $\partial L / \partial (\text{this node})$ — a scalar-per-entry because $L$ is scalar. *Kill it with:* explicitly name `.grad` as "sensitivity of the loss to this number."
9. **"You can detach a tensor to fix numerical issues without affecting training."** *Why natural:* `.detach()` looks like a performance hint. *Why broken:* it severs the autograd graph — gradients will not flow through the detached tensor back to the parameters that produced it. *Kill it with:* a widget that `.detach()`s an intermediate and shows upstream `.grad` silently stays at 0.
10. **"Backprop is slow because of all the chain rule."** *Why natural:* seems like compounding multiplications. *Why broken:* reverse-mode touches each edge exactly once, so backward is roughly the same cost as forward (the constant is typically 2–3×). *Kill it with:* Olah's factoring explanation and a stopwatch in the widget. [Source](https://colah.github.io/posts/2015-08-Backprop/)
11. **"A bigger graph means more paths means more * *Why natural:* Olah's path-counting example looks combinatorial. *Why broken:* paths are summed lazily *at each node*; reverse-mode cost is linear in the number of edges, not paths. *Kill it with:* show paths exploding in the naive view, then collapse them back via the factoring animation.
12. **"Forward-mode autodiff and reverse-mode autodiff compute different things."** *Why natural:* the algorithms look completely different. *Why broken:* they compute the same Jacobian; they differ in *which direction* they traverse it. Forward is efficient for $n \ll m$; reverse is efficient for $m \ll n$; deep learning has $n$ = millions of parameters and $m = 1$ (the loss), so reverse wins. *Kill it with:* a side-by-side widget where the learner sets $(n, m)$ and watches the op count for each mode. [Source](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec08.pdf)

## 4. Interactive widget suggestions

Opinionated stance up front: **build micrograd's `Value` class scalar-by-scalar before touching tensors.** Karpathy's order is correct and I recommend preserving it. The reason: the moment you go to tensors, the learner loses the mental picture of "one node, one edge, one local derivative" and starts memorizing matrix identities. Keep scalars for Lessons A–C; earn the right to tensors in Lesson D.

### (a) `graphForgeBackprop`
- **User manipulates:** drags op tiles (`+`, `*`, `**`, `tanh`, `relu`, `exp`) onto a canvas; draws edges by click-dragging from a node's output port to another's input port; types scalar values into leaf nodes; clicks "Forward" then "Backward".
- **Live updates:** each node displays two numbers — `.data` on the left, `.grad` on the right. On Forward, `.data` values fill in via a wave animation along the edges. On Backward, `.grad` values fill in via a *reverse* wave, each edge briefly highlighting with its local derivative while the wave passes. A node used twice visibly pulses twice during the backward wave (the `+=`).
- *gible concept:** the `computational-graph` *is* the picture; the `backward-pass-implementation` is the reverse wave; `gradient-accumulation` is the double pulse.
- **Beats a slider:** the learner is *wiring* the graph, not adjusting a number — they own the topology. Every edge they wire later turns into a line of `_backward` code.
- **Prior art:** Karpathy's `draw_dot` graphviz output in the micrograd repo ([Source](https://github.com/karpathy/micrograd)); the static backprop visualization at [sajalsharma.com/backprop-visualization](https://sajalsharma.com/backprop-visualization/); TensorFlow Playground's animated forward pass on weights ([Source](https://playground.tensorflow.org/)).

### (b) `edgeInspector`
- **User manipulates:** hovers/clicks any edge in the graph from (a); optionally scrubs a parent node's `.data` value to see the local derivative update in real time.
- **Live updates:** pops a tooltip showing the op's local-derivative formula (e.g., for edge into a `*`: "∂out/∂this = other.data ), the numeric value of the local derivative *right now*, and the contribution this edge just made to the parent's `.grad` (= upstream × local). The edge's stroke width encodes |local derivative|.
- **Tangible concept:** `local-derivative` and `upstream-vs-local-gradient` as properties of the *edge*, not the node.
- **Beats a slider:** makes the chain-rule factor on *one specific edge* a clickable object. Reading "∂/∂x = y" from a textbook teaches the rule; clicking an edge and watching its local derivative change as you scrub the sibling input teaches the *reason*.
- **Prior art:** Tensorflow.js playground's per-weight hover tooltip; Distill's "Building Blocks" interactive annotations over feature maps ([Source](https://distill.pub/2018/building-blocks/)).

### (c) `topoSortWalkthrough`
- **User manipulates:** takes a diamond-shaped graph (Example 7 above) and drags nodes into an ordering slot at the bottom; clicks "Run backward in this order" to watch what happens; can also click "DFS build" to animae recursive construction.
- **Live updates:** the final `.grad` of each node turns green if correct, red if wrong. When an order is wrong (e.g., `a` placed before `b`), the backward pass runs, the learner watches `a.grad` get set to `6` (from the `b` path), then the `c` path attempts to contribute but `c`'s own `.grad` is still 0 because `d`'s pass hasn't hit `c` yet. Shows the *mechanism* of the bug.
- **Tangible concept:** `topological-sort` is no longer an implementation detail — it's literally the thing that makes the diamond give 15 instead of 6 or 9.
- **Beats a slider:** the learner produces the bug with their own hands before seeing the fix.
- **Prior art:** Karpathy's in-video explanation of "fixing a backprop bug when one node is used multiple times" in the spelled-out intro video ([Source](https://www.youtube.com/watch?v=VMj-3S1tku0)); the diamond-gradient discussion in [Jhidding's Micrograd in Julia post](https://jhidding.github.io/micrograd-julia/).

### (d) `valueClassBuilder`
- **User manipates:** a live Python editor running in-browser (Pyodide) with a scaffolded `Value` class; writes the body of `__mul__._backward`, `__add__._backward`, `tanh._backward` one at a time; each gated step runs a tiny test that compares to PyTorch.
- **Live updates:** the test output goes green (matches PyTorch to 1e-7) or red (with a specific diff); the graph-viz visualization updates with the learner's current implementation applied to a fixed expression.
- **Tangible concept:** `per-op-backward-closures` and the entire `value-class` — the learner is literally writing the engine.
- **Beats a slider:** it *is* the keystone moment. The learner's own code is executed and compared to a reference.
- **Prior art:** Karpathy's `test_engine.py` in micrograd ([Source](https://github.com/karpathy/micrograd)); Selenium-style notebook builds like [Ryan Killian's Karpathy workbooks](https://github.com/ryankillian/karpathy-lectures-notebooks).

### (e) `broadcastReducer`
- **User manipulates:** drags to choose an input sha ($N, M$) and a bias-vector shape ($M$), hits "forward" to see broadcasting, then hits "backward" and must click the axes to *sum over*. If they skip an axis, the framework refuses to complete the backward (shape mismatch) and displays the `unbroadcast` logic they missed.
- **Live updates:** shape badges on each tensor in the DAG; an "unbroadcast" node visibly appears on the backward edge once the learner correctly identifies the broadcast axes.
- **Tangible concept:** `broadcasting-gradient` — "broadcast forward, reduce backward."
- **Beats a slider:** the shape mismatch *is* the lesson; the learner feels the error in their fingers.
- **Prior art:** PyTorch error messages as the de facto negative teacher; [Douglas Orr's autodiff-from-scratch post](https://douglasorr.github.io/2021-11-autodiff/article.html) explicitly names the "broadcast → reduce-sum" duality.

### (f) `forwardVsReverseRaceTrack`
- **User manipulates:** drags sliders for `n` (input dim) and `m` (output dim) of a random DAG; hits "Race,ich runs forward-mode autodiff (one sweep per input) and reverse-mode (one sweep per output) side by side, counting edge visits.
- **Live updates:** two counters tick up live; at the end, two bars show total ops. The color of each mode darkens as its cost dominates. A toggle "Loss function mode" pins $m=1$ and shows reverse-mode as a flat line while forward-mode blows up with $n$.
- **Tangible concept:** `why-reverse-mode-for-dl` — the one-slide reason deep learning picked reverse.
- **Beats a slider:** the slider alone would just be a number; the race lets the learner *feel* the asymmetry.
- **Prior art:** MIT 18.S096 lecture notes on forward vs. reverse mode ([Source](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec08.pdf)); the auto-eD visualization tool ([Source](https://auto-ed.readthedocs.io/en/latest/mod3.html)).

### (g) `gradientCheckBench`
- **User manipulates:** writes (or inherits from Lesson C) a custom `_backward` for a w op (say, `sin`); runs "Grad check," which perturbs each input by ε and compares to the analytical gradient; can scrub ε on a log scale.
- **Live updates:** a log-log plot of relative error vs. ε appears with a characteristic "U" curve — truncation error dominates at large ε, round-off at small ε. A green band at $10^{-7}$ relative error confirms correctness; the sweet spot around $\varepsilon \approx 10^{-5}$ is highlighted.
- **Tangible concept:** `gradient-checking` as a *test*, plus the numerical reality of floating point.
- **Beats a slider:** the slider *has* meaning — the learner discovers the U-curve empirically, which is the whole pedagogical point.
- **Prior art:** Stanford's UFLDL gradient-check page ([Source](http://deeplearning.stanford.edu/tutorial/supervised/DebuggingGradientChecking/)); Andrew Ng's "fraud-detection" gradient check assignment in the Deep Learning specialization.

## 5. Key formulas

**Chain rule on a graph**
```
\frac{\partial L}{\partial x} = \sum_{y \in \text{chil \frac{\partial L}{\partial y}\,\frac{\partial y}{\partial x}
```
```
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial x}
```
```
\text{grad}_x \mathrel{+}= \text{grad}_y \cdot \frac{\partial y}{\partial x}
```

**Scalar backward rules per op** (given `out = op(a, b)` with upstream gradient `g = ∂L/∂out`)
```
c = a + b: \quad \frac{\partial L}{\partial a}\mathrel{+}= g,\ \frac{\partial L}{\partial b}\mathrel{+}= g
```
```
c = a - b: \quad \frac{\partial L}{\partial a}\mathrel{+}= g,\ \frac{\partial L}{\partial b}\mathrel{+}= -g
```
```
c = a \cdot b: \quad \frac{\partial L}{\partial a}\mathrel{+}= g\cdot b,\ \frac{\partial L}{\partial b}\mathrel{+}= g\cdot a
```
```
c = a / b: \quad \frac{\partial L}{\partial a}\mathrel{+}= g/b,\ \frac{\partial L}{\partial b}\mathrel{+}= -g\cdot a/b^{2}
```
```
c = a^{n}: \quad \frac{\partial L}{\partial a}\mathrel{+}= g\cdot n\, a^{n-1}
```
```
c = e^{a}: \quad \frac{\partial L}{\partial a}\mathrel{+}= g\cdot e^{a} = g\cdot c
```
c = \log a: \quad \frac{\partial L}{\partial a}\mathrel{+}= g\,/\,a
```
```
c = \tanh(a): \quad \frac{\partial L}{\partial a}\mathrel{+}= g\,(1 - \tanh^{2}(a)) = g\,(1-c^{2})
```
```
c = \sigma(a): \quad \frac{\partial L}{\partial a}\mathrel{+}= g\,\sigma(a)(1-\sigma(a)) = g\,c(1-c)
```
```
c = \mathrm{ReLU}(a): \quad \frac{\partial L}{\partial a}\mathrel{+}= g\cdot \mathbb{1}[a>0]
```

**Matmul backward**
```
Y = X W
```
```
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}\, W^{\top}
```
```
\frac{\partial L}{\partial W} = X^{\top}\, \frac{\partial L}{\partial Y}
```
```
Y = X W + b,\quad \frac{\partial L}{\partial b} = \sum_{n=1}^{N} \frac{\partial L}{\partial Y_{n,:}}
```

**Softmax + cross-entropy backward**
```
p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}
```
```
L = -\sum_i y_i \log p_i
```
```
\frac{\partial L}{\partial z} = p - y
```

**Broadcasting backward** (input shape `s_a` broadcast to output shape `s_out`)
```
\frac{\partial L}{\partial a} = \mathrm{unbroadcast}\!\left(\frac{\partial L}{\partial \mathrm{out}},\ s_a\right) = \text{sum over broadcast axes of } \frac{\partial L}{\partial \mathrm{out}}
```

**Gradient check**
```
\tilde g = \frac{f(\theta+\varepsilon) - f(\theta-\varepsilon)}{2\varepsilon}
```
```
\text{relerr} = \frac{\lVert g_\text{analytical} - \tilde g \rVert}{\lVert g_\text{analytical}\rVert + \lVert \tilde g\rVert} < 10^{-7}
```

## 6. Lesson decomposition

Five lessons. Spine follows Karpathy's order because it's the right order: scalars build the mental model, tensors then slot in as "the same thing, faster." Total estimated seat time: ~2.5–3 hours of active learner time — i.e., Karpathy's video, chunked and gated.

### Lesson 12.1 — "Draw the graph"
- **Summary:** Take any expression you already know how to evaluate and turn it into a graph you can walk node by node.
- **Steps (≈ 45–55 min):**
  1. *Prose-widget.* The three-node graph for $f = (a+b)\cdot c$; click each node to evaluate. [uses `graphForgeBackprop`]
  2. *Prose.* "Values live on nodes, opive on edges — and every edge has a local derivative."
  3. *StepCheck.* With $a=2, b=-3, c=10$, what is $f$? **Answer: -10.**
  4. *Widget.* Click each edge to see its local derivative. [uses `edgeInspector`]
  5. *StepCheck.* On the edge from $e$ to $f$ (with $f = e\cdot c$), what is $\partial f/\partial e$? **Answer: 10 (i.e., $c$).**
  6. *Prose.* The sum-over-paths rule; visual of the diamond graph from Olah.
  7. *Widget.* Build your own graph for $f(x,y) = xy + \sin(x)$ at $x=\pi/2, y=2$. Evaluate forward.
  8. *StepCheck.* What is $f$? **Answer: $\pi + 1 \approx 4.1416$.**
  9. *Prose.* Endgame callback.
- **Widgets used:** `graphForgeBackprop`, `edgeInspector`.

### Lesson 12.2 — "Walk it backward"
- **Summary:** The backward pass is a reverse sweep of the same graph, propagating one gradient per node.
- **Steps (≈ 45–55 min):**
  1. *Prose.* Seed `dL/dL = 1`. Every node gets a `.grad`.
  2. *Widget.* Run backward on the $(a+b)\cdot c$ graph; watch gradients flow. [uses `graphForgeBackprop`StepCheck.* What is $\partial f/\partial a$ with $c=10$? **Answer: 10.**
  4. *StepCheck.* What is $\partial f/\partial b$? **Answer: 10.** (Teaches that `+` copies.)
  5. *Prose-widget.* The one-neuron example $o = \tanh(w_1 x_1 + w_2 x_2 + b)$ — Karpathy's canonical setup with $x_1=2, x_2=0, w_1=-3, w_2=1$.
  6. *StepCheck.* With $n \approx 0.8814$, what is $1 - \tanh^2(n)$ (to 2 decimals)? **Answer: 0.50.**
  7. *StepCheck.* What is $\partial o/\partial w_1$? **Answer: 1.0.**
  8. *Prose.* The diamond graph; why a naïve recursive backward gives the wrong answer.
  9. *Widget.* Misorder the nodes of the diamond on purpose; watch the bug. [uses `topoSortWalkthrough`]
  10. *Prose.* Topological sort as the fix. One sentence, one picture.
  11. *StepCheck.* In Example 7's diamond, $\partial d/\partial a = ?$ **Answer: 15.**
  12. *Prose.* Endgame callback.
- **Widgets used:** `graphForgeBackprop`, `topoSortWalkthrough`.

### Lesson 12.3 — "Build the Value class"
- **Summary:** Now write the code. Every grou've walked by hand becomes a `Value` object with a `_backward` closure.
- **Steps (≈ 45–55 min):**
  1. *Prose.* Intro the `Value` dataclass: `.data`, `.grad`, `._prev`, `._op`, `._backward`. Show the complete skeleton (no backwards filled in). [Source](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py)
  2. *Widget.* Write `__add__._backward`. Hit run; tested against PyTorch. [uses `valueClassBuilder`]
  3. *Widget.* Write `__mul__._backward`. (This is the moment `+=` appears for the first time; point out the `a*a` case.)
  4. *StepCheck — interpretation.* "Why `+=` and not `=` inside `_backward`?" *Free-text or multiple choice; canonical answer: a node can be used multiple times, contributions must be summed.*
  5. *Widget.* Write `__pow__._backward`, `exp._backward`, `tanh._backward`.
  6. *Prose.* The top-level `backward()` method: build topo, seed `1.0`, iterate in reverse.
  7. *Widget.* Implement the topological sort via DFS; tests run on the diamond graph.
  8. *Widget.* a tiny 2-input, 1-hidden-layer MLP to fit a single sample (binary classifier on moons), manually looping over parameters and calling `p.data -= 0.05 * p.grad`. Plot the loss decreasing.
  9. *Prose.* Zero the grads. Show the bug Karpathy himself made on camera (gradients accumulating across iterations), and the fix. [Source](https://whatilearnedtoday.net/2024/09/23/intro-to-llms-andrej-karpathys-micrograd/)
  10. *StepCheck — debugging.* Given buggy training code missing `.zero_grad()`, what do gradients look like after 3 iterations compared to 1? *Canonical answer: ≈3× too large.*
  11. *Prose.* Endgame callback. You just built micrograd.
- **Widgets used:** `valueClassBuilder`.

### Lesson 12.4 — "From scalars to tensors"
- **Summary:** Stack 64 scalars into a vector, stack 768 vectors into a matrix, and the forward and backward rules still hold — but now `dL/dW` is a matrix.
- **Steps (≈ 30–40 min):**
  1. *Prose.* The scaling problem: a `Neuron` with 768 inputs and 64 outputs would be 768×parate `Value` objects. One matmul op does it in one line and is ~1000× faster on a GPU.
  2. *Prose-widget.* Derive `dL/dX = dL/dY @ W.T` and `dL/dW = X.T @ dL/dY` by checking shapes. Show the index-sum derivation as an optional reveal.
  3. *StepCheck.* Given `X.shape = (32, 100)`, `W.shape = (100, 10)`, `dY.shape = (32, 10)`, what shape is `dW`? **Answer: (100, 10).**
  4. *Widget.* Implement `matmul.backward` in a tensor micrograd stub; test vs PyTorch.
  5. *Prose.* Elementwise nonlinearities — same rule as before, just tensor-wise. `dL/dX = dL/dY * f'(X)`.
  6. *Widget.* Bias broadcasting: forward is easy, backward requires summing along the batch axis. [uses `broadcastReducer`]
  7. *StepCheck.* Given `dY.shape = (32, 64)` and bias shape `(64,)`, what is `db`? *Canonical: `dY.sum(axis=0)`, shape `(64,)`.*
  8. *Prose.* The famous softmax+CE collapse: derive $p - y$.
  9. *StepCheck.* For logits $[2.0, 1.0, 0.1]$ and target $y = [1, 0, 0]$, compute $p - y$ to 3 decimals. *Canonical: $[-0.341, 0.242.099]$.*
  10. *Prose.* Endgame callback.
- **Widgets used:** `broadcastReducer`, `valueClassBuilder` (tensor mode).

### Lesson 12.5 — "Check it, break it, fix it"
- **Summary:** Every autodiff bug is caught by a gradient check or diagnosed by knowing the four classic footguns.
- **Steps (≈ 30–40 min):**
  1. *Prose.* Why gradient checks: subtly buggy backwards still train (Karpathy: "the model still worked"). Fraud detection framing from Andrew Ng's assignment.
  2. *Widget.* The two-sided finite-difference formula; the U-curve of error vs. ε. [uses `gradientCheckBench`]
  3. *StepCheck.* At what ε (to one significant figure) is the minimum error? **Answer: ~1e-5.**
  4. *Widget.* Write a grad-check that asserts relative error < 1e-7; run it on your `tanh._backward` from lesson 12.3.
  5. *Prose.* The debugging hall of shame: missing `.zero_grad()`, in-place ops, `.detach()`, swapped matmul operands, broadcast axes.
  6. *StepCheck — debugging.* Spot the bug: given `dW = dY @ X.T` with shapes `(`(N,D)`, why is this wrong? *Canonical: should be `X.T @ dY`.*
  7. *StepCheck — debugging.* Spot the bug: `def _backward(): self.grad = out.grad * other.data`. *Canonical: should be `+=`.*
  8. *StepCheck — debugging.* After `x = x * 2` inside a `torch.no_grad()` block, will gradients flow? *Canonical: no.*
  9. *Prose.* Forward vs reverse autodiff side-by-side — why deep learning uses reverse mode.
  10. *Widget.* Race forward vs reverse on random DAGs as a function of $(n, m)$. [uses `forwardVsReverseRaceTrack`]
  11. *Prose.* Final endgame callback. The learner has just built the engine that trains every neural network on earth.
- **Widgets used:** `gradientCheckBench`, `forwardVsReverseRaceTrack`.

## 7. Problem bank

1. **[novice / computation / computational-graph, chain-rule-on-a-graph]** For $f(a,b,c) = (a+b)\cdot c$ with $a=2, b=-3, c=10$, compute $\partial f/\partial a$, $\partial f/\partial b$, $\partial f/\partial c$. *Answer:* 10, 10, −1.

2. **[novice / computation / local-derivative]y = x_1 \cdot x_2$, give both local derivatives. *Answer:* $\partial y/\partial x_1 = x_2$, $\partial y/\partial x_2 = x_1$.

3. **[novice / computation / chain-rule-on-a-graph]** For $g(x) = \tanh(3x + 1)$ at $x=0$, compute $g'(0)$. *Answer:* $(1-\tanh^2(1))\cdot 3 \approx 3 \cdot 0.41998 \approx 1.2600$.

4. **[novice / computation / local-derivative]** Give the local derivative of $y = \text{ReLU}(x)$ at $x = -0.3$ and at $x = 2.1$. *Answer:* 0 and 1.

5. **[novice / computation / chain-rule-on-a-graph]** For $f(x,y) = xy + \sin(x)$ at $(x,y)=(\pi/2, 2)$, compute the gradient. *Answer:* $(\partial f/\partial x,\partial f/\partial y) = (y + \cos(x), x) = (2, \pi/2)$.

6. **[intermediate / computation / gradient-accumulation]** For $f(a) = a \cdot a + a$ at $a=4$, use the graph to compute $f'(a)$ by identifying each path's contribution. *Answer:* $2a + 1 = 9$; contributions are $a$ (from left `*`), $a$ (from right `*`), $1$ (from `+`).

7. **[intermediate / construction / per-op-backward-closures]** Write the `_backward` closure in Python for `out = a ** n` (n constant, a a Value). *Answer:*
    ```python
    def _backward():
        a.grad += (n * a.data**(n-1)) * out.grad
    ```

8. **[intermediate / computation / chain-rule-on-a-graph]** For the neuron $o=\tanh(w_1 x_1 + w_2 x_2 + b)$ at $x_1=2, x_2=0, w_1=-3, w_2=1, b\approx 6.8814$, compute $\partial o/\partial x_1$. *Answer:* $(1-\tanh^2(n))\cdot w_1 = 0.5 \cdot (-3) = -1.5$.

9. **[intermediate / debugging / common-backprop-bugs]** The following `_backward` for multiplication is buggy. Find the bug and fix it.
    ```python
    def _backward():
        self.grad = out.grad * other.data
        other.grad = out.grad * self.data
    ```
    *Answer:* `=` should be `+=` on both lines; otherwise, when `self` or `other` is a leaf used by multiple parents, contributions from other paths get clobbered.

10. **[intermediate / computation / matmul-backward]** Given $Y = XW$ with `X.shape = (64, 128)`, `W.shape = (128, 10)`, `dL/dY.shape = (64, 10)`, what are the shapes of `dL/dX` and `dL/dW`, and how are they computed? *Answer:* `dL/dX.shape = (64, 128) = dL/dY @ W.T`; `dL/dW.shape = (128, 10) = X.T @ dL/dY`.

11. **[intermediate / computation / softmax-ce-fused-backward]** For logits $z = [2.0, 1.0, 0.1]$ and one-hot target $y = [1, 0, 0]$, compute $\partial L/\partial z$ to 3 decimals. *Answer:* $p = [0.659, 0.242, 0.099]$; $p - y = [-0.341, 0.242, 0.099]$.

12. **[intermediate / computation / broadcasting-gradient]** Forward: `Y = X + b` with `X.shape = (32, 10)`, `b.shape = (10,)` (broadcast along batch). Given `dL/dY.shape = (32, 10)`, what is `dL/db` and its shape? *Answer:* `dL/db = dL/dY.sum(axis=0)`, shape `(10,)`.

13. **[intermediate / debugging / topological-sort]** A learner's `backward()` method calls `_backward` in the insertion order of `_prev` rather than topo order. On a diamond $d = b \cdot c,\ b = a+1,\ c = 3a$, a = 2, what (wrong) gradient might they compute for `a`? Explain. *Answer:* Likely 6 or 9 instead of 15 — whichever path's contrition gets overwritten or runs before the other has propagated into the shared parent.

14. **[intermediate / debugging / common-backprop-bugs]** In a training loop:
    ```python
    for epoch in range(100):
        loss = model(x).loss(y)
        loss.backward()
        for p in model.parameters():
            p.data -= 0.1 * p.grad
    ```
    Why does the loss diverge after a few iterations? *Answer:* `.grad` is never zeroed; after $k$ iterations, the gradient used is effectively the sum of all past gradients, ≈k× the correct magnitude, so the step size grows without bound.

15. **[advanced / construction / value-class]** Implement a complete `Value.__truediv__` (division) that correctly supports backward. Use only previously-defined ops. *Answer:* `return self * other**-1`; relies on `__mul__` and `__pow__` being correct. This gives `da/d(a/b) = 1/b` and `db/d(a/b) = -a/b^2` for free.

16. **[advanced / proof-scaffold / why-reverse-mode-for-dl]** Let $f: \mathbb{R}^n \to \mathbb{R}^m$ be computed by AG with $E$ edges. State and justify: (a) the cost of forward-mode autodiff; (b) the cost of reverse-mode; (c) why deep learning uses reverse mode. *Answer:* (a) $O(n\cdot E)$; (b) $O(m\cdot E)$; (c) in deep learning $m=1$ (scalar loss) while $n$ is in the millions or billions, so reverse is cheaper by a factor of $n$.

17. **[advanced / debugging / common-backprop-bugs]** A student's custom `ReLU._backward` is:
    ```python
    def _backward():
        self.grad += out.grad * (out.data > 0)
    ```
    Gradient check says "relative error 3e-3" — bad. Identify the bug. *Answer:* The indicator should be on the *input* `self.data > 0`, not on the output `out.data > 0`. For ReLU they coincide *except* at $x=0^-$ vs $x=0$; more importantly, if the op were ever `LeakyReLU` the distinction matters, and the student's conceptual model is wrong. (Acceptable alternate answer: both are equivalent for standard ReLU, but the numerical error comes from a different source — e.g., a different line.)

18. **[advanced /struction / scalar-to-tensor-vectorization]** Write a vectorized forward and backward for a single linear layer `Y = X @ W + b` using only numpy (not PyTorch). Include shape comments. *Answer:*
    ```python
    # forward; X:(N,D), W:(D,M), b:(M,)
    Y = X @ W + b                    # (N, M)
    # given dY:(N,M), backward:
    dX = dY @ W.T                    # (N, D)
    dW = X.T @ dY                    # (D, M)
    db = dY.sum(axis=0)              # (M,)
    ```

19. **[advanced / interpretation / autograd-as-tape]** In PyTorch, `x = torch.tensor([1., 2.], requires_grad=True)`, `y = x.detach() * 3; y.sum().backward()`. What is `x.grad`? Explain. *Answer:* `None` (or all zeros). `.detach()` returns a tensor that shares storage but is *not* part of the autograd graph, so gradient flow from `y` stops at the detach boundary; `x` receives no contribution.

20. **[advanced / debugging / matmul-backward, scalar-to-tensor-vectorization]** A learner writes `dW = dY @ X.T` and their gradient check fails with 100% relative error. The shapes happen to be compatible because the problem has `N = D`. What's the bug and how would they have caught it before a grad check? *Answer:* Correct formula is `dW = X.T @ dY` (the operand order matters; matmul isn't commutative). They'd catch it earlier by testing with non-square matrices where `N ≠ D` — the incorrect formula would throw a shape error immediately.

## 8. Endgame callback — refined

Starter: *"You just built micrograd. loss.backward() is no longer magic — it's your code."*

Candidates, in ascending order of my preference:

**C1.** *"You just built micrograd. Every deep-learning framework on earth — PyTorch, JAX, TensorFlow — is a faster, GPU-aware version of what you just wrote by hand. `loss.backward()` is no longer magic; it's your code."*

**C2.** *"Before this module, `loss.backward()` was a call into someone else's library. Now it's a reverse walk of a graph you built, running closures you wrote. Everything from a 2-parameter fit to a 70-billion-paramins on this exact algorithm."*

**C3 (recommended).** *"Every neural network on earth — every GPT, every diffusion model, every AlphaFold — is trained by the algorithm you just implemented in about 100 lines of Python. The rest is efficiency."*

Why C3 wins: it (a) lands the keystone emotionally (the learner has joined a club), (b) uses concrete brand-name models the learner knows, (c) echoes Karpathy's own canonical line "the rest is just efficiency" ([Source](http://karpathy.github.io/2026/02/12/microgpt/)) which becomes an inside-reference the learner will recognize the first time they watch his video, and (d) is short enough to be the tab-closer at the bottom of every lesson in the module.

## 9. Sources (licensing-aware)

1. **"The spelled-out intro to neural networks and backpropagation: building micrograd"** — Andrej Karpathy, YouTube video (2.5 hours) — [https://www.youtube.com/watch?v=VMj-3S1tku0](https://www.youtube.com/watch?v=VMj-3S1tku0). **Medium:** video. **License:** standard YouTube redistribution). **Tag: [REFERENCE-ONLY]**. *Use for:* the exact pedagogical order of operations for this module; do not re-host or transcribe at length. Timestamps and short quoted titles are fine under fair use. [Source](https://karpathy.ai/zero-to-hero.html)

2. **micrograd** — Andrej Karpathy, GitHub repo — [https://github.com/karpathy/micrograd](https://github.com/karpathy/micrograd). **Medium:** code. **License:** MIT (explicit: "The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy"). **Tag: [ADAPT]**. *Use for:* direct adaptation of the `Value` class as the backbone of Lesson 12.3. Keep the MIT notice in the relevant files in our repo; credit Karpathy by name in the module footer. [Source](https://github.com/karpathy/micrograd/blob/master/LICENSE)

3. **"Calculus on Computational Graphs: Backpropagation"** — Christopher Olah, colah.github.io (August 31, 2015) — [https://colah.github.io/posts/2015-08-Backprop/](https://colah.github.io/posts/2015-08-Backprop/). **Medium:** blog. **License:s blog is generally understood to be shared under terms that preclude commercial reuse (no explicit license on the post; Olah's other work on Distill is CC-BY 4.0, but colah.github.io itself does not carry an equivalent declaration, and the commercial safe default is to treat it as all-rights-reserved). **Tag: [REFERENCE-ONLY]**. *Use for:* the "sum over paths" visualization and the forward-vs-reverse framing, re-drawn in our own style. Do not embed figures. [Source](https://colah.github.io/posts/2015-08-Backprop/)

4. **CS231n Backpropagation notes and handouts** — Justin Johnson, Andrej Karpathy et al., Stanford — [https://cs231n.github.io/optimization-2/](https://cs231n.github.io/optimization-2/) and the linear-backprop / derivatives handouts at [https://cs231n.stanford.edu/handouts/linear-backprop.pdf](https://cs231n.stanford.edu/handouts/linear-backprop.pdf) and [https://cs231n.stanford.edu/handouts/derivatives.pdf](https://cs231n.stanford.edu/handouts/derivatives.pdf). **Medium:** course notes. **nse:** the `cs231n/cs231n.github.io` repository is MIT-licensed per its GitHub repo metadata [Source](https://github.com/cs231n). Individual handout PDFs on `cs231n.stanford.edu/handouts/` are not individually license-tagged; treat those as **[REFERENCE-ONLY]**. **Tag: [ADAPT]** for the main notes on [cs231n.github.io](https://cs231n.github.io/) (with attribution); **[REFERENCE-ONLY]** for the standalone PDF handouts. *Use for:* the canonical backprop-by-graph worked example ($f = (a+b)\cdot c$) and the linear-layer derivation shape-check pedagogy.

5. **Rumelhart, Hinton & Williams (1986), "Learning representations by back-propagating errors"** — *Nature* 323: 533–536 — [https://www.nature.com/articles/323533a0](https://www.nature.com/articles/323533a0). Author-hosted PDF at [https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf). **Medium:** paper. **License:** Nature copyright. **Tag: [REFERENCE-ONLY]**. *Use for:* the historical anchor sidebar paper that put this algorithm on the map," along with the fact that Werbos had it in 1974 [Source](https://en.wikipedia.org/wiki/Paul_Werbos) and Linnainmaa in 1970 [Source](https://people.idsia.ch/~juergen/who-invented-backpropagation.html)). Do not reproduce figures.

6. **Werbos, P.J. (1974)** — *Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences*, PhD thesis, Harvard. Reprinted in Werbos, P.J. (1994), *The Roots of Backpropagation*, Wiley. **Medium:** thesis / book. **License:** copyrighted. **Tag: [REFERENCE-ONLY]**. *Use for:* citation in the "who invented this" sidebar; the historical precedence over the 1986 paper is important for an adult technical audience who will meet these names in follow-on reading. [Source](https://www.werbos.com/AD2004.pdf)

7. **"Automatic differentiation"** — Wikipedia — [https://en.wikipedia.org/wiki/Automatic_differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation). **Medium:** encyclopedia article. **License:** CC 4.0 (Wikipedia). **Tag: [REFERENCE-ONLY]** (the ShareAlike clause makes incorporation incompatible with a paid closed course). *Use for:* precise terminology (forward accumulation, reverse accumulation, Wengert list) that learners will meet elsewhere.

8. **"Backpropagation through a fully-connected layer"** — Eli Bendersky, 2018 — [https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/](https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/). **Medium:** blog. **License:** the site footer lists no explicit license; treat as all-rights-reserved by default. **Tag: [REFERENCE-ONLY]**. *Use for:* the cleanest complete derivation of $\partial L/\partial W = X^\top\, \partial L/\partial Y$ from the scalar Jacobian; redraw in our own words.

9. **PyTorch `torch.autograd` documentation** — [https://docs.pytorch.org/docs/stable/autograd.html](https://docs.pytorch.org/docs/stable/autograd.html). **Medium:** docs. **License:** PyTorch docs are BSD-3-Cla*Tag: [ADAPT]** for short code examples with attribution. *Use for:* the "autograd-as-tape" framing and concrete examples of `detach()`, `requires_grad`, in-place corruption errors. [Source](https://docs.pytorch.org/docs/stable/autograd.html)

10. **TensorFlow Playground** — Daniel Smilkov, Shan Carter — [https://playground.tensorflow.org/](https://playground.tensorflow.org/) and [https://github.com/tensorflow/playground](https://github.com/tensorflow/playground). **Medium:** interactive widget + code. **License:** Apache 2.0. **Tag: [ADAPT]** (with attribution; Apache is commercial-friendly). *Use for:* inspiration only for our widget visual language; we will not embed their widget. [Source](https://playground.tensorflow.org/)

(Out of scope for this module but worth flagging: *Goodfellow, Bengio, Courville, Deep Learning* chapter 6 — copyrighted, **[REFERENCE-ONLY]**; *Roger Grosse's CSC 311 backprop lecture notes* — university copyright, **[REFERENCE-ONLY]**; *Bishop's PRML* — copyrighted, **[RNLY]**.)

## 10. Pedagogical traps

1. **"It's just the chain rule" framing.** *What happens:* the instructor correctly reduces backprop to the chain rule and thinks pedagogical victory has been achieved. The learner leaves with a formula but cannot implement backprop for a graph they haven't seen. *Why it happens:* the chain rule *is* the math; the pedagogical content is everything else — the graph, the topo sort, the `+=`, the tape. *Mitigation:* the first step of Lesson 12.1 is "draw the graph" — before any formula appears. Introduce the chain rule only *after* the learner has manually walked one graph forward and one backward. Explicitly say "backprop = chain rule + dynamic programming over a DAG" and build each word in order.

2. **Jumping to vectorized/tensor backward formulas before scalar is rock solid.** *What happens:* the learner memorizes "$X^\top dY$" as a pattern but can't reconstruct it when the shapes or indices are unusual. *Why it happens:* tensor notation is compact and tempting. *Mition:* hard constraint — no tensors until Lesson 12.4. Karpathy's order. All of 12.1–12.3 lives at the scalar level. When tensors arrive, derive `dW = X.T @ dY` by shape-checking and by index-sum from the scalar rule, in that order.

3. **Hand-waving "and PyTorch handles the rest."** *What happens:* at the very moment the learner is about to cross over to "ML implementer," the instructor shortcuts to the library, and the keystone moment evaporates. *Why it happens:* the material is long and the temptation to "efficiency-bump" is strong. *Mitigation:* Lesson 12.3's final StepCheck is *writing the top-level `backward()`*. Do not mention PyTorch until Lesson 12.4, and when you do, frame it as "the same thing, vectorized, on a GPU." The endgame callback (Section 8, C3) is the moment that pays this off.

4. **Over-indexing on the math derivation, under-indexing on the implementation.** *What happens:* the learner can derive softmax+CE → p−y on paper but freezes when asked to write the 6-line `softmax_ce._`. *Why it happens:* math-first instincts from a calc background. *Mitigation:* every lesson has a "write the code" widget step (`valueClassBuilder`) and at least one StepCheck that is an *implementation*, not a derivation. The assessment bank (Section 7) is roughly 40% computation, 25% construction/implementation, 20% debugging, 15% interpretation — over-weight the practical.

5. **Skipping topological sort as "an implementation detail."** *What happens:* diamond-shaped graphs (which every MLP contains — every weight is used by many outputs) produce mysteriously wrong gradients. *Why it happens:* on linear-chain graphs the recursive naive backward happens to produce the right order. *Mitigation:* Lesson 12.2 specifically builds a diamond, specifically runs a naive backward, specifically produces the wrong number, and only then introduces topo sort. The `topoSortWalkthrough` widget makes the bug physical. Karpathy issue #67 on his own repo catches a topo bug in the wild — worth citing. [Source](https:ub.com/karpathy/micrograd/issues/67)

6. **Not making the learner physically watch gradients accumulate.** *What happens:* the learner writes `self.grad = ...` instead of `self.grad += ...` on their first try, it passes trivial tests, and breaks silently on any graph where a node is used twice (i.e., all of them, once you include bias or residual connections). *Why it happens:* `=` is the instinctive Python reflex, and most pedagogical examples never revisit a node. *Mitigation:* Example 3 in Section 2 is explicitly `f(a) = a*a` — the minimal graph that exposes the bug. The `graphForgeBackprop` widget animates the *double pulse* on a reused node. The StepCheck in Lesson 12.3 step 4 asks the learner to explain why `+=` is necessary, in their own words, before they continue.

7. **(Bonus) Letting the learner believe autograd is symbolic.** *What happens:* the learner assumes PyTorch builds an algebraic formula, so Python control flow should break it, so they write awkward tensor-only code to avoid `if`. *Whit happens:* every previous exposure (Mathematica, SymPy, Wolfram Alpha) has been symbolic. *Mitigation:* the `autograd-as-tape` concept in Section 1 (item 22) is introduced explicitly in Lesson 12.5, with a widget step that runs the same function with different `if` branches and shows the tape recording whichever branch executed. Wikipedia's AD article is a good vocabulary anchor. [Source](https://en.wikipedia.org/wiki/Automatic_differentiation)

— End of Module 12 research brief.
