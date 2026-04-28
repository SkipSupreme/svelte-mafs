# Module 11 — Neural Network Fundamentals: Research Brief

*Tinker / Arc 2: ML Foundations. Sits between M10 (Optimization) and M12 (Backpropagation / micrograd).*

---

## 1. Concept dependency graph

A topologically sorted list. Concept ids are kebab-case; prerequisites refer to earlier-module ids (e.g. `m7-linear-algebra`) or this module's own ids.

1. **`weighted-sum`** — A neuron computes `w·x + b`, a single scalar dot-product plus an offset. *Prereqs:* `m7-linear-algebra` (dot product), `m4-pre-algebra` (sums).
2. **`bias-as-offset`** — The bias `b` is the scalar that shifts the decision boundary off the origin; without it every hyperplane passes through `0`. *Prereqs:* `weighted-sum`.
3. **`perceptron`** — A weighted sum followed by a hard step (Heaviside) used as a binary classifier. *Prereqs:* `weighted-sum`, `bias-as-offset`.
4. **`linear-separability`** — A dataset is linearly separable iff a single hyperplane perfectly partitions the two classes. *Prereqs:* `perceptron`, `m7-lin` (hyperplanes).
5. **`decision-boundary`** — The level set `{x : w·x + b = 0}`; in 2D a line, in 3D a plane, in n-D a hyperplane. *Prereqs:* `weighted-sum`, `bias-as-offset`.
6. **`xor-problem`** — XOR is the canonical 4-point dataset that is *not* linearly separable; it is the existence proof that a single perceptron is strictly weaker than logical computation. *Prereqs:* `perceptron`, `linear-separability`.
7. **`linear-layer`** — `y = Wx + b`: a vector of perceptrons sharing the same input, computed as one matrix-vector product. *Prereqs:* `weighted-sum`, `m7-linear-algebra` (matvec as linear combination of columns).
8. **`linear-composition-collapse`** — Stacking linear layers without nonlinearity is mathematically equivalent to a single linear layer: `W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)`. *Prereqs:* `linear-layer`, `m7-linear-algebra` (matrix multiplication associativity).
9. **`activation-function`** — A scalar nonlinearity applied elementwise after a linear layer; you anything. *Prereqs:* `linear-composition-collapse`.
10. **`sigmoid`** — `σ(x) = 1/(1+e^{-x})`: smooth, bounded in (0,1), historically the first hidden-unit activation, now mostly a probability-output function. *Prereqs:* `activation-function`, `m5-calculus` (exponentials).
11. **`tanh`** — `tanh(x) = (e^x − e^{−x})/(e^x + e^{−x})`: zero-centered sigmoid in (−1, 1); the activation Karpathy uses in micrograd. *Prereqs:* `sigmoid`.
12. **`relu`** — `ReLU(x) = max(0, x)`: piecewise-linear, non-saturating for positive inputs, the modern default for hidden layers. *Prereqs:* `activation-function`.
13. **`relu-kink-as-hinge`** — A single ReLU neuron carves the input space into a "dead" half-plane and an "active" half-plane joined by a fold along its decision boundary. *Prereqs:* `relu`, `decision-boundary`.
14. **`dead-relu`** — A ReLU neuron whose pre-activation is negative for *every* training input contributes zero gradient and never learns; pathological but visualizable. *Prereqs:* `relun` (gradients).
15. **`gelu`** — `GELU(x) = x·Φ(x)`: a smooth ReLU-like activation that gates inputs by their Gaussian CDF; the activation in BERT/GPT-2 feed-forward blocks. *Prereqs:* `relu`, `sigmoid`.
16. **`hidden-layer`** — A linear layer whose output is *not* the prediction but the input to a further layer; its outputs are learned features. *Prereqs:* `linear-layer`, `activation-function`.
17. **`mlp`** — Multilayer perceptron: a stack of (linear, nonlinearity) blocks ending in a task-appropriate output layer. *Prereqs:* `hidden-layer`, `linear-composition-collapse`.
18. **`forward-pass`** — Sequentially applying each (matmul + bias + nonlinearity) from input to prediction. *Prereqs:* `mlp`.
19. **`feature-space-warping`** — Each hidden layer is a learned change of coordinates that bends the input space so that classes become linearly separable in the final layer. *Prereqs:* `mlp`, `xor-problem`.
20. **`output-layer-choice`** — Sigmoid on output for binary classification, softmax (next mti-class, linear for regression — the output layer differs from hidden layers. *Prereqs:* `sigmoid`, `mlp`.
21. **`universal-approximation`** — An MLP with one wide enough hidden layer and a non-polynomial activation can approximate any continuous function on a compact set to arbitrary accuracy — *existence only*, no constructive bound on width or training. *Prereqs:* `mlp`, `feature-space-warping`.
22. **`parameter-count`** — The number of learnable scalars in a layer is `(in_dim · out_dim) + out_dim`; total params is the sum across layers and dominates memory/compute. *Prereqs:* `linear-layer`.
23. **`mlp-as-ffn-block`** — In a transformer, the per-token feed-forward block is exactly `Linear → activation → Linear` — i.e. a 2-layer MLP — applied independently to every token. *Prereqs:* `mlp`, `forward-pass`.

---

## 2. Canonical worked examples

These six are the *defaults* across CS231n, MIT 6.S191, Karpathy ZTH, 3Blue1Brown, and Goodfellow et al. Pick these because students will encouher resource they consult.

### 2.1 — A single perceptron computing AND

**Problem.** Find weights $w_1, w_2$ and bias $b$ such that the perceptron $y = \text{step}(w_1 x_1 + w_2 x_2 + b)$ outputs 1 on $(1,1)$ and 0 on $(0,0), (0,1), (1,0)$.

**Solution.** Choose $w_1 = w_2 = 1$, $b = -1.5$. Verify: $(1,1) \to 0.5 \to 1$; $(1,0) \to -0.5 \to 0$; $(0,1) \to -0.5 \to 0$; $(0,0) \to -1.5 \to 0$. The decision boundary is the line $x_1 + x_2 = 1.5$.

**Pedagogical point.** A perceptron *is* a hyperplane plus a threshold; learning weights = placing a line. The bias is what lets the line miss the origin.

**Most common student mistake.** Trying $b = 0$ "because zero is the natural starting point" — produces a boundary forced through the origin that cannot fit AND. This is the moment to make bias non-optional.

---

### 2.2 — XOR is not linearly separable

**Problem.** Prove no single perceptron $y = \text{step}(w_1 x_1 + w_2 x_2 + b)$ can compute XOR: $(0,0)\to 0, (0,1)\to 1, (1,0)\to 1, (1,1)\to 0$.

**Solu* Suppose such weights exist. From $(0,0)\to 0$: $b < 0$. From $(1,0)\to 1$: $w_1 + b > 0 \Rightarrow w_1 > -b > 0$. From $(0,1)\to 1$: similarly $w_2 > 0$. Then $w_1 + w_2 + b > -b + (-b) + b = -b > 0$, forcing $(1,1) \to 1$. Contradiction. ([Berkeley CS189 lecture notes](https://people.eecs.berkeley.edu/~jrs/189s19/lec/17.pdf))

**Pedagogical point.** This is the historic moment ([Minsky & Papert 1969](https://en.wikipedia.org/wiki/Perceptrons_(book))) that motivates everything that follows. It's a 4-line proof a software engineer can do over coffee, and it forces the question: *what's the smallest extension that fixes this?*

**Most common student mistake.** Believing "XOR is just hard" rather than "XOR is the smallest example of a non-linearly-separable problem"; missing that the *exact same proof technique* shows any topologically equivalent dataset (concentric rings, two interlocking spirals) is also impossible for a single linear layer.

---

### 2.3 — Solving XOR with a 2-2-1 MLP

**Problem.** Conruct an MLP with two hidden ReLU units and one linear output unit that computes XOR exactly on $\{0,1\}^2$.

**Solution.** Hidden layer:
$$h_1 = \text{ReLU}(x_1 + x_2 - 0.5), \quad h_2 = \text{ReLU}(x_1 + x_2 - 1.5)$$
Output: $y = h_1 - 2 h_2$. Check: $(0,0)\to 0-2\cdot 0=0$; $(0,1)\to 0.5 - 0 = 0.5$; $(1,0)\to 0.5 - 0 = 0.5$; $(1,1)\to 1.5 - 2(0.5) = 0.5$. Threshold at $0.25$. (Alternative: NAND+OR composition à la [Towards Data Science](https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7/).)

**Pedagogical point.** The hidden layer warps the input plane so the two classes become linearly separable. You can *hand-design* the warp; the network's job is to discover it.

**Most common student mistake.** Drawing the hidden units' decision boundaries on the *input* plane and trying to read XOR off of them directly. The right picture is two views: the (x₁, x₂) plane and the (h₁, h₂) plane — only the second one is linearly separable.

---

### 2.4 — Forward pass arit2-3-1 MLP

**Problem.** Given $W^{(1)} = \begin{pmatrix} 0.5 & -0.2 \\ 0.1 & 0.4 \\ -0.3 & 0.6 \end{pmatrix}$, $b^{(1)} = (0, 0.1, -0.2)^\top$, $W^{(2)} = (1.0, -1.0, 0.5)$, $b^{(2)} = 0$, sigmoid hidden activations, and input $x = (1, 1)^\top$, compute $\hat y$.

**Solution.**
$z^{(1)} = W^{(1)} x + b^{(1)} = (0.3, 0.6, 0.1)^\top$.
$h = \sigma(z^{(1)}) \approx (0.5744, 0.6457, 0.5250)^\top$.
$z^{(2)} = W^{(2)} h + b^{(2)} = 0.5744 - 0.6457 + 0.5\cdot 0.5250 = 0.1912$.
$\hat y = z^{(2)} \approx 0.191$ (linear output).

**Pedagogical point.** A forward pass is just two matvecs and one elementwise sigmoid. There is no magic; it's high-school arithmetic in a loop.

**Most common student mistake.** Forgetting the bias on either layer, or applying the activation to the *output* layer when the task is regression. (Karpathy lists "passed softmaxed outputs to a loss that expects raw logits" as one of his canonical NN debugging gotchas; same family of bug.)

---

### 2.5 — Why ReLU stacks beat sigmoid stacks (vaniing gradient demo)

**Problem.** Consider an 8-layer MLP with sigmoid activations everywhere. Initial weights are Gaussian, $w \sim \mathcal N(0, 1)$. Why does training stall?

**Solution sketch.** $\sigma'(z) = \sigma(z)(1-\sigma(z)) \le 0.25$. The chain rule sends gradients backward as a product of $\sigma'$ terms times weight matrices. After 8 layers the signal is attenuated by at least $0.25^8 \approx 1.5 \times 10^{-5}$ before *any* weight effect. With ReLU, the derivative is exactly 1 on the active half-line, so the gradient is preserved through every active path.

**Pedagogical point.** The reason ReLU became default is *not* aesthetic; it's that $\max(0, x)$ has gradient $1$ on its active region while $\sigma$ saturates. (See [the canonical ReLU-vs-sigmoid discussion](https://wandb.ai/ayush-thakur/dl-question-bank/reports/ReLU-vs-Sigmoid-Function-in-Deep-Neural-Networks--VmlldzoyMDk0MzI).)

**Most common student mistake.** Concluding "ReLU is always better." Sigmoid is still right for a binary-probability output; tanh is still right inside LSTM gates and many small models (it's what micrograd uses); GELU is the right default in transformers.

---

### 2.6 — MNIST-style 784-128-10 net parameter count

**Problem.** A network with input dim 784 (28×28 pixel image), one hidden layer of 128 units with ReLU, and 10-dim output. How many learnable parameters?

**Solution.** Layer 1: $784 \cdot 128 + 128 = 100{,}480$. Layer 2: $128 \cdot 10 + 10 = 1{,}290$. Total: $101{,}770$. Almost all parameters live in the first layer because input dimensionality dominates.

**Pedagogical point.** Parameter count is dominated by the widest matmul. Matters in M16 when we discuss why transformer FFN blocks use a 4× expansion factor — that's where a transformer's parameters mostly live.

**Most common student mistake.** Forgetting biases (off by $\approx 1\%$ here, but on a 1-billion-param model that's millions of parameters), or computing $784 \cdot 128 \cdot 10$ as if the network were one giant tensor product.

---

# — Universal approximation by sigmoid bumps (Nielsen-style intuition)

**Problem.** Sketch how a single-hidden-layer sigmoid network approximates an arbitrary continuous $f: [0,1] \to \mathbb R$.

**Solution sketch.** Use two hidden sigmoids per "bump": one with large positive weight at threshold $s$, one with large negative weight at threshold $s + \varepsilon$. Their difference, scaled, approximates a narrow rectangular pulse of arbitrary height at position $s$. Sum many such bumps in the output layer to approximate any Riemann-integrable function. ([Visual proof, Nielsen Ch. 4](http://neuralnetworksanddeeplearning.com/chap4.html) — note: CC BY-NC, reference only.)

**Pedagogical point.** The theorem is *not* magic. It's "Riemann sums, but with sigmoids instead of indicators." It says **existence**, says nothing about how many neurons you need or whether SGD will find them.

**Most common student mistake.** Concluding "so deep networks are unnecessary, one wide layer suffices." The universal approxima theorem is silent on the *width* required — and in practice required width is exponential in input dimension for most useful functions, while depth lets you reuse subcomputations. (See [Mitliagkas IFT-6085 Lecture 10](http://mitliagkas.github.io/ift6169-2022/ift-6169-lecture-10-notes.pdf).)

---

## 3. Common misconceptions

1. **"A neuron is a tiny model of a brain neuron."** *Why it's natural:* the name. The McCulloch-Pitts paper genuinely pitched it that way in 1943. *Kill with:* the side-by-side that an artificial neuron is one dot product plus one scalar function, while a real cortical neuron involves dendritic compartments, neurotransmitter dynamics, and ~10⁴ synapses — see [MIT News on grid-cell results](https://news.mit.edu/2022/neural-networks-brain-function-1102) and [the Sophos comparison](https://www.sophos.com/en-us/blog/man-vs-machine-comparing-artificial-and-biological-neural-networks). Reframe: "a neuron is just a weighted sum and a squash."

2. **"Deep just means lots of layers — metter."** *Why natural:* the marketing. *Kill with:* the linear-composition-collapse derivation `W₂(W₁x) = (W₂W₁)x` followed by a widget where the learner stacks five linear-only layers and confirms the decision boundary stays straight no matter how many layers they add.

3. **"The activation function is a hyperparameter you tune like learning rate."** *Why natural:* every tutorial lists ReLU/sigmoid/tanh/GELU as a dropdown. *Kill with:* a "what would break if I removed the nonlinearity?" diff-gym widget that lets the learner replace ReLU with identity and watch the network collapse to a line. Then introduce variants as fixes for *specific* failure modes, not as a zoo.

4. **"Bias is just a small correction — I can ignore it."** *Why natural:* it's the one parameter not visually depicted in many diagrams. *Kill with:* the widget where toggling bias = 0 forces the decision boundary through the origin, so the network can't even fit AND.

5. **"Universal approximation means MLPs can solve anything."*ural:* the name "universal." *Kill with:* explicit framing: existence theorem, not a constructive one; says nothing about (a) how many neurons, (b) whether gradient descent finds them, (c) generalization to new data. ([Wikipedia is precise on this](https://en.wikipedia.org/wiki/Universal_approximation_theorem); "These are existence theorems… they do not provide a method for finding the network's parameters.")

6. **"Sigmoid is the 'default' activation because it looks biological."** *Why natural:* every textbook still leads with it. *Kill with:* the 8-layer-vanishing-gradient demo (Example 2.5). Sigmoid hasn't been the default hidden activation since ~2012.

7. **"More parameters = more accuracy."** *Why natural:* big-model triumphalism. *Kill with:* a fixed XOR dataset with sliders for hidden width 2 → 200 — accuracy plateaus at 4 hidden units, then training instability rises. (Conceptually, this is a soft preview of overfitting which lands properly in M13.)

8. **"The MLP draws a curve around the da *Why natural:* the colored decision-boundary visualizations look like curves. *Kill with:* the layered transformation widget — show that what's actually happening is the input space is being *bent* so that a *straight* line in the final hidden representation classifies everything. The curve in input space is the pullback of a straight line. (This is exactly [Olah's manifolds-and-topology framing](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/).)

9. **"Hidden-layer outputs are meaningless soup."** *Why natural:* you can't read them. *Kill with:* a ConvNetJS-style widget that lets the learner select two hidden units and plot every input point in the (h₁, h₂) plane — they will see XOR-data become *linearly separable* in this learned coordinate system.

10. **"ReLU is broken because it's not differentiable at zero."** *Why natural:* calculus pedantry. *Kill with:* show that `0` is a measure-zero set; in practice frameworks define $\partial \text{ReLU}/\partial x|_{x=0} := 0$; in 8 years ction deep learning this has caused exactly zero real bugs.

11. **"GELU is just a smoother ReLU; the choice doesn't matter."** *Why natural:* visually they look similar. *Kill with:* the BERT/GPT-2 reference — at transformer scale, GELU consistently outperforms ReLU because gradient flow through the negative tail (near zero but nonzero) prevents dead-neuron pathologies in very deep stacks. Cite [Hendrycks & Gimpel 2016](https://arxiv.org/pdf/2305.12073).

12. **"The forward pass is the same thing as inference; backprop is something extra."** *Why natural:* they're presented sequentially. *Kill with:* an explicit foreshadow — "Module 12 builds backprop on the *exact same computation graph* you just built. The forward pass is the structure; backprop just walks it backwards." Pin this to the endgame callback.

---

## 4. Interactive widget suggestions

Every widget below grabs a *named mathematical object* (a weight vector, a bias, a ReLU kink, an input point, a hidden activation) and manipulates it direc Sliders are forbidden unless the slider value is itself the object the lesson named.

### 4.1 — `perceptronLine`

**User manipulates directly.** Drags the *decision-boundary line itself* across a 2D scatter plot. Two grab handles: one rotates (sets $w$ direction), one translates perpendicular (sets $b$). Optional: drag the weight vector arrow at the origin and watch the orthogonal line update.

**Live updates.** Counts of correctly classified points update in real time; misclassified points flash red. The scalar values of $w_1, w_2, b$ display, derived from the line position. The half-plane is shaded.

**Concept.** Direct kinesthetic mapping between "geometric line in the plane" and "$w \cdot x + b = 0$." Shows that "training" = "moving this line."

**Beats slider+figure?** Yes — sliders for $w_1, w_2, b$ require the learner to mentally invert the parameterization. Dragging the *line* is the conceptual primitive; weights are the dependent readouts.

**Prior art.** [TensorFlow Playground](https://playgr.tensorflow.org/) (Apache 2.0) shows decision boundaries but does not let the user grab the line. [ConvNetJS classify2d demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html) (MIT) renders the boundary but again read-only. We exceed prior art by making the line draggable.

---

### 4.2 — `xorPlayground`

**User manipulates directly.** Drags the four XOR points $(0,0), (0,1), (1,0), (1,1)$ around the plane (with snap-back button) and tries to draw a *single* separating line by dragging the line as in §4.1. Then unlocks a "second neuron" — the user gets a second draggable line, and the widget renders the two-neuron hidden-space view side-by-side.

**Live updates.** Left panel: input space with the user's draggable line(s) and the four colored points. Right panel: $(h_1, h_2)$ space showing where each input lands after passing through the two ReLU neurons; updates as the user drags either line. A status indicator shows "linearly separable in hidden space?" yes/no.

**Concept.** The ataset is fixed in input space and **is not** linearly separable, but the two hidden ReLUs can warp it into a space where it **is**. The learner sees both pictures simultaneously.

**Beats slider+figure?** This widget is the entire pedagogical payload of the module. There is no slider equivalent.

**Prior art.** [Olah's manifolds-and-topology post](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) shows static layer-by-layer warps; ConvNetJS classify2d ([demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html)) shows hidden-layer transforms in a fixed grid. The novel piece here is *both* spaces live, with both the boundary and the input points draggable.

---

### 4.3 — `reluKinkDrag`

**User manipulates directly.** A 1D plot of a single ReLU neuron $f(x) = \max(0, wx + b)$. The learner drags the **kink point** along the x-axis. Internally this changes $b/w$ but the learner is grabbing the geometric kink, not a slider for `b`.

**Live updates.** The piecewise-linear functi plot redraws live. The slope handle (a separate rotational handle on the active arm) sets $w$. A live readout shows $w$, $b$, and the kink location $-b/w$. A second tab adds two ReLUs and shows their sum — illustrating how a *bump* is built from two ReLUs (the universal-approximation construction).

**Concept.** A ReLU isn't an "activation function" abstractly — it's a geometric hinge with a draggable corner. The location of the corner is determined by the bias.

**Beats slider+figure?** Yes — sliders separate the action ("change $b$") from the visual consequence ("the kink moves"). Dragging the kink fuses them.

**Prior art.** I am not aware of a widget that does exactly this; Distill articles on activations are all read-only static plots. This widget would be original to Tinker.

---

### 4.4 — `forwardPassScrubber`

**User manipulates directly.** Places an input point $x$ in the input plane (drag), then *scrubs a timeline* labeled "input → linear 1 → activation 1 → linear 2 → activation each scrub position the visualization shows the current state of the activations as a colored vector / heatmap.

**Live updates.** As the timeline scrubs, the visualization morphs continuously between successive layers' activations (interpolating linearly between layer outputs for animation purposes — make this clear, since real networks don't have intermediate states). Dimensionality changes are shown by adding/removing colored squares.

**Concept.** A forward pass is a *sequence of transformations*. The input is gradually warped, layer by layer, into the output. Demystifies "what does the network do internally."

**Beats slider+figure?** Yes — a static figure of layers stacked vertically with arrows is in every textbook. The pedagogical move is *animation under direct control*.

**Prior art.** [Distill "The Building Blocks of Interpretability"](https://distill.pub/2018/building-blocks/) (CC BY 4.0 — adaptable!) shows layer-by-layer activations on a frozen image. We extend by letting the learner picknput *and* scrub the depth axis.

---

### 4.5 — `decisionBoundarySculptor`

**User manipulates directly.** Click anywhere on a 2D canvas to add a labeled training point (left-click = class A, right-click = class B). A small MLP (configurable, default 2-4-4-1 with ReLU) trains in the browser via gradient descent on each click and the decision boundary updates.

**Live updates.** The colored decision-region rendering updates as training proceeds. Crucially, the user can also drag the *individual hidden-unit boundaries* (the kinks of each ReLU as draggable lines in the input plane) and watch the resulting decision region update — this connects "individual neurons each draw a line" to "the network's combined boundary is a polygonal mosaic of half-planes."

**Concept.** An MLP's decision boundary is *built from* the hidden neurons' individual hyperplane decision boundaries. The shape of the polygon you see is the union/intersection structure of the hidden ReLUs.

**Beats slider+figure?** Yes — TensorFlow ound has the dataset+architecture sliders but the *individual neurons' boundaries are not interactive*. Making each hidden unit's hyperplane a separately draggable object is the pedagogical leap.

**Prior art.** [TensorFlow Playground](https://playground.tensorflow.org/) (Apache 2.0; design ideas adaptable with attribution); [ConvNetJS classify2d](https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html) (MIT). We can adapt design directly while crediting both. Consider also reading [Smilkov & Carter's Playground source on GitHub](https://github.com/tensorflow/playground/blob/master/LICENSE).

---

### 4.6 — `activationDiff`

**User manipulates directly.** A 2D classification dataset (selectable: XOR, two moons, concentric rings) with a 2-8-1 MLP. The learner picks an activation function from a *visual* palette: dragging a function-graph icon (sigmoid curve, tanh curve, ReLU hinge, GELU smooth-curve, *identity*) onto the hidden layer. Each icon is itself a small functional plot.

**Live updas.** Decision boundary, training loss curve, and a histogram of hidden-unit activations update on each new activation function chosen. Critical case: when the user drops "identity" on the hidden layer, the boundary collapses to a straight line *while training continues* — the linear-composition-collapse made experimentally visible.

**Concept.** "What would break if I removed the nonlinearity?" — the question that motivates activations. Plus: comparison of sigmoid (saturates), ReLU (kinked, fast), GELU (smooth ReLU). Same dataset, same architecture, same training time → different qualitative behavior.

**Beats slider+figure?** Yes — the "drag the function onto the layer" interaction physicalizes the abstract slot of "activation function" as a *thing you can swap*. A multiple-choice dropdown does not.

**Prior art.** None precise. TensorFlow Playground has an activation dropdown; we differ by making the function graphs themselves the draggable objects, and by featuring identity as a "what breaks?" deion.

---

### 4.7 — `paramCounter`

**User manipulates directly.** A network architecture is shown as draggable column-of-circles diagrams. The user drags layer widths up/down (each layer is a column the user resizes vertically by dragging its top edge) and adds/removes layers by drag-drop.

**Live updates.** Total parameter count updates per layer and overall. Memory cost in MB at fp32 displays. A bar chart shows what fraction of parameters lives in each layer. When the user makes the architecture absurdly wide, a warning indicates "this exceeds the GPT-2 small parameter count."

**Concept.** Parameter count scales like the product of adjacent layer widths plus output bias. It's the budget that bounds models.

**Beats slider+figure?** Yes — a width slider with a number printout could do this trivially, but resizing the layer *as a visual object* (the circles fan out, the connecting edges multiply) ties the symbolic count to the visual object. This is borderline — included because parameter literacy atter in M16-18.

**Prior art.** [Netron](https://netron.app/) (MIT) visualizes architectures statically; this widget makes the visualization mutable.

---

## 5. Key formulas

Drop directly into MDX. Using KaTeX-compatible LaTeX.

**Perceptron**
- `y = \text{step}(w \cdot x + b)`
- `\text{step}(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}`
- `\text{decision boundary: } \{x : w \cdot x + b = 0\}`

**Linear layer**
- `y = Wx + b, \quad W \in \mathbb{R}^{m \times n},\ x \in \mathbb{R}^n,\ b \in \mathbb{R}^m`
- `(Wx)_i = \sum_{j=1}^{n} W_{ij} x_j`
- `\#\text{params} = m \cdot n + m`

**Activations**
- `\sigma(x) = \frac{1}{1 + e^{-x}}`
- `\sigma'(x) = \sigma(x)(1 - \sigma(x))`
- `\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}`
- `\tanh'(x) = 1 - \tanh^2(x)`
- `\text{ReLU}(x) = \max(0, x)`
- `\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}`
- `\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5\,x\bigl(1 + \tanh\bigl[\sqrt{2/\pi}(x + 0.044715\,x^3)\bigr]\bigr)`

**MLP forward pass (L-layer net)**
- `h^{(0)} = x`
- `z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}`
- `h^{(\ell)} = \phi\bigl(z^{(\ell)}\bigr) \quad \text{for } \ell = 1, \ldots, L-1`
- `\hat{y} = z^{(L)} \quad \text{(regression)} \quad \text{or} \quad \hat{y} = \sigma\bigl(z^{(L)}\bigr) \quad \text{(binary classification)}`
- Compact: `\hat{y} = \phi_L\bigl(W^{(L)} \phi_{L-1}\bigl(W^{(L-1)} \cdots \phi_1(W^{(1)} x + b^{(1)}) \cdots\bigr) + b^{(L)}\bigr)`

**Linear composition collapse (the negative result)**
- `W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)`

**Decision boundary (general)**
- `\{x \in \mathbb{R}^n : f_\theta(x) = 0.5\} \quad \text{(binary, sigmoid output)}`
- `\{x : w \cdot x + b = 0\} \quad \text{(linear classifier)}`

**Universal approximation (informal)**
- `\forall f \in C(K, \mathbb{R}),\ \forall \varepsilon > 0,\ \exists\, N, \{w_i, b_i, c_i\}_{i=1}^N : \left\| f(x) - \sum_{i=1}^N c_i\, \phi(w_i \cdot x + b_i) \right\|_\infty < \varepsilon \quad \forall x \in K`

**Transformer FFN preview (callback formula)**
- `\text{FFN}(x) = W_2\,\phi(W_1 x + b_1) + b_2 \quad \text{← this is just a 2-layer MLP}`

---

## 6. Lesson decomposition

Five lessons. Each is 8–15 gated steps. StepCheck answers are scalar or short symbolic.

### Lesson 11.1 — "What's a perceptron, really?"
**Summary.** A single neuron is a weighted sum, a bias, and a threshold — and it draws a line.
**StepChecks.**
- *Compute:* For $w = (1, 1)$, $b = -1.5$, $x = (1, 1)$: what's the pre-activation? **Answer: 0.5.**
- *Construct:* Find $b$ such that with $w = (1, 1)$ the decision boundary passes through $(0.5, 0.5)$. **Answer: $b = -1$.**
- *Interpret:* A perceptron with $w = (3, -2)$, $b = 1$ classifies $(0, 0)$ as which class? **Answer: class 1 (since $0 + 0 + 1 > 0$).**
**Widgets.** `perceptronLine`.
**Estimated minutes.** 15.

### Lesson 11.2 — "The XOR moment"
**Summary.** A single perceptron can't do XOR — and this 1969 fact is the reason neural networks have hidden layers.
**StepChecks.**
- *Proof-scaffold:* Fill in thequality: from XOR's $(0,1)\to 1$, we conclude $w_2 + b > \_$. **Answer: 0.**
- *Construct:* Find a 2-2-1 ReLU MLP that solves XOR. (Free-form, validated against truth table; multiple correct answers.) **Sample answer: $W_1 = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, $b_1 = (-0.5, -1.5)$, $w_2 = (1, -2)$, $b_2 = 0$, threshold at 0.25.**
- *Interpret:* In hidden-layer coordinates, what shape do the four XOR points form? **Answer: collinear / on a line.**
**Widgets.** `xorPlayground`, `perceptronLine`.
**Estimated minutes.** 18.

### Lesson 11.3 — "Why a stack of linear layers is just one linear layer"
**Summary.** Without a nonlinearity between them, depth gets you nothing. This is why activations exist.
**StepChecks.**
- *Compute:* Given $W_1 = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$, $W_2 = \begin{pmatrix} 1 & 1 \end{pmatrix}$, what is the equivalent single matrix $W = W_2 W_1$? **Answer: $(2,\ 3)$.**
- *Debugging:* The student stacks 5 linear layers, no activations, and sees a straight decisi boundary. Is this a bug or expected? **Answer: expected.**
- *Construction:* Smallest activation function that "fixes" Lesson 11.3's problem? **Answer: any non-polynomial nonlinearity (ReLU, sigmoid, tanh, GELU).**
**Widgets.** `activationDiff` (with identity option) — drop in identity, watch boundary go straight.
**Estimated minutes.** 12.

### Lesson 11.4 — "ReLU, and the activation zoo"
**Summary.** ReLU is the default for hidden layers in 2026, but you should know sigmoid (binary outputs), tanh (zero-centered, used in micrograd), and GELU (transformers).
**StepChecks.**
- *Compute:* $\text{ReLU}(-3) = ?$, $\text{ReLU}(2.5) = ?$. **Answer: 0, 2.5.**
- *Compute:* $\sigma(0) = ?$, $\tanh(0) = ?$. **Answer: 0.5, 0.**
- *Interpretation:* Why isn't $\sigma$ the default hidden activation in modern deep nets? **Answer: vanishing gradient (its derivative maxes at 0.25, products attenuate fast).**
**Widgets.** `reluKinkDrag`, `activationDiff`.
**Estimated minutes.** 18.

### Lesson 11.5 — "Forward pass, ennd"
**Summary.** Run a real MLP from input to output by hand. Then count its parameters. Then take the universal approximation theorem seriously — but no further than it earns.
**StepChecks.**
- *Compute:* For the 2-3-1 net in Example 2.4 with input $(1, 1)$, what is the output (to 3 decimals)? **Answer: 0.191.**
- *Compute:* Parameters in a 784-128-64-10 network with biases? **Answer: $784\cdot 128 + 128 + 128\cdot 64 + 64 + 64 \cdot 10 + 10 = 109{,}386$.**
- *Interpretation:* "Universal approximation says any continuous function can be represented to arbitrary precision. So an infinite-width 1-hidden-layer MLP can solve anything." Identify the *two* things this claim oversells. **Answer: (1) it's existence-only — doesn't tell us how to find weights; (2) "continuous on a compact set," and width may be exponential in input dim.**
**Widgets.** `forwardPassScrubber`, `paramCounter`, `decisionBoundarySculptor`.
**Estimated minutes.** 22.

---

## 7. Problem bank (20 problems)

### Novice tier

**P1.** *(coation, `weighted-sum`)* For $w = (2, -1)$, $b = 0.5$, $x = (3, 4)$, compute $w \cdot x + b$. **Answer: 2.5.**

**P2.** *(computation, `sigmoid`)* Compute $\sigma(0)$, $\sigma(2)$ to 3 decimals. **Answer: 0.500, 0.881.**

**P3.** *(computation, `relu`)* Compute $\text{ReLU}(\text{input})$ for input $= (-2, 0, 0.5, 7)$. **Answer: $(0, 0, 0.5, 7)$.**

**P4.** *(computation, `tanh`)* Compute $\tanh(0)$ and $\tanh'(0)$. **Answer: 0, 1.**

**P5.** *(interpretation, `decision-boundary`)* A perceptron has $w = (1, 1)$, $b = -2$. Which side of its decision boundary is $(0, 0)$ on? Where does the boundary cross the $x_1$-axis? **Answer: negative side; crosses at $x_1 = 2$.**

**P6.** *(computation, `linear-layer`)* Given $W = \begin{pmatrix} 1 & 0 \\ 0 & -1 \\ 1 & 1 \end{pmatrix}$, $b = (0, 0, 1)^\top$, $x = (2, 3)^\top$: compute $Wx + b$. **Answer: $(2, -3, 6)^\top$.**

**P7.** *(computation, `parameter-count`)* How many learnable parameters in a layer that maps $\mathbb{R}^{50} \to \mathbb{R}^{20}$ (with bias)? **Answer: $50 \cdot 20 + 20 = 1020$.**

### Intermediate tier

**P8.** *(construction, `perceptron`)* Find weights and bias for a perceptron that computes the OR function. **Answer: any with $w_1, w_2 > 0$, $-\min(w_1, w_2) < b < 0$. E.g. $w = (1,1), b = -0.5$.**

**P9.** *(proof-scaffold, `xor-problem`)* Show that NO weights $(w_1, w_2, b)$ make a perceptron compute XOR by deriving a contradiction from the four XOR constraints. (Free response; validate against the four inequalities above.)

**P10.** *(construction, `mlp`)* Construct a 2-2-1 MLP with ReLU hidden activations that computes the AND-NOT function $f(x_1, x_2) = x_1 \land \neg x_2$. **Answer (one solution):** Hidden $h = \text{ReLU}(W_1 x + b_1)$ with $W_1 = \begin{pmatrix} 1 & -1 \\ 0 & 0\end{pmatrix}$, $b_1 = (-0.5, 0)$; output $w_2 = (1, 0)$, $b_2 = 0$, threshold at 0.25. (Many valid answers; grader checks truth table.)

**P11.** *(computation, `forward-pass`)* For a 2-2-1 MLP with $W^{(1)} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $b^{(1)} = (0, 0)$, ReLU hidden, $W^{(2)} = (1, 1)$, $b^{(2)} = -1$, identity output: compute $\hat y$ for $x = (0.5, 1.0)$. **Answer: $\hat y = 0.5 + 1.0 - 1 = 0.5$.**

**P12.** *(debugging, `linear-composition-collapse`)* A learner builds a 4-layer MLP with no activation functions and is confused why training plateaus immediately. Explain in one sentence what's happening and what to add to fix it. **Answer: stacked linear layers compose to a single linear layer, so the model is mathematically equivalent to logistic regression; insert a non-polynomial activation (e.g. ReLU) after each hidden layer.**

**P13.** *(interpretation, `dead-relu`)* A ReLU neuron has weights $w = (-2, -1)$ and bias $b = -3$. For inputs in $[0, 1]^2$, will this neuron ever fire (output > 0)? **Answer: no — for $x \in [0,1]^2$, $w \cdot x + b \le 0 + 0 - 3 = -3 < 0$. It is a dead neuron with respect to this input domain.**

**P14.** *(computation, `parameter-count`)* For a 784-256-256-10 MLP with biases, total parameter count? **Answer: 84\cdot 256 + 256 + 256\cdot 256 + 256 + 256\cdot 10 + 10 = 269{,}322$.**

**P15.** *(construction, `decision-boundary`)* Sketch the decision regions of an MLP with two ReLU hidden units having decision boundaries $x_1 = 0$ and $x_2 = 0$, with output $w_2 = (1, 1)$, $b_2 = -0.5$, threshold at 0. (Free response; the answer is: the network outputs 1 only in the region $x_1 + x_2 > 0.5$ where both hidden units are active.)

### Advanced tier

**P16.** *(proof-scaffold, `linear-composition-collapse`)* Show that a 3-layer MLP $f(x) = W_3 \phi(W_2 \phi(W_1 x + b_1) + b_2) + b_3$ with $\phi(z) = z$ (identity activation) is equivalent to a single linear layer. State the equivalent $W$ and $b$. **Answer: $W = W_3 W_2 W_1$, $b = W_3 W_2 b_1 + W_3 b_2 + b_3$.**

**P17.** *(interpretation, `universal-approximation`)* The universal approximation theorem says a single-hidden-layer MLP can approximate any continuous function on a compact set. Name three things the theorem does *not* tell you. **Answer (any three of):** how many hidden units are needed; whether SGD will find them; whether the network will generalize beyond the compact set; the case of discontinuous targets; computational tractability.

**P18.** *(construction, `mlp-as-ffn-block`)* Write down, in matrix notation, a transformer feed-forward block that takes a 4-dim token vector to a 4-dim output via a 16-dim hidden layer with GELU. State the parameter count. **Answer: $\text{FFN}(x) = W_2\, \text{GELU}(W_1 x + b_1) + b_2$ with $W_1 \in \mathbb{R}^{16 \times 4}$, $b_1 \in \mathbb{R}^{16}$, $W_2 \in \mathbb{R}^{4 \times 16}$, $b_2 \in \mathbb{R}^4$. Parameter count: $4\cdot 16 + 16 + 16\cdot 4 + 4 = 148$.**

**P19.** *(debugging, `dead-relu`)* In a 2-layer ReLU network trained on a centered Gaussian dataset, a learner notices that 60% of hidden neurons have zero output for *all* training inputs. State two distinct causes and one mitigation each. **Answer (sample):** (1) Learning rate too high causing a large negative weight update — mitigation: lower learning te. (2) Symmetric initialization combined with all-negative biases — mitigation: switch to He / Kaiming initialization or use Leaky ReLU / GELU. (See [Lu et al. on dying ReLU](https://arxiv.org/abs/1903.06733).)

**P20.** *(construction + interpretation, `feature-space-warping`)* The two-spirals dataset cannot be linearly separated in the input plane. Sketch what the *minimum* hidden-space dimension and approximate hidden-layer width must be such that an MLP can separate it, and explain in one paragraph why. **Sample answer:** an MLP with one hidden layer of $\sim 16$ ReLU units in 2D input typically suffices; the hidden layer learns a 16-dim representation in which the spirals become approximately linearly separable. Lower bound: the construction must "unwind" each spiral arm, so the number of hidden ReLUs effectively bounds the polygonal complexity of the decision region — and a 2-arm spiral wrapping ~2× requires ~8–16 distinct hyperplanes worth of hinge points to approximate.

---

## 8. Endgame c — refined

The starter — *"The transformer's feed-forward block is literally an MLP"* — is good. Three sharper alternatives:

**Option A (preferred).**
> *The transformer's feed-forward block is a 2-layer MLP with GELU. Same matmul, same bias, same nonlinearity. You're already done with half of it.*

Why this wins: it's specific (2 layers, GELU), it ties to the exact thing they just learned (linear, activation, linear), and it foreshadows the Module 16 reveal without spoiling.

**Option B.**
> *Every transformer is a stack of attention blocks and MLP blocks. You've now built the MLP block. Two modules from now, we'll add attention — and that's the whole architecture.*

Why this is good: most explicit roadmap; works as a "you are here" marker.

**Option C.**
> *Modern LLMs spend roughly two-thirds of their parameters inside MLP blocks. The architecture you wrote on paper today is where most of GPT actually lives.*

Why this is good: a striking fact, grounded in [3Blue1Brown's MLP-in-transformer chapps://www.3blue1brown.com/lessons/mlp/) ("facts seem to be stored inside a specific part of these networks known as the multi-layer perceptrons"). Use this if the audience needs motivation more than roadmap.

**Recommendation: A as the standard footer, C reserved for Lesson 11.5 specifically.**

---

## 9. Sources (licensing-aware)

**[ADAPT] Tensorflow Playground (Smilkov & Carter, Google).**
URL: https://playground.tensorflow.org/ (source: https://github.com/tensorflow/playground)
Medium: widget. License: **Apache 2.0**.
Use for: directly adapt design language for `decisionBoundarySculptor`, `xorPlayground`. Attribute clearly in widget credits.

**[ADAPT] ConvNetJS (Andrej Karpathy).**
URL: https://cs.stanford.edu/people/karpathy/convnetjs/ (https://github.com/karpathy/convnetjs)
Medium: widget library + demos. License: **MIT**.
Use for: training-loop reference; `classify2d.html` demo as the precedent for hidden-space visualization in `xorPlayground`. Attribute.

**[ADAPT] micrograd (Andrej Karpathy).**
URL: https://github.com/karpathy/micrograd
Medium: code. License: **MIT**.
Use for: Module 12's autograd skeleton. For Module 11, micrograd's tanh-MLP demo is the reference for "smallest possible MLP that works." Attribute.

**[ADAPT] MIT 6.S191 Introduction to Deep Learning (Amini & Amini) — code only.**
URL: https://introtodeeplearning.com/ ; https://github.com/MITDeepLearning/introtodeeplearning
Medium: code (labs) + slides. License: **code is MIT-licensed; lecture slides are copyrighted with reference-attribution requirement.** Treat code [ADAPT], slides [REFERENCE-ONLY].
Use for: lab 1 dense-layer-from-scratch as a pedagogical template; cite slides for sequencing inspiration only.

**[REFERENCE-ONLY] Michael Nielsen, *Neural Networks and Deep Learning*.**
URL: http://neuralnetworksanddeeplearning.com/
Medium: book (online). License: **CC BY-NC 3.0** — non-commercial.
Use for: pedagogical reference for the universal approximation chapter ([Ch. 4](http://neuralnetworksanddeeplearning.com/chap4.html)) ahe perceptron→sigmoid narrative arc ([Ch. 1](http://neuralnetworksanddeeplearning.com/chap1.html)). **Do not adapt text or figures.** Cite as inspiration in instructor notes.

**[REFERENCE-ONLY] CS231n notes (Stanford / Karpathy & Fei-Fei et al.).**
URL: https://cs231n.github.io/
Medium: course notes. License: **not openly licensed** (default copyright; the GitHub repo is for site source only).
Use for: see the canonical sequencing of "linear classifier → loss → MLP → backprop"; cite for pedagogy choices.

**[REFERENCE-ONLY] Goodfellow, Bengio, Courville, *Deep Learning*.**
URL: https://www.deeplearningbook.org/
Medium: textbook. License: **MIT Press copyright**; HTML web version exists explicitly as DRM. **Do not copy text, figures, exercises, or examples.**
Use for: pedagogy reference and citations in instructor notes only. Especially Ch. 6 (Deep Feedforward Networks) for sequencing and the canonical XOR worked example structure.

**[REFERENCE-ONLY] 3Blue1Brown Neural Networks series (Grant SanderURL: https://www.3blue1brown.com/topics/neural-networks ; YouTube playlist.
Medium: video + companion text. License: **copyright Grant Sanderson; videos free to embed but figures/animations not openly licensed**.
Use for: visual reference for hidden-layer feature interpretation ([Ch. 1 lesson](https://www.3blue1brown.com/lessons/neural-networks/)); the [MLP-in-transformer lesson](https://www.3blue1brown.com/lessons/mlp/) directly supports the Module 11 endgame callback. **Embed videos, do not copy frames.**

**[ADAPT] Karpathy "Neural Networks: Zero to Hero" lecture notebooks.**
URL: https://github.com/karpathy/nn-zero-to-hero
Medium: notebooks + videos. License: **MIT** for the notebooks; videos are copyright Karpathy.
Use for: sequencing of "scalar-valued → vector-valued → MLP" matches Tinker's progression; adapt notebook code patterns with attribution.

**[REFERENCE-ONLY] Christopher Olah, "Neural Networks, Manifolds, and Topology."**
URL: https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
um: blog post. License: **default copyright; no open license stated**.
Use for: pedagogy reference for the feature-space-warping concept and `xorPlayground` design. **Cite, do not copy figures.**

**[ADAPT] Distill.pub articles (e.g. "The Building Blocks of Interpretability," "Visualizing Neural Networks with the Grand Tour").**
URLs: https://distill.pub/2018/building-blocks/ ; https://distill.pub/2020/grand-tour/
Medium: interactive articles. License: **CC BY 4.0** — adaptable with attribution.
Use for: you can adapt visualizations and code (the Distill code is on GitHub) as long as you attribute. This is one of the few high-quality CC BY sources. Prioritize.

**Avoid for adaptation (reference only): Khan Academy (CC BY-NC-SA), Wikipedia (CC BY-SA), MIT OCW lecture notes (CC BY-NC-SA).** All three appear in the research above; in this commercial product, they're for cross-checking definitions and pedagogy only, never copied or remixed into the lessons.

---

## 10. Pedagogical traps

### Trap 1 — Skippthe single neuron and starting at "deep learning"
**Why it happens.** "Deep learning" is what's exciting and what students think they signed up for. Many courses spend 90 seconds on a perceptron and 90 minutes on convnets.
**Mitigation.** Lesson 11.1 spends 15 minutes on *exactly one neuron* with `perceptronLine` as the central artifact. The student should be able to compute the AND function by hand before they ever see a hidden layer. Lock progression: Lesson 11.2 (the XOR moment) is gated on completing 11.1's StepChecks.

### Trap 2 — Teaching activations as a memorize-the-zoo exercise
**Why it happens.** Sigmoid/tanh/ReLU/Leaky ReLU/PReLU/ELU/SELU/Swish/GELU/Mish — a textbook lists ten and asks the student to remember which is which.
**Mitigation.** Frame the entire activation-functions lesson around one question: *"What would break if I removed the nonlinearity?"* Lesson 11.3 demonstrates linear-composition collapse first; activations are introduced as the *fix*. Then in Lesson 11.4, only **four** aations get name-and-shape: sigmoid (output for binary), tanh (the micrograd one), ReLU (the default), GELU (the transformer one). The rest are footnoted in the problem bank.

### Trap 3 — Universal approximation as "MLPs can do anything"
**Why it happens.** The name. Pop-science writeups. The visual proof in Nielsen Ch. 4 *looks* constructive.
**Mitigation.** When the theorem is introduced (Lesson 11.5, ~minute 18), pair it immediately with three caveats baked into the StepCheck (P17 in the bank): existence-only, no width bound, no training guarantee. The widget on this page is `forwardPassScrubber` rather than a "watch the universal approximator approximate" widget — we deliberately don't dramatize the theorem visually. Cite [Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem)'s phrasing: *"They guarantee that a network with the right structure exists, but they do not provide a method for finding the network's parameters."*

### Trap 4 — "Draw a curve around the dots"
**Why it h.** Decision-boundary heatmaps from TensorFlow Playground look like the network is drawing a smooth curve in the input plane.
**Mitigation.** `xorPlayground` and `decisionBoundarySculptor` always show *two* views: input space and hidden representation. The pedagogical claim is reinforced verbally and visually: *the network does not draw curves; it bends the space and then draws a line.* Olah's manifolds-and-topology framing is the spine. Reinforced in P15 (sketch decision regions from hidden-unit boundaries) and P20 (spirals).

### Trap 5 — Letting "biological neuron" linger
**Why it happens.** Every textbook opens with the McCulloch-Pitts illustration. Students leave with a fuzzy "it's like a brain" mental model that is wrong in detail and breeds magical thinking later.
**Mitigation.** Acknowledge the historical motivation for ~30 seconds in Lesson 11.1, then **explicitly retire the analogy** with the Sophos-style comparison: real neurons have 10⁴ synapses, complex dendritic dynamics, neurotransmitter istry; an artificial neuron is one dot product and one scalar function. After that point, in all five lessons, the language is exclusively "unit," "neuron-as-named-by-history," or "ReLU unit" — never "brain neuron." This matters because in Module 12 (backprop), the brain analogy directly impedes understanding of the chain rule.

### Trap 6 — Forward pass before backprop being treated as inert
**Why it happens.** Students see the forward pass as "the answer" and backprop as "magic that finds weights."
**Mitigation.** The endgame callback for the entire module foreshadows Module 12 explicitly: *"You just built the computation graph. Next module, we walk it backwards."* Lesson 11.5's `forwardPassScrubber` widget keeps a hidden affordance — at the end of training, the same scrubber is reused in Module 12 with arrows reversed for backprop. Forward pass is presented as *the structure that backprop will exploit*, not as a self-contained skill.
